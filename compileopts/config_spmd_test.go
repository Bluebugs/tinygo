package compileopts

import (
	"strings"
	"testing"
)

func TestFeaturesAutoSIMD128(t *testing.T) {
	tests := []struct {
		name         string
		goExperiment string
		goarch       string
		features     string
		llvmFeatures string
		want         string
	}{
		{
			name:         "SPMD+WASM adds simd128 and relaxed-simd",
			goExperiment: "spmd",
			goarch:       "wasm",
			want:         "+simd128,+relaxed-simd",
		},
		{
			name:         "SPMD+WASM with existing features",
			goExperiment: "spmd",
			goarch:       "wasm",
			features:     "+bulk-memory,+sign-ext",
			want:         "+bulk-memory,+sign-ext,+simd128,+relaxed-simd",
		},
		{
			name:         "SPMD+WASM already has simd128",
			goExperiment: "spmd",
			goarch:       "wasm",
			features:     "+bulk-memory,+simd128,+sign-ext",
			want:         "+bulk-memory,+simd128,+sign-ext,+relaxed-simd",
		},
		{
			name:         "SPMD+WASM already has relaxed-simd",
			goExperiment: "spmd",
			goarch:       "wasm",
			features:     "+bulk-memory,+simd128,+relaxed-simd",
			want:         "+bulk-memory,+simd128,+relaxed-simd",
		},
		{
			name:         "no SPMD does not add simd128",
			goExperiment: "",
			goarch:       "wasm",
			features:     "+bulk-memory",
			want:         "+bulk-memory",
		},
		{
			name:         "SPMD+non-WASM does not add simd128",
			goExperiment: "spmd",
			goarch:       "arm",
			features:     "+armv7",
			want:         "+armv7",
		},
		{
			name:         "SPMD+WASM merges llvm-features",
			goExperiment: "spmd",
			goarch:       "wasm",
			features:     "+bulk-memory",
			llvmFeatures: "+sign-ext",
			want:         "+bulk-memory,+sign-ext,+simd128,+relaxed-simd",
		},
		{
			name:         "SPMD+WASM llvm-features has simd128",
			goExperiment: "spmd",
			goarch:       "wasm",
			features:     "+bulk-memory",
			llvmFeatures: "+simd128",
			want:         "+bulk-memory,+simd128,+relaxed-simd",
		},
		{
			name:         "empty everything",
			goExperiment: "",
			goarch:       "wasm",
			want:         "",
		},
		{
			name:         "nospmd does not add simd128",
			goExperiment: "nospmd",
			goarch:       "wasm",
			features:     "+bulk-memory",
			want:         "+bulk-memory",
		},
		{
			name:         "SPMD among multiple experiments",
			goExperiment: "foo,spmd,bar",
			goarch:       "wasm",
			want:         "+simd128,+relaxed-simd",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := &Config{
				Options: &Options{
					GOExperiment: tt.goExperiment,
					LLVMFeatures: tt.llvmFeatures,
				},
				Target: &TargetSpec{
					GOARCH:   tt.goarch,
					Features: tt.features,
				},
			}
			got := c.Features()
			if got != tt.want {
				t.Errorf("Features() = %q, want %q", got, tt.want)
			}
		})
	}
}

func TestSIMDDisabledSuppressesFeatures(t *testing.T) {
	c := &Config{
		Options: &Options{
			GOExperiment: "spmd",
			SIMD:         "false",
		},
		Target: &TargetSpec{GOARCH: "wasm"},
	}
	features := c.Features()
	if strings.Contains(features, "+simd128") {
		t.Errorf("expected no +simd128 with -simd=false, got: %s", features)
	}
}

func TestSIMDDefaultEnabled(t *testing.T) {
	c := &Config{
		Options: &Options{GOExperiment: "spmd"},
		Target:  &TargetSpec{GOARCH: "wasm"},
	}
	features := c.Features()
	if !strings.Contains(features, "+simd128") {
		t.Errorf("expected +simd128 by default with SPMD+WASM, got: %s", features)
	}
}

func TestSIMDEnabledMethod(t *testing.T) {
	c := &Config{
		Options: &Options{GOExperiment: "spmd"},
		Target:  &TargetSpec{GOARCH: "wasm"},
	}
	if !c.SIMDEnabled() {
		t.Error("expected SIMDEnabled() == true for SPMD+WASM")
	}
	c.Options.SIMD = "false"
	if c.SIMDEnabled() {
		t.Error("expected SIMDEnabled() == false with -simd=false")
	}
}

func TestSIMDRegisterSizeAVX2(t *testing.T) {
	c := &Config{
		Target:  &TargetSpec{GOARCH: "amd64"},
		Options: &Options{GOExperiment: "spmd", LLVMFeatures: "+ssse3,+sse4.2,+avx2"},
	}
	if got := c.SIMDRegisterSize(); got != 32 {
		t.Errorf("SIMDRegisterSize() = %d, want 32 for AVX2", got)
	}
}

func TestSIMDRegisterSizeAVX512(t *testing.T) {
	c := &Config{
		Target:  &TargetSpec{GOARCH: "amd64"},
		Options: &Options{GOExperiment: "spmd", LLVMFeatures: "+avx512f"},
	}
	if got := c.SIMDRegisterSize(); got != 64 {
		t.Errorf("SIMDRegisterSize() = %d, want 64 for AVX-512", got)
	}
}

func TestSIMDRegisterSizeSSEDefault(t *testing.T) {
	c := &Config{
		Target:  &TargetSpec{GOARCH: "amd64"},
		Options: &Options{GOExperiment: "spmd", LLVMFeatures: "+ssse3"},
	}
	if got := c.SIMDRegisterSize(); got != 16 {
		t.Errorf("SIMDRegisterSize() = %d, want 16 for SSE", got)
	}
}

func TestSIMDRegisterSizeWASM(t *testing.T) {
	c := &Config{
		Target:  &TargetSpec{GOARCH: "wasm"},
		Options: &Options{GOExperiment: "spmd"},
	}
	if got := c.SIMDRegisterSize(); got != 16 {
		t.Errorf("SIMDRegisterSize() = %d, want 16 for WASM", got)
	}
}

func TestSIMDRegisterSizeScalar(t *testing.T) {
	c := &Config{
		Target:  &TargetSpec{GOARCH: "amd64"},
		Options: &Options{GOExperiment: "spmd", SIMD: "false", LLVMFeatures: "+avx2"},
	}
	if got := c.SIMDRegisterSize(); got != 1 {
		t.Errorf("SIMDRegisterSize() = %d, want 1 for scalar mode", got)
	}
}

func TestGOExperiment(t *testing.T) {
	c := &Config{
		Options: &Options{GOExperiment: "spmd"},
		Target:  &TargetSpec{},
	}
	if got := c.GOExperiment(); got != "spmd" {
		t.Errorf("GOExperiment() = %q, want %q", got, "spmd")
	}

	c.Options.GOExperiment = ""
	if got := c.GOExperiment(); got != "" {
		t.Errorf("GOExperiment() = %q, want %q", got, "")
	}
}
