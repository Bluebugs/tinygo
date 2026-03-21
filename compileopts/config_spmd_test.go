package compileopts

import "testing"

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
