package compiler

import (
	"encoding/binary"
	"os/exec"
	"testing"
)

func TestGPUSPIRVFromMandelbrotWGSL(t *testing.T) {
	if _, err := exec.LookPath(nagaPath()); err != nil {
		t.Skip("naga not available")
	}
	plan := parseAndAnalyzeGPULoop(t, mandelbrotFlatGPUSrc, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatal(err)
	}
	spv, err := wgslToSPIRV(k.WGSL)
	if err != nil {
		t.Fatalf("wgslToSPIRV: %v", err)
	}
	if len(spv) < 20 || len(spv)%4 != 0 {
		t.Fatalf("bad SPIR-V length %d", len(spv))
	}
	if magic := binary.LittleEndian.Uint32(spv); magic != 0x07230203 {
		t.Fatalf("bad SPIR-V magic %#x", magic)
	}
	again, err := wgslToSPIRV(k.WGSL)
	if err != nil || &again[0] != &spv[0] {
		t.Fatalf("second conversion must hit the cache (err=%v)", err)
	}
}

func TestGPUSPIRVInvalidWGSL(t *testing.T) {
	if _, err := exec.LookPath(nagaPath()); err != nil {
		t.Skip("naga not available")
	}
	if _, err := wgslToSPIRV("fn broken( {"); err == nil {
		t.Fatal("expected an error for invalid WGSL")
	}
}

// TestGPURegisterPayload locks down the spmdGPURegister call shape chosen at
// emission time: the vulkan host gets the SPIR-V bytes plus a buffer count,
// every other host keeps the 3-argument WGSL call.
func TestGPURegisterPayload(t *testing.T) {
	k := &gpuKernel{
		WGSL:    "wgsl-source",
		SPIRV:   []byte{0x03, 0x02, 0x23, 0x07},
		Buffers: make([]gpuFreeVar, 3),
	}
	for _, tc := range []struct {
		host        string
		shader      string
		wantCount   bool
		bufferCount int32
	}{
		{"", "wgsl-source", false, 0},
		{"browser", "wgsl-source", false, 0},
		{"vulkan", string(k.SPIRV), true, 3},
	} {
		t.Run("host="+tc.host, func(t *testing.T) {
			p := gpuRegisterPayload(tc.host, k)
			if p.shader != tc.shader || p.withBufCount != tc.wantCount || p.bufCount != tc.bufferCount {
				t.Errorf("got %+v, want shader=%q withBufCount=%v bufCount=%d", p, tc.shader, tc.wantCount, tc.bufferCount)
			}
		})
	}
}
