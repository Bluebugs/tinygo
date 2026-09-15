package compiler

// WGSL -> SPIR-V conversion for the direct Vulkan host (-gpu-host=vulkan).
// The Vulkan runtime consumes SPIR-V; the WGSL transpiler stays the single
// source of truth, and naga (the same translator wgpu uses internally)
// converts its output at build time.

import (
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"sync"

	"github.com/tinygo-org/tinygo/goenv"
)

var spirvCache sync.Map // [sha256.Size]byte -> []byte

// nagaPath returns the naga executable used by both the build and the tests.
func nagaPath() string {
	if p := goenv.Get("NAGA"); p != "" {
		return p
	}
	return "naga"
}

// wgslToSPIRV compiles a WGSL module to SPIR-V with naga, caching by content.
func wgslToSPIRV(wgsl string) ([]byte, error) {
	key := sha256.Sum256([]byte(wgsl))
	if v, ok := spirvCache.Load(key); ok {
		return v.([]byte), nil
	}
	dir, err := os.MkdirTemp("", "tinygo-spirv-")
	if err != nil {
		return nil, err
	}
	defer os.RemoveAll(dir)
	in := filepath.Join(dir, "kernel.wgsl")
	out := filepath.Join(dir, "kernel.spv")
	if err := os.WriteFile(in, []byte(wgsl), 0o600); err != nil {
		return nil, err
	}
	cmd := exec.Command(nagaPath(), in, out)
	if msg, err := cmd.CombinedOutput(); err != nil {
		return nil, fmt.Errorf("naga: %v: %s", err, msg)
	}
	spv, err := os.ReadFile(out)
	if err != nil {
		return nil, err
	}
	if len(spv) < 20 || len(spv)%4 != 0 || binary.LittleEndian.Uint32(spv) != 0x07230203 {
		return nil, fmt.Errorf("naga produced invalid SPIR-V (%d bytes)", len(spv))
	}
	actual, _ := spirvCache.LoadOrStore(key, spv)
	return actual.([]byte), nil
}
