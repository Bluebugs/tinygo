package compileopts

import (
	"bufio"
	"go/build/constraint"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// gpuRuntimeFiles must partition every build: the compiler emits calls to
// spmdGPU* on every target, so zero matching files is a link/verify error
// and two is a duplicate-definition error.
var gpuRuntimeFiles = []string{"gpu_wasm.go", "gpu_native.go", "gpu_vulkan.go", "gpu_stub.go"}

// gpuTagAtoms are every build tag the four files' constraints mention.
var gpuTagAtoms = []string{
	"tinygo.wasm", "js", "linux", "amd64",
	"spmd.gpu.webgpu", "spmd.gpu.host.browser", "spmd.gpu.host.vulkan",
}

func readBuildConstraint(t *testing.T, path string) constraint.Expr {
	t.Helper()
	f, err := os.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		line := sc.Text()
		if constraint.IsGoBuild(line) {
			expr, err := constraint.Parse(line)
			if err != nil {
				t.Fatalf("%s: %v", path, err)
			}
			return expr
		}
		if line != "" && !strings.HasPrefix(line, "//") {
			break
		}
	}
	t.Fatalf("%s: no //go:build line", path)
	return nil
}

func matchingGPUFiles(exprs []constraint.Expr, tags map[string]bool) []string {
	var match []string
	for i, e := range exprs {
		if e.Eval(func(tag string) bool { return tags[tag] }) {
			match = append(match, gpuRuntimeFiles[i])
		}
	}
	return match
}

func TestGPURuntimeTagPartition(t *testing.T) {
	exprs := make([]constraint.Expr, len(gpuRuntimeFiles))
	for i, name := range gpuRuntimeFiles {
		exprs[i] = readBuildConstraint(t, filepath.Join("..", "src", "runtime", name))
	}

	// Exhaustive: every subset of the atoms selects exactly one file.
	for mask := 0; mask < 1<<len(gpuTagAtoms); mask++ {
		tags := map[string]bool{}
		for i, atom := range gpuTagAtoms {
			if mask&(1<<i) != 0 {
				tags[atom] = true
			}
		}
		if m := matchingGPUFiles(exprs, tags); len(m) != 1 {
			t.Errorf("tags %v: matched %v, want exactly one file", tags, m)
		}
	}

	// Named configurations, including which file must win.
	tests := []struct {
		name string
		tags []string
		want string
	}{
		{"wasm js webgpu", []string{"tinygo.wasm", "js", "spmd.gpu.webgpu"}, "gpu_wasm.go"},
		{"wasm js no gpu", []string{"tinygo.wasm", "js", "spmd.gpu.none"}, "gpu_stub.go"},
		{"wasip1 browser host", []string{"tinygo.wasm", "wasip1", "spmd.gpu.webgpu", "spmd.gpu.host.browser"}, "gpu_wasm.go"},
		{"wasip1 webgpu no host", []string{"tinygo.wasm", "wasip1", "spmd.gpu.webgpu"}, "gpu_stub.go"},
		{"linux amd64 webgpu", []string{"linux", "amd64", "spmd.gpu.webgpu"}, "gpu_native.go"},
		{"linux amd64 webgpu vulkan", []string{"linux", "amd64", "spmd.gpu.webgpu", "spmd.gpu.host.vulkan"}, "gpu_vulkan.go"},
		{"linux amd64 no gpu", []string{"linux", "amd64", "spmd.gpu.none"}, "gpu_stub.go"},
		{"linux arm64 webgpu", []string{"linux", "arm64", "spmd.gpu.webgpu"}, "gpu_stub.go"},
		{"darwin amd64 webgpu", []string{"darwin", "amd64", "spmd.gpu.webgpu"}, "gpu_stub.go"},
		{"darwin amd64 webgpu vulkan", []string{"darwin", "amd64", "spmd.gpu.webgpu", "spmd.gpu.host.vulkan"}, "gpu_stub.go"},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tags := map[string]bool{}
			for _, tag := range tc.tags {
				tags[tag] = true
			}
			m := matchingGPUFiles(exprs, tags)
			if len(m) != 1 || m[0] != tc.want {
				t.Errorf("matched %v, want [%s]", m, tc.want)
			}
		})
	}
}
