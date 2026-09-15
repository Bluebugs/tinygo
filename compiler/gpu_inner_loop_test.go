package compiler

import (
	"go/ast"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tinygo-org/tinygo/compileopts"
	"golang.org/x/tools/go/ssa"
)

const innerLoopSummarySrc = `
package p

import "lanes"

const tbl = "\x00\x01\x00\x02\x00\x04\x00\x08\x00\x10\x00\x20\x00\x40\x00\x80"

func summary(sum []uint32, text []byte) {
	go for w := range len(sum) {
		var acc lanes.Varying[uint32]
		for j := range 32 {
			c := tbl[text[32*w+j]&15]
			acc = acc | lanes.Varying[uint32](c)
		}
		sum[w] = acc
	}
}
`

func TestGPUInnerLoopSummaryEligible(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, innerLoopSummarySrc, 1)
	if plan.Reject != "" {
		t.Fatalf("rejected: %s", plan.Reject)
	}
	wgsl := transpileOK(t, innerLoopSummarySrc)
	for _, want := range []string{"var<storage, read_write> sum: array<u32>;", "let j: i32 = 0;"} {
		if !strings.Contains(wgsl, want) {
			t.Errorf("WGSL missing %q:\n%s", want, wgsl)
		}
	}
	// The accumulator must be declared inside the entry function body (per
	// invocation / per lane), not at module scope.
	entry := wgsl[strings.Index(wgsl, "fn spmd_kernel_"):]
	if !strings.Contains(entry, "var acc") {
		t.Errorf("accumulator not declared in invocation scope:\n%s", wgsl)
	}
	if strings.Index(wgsl, "var acc") < strings.Index(wgsl, "fn spmd_kernel_") {
		t.Errorf("accumulator declared at module scope:\n%s", wgsl)
	}
}

// TestGPUInnerLoopSummaryGolden pins the kernel shape when the inner loop is
// too long to unroll.
func TestGPUInnerLoopSummaryGolden(t *testing.T) {
	const golden = `struct Params {
  n: i32,
  _pad0: i32,
  _pad1: i32,
  _pad2: i32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> sum: array<u32>;
@group(0) @binding(2) var<storage, read> text: array<u32>;
var<private> tbl_tbl: array<u32, 16> = array<u32, 16>(0u, 1u, 0u, 2u, 0u, 4u, 0u, 8u, 0u, 16u, 0u, 32u, 0u, 64u, 0u, 128u);
@compute @workgroup_size(64)
fn spmd_kernel_0(@builtin(global_invocation_id) gid: vec3<u32>) {
  let w: i32 = i32(gid.x);
  if (w >= params.n) { return; }
  var acc: u32 = 0;
  for (var j: i32 = 0; j < 33; j = (j + 1)) {
    var c: u32 = tbl_tbl[u32((((text[u32(((32 * w) + j)) >> 2u] >> ((u32(((32 * w) + j)) & 3u) * 8u)) & 0xffu) & 15))];
    acc = (acc | u32(c));
  }
  sum[w] = acc;
}
`
	// Above gpuUnrollMaxTrips the inner loop stays a WGSL for loop.
	src := strings.Replace(innerLoopSummarySrc, "range 32", "range 33", 1)
	if got := transpileOK(t, src); got != golden {
		t.Fatalf("WGSL mismatch.\n--- got ---\n%s\n--- want ---\n%s", got, golden)
	}
}

// TestGPUInnerLoopCostUsesTripCount checks that a constant `range K` inner
// loop is costed by K rather than the dynamic-bound multiplier. The body has
// fixed costs outside the inner loop, so the difference between K=32 and K=4
// is 28 times a per-trip cost, not a proportional scaling.
func TestGPUInnerLoopCostUsesTripCount(t *testing.T) {
	src4 := strings.Replace(innerLoopSummarySrc, "range 32", "range 4", 1)
	p32 := parseAndAnalyzeGPULoop(t, innerLoopSummarySrc, 1)
	p4 := parseAndAnalyzeGPULoop(t, src4, 1)
	if p32.BodyCost <= p4.BodyCost {
		t.Fatalf("BodyCost range 32 = %d, range 4 = %d; want range 32 > range 4", p32.BodyCost, p4.BodyCost)
	}
	if d := p32.BodyCost - p4.BodyCost; d%28 != 0 {
		t.Errorf("BodyCost range 32 = %d, range 4 = %d; difference %d is not a multiple of 28", p32.BodyCost, p4.BodyCost, d)
	}
}

// TestGPUInnerLoopCostNamedConstBound checks that a classic for loop bounded
// by a named constant is costed by that constant.
func TestGPUInnerLoopCostNamedConstBound(t *testing.T) {
	lit := strings.Replace(innerLoopSummarySrc, "for j := range 32 {", "for j := 0; j < 32; j++ {", 1)
	named := strings.Replace(innerLoopSummarySrc, "for j := range 32 {", "for j := 0; j < width; j++ {", 1) + "\nconst width = 32\n"
	pl := parseAndAnalyzeGPULoop(t, lit, 1)
	pn := parseAndAnalyzeGPULoop(t, named, 1)
	pr := parseAndAnalyzeGPULoop(t, innerLoopSummarySrc, 1)
	if pn.BodyCost != pl.BodyCost || pr.BodyCost != pl.BodyCost {
		t.Errorf("BodyCost literal = %d, named const = %d, range = %d; want equal", pl.BodyCost, pn.BodyCost, pr.BodyCost)
	}
}

func TestGPUInnerLoopShapeReject(t *testing.T) {
	entry := &ssa.BasicBlock{Index: 0}
	entry.Instrs = []ssa.Instruction{&ssa.Jump{}}
	done := &ssa.BasicBlock{Index: 1}
	for _, tc := range []struct {
		name string
		trip int64
		want string
	}{
		{"non-constant bound", -1, "inner loop has non-constant bound"},
		{"constant bound", 32, ""},
	} {
		t.Run(tc.name, func(t *testing.T) {
			info := &ssa.SPMDLoopInfo{
				EntryBlock: entry,
				DoneBlock:  done,
				InnerLoops: []ssa.SPMDInnerLoop{{TripCount: tc.trip}},
			}
			if got := gpuLoopShapeReject(info); got != tc.want {
				t.Errorf("gpuLoopShapeReject = %q, want %q", got, tc.want)
			}
		})
	}
}

// TestGPUInnerLoopAccumulatorTyping checks that int32 and float32 body-local
// accumulators get a correctly typed WGSL declaration and initializer. When
// the naga CLI is available the shader is also validated.
func TestGPUInnerLoopAccumulatorTyping(t *testing.T) {
	for _, tc := range []struct{ goTy, decl, update string }{
		{"int32", "var acc: i32 = 0;", "acc = (acc + i32(c));"},
		{"float32", "var acc: f32 = 0.0;", "acc = (acc + f32(c));"},
	} {
		t.Run(tc.goTy, func(t *testing.T) {
			src := strings.NewReplacer(
				"sum []uint32", "sum []"+tc.goTy,
				"var acc lanes.Varying[uint32]", "var acc lanes.Varying["+tc.goTy+"]",
				"acc = acc | lanes.Varying[uint32](c)", "acc = acc + lanes.Varying["+tc.goTy+"](c)",
			).Replace(innerLoopSummarySrc)
			wgsl := transpileOK(t, src)
			entry := wgsl[strings.Index(wgsl, "fn spmd_kernel_"):]
			for _, want := range []string{tc.decl, tc.update} {
				if !strings.Contains(entry, want) {
					t.Errorf("WGSL entry missing %q:\n%s", want, wgsl)
				}
			}
			nagaValidate(t, wgsl)
		})
	}
}

// nagaValidate runs `naga <file>.wgsl`, which parses and validates the shader.
// It is a no-op when naga is not installed.
func nagaValidate(t *testing.T, wgsl string) {
	t.Helper()
	naga := os.Getenv("NAGA")
	if naga == "" {
		naga, _ = exec.LookPath("naga")
	}
	if naga == "" {
		if home, err := os.UserHomeDir(); err == nil {
			if p := filepath.Join(home, ".cargo", "bin", "naga"); fileExists(p) {
				naga = p
			}
		}
	}
	if naga == "" {
		t.Log("naga not found; skipping WGSL validation")
		return
	}
	file := filepath.Join(t.TempDir(), "kernel.wgsl")
	if err := os.WriteFile(file, []byte(wgsl), 0o644); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(naga, file).CombinedOutput(); err != nil {
		t.Errorf("naga rejected WGSL: %v\n%s\n%s", err, out, wgsl)
	}
}

func fileExists(p string) bool {
	_, err := os.Stat(p)
	return err == nil
}

// TestGPUInnerLoopRuntimeBoundSource compiles the summary kernel with a
// runtime inner bound through the full pipeline under -gpu=webgpu. The
// peeler leaves such a loop unpeeled, and the go for index-dependent read
// makes the CPU lowering reject it with a precise error before GPU analysis
// runs; it must never degrade to the generic "not peeled" reason.
func TestGPUInnerLoopRuntimeBoundSource(t *testing.T) {
	src := strings.NewReplacer(
		"package p", "package main",
		"text []byte)", "text []byte, n int)",
		"for j := range 32 {", "for j := range n {",
	).Replace(innerLoopSummarySrc) + "\nfunc main() { summary(make([]uint32, 8), make([]byte, 256), 32) }\n"
	opts := compileopts.Options{GOOS: "linux", GOARCH: "amd64", LLVMFeatures: "+ssse3,+sse4.2,+avx2",
		GOExperiment: "spmd", GPU: "webgpu", GPUThresholdOps: 1}
	_, errs := compileSPMDSourceErrors(t, src, opts)
	all := strings.Join(errs, "\n")
	const want = "go for body with a nested loop that indexes memory by the go for index is not supported unless the loop can be peeled"
	if !strings.Contains(all, want) {
		t.Fatalf("errors = %q, want one containing %q", all, want)
	}
	if strings.Contains(all, "(not peeled)") {
		t.Errorf("generic not-peeled reason reported: %s", all)
	}
}

// TestGPUInnerLoopNotPeeledReason builds a go for whose runtime-bound inner
// loop does not index memory by the go for index. The peeler leaves it
// unpeeled and CPU codegen succeeds (SSE), so the GPU gate must report the
// precise inner-loop reason, not the generic "not peeled" one.
func TestGPUInnerLoopNotPeeledReason(t *testing.T) {
	const src = `package main

import "lanes"

func summary(sum []uint32, n int) {
	go for w := range len(sum) {
		var acc lanes.Varying[uint32]
		for j := range n {
			acc = acc + lanes.Varying[uint32](j)
		}
		sum[w] = acc
	}
}

func main() { summary(make([]uint32, 8), 32) }
`
	opts := compileopts.Options{GOOS: "linux", GOARCH: "amd64", LLVMFeatures: "+ssse3,+sse4.2",
		GOExperiment: "spmd", GPU: "webgpu", GPUThresholdOps: 1, GPUVerbose: true}
	var errs []string
	stderr := captureStderr(t, func() { _, errs = compileSPMDSourceErrors(t, src, opts) })
	if len(errs) > 0 {
		t.Fatalf("compile errors:\n%s", strings.Join(errs, "\n"))
	}
	if !strings.Contains(stderr, "skipped: "+gpuInnerLoopNotPeeledReason) {
		t.Errorf("missing inner-loop skip reason in -gpu-verbose output:\n%s", stderr)
	}
	if strings.Contains(stderr, "(not peeled)") {
		t.Errorf("generic not-peeled reason reported:\n%s", stderr)
	}
}

// captureStderr returns what fn writes to os.Stderr.
func captureStderr(t *testing.T, fn func()) string {
	t.Helper()
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	old := os.Stderr
	os.Stderr = w
	done := make(chan string)
	go func() {
		var sb strings.Builder
		buf := make([]byte, 4096)
		for {
			n, err := r.Read(buf)
			sb.Write(buf[:n])
			if err != nil {
				break
			}
		}
		done <- sb.String()
	}()
	defer func() { os.Stderr = old }()
	fn()
	w.Close()
	os.Stderr = old
	return <-done
}

// TestGPUInnerLoopPackedReadOneLane checks that a kernel reading a packed byte
// slice inside an inner loop runs one iteration per invocation: on RADV an
// inner loop nested in the packed lane loop is ~10x slower, and removing the
// lane loop recovers it (prototype E). Reads keep the packed unpack, and the
// launch dispatches one invocation per trip.
func TestGPUInnerLoopPackedReadOneLane(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, innerLoopSummarySrc, 1)
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatal(err)
	}
	if k.LanesPerInvocation != 1 {
		t.Errorf("LanesPerInvocation = %d, want 1", k.LanesPerInvocation)
	}
	if strings.Contains(k.WGSL, "lane") {
		t.Errorf("inner-loop kernel must not emit a lane loop:\n%s", k.WGSL)
	}
	for _, want := range []string{"let w: i32 = i32(gid.x);", "if (w >= params.n) { return; }", ">> 2u]"} {
		if !strings.Contains(k.WGSL, want) {
			t.Errorf("WGSL missing %q:\n%s", want, k.WGSL)
		}
	}
	nagaValidate(t, k.WGSL)
}

// TestGPUInnerLoopPackedWriteKeepsLaneLoop checks that a body with an inner
// loop that WRITES a packed byte slice keeps four lanes per invocation: a u32
// word holds four bytes, so one invocation must own the whole word.
func TestGPUInnerLoopPackedWriteKeepsLaneLoop(t *testing.T) {
	const src = `
package p

import "lanes"

func f(dst []byte, src []uint32) {
	go for i := range len(dst) {
		var acc lanes.Varying[uint32]
		for j := range 8 {
			acc = acc + src[i] + lanes.Varying[uint32](j)
		}
		dst[i] = lanes.Varying[byte](acc)
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject != "" {
		t.Fatalf("rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatal(err)
	}
	if k.LanesPerInvocation != 4 || !strings.Contains(k.WGSL, "lane < 4u") {
		t.Errorf("LanesPerInvocation = %d, want 4 with a lane loop:\n%s", k.LanesPerInvocation, k.WGSL)
	}
}

// TestGPUInnerLoopUnrolledGolden checks that a constant 32-trip inner loop is
// unrolled: each copy gets its own block with the index bound to a constant,
// and the accumulator chains across copies. On RADV the unrolled body is
// ~1.5x faster than the loop (prototype CE vs E).
func TestGPUInnerLoopUnrolledGolden(t *testing.T) {
	wgsl := transpileOK(t, innerLoopSummarySrc)
	if strings.Contains(wgsl, "for (") {
		t.Errorf("constant inner loop not unrolled:\n%s", wgsl)
	}
	entry := wgsl[strings.Index(wgsl, "fn spmd_kernel_"):]
	var want strings.Builder
	want.WriteString("fn spmd_kernel_0(@builtin(global_invocation_id) gid: vec3<u32>) {\n  let w: i32 = i32(gid.x);\n  if (w >= params.n) { return; }\n  var acc: u32 = 0;\n")
	for j := 0; j < 32; j++ {
		want.WriteString("  {\n")
		want.WriteString("    let j: i32 = " + strconv.Itoa(j) + ";\n")
		want.WriteString("    var c: u32 = tbl_tbl[u32((((text[u32(((32 * w) + j)) >> 2u] >> ((u32(((32 * w) + j)) & 3u) * 8u)) & 0xffu) & 15))];\n")
		want.WriteString("    acc = (acc | u32(c));\n")
		want.WriteString("  }\n")
	}
	want.WriteString("  sum[w] = acc;\n}\n")
	if entry != want.String() {
		t.Fatalf("WGSL mismatch.\n--- got ---\n%s\n--- want ---\n%s", entry, want.String())
	}
	nagaValidate(t, wgsl)
}

// TestGPUInnerLoopUnrollClassicFor checks the `j := 0; j < K; j++` form.
func TestGPUInnerLoopUnrollClassicFor(t *testing.T) {
	src := strings.Replace(innerLoopSummarySrc, "for j := range 32 {", "for j := 0; j < width; j++ {", 1) + "\nconst width = 32\n"
	wgsl := transpileOK(t, src)
	if strings.Contains(wgsl, "for (") || !strings.Contains(wgsl, "let j: i32 = 31;") || strings.Contains(wgsl, "let j: i32 = 32;") {
		t.Errorf("classic constant loop not unrolled 0..31:\n%s", wgsl)
	}
	nagaValidate(t, wgsl)
}

// TestGPUInnerLoopUnrollKeepsFor checks the cases that must keep the WGSL loop.
func TestGPUInnerLoopUnrollKeepsFor(t *testing.T) {
	for _, tc := range []struct{ name, from, to string }{
		{"trips above cap", "range 32", "range " + strconv.Itoa(gpuUnrollMaxTrips+1)},
		{"continue inside", "c := tbl[text[32*w+j]&15]", "if j == 3 {\n\t\t\t\tcontinue\n\t\t\t}\n\t\t\tc := tbl[text[32*w+j]&15]"},
		{"index modified", "for j := range 32 {", "for j := 0; j < 32; j += 2 {"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := strings.Replace(innerLoopSummarySrc, tc.from, tc.to, 1)
			plan := parseAndAnalyzeGPULoop(t, src, 1)
			if plan.Reject != "" {
				t.Skipf("rejected: %s", plan.Reject)
			}
			k, err := transpileWGSL(plan, 0)
			if err != nil {
				t.Fatal(err)
			}
			if !strings.Contains(k.WGSL, "for (") || strings.Contains(k.WGSL, "let j") {
				t.Errorf("loop must be kept:\n%s", k.WGSL)
			}
			nagaValidate(t, k.WGSL)
		})
	}
}

// TestGPUInnerLoopUnrollSizeCap checks that a body whose trips times statement
// count exceeds the size cap keeps the loop even with few trips.
func TestGPUInnerLoopUnrollSizeCap(t *testing.T) {
	var body strings.Builder
	for i := 0; i < gpuUnrollMaxStmts/32+1; i++ {
		body.WriteString("\t\t\tacc = acc | lanes.Varying[uint32](c)\n")
	}
	src := strings.Replace(innerLoopSummarySrc, "\t\t\tacc = acc | lanes.Varying[uint32](c)\n", body.String(), 1)
	wgsl := transpileOK(t, src)
	if !strings.Contains(wgsl, "for (") {
		t.Errorf("oversized body must keep the loop:\n%s", wgsl)
	}
}

// TestGPUInnerLoopUnrollNestedScoping unrolls two nested constant loops that
// both declare locals; every copy must be block scoped so the shader is valid.
func TestGPUInnerLoopUnrollNestedScoping(t *testing.T) {
	const src = `
package p

import "lanes"

func f(sum []uint32, text []byte) {
	go for w := range len(sum) {
		var acc lanes.Varying[uint32]
		for j := range 4 {
			x := lanes.Varying[uint32](text[16*w+4*j])
			for k := range 4 {
				y := x + lanes.Varying[uint32](text[16*w+4*j+k])
				acc = acc + y
			}
		}
		sum[w] = acc
	}
}
`
	wgsl := transpileOK(t, src)
	if strings.Contains(wgsl, "for (") {
		t.Errorf("nested constant loops not unrolled:\n%s", wgsl)
	}
	if got := strings.Count(wgsl, "var y: u32"); got != 16 {
		t.Errorf("var y declared %d times, want 16:\n%s", got, wgsl)
	}
	if got := strings.Count(wgsl, "var x: u32"); got != 4 {
		t.Errorf("var x declared %d times, want 4:\n%s", got, wgsl)
	}
	nagaValidate(t, wgsl)
}

// TestGPUInnerLoopUnrollParenIndexWrite checks the emitter's own guard: a
// parenthesized write to the index must keep the loop. Eligibility may reject
// such bodies first, so unrollable is called directly on the inner loop.
func TestGPUInnerLoopUnrollParenIndexWrite(t *testing.T) {
	for _, tc := range []struct{ name, stmt string }{
		{"increment", "(j)++"},
		{"assignment", "(j) = 1"},
		{"address", "_ = &(j)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := strings.Replace(innerLoopSummarySrc, "c := tbl[text[32*w+j]&15]", tc.stmt+"\n\t\t\tc := tbl[text[32*w+j]&15]", 1)
			plan := parseAndAnalyzeGPULoop(t, src, 1)
			var inner *ast.RangeStmt
			ast.Inspect(plan.Body, func(n ast.Node) bool {
				if rs, ok := n.(*ast.RangeStmt); ok && inner == nil {
					inner = rs
				}
				return inner == nil
			})
			if inner == nil {
				t.Fatal("inner loop not found")
			}
			e := &wgslEmitter{info: plan.Loop.TypesInfo}
			if _, ok := e.unrollable(inner); ok {
				t.Errorf("loop writing %s must not be unrolled", tc.stmt)
			}
		})
	}
}
