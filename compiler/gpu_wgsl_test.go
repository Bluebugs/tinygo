// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/token"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// mandelbrotFlatGPUSrc is the flat single-loop SPMD mandelbrot kernel from
// examples/mandelbrot/main.go (mandelSPMD inlined into mandelbrotFlat's
// `go for`), copied verbatim (including mandelSPMD's real
// `for iter := range maxIter` divergence loop) and hand-verified
// statement-by-statement against that file. Fix round 1 added narrow
// *ast.RangeStmt allowlisting to gpu_eligible.go's checkStmt for exactly
// this "for i := range n" shape (see its case there), removing the earlier
// classic-for-loop workaround this test used.
const mandelbrotFlatGPUSrc = `
package p

import "lanes"

const (
	X0 = -2.5
	Y0 = -1.25
	X1 = 1.5
	Y1 = 1.25
)

func mandelSPMD(cRe, cIm lanes.Varying[float32], maxIter int) lanes.Varying[int] {
	var zRe lanes.Varying[float32] = cRe
	var zIm lanes.Varying[float32] = cIm
	var iterations lanes.Varying[int] = maxIter
	for iter := range maxIter {
		magSquared := zRe*zRe + zIm*zIm
		diverged := magSquared > 4.0
		if diverged {
			iterations = iter
			break
		}
		newRe := zRe*zRe - zIm*zIm
		newIm := 2.0 * zRe * zIm
		zRe = cRe + newRe
		zIm = cIm + newIm
	}
	return iterations
}

func mandelbrotFlat(width, height, maxIter int, output []int) {
	dx := (X1 - X0) / float32(width)
	dy := (Y1 - Y0) / float32(height)
	go for idx := range width * height {
		i := idx % width
		j := idx / width
		x := X0 + lanes.Varying[float32](i)*dx
		y := Y0 + lanes.Varying[float32](j)*dy
		iterations := mandelSPMD(x, y, maxIter)
		output[idx] = iterations
	}
}
`

// (The former TestGPUWGSLMandelbrotFlatDump was a t.Logf-only "test" that
// asserted nothing; it was dead weight beside TestGPUWGSLMandelbrotFlatGolden
// below, which transpiles the same source and compares it byte-for-byte.
// Deleted rather than kept as unassertive noise.)

func paramNames(fv []gpuFreeVar) []string {
	var s []string
	for _, f := range fv {
		s = append(s, f.Obj.Name()+":"+f.WGSLTy)
	}
	return s
}

// runNaga writes wgsl to a temp file and validates it with the naga CLI,
// skipping (not failing) if naga is not on PATH.
func runNaga(t *testing.T, wgsl string) {
	t.Helper()
	nagaPath, err := exec.LookPath("naga")
	if err != nil {
		t.Skip("naga not found on PATH; skipping WGSL validation")
	}
	dir := t.TempDir()
	file := filepath.Join(dir, "kernel.wgsl")
	if err := os.WriteFile(file, []byte(wgsl), 0o644); err != nil {
		t.Fatalf("write wgsl: %v", err)
	}
	cmd := exec.Command(nagaPath, file)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("naga validation failed: %v\n%s\n--- wgsl ---\n%s", err, out, wgsl)
	}
}

func TestGPUWGSLMandelbrotFlatGolden(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, mandelbrotFlatGPUSrc, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}

	const golden = `struct Params {
  n: i32,
  dx: f32,
  dy: f32,
  height: i32,
  maxIter: i32,
  width: i32,
  _pad0: i32,
  _pad1: i32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read_write> output: array<i32>;
@compute @workgroup_size(64)
fn spmd_kernel_0(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: i32 = i32(gid.x);
  if (idx >= params.n) { return; }
  var i: i32 = (idx % params.width);
  var j: i32 = (idx / params.width);
  var x: f32 = (-2.5 + (f32(i) * params.dx));
  var y: f32 = (-1.25 + (f32(j) * params.dy));
  var iterations: i32;
  var cRe_i1: f32 = x;
  var cIm_i1: f32 = y;
  var maxIter_i1: i32 = params.maxIter;
  var zRe_i1: f32 = cRe_i1;
  var zIm_i1: f32 = cIm_i1;
  var iterations_i1: i32 = maxIter_i1;
  for (var iter_i1: i32 = 0; iter_i1 < maxIter_i1; iter_i1 = (iter_i1 + 1)) {
    var magSquared_i1: f32 = ((zRe_i1 * zRe_i1) + (zIm_i1 * zIm_i1));
    var diverged_i1: bool = (magSquared_i1 > 4.0);
    if (diverged_i1) {
      iterations_i1 = iter_i1;
      break;
    }
    var newRe_i1: f32 = ((zRe_i1 * zRe_i1) - (zIm_i1 * zIm_i1));
    var newIm_i1: f32 = ((2.0 * zRe_i1) * zIm_i1);
    zRe_i1 = (cRe_i1 + newRe_i1);
    zIm_i1 = (cIm_i1 + newIm_i1);
  }
  iterations = iterations_i1;
  output[idx] = iterations;
}
`
	if k.WGSL != golden {
		t.Fatalf("WGSL mismatch.\n--- got ---\n%s\n--- want ---\n%s", k.WGSL, golden)
	}
	if k.Entry != "spmd_kernel_0" {
		t.Fatalf("Entry = %q", k.Entry)
	}
	if k.ParamsSize != 32 {
		t.Fatalf("ParamsSize = %d, want 32", k.ParamsSize)
	}
	wantParams := "n:i32 dx:f32 dy:f32 height:i32 maxIter:i32 width:i32"
	if got := strings.Join(paramNames(k.Params), " "); got != wantParams {
		t.Fatalf("Params = %q, want %q", got, wantParams)
	}
	wantBuffers := "output:array<i32>"
	if got := strings.Join(paramNames(k.Buffers), " "); got != wantBuffers {
		t.Fatalf("Buffers = %q, want %q", got, wantBuffers)
	}

	runNaga(t, k.WGSL)
}

// saxpyGPUSrc: out[idx] = a*xs[idx] + ys[idx], xs/ys read-only, out read_write.
const saxpyGPUSrc = `
package p

func saxpy(a float32, xs, ys, out []float32) {
	go for idx := range len(xs) {
		out[idx] = a*xs[idx] + ys[idx]
	}
}
`

func TestGPUWGSLSaxpyGolden(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, saxpyGPUSrc, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 1)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}

	const golden = `struct Params {
  n: i32,
  a: f32,
  _pad0: i32,
  _pad1: i32,
}
@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> xs: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<storage, read> ys: array<f32>;
@compute @workgroup_size(64)
fn spmd_kernel_1(@builtin(global_invocation_id) gid: vec3<u32>) {
  let idx: i32 = i32(gid.x);
  if (idx >= params.n) { return; }
  out[idx] = ((params.a * xs[idx]) + ys[idx]);
}
`
	if k.WGSL != golden {
		t.Fatalf("WGSL mismatch.\n--- got ---\n%s\n--- want ---\n%s", k.WGSL, golden)
	}
	wantBuffers := "xs:array<f32> out:array<f32> ys:array<f32>"
	if got := strings.Join(paramNames(k.Buffers), " "); got != wantBuffers {
		t.Fatalf("Buffers = %q, want %q", got, wantBuffers)
	}

	runNaga(t, k.WGSL)
}

// twoInlinedCallsGPUSrc exercises two inlined SPMD-call expansions in one
// kernel body (Fix round 1, Finding 2): declareLocal must give the two
// expansions of "double" distinct local names, or the second "var a_iN:"
// declaration would redeclare the first's locals in the same WGSL scope.
const twoInlinedCallsGPUSrc = `
package p

import "lanes"

func double(x lanes.Varying[float32]) lanes.Varying[float32] {
	a := x * 2.0
	return a
}

func run(n int, xs, out []float32) {
	go for idx := range n {
		p := double(lanes.Varying[float32](xs[idx]))
		q := double(p)
		out[idx] = q
	}
}
`

func TestGPUWGSLTwoInlinedCallsDistinctLocals(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, twoInlinedCallsGPUSrc, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 2)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}
	if !strings.Contains(k.WGSL, "a_i1") || !strings.Contains(k.WGSL, "a_i2") {
		t.Fatalf("expected distinct per-call-site locals a_i1 and a_i2, got:\n%s", k.WGSL)
	}
	if strings.Count(k.WGSL, "var a_i1: f32") != 1 || strings.Count(k.WGSL, "var a_i2: f32") != 1 {
		t.Fatalf("expected exactly one declaration each of a_i1/a_i2, got:\n%s", k.WGSL)
	}
	runNaga(t, k.WGSL)
}

// lanesCountRejectedGPUSrc / lanesIndexRejectedGPUSrc: Fix round 1, Finding
// 3 - lanes.Count and lanes.Index are lane-count-dependent (SIMD-width- and
// intra-vector-lane-index-scoped) and have no sound GPU meaning; they must
// be rejected by analyzeGPULoop, not silently miscompiled.
const lanesCountRejectedGPUSrc = `
package p

import "lanes"

func run(n int, xs []float32) {
	go for idx := range n {
		v := lanes.Varying[float32](xs[idx])
		c := lanes.Count(v)
		xs[idx] = float32(c)
	}
}
`

const lanesIndexRejectedGPUSrc = `
package p

import "lanes"

func run(n int, out []int) {
	go for idx := range n {
		li := lanes.Index()
		out[idx] = li
	}
}
`

func TestGPUEligibleLanesCountRejected(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, lanesCountRejectedGPUSrc, 1)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for lanes.Count, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "lane-count-dependent") || !strings.Contains(plan.Reject, "lanes.Count") {
		t.Fatalf("reject reason = %q, want it to mention lane-count-dependent and lanes.Count", plan.Reject)
	}
}

func TestGPUEligibleLanesIndexRejected(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, lanesIndexRejectedGPUSrc, 1)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for lanes.Index, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "lane-count-dependent") || !strings.Contains(plan.Reject, "lanes.Index") {
		t.Fatalf("reject reason = %q, want it to mention lane-count-dependent and lanes.Index", plan.Reject)
	}
}

// lanesFMAGPUSrc: lanes.FMA(a, xs[idx], ys[idx]) must transpile to WGSL's
// builtin fma(...) (Fix round 1, Finding 3).
const lanesFMAGPUSrc = `
package p

import "lanes"

func run(n int, a float32, xs, ys, out []float32) {
	go for idx := range n {
		av := lanes.Varying[float32](a)
		out[idx] = lanes.FMA(av, xs[idx], ys[idx])
	}
}
`

func TestGPUWGSLLanesFMA(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, lanesFMAGPUSrc, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 3)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}
	if !strings.Contains(k.WGSL, "fma(") {
		t.Fatalf("expected fma(...) in output, got:\n%s", k.WGSL)
	}
	runNaga(t, k.WGSL)
}

// --- Final-review fix wave: emission regressions ------------------------

// C2: a callee that assigns to its OWN parameter must not mutate the
// caller's local. emitInlinedCall used to bind each parameter to the
// ARGUMENT TEXT, so `x = x * 2` in the callee emitted `x = (x * 2);` into
// the CALLER's scope -- correct on the CPU path (Go parameters are by
// value), silently wrong on the GPU path.
const inlinedParamWriteGPUSrc = `
package p

import "lanes"

func twice(x lanes.Varying[int]) lanes.Varying[int] {
	x = x * 2
	return x
}

func run(n int, out []int) {
	go for idx := range n {
		x := idx + 1
		y := twice(x)
		out[idx] = y + x
	}
}
`

func TestGPUWGSLInlinedParamIsByValue(t *testing.T) {
	plan := parseAndAnalyzeGPULoop(t, inlinedParamWriteGPUSrc, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}
	// A fresh, suffixed copy of the parameter must be declared and written.
	if !strings.Contains(k.WGSL, "var x_i1: i32 = x;") {
		t.Errorf("missing by-value parameter copy `var x_i1: i32 = x;`\n%s", k.WGSL)
	}
	if !strings.Contains(k.WGSL, "x_i1 = (x_i1 * 2);") {
		t.Errorf("callee write should target the parameter copy\n%s", k.WGSL)
	}
	// And the caller's own `x` must NEVER be assigned by the callee body.
	if strings.Contains(k.WGSL, "\n  x = (x * 2);") || strings.Contains(k.WGSL, "  x = (x * 2);\n") {
		t.Errorf("callee mutated the caller's local x (by-value semantics broken)\n%s", k.WGSL)
	}
	runNaga(t, k.WGSL)
}

// I5: buffer identifiers are emitted verbatim into the module, so a Go slice
// named `params` collided with the mandatory uniform binding and one named
// after a WGSL keyword produced an invalid shader -- which, before I2, was
// an entirely silent wrong answer.
func TestGPUWGSLBufferNameHygiene(t *testing.T) {
	src := `package p

func run(n int, params []int, loop []int) {
	go for idx := range n {
		params[idx] = loop[idx] + idx
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}
	// The uniform binding keeps the name `params`; the slice is renamed.
	if !strings.Contains(k.WGSL, "var<uniform> params: Params;") {
		t.Errorf("uniform binding lost its name\n%s", k.WGSL)
	}
	if strings.Contains(k.WGSL, "> params: array<i32>") {
		t.Errorf("slice named `params` collided with the uniform binding\n%s", k.WGSL)
	}
	if !strings.Contains(k.WGSL, "params_1: array<i32>") || !strings.Contains(k.WGSL, "params_1[idx]") {
		t.Errorf("slice named `params` was not uniquified\n%s", k.WGSL)
	}
	// `loop` is a WGSL reserved word.
	if strings.Contains(k.WGSL, "> loop: array<i32>") {
		t.Errorf("slice named after the WGSL reserved word `loop` emitted verbatim\n%s", k.WGSL)
	}
	if !strings.Contains(k.WGSL, "loop_: array<i32>") || !strings.Contains(k.WGSL, "loop_[idx]") {
		t.Errorf("reserved-word slice name was not escaped\n%s", k.WGSL)
	}
	runNaga(t, k.WGSL)
}

// C1 (emitter half): every operator gpu_eligible.go declares eligible must
// have an explicit mapping, and every other operator must be reported as an
// internal error rather than silently lowered to a plain `=`.
func TestGPUWGSLAssignOperatorAllowlistsAgree(t *testing.T) {
	for tok := range gpuEligibleAssignToks {
		if tok == token.DEFINE || tok == token.ASSIGN {
			continue
		}
		if _, ok := compoundAssignOp(tok); !ok {
			t.Errorf("%s is eligible but has no compoundAssignOp mapping", tok)
		}
	}
	for _, tok := range []token.Token{
		token.OR_ASSIGN, token.AND_ASSIGN, token.XOR_ASSIGN,
		token.SHL_ASSIGN, token.SHR_ASSIGN, token.AND_NOT_ASSIGN,
	} {
		if _, ok := compoundAssignOp(tok); ok {
			t.Errorf("%s must not be mapped without also being tested end to end", tok)
		}
		if gpuEligibleAssignToks[tok] {
			t.Errorf("%s must not be eligible while unmapped", tok)
		}
	}
}
