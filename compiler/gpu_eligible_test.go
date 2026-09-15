// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/ast"
	"go/importer"
	"go/parser"
	"go/token"
	"go/types"
	"strings"
	"testing"
)

// parseAndAnalyzeGPULoop parses src (which must contain exactly one `go for`
// loop), type-checks it with the standard library importer (lanes/reduce are
// part of the forked GOROOT, so importer.Default() resolves them), builds a
// *SPMDLoopInfo the same way extractSPMDLoops does, and runs analyzeGPULoop
// on it.
func parseAndAnalyzeGPULoop(t *testing.T, src string, thresholdOps uint64) *gpuLoopPlan {
	t.Helper()
	t.Setenv("GOEXPERIMENT", "spmd")

	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, "test.go", src, 0)
	if err != nil {
		t.Fatalf("parse error: %v", err)
	}

	info := &types.Info{
		Types: map[ast.Expr]types.TypeAndValue{},
		Defs:  map[*ast.Ident]types.Object{},
		Uses:  map[*ast.Ident]types.Object{},
	}
	conf := types.Config{Importer: importer.Default()}
	if _, err := conf.Check("test", fset, []*ast.File{f}, info); err != nil {
		t.Fatalf("type-check error: %v", err)
	}

	var rangeStmt *ast.RangeStmt
	ast.Inspect(f, func(n ast.Node) bool {
		if rs, ok := n.(*ast.RangeStmt); ok && rs.IsSpmd {
			rangeStmt = rs
			return false
		}
		return true
	})
	if rangeStmt == nil {
		t.Fatalf("no `go for` loop found in test source")
	}

	loopInfo := &SPMDLoopInfo{
		ForPos:    rangeStmt.For,
		BodyStart: rangeStmt.Body.Lbrace,
		BodyEnd:   rangeStmt.Body.Rbrace,
		RangeStmt: rangeStmt,
		TypesInfo: info,
		Files:     []*ast.File{f},
	}

	return analyzeGPULoop(loopInfo, thresholdOps)
}

func freeVarKind(t *testing.T, plan *gpuLoopPlan, name string) (gpuFreeVar, bool) {
	t.Helper()
	for _, fv := range plan.Free {
		if fv.Obj.Name() == name {
			return fv, true
		}
	}
	return gpuFreeVar{}, false
}

// (a) constant-bound version of a flat-mandelbrot-shaped loop: eligible,
// Free classifies output as gpuSliceRW and width/height/maxIter as
// gpuScalar, and BodyCost uses the constant inner-loop bound (8) as the
// multiplier, not the ×16 dynamic-bound fallback.
func TestGPUEligibleFlatMandelbrotConstantBound(t *testing.T) {
	src := `package test

import "lanes"

func kernel(width, height, maxIter int, output []int) {
	go for idx := range width * height {
		i := idx % width
		j := idx / width
		v := lanes.Varying[int](i + j)
		for k := 0; k < 8; k++ {
			v = v + lanes.Varying[int](maxIter)
		}
		output[idx] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject != "" {
		t.Fatalf("expected eligible loop, got Reject=%q", plan.Reject)
	}

	out, ok := freeVarKind(t, plan, "output")
	if !ok {
		t.Fatalf("expected free var %q, Free=%+v", "output", plan.Free)
	}
	if out.Kind != gpuSliceRW {
		t.Errorf("output: expected gpuSliceRW, got %d", out.Kind)
	}
	if out.WGSLTy != "array<i32>" {
		t.Errorf("output: expected WGSLTy array<i32>, got %q", out.WGSLTy)
	}

	for _, name := range []string{"width", "height", "maxIter"} {
		fv, ok := freeVarKind(t, plan, name)
		if !ok {
			t.Fatalf("expected free var %q, Free=%+v", name, plan.Free)
		}
		if fv.Kind != gpuScalar {
			t.Errorf("%s: expected gpuScalar, got %d", name, fv.Kind)
		}
		if fv.WGSLTy != "i32" {
			t.Errorf("%s: expected WGSLTy i32, got %q", name, fv.WGSLTy)
		}
	}

	// idx % width (REM, weight 8) + idx / width (QUO, weight 8) +
	// i + j (ADD, weight 1) = 17 outside the inner loop, plus the inner
	// loop's "v + maxIter" (ADD, weight 1) multiplied by its constant
	// bound 8 = 8, for a total of 25.
	const want = 25
	if plan.BodyCost != want {
		t.Errorf("BodyCost = %d, want %d", plan.BodyCost, want)
	}
	wantMinTrip := (thresholdCeilDiv(1_000_000, want))
	if plan.MinTrip != wantMinTrip {
		t.Errorf("MinTrip = %d, want %d", plan.MinTrip, wantMinTrip)
	}
}

// thresholdCeilDiv mirrors the ceil(threshold/bodyCost) rule from §D3, used
// only to compute the expected value in the test above without duplicating
// analyzeGPULoop's internals.
func thresholdCeilDiv(threshold, bodyCost uint64) uint64 {
	if bodyCost == 0 {
		bodyCost = 1
	}
	trip := (threshold + bodyCost - 1) / bodyCost
	if trip < 1 {
		trip = 1
	}
	return trip
}

// Same loop shape as above, but the inner loop's bound is a variable
// (dynamic), not a constant literal, so the ×16 fallback multiplier
// applies instead of the constant bound 8. BodyCost must come out larger
// than the constant-bound case's inner contribution (8) scaled by 16
// instead of 8.
func TestGPUEligibleDynamicBound(t *testing.T) {
	src := `package test

import "lanes"

func kernel(width, height, maxIter int, output []int) {
	go for idx := range width * height {
		i := idx % width
		j := idx / width
		v := lanes.Varying[int](i + j)
		for k := 0; k < maxIter; k++ {
			v = v + lanes.Varying[int](maxIter)
		}
		output[idx] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject != "" {
		t.Fatalf("expected eligible loop, got Reject=%q", plan.Reject)
	}

	// Same 17 outside the inner loop, plus inner "v + maxIter" (weight 1)
	// times the ×16 dynamic-bound fallback = 16, for a total of 33.
	const want = 33
	if plan.BodyCost != want {
		t.Errorf("BodyCost = %d, want %d", plan.BodyCost, want)
	}
}

// (b) a body using a cross-lane op (lanes.Rotate) must be rejected, with the
// reason mentioning "cross-lane".
func TestGPUEligibleCrossLaneRejected(t *testing.T) {
	src := `package test

import "lanes"

func kernel(n int, output []int) {
	go for idx := range n {
		v := lanes.Varying[int](idx)
		output[idx] = lanes.Rotate(v, 1)
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for cross-lane op, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "cross-lane") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "cross-lane")
	}
}

// (c) a body using reduce.Add must be rejected.
func TestGPUEligibleReduceRejected(t *testing.T) {
	src := `package test

import (
	"lanes"
	"reduce"
)

func kernel(n int, output []int) {
	go for idx := range n {
		v := lanes.Varying[int](idx)
		output[idx] = reduce.Add(v)
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for reduce.Add, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "reduce") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "reduce")
	}
}

// (d) a body that calls an SPMD function defined in the same package must
// have that call recorded in Inlined.
func TestGPUEligibleInlinedSPMDCall(t *testing.T) {
	src := `package test

import "lanes"

func double(v lanes.Varying[int]) lanes.Varying[int] {
	return v + v
}

func kernel(n int, output []int) {
	go for idx := range n {
		v := lanes.Varying[int](idx)
		output[idx] = double(v)
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject != "" {
		t.Fatalf("expected eligible loop, got Reject=%q", plan.Reject)
	}
	if len(plan.Inlined) != 1 {
		t.Fatalf("expected 1 inlined call, got %d: %+v", len(plan.Inlined), plan.Inlined)
	}
	for call, decl := range plan.Inlined {
		if decl == nil || decl.Name == nil || decl.Name.Name != "double" {
			t.Errorf("expected inlined call to resolve to func %q, got %+v (call=%v)", "double", decl, call)
		}
	}
}

// (e) a body calling fmt.Println must be rejected.
func TestGPUEligibleFmtRejected(t *testing.T) {
	src := `package test

import (
	"fmt"
	"lanes"
)

func kernel(n int, output []int) {
	go for idx := range n {
		v := lanes.Varying[int](idx)
		fmt.Println(idx)
		output[idx] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for fmt.Println, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "fmt") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "fmt")
	}
}

// (f) a body reading/writing a free []float64 slice must be rejected
// because float64 is unsupported in WGSL.
func TestGPUEligibleFloat64SliceRejected(t *testing.T) {
	src := `package test

import "lanes"

func kernel(n int, output []float64) {
	go for idx := range n {
		v := lanes.Varying[float64](idx)
		output[idx] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for []float64, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "float64") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "float64")
	}
}

// Fix round 1 additions: the eligibility walk is now a default-REJECT
// allowlist (Critical 1), slice writes via IncDecStmt are detected
// (Critical 2), slice arguments to function calls are rejected outright
// (Critical 2), and writes to free scalars are rejected (Critical 3).

// An external, non-approved package function call must be rejected.
func TestGPUEligibleExternalCallRejected(t *testing.T) {
	src := `package test

import "strconv"

func kernel(n int, output []int) {
	go for idx := range n {
		strconv.Itoa(n)
		output[idx] = idx
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for strconv.Itoa, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "unapproved package function") || !strings.Contains(plan.Reject, "strconv") {
		t.Errorf("Reject = %q, want it to mention %q and %q", plan.Reject, "unapproved package function", "strconv")
	}
}

// A closure literal must be rejected.
func TestGPUEligibleClosureRejected(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		g := func() int { return 1 }
		output[idx] = idx + g()
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for closure, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "closure") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "closure")
	}
}

// A method call must be rejected.
func TestGPUEligibleMethodCallRejected(t *testing.T) {
	src := `package test

type counter struct{}

func (c counter) get() int { return 1 }

func kernel(n int, output []int) {
	go for idx := range n {
		var c counter
		output[idx] = idx + c.get()
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for method call, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "method call") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "method call")
	}
}

// A type assertion must be rejected.
func TestGPUEligibleTypeAssertionRejected(t *testing.T) {
	src := `package test

func kernel(x interface{}, n int, output []int) {
	go for idx := range n {
		v := x.(int)
		output[idx] = idx + v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for type assertion, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "type assertion") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "type assertion")
	}
}

// A pointer dereference must be rejected.
func TestGPUEligiblePointerDerefRejected(t *testing.T) {
	src := `package test

func kernel(p *int, n int, output []int) {
	go for idx := range n {
		output[idx] = idx + *p
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for pointer dereference, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "pointer dereference") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "pointer dereference")
	}
}

// "output[idx]++" must be detected as a write, classifying output as
// gpuSliceRW (not the misclassified gpuSliceRead a plain AssignStmt-only
// scan would produce).
func TestGPUEligibleIncDecSliceWriteClassifiedRW(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx]++
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject != "" {
		t.Fatalf("expected eligible loop, got Reject=%q", plan.Reject)
	}
	out, ok := freeVarKind(t, plan, "output")
	if !ok {
		t.Fatalf("expected free var %q, Free=%+v", "output", plan.Free)
	}
	if out.Kind != gpuSliceRW {
		t.Errorf("output: expected gpuSliceRW (from output[idx]++), got %d", out.Kind)
	}
}

// Passing a free slice to a function call must be rejected, even when the
// callee is itself an inlinable local function (no free-var/write analysis
// is threaded into inlined callee bodies yet, so allowing this would let a
// write inside the callee go undetected).
func TestGPUEligibleSliceArgumentToCallRejected(t *testing.T) {
	src := `package test

func helper(s []int) {
}

func kernel(n int, output []int) {
	go for idx := range n {
		helper(output)
		output[idx] = idx
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for slice argument to call, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "slice or pointer argument") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "slice or pointer argument")
	}
}

// A write to a free scalar (a uniform-buffer parameter, read-only from the
// GPU's perspective) must be rejected.
func TestGPUEligibleWrittenFreeScalarRejected(t *testing.T) {
	src := `package test

func kernel(maxIter, n int, output []int) {
	go for idx := range n {
		maxIter = maxIter / 2
		output[idx] = idx + maxIter
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject == "" {
		t.Fatalf("expected rejection for written free scalar, got eligible plan")
	}
	if !strings.Contains(plan.Reject, "free scalar variable written") {
		t.Errorf("Reject = %q, want it to mention %q", plan.Reject, "free scalar variable written")
	}
}

// --- Final-review fix wave: fail-closed eligibility regressions ----------

// C1: a bitwise/shift compound assignment must be REJECTED. compoundAssignOp
// maps only += -= *= /= %=; before this fix `acc |= mask` was declared
// eligible and emitted as a plain `acc = mask;` -- valid WGSL, wrong answer,
// no diagnostic anywhere.
func TestGPUEligibleBitwiseCompoundAssignRejected(t *testing.T) {
	for _, tc := range []struct{ name, op string }{
		{"or", "|="},
		{"and", "&="},
		{"xor", "^="},
		{"shl", "<<="},
		{"shr", ">>="},
		{"andnot", "&^="},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := `package test

func kernel(n int, out []int) {
	go for idx := range n {
		acc := idx
		acc ` + tc.op + ` 7
		out[idx] = acc
	}
}
`
			plan := parseAndAnalyzeGPULoop(t, src, 1)
			if plan.Reject == "" {
				t.Fatalf("expected %s to be rejected, got eligible", tc.op)
			}
			if !strings.Contains(plan.Reject, "assignment operator") {
				t.Errorf("Reject = %q, want it to mention the assignment operator", plan.Reject)
			}
		})
	}
}

// The arithmetic compound assignments compoundAssignOp DOES map stay
// eligible -- the C1 fix must not over-reject.
func TestGPUEligibleArithmeticCompoundAssignAccepted(t *testing.T) {
	for _, op := range []string{"+=", "-=", "*=", "/=", "%="} {
		src := `package test

func kernel(n int, out []int) {
	go for idx := range n {
		acc := idx
		acc ` + op + ` 7
		out[idx] = acc
	}
}
`
		plan := parseAndAnalyzeGPULoop(t, src, 1)
		if plan.Reject != "" {
			t.Errorf("%s: expected eligible, got Reject=%q", op, plan.Reject)
		}
	}
}

// C3: a return that is not the final statement of an inlined callee body has
// no early exit in the emitter (it lowers to `retTarget = v;` and falls
// through), so both assignments would run and the LAST one would always win.
func TestGPUEligibleNonTailReturnRejected(t *testing.T) {
	src := `package test

import "lanes"

func pick(a lanes.Varying[int]) lanes.Varying[int] {
	if a > 3 {
		return a
	}
	return 0
}

func kernel(n int, out []int) {
	go for idx := range n {
		v := pick(idx)
		out[idx] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject == "" {
		t.Fatalf("expected non-tail return to be rejected, got eligible")
	}
	if !strings.Contains(plan.Reject, "non-tail return") {
		t.Errorf("Reject = %q, want it to mention a non-tail return", plan.Reject)
	}
}

// A callee whose ONLY return is its final statement stays eligible.
func TestGPUEligibleTailReturnAccepted(t *testing.T) {
	src := `package test

import "lanes"

func twice(a lanes.Varying[int]) lanes.Varying[int] {
	b := a * 2
	return b
}

func kernel(n int, out []int) {
	go for idx := range n {
		v := twice(idx)
		out[idx] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject != "" {
		t.Fatalf("expected eligible, got Reject=%q", plan.Reject)
	}
}

// I1: body-local variables were never type-checked against WGSL (classify
// only ever saw FREE vars), so a local `var c int64` / `float64` reached
// wgslTypeOfIdent, whose error was swallowed into a hardcoded "i32" --
// a silent truncation on the GPU path only.
func TestGPUEligibleBodyLocalWideTypeRejected(t *testing.T) {
	for _, tc := range []struct{ name, decl string }{
		{"int64", "var c lanes.Varying[int64] = int64(idx)"},
		{"float64", "var c lanes.Varying[float64] = float64(idx)"},
		{"int64-shortdecl", "c := int64(idx)"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			imp := ""
			if strings.Contains(tc.decl, "lanes.") {
				imp = "import \"lanes\"\n\n"
			}
			src := `package test

` + imp + `func kernel(n int, out []int) {
	go for idx := range n {
		` + tc.decl + `
		out[idx] = int(c)
	}
}
`
			plan := parseAndAnalyzeGPULoop(t, src, 1)
			if plan.Reject == "" {
				t.Fatalf("expected %s local to be rejected, got eligible", tc.name)
			}
			if !strings.Contains(plan.Reject, "not representable in WGSL") {
				t.Errorf("Reject = %q, want it to mention WGSL representability", plan.Reject)
			}
		})
	}
}

// I1 (inlined callee): a wide local inside an inlined callee body must be
// rejected too.
func TestGPUEligibleInlinedCalleeWideLocalRejected(t *testing.T) {
	src := `package test

import "lanes"

func widen(a lanes.Varying[int]) lanes.Varying[int] {
	var c lanes.Varying[float64] = lanes.Varying[float64](a)
	return lanes.Varying[int](c)
}

func kernel(n int, out []int) {
	go for idx := range n {
		v := widen(idx)
		out[idx] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject == "" {
		t.Fatalf("expected eligible=false, got eligible")
	}
	if !strings.Contains(plan.Reject, "not representable in WGSL") {
		t.Errorf("Reject = %q, want it to mention WGSL representability", plan.Reject)
	}
}

// I4: lookupLocalFunc used to match by NAME, so a local (here: a parameter)
// shadowing a package-level function inlined the WRONG body. Resolution now
// goes through the type-checker object, so the shadowed call is simply not a
// recognized target and the loop falls back to the CPU.
func TestGPUEligibleShadowedFuncNameNotInlined(t *testing.T) {
	src := `package test

import "lanes"

func double(x lanes.Varying[int]) lanes.Varying[int] {
	return x * 2
}

func kernel(n int, out []int, double func(lanes.Varying[int]) lanes.Varying[int]) {
	go for idx := range n {
		out[idx] = double(idx)
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject == "" {
		t.Fatalf("expected the shadowed call to be rejected, got eligible")
	}
	if !strings.Contains(plan.Reject, "unrecognized function") {
		t.Errorf("Reject = %q, want it to mention an unrecognized function", plan.Reject)
	}
	if len(plan.Inlined) != 0 {
		t.Errorf("expected no inlined callee, got %d", len(plan.Inlined))
	}
}

// --- write-only buffer classification (upload-skipping optimization) ------
//
// A gpuSliceRW free variable is additionally flagged WriteOnly when the
// kernel provably writes every element of [0, n) and never reads the slice.
// The flag only makes the buffer a CANDIDATE -- gpuBuildBuffers turns it
// into a runtime `n == len(slice)` check that picks descriptor mode 2 (skip
// the upload) or mode 1 (upload, as before). These tests pin the classifier
// itself: every condition that makes the optimization unsound must clear
// the flag.

func gpuWriteOnlyFlag(t *testing.T, src, name string) (gpuFreeVar, *gpuLoopPlan) {
	t.Helper()
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject != "" {
		t.Fatalf("expected eligible loop, got Reject=%q", plan.Reject)
	}
	fv, ok := freeVarKind(t, plan, name)
	if !ok {
		t.Fatalf("expected free var %q, Free=%+v", name, plan.Free)
	}
	if fv.Kind != gpuSliceRW {
		t.Fatalf("%s: expected gpuSliceRW, got %d", name, fv.Kind)
	}
	return fv, plan
}

// The canonical accepted shape: an unconditional `output[idx] = expr` at
// loop-body top level, with the slice never read. This is exactly
// mandelbrotFlat's and intOffloadFlat's store.
func TestGPUEligibleWriteOnlyBareIndexStoreAccepted(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx] = idx*3 + 7
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if !out.WriteOnly {
		t.Errorf("output: expected WriteOnly=true for top-level output[idx] = expr")
	}
}

// A slice that is READ as well as written must not be write-only: skipping
// the upload would make the kernel observe undefined device memory.
func TestGPUEligibleWriteOnlyRejectedWhenRead(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx] = output[idx] + 1
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if out.WriteOnly {
		t.Errorf("output: expected WriteOnly=false, slice is read on the RHS")
	}
}

// A read through a *different* index could observe another invocation's
// write, so the whole loop is ineligible (not merely non-write-only).
func TestGPUEligibleWriteOnlyRejectedWhenReadAtOtherIndex(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx] = output[0] + 1
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if !strings.Contains(plan.Reject, "both read and written") {
		t.Errorf("Reject = %q, want a read-while-written rejection", plan.Reject)
	}
}

// A conditional store writes only some elements of [0, n); the rest would
// come back as device garbage.
func TestGPUEligibleWriteOnlyRejectedWhenConditional(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		if idx > 3 {
			output[idx] = idx
		}
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if out.WriteOnly {
		t.Errorf("output: expected WriteOnly=false, store is nested inside an if")
	}
}

// A `continue` at loop-body depth is explicitly ELIGIBLE (checkStmt allows
// it), and it makes every following statement conditional on control
// reaching it -- including a store that is syntactically a direct child of
// body.List. `output[idx] = idx` below runs only for idx <= 3, so [4, n) is
// never written and mode 2 would read back undefined device memory over the
// caller's live data. Fail closed.
func TestGPUEligibleWriteOnlyRejectedAfterContinue(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		if idx > 3 {
			continue
		}
		output[idx] = idx
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if out.WriteOnly {
		t.Errorf("output: expected WriteOnly=false, a body-level continue makes the store conditional")
	}
}

// A `continue` bound to a NESTED loop cannot skip the body-level store, so
// it must not disqualify the candidate -- otherwise the guard would be
// needlessly broad.
func TestGPUEligibleWriteOnlyAllowedWithNestedLoopContinue(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		acc := 0
		for k := 0; k < 8; k++ {
			if k == 3 {
				continue
			}
			acc += k
		}
		output[idx] = acc
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if !out.WriteOnly {
		t.Errorf("output: expected WriteOnly=true, the continue binds to the nested for")
	}
}

// A store at a derived index need not cover [0, n). output[idx/2] is also
// genuinely racy (idx=0 and idx=1 both write output[0]), so the write-index
// distinctness gate (gpu_write_index.go) now rejects the loop before the
// WriteOnly classification is even reached.
func TestGPUEligibleWriteOnlyRejectedWhenNonIdentityIndex(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx/2] = idx
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if !strings.Contains(plan.Reject, "slice write index") {
		t.Fatalf("Reject = %q, want a slice write index rejection (idx/2 is not affine)", plan.Reject)
	}
}

// `output[idx]++` reads the old value, so it is not write-only (and it is
// also not a plain `=` store).
func TestGPUEligibleWriteOnlyRejectedForIncDec(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx]++
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if out.WriteOnly {
		t.Errorf("output: expected WriteOnly=false for output[idx]++")
	}
}

// A compound assignment reads the old value too.
func TestGPUEligibleWriteOnlyRejectedForCompoundAssign(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx] += idx
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if out.WriteOnly {
		t.Errorf("output: expected WriteOnly=false for output[idx] += idx")
	}
}

// A second, differently-indexed store disqualifies: the recognised
// top-level store is no longer the object's only write. output[idx] and
// output[1] also genuinely collide (at idx=1), so the write-index
// distinctness gate (gpu_write_index.go) now rejects the loop before the
// WriteOnly classification is even reached.
func TestGPUEligibleWriteOnlyRejectedForSecondStore(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		output[idx] = idx
		if idx == 0 {
			output[1] = 9
		}
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if !strings.Contains(plan.Reject, "slice write index") {
		t.Fatalf("Reject = %q, want a slice write index rejection (output[idx] and output[1] can collide)", plan.Reject)
	}
}

// A store nested inside an inner `for` is not top-level, so its coverage of
// [0, n) is not established by this classifier.
func TestGPUEligibleWriteOnlyRejectedInsideNestedLoop(t *testing.T) {
	src := `package test

func kernel(n int, output []int) {
	go for idx := range n {
		for k := 0; k < 4; k++ {
			output[idx] = k
		}
	}
}
`
	out, _ := gpuWriteOnlyFlag(t, src, "output")
	if out.WriteOnly {
		t.Errorf("output: expected WriteOnly=false, store is inside a nested for")
	}
}

// A read-only slice is never flagged: there is nothing to skip uploading,
// and its contents are exactly what the kernel needs.
func TestGPUEligibleWriteOnlyNotSetForReadOnlySlice(t *testing.T) {
	src := `package test

func kernel(n int, input []int, output []int) {
	go for idx := range n {
		output[idx] = input[idx] * 2
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1_000_000)
	if plan.Reject != "" {
		t.Fatalf("expected eligible loop, got Reject=%q", plan.Reject)
	}
	in, ok := freeVarKind(t, plan, "input")
	if !ok {
		t.Fatalf("expected free var input, Free=%+v", plan.Free)
	}
	if in.Kind != gpuSliceRead {
		t.Errorf("input: expected gpuSliceRead, got %d", in.Kind)
	}
	if in.WriteOnly {
		t.Errorf("input: WriteOnly must never be set on a read-only slice")
	}
	out, ok := freeVarKind(t, plan, "output")
	if !ok {
		t.Fatalf("expected free var output, Free=%+v", plan.Free)
	}
	if !out.WriteOnly {
		t.Errorf("output: expected WriteOnly=true (input is read, output is not)")
	}
}
