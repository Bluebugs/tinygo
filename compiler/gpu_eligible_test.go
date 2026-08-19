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
