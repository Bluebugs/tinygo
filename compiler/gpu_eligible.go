// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

// This file analyzes a `go for` loop's body (pure go/ast + go/types, no
// LLVM) to decide whether it is eligible for GPU offload and, if so,
// estimates its compile-time cost per §D3/§D4 of the GPU-offload design
// (docs/superpowers/plans/2026-08-18-gpu-offload-go-for.md). It produces a
// gpuLoopPlan that later tasks (gpu_wgsl.go, gpu_offload.go) consume to
// generate the WGSL kernel and the dual-path emission.

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
)

// gpuFreeVar classifies a single free variable referenced by an
// offload-eligible loop body.
type gpuFreeVar struct {
	Obj    types.Object
	Kind   int    // gpuScalar | gpuSliceRead | gpuSliceRW
	WGSLTy string // "i32", "u32", "f32", "array<i32>", "array<f32>"
}

const (
	gpuScalar    = iota // uniform scalar, becomes a Params uniform-buffer field
	gpuSliceRead        // free slice only read inside the body
	gpuSliceRW          // free slice written (also) inside the body
)

// gpuLoopPlan is the result of analyzeGPULoop: either an eligible loop with
// its cost estimate and free-variable classification, or a rejection
// reason.
type gpuLoopPlan struct {
	Loop      *SPMDLoopInfo
	Body      *ast.BlockStmt                  // the go-for RangeStmt body
	IterIdent *ast.Ident                      // loop variable
	Inlined   map[*ast.CallExpr]*ast.FuncDecl // SPMD fn calls to inline
	BodyCost  uint64
	MinTrip   uint64 // ceil(threshold / BodyCost), clamped >= 1
	Free      []gpuFreeVar
	Reject    string // "" = eligible; else human-readable reason
}

// gpuCrossLaneOps is the set of lanes.* functions that are meaningless on a
// GPU where one invocation is one lane and workgroups don't align with SIMD
// lane groups (§D4).
var gpuCrossLaneOps = map[string]bool{
	"From":             true,
	"Broadcast":        true,
	"Rotate":           true,
	"Swizzle":          true,
	"RotateWithin":     true,
	"SwizzleWithin":    true,
	"ShiftLeftWithin":  true,
	"ShiftRightWithin": true,
	"ShiftLeft":        true,
	"ShiftRight":       true,
}

// analyzeGPULoop decides whether the `go for` loop described by info is
// eligible for GPU offload and, if so, computes its cost estimate and
// free-variable classification. It consumes only info (RangeStmt +
// TypesInfo, both stashed by extractSPMDLoops), never a *loader.Package.
func analyzeGPULoop(info *SPMDLoopInfo, thresholdOps uint64) *gpuLoopPlan {
	plan := &gpuLoopPlan{
		Loop:    info,
		Inlined: map[*ast.CallExpr]*ast.FuncDecl{},
	}

	if info == nil || info.RangeStmt == nil || info.TypesInfo == nil {
		plan.Reject = "missing AST or type information for GPU eligibility analysis"
		return plan
	}

	rangeStmt := info.RangeStmt
	plan.Body = rangeStmt.Body
	if ident, ok := rangeStmt.Key.(*ast.Ident); ok {
		plan.IterIdent = ident
	}

	a := &gpuAnalyzer{
		info:      info.TypesInfo,
		files:     info.Files,
		rangeStmt: rangeStmt,
		inlined:   plan.Inlined,
	}

	// §D4 eligibility walk (also inlines same-package SPMD calls one level
	// deep).
	if reject := a.checkEligible(rangeStmt.Body, 0); reject != "" {
		plan.Reject = reject
		return plan
	}

	// Free-variable discovery + classification.
	free, reject := a.freeVars(rangeStmt)
	if reject != "" {
		plan.Reject = reject
		return plan
	}
	plan.Free = free

	// §D3 cost estimate.
	plan.BodyCost = a.bodyCost(rangeStmt.Body.List)
	if plan.BodyCost == 0 {
		plan.BodyCost = 1
	}
	trip := (thresholdOps + plan.BodyCost - 1) / plan.BodyCost
	if trip < 1 {
		trip = 1
	}
	plan.MinTrip = trip

	return plan
}

// gpuAnalyzer carries the shared state for the eligibility walk, free-var
// discovery, and cost estimate.
type gpuAnalyzer struct {
	info      *types.Info
	files     []*ast.File
	rangeStmt *ast.RangeStmt
	inlined   map[*ast.CallExpr]*ast.FuncDecl
}

// pkgNameOf resolves a *ast.Ident used as the package qualifier of a
// SelectorExpr (e.g. the "lanes" in "lanes.Rotate") to its import path,
// or "" if it does not resolve to a package name.
func (a *gpuAnalyzer) pkgNameOf(ident *ast.Ident) string {
	obj := a.info.Uses[ident]
	if pkgName, ok := obj.(*types.PkgName); ok {
		return pkgName.Imported().Path()
	}
	return ""
}

// checkEligible walks node (a loop body or an inlined callee body) enforcing
// §D4 as a default-REJECT allowlist: every statement and expression must be
// explicitly recognized as safe, or the loop is rejected. This is
// deliberately conservative — analyzeGPULoop is a correctness GATE (a wrong
// "eligible" verdict silently runs a loop on the GPU and can produce wrong
// results; a wrong "ineligible" verdict only costs a missed optimization).
// depth is the SPMD-call inlining depth (0 = the loop body itself, 1 = one
// level of inlining; depth > 1 is rejected).
func (a *gpuAnalyzer) checkEligible(node ast.Node, depth int) string {
	switch x := node.(type) {
	case *ast.BlockStmt:
		for _, stmt := range x.List {
			if reject := a.checkEligible(stmt, depth); reject != "" {
				return reject
			}
		}
		return ""
	case nil:
		return ""
	}

	if stmt, ok := node.(ast.Stmt); ok {
		return a.checkStmt(stmt, depth)
	}
	if expr, ok := node.(ast.Expr); ok {
		return a.checkExpr(expr, depth)
	}
	return fmt.Sprintf("unsupported AST node not eligible for GPU offload: %T", node)
}

// checkStmt allowlists the statement forms §D4 permits. Anything not listed
// here is rejected by the default case.
func (a *gpuAnalyzer) checkStmt(stmt ast.Stmt, depth int) string {
	switch x := stmt.(type) {
	case *ast.BlockStmt:
		return a.checkEligible(x, depth)

	case *ast.ExprStmt:
		return a.checkExpr(x.X, depth)

	case *ast.AssignStmt:
		for _, rhs := range x.Rhs {
			if reject := a.checkExpr(rhs, depth); reject != "" {
				return reject
			}
		}
		for _, lhs := range x.Lhs {
			switch lv := lhs.(type) {
			case *ast.Ident:
				// Assignment to a local or free scalar; free-scalar writes
				// are separately rejected by writtenScalarObjs/classify.
				continue
			case *ast.IndexExpr:
				if reject := a.checkExpr(lv.X, depth); reject != "" {
					return reject
				}
				if reject := a.checkExpr(lv.Index, depth); reject != "" {
					return reject
				}
			default:
				return fmt.Sprintf("unsupported assignment target not eligible for GPU offload: %T", lhs)
			}
		}
		return ""

	case *ast.IncDecStmt:
		switch tv := x.X.(type) {
		case *ast.Ident:
			return ""
		case *ast.IndexExpr:
			if reject := a.checkExpr(tv.X, depth); reject != "" {
				return reject
			}
			return a.checkExpr(tv.Index, depth)
		default:
			return fmt.Sprintf("unsupported increment/decrement target not eligible for GPU offload: %T", x.X)
		}

	case *ast.DeclStmt:
		gen, ok := x.Decl.(*ast.GenDecl)
		if !ok {
			return "unsupported local declaration not eligible for GPU offload"
		}
		if gen.Tok == token.TYPE {
			return "local type declaration not eligible for GPU offload"
		}
		for _, spec := range gen.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				continue
			}
			for _, v := range vs.Values {
				if reject := a.checkExpr(v, depth); reject != "" {
					return reject
				}
			}
		}
		return ""

	case *ast.IfStmt:
		if x.Init != nil {
			if reject := a.checkStmt(x.Init, depth); reject != "" {
				return reject
			}
		}
		if reject := a.checkExpr(x.Cond, depth); reject != "" {
			return reject
		}
		if reject := a.checkEligible(x.Body, depth); reject != "" {
			return reject
		}
		if x.Else != nil {
			return a.checkStmt(x.Else, depth)
		}
		return ""

	case *ast.ForStmt:
		if x.Init != nil {
			if reject := a.checkStmt(x.Init, depth); reject != "" {
				return reject
			}
		}
		if x.Cond != nil {
			if reject := a.checkExpr(x.Cond, depth); reject != "" {
				return reject
			}
		}
		if x.Post != nil {
			if reject := a.checkStmt(x.Post, depth); reject != "" {
				return reject
			}
		}
		return a.checkEligible(x.Body, depth)

	case *ast.RangeStmt:
		// Narrowly allowlist "for i := range n" over an integer bound
		// (Go 1.22+ range-over-int) as a nested loop inside an inlined
		// SPMD callee (e.g. mandelSPMD's "for iter := range maxIter"
		// divergence loop) -- NOT range over a slice/map/string/channel/
		// func, and NOT a two-variable "for i, v := range ...", both of
		// which have no meaning identical to a plain counted loop here.
		// Everything else about x is rejected by falling through to the
		// default case below via the explicit checks; the default switch
		// case is NOT reopened.
		if x.Value != nil || x.Tok != token.DEFINE {
			return "range-over-value or non-defining range not eligible for GPU offload"
		}
		if _, ok := x.Key.(*ast.Ident); !ok {
			return "unsupported range key not eligible for GPU offload"
		}
		xt := a.info.TypeOf(x.X)
		basic, ok := xt.Underlying().(*types.Basic)
		if !ok || basic.Info()&types.IsInteger == 0 {
			return "range-over-non-integer not eligible for GPU offload"
		}
		if reject := a.checkExpr(x.X, depth); reject != "" {
			return reject
		}
		return a.checkEligible(x.Body, depth)

	case *ast.ReturnStmt:
		// A return inside the loop body itself would exit the loop under a
		// varying condition, which §D4 forbids. A return inside an inlined
		// callee's body is just an ordinary function return and is fine.
		if depth == 0 {
			return "return not eligible for GPU offload"
		}
		for _, r := range x.Results {
			if reject := a.checkExpr(r, depth); reject != "" {
				return reject
			}
		}
		return ""

	case *ast.BranchStmt:
		switch x.Tok {
		case token.CONTINUE:
			return ""
		case token.BREAK:
			if depth == 0 {
				return "break not eligible for GPU offload"
			}
			return ""
		default:
			return fmt.Sprintf("branch statement %s not eligible for GPU offload", x.Tok)
		}

	default:
		return fmt.Sprintf("statement type %T not eligible for GPU offload", stmt)
	}
}

// checkExpr allowlists the expression forms §D4 permits. Anything not
// listed here is rejected by the default case.
func (a *gpuAnalyzer) checkExpr(expr ast.Expr, depth int) string {
	switch x := expr.(type) {
	case nil:
		return ""
	case *ast.Ident:
		return ""
	case *ast.BasicLit:
		return ""
	case *ast.ParenExpr:
		return a.checkExpr(x.X, depth)

	case *ast.BinaryExpr:
		if reject := a.checkExpr(x.X, depth); reject != "" {
			return reject
		}
		return a.checkExpr(x.Y, depth)

	case *ast.UnaryExpr:
		switch x.Op {
		case token.AND:
			return "address-of not eligible for GPU offload"
		case token.ARROW:
			return "channel receive not eligible for GPU offload"
		default:
			return a.checkExpr(x.X, depth)
		}

	case *ast.StarExpr:
		return "pointer dereference not eligible for GPU offload"

	case *ast.IndexExpr:
		if reject := a.checkExpr(x.X, depth); reject != "" {
			return reject
		}
		return a.checkExpr(x.Index, depth)

	case *ast.CallExpr:
		return a.checkCall(x, depth)

	case *ast.FuncLit:
		return "closure not eligible for GPU offload"

	case *ast.TypeAssertExpr:
		return "type assertion not eligible for GPU offload"

	case *ast.CompositeLit:
		return "composite literal not eligible for GPU offload"

	case *ast.SliceExpr:
		return "slice expression not eligible for GPU offload"

	case *ast.SelectorExpr:
		// A standalone selector outside a CallExpr.Fun position (field
		// access, package constant, etc.) is not something the transpiler
		// currently understands; checkCall handles the CallExpr.Fun case
		// separately before recursing into arguments.
		return fmt.Sprintf("unsupported selector expression not eligible for GPU offload: %s.%s", exprString(x.X), x.Sel.Name)

	default:
		return fmt.Sprintf("expression type %T not eligible for GPU offload", expr)
	}
}

// exprString renders expr for error messages without pulling in go/printer;
// good enough for the identifier/selector forms checkExpr's messages need.
func exprString(expr ast.Expr) string {
	if ident, ok := expr.(*ast.Ident); ok {
		return ident.Name
	}
	return "<expr>"
}

// checkCall allowlists call targets: predeclared type conversions
// (int(x), float32(x), ...), the "lanes.Varying[T](x)" conversion, a small
// set of approved lanes/math builtins, and same-package private SPMD
// functions (inlined one level deep). Everything else — external package
// calls, method calls, calls through a value — is rejected by default.
// Every call's arguments are additionally checked: a slice- or
// pointer-typed argument is always rejected, because §D2's read/write
// classification is derived from a syntactic scan of the loop body (and,
// for inlined calls, the callee body) and passing such a value into an
// unanalyzed call site would let the callee silently read/write it without
// that access being reflected in Free's classification.
func (a *gpuAnalyzer) checkCall(call *ast.CallExpr, depth int) string {
	approved := false

	switch fun := call.Fun.(type) {
	case *ast.Ident:
		if obj := a.info.Uses[fun]; obj != nil {
			if tn, ok := obj.(*types.TypeName); ok && tn.Parent() == types.Universe {
				// Predeclared type conversion: int(x), float32(x), etc.
				approved = true
				break
			}
		}
		if decl := a.lookupLocalFunc(fun.Name); decl != nil {
			if depth >= 1 {
				return "nested SPMD call depth"
			}
			a.inlined[call] = decl
			if reject := a.checkEligible(decl.Body, depth+1); reject != "" {
				return reject
			}
			approved = true
			break
		}
		return fmt.Sprintf("call to unrecognized function not eligible for GPU offload: %s", fun.Name)

	case *ast.SelectorExpr:
		reject, ok := a.checkPackageCall(fun)
		if !ok {
			return reject
		}
		approved = true

	case *ast.IndexExpr:
		// Generic instantiation used as a call target, e.g.
		// "lanes.Varying[int](x)".
		sel, ok := fun.X.(*ast.SelectorExpr)
		if !ok {
			return "unsupported generic call not eligible for GPU offload"
		}
		reject, ok := a.checkPackageCall(sel)
		if !ok {
			return reject
		}
		approved = true

	default:
		return fmt.Sprintf("unsupported call expression not eligible for GPU offload: %T", call.Fun)
	}

	if !approved {
		return "call not eligible for GPU offload"
	}

	for _, arg := range call.Args {
		if reject := a.checkExpr(arg, depth); reject != "" {
			return reject
		}
		if t := a.info.TypeOf(arg); t != nil {
			switch t.Underlying().(type) {
			case *types.Slice, *types.Pointer:
				return fmt.Sprintf("slice or pointer argument to function call not eligible for GPU offload: %s", exprString(arg))
			}
		}
	}
	return ""
}

// checkPackageCall resolves a "pkg.Name(...)" selector call target against
// the small allowlist of package functions the transpiler understands. It
// returns (rejectReason, false) on rejection, or ("", true) on approval.
func (a *gpuAnalyzer) checkPackageCall(sel *ast.SelectorExpr) (string, bool) {
	ident, ok := sel.X.(*ast.Ident)
	if !ok {
		return fmt.Sprintf("method call not eligible for GPU offload: .%s", sel.Sel.Name), false
	}
	obj := a.info.Uses[ident]
	pkgName, isPkg := obj.(*types.PkgName)
	if !isPkg {
		// The selector's receiver resolves to a value (variable, field,
		// etc.), not a package: this is a method call, not a package
		// function call.
		return fmt.Sprintf("method call not eligible for GPU offload: %s.%s", ident.Name, sel.Sel.Name), false
	}
	pkgPath := pkgName.Imported().Path()
	switch pkgPath {
	case "lanes":
		if gpuCrossLaneOps[sel.Sel.Name] {
			return fmt.Sprintf("cross-lane operation not eligible for GPU offload: lanes.%s", sel.Sel.Name), false
		}
		switch sel.Sel.Name {
		case "Count", "Index":
			// lanes.Count is the SIMD register width and lanes.Index is the
			// lane index *within* a vector (0..Count-1); the loop index is
			// already the GPU's gid.x. With one GPU invocation per lane
			// neither has a sound meaning, and per-lane use of them is
			// exactly the lane-count-dependent anti-pattern documented in
			// CLAUDE.md as not dual-mode safe -- offloading them would
			// silently change results depending on workgroup size.
			return fmt.Sprintf("lane-count-dependent operation not eligible for GPU offload: lanes.%s", sel.Sel.Name), false
		}
		if gpuApprovedLanesFuncs[sel.Sel.Name] {
			return "", true
		}
		return fmt.Sprintf("unsupported lanes function not eligible for GPU offload: lanes.%s", sel.Sel.Name), false
	case "reduce":
		return fmt.Sprintf("reduce operations not eligible for GPU offload: reduce.%s", sel.Sel.Name), false
	case "fmt":
		return fmt.Sprintf("fmt calls not eligible for GPU offload: fmt.%s", sel.Sel.Name), false
	case "math":
		if gpuApprovedMathFuncs[sel.Sel.Name] {
			return "", true
		}
		return fmt.Sprintf("unsupported math function not eligible for GPU offload: math.%s", sel.Sel.Name), false
	default:
		return fmt.Sprintf("call to unapproved package function not eligible for GPU offload: %s.%s", ident.Name, sel.Sel.Name), false
	}
}

// gpuApprovedLanesFuncs is the set of lanes.* functions known to be safe on
// a GPU (the Varying[T] conversion and uniform lane metadata), disjoint from
// gpuCrossLaneOps.
var gpuApprovedLanesFuncs = map[string]bool{
	"Varying": true,
	"FMA":     true,
}

// gpuApprovedMathFuncs is the set of math.* functions the §D3 cost model
// and (eventually) the WGSL transpiler know how to handle.
var gpuApprovedMathFuncs = map[string]bool{
	"Sqrt": true,
}

// lookupLocalFunc returns the package-level, unexported FuncDecl named name,
// or nil. Only such functions are candidates for one-level SPMD inlining.
func (a *gpuAnalyzer) lookupLocalFunc(name string) *ast.FuncDecl {
	if ast.IsExported(name) {
		return nil
	}
	for _, file := range a.files {
		for _, decl := range file.Decls {
			fn, ok := decl.(*ast.FuncDecl)
			if !ok || fn.Recv != nil || fn.Name.Name != name {
				continue
			}
			return fn
		}
	}
	return nil
}

// freeVars discovers the loop body's free variables (objects declared
// outside the RangeStmt's body and used inside it) via types.Info.Uses, and
// classifies each as a uniform scalar or a read/read-write slice per §D2.
func (a *gpuAnalyzer) freeVars(rangeStmt *ast.RangeStmt) ([]gpuFreeVar, string) {
	bodyStart := rangeStmt.Body.Lbrace
	bodyEnd := rangeStmt.Body.Rbrace

	writes := a.writtenSliceObjs(rangeStmt.Body)
	scalarWrites := a.writtenScalarObjs(rangeStmt.Body)

	// The loop's own iteration variable(s) (e.g. "idx" in
	// "go for idx := range n") are not free: exclude the objects they
	// define, keyed by identity via info.Defs.
	iterVars := map[types.Object]bool{}
	for _, e := range []ast.Expr{rangeStmt.Key, rangeStmt.Value} {
		if ident, ok := e.(*ast.Ident); ok {
			if obj := a.info.Defs[ident]; obj != nil {
				iterVars[obj] = true
			}
		}
	}

	seen := map[types.Object]bool{}
	var free []gpuFreeVar
	var reject string

	// Scan both the loop bound expression (rangeStmt.X, e.g. "width *
	// height") and the body: variables feeding the trip count must also be
	// passed to the kernel launch as scalars.
	visit := func(n ast.Node) {
		ast.Inspect(n, func(n ast.Node) bool {
			if reject != "" {
				return false
			}
			ident, ok := n.(*ast.Ident)
			if !ok {
				return true
			}
			obj := a.info.Uses[ident]
			v, ok := obj.(*types.Var)
			if !ok {
				return true
			}
			if iterVars[v] {
				return true
			}
			// Declared inside the loop body -> not free.
			if v.Pos() >= bodyStart && v.Pos() < bodyEnd {
				return true
			}
			if seen[v] {
				return true
			}
			seen[v] = true

			kind, wgslTy, err := a.classify(v, writes[v], scalarWrites[v])
			if err != "" {
				reject = err
				return false
			}
			free = append(free, gpuFreeVar{Obj: v, Kind: kind, WGSLTy: wgslTy})
			return true
		})
	}

	visit(rangeStmt.X)
	visit(rangeStmt.Body)

	if reject != "" {
		return nil, reject
	}
	return free, ""
}

// writtenSliceObjs returns the set of *types.Var free slice objects that
// appear as the target of an IndexExpr assignment ("slice[i] = ...") or an
// increment/decrement ("slice[i]++") inside body.
func (a *gpuAnalyzer) writtenSliceObjs(body *ast.BlockStmt) map[*types.Var]bool {
	writes := map[*types.Var]bool{}
	record := func(target ast.Expr) {
		idx, ok := target.(*ast.IndexExpr)
		if !ok {
			return
		}
		ident, ok := idx.X.(*ast.Ident)
		if !ok {
			return
		}
		if v, ok := a.info.Uses[ident].(*types.Var); ok {
			writes[v] = true
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range x.Lhs {
				record(lhs)
			}
		case *ast.IncDecStmt:
			record(x.X)
		}
		return true
	})
	return writes
}

// writtenScalarObjs returns the set of *types.Var objects assigned via a
// bare identifier (assignment, compound assignment, or increment/decrement)
// anywhere inside body. Local variables declared inside the body will
// spuriously appear here too; callers must combine this with a
// declared-outside-body check (as freeVars does) before treating it as a
// free-scalar write.
func (a *gpuAnalyzer) writtenScalarObjs(body *ast.BlockStmt) map[*types.Var]bool {
	writes := map[*types.Var]bool{}
	record := func(target ast.Expr) {
		ident, ok := target.(*ast.Ident)
		if !ok {
			return
		}
		if v, ok := a.info.Uses[ident].(*types.Var); ok {
			writes[v] = true
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			if x.Tok == token.DEFINE {
				return true
			}
			for _, lhs := range x.Lhs {
				record(lhs)
			}
		case *ast.IncDecStmt:
			record(x.X)
		}
		return true
	})
	return writes
}

// classify maps a free variable's Go type to its GPU kind and WGSL type
// name, rejecting types the transpiler cannot represent (§D2), and
// rejecting free scalars that the loop body writes: a uniform-buffer
// parameter is read-only from the GPU's perspective, so such a write would
// be silently dropped on the GPU path while the CPU path performs it.
func (a *gpuAnalyzer) classify(v *types.Var, sliceWritten, scalarWritten bool) (kind int, wgslTy string, reject string) {
	switch t := v.Type().Underlying().(type) {
	case *types.Basic:
		if scalarWritten {
			return 0, "", fmt.Sprintf("free scalar variable written inside GPU-offloaded loop body (uniform buffers are read-only): %s", v.Name())
		}
		wgslTy, reject = basicWGSLType(t)
		if reject != "" {
			return 0, "", reject
		}
		return gpuScalar, wgslTy, ""
	case *types.Slice:
		elemWgslTy, reject := basicWGSLType(asBasic(t.Elem()))
		if reject != "" {
			return 0, "", reject
		}
		arrTy := "array<" + elemWgslTy + ">"
		if sliceWritten {
			return gpuSliceRW, arrTy, ""
		}
		return gpuSliceRead, arrTy, ""
	default:
		return 0, "", fmt.Sprintf("unsupported free-variable type for GPU offload: %s", v.Type().String())
	}
}

// asBasic returns t as *types.Basic, or nil if t is not one (in which case
// basicWGSLType will report the rejection).
func asBasic(t types.Type) *types.Basic {
	b, _ := t.Underlying().(*types.Basic)
	return b
}

// basicWGSLType maps an int/int32/uint32/float32/bool basic type to its
// WGSL scalar name, rejecting unsupported basic kinds (notably float64,
// per §D2/§D4).
func basicWGSLType(t *types.Basic) (string, string) {
	if t == nil {
		return "", "unsupported non-basic element type for GPU offload"
	}
	switch t.Kind() {
	case types.Int, types.Int32:
		return "i32", ""
	case types.Uint, types.Uint32:
		return "u32", ""
	case types.Float32:
		return "f32", ""
	case types.Float64:
		return "", "float64 unsupported in WGSL"
	case types.Bool:
		// WGSL has a bool type but no dedicated buffer element mapping is
		// defined by the design doc; treat as i32-compatible (0/1) for now.
		return "i32", ""
	default:
		return "", fmt.Sprintf("unsupported basic type for GPU offload: %s", t.String())
	}
}

// bodyCost computes the §D3 weighted AST op count for a statement list:
// arithmetic ops (+,-,*,&,|,^,<<,>>,&&,||) weight 1; division/remainder
// (/, %) and sqrt calls weight 8; a nested uniform `for` loop multiplies
// its own body's cost by its constant bound if the loop condition is of
// the form `x < <int literal>`, or by 16 if the bound is not a compile-time
// constant.
func (a *gpuAnalyzer) bodyCost(stmts []ast.Stmt) uint64 {
	var total uint64
	for _, stmt := range stmts {
		total += a.stmtCost(stmt)
	}
	return total
}

func (a *gpuAnalyzer) stmtCost(stmt ast.Stmt) uint64 {
	if forStmt, ok := stmt.(*ast.ForStmt); ok {
		innerCost := a.bodyCost(forStmt.Body.List)
		mult := uint64(dynamicBoundMultiplier)
		if n, ok := constLoopBound(forStmt.Cond); ok {
			mult = n
		}
		return innerCost * mult
	}
	if rangeStmt, ok := stmt.(*ast.RangeStmt); ok {
		// "for i := range n" over an integer bound (allowlisted by
		// checkStmt above): no compile-time-constant-bound fast path
		// (unlike the classic for's literal-comparison form), so always
		// use the dynamic-bound multiplier.
		innerCost := a.bodyCost(rangeStmt.Body.List)
		return innerCost * dynamicBoundMultiplier
	}

	var total uint64
	ast.Inspect(stmt, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.ForStmt:
			// Nested for-loops are handled recursively by stmtCost (to get
			// the correct multiplier and avoid double counting); stop
			// descending into this subtree here.
			total += a.stmtCost(x)
			return false
		case *ast.RangeStmt:
			total += a.stmtCost(x)
			return false
		case *ast.BinaryExpr:
			switch x.Op {
			case token.ADD, token.SUB, token.MUL, token.AND, token.OR, token.XOR,
				token.SHL, token.SHR, token.LAND, token.LOR:
				total++
			case token.QUO, token.REM:
				total += divSqrtCost
			}
		case *ast.CallExpr:
			if sel, ok := x.Fun.(*ast.SelectorExpr); ok && sel.Sel.Name == "Sqrt" {
				total += divSqrtCost
			}
		}
		return true
	})
	return total
}

const (
	divSqrtCost            = 8
	dynamicBoundMultiplier = 16
)

// constLoopBound reports whether cond is of the form `x < N` (or `x <= N`)
// for an integer literal N, returning N (adjusted for <=) as the loop's
// compile-time-constant trip count.
func constLoopBound(cond ast.Expr) (uint64, bool) {
	bin, ok := cond.(*ast.BinaryExpr)
	if !ok {
		return 0, false
	}
	lit, ok := bin.Y.(*ast.BasicLit)
	if !ok || lit.Kind != token.INT {
		return 0, false
	}
	var n uint64
	if _, err := fmt.Sscanf(lit.Value, "%d", &n); err != nil {
		return 0, false
	}
	switch bin.Op {
	case token.LSS:
		return n, true
	case token.LEQ:
		return n + 1, true
	default:
		return 0, false
	}
}
