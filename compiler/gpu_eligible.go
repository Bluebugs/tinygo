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
	"go/constant"
	"go/token"
	"go/types"
)

// gpuFreeVar classifies a single free variable referenced by an
// offload-eligible loop body.
type gpuFreeVar struct {
	Obj    types.Object
	Kind   int    // gpuScalar | gpuSliceRead | gpuSliceRW
	WGSLTy string // "i32", "u32", "f32", "array<i32>", "array<f32>"

	// WriteOnly is only meaningful for gpuSliceRW.  It records that the
	// loop body (and every inlined callee) NEVER reads this slice and its
	// ONLY write is an unconditional `s[idx] = expr` at loop-body top
	// level, with idx the bare loop induction variable.  Under those
	// conditions every element in [0, n) is provably written by the
	// kernel, so if the slice's length is exactly n at launch time there
	// is no need to upload its prior contents to the device.
	//
	// This is a CANDIDATE flag, not a decision: `len(s) == n` cannot be
	// proven statically (the slice may be longer than the trip count, in
	// which case the tail elements are never written by the kernel and
	// their pre-launch contents must survive the round trip).  The
	// descriptor emitter in gpu_offload.go therefore turns this into a
	// RUNTIME `n == len(s)` check that selects buffer mode 2 (skip the
	// upload, still read back) or falls back to mode 1 (upload + read
	// back).  See gpuBuildBuffers.
	WriteOnly bool
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

	if reject := a.rangeValueReject(rangeStmt); reject != "" {
		plan.Reject = reject
		return plan
	}

	// §D4 eligibility walk (also inlines same-package SPMD calls one level
	// deep).
	if reject := a.checkEligible(rangeStmt.Body, 0); reject != "" {
		plan.Reject = reject
		return plan
	}

	// I1: body-local (and inlined-callee-local) declared types must also be
	// representable in WGSL.
	if reject := a.checkLocalTypes(rangeStmt.Body); reject != "" {
		plan.Reject = reject
		return plan
	}
	for _, decl := range plan.Inlined {
		if reject := a.checkLocalTypes(decl); reject != "" {
			plan.Reject = reject
			return plan
		}
	}

	// Concurrent GPU invocations must never write the same slice element.
	if reject := a.writeIndexReject(rangeStmt.Body, plan.IterIdent); reject != "" {
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

// rangeValueReject accepts the value variable of the `go for` itself
// (`go for i, v := range s`) only when the WGSL emitter can declare it as the
// element read `s[i]`: a named key and a free slice identifier as the range
// operand. The checkEligible walk covers the body only, so without this gate
// any range value reached the emitter undeclared.
func (a *gpuAnalyzer) rangeValueReject(rs *ast.RangeStmt) string {
	v, ok := rs.Value.(*ast.Ident)
	if rs.Value == nil || (ok && v.Name == "_") {
		return ""
	}
	if !ok || rs.Tok != token.DEFINE {
		return "non-identifier or non-defining range value not eligible for GPU offload"
	}
	if key, ok := rs.Key.(*ast.Ident); !ok || key.Name == "_" {
		return "range value without a named range key not eligible for GPU offload"
	}
	x, ok := rs.X.(*ast.Ident)
	if !ok {
		return "range value over a non-identifier operand not eligible for GPU offload"
	}
	obj, ok := a.info.Uses[x].(*types.Var)
	if !ok {
		return "range value over a non-variable operand not eligible for GPU offload"
	}
	if _, ok := gpuUnwrapSPMD(obj.Type()).Underlying().(*types.Slice); !ok {
		return "range value over a non-slice operand not eligible for GPU offload"
	}
	return ""
}

// gpuAnalyzer carries the shared state for the eligibility walk, free-var
// discovery, and cost estimate.
type gpuAnalyzer struct {
	info      *types.Info
	files     []*ast.File
	rangeStmt *ast.RangeStmt
	inlined   map[*ast.CallExpr]*ast.FuncDecl

	// tailReturn is the single *ast.ReturnStmt that terminates the callee
	// body currently being walked (nil outside an inlined callee). Any other
	// return statement is rejected -- see checkStmt's *ast.ReturnStmt case.
	tailReturn *ast.ReturnStmt
}

// gpuEligibleAssignToks is the set of assignment operators the WGSL emitter
// explicitly handles (see compoundAssignOp in gpu_wgsl.go). It is
// deliberately a closed allowlist: an operator present here but missing from
// compoundAssignOp is an internal error, and an operator missing from here
// simply falls back to the CPU path.
var gpuEligibleAssignToks = map[token.Token]bool{
	token.DEFINE:     true,
	token.ASSIGN:     true,
	token.ADD_ASSIGN: true,
	token.SUB_ASSIGN: true,
	token.MUL_ASSIGN: true,
	token.QUO_ASSIGN: true,
	token.REM_ASSIGN: true,
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
		// C1 (fail-closed): only the assignment operators the WGSL emitter
		// explicitly maps are eligible. Anything else -- notably the
		// bitwise/shift compound forms |=, &=, ^=, <<=, >>=, &^= -- must be
		// rejected here rather than silently reaching emitAssign, which
		// would otherwise emit a plain `lhs = rhs` and produce a WRONG
		// answer with no diagnostic. Adding an operator to
		// gpuEligibleAssignToks REQUIRES adding it to compoundAssignOp too.
		if !gpuEligibleAssignToks[x.Tok] {
			return fmt.Sprintf("assignment operator %s not eligible for GPU offload", x.Tok)
		}
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
				// The emitter asserts *ast.ValueSpec unconditionally;
				// tolerating anything else here would turn into a compiler
				// panic later. Reject cleanly instead.
				return fmt.Sprintf("unsupported declaration specifier not eligible for GPU offload: %T", spec)
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
		// varying condition, which §D4 forbids.
		if depth == 0 {
			return "return not eligible for GPU offload"
		}
		// C3 (fail-closed): inside an inlined callee, the emitter lowers
		// `return v` to `retTarget = v;` and then FALLS THROUGH -- there is
		// no early exit. That is only correct for a return which is the
		// final statement of the callee body. A non-tail return (e.g.
		// `if cond { return a }; return b`) would emit both assignments
		// unconditionally and let the last one win, silently producing a
		// wrong answer. Reject anything but the tail return.
		if x != a.tailReturn {
			return "non-tail return in inlined function not eligible for GPU offload"
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
		if x.Op == token.SHL || x.Op == token.SHR {
			if reject := a.checkShift(x); reject != "" {
				return reject
			}
		}
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
		if s, _, ok := constStringOf(a.info, x.X); ok {
			if len(s) == 0 || len(s) > 256 {
				return fmt.Sprintf("constant table index: table length %d not eligible for GPU offload (1..256)", len(s))
			}
			if reject := a.checkExpr(x.Index, depth); reject != "" {
				return reject
			}
			m, proven := a.gpuIndexMax(x.Index)
			if !proven || m >= uint64(len(s)) {
				return fmt.Sprintf("constant table index into %s cannot be proven < %d (GPU clamps where Go panics)", exprString(x.X), len(s))
			}
			return ""
		}
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

// checkShift fails closed on shift counts Go and WGSL disagree on: Go
// defines a count >= the operand width (0 or sign fill), WGSL does not.
// Only constant counts below the width are eligible. int/uint/uintptr are
// held to 32 bits because they lower to 32-bit WGSL types.
func (a *gpuAnalyzer) checkShift(x *ast.BinaryExpr) string {
	if a.info.Types[x].Value != nil {
		return "" // folded by the type checker
	}
	count, ok := a.constInt(x.Y)
	if !ok {
		return fmt.Sprintf("shift count %s is not a compile-time constant (not eligible for GPU offload)", exprString(x.Y))
	}
	width := int64(32)
	if b, ok := gpuUnwrapSPMD(a.info.TypeOf(x.X)).Underlying().(*types.Basic); ok {
		switch b.Kind() {
		case types.Int8, types.Uint8:
			width = 8
		case types.Int16, types.Uint16:
			width = 16
		}
	}
	if count < 0 || count >= width {
		return fmt.Sprintf("shift count %d >= operand width %d not eligible for GPU offload", count, width)
	}
	return ""
}

// constStringOf reports whether e names a string constant, returning its
// value and object.
func constStringOf(info *types.Info, e ast.Expr) (string, *types.Const, bool) {
	ident, ok := e.(*ast.Ident)
	if !ok {
		return "", nil, false
	}
	c, ok := info.Uses[ident].(*types.Const)
	if !ok || c.Val().Kind() != constant.String {
		return "", nil, false
	}
	return constant.StringVal(c.Val()), c, true
}

// gpuIndexMax proves an upper bound on an index expression. WGSL clamps an
// out-of-range array index where Go panics, so a table lookup is only
// offloaded when its index provably stays in range.
func (a *gpuAnalyzer) gpuIndexMax(e ast.Expr) (uint64, bool) {
	if tv, ok := a.info.Types[e]; ok && tv.Value != nil {
		v := constant.ToInt(tv.Value)
		if v.Kind() == constant.Int && constant.Sign(v) >= 0 {
			if u, exact := constant.Uint64Val(v); exact {
				return u, true
			}
		}
		return 0, false
	}
	switch x := e.(type) {
	case *ast.ParenExpr:
		return a.gpuIndexMax(x.X)
	case *ast.BinaryExpr:
		switch x.Op {
		case token.AND:
			lm, lok := a.gpuIndexMax(x.X)
			rm, rok := a.gpuIndexMax(x.Y)
			switch {
			case lok && rok:
				return min(lm, rm), true
			case lok:
				if _, isConst := a.info.Types[x.X]; isConst && a.info.Types[x.X].Value != nil {
					return lm, true
				}
			case rok:
				if a.info.Types[x.Y].Value != nil {
					return rm, true
				}
			}
		case token.SHR:
			if a.info.Types[x.Y].Value == nil {
				return 0, false
			}
			k, kok := a.gpuIndexMax(x.Y)
			m, mok := a.gpuIndexMax(x.X)
			if kok && mok && k < 64 {
				return m >> k, true
			}
		case token.REM:
			if a.info.Types[x.Y].Value != nil && gpuIsUnsigned(a.info.TypeOf(x.X)) {
				if m, ok := a.gpuIndexMax(x.Y); ok && m > 0 {
					return m - 1, true
				}
			}
		}
	}
	if gpuIsUint8(a.info.TypeOf(e)) {
		return 255, true
	}
	return 0, false
}

func gpuUnwrapSPMD(t types.Type) types.Type {
	if s, ok := t.(*types.SPMDType); ok {
		return s.Elem()
	}
	return t
}

func gpuIsUint8(t types.Type) bool {
	t = gpuUnwrapSPMD(t)
	if t == nil {
		return false
	}
	b, ok := t.Underlying().(*types.Basic)
	return ok && b.Kind() == types.Uint8
}

func gpuIsUnsigned(t types.Type) bool {
	t = gpuUnwrapSPMD(t)
	if t == nil {
		return false
	}
	b, ok := t.Underlying().(*types.Basic)
	return ok && b.Info()&types.IsUnsigned != 0
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
				if tn.Name() == "byte" || tn.Name() == "uint8" {
					if len(call.Args) == 1 {
						if b := asByteConversionSourceBasic(a.info, call.Args[0]); b != nil && b.Info()&types.IsFloat != 0 {
							return "conversion from float to byte not eligible for GPU offload (out-of-range semantics differ)"
						}
					}
				}
				approved = true
				break
			}
		}
		if decl := a.lookupLocalFunc(fun); decl != nil {
			if depth >= 1 {
				return "nested SPMD call depth"
			}
			a.inlined[call] = decl
			savedTail := a.tailReturn
			a.tailReturn = tailReturnOf(decl.Body)
			reject := a.checkEligible(decl.Body, depth+1)
			a.tailReturn = savedTail
			if reject != "" {
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

// lookupLocalFunc returns the package-level, unexported FuncDecl that ident
// RESOLVES TO, or nil. Only such functions are candidates for one-level SPMD
// inlining.
//
// I4: resolution goes through the type checker (info.Uses -> *types.Func ->
// that object's own declaring *ast.Ident), never by name. A local variable
// (or a local func literal bound to a name) that shadows a package-level
// function must NOT cause that package function's body to be inlined -- a
// name-only match silently transpiles the wrong code.
func (a *gpuAnalyzer) lookupLocalFunc(ident *ast.Ident) *ast.FuncDecl {
	fn, ok := a.info.Uses[ident].(*types.Func)
	if !ok {
		return nil // not a function at all (e.g. a shadowing local variable)
	}
	if fn.Pkg() == nil || fn.Parent() != fn.Pkg().Scope() {
		return nil // not declared at package level
	}
	if sig, ok := fn.Type().(*types.Signature); !ok || sig.Recv() != nil {
		return nil // methods are not inlined
	}
	if ast.IsExported(fn.Name()) {
		return nil
	}
	for _, file := range a.files {
		for _, decl := range file.Decls {
			d, ok := decl.(*ast.FuncDecl)
			if !ok || d.Recv != nil {
				continue
			}
			// Identity match on the declaring object, not on the name.
			if a.info.Defs[d.Name] == fn {
				return d
			}
		}
	}
	return nil
}

// tailReturnOf returns body's final statement when it is a *ast.ReturnStmt,
// else nil. See checkStmt's *ast.ReturnStmt case (C3).
func tailReturnOf(body *ast.BlockStmt) *ast.ReturnStmt {
	if body == nil || len(body.List) == 0 {
		return nil
	}
	ret, _ := body.List[len(body.List)-1].(*ast.ReturnStmt)
	return ret
}

// checkLocalTypes rejects the loop (I1) when any variable DECLARED inside
// node has a type the WGSL emitter cannot spell. classify/basicWGSLType only
// ever see FREE variables, so without this a body-local `var c int64` or
// `float64` would reach wgslTypeOfIdent, whose error used to be swallowed
// into a hardcoded "i32" -- a silent truncation on the GPU path only.
func (a *gpuAnalyzer) checkLocalTypes(node ast.Node) string {
	reject := ""
	ast.Inspect(node, func(n ast.Node) bool {
		if reject != "" {
			return false
		}
		ident, ok := n.(*ast.Ident)
		if !ok {
			return true
		}
		obj, ok := a.info.Defs[ident].(*types.Var)
		if !ok || obj == nil {
			return true
		}
		if _, err := wgslTypeOf(obj.Type()); err != nil {
			reject = fmt.Sprintf("local variable %s has a type not representable in WGSL: %s", ident.Name, obj.Type().String())
			return false
		}
		return true
	})
	return reject
}

// freeVars discovers the loop body's free variables (objects declared
// outside the RangeStmt's body and used inside it) via types.Info.Uses, and
// classifies each as a uniform scalar or a read/read-write slice per §D2.
func (a *gpuAnalyzer) freeVars(rangeStmt *ast.RangeStmt) ([]gpuFreeVar, string) {
	bodyStart := rangeStmt.Body.Lbrace
	bodyEnd := rangeStmt.Body.Rbrace

	writes := a.writtenSliceObjs(rangeStmt.Body)
	scalarWrites := a.writtenScalarObjs(rangeStmt.Body)
	// Write-only candidates for the upload-skipping optimization (see
	// gpuFreeVar.WriteOnly).  iterIdent is rangeStmt.Key when it is a
	// plain identifier; anything else yields no candidates at all.
	iterIdent, _ := rangeStmt.Key.(*ast.Ident)
	writeOnly := a.writeOnlySliceObjs(rangeStmt.Body, iterIdent)

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
			free = append(free, gpuFreeVar{
				Obj:    v,
				Kind:   kind,
				WGSLTy: wgslTy,
				// Only a read-write slice can be write-only: a
				// gpuSliceRead buffer is never written at all.
				WriteOnly: kind == gpuSliceRW && writeOnly[v],
			})
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

// readSliceObjs returns the set of *types.Var slice objects that appear as
// an *rvalue* index expression (`... = s[i]`, `f(s[i])`, `s[i] + 1`, ...)
// anywhere in n.  It deliberately walks EVERY IndexExpr and then subtracts
// the ones that are pure assignment targets, so any use it does not
// recognise counts as a read -- the fail-closed direction for the
// write-only-buffer optimization, whose soundness depends on the kernel
// never observing the buffer's prior contents.
//
// Note `s[i] += x` and `s[i]++` ARE reads (the old value is consumed), and
// this function reports them as such because record() below only removes
// plain `=` assignment targets from the set.
func (a *gpuAnalyzer) readSliceObjs(n ast.Node) map[*types.Var]bool {
	reads := map[*types.Var]bool{}
	// Assignment targets of a plain `=` are writes, not reads.  Collect
	// their IndexExpr nodes by identity so the general walk below can skip
	// exactly those nodes and nothing else.
	pureWriteTargets := map[ast.Expr]bool{}
	ast.Inspect(n, func(nd ast.Node) bool {
		if as, ok := nd.(*ast.AssignStmt); ok && as.Tok == token.ASSIGN {
			for _, lhs := range as.Lhs {
				if _, ok := lhs.(*ast.IndexExpr); ok {
					pureWriteTargets[lhs] = true
				}
			}
		}
		return true
	})
	ast.Inspect(n, func(nd ast.Node) bool {
		idx, ok := nd.(*ast.IndexExpr)
		if !ok {
			return true
		}
		if pureWriteTargets[idx] {
			// Still descend into the index expression itself: `a[b[i]] = x`
			// reads b.
			ast.Inspect(idx.Index, func(inner ast.Node) bool {
				if ie, ok := inner.(*ast.IndexExpr); ok {
					if ident, ok := ie.X.(*ast.Ident); ok {
						if v, ok := a.info.Uses[ident].(*types.Var); ok {
							reads[v] = true
						}
					}
				}
				return true
			})
			return true
		}
		if ident, ok := idx.X.(*ast.Ident); ok {
			if v, ok := a.info.Uses[ident].(*types.Var); ok {
				reads[v] = true
			}
		}
		return true
	})
	return reads
}

// bodyHasTopLevelBranch reports whether body contains a `continue` or
// `break` (labeled or not) that is NOT lexically enclosed by a nested
// `for`/`range` statement inside body -- i.e. a branch that acts on the
// `go for` loop itself.
//
// This exists purely for the write-only-buffer optimization. The
// eligibility walk deliberately ALLOWS `continue` at loop-body depth
// (checkStmt, *ast.BranchStmt), and a `continue` skips every statement
// after it:
//
//	go for idx := range n {
//	    if idx > 3 { continue }
//	    output[idx] = idx      // a direct child of body.List, yet only
//	}                          // executed for idx <= 3
//
// The store above is syntactically top-level, bare-indexed, unconditional
// and unread, so every other condition in writeOnlySliceObjs holds -- but
// [4, n) is never written, which is exactly what mode 2 must never assume.
// Rather than model reachability, this fails closed: any body-level branch
// disqualifies every write-only candidate in that body.
//
// (`break` at body level is already rejected by checkStmt, and a labeled
// branch is rejected with labeled statements; both are matched here anyway
// so this predicate does not silently depend on those rules holding.)
func bodyHasTopLevelBranch(body *ast.BlockStmt) bool {
	found := false
	ast.Inspect(body, func(n ast.Node) bool {
		if found || n == nil {
			return false
		}
		switch x := n.(type) {
		case *ast.ForStmt:
			// A branch inside a nested loop binds to THAT loop, not to the
			// `go for` body, so it cannot skip the body-level store.
			return false
		case *ast.RangeStmt:
			return false
		case *ast.BranchStmt:
			if x.Tok == token.CONTINUE || x.Tok == token.BREAK {
				found = true
			}
			return false
		}
		return true
	})
	return found
}

// writeOnlySliceObjs returns the set of *types.Var slice objects that are
// WRITE-ONLY CANDIDATES for the upload-skipping optimization: within body
// (and every inlined callee), the object is never read, and its only write
// is a single unconditional `s[iterIdent] = expr` statement at body TOP
// LEVEL -- not nested inside an `if`, a `for`, or an inlined callee, and
// indexed by the bare loop induction variable rather than any derived
// expression.
//
// Every one of those conditions is load-bearing for soundness:
//   - never read: the kernel must not observe the buffer's prior contents,
//     since the optimization leaves them undefined on the device.
//   - top level / unconditional: a store under an `if` writes only some
//     elements; the rest would come back as device garbage.
//   - bare loop index: `s[f(idx)]` need not be a bijection onto [0, n).
//   - exactly one write: a second write elsewhere may be conditional or
//     differently-indexed, so anything beyond the single recognised store
//     disqualifies the object.
//   - no body-level `continue`/`break`: those make a syntactically
//     top-level store conditional on control reaching it. See
//     bodyHasTopLevelBranch.
//
// Note this says nothing about len(s) vs. n -- that is checked at runtime
// by the descriptor emitter, see gpuFreeVar.WriteOnly.
func (a *gpuAnalyzer) writeOnlySliceObjs(body *ast.BlockStmt, iterIdent *ast.Ident) map[*types.Var]bool {
	if iterIdent == nil {
		return nil
	}
	iterObj := a.info.Defs[iterIdent]
	if iterObj == nil {
		return nil
	}
	// A body-level `continue` (which the eligibility walk allows) makes
	// every statement after it conditional, including a store that is
	// syntactically a direct child of body.List. Fail closed.
	if bodyHasTopLevelBranch(body) {
		return nil
	}

	// Candidates: objects with a top-level `s[iter] = expr` statement.
	cand := map[*types.Var]bool{}
	for _, stmt := range body.List {
		as, ok := stmt.(*ast.AssignStmt)
		if !ok || as.Tok != token.ASSIGN {
			continue
		}
		for _, lhs := range as.Lhs {
			idx, ok := lhs.(*ast.IndexExpr)
			if !ok {
				continue
			}
			xIdent, ok := idx.X.(*ast.Ident)
			if !ok {
				continue
			}
			iIdent, ok := idx.Index.(*ast.Ident)
			if !ok || a.info.Uses[iIdent] != iterObj {
				continue
			}
			if v, ok := a.info.Uses[xIdent].(*types.Var); ok {
				cand[v] = true
			}
		}
	}
	if len(cand) == 0 {
		return nil
	}

	// Count every write to each candidate anywhere in the body: more than
	// the single top-level store recognised above disqualifies it.
	writeCount := map[*types.Var]int{}
	countWrite := func(target ast.Expr) {
		idx, ok := target.(*ast.IndexExpr)
		if !ok {
			return
		}
		ident, ok := idx.X.(*ast.Ident)
		if !ok {
			return
		}
		if v, ok := a.info.Uses[ident].(*types.Var); ok {
			writeCount[v]++
		}
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch x := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range x.Lhs {
				countWrite(lhs)
			}
		case *ast.IncDecStmt:
			countWrite(x.X)
		}
		return true
	})

	// Reads anywhere in the body OR in any inlined callee disqualify.
	reads := a.readSliceObjs(body)
	for _, decl := range a.inlined {
		for v := range a.readSliceObjs(decl) {
			reads[v] = true
		}
		// A write inside a callee is by definition not the single
		// top-level store, so count it too.
		ast.Inspect(decl, func(n ast.Node) bool {
			switch x := n.(type) {
			case *ast.AssignStmt:
				for _, lhs := range x.Lhs {
					countWrite(lhs)
				}
			case *ast.IncDecStmt:
				countWrite(x.X)
			}
			return true
		})
	}

	out := map[*types.Var]bool{}
	for v := range cand {
		if reads[v] || writeCount[v] != 1 {
			continue
		}
		out[v] = true
	}
	if len(out) == 0 {
		return nil
	}
	return out
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

// asByteConversionSourceBasic resolves arg's Go type to its underlying
// *types.Basic, unwrapping a *types.SPMDType (lanes.Varying[T]) first, for
// the sole purpose of checking whether a byte(x)/uint8(x) conversion's
// argument is a float. Returns nil if arg's type is not a basic type (or is
// unresolvable), which the caller treats as "not a float" -- the reverse
// mistake (accepting a float conversion) is the one that produces silently
// wrong results, not the other way around.
func asByteConversionSourceBasic(info *types.Info, arg ast.Expr) *types.Basic {
	t := info.TypeOf(arg)
	if t == nil {
		return nil
	}
	if s, ok := t.(*types.SPMDType); ok {
		t = s.Elem()
	}
	if t == nil {
		return nil
	}
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
	case types.Uint8:
		// A Go byte is carried as a WGSL u32 holding a value in [0, 255];
		// the WGSL emitter re-establishes that range after every
		// overflowing operation (see wgslEmitter.isByteExpr).
		return "u32", ""
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
