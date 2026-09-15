// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

// This file transpiles an offload-eligible `go for` loop body (as analyzed
// by gpu_eligible.go's analyzeGPULoop into a *gpuLoopPlan) into a WGSL
// compute-shader module. It operates purely on the pre-predication go/ast +
// go/types data the plan carries: no mask machinery, because on the GPU one
// compute invocation is one SPMD lane, so a varying `break` inside the loop
// body (or an inlined callee) is emitted as an ordinary WGSL `break`.
//
// The emitter here is hand-written for the restricted AST subset
// gpu_eligible.go's checkStmt/checkExpr allowlist admits (see RULING A in
// the task brief: gosl's full go/printer fork is not vendored — its bulk is
// generic Go-source formatting irrelevant to this narrow subset). Two small
// pieces adapted from cogentcore.org/lab/gosl live in
// compiler/internal/wgslprint (BSD-3-Clause, see that file's header).

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"sort"
	"strconv"
	"strings"

	"github.com/tinygo-org/tinygo/compiler/internal/wgslprint"
)

// gpuKernel is the transpiled WGSL module for one offload-eligible `go for`
// loop.
type gpuKernel struct {
	ID         int32
	Entry      string       // "spmd_kernel_<n>"
	WGSL       string       // full module source
	Params     []gpuFreeVar // uniform scalars, in Params struct field order (excluding trailing alignment padding); Task 6 iterates this to fill the uniform buffer
	ParamsSize uint32       // total byte size of the Params struct incl. padding (wgslprint.PadTo16-computed); Task 6 asserts LLVM struct size equals this
	Buffers    []gpuFreeVar // slices, in @binding order starting at 1
	SPIRV      []byte       // naga-compiled WGSL; set only for -gpu-host=vulkan

	// LanesPerInvocation is how many loop iterations one compute invocation
	// runs: 4 when any buffer is a byte slice (bytes are packed four per
	// u32 word, and an invocation owns whole words), else 1. The launch n
	// argument is ceil(trip count / LanesPerInvocation); Params.n stays the
	// trip count.
	LanesPerInvocation int
}

// wgslWorkgroupSize is the fixed 1-D workgroup size used for every kernel.
const wgslWorkgroupSize = 64

// transpileWGSL turns an eligible plan into a WGSL compute-shader module.
// plan.Reject must be "" (analyzeGPULoop's eligibility gate); id numbers the
// kernel within the compilation unit and names its entry point.
func transpileWGSL(plan *gpuLoopPlan, id int32) (*gpuKernel, error) {
	if plan == nil {
		return nil, fmt.Errorf("transpileWGSL: nil plan")
	}
	if plan.Reject != "" {
		return nil, fmt.Errorf("transpileWGSL: plan is not GPU-offload eligible: %s", plan.Reject)
	}
	if plan.IterIdent == nil {
		return nil, fmt.Errorf("transpileWGSL: plan has no loop iteration variable")
	}

	e := &wgslEmitter{
		info:      plan.Loop.TypesInfo,
		iterIdent: plan.IterIdent.Name,
		iterObj:   plan.Loop.TypesInfo.Defs[plan.IterIdent],
		freeSubst: map[types.Object]string{},
		locals:    map[types.Object]string{},
		inlined:   plan.Inlined,
		byteBufs:  map[types.Object]bool{},
		usedBufs:  map[types.Object]bool{},
		tables:    map[*types.Const]string{},
	}

	// Params: RULING B - "n" (the launch trip count) is always the
	// mandatory first uniform field; the bound guard always compares
	// against params.n, never arrayLength (which is only correct when the
	// written buffer's length happens to equal the trip count).
	nObj := types.NewVar(token.NoPos, nil, "n", types.Typ[types.Int32])
	params := []gpuFreeVar{{Obj: nObj, Kind: gpuScalar, WGSLTy: "i32"}}

	var scalars []gpuFreeVar
	var buffers []gpuFreeVar
	for _, fv := range plan.Free {
		switch fv.Kind {
		case gpuScalar:
			scalars = append(scalars, fv)
		case gpuSliceRead, gpuSliceRW:
			buffers = append(buffers, fv)
		default:
			return nil, fmt.Errorf("transpileWGSL: free variable %s has unknown kind %d", fv.Obj.Name(), fv.Kind)
		}
	}
	// Uniform struct fields (after the mandatory "n") are ordered
	// alphabetically, per the brief's golden shape.
	sort.Slice(scalars, func(i, j int) bool { return scalars[i].Obj.Name() < scalars[j].Obj.Name() })
	params = append(params, scalars...)

	// I5: buffer identifiers are emitted verbatim into the WGSL module, so
	// a Go slice named `params`, `select`, `loop`, `fn`, `let`, ... would
	// either collide with the mandatory uniform binding or hit a WGSL
	// reserved word. Either one is a shader COMPILE failure, which (before
	// I2) surfaced as a silent wrong answer. Sanitize and uniquify here the
	// same way the Params fields are handled below.
	bufNames := make(map[types.Object]string, len(buffers))
	usedBufNames := map[string]bool{"params": true}
	for i, b := range buffers {
		name := wgslSafeIdent(b.Obj.Name())
		for j := 1; usedBufNames[name]; j++ {
			name = fmt.Sprintf("%s_%d", wgslSafeIdent(b.Obj.Name()), j)
		}
		usedBufNames[name] = true
		bufNames[b.Obj] = name
		e.freeSubst[b.Obj] = name
		buffers[i].Obj = b.Obj // no-op, keeps order explicit
		if gpuIsByteSlice(b.Obj.Type()) {
			e.byteBufs[b.Obj] = true
		}
	}
	lanes := 1
	if len(e.byteBufs) > 0 {
		lanes = 4
	}

	// The packed byte store declares WGSL lets; pick names that cannot
	// shadow any Go identifier or buffer the store's operands refer to.
	taken := map[string]bool{"params": true}
	for _, name := range bufNames {
		taken[name] = true
	}
	collect := func(n ast.Node) {
		ast.Inspect(n, func(n ast.Node) bool {
			if id, ok := n.(*ast.Ident); ok {
				taken[id.Name] = true
			}
			return true
		})
	}
	collect(plan.Body)
	for _, decl := range plan.Inlined {
		collect(decl)
	}
	fresh := func(base string) string {
		name := base
		for i := 1; taken[name] || wgslReservedWords[name]; i++ {
			name = fmt.Sprintf("%s_%d", base, i)
		}
		taken[name] = true
		return name
	}
	e.tmpIdx, e.tmpShift, e.tmpRet = fresh("j"), fresh("sh"), fresh("spmd_bret")
	e.freshName = fresh
	e.lanes = lanes

	padFields := wgslprint.PadTo16(len(params))
	paramsSize := uint32((len(params) + padFields) * 4)

	entry := fmt.Sprintf("spmd_kernel_%d", id)

	// A free variable can legitimately be named "n" in Go source (e.g. a
	// parameter literally called n); disambiguate the WGSL struct field
	// name in that case so it does not collide with the mandatory
	// trip-count field "n" (RULING B). The reserved Params field order
	// (and therefore Task 6's positional iteration) is unaffected --
	// only the WGSL spelling of the field and its params.<name> references
	// changes.
	fieldName := make(map[types.Object]string, len(params))
	used := map[string]bool{}
	for _, p := range params {
		name := wgslSafeIdent(p.Obj.Name())
		for i := 1; used[name]; i++ {
			name = fmt.Sprintf("%s_%d", wgslSafeIdent(p.Obj.Name()), i)
		}
		used[name] = true
		fieldName[p.Obj] = name
		e.freeSubst[p.Obj] = "params." + name
		taken[name] = true
	}

	var sb strings.Builder
	sb.WriteString("struct Params {\n")
	for _, p := range params {
		fmt.Fprintf(&sb, "  %s: %s,\n", fieldName[p.Obj], p.WGSLTy)
	}
	for i := 0; i < padFields; i++ {
		fmt.Fprintf(&sb, "  _pad%d: i32,\n", i)
	}
	sb.WriteString("}\n")
	sb.WriteString("@group(0) @binding(0) var<uniform> params: Params;\n")
	var fn strings.Builder
	fmt.Fprintf(&fn, "@compute @workgroup_size(%d)\n", wgslWorkgroupSize)
	fmt.Fprintf(&fn, "fn %s(@builtin(global_invocation_id) gid: vec3<u32>) {\n", entry)
	e.sb = &fn
	if lanes == 1 {
		fmt.Fprintf(&fn, "  let %s: i32 = i32(gid.x);\n", e.iterIdent)
		fmt.Fprintf(&fn, "  if (%s >= params.n) { return; }\n", e.iterIdent)
		e.indent = 1
	} else {
		// A body-level Go `continue` lowers to WGSL `continue;`, which
		// now targets this lane loop: exactly "next iteration". Body-level
		// `break` is ineligible (checkStmt), so no Go break can end it.
		w := fresh("w")
		lane := fresh("lane")
		fmt.Fprintf(&fn, "  let %s: u32 = gid.x;\n", w)
		fmt.Fprintf(&fn, "  for (var %s: u32 = 0u; %s < %du; %s++) {\n", lane, lane, lanes, lane)
		fmt.Fprintf(&fn, "    let %s: i32 = i32(%s * %du + %s);\n", e.iterIdent, w, lanes, lane)
		fmt.Fprintf(&fn, "    if (%s >= params.n) { break; }\n", e.iterIdent)
		e.indent = 2
	}
	if err := e.emitRangeValue(plan.Loop.RangeStmt); err != nil {
		return nil, err
	}
	if err := e.emitStmts(plan.Body.List, ""); err != nil {
		return nil, err
	}
	if lanes != 1 {
		fn.WriteString("  }\n")
	}
	fn.WriteString("}\n")

	// Declare only the buffers the shader references. A slice used solely as
	// the range bound (its length is the SSA trip count) is never read, and a
	// `layout: 'auto'` pipeline drops an unreferenced binding from its layout
	// while both hosts bind every kernel buffer, so declaring it made wgpu
	// reject the bind group. Filtering Buffers keeps compiler, native and JS
	// hosts consistent with no host change.
	kept := buffers[:0]
	for _, b := range buffers {
		if e.usedBufs[b.Obj] {
			kept = append(kept, b)
		}
	}
	buffers = kept
	for i, b := range buffers {
		access := "read"
		if b.Kind == gpuSliceRW {
			access = "read_write"
		}
		fmt.Fprintf(&sb, "@group(0) @binding(%d) var<storage, %s> %s: %s;\n", i+1, access, bufNames[b.Obj], b.WGSLTy)
	}

	// Constant lookup tables are emitted at module scope, before the entry
	// point, but table names and references are only known after the body
	// has been transpiled (a table's first use is discovered while walking
	// the body). tableOrder records first-use order for determinism.
	for _, c := range e.tableOrder {
		s := constant.StringVal(c.Val())
		elems := make([]string, len(s))
		for i := 0; i < len(s); i++ {
			elems[i] = strconv.Itoa(int(s[i])) + "u"
		}
		fmt.Fprintf(&sb, "var<private> %s: array<u32, %d> = array<u32, %d>(%s);\n",
			e.tables[c], len(s), len(s), strings.Join(elems, ", "))
	}

	return &gpuKernel{
		ID:         id,
		Entry:      entry,
		WGSL:       sb.String() + fn.String(),
		Params:     params,
		ParamsSize: paramsSize,
		Buffers:    buffers,

		LanesPerInvocation: lanes,
	}, nil
}

// emitRangeValue declares the value variable of `go for i, v := range s` as
// the element read s[i], lowered exactly like the explicit IndexExpr (packed
// for byte slices). rangeValueReject guarantees the shape.
func (e *wgslEmitter) emitRangeValue(rs *ast.RangeStmt) error {
	if rs == nil {
		return nil
	}
	v, ok := rs.Value.(*ast.Ident)
	if !ok || v.Name == "_" {
		return nil
	}
	ty, err := e.wgslTypeOfIdent(v)
	if err != nil {
		return err
	}
	read, err := e.emitExpr(&ast.IndexExpr{X: rs.X, Index: rs.Key})
	if err != nil {
		return err
	}
	name := e.declareLocal(v)
	e.writeIndent()
	fmt.Fprintf(e.sb, "var %s: %s = %s;\n", name, ty, read)
	return nil
}

// wgslEmitter carries the state needed to lower the plan's body (and, while
// inlining, an SPMD callee's body) statement-by-statement into WGSL text.
type wgslEmitter struct {
	info      *types.Info
	iterIdent string       // the loop's own iteration variable; passed through unchanged
	iterObj   types.Object // iterIdent's defining object
	freeSubst map[types.Object]string
	locals    map[types.Object]string // ephemeral: active only while inlining a callee (param args + renamed callee locals)
	inlined   map[*ast.CallExpr]*ast.FuncDecl

	sb            *strings.Builder
	indent        int
	inlineSuffix  string // "" when not inlining; else "_iN" for the Nth inline expansion (monotonic per call site, not per depth)
	inlineCounter int    // monotonically increasing across the whole transpile; never reset on inline-exit

	byteBufs                 map[types.Object]bool // free byte-slice buffers, packed 4 per u32 word
	usedBufs                 map[types.Object]bool // free slice buffers referenced by the emitted shader
	tmpIdx, tmpShift, tmpRet string                // collision-free WGSL names used by packed byte stores
	lanes                    int                   // loop iterations per compute invocation (LanesPerInvocation)
	loopDepth                int                   // WGSL loops enclosing the statement being emitted, excluding the lane loop

	// tables/tableOrder/freshName support constant string lookup tables
	// (T[x] where T is a Go string constant). tables maps the constant
	// object to its emitted WGSL name (first use wins); tableOrder records
	// that first-use order for deterministic module-scope emission.
	// freshName is transpileWGSL's collision-free name generator, shared
	// with the packed-byte-store temporaries so table names cannot shadow
	// any Go identifier, buffer, or Params field.
	tables     map[*types.Const]string
	tableOrder []*types.Const
	freshName  func(string) string
}

func (e *wgslEmitter) writeIndent() {
	e.sb.WriteString(strings.Repeat("  ", e.indent))
}

// emitStmts lowers a statement list. retTarget, when non-empty, is the WGSL
// lvalue text a bare `return expr` (only reachable while inlining an SPMD
// callee, per gpu_eligible.go's depth==0 return rejection) should assign
// into instead of emitting a WGSL `return`.
func (e *wgslEmitter) emitStmts(stmts []ast.Stmt, retTarget string) error {
	for _, stmt := range stmts {
		if err := e.emitStmt(stmt, retTarget); err != nil {
			return err
		}
	}
	return nil
}

func (e *wgslEmitter) emitStmt(stmt ast.Stmt, retTarget string) error {
	switch x := stmt.(type) {
	case *ast.BlockStmt:
		return e.emitStmts(x.List, retTarget)

	case *ast.ExprStmt:
		// Only reachable form here is an inlined-callee call used as a
		// statement with a discarded result, which does not occur in the
		// supported subset; reject defensively.
		return fmt.Errorf("transpileWGSL: unsupported bare expression statement")

	case *ast.DeclStmt:
		// Checked assertions: gpu_eligible.go rejects any declaration that
		// is not a GenDecl of ValueSpecs, so a mismatch here is an internal
		// inconsistency -- report it as a transpile error, never panic.
		gen, ok := x.Decl.(*ast.GenDecl)
		if !ok {
			return fmt.Errorf("transpileWGSL: unsupported declaration %T", x.Decl)
		}
		for _, spec := range gen.Specs {
			vs, ok := spec.(*ast.ValueSpec)
			if !ok {
				return fmt.Errorf("transpileWGSL: unsupported declaration specifier %T", spec)
			}
			for i, name := range vs.Names {
				ty, err := e.wgslTypeOfIdent(name)
				if err != nil {
					return err
				}
				var rhs string
				if i < len(vs.Values) {
					s, err := e.emitExpr(vs.Values[i])
					if err != nil {
						return err
					}
					rhs = s
				} else {
					rhs = wgslZeroValue(ty)
				}
				wgslName := e.declareLocal(name)
				e.writeIndent()
				fmt.Fprintf(e.sb, "var %s: %s = %s;\n", wgslName, ty, rhs)
			}
		}
		return nil

	case *ast.AssignStmt:
		return e.emitAssign(x, retTarget)

	case *ast.IncDecStmt:
		if idx, ok := e.packedByteTarget(x.X); ok {
			op := "+"
			if x.Tok == token.DEC {
				op = "-"
			}
			return e.emitPackedByteUpdate(idx, op, "1")
		}
		lv, rhs, err := e.incDecText(x)
		if err != nil {
			return err
		}
		e.writeIndent()
		fmt.Fprintf(e.sb, "%s = %s;\n", lv, rhs)
		return nil

	case *ast.IfStmt:
		cond, err := e.emitExpr(x.Cond)
		if err != nil {
			return err
		}
		e.writeIndent()
		fmt.Fprintf(e.sb, "if (%s) {\n", cond)
		e.indent++
		if err := e.emitStmts(x.Body.List, retTarget); err != nil {
			return err
		}
		e.indent--
		if x.Else != nil {
			e.writeIndent()
			e.sb.WriteString("} else {\n")
			e.indent++
			if err := e.emitStmt(x.Else, retTarget); err != nil {
				return err
			}
			e.indent--
		}
		e.writeIndent()
		e.sb.WriteString("}\n")
		return nil

	case *ast.ForStmt:
		var initTxt, postTxt string
		if init, ok := x.Init.(*ast.AssignStmt); ok && x.Init != nil {
			s, err := e.forInitText(init)
			if err != nil {
				return err
			}
			initTxt = s
		}
		condTxt := ""
		if x.Cond != nil {
			s, err := e.emitExpr(x.Cond)
			if err != nil {
				return err
			}
			condTxt = s
		}
		switch post := x.Post.(type) {
		case nil:
		case *ast.IncDecStmt:
			if _, packed := e.packedByteTarget(post.X); packed {
				return fmt.Errorf("transpileWGSL: unsupported byte-slice element as for-loop post target")
			}
			lv, rhs, err := e.incDecText(post)
			if err != nil {
				return err
			}
			postTxt = fmt.Sprintf("%s = %s", lv, rhs)
		case *ast.AssignStmt:
			s, err := e.forPostAssignText(post)
			if err != nil {
				return err
			}
			postTxt = s
		default:
			return fmt.Errorf("transpileWGSL: unsupported for-loop post statement %T", x.Post)
		}
		e.writeIndent()
		fmt.Fprintf(e.sb, "for (%s; %s; %s) {\n", initTxt, condTxt, postTxt)
		e.indent++
		e.loopDepth++
		if err := e.emitStmts(x.Body.List, retTarget); err != nil {
			return err
		}
		e.loopDepth--
		e.indent--
		e.writeIndent()
		e.sb.WriteString("}\n")
		return nil

	case *ast.RangeStmt:
		// "for i := range n" over an integer bound, allowlisted narrowly by
		// gpu_eligible.go's checkStmt (key-only, DEFINE, integer X); lowers
		// to the equivalent WGSL counted for-loop.
		ident, ok := x.Key.(*ast.Ident)
		if !ok {
			return fmt.Errorf("transpileWGSL: unsupported range-loop key")
		}
		bound, err := e.emitExpr(x.X)
		if err != nil {
			return err
		}
		name := e.declareLocal(ident)
		e.writeIndent()
		fmt.Fprintf(e.sb, "for (var %s: i32 = 0; %s < %s; %s = (%s + 1)) {\n", name, name, bound, name, name)
		e.indent++
		e.loopDepth++
		if err := e.emitStmts(x.Body.List, retTarget); err != nil {
			return err
		}
		e.loopDepth--
		e.indent--
		e.writeIndent()
		e.sb.WriteString("}\n")
		return nil

	case *ast.BranchStmt:
		e.writeIndent()
		switch x.Tok {
		case token.BREAK:
			e.sb.WriteString("break;\n")
		case token.CONTINUE:
			if e.loopDepth == 0 && e.lanes == 1 {
				// No lane loop encloses the body at one lane per
				// invocation; ending the invocation is "next iteration".
				e.sb.WriteString("return;\n")
			} else {
				e.sb.WriteString("continue;\n")
			}
		default:
			return fmt.Errorf("transpileWGSL: unsupported branch statement %s", x.Tok)
		}
		return nil

	case *ast.ReturnStmt:
		if retTarget == "" {
			return fmt.Errorf("transpileWGSL: return outside of an inlined callee")
		}
		if len(x.Results) != 1 {
			return fmt.Errorf("transpileWGSL: only single-return inlined callees are supported")
		}
		v, err := e.emitExpr(x.Results[0])
		if err != nil {
			return err
		}
		e.writeIndent()
		fmt.Fprintf(e.sb, "%s = %s;\n", retTarget, v)
		return nil

	default:
		return fmt.Errorf("transpileWGSL: unsupported statement type %T", stmt)
	}
}

// forInitText lowers a classic C-style for-loop's `x := 0` init clause to
// WGSL's `var x: T = 0` init-clause spelling.
func (e *wgslEmitter) forInitText(init *ast.AssignStmt) (string, error) {
	if len(init.Lhs) != 1 || len(init.Rhs) != 1 {
		return "", fmt.Errorf("transpileWGSL: unsupported multi-value for-loop init")
	}
	ident, ok := init.Lhs[0].(*ast.Ident)
	if !ok {
		return "", fmt.Errorf("transpileWGSL: unsupported for-loop init target")
	}
	rhs, err := e.emitExpr(init.Rhs[0])
	if err != nil {
		return "", err
	}
	ty, err := e.wgslTypeOfIdent(ident)
	if err != nil {
		return "", err
	}
	name := e.declareLocal(ident)
	return fmt.Sprintf("var %s: %s = %s", name, ty, rhs), nil
}

// emitAssign lowers `:=`/`=`/compound-assign statements, including the
// inlined-SPMD-callee-call special case: a single-value `lhs := callee(...)`
// or `lhs = callee(...)` where callee is a plan.Inlined entry splices the
// callee's body in place (see the file comment).
func (e *wgslEmitter) emitAssign(x *ast.AssignStmt, retTarget string) error {
	if len(x.Lhs) != 1 || len(x.Rhs) != 1 {
		return fmt.Errorf("transpileWGSL: unsupported multi-value assignment")
	}
	if call, ok := x.Rhs[0].(*ast.CallExpr); ok {
		if decl, isInlined := e.inlined[call]; isInlined {
			return e.emitInlinedCall(x, call, decl, retTarget)
		}
	}

	rhs, err := e.emitExpr(x.Rhs[0])
	if err != nil {
		return err
	}

	if idx, ok := e.packedByteTarget(x.Lhs[0]); ok {
		if x.Tok == token.ASSIGN {
			return e.emitPackedByteStore(idx, rhs)
		}
		op, ok := compoundAssignOp(x.Tok)
		if !ok {
			return fmt.Errorf("transpileWGSL: internal error: unmapped assignment operator %s (eligibility allowlist and compoundAssignOp disagree)", x.Tok)
		}
		return e.emitPackedByteUpdate(idx, op, rhs)
	}

	if x.Tok == token.DEFINE {
		ident, ok := x.Lhs[0].(*ast.Ident)
		if !ok {
			return fmt.Errorf("transpileWGSL: unsupported := target %T", x.Lhs[0])
		}
		ty, err := e.wgslTypeOfIdent(ident)
		if err != nil {
			return err
		}
		name := e.declareLocal(ident)
		e.writeIndent()
		fmt.Fprintf(e.sb, "var %s: %s = %s;\n", name, ty, rhs)
		return nil
	}

	lv, err := e.emitExpr(x.Lhs[0])
	if err != nil {
		return err
	}
	if x.Tok == token.ASSIGN {
		e.writeIndent()
		fmt.Fprintf(e.sb, "%s = %s;\n", lv, rhs)
		return nil
	}
	// C1: an operator that reaches here without an explicit mapping is an
	// INTERNAL ERROR (gpu_eligible.go's gpuEligibleAssignToks allowlist and
	// compoundAssignOp have drifted apart), not something to silently lower
	// as a plain `=` -- that would emit valid WGSL computing the wrong
	// value with no diagnostic anywhere.
	op, ok := compoundAssignOp(x.Tok)
	if !ok {
		return fmt.Errorf("transpileWGSL: internal error: unmapped assignment operator %s (eligibility allowlist and compoundAssignOp disagree)", x.Tok)
	}
	computed := fmt.Sprintf("(%s %s %s)", lv, op, rhs)
	switch x.Tok {
	case token.ADD_ASSIGN, token.SUB_ASSIGN, token.MUL_ASSIGN:
		// Mirrors the *ast.BinaryExpr ADD/SUB/MUL/SHL masking above: these
		// compound forms can leave [0, 255] on a byte target. QUO_ASSIGN
		// (/=) and REM_ASSIGN (%=) are range-preserving, matching the
		// unmasked QUO/REM binary case, so they are deliberately excluded.
		if e.isByteExpr(x.Lhs[0]) {
			computed = "(" + computed + " & 0xffu)"
		}
	}
	e.writeIndent()
	fmt.Fprintf(e.sb, "%s = %s;\n", lv, computed)
	return nil
}

// emitInlinedCall splices decl's body into the current statement stream:
// callee locals are renamed with a unique per-call-site suffix to avoid
// collisions with the caller's own locals, callee parameters are bound
// directly to the (already-substituted) argument expression text, and the
// callee's single `return expr` becomes an assignment into the
// caller-side lhs. Only single-return callees are supported (Task 3 rejects
// multi-return SPMD functions before a plan reaches here).
func (e *wgslEmitter) emitInlinedCall(assign *ast.AssignStmt, call *ast.CallExpr, decl *ast.FuncDecl, outerRetTarget string) error {
	var lhsName string
	var packedIdx *ast.IndexExpr // non-nil: lhsName is a temporary to store into this packed byte element
	var lhsWGSLTy string
	if assign.Tok == token.DEFINE {
		ident, ok := assign.Lhs[0].(*ast.Ident)
		if !ok {
			return fmt.Errorf("transpileWGSL: unsupported := target %T", assign.Lhs[0])
		}
		ty, err := e.wgslTypeOfIdent(ident)
		if err != nil {
			return err
		}
		lhsName = e.declareLocal(ident)
		lhsWGSLTy = ty
		e.writeIndent()
		fmt.Fprintf(e.sb, "var %s: %s;\n", lhsName, lhsWGSLTy)
	} else if idx, ok := e.packedByteTarget(assign.Lhs[0]); ok {
		// A packed byte element is not a WGSL lvalue: collect the callee's
		// result in a temporary and store it once the body is spliced.
		if assign.Tok != token.ASSIGN {
			return fmt.Errorf("transpileWGSL: unsupported %s of an inlined call into a byte slice", assign.Tok)
		}
		tmp := e.freshName(e.tmpRet)
		e.writeIndent()
		fmt.Fprintf(e.sb, "var %s: u32;\n", tmp)
		packedIdx = idx
		lhsName = tmp
	} else {
		lv, err := e.emitExpr(assign.Lhs[0])
		if err != nil {
			return err
		}
		lhsName = lv
	}

	// C2: bind each callee parameter to a FRESH WGSL variable initialised
	// from the argument text, NOT to the argument text itself. Go parameters
	// are by-value: a callee that assigns to its own parameter must not
	// mutate the caller's local. Binding to the argument text emitted
	// `x = (x * 2);` straight into the CALLER's scope -- correct on the CPU
	// path, silently wrong on the GPU path.
	//
	// The fresh names carry this call site's inline suffix, so two inlined
	// calls in the same scope cannot collide either. Argument expressions
	// are emitted BEFORE e.locals is swapped, so they resolve in the
	// caller's scope.
	sig := decl.Type.Params.List
	argIdx := 0
	savedLocals := e.locals
	newLocals := map[types.Object]string{}
	for k, v := range savedLocals {
		newLocals[k] = v
	}
	e.inlineCounter++
	savedSuffix := e.inlineSuffix
	suffix := fmt.Sprintf("_i%d", e.inlineCounter)
	for _, field := range sig {
		for _, nameIdent := range field.Names {
			obj := e.info.Defs[nameIdent]
			if argIdx >= len(call.Args) {
				return fmt.Errorf("transpileWGSL: inlined call to %s has fewer arguments than parameters", decl.Name.Name)
			}
			argTxt, err := e.emitExpr(call.Args[argIdx])
			if err != nil {
				return err
			}
			paramTy, err := e.wgslTypeOfIdent(nameIdent)
			if err != nil {
				return err
			}
			paramName := nameIdent.Name + suffix
			e.writeIndent()
			fmt.Fprintf(e.sb, "var %s: %s = %s;\n", paramName, paramTy, argTxt)
			newLocals[obj] = paramName
			argIdx++
		}
	}

	e.locals = newLocals
	e.inlineSuffix = suffix
	err := e.emitStmts(decl.Body.List, lhsName)
	e.inlineSuffix = savedSuffix
	e.locals = savedLocals
	if err != nil {
		return err
	}
	if packedIdx != nil {
		if err := e.emitPackedByteStore(packedIdx, lhsName); err != nil {
			return err
		}
	}

	_ = outerRetTarget // an inlined callee always assigns lhsName directly; nested inlining is depth-limited by gpu_eligible.go
	return nil
}

// declareLocal registers ident's defining object and returns its WGSL
// spelling: the bare Go name, unless it collides with an active inlined
// parameter binding or a previously declared local of a different object,
// in which case it is suffixed to stay unique.
func (e *wgslEmitter) declareLocal(ident *ast.Ident) string {
	obj := e.info.Defs[ident]
	name := ident.Name
	if obj == nil {
		return name
	}
	if e.inlineSuffix != "" {
		name = ident.Name + e.inlineSuffix
	}
	e.locals[obj] = name
	return name
}

// isByteExpr reports whether x has Go type uint8 (possibly as
// lanes.Varying[uint8]). Such values are carried as WGSL u32 restricted to
// [0, 255]; overflowing operations must mask back into that range.
func (e *wgslEmitter) isByteExpr(x ast.Expr) bool {
	t := e.info.TypeOf(x)
	if s, ok := t.(*types.SPMDType); ok {
		t = s.Elem()
	}
	if t == nil {
		return false
	}
	b, ok := t.Underlying().(*types.Basic)
	return ok && b.Kind() == types.Uint8
}

// isByteSlice reports whether x names a free byte-slice buffer. Such a
// buffer holds the Go bytes unmodified, packed four per u32 word
// (little-endian: word w holds bytes 4w..4w+3), so its elements are read
// and written through packedByteRead / emitPackedByteStore.
func (e *wgslEmitter) isByteSlice(x ast.Expr) bool {
	ident, ok := x.(*ast.Ident)
	if !ok {
		return false
	}
	return e.byteBufs[e.info.Uses[ident]]
}

// packedByteTarget returns lhs as an index expression when it is an element
// of a packed byte-slice buffer.
func (e *wgslEmitter) packedByteTarget(lhs ast.Expr) (*ast.IndexExpr, bool) {
	idx, ok := lhs.(*ast.IndexExpr)
	if !ok || !e.isByteSlice(idx.X) {
		return nil, false
	}
	return idx, true
}

// packedByteRead extracts byte idx from a packed u32-word buffer.
func packedByteRead(buf, idx string) string {
	return fmt.Sprintf("((%s[u32(%s) >> 2u] >> ((u32(%s) & 3u) * 8u)) & 0xffu)", buf, idx, idx)
}

// emitPackedByteStore writes the byte value text v into element idx of a
// packed byte buffer by rewriting only that byte of its owning word. The
// eligibility gate (writeIndexReject) guarantees no other invocation
// touches the word.
func (e *wgslEmitter) emitPackedByteStore(idx *ast.IndexExpr, v string) error {
	buf, err := e.emitExpr(idx.X)
	if err != nil {
		return err
	}
	i, err := e.emitExpr(idx.Index)
	if err != nil {
		return err
	}
	j, sh := e.tmpIdx, e.tmpShift
	e.writeIndent()
	fmt.Fprintf(e.sb, "{ let %s = u32(%s); let %s = (%s & 3u) * 8u; %s[%s >> 2u] = (%s[%s >> 2u] & ~(0xffu << %s)) | ((%s & 0xffu) << %s); }\n",
		j, i, sh, j, buf, j, buf, j, sh, v, sh)
	return nil
}

// emitPackedByteUpdate lowers `s[E] op= rhs` (and ++/-- with rhs "1") on a
// packed byte element: the old value is the packed read, and the store's
// `& 0xffu` wraps the result like Go's byte arithmetic.
func (e *wgslEmitter) emitPackedByteUpdate(idx *ast.IndexExpr, op, rhs string) error {
	old, err := e.emitExpr(idx)
	if err != nil {
		return err
	}
	return e.emitPackedByteStore(idx, fmt.Sprintf("(%s %s %s)", old, op, rhs))
}

// forPostAssignText lowers a for-loop post assignment (e.g. `c += 2`)
// through emitAssign, so it gets the same byte masking, and returns the
// single statement without indent or `;` for the for header.
func (e *wgslEmitter) forPostAssignText(x *ast.AssignStmt) (string, error) {
	if _, packed := e.packedByteTarget(x.Lhs[0]); packed || x.Tok == token.DEFINE {
		return "", fmt.Errorf("transpileWGSL: unsupported for-loop post assignment")
	}
	saved, savedIndent := e.sb, e.indent
	var tmp strings.Builder
	e.sb, e.indent = &tmp, 0
	err := e.emitAssign(x, "")
	e.sb, e.indent = saved, savedIndent
	if err != nil {
		return "", err
	}
	s := strings.TrimSuffix(tmp.String(), ";\n")
	if strings.ContainsAny(s, ";\n") {
		return "", fmt.Errorf("transpileWGSL: for-loop post assignment does not lower to one statement")
	}
	return s, nil
}

// incDecText lowers an IncDecStmt's `x++`/`x--` to its WGSL lvalue text and
// the masked-as-needed RHS text (`(lv + 1)` or `(lv - 1)`, wrapped in
// `(... & 0xffu)` when x.X is a byte value). Shared by emitStmt's
// *ast.IncDecStmt case and the *ast.ForStmt post-clause handling below, so
// a byte-typed classic-for counter (`for b := byte(250); b != 4; b++`)
// wraps at 256 exactly like Go on both paths.
func (e *wgslEmitter) incDecText(x *ast.IncDecStmt) (lv, rhs string, err error) {
	lv, err = e.emitExpr(x.X)
	if err != nil {
		return "", "", err
	}
	op := "+"
	if x.Tok == token.DEC {
		op = "-"
	}
	rhs = fmt.Sprintf("(%s %s 1)", lv, op)
	if e.isByteExpr(x.X) {
		// ++/-- on a byte value is ADD/SUB in disguise and can leave
		// [0, 255] just like the binary-operator case above.
		rhs = "(" + rhs + " & 0xffu)"
	}
	return lv, rhs, nil
}

// emitExpr lowers a single expression to WGSL text.
func (e *wgslEmitter) emitExpr(expr ast.Expr) (string, error) {
	switch x := expr.(type) {
	case *ast.Ident:
		obj := e.info.Uses[x]
		if obj == nil {
			obj = e.info.Defs[x]
		}
		if s, ok := e.locals[obj]; ok {
			return s, nil
		}
		if s, ok := e.freeSubst[obj]; ok {
			if _, isBuf := obj.Type().Underlying().(*types.Slice); isBuf {
				e.usedBufs[obj] = true
			}
			return s, nil
		}
		if c, ok := obj.(*types.Const); ok {
			ty, err := e.wgslTypeOfIdent(x)
			if err != nil {
				return "", err
			}
			return constLiteral(c.Val(), ty), nil
		}
		if v, ok := obj.(*types.Var); ok {
			if e.iterObj != nil && v == e.iterObj {
				return e.iterIdent, nil
			}
			// Neither a local, a free substitution, nor the iteration
			// variable: the shader would reference an undeclared name and
			// fail at shader creation. Fail closed to the CPU path.
			return "", fmt.Errorf("transpileWGSL: variable %s is not declared in the kernel", x.Name)
		}
		return x.Name, nil

	case *ast.BasicLit:
		return x.Value, nil

	case *ast.ParenExpr:
		s, err := e.emitExpr(x.X)
		if err != nil {
			return "", err
		}
		return "(" + s + ")", nil

	case *ast.BinaryExpr:
		lhs, err := e.emitExpr(x.X)
		if err != nil {
			return "", err
		}
		rhs, err := e.emitExpr(x.Y)
		if err != nil {
			return "", err
		}
		op, err := wgslBinOp(x.Op)
		if err != nil {
			return "", err
		}
		out := fmt.Sprintf("(%s %s %s)", lhs, op, rhs)
		switch x.Op {
		case token.ADD, token.SUB, token.MUL, token.SHL:
			// A uint8 result of one of these ops can leave [0, 255]; every
			// other Go byte op (/ % >> & | ^, comparisons) is
			// range-preserving and needs no mask -- see the file comment
			// and isByteExpr.
			if e.isByteExpr(x) {
				out = "(" + out + " & 0xffu)"
			}
		}
		return out, nil

	case *ast.UnaryExpr:
		s, err := e.emitExpr(x.X)
		if err != nil {
			return "", err
		}
		switch x.Op {
		case token.SUB:
			if e.isByteExpr(x) {
				// WGSL has no unary minus on u32.
				return "((0u - " + s + ") & 0xffu)", nil
			}
			return "(-" + s + ")", nil
		case token.NOT:
			return "(!" + s + ")", nil
		default:
			return "", fmt.Errorf("transpileWGSL: unsupported unary operator %s", x.Op)
		}

	case *ast.IndexExpr:
		if _, c, ok := constStringOf(e.info, x.X); ok {
			name, seen := e.tables[c]
			if !seen {
				name = e.freshName(wgslSafeIdent(c.Name()) + "_tbl")
				e.tables[c] = name
				e.tableOrder = append(e.tableOrder, c)
			}
			idx, err := e.emitExpr(x.Index)
			if err != nil {
				return "", err
			}
			return fmt.Sprintf("%s[u32(%s)]", name, idx), nil
		}
		base, err := e.emitExpr(x.X)
		if err != nil {
			return "", err
		}
		idx, err := e.emitExpr(x.Index)
		if err != nil {
			return "", err
		}
		if e.isByteSlice(x.X) {
			return packedByteRead(base, idx), nil
		}
		return fmt.Sprintf("%s[%s]", base, idx), nil

	case *ast.CallExpr:
		return e.emitCall(x)

	default:
		return "", fmt.Errorf("transpileWGSL: unsupported expression type %T", expr)
	}
}

// emitCall lowers predeclared type conversions, the lanes.Varying[T](x)
// conversion, approved math.* builtins, and lanes.Count/Index/FMA. Inlined
// SPMD calls are handled separately by emitInlinedCall (they only occur as
// the direct RHS of an assignment, which emitAssign intercepts before
// reaching here).
func (e *wgslEmitter) emitCall(call *ast.CallExpr) (string, error) {
	switch fun := call.Fun.(type) {
	case *ast.Ident:
		if obj := e.info.Uses[fun]; obj != nil {
			if _, ok := obj.(*types.TypeName); ok {
				return e.emitConversion(fun.Name, call.Args)
			}
		}
		return "", fmt.Errorf("transpileWGSL: unsupported call to %s", fun.Name)

	case *ast.IndexExpr:
		// lanes.Varying[T](x): a same-lane no-op conversion to T's scalar
		// WGSL type, since on the GPU one invocation already is one lane.
		sel, ok := fun.X.(*ast.SelectorExpr)
		if !ok {
			return "", fmt.Errorf("transpileWGSL: unsupported generic call")
		}
		if pkgIdent, ok := sel.X.(*ast.Ident); ok && sel.Sel.Name == "Varying" {
			if pn, ok := e.info.Uses[pkgIdent].(*types.PkgName); ok && pn.Imported().Path() == "lanes" {
				tv := e.info.TypeOf(fun.Index)
				// byte keeps its [0, 255] masking; wgslTypeOf maps it to
				// plain u32, which would drop the mask.
				if b, ok := tv.Underlying().(*types.Basic); ok && b.Kind() == types.Uint8 {
					return e.emitConversion("byte", call.Args)
				}
				wty, err := wgslTypeOf(tv)
				if err != nil {
					return "", err
				}
				return e.emitConversion(wty, call.Args)
			}
		}
		return "", fmt.Errorf("transpileWGSL: unsupported generic call")

	case *ast.SelectorExpr:
		pkgIdent, ok := fun.X.(*ast.Ident)
		if !ok {
			return "", fmt.Errorf("transpileWGSL: unsupported method-style call")
		}
		pn, ok := e.info.Uses[pkgIdent].(*types.PkgName)
		if !ok {
			return "", fmt.Errorf("transpileWGSL: unsupported call target")
		}
		switch pn.Imported().Path() {
		case "math":
			wgslName, ok := wgslprint.MathFuncs[fun.Sel.Name]
			if !ok {
				return "", fmt.Errorf("transpileWGSL: unsupported math function %s", fun.Sel.Name)
			}
			return e.emitConversion(wgslName, call.Args)
		case "lanes":
			if fun.Sel.Name == "FMA" {
				return e.emitConversion("fma", call.Args)
			}
			return "", fmt.Errorf("transpileWGSL: unsupported lanes function %s", fun.Sel.Name)
		default:
			return "", fmt.Errorf("transpileWGSL: unsupported package call %s.%s", pkgIdent.Name, fun.Sel.Name)
		}

	default:
		return "", fmt.Errorf("transpileWGSL: unsupported call expression %T", call.Fun)
	}
}

// emitConversion renders `wgslFn(args...)`, used both for genuine type
// conversions (int(x) -> i32(x)) and for builtin function calls
// (sqrt(x)) which share the same "name(args)" shape.
func (e *wgslEmitter) emitConversion(name string, args []ast.Expr) (string, error) {
	// byte(x)/uint8(x) mask their result into [0, 255] like every other
	// overflowing byte op (see the file comment); handled here, before the
	// generic path below, because that path joins args with the target
	// name as a plain call and has no notion of masking.
	if name == "byte" || name == "uint8" {
		if len(args) != 1 {
			return "", fmt.Errorf("transpileWGSL: %s conversion takes one argument", name)
		}
		s, err := e.emitExpr(args[0])
		if err != nil {
			return "", err
		}
		return "(u32(" + s + ") & 0xffu)", nil
	}

	wgslName := name
	switch name {
	case "int", "int32":
		wgslName = "i32"
	case "uint", "uint32":
		wgslName = "u32"
	case "float32":
		wgslName = "f32"
	}
	var parts []string
	for _, a := range args {
		s, err := e.emitExpr(a)
		if err != nil {
			return "", err
		}
		parts = append(parts, s)
	}
	return fmt.Sprintf("%s(%s)", wgslName, strings.Join(parts, ", ")), nil
}

// wgslReservedWords are WGSL keywords/reserved words (and the predeclared
// scalar type names) that must never be emitted as a user identifier. The
// list is not exhaustive of the full WGSL reserved-word set, but covers
// every keyword and builtin type name a Go identifier could plausibly be,
// and anything missed is only a missed optimization now that a shader
// compile failure is reported back to Go (I2) instead of silently ignored.
var wgslReservedWords = map[string]bool{
	"alias": true, "break": true, "case": true, "const": true, "const_assert": true,
	"continue": true, "continuing": true, "default": true, "diagnostic": true,
	"discard": true, "else": true, "enable": true, "false": true, "fn": true,
	"for": true, "if": true, "let": true, "loop": true, "override": true,
	"requires": true, "return": true, "struct": true, "switch": true, "true": true,
	"var": true, "while": true, "select": true,
	"bool": true, "f16": true, "f32": true, "i32": true, "u32": true,
	"vec2": true, "vec3": true, "vec4": true, "mat2x2": true, "array": true,
	"atomic": true, "ptr": true, "sampler": true, "texture_1d": true,
}

// wgslSafeIdent maps a Go identifier onto a legal, non-reserved WGSL
// identifier: non-alphanumeric characters become '_', a leading digit or
// a leading double underscore (reserved by the WGSL spec) is prefixed, and
// reserved words are suffixed. Uniqueness against other emitted names is
// the caller's job.
func wgslSafeIdent(name string) string {
	if name == "" {
		return "v_"
	}
	var sb strings.Builder
	for i, r := range name {
		switch {
		case r >= 'a' && r <= 'z', r >= 'A' && r <= 'Z', r == '_':
			sb.WriteRune(r)
		case r >= '0' && r <= '9' && i > 0:
			sb.WriteRune(r)
		default:
			sb.WriteByte('_')
		}
	}
	out := sb.String()
	if out[0] >= '0' && out[0] <= '9' {
		out = "v_" + out
	}
	if strings.HasPrefix(out, "__") {
		out = "v" + out
	}
	if wgslReservedWords[out] {
		out += "_"
	}
	return out
}

// wgslTypeOfIdent resolves ident's declared Go type to its WGSL scalar
// spelling.
// I1: the error is PROPAGATED, never swallowed into a hardcoded "i32" --
// silently truncating an int64/float64 local to i32 on the GPU path only is
// exactly the class of invisible wrong answer this pipeline must not have.
// gpu_eligible.go's checkLocalTypes rejects such loops up front, so an error
// here means the two allowlists disagree: a transpile failure, which
// gpuAnalyzeLoop turns into a clean CPU-only fallback.
func (e *wgslEmitter) wgslTypeOfIdent(ident *ast.Ident) (string, error) {
	obj := e.info.Defs[ident]
	if obj == nil {
		obj = e.info.Uses[ident]
	}
	if obj == nil {
		return "", fmt.Errorf("transpileWGSL: cannot resolve identifier %s to an object", ident.Name)
	}
	t := obj.Type()
	// An untyped package-level constant (e.g. `const X0 = -2.5`) has no
	// WGSL spelling of its own; the type checker records the type it was
	// CONVERTED TO at this use site, which is the one to emit. If that is
	// still untyped (the constant is used in an untyped context), fall
	// through to wgslTypeOf, which rejects it -- fail closed.
	if basic, ok := t.Underlying().(*types.Basic); ok && basic.Info()&types.IsUntyped != 0 {
		if tv, ok := e.info.Types[ident]; ok && tv.Type != nil {
			t = tv.Type
		}
	}
	ty, err := wgslTypeOf(t)
	if err != nil {
		return "", fmt.Errorf("transpileWGSL: identifier %s: %w", ident.Name, err)
	}
	return ty, nil
}

// wgslTypeOf maps a Go type (basic, or lanes.Varying[T] via *types.SPMDType)
// to its WGSL scalar spelling.
func wgslTypeOf(t types.Type) (string, error) {
	if t == nil {
		return "", fmt.Errorf("transpileWGSL: nil type")
	}
	if spmd, ok := t.(*types.SPMDType); ok {
		return wgslTypeOf(spmd.Elem())
	}
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return "", fmt.Errorf("transpileWGSL: unsupported type %s", t.String())
	}
	switch basic.Kind() {
	case types.Int, types.Int32:
		return "i32", nil
	case types.Uint, types.Uint32, types.Uint8:
		return "u32", nil
	case types.Float32:
		return "f32", nil
	case types.Bool:
		return "bool", nil
	case types.UntypedInt, types.UntypedRune:
		// An untyped constant used inside a kernel. WGSL's abstract-numeric
		// literals convert implicitly, and every float64 variable is already
		// rejected by the eligibility gate, so an untyped int can only ever
		// be consumed as an i32 (or widened to f32 by WGSL itself) and an
		// untyped float only as an f32.
		return "i32", nil
	case types.UntypedFloat:
		return "f32", nil
	case types.UntypedBool:
		return "bool", nil
	default:
		return "", fmt.Errorf("transpileWGSL: unsupported basic type %s", basic.String())
	}
}

// wgslZeroValue returns the WGSL zero-literal for a scalar type name.
func wgslZeroValue(wgslTy string) string {
	switch wgslTy {
	case "f32":
		return "0.0"
	case "bool":
		return "false"
	default:
		return "0"
	}
}

// wgslBinOp maps a Go binary operator token to its WGSL spelling; the Go and
// WGSL operator sets agree on spelling and semantics for every operator the
// eligibility allowlist (gpu_eligible.go's checkExpr) admits.
func wgslBinOp(op token.Token) (string, error) {
	switch op {
	case token.ADD:
		return "+", nil
	case token.SUB:
		return "-", nil
	case token.MUL:
		return "*", nil
	case token.QUO:
		return "/", nil
	case token.REM:
		return "%", nil
	case token.AND:
		return "&", nil
	case token.OR:
		return "|", nil
	case token.XOR:
		return "^", nil
	case token.SHL:
		return "<<", nil
	case token.SHR:
		return ">>", nil
	case token.LAND:
		return "&&", nil
	case token.LOR:
		return "||", nil
	case token.EQL:
		return "==", nil
	case token.NEQ:
		return "!=", nil
	case token.LSS:
		return "<", nil
	case token.LEQ:
		return "<=", nil
	case token.GTR:
		return ">", nil
	case token.GEQ:
		return ">=", nil
	default:
		return "", fmt.Errorf("transpileWGSL: unsupported binary operator %s", op)
	}
}

// compoundAssignOp maps a compound-assignment token (+=, -=, ...) to the
// WGSL binary operator to expand it into `lhs = (lhs op rhs)`. ok is false
// for any token it does not explicitly handle -- callers MUST treat that as
// an error rather than falling back to a plain `=` (C1). This set and
// gpu_eligible.go's gpuEligibleAssignToks must stay in sync.
func compoundAssignOp(tok token.Token) (string, bool) {
	switch tok {
	case token.ADD_ASSIGN:
		return "+", true
	case token.SUB_ASSIGN:
		return "-", true
	case token.MUL_ASSIGN:
		return "*", true
	case token.QUO_ASSIGN:
		return "/", true
	case token.REM_ASSIGN:
		return "%", true
	default:
		return "", false
	}
}

// constLiteral renders a Go package-level constant's value as a WGSL
// literal of the given target scalar type.
func constLiteral(v constant.Value, wgslTy string) string {
	if wgslTy == "f32" {
		f, _ := constant.Float64Val(v)
		s := strconv.FormatFloat(f, 'g', -1, 64)
		if !strings.ContainsAny(s, ".eE") {
			s += ".0"
		}
		return s
	}
	return v.String()
}
