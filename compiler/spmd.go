package compiler

// This file extracts SPMD metadata from the typed AST for use during LLVM IR
// generation. TinyGo uses golang.org/x/tools/go/ssa which has no SPMD support,
// so we build a side-table from go/ast and go/types (which our Go fork extends).

import (
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strconv"
	"strings"

	"github.com/tinygo-org/tinygo/loader"
	"golang.org/x/tools/go/ssa"
	"tinygo.org/x/go-llvm"
)

// SPMDLoopInfo holds metadata about a go for loop extracted from the AST.
type SPMDLoopInfo struct {
	ForPos     token.Pos // position of "for" keyword
	BodyStart  token.Pos // start of loop body (opening brace)
	BodyEnd    token.Pos // end of loop body (closing brace)
	LaneCount  int64     // SIMD lane count (from type checker)
	Constraint int64     // range[N] constraint; -1 if unconstrained
}

// SPMDParamInfo holds info about a single varying parameter.
type SPMDParamInfo struct {
	Index      int
	ElemType   types.Type
	Constraint int64 // -1 if unconstrained
}

// SPMDFuncInfo holds metadata about a function with varying parameters.
type SPMDFuncInfo struct {
	HasVaryingParams  bool
	HasVaryingResults bool
	VaryingParams     []SPMDParamInfo
}

// SPMDInfo holds aggregated SPMD metadata for a package.
type SPMDInfo struct {
	Loops      map[token.Pos]*SPMDLoopInfo // keyed by ForPos
	Funcs      map[*types.Func]*SPMDFuncInfo
	loopRanges []spmdPosRange // sorted by Start for binary search
}

// spmdPosRange is a position range for "inside SPMD loop?" queries.
type spmdPosRange struct {
	start, end token.Pos
	info       *SPMDLoopInfo
}

// extractSPMDLoops walks the AST and extracts all SPMD loop metadata.
// Closures inside SPMD loops are handled by position: their instructions
// have positions inside the loop body, so isInSPMDLoop works for them.
func extractSPMDLoops(pkg *loader.Package) map[token.Pos]*SPMDLoopInfo {
	loops := make(map[token.Pos]*SPMDLoopInfo)

	for _, file := range pkg.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			rangeStmt, ok := n.(*ast.RangeStmt)
			if !ok {
				return true
			}

			// Check if this is an SPMD loop (go for).
			// The IsSpmd field is set by our Go fork's parser.
			if !rangeStmt.IsSpmd {
				return true
			}

			// Extract constraint if present.
			constraint := int64(-1) // -1 means unconstrained
			if rangeStmt.Constraint != nil {
				if basicLit, ok := rangeStmt.Constraint.(*ast.BasicLit); ok {
					if basicLit.Kind == token.INT {
						if val, err := strconv.ParseInt(basicLit.Value, 10, 64); err == nil {
							constraint = val
						}
					}
				}
			}

			// Create loop info.
			info := &SPMDLoopInfo{
				ForPos:     rangeStmt.For,
				BodyStart:  rangeStmt.Body.Lbrace,
				BodyEnd:    rangeStmt.Body.Rbrace,
				LaneCount:  rangeStmt.LaneCount, // set by type checker
				Constraint: constraint,
			}

			loops[rangeStmt.For] = info
			return true
		})
	}

	return loops
}

// extractSPMDFuncs walks the package scope and extracts functions with varying
// parameters or results.
func extractSPMDFuncs(pkg *loader.Package) map[*types.Func]*SPMDFuncInfo {
	funcs := make(map[*types.Func]*SPMDFuncInfo)

	// Process top-level functions.
	scope := pkg.Pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if fn, ok := obj.(*types.Func); ok {
			if info := analyzeSPMDSignature(fn.Type().(*types.Signature)); info != nil {
				funcs[fn] = info
			}
		}

		// Also check methods on named types.
		if typeName, ok := obj.(*types.TypeName); ok {
			if namedType, ok := typeName.Type().(*types.Named); ok {
				for i := 0; i < namedType.NumMethods(); i++ {
					method := namedType.Method(i)
					if info := analyzeSPMDSignature(method.Type().(*types.Signature)); info != nil {
						funcs[method] = info
					}
				}
			}
		}
	}

	return funcs
}

// analyzeSPMDSignature checks if a function signature has varying parameters
// or results. Returns nil if the function is not an SPMD function.
func analyzeSPMDSignature(sig *types.Signature) *SPMDFuncInfo {
	info := &SPMDFuncInfo{}

	// Check parameters.
	params := sig.Params()
	for i := 0; i < params.Len(); i++ {
		param := params.At(i)
		if spmdType, ok := param.Type().(*types.SPMDType); ok {
			if spmdType.IsVarying() {
				info.HasVaryingParams = true
				info.VaryingParams = append(info.VaryingParams, SPMDParamInfo{
					Index:      i,
					ElemType:   spmdType.Elem(),
					Constraint: spmdType.Constraint(),
				})
			}
		}
	}

	// Check results.
	results := sig.Results()
	for i := 0; i < results.Len(); i++ {
		result := results.At(i)
		if spmdType, ok := result.Type().(*types.SPMDType); ok {
			if spmdType.IsVarying() {
				info.HasVaryingResults = true
			}
		}
	}

	// Return nil if no SPMD characteristics found.
	if !info.HasVaryingParams && !info.HasVaryingResults {
		return nil
	}

	return info
}

// loadSPMDInfo extracts SPMD metadata from a package and stores it in the
// compiler context. If the package contains no SPMD code, spmdInfo remains nil.
func (c *compilerContext) loadSPMDInfo(pkg *loader.Package) {
	loops := extractSPMDLoops(pkg)
	funcs := extractSPMDFuncs(pkg)

	// If no SPMD code, leave spmdInfo as nil.
	if len(loops) == 0 && len(funcs) == 0 {
		return
	}

	// Build the SPMDInfo structure.
	info := &SPMDInfo{
		Loops: loops,
		Funcs: funcs,
	}

	// Build sorted position ranges for binary search.
	info.loopRanges = make([]spmdPosRange, 0, len(loops))
	for _, loopInfo := range loops {
		info.loopRanges = append(info.loopRanges, spmdPosRange{
			start: loopInfo.BodyStart,
			end:   loopInfo.BodyEnd,
			info:  loopInfo,
		})
	}

	// Sort by start position for binary search.
	sort.Slice(info.loopRanges, func(i, j int) bool {
		return info.loopRanges[i].start < info.loopRanges[j].start
	})

	c.spmdInfo = info
}

// isInSPMDLoop checks if a given position is inside an SPMD loop body.
// Returns the loop info if found, nil otherwise.
func (c *compilerContext) isInSPMDLoop(pos token.Pos) *SPMDLoopInfo {
	if c.spmdInfo == nil {
		return nil
	}

	// Binary search through sorted ranges.
	idx := sort.Search(len(c.spmdInfo.loopRanges), func(i int) bool {
		return c.spmdInfo.loopRanges[i].start > pos
	})

	// Check the range before the found index (if it exists).
	if idx > 0 {
		r := &c.spmdInfo.loopRanges[idx-1]
		if r.start <= pos && pos <= r.end {
			return r.info
		}
	}

	return nil
}

// getSPMDLoopAt returns the SPMD loop info for a loop starting at the given
// position (the "for" keyword position). Returns nil if not found.
func (c *compilerContext) getSPMDLoopAt(forPos token.Pos) *SPMDLoopInfo {
	if c.spmdInfo == nil {
		return nil
	}
	return c.spmdInfo.Loops[forPos]
}

// getSPMDFuncInfo returns the SPMD function info for a given function.
// Returns nil if the function is not an SPMD function.
func (c *compilerContext) getSPMDFuncInfo(fn *types.Func) *SPMDFuncInfo {
	if c.spmdInfo == nil {
		return nil
	}
	return c.spmdInfo.Funcs[fn]
}

// isSPMDFunction checks if an SSA function is an SPMD function (has varying
// parameters or results).
func (c *compilerContext) isSPMDFunction(fn *ssa.Function) bool {
	if c.spmdInfo == nil {
		return false
	}

	// Try to get the function object.
	if obj := fn.Object(); obj != nil {
		if typesFn, ok := obj.(*types.Func); ok {
			return c.spmdInfo.Funcs[typesFn] != nil
		}
	}

	// Fallback: analyze the signature directly for cross-package calls.
	return analyzeSPMDSignature(fn.Signature) != nil
}

// hasSPMDCode returns true if the current package contains any SPMD code
// (loops or functions).
func (c *compilerContext) hasSPMDCode() bool {
	return c.spmdInfo != nil
}

// spmdLaneCount returns the number of SIMD lanes for a given LLVM element type.
// For WASM SIMD128: 128 bits / element size in bits.
func (c *compilerContext) spmdLaneCount(elemType llvm.Type) int {
	elemSize := c.targetData.TypeAllocSize(elemType)
	if elemSize == 0 {
		return 1
	}
	return 16 / int(elemSize) // 128-bit SIMD
}

// spmdEffectiveLaneCount returns the lane count for an SPMDType, respecting constraints.
// If the type has an explicit constraint (e.g., Varying[int, 8]), that value is used.
// Otherwise, the lane count is derived from the SIMD register width.
func (c *compilerContext) spmdEffectiveLaneCount(spmdType *types.SPMDType, elemLLVM llvm.Type) int {
	if spmdType.IsConstrained() && spmdType.Constraint() > 0 {
		return int(spmdType.Constraint())
	}
	return c.spmdLaneCount(elemLLVM)
}

// splatScalar broadcasts a scalar value to fill all lanes of a vector type.
func (b *builder) splatScalar(scalar llvm.Value, vecType llvm.Type) llvm.Value {
	undef := llvm.Undef(vecType)
	zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
	ins := b.CreateInsertElement(undef, scalar, zero, "")
	mask := llvm.ConstNull(llvm.VectorType(b.ctx.Int32Type(), vecType.VectorSize()))
	return b.CreateShuffleVector(ins, undef, mask, "splat")
}

// arrayToVector converts an LLVM [N x T] array value to a <N x T> vector
// by extracting each element and inserting it into a vector.
func (b *builder) arrayToVector(arr llvm.Value, vecType llvm.Type) llvm.Value {
	n := vecType.VectorSize()
	vec := llvm.Undef(vecType)
	for i := 0; i < n; i++ {
		elem := b.CreateExtractValue(arr, i, "")
		idx := llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
		vec = b.CreateInsertElement(vec, elem, idx, "")
	}
	return vec
}

// spmdBroadcastMatch ensures both operands have matching types for SPMD operations.
// If one operand is a vector and the other is a scalar, the scalar is splatted.
// If both are vectors with different lane counts, the wider one is resized to match the narrower.
func (b *builder) spmdBroadcastMatch(x, y llvm.Value) (llvm.Value, llvm.Value) {
	xIsVec := x.Type().TypeKind() == llvm.VectorTypeKind
	yIsVec := y.Type().TypeKind() == llvm.VectorTypeKind
	if xIsVec && !yIsVec {
		y = b.splatScalar(y, x.Type())
	} else if !xIsVec && yIsVec {
		x = b.splatScalar(x, y.Type())
	} else if xIsVec && yIsVec && x.Type().VectorSize() != y.Type().VectorSize() {
		// Both vectors but different lane counts. The narrower width is authoritative
		// (determined by the SPMD loop's effective lane count). Resize the wider one.
		xSize := x.Type().VectorSize()
		ySize := y.Type().VectorSize()
		if xSize < ySize {
			y = b.spmdResizeVector(y, xSize, x.Type().ElementType())
		} else {
			x = b.spmdResizeVector(x, ySize, y.Type().ElementType())
		}
	}
	return x, y
}

// spmdResizeVector resizes a vector to the target lane count.
// Used when two SPMD vectors have mismatched widths (e.g., Varying[byte] constant
// with 16 lanes vs loop-effective 4 lanes). Truncates wider vectors or re-splats.
func (b *builder) spmdResizeVector(vec llvm.Value, targetLanes int, elemType llvm.Type) llvm.Value {
	targetVecType := llvm.VectorType(elemType, targetLanes)
	// Use shuffle to truncate. LLVM constant-folds shuffles on constant vectors,
	// so there is no need for a separate constant fast-path.
	srcLanes := vec.Type().VectorSize()
	if targetLanes <= srcLanes {
		mask := make([]llvm.Value, targetLanes)
		for i := range mask {
			mask[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
		}
		maskVec := llvm.ConstVector(mask, false)
		return b.CreateShuffleVector(vec, llvm.Undef(vec.Type()), maskVec, "spmd.resize")
	}
	// Extending (shouldn't happen in practice): splat element 0.
	elem := b.CreateExtractElement(vec, llvm.ConstInt(b.ctx.Int32Type(), 0, false), "")
	return b.splatScalar(elem, targetVecType)
}

// createSPMDConst creates a splatted vector constant for an SPMD varying type.
func (c *compilerContext) createSPMDConst(expr *ssa.Const, spmdType *types.SPMDType, pos token.Pos) llvm.Value {
	vecType := c.getLLVMType(spmdType)
	if expr.Value == nil {
		return llvm.ConstNull(vecType)
	}

	// Create scalar constant using the element type.
	scalarConst := ssa.NewConst(expr.Value, spmdType.Elem())
	scalar := c.createConst(scalarConst, pos)

	// Splat scalar across all lanes.
	laneCount := vecType.VectorSize()
	elts := make([]llvm.Value, laneCount)
	for i := range elts {
		elts[i] = scalar
	}
	return llvm.ConstVector(elts, false)
}

// spmdConstIntOrSplat creates an integer constant, splatting it into a vector if typ is a vector type.
// This is used for creating constants that need to match the type of a potentially-vector operand,
// such as bounds check comparisons and shift overflow checks.
func spmdConstIntOrSplat(typ llvm.Type, val uint64, signed bool) llvm.Value {
	if typ.TypeKind() == llvm.VectorTypeKind {
		elemType := typ.ElementType()
		scalar := llvm.ConstInt(elemType, val, signed)
		laneCount := typ.VectorSize()
		elts := make([]llvm.Value, laneCount)
		for i := range elts {
			elts[i] = scalar
		}
		return llvm.ConstVector(elts, false)
	}
	return llvm.ConstInt(typ, val, signed)
}

// spmdLoopState holds per-function SPMD loop analysis results.
type spmdLoopState struct {
	activeLoops map[ssa.Value]*spmdActiveLoop // iter phi -> active loop
	bodyBlocks  map[int]*spmdActiveLoop       // body block index -> loop
	loopBlocks  map[int]*spmdActiveLoop       // loop block index -> loop
}

// spmdActiveLoop holds state for one SPMD loop during compilation.
type spmdActiveLoop struct {
	info       *SPMDLoopInfo
	iterPhi    *ssa.Phi   // the "rangeint.iter" phi in body block (nil for rangeindex)
	laneCount  int        // e.g. 4 for int32 on WASM SIMD128
	boundValue ssa.Value  // N in "range N"
	incrBinOp  *ssa.BinOp // the ADD in the loop block

	// Range-over-slice specific fields (isRangeIndex == true):
	isRangeIndex  bool      // true for rangeindex pattern (range-over-slice)
	bodyIterValue ssa.Value // value body uses as index: iterPhi (rangeint) or incrBinOp (rangeindex)
	initEdgeIndex int       // phi edge index for the entry predecessor (rangeindex only; -1 for rangeint)

	// Set during IR generation:
	laneIndices   llvm.Value // <iter, iter+1, ..., iter+laneCount-1>
	tailMask      llvm.Value // per-lane bounds check
	scalarIterVal llvm.Value // scalar LLVM value (before override to lane indices)
}

// analyzeSPMDLoops performs two-pass pre-analysis of SPMD loops before block compilation.
//
// Pass 1 detects range-over-int (rangeint) patterns via "rangeint.body" block comments.
// The iter phi is in the body block with comment "rangeint.iter".
//
// Pass 2 detects range-over-slice (rangeindex) patterns via "rangeindex.body" block comments.
// The iter phi is in the loop block (not body) with comment "rangeindex", and the body
// uses the increment BinOp (loopPhi + 1) as its index.
//
// This relies on golang.org/x/tools/go/ssa's block and phi comment conventions.
// If go/ssa internals change in x/tools, this detection may need updates.
func (b *builder) analyzeSPMDLoops() *spmdLoopState {
	if b.spmdInfo == nil {
		return nil
	}

	state := &spmdLoopState{
		activeLoops: make(map[ssa.Value]*spmdActiveLoop),
		bodyBlocks:  make(map[int]*spmdActiveLoop),
		loopBlocks:  make(map[int]*spmdActiveLoop),
	}

	// seenLoopInfo tracks which SPMDLoopInfo pointers have already been assigned
	// to a body block in either pass (rangeint or rangeindex). Since go/ssa
	// creates blocks via append in AST traversal order, the outer SPMD go for
	// body block always has a lower index than any nested for loop body blocks.
	// Only the first (lowest-index) body block per SPMD loop is the actual SPMD
	// body; subsequent matches are nested regular for-range loops whose body
	// block instructions fall positionally inside the parent SPMD loop body,
	// causing isInSPMDLoop to return the parent's info. A single unified map
	// handles cross-pattern nesting (rangeint outer + rangeindex inner, or
	// vice versa).
	seenLoopInfo := make(map[*SPMDLoopInfo]bool)

	// Iterate over ALL blocks (not just DomPreorder) to find rangeint patterns.
	for _, block := range b.fn.Blocks {
		// Look for rangeint.body blocks.
		if block.Comment != "rangeint.body" {
			continue
		}

		// Find the rangeint.iter phi instruction.
		var iterPhi *ssa.Phi
		for _, instr := range block.Instrs {
			if phi, ok := instr.(*ssa.Phi); ok && phi.Comment == "rangeint.iter" {
				iterPhi = phi
				break
			}
		}
		if iterPhi == nil {
			continue
		}
		// Check if this block is inside an SPMD loop using any instruction with a valid position.
		// The phi itself has NoPos (synthetic from SSA lift), so use body block instructions.
		var loopInfo *SPMDLoopInfo
		for _, instr := range block.Instrs {
			if pos := instr.(interface{ Pos() token.Pos }).Pos(); pos != token.NoPos {
				loopInfo = b.isInSPMDLoop(pos)
				if loopInfo != nil {
					break
				}
			}
		}
		if loopInfo == nil {
			continue
		}

		// Deduplicate: if this SPMDLoopInfo was already claimed (by either
		// pass), the current block is a nested regular loop — skip it.
		if seenLoopInfo[loopInfo] {
			continue
		}
		seenLoopInfo[loopInfo] = true

		// Find the loop block containing the increment (iterPhi + 1) and bounds check.
		// The loop block may be:
		//   - The body block itself (merged body+loop, simple cases)
		//   - A direct successor with comment "rangeint.loop" (split case, no body control flow)
		//   - A predecessor of body (split case with control flow: body → if.then/else → if.done → body)
		var loopBlock *ssa.BasicBlock
		var incrBinOp *ssa.BinOp
		var boundValue ssa.Value

		// Helper to search a candidate block for increment and bounds check.
		searchBlock := func(candidate *ssa.BasicBlock) bool {
			var incr *ssa.BinOp
			var bound ssa.Value
			for _, instr := range candidate.Instrs {
				if binOp, ok := instr.(*ssa.BinOp); ok {
					if binOp.Op == token.ADD && binOp.X == iterPhi {
						incr = binOp
					}
					if binOp.Op == token.LSS && incr != nil && binOp.X == incr {
						bound = binOp.Y
					}
				}
			}
			if incr != nil && bound != nil {
				loopBlock = candidate
				incrBinOp = incr
				boundValue = bound
				return true
			}
			return false
		}

		// 1. Check the body block itself (merged body+loop).
		if !searchBlock(block) {
			// 2. Check direct successors (simple split with "rangeint.loop").
			for _, succ := range block.Succs {
				if searchBlock(succ) {
					break
				}
			}
		}
		if loopBlock == nil {
			// 3. Check predecessors (body has control flow, loop-back comes through if.done).
			for _, pred := range block.Preds {
				if searchBlock(pred) {
					break
				}
			}
		}
		if loopBlock == nil {
			continue
		}

		// Compute lane count based on the iter phi's element type.
		// The phi has Go's int type, which on WASM is i32.
		elemType := b.getLLVMType(iterPhi.Type())
		laneCount := b.spmdLaneCount(elemType)

		// Create the active loop entry.
		loop := &spmdActiveLoop{
			info:          loopInfo,
			iterPhi:       iterPhi,
			laneCount:     laneCount,
			boundValue:    boundValue,
			incrBinOp:     incrBinOp,
			isRangeIndex:  false,
			bodyIterValue: iterPhi, // rangeint: body uses the phi directly
			initEdgeIndex: -1,      // unused for rangeint; -1 ensures no accidental match
		}

		// Populate the maps for quick lookup.
		state.activeLoops[iterPhi] = loop
		state.bodyBlocks[block.Index] = loop
		state.loopBlocks[loopBlock.Index] = loop
	}

	// Second pass: detect range-over-slice (rangeindex) patterns.
	//
	// The SSA pattern for "go for i, v := range slice" differs from range-over-int:
	//   rangeindex.loop: phi "rangeindex" = [entry:-1, body:incr]
	//                    incr = phi + 1
	//                    if incr < len(slice): goto body else done
	//   rangeindex.body: (no iter phi here; body uses incr from loop block)
	//                    ptr = IndexAddr(slice, incr)
	//                    ...
	//
	// Pass 2 uses the same seenLoopInfo map to handle cross-pattern nesting
	// (e.g., rangeint outer + rangeindex inner).

	for _, block := range b.fn.Blocks {
		if block.Comment != "rangeindex.body" {
			continue
		}

		// The loop block is a predecessor of the body block with comment "rangeindex.loop".
		var loopBlock *ssa.BasicBlock
		for _, pred := range block.Preds {
			if pred.Comment == "rangeindex.loop" {
				loopBlock = pred
				break
			}
		}
		if loopBlock == nil {
			continue
		}

		// Find the "rangeindex" phi in the loop block.
		var loopPhi *ssa.Phi
		for _, instr := range loopBlock.Instrs {
			if phi, ok := instr.(*ssa.Phi); ok && phi.Comment == "rangeindex" {
				loopPhi = phi
				break
			}
		}
		if loopPhi == nil {
			continue
		}

		// Check SPMD membership using body block instruction positions.
		// The phi is in the loop block, not the body, so we must check body instructions.
		var loopInfo *SPMDLoopInfo
		for _, instr := range block.Instrs {
			if pos := instr.Pos(); pos.IsValid() {
				if info := b.isInSPMDLoop(pos); info != nil {
					loopInfo = info
					break
				}
			}
		}
		if loopInfo == nil {
			// No SPMD loop found via body instruction positions. This can happen
			// if the body block contains only synthetic instructions with no source
			// position. In practice, go for range-over-slice bodies always contain
			// at least one user instruction (IndexAddr, etc.).
			continue
		}

		// Deduplicate: if this SPMDLoopInfo was already claimed (by either
		// pass), the current block is a nested regular loop — skip it.
		if seenLoopInfo[loopInfo] {
			continue
		}
		seenLoopInfo[loopInfo] = true

		// Find the increment BinOp (loopPhi + 1) and the bounds check (incr < len) in the loop block.
		var incrBinOp *ssa.BinOp
		var boundValue ssa.Value
		for _, instr := range loopBlock.Instrs {
			if binOp, ok := instr.(*ssa.BinOp); ok {
				if binOp.Op == token.ADD && binOp.X == loopPhi {
					incrBinOp = binOp
				}
				if binOp.Op == token.LSS && incrBinOp != nil && binOp.X == incrBinOp {
					boundValue = binOp.Y
				}
			}
		}
		if incrBinOp == nil || boundValue == nil {
			continue
		}

		// Determine initEdgeIndex: the phi edge whose predecessor is NOT the body block.
		// The rangeindex phi starts at -1 from the entry predecessor.
		initEdgeIndex := -1
		for i, pred := range loopBlock.Preds {
			if pred != block {
				initEdgeIndex = i
				break
			}
		}
		if initEdgeIndex < 0 {
			continue
		}

		// Compute lane count from the increment value's type.
		elemType := b.getLLVMType(incrBinOp.Type())
		laneCount := b.spmdLaneCount(elemType)

		loop := &spmdActiveLoop{
			info:          loopInfo,
			iterPhi:       nil, // rangeindex has no iter phi in body
			laneCount:     laneCount,
			boundValue:    boundValue,
			incrBinOp:     incrBinOp,
			isRangeIndex:  true,
			bodyIterValue: incrBinOp, // rangeindex: body uses the incr BinOp
			initEdgeIndex: initEdgeIndex,
		}

		// Register both loopPhi and incrBinOp as keys so IndexAddr contiguous
		// detection can find this loop via either value.
		state.activeLoops[loopPhi] = loop
		state.activeLoops[incrBinOp] = loop
		state.bodyBlocks[block.Index] = loop
		state.loopBlocks[loopBlock.Index] = loop
	}

	if len(state.activeLoops) == 0 {
		return nil
	}

	return state
}

// spmdLaneOffsetConst creates a constant vector <0, 1, 2, ..., laneCount-1>.
func (c *compilerContext) spmdLaneOffsetConst(laneCount int, elemType llvm.Type) llvm.Value {
	elts := make([]llvm.Value, laneCount)
	for i := 0; i < laneCount; i++ {
		elts[i] = llvm.ConstInt(elemType, uint64(i), false)
	}
	return llvm.ConstVector(elts, false)
}

// emitSPMDBodyPrologue emits the lane indices and tail mask after phi compilation.
// This transforms the scalar loop iterator into a vector of lane indices, and
// computes a per-lane bounds check mask.
//
// For rangeint loops (isRangeIndex == false), the scalar base value is the iter
// phi already compiled in the body block.
// For rangeindex loops (isRangeIndex == true), the scalar base value is the
// incrBinOp (loopPhi + 1) compiled in the loop block, which dominates the body
// block so its LLVM value is already present in b.locals via DomPreorder.
func (b *builder) emitSPMDBodyPrologue(loop *spmdActiveLoop) {
	// Obtain the scalar iteration value that the body block uses as its index.
	var scalarPhi llvm.Value
	if loop.isRangeIndex {
		// rangeindex: the incr BinOp (loopPhi+1) is the index seen by body instructions.
		// It was compiled in the loop block (which dominates body) so b.locals has it.
		scalarPhi = b.locals[loop.bodyIterValue]
	} else {
		// rangeint: the iter phi is in the body block itself.
		scalarPhi = b.locals[loop.iterPhi]
	}

	// Save the scalar value for later use in contiguous IndexAddr detection.
	loop.scalarIterVal = scalarPhi

	// Get element type from the scalar phi (e.g., i32 for int on WASM).
	elemType := scalarPhi.Type()

	// Create vector type for the lane count.
	vecType := llvm.VectorType(elemType, loop.laneCount)

	// Splat the scalar iterator across all lanes.
	iterVec := b.splatScalar(scalarPhi, vecType)

	// Create the offset constant <0, 1, 2, ..., laneCount-1>.
	offsetVec := b.spmdLaneOffsetConst(loop.laneCount, elemType)

	// Compute lane indices: <iter, iter+1, iter+2, ..., iter+laneCount-1>.
	laneIndices := b.CreateAdd(iterVec, offsetVec, "spmd.lane.idx")

	// Get the bound value and splat it.
	boundScalar := b.getValue(loop.boundValue, token.NoPos)
	boundVec := b.splatScalar(boundScalar, vecType)

	// Compute tail mask: laneIndices < bound (per-lane comparison).
	// Use IntSLT for signed comparison since Go's int is signed.
	// On WASM, sign-extend the <N x i1> result to <N x i32> so that downstream
	// uses (bitselect, mask stack AND/NOT) work without redundant conversions.
	tailMaskI1 := b.CreateICmp(llvm.IntSLT, laneIndices, boundVec, "spmd.tail.mask")
	tailMask := b.spmdWrapMask(tailMaskI1, loop.laneCount)

	// Store the results in the loop state.
	loop.laneIndices = laneIndices
	loop.tailMask = tailMask
}

// spmdPushMask pushes a new execution mask onto the stack.
func (b *builder) spmdPushMask(mask llvm.Value) {
	b.spmdMaskStack = append(b.spmdMaskStack, mask)
}

// spmdPopMask removes the top mask from the stack.
func (b *builder) spmdPopMask() {
	if len(b.spmdMaskStack) > 0 {
		b.spmdMaskStack = b.spmdMaskStack[:len(b.spmdMaskStack)-1]
	}
}

// spmdCurrentMask returns the top of the mask stack, or a nil Value if empty.
func (b *builder) spmdCurrentMask() llvm.Value {
	if len(b.spmdMaskStack) > 0 {
		return b.spmdMaskStack[len(b.spmdMaskStack)-1]
	}
	return llvm.Value{}
}

// spmdVaryingIf holds analysis results for a varying (vector) if/else construct.
type spmdVaryingIf struct {
	cond           llvm.Value // vector condition: <N x i1> on non-WASM, <N x i32> on WASM
	ifBlockIndex   int        // block with the If instruction
	thenEntryIndex int        // Succs[0] of if-block
	elseEntryIndex int        // Succs[1] of if-block (or merge for if-without-else)
	mergeIndex     int        // common successor (merge point)
	hasElse        bool       // true if then/else are distinct from merge
}

// spmdMergePhiOverride tracks a phi at a multi-predecessor merge block
// where a then/else pair of edges has been combined via select.
// This handles cases like varying if/else inside loop bodies where the merge
// is the loop header with 3+ predecessors (entry, then-exit, else-exit).
// The select is created during phi resolution, not during phi creation.
type spmdMergePhiOverride struct {
	thenEdgeIdx int           // phi edge index for the then-branch
	elseEdgeIdx int           // phi edge index for the else-branch
	info        *spmdVaryingIf // varying if info (for condition)
	llvmBlock   llvm.BasicBlock // LLVM block to use as predecessor for the selected value
}

// spmdMaskTransition describes how the execution mask changes at a block boundary.
type spmdMaskTransition struct {
	kind string     // "pushThen", "swapElse", "pop"
	cond llvm.Value // vector condition (for pushThen/swapElse)
}

// spmdContiguousInfo tracks an IndexAddr result that was detected as contiguous SPMD access.
type spmdContiguousInfo struct {
	scalarPtr llvm.Value      // scalar GEP result (base of contiguous access)
	loop      *spmdActiveLoop // owning loop (for lane count)
}

// isBlockInSPMDBody checks if a given SSA block is inside an SPMD loop body.
// This extends beyond just rangeint.body blocks to include if.then/if.else/if.done.
// It also returns a non-nil sentinel for SPMD function bodies (functions with varying
// parameters but no go-for loops), where all blocks are part of the SPMD region.
func (b *builder) isBlockInSPMDBody(block *ssa.BasicBlock) *SPMDLoopInfo {
	if b.spmdInfo == nil {
		return nil
	}

	// SPMD function body: all blocks belong to the SPMD region.
	// Return a non-nil sentinel; callers only check nil vs non-nil.
	if b.spmdFuncIsBody {
		return &SPMDLoopInfo{}
	}

	// Check if any instruction position in the block falls inside an SPMD loop body.
	for _, instr := range block.Instrs {
		if loopInfo := b.isInSPMDLoop(instr.Pos()); loopInfo != nil {
			return loopInfo
		}
	}

	// Fallback: check if any SPMD body block dominates this block.
	// This handles blocks like if.done whose instructions are synthetic (NoPos)
	// but are structurally inside the SPMD loop (dominated by the body block).
	// Note: this may also match post-loop blocks (e.g. rangeint.done) that are
	// dominated by body. This is safe because spmdValueOverride only maps
	// loop-local SSA values (iterPhi) which are not referenced after the loop.
	if b.spmdLoopState != nil {
		for bodyIdx, loop := range b.spmdLoopState.bodyBlocks {
			bodyBlock := b.fn.Blocks[bodyIdx]
			if bodyBlock.Dominates(block) {
				return loop.info
			}
		}
	}

	return nil
}

// preDetectVaryingIfs scans all blocks in the function for If instructions
// with varying conditions and calls spmdAnalyzeVaryingIf to populate the
// spmdMergeSelects map. This must be done before compiling blocks so that
// phis at merge blocks can be converted to selects.
func (b *builder) preDetectVaryingIfs() {
	for _, block := range b.fn.Blocks {
		// Check if block is in SPMD context using the same logic as isBlockInSPMDBody.
		if b.isBlockInSPMDBody(block) == nil {
			continue
		}

		// Check if last instruction is an If.
		if len(block.Instrs) == 0 {
			continue
		}
		ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If)
		if !ok {
			continue
		}

		// Check if condition is varying (SPMDType).
		if _, ok := ifInstr.Cond.Type().(*types.SPMDType); ok {
			b.spmdAnalyzeVaryingIf(block)
		}
	}
}

// spmdAnalyzeVaryingIf analyzes a varying if/else construct at ifBlock and populates
// spmdVaryingIfs, spmdThenExitRedirects, and spmdMergeSelects maps. This is the
// analysis-only version called during pre-detection. The condition LLVM value will
// be filled in later when the If instruction is actually compiled.
func (b *builder) spmdAnalyzeVaryingIf(ifBlock *ssa.BasicBlock) {
	thenEntry := ifBlock.Succs[0]
	elseEntry := ifBlock.Succs[1]

	// Find the merge block (common successor).
	merge := b.spmdFindMerge(ifBlock, thenEntry, elseEntry)
	if merge == nil {
		// No merge found (possibly unreachable code or exit branches).
		return
	}

	// Determine if this is if-with-else or if-without-else.
	// If-without-else: elseEntry IS the merge block.
	hasElse := (elseEntry != merge)

	info := &spmdVaryingIf{
		cond:           llvm.Value{}, // Will be filled in during compilation
		ifBlockIndex:   ifBlock.Index,
		thenEntryIndex: thenEntry.Index,
		elseEntryIndex: elseEntry.Index,
		mergeIndex:     merge.Index,
		hasElse:        hasElse,
	}

	// Register the varying if.
	b.spmdVaryingIfs[ifBlock.Index] = info
	b.spmdMergeSelects[merge.Index] = info

	if hasElse {
		// For if-with-else: record that then-exit blocks need to be redirected.
		// We'll populate the LLVM blocks later during compilation.
		// For now, just find the then-exit blocks.
		_ = b.spmdFindThenExits(thenEntry, merge)
	}

	// Record mask transitions for block-level mask stack management.
	if b.spmdMaskTransitions != nil {
		// Condition will be filled in during compilation.
		b.spmdMaskTransitions[thenEntry.Index] = &spmdMaskTransition{kind: "pushThen", cond: llvm.Value{}}
		if hasElse {
			b.spmdMaskTransitions[elseEntry.Index] = &spmdMaskTransition{kind: "swapElse", cond: llvm.Value{}}
		}
		// When merge is a loop header, don't register "pop" at merge (it would
		// fire on every loop iteration, including from the entry edge). Instead,
		// mark the varying-if info so the Jump handler pops the mask when jumping
		// from the else-exit to the loop header.
		isLoopHeader := false
		if b.spmdLoopState != nil {
			_, isLoopHeader = b.spmdLoopState.loopBlocks[merge.Index]
		}
		if !isLoopHeader {
			b.spmdMaskTransitions[merge.Index] = &spmdMaskTransition{kind: "pop"}
		}
		// For loop-header merges, the pop is handled by spmdShouldPopBeforeJump.
	}
}

// spmdDetectVaryingIf detects and analyzes a varying if/else construct at ifBlock.
// The condition is a vector (<N x i1>), so both branches must execute for their
// respective lanes. Updates the pre-detected info with the actual LLVM condition value
// and creates the then-exit redirects.
func (b *builder) spmdDetectVaryingIf(ifBlock *ssa.BasicBlock, cond llvm.Value) {
	// Look up the pre-detected info.
	info, ok := b.spmdVaryingIfs[ifBlock.Index]
	if !ok {
		// Not pre-detected (shouldn't happen, but handle gracefully).
		b.spmdAnalyzeVaryingIf(ifBlock)
		info = b.spmdVaryingIfs[ifBlock.Index]
		if info == nil {
			return
		}
	}

	// Fill in the LLVM condition.
	info.cond = cond

	// Fill in LLVM blocks for then-exit redirects.
	if info.hasElse {
		thenEntry := b.fn.Blocks[info.thenEntryIndex]
		elseEntry := b.fn.Blocks[info.elseEntryIndex]
		merge := b.fn.Blocks[info.mergeIndex]
		thenExits := b.spmdFindThenExits(thenEntry, merge)
		elseLLVMBlock := b.blockInfo[elseEntry.Index].entry
		for _, exitBlock := range thenExits {
			b.spmdThenExitRedirects[exitBlock.Index] = elseLLVMBlock
		}
	}

	// Update mask transitions with actual condition.
	if b.spmdMaskTransitions != nil {
		if trans, ok := b.spmdMaskTransitions[info.thenEntryIndex]; ok {
			trans.cond = cond
		}
		if info.hasElse {
			if trans, ok := b.spmdMaskTransitions[info.elseEntryIndex]; ok {
				trans.cond = cond
			}
		}
	}
}

// spmdFindMerge finds the merge block (common successor) of then and else branches.
// Uses ifBlock as a barrier to prevent DFS from following loop back-edges through the if-block.
// This is critical for varying if/else inside loop bodies where both branches jump back to
// the loop header: without the barrier, we'd incorrectly detect else-entry as the merge.
func (b *builder) spmdFindMerge(ifBlock, thenEntry, elseEntry *ssa.BasicBlock) *ssa.BasicBlock {
	// Handle if-without-else: elseEntry itself is the merge.
	if b.spmdIsReachableFrom(thenEntry, elseEntry, ifBlock) {
		return elseEntry
	}

	// Build reachable set from thenEntry (excluding elseEntry subtree and ifBlock barrier).
	visited := make(map[int]bool)
	var walkThen func(*ssa.BasicBlock)
	walkThen = func(block *ssa.BasicBlock) {
		if visited[block.Index] || block == elseEntry || block == ifBlock {
			return
		}
		visited[block.Index] = true
		for _, succ := range block.Succs {
			walkThen(succ)
		}
	}
	walkThen(thenEntry)

	// Find first block reachable from elseEntry that's also in thenEntry's reachable set.
	// Use ifBlock as barrier to prevent walking back through it.
	var findIntersection func(*ssa.BasicBlock) *ssa.BasicBlock
	elseVisited := make(map[int]bool)
	findIntersection = func(block *ssa.BasicBlock) *ssa.BasicBlock {
		if elseVisited[block.Index] || block == ifBlock {
			return nil
		}
		elseVisited[block.Index] = true

		if visited[block.Index] {
			return block
		}

		for _, succ := range block.Succs {
			if result := findIntersection(succ); result != nil {
				return result
			}
		}
		return nil
	}

	return findIntersection(elseEntry)
}

// spmdFindThenExits finds all blocks in the then-branch that jump to the merge block.
// These are the "then-exit" blocks that need to be redirected to else-entry.
func (b *builder) spmdFindThenExits(thenEntry, merge *ssa.BasicBlock) []*ssa.BasicBlock {
	var exits []*ssa.BasicBlock
	visited := make(map[int]bool)

	var walk func(*ssa.BasicBlock)
	walk = func(block *ssa.BasicBlock) {
		if visited[block.Index] || block == merge {
			return
		}
		visited[block.Index] = true

		// Check if this block's last instruction is a Jump to merge.
		if len(block.Instrs) > 0 {
			if _, ok := block.Instrs[len(block.Instrs)-1].(*ssa.Jump); ok {
				if len(block.Succs) == 1 && block.Succs[0] == merge {
					exits = append(exits, block)
					return // Don't recurse past the exit.
				}
			}
		}

		// Recurse to successors.
		for _, succ := range block.Succs {
			walk(succ)
		}
	}

	walk(thenEntry)
	return exits
}

// spmdShouldPopBeforeJump checks if a Jump from this block needs to pop the mask
// stack before branching. This handles the case where a varying if/else inside a
// loop has the loop header as its merge: the "pop" can't be at the merge (it would
// fire on every loop entry), so instead the else-exit block pops before jumping.
func (b *builder) spmdShouldPopBeforeJump(block *ssa.BasicBlock) bool {
	if b.spmdMergeSelects == nil || b.spmdLoopState == nil {
		return false
	}
	// Check if this block's Jump target is a merge block that's also a loop header.
	if len(block.Succs) != 1 {
		return false
	}
	target := block.Succs[0]
	if _, isMerge := b.spmdMergeSelects[target.Index]; !isMerge {
		return false
	}
	if _, isLoop := b.spmdLoopState.loopBlocks[target.Index]; !isLoop {
		return false
	}
	// This block is an else-exit jumping to a loop-header merge. Pop before jump.
	return true
}

// spmdShouldRedirectJump checks if a Jump instruction at the given block should
// be redirected to the else-entry (for then-exit blocks in if-with-else).
// Returns (elseLLVMBlock, true) if redirect is needed, (zero, false) otherwise.
func (b *builder) spmdShouldRedirectJump(block *ssa.BasicBlock) (llvm.BasicBlock, bool) {
	if b.spmdThenExitRedirects == nil {
		return llvm.BasicBlock{}, false
	}
	elseLLVMBlock, ok := b.spmdThenExitRedirects[block.Index]
	return elseLLVMBlock, ok
}

// spmdCreateMergeSelect converts a phi at a merge block into a select instruction
// when the phi results from a varying if/else. Returns (value, true) if a select
// was created, (zero, false) if this phi is not a varying merge phi.
func (b *builder) spmdCreateMergeSelect(phi *ssa.Phi) (llvm.Value, bool) {
	if b.spmdMergeSelects == nil {
		return llvm.Value{}, false
	}

	info, ok := b.spmdMergeSelects[phi.Block().Index]
	if !ok {
		return llvm.Value{}, false
	}

	// Phi must have exactly 2 edges for standard merge.
	if len(phi.Edges) != 2 {
		// Multi-predecessor merge (e.g., loop header with if/else back-edges).
		// Find the then and else edge indices, merge them with select,
		// and let the remaining edges go through normal phi resolution.
		if len(phi.Edges) > 2 && info.hasElse {
			return b.spmdCreateMultiPredMergeSelect(phi, info)
		}
		return llvm.Value{}, false
	}

	block := phi.Block()
	preds := block.Preds

	// Determine which edge is "then" and which is "else".
	// For if-without-else: predecessor matching ifBlockIndex is "else" (default), other is "then".
	// For if-with-else: determine based on reachability from then-entry vs else-entry.
	var thenValue, elseValue llvm.Value
	if !info.hasElse {
		// If-without-else: elseEntry IS merge, so ifBlock predecessor is "else" edge.
		for i := 0; i < len(preds); i++ {
			if preds[i].Index == info.ifBlockIndex {
				// This edge comes from ifBlock (the else/default path).
				elseValue = b.getValue(phi.Edges[i], token.NoPos)
			} else {
				// This edge comes from then-branch.
				thenValue = b.getValue(phi.Edges[i], token.NoPos)
			}
		}
	} else {
		// If-with-else: determine based on reachability.
		thenEntry := b.fn.Blocks[info.thenEntryIndex]
		for i := 0; i < len(preds); i++ {
			if b.spmdIsReachableFrom(thenEntry, preds[i], block) {
				// This edge comes from then-branch.
				thenValue = b.getValue(phi.Edges[i], token.NoPos)
			} else {
				// This edge comes from else-branch.
				elseValue = b.getValue(phi.Edges[i], token.NoPos)
			}
		}
	}

	// Ensure both values are non-nil.
	if thenValue.IsNil() || elseValue.IsNil() {
		return llvm.Value{}, false
	}

	// Handle type mismatches via broadcast.
	thenValue, elseValue = b.spmdBroadcastMatch(thenValue, elseValue)

	// Determine select type based on operand types.
	thenIsVec := thenValue.Type().TypeKind() == llvm.VectorTypeKind
	elseIsVec := elseValue.Type().TypeKind() == llvm.VectorTypeKind

	if thenIsVec || elseIsVec {
		// At least one operand is a vector → vector masked select.
		// On WASM the mask is <N x i32> so use spmdMaskSelect instead of CreateSelect
		// (which requires an <N x i1> condition).
		return b.spmdMaskSelect(info.cond, thenValue, elseValue), true
	}

	// Both are scalars → reduce condition to scalar boolean and use scalar select.
	// Scalar phis at a varying merge represent uniform values. We use any-true
	// reduction: if any lane took the then-branch, use the then-value. This is
	// safe because the SPMD type checker forbids varying-dependent mutation of
	// uniform variables, so both edges carry the same value in practice.
	scalarCond := b.spmdVectorAnyTrue(info.cond)
	return b.CreateSelect(scalarCond, thenValue, elseValue, ""), true
}

// spmdCreateMultiPredMergeSelect handles phis at merge blocks with >2 predecessors,
// such as loop headers where an if/else merges back along with the entry edge.
// It creates an LLVM phi and records override info so that phi resolution can
// create the select and add edges properly. The select creation is deferred until
// phi resolution to avoid circular dependencies when edge values reference the phi itself.
func (b *builder) spmdCreateMultiPredMergeSelect(phi *ssa.Phi, info *spmdVaryingIf) (llvm.Value, bool) {
	block := phi.Block()
	preds := block.Preds

	// Find then and else edge indices by checking reachability.
	thenEntry := b.fn.Blocks[info.thenEntryIndex]
	elseEntry := b.fn.Blocks[info.elseEntryIndex]
	thenIdx := -1
	elseIdx := -1

	for i, pred := range preds {
		if pred.Index == info.thenEntryIndex || b.spmdIsReachableFrom(thenEntry, pred, block) {
			if thenIdx < 0 {
				thenIdx = i
			}
		}
		if pred.Index == info.elseEntryIndex || b.spmdIsReachableFrom(elseEntry, pred, block) {
			if elseIdx < 0 {
				elseIdx = i
			}
		}
	}

	if thenIdx < 0 || elseIdx < 0 {
		return llvm.Value{}, false
	}

	// Create LLVM phi for normal resolution of non-then/else edges.
	phiType := b.getLLVMType(phi.Type())
	llvmPhi := b.CreatePHI(phiType, "")
	b.phis = append(b.phis, phiNode{phi, llvmPhi})

	// Record override: during phi resolution, skip then-edge and replace else-edge
	// with a select(cond, thenVal, elseVal). Use the else-block's LLVM exit as the
	// predecessor because after linearization, if.then jumps to if.else.
	b.spmdMergePhiOverrides[phi] = spmdMergePhiOverride{
		thenEdgeIdx: thenIdx,
		elseEdgeIdx: elseIdx,
		info:        info,
		llvmBlock:   b.blockInfo[preds[elseIdx].Index].exit,
	}

	return llvmPhi, true
}

// spmdVectorAnyTrue reduces an SPMD mask vector to a scalar i1.
// Returns true if any lane is active.
// On WASM targets, uses the native v128.any_true instruction via the
// @llvm.wasm.anytrue LLVM intrinsic (single WASM instruction, no bitcast).
// On other targets the mask is <N x i1>, so we bitcast to iN and compare != 0.
func (b *builder) spmdVectorAnyTrue(mask llvm.Value) llvm.Value {
	if b.spmdIsWASM() {
		// Use native WASM v128.any_true instruction.
		i32Result := b.spmdWasmAnyTrue(mask)
		zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
		return b.CreateICmp(llvm.IntNE, i32Result, zero, "")
	}
	// Non-WASM: <N x i1> → iN bitcast, compare != 0.
	vecSize := mask.Type().VectorSize()
	intType := b.ctx.IntType(vecSize)
	intVal := b.CreateBitCast(mask, intType, "")
	zero := llvm.ConstNull(intType)
	return b.CreateICmp(llvm.IntNE, intVal, zero, "")
}

// spmdVectorAllTrue reduces an SPMD mask vector to a scalar i1.
// Returns true only when every lane is active (stronger than spmdVectorAnyTrue).
// On WASM targets, uses the native v128.alltrue instruction via the
// @llvm.wasm.alltrue LLVM intrinsic (single WASM instruction, no bitcast).
// On other targets the mask is <N x i1>.
func (b *builder) spmdVectorAllTrue(mask llvm.Value) llvm.Value {
	if b.spmdIsWASM() {
		// Use native WASM i32x4.all_true instruction.
		i32Result := b.spmdWasmAllTrue(mask)
		zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
		return b.CreateICmp(llvm.IntNE, i32Result, zero, "")
	}
	// Non-WASM: <N x i1> → iN bitcast, compare == all-ones.
	vecSize := mask.Type().VectorSize()
	intType := b.ctx.IntType(vecSize)
	intVal := b.CreateBitCast(mask, intType, "")
	allOnes := llvm.ConstAllOnes(intType)
	return b.CreateICmp(llvm.IntEQ, intVal, allOnes, "")
}

// spmdWasmAnyTrue calls @llvm.wasm.anytrue on a vector.
// Returns i32 (0 or 1). Only valid for WASM targets.
func (b *builder) spmdWasmAnyTrue(mask llvm.Value) llvm.Value {
	vecType := mask.Type()
	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.wasm.anytrue." + suffix
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(i32Type, []llvm.Type{vecType}, false)
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{mask}, "")
}

// spmdWasmAllTrue calls @llvm.wasm.alltrue on a vector.
// Returns i32 (0 or 1). Only valid for WASM targets.
func (b *builder) spmdWasmAllTrue(mask llvm.Value) llvm.Value {
	vecType := mask.Type()
	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.wasm.alltrue." + suffix
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(i32Type, []llvm.Type{vecType}, false)
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{mask}, "")
}

// spmdIsReachableFrom checks if target is reachable from start without going
// through barrier. Returns true if a path exists from start to target.
func (b *builder) spmdIsReachableFrom(start, target, barrier *ssa.BasicBlock) bool {
	if start == target {
		return true
	}
	if start == barrier {
		return false
	}

	visited := make(map[int]bool)
	var dfs func(*ssa.BasicBlock) bool
	dfs = func(block *ssa.BasicBlock) bool {
		if block == target {
			return true
		}
		if block == barrier || visited[block.Index] {
			return false
		}
		visited[block.Index] = true

		for _, succ := range block.Succs {
			if dfs(succ) {
				return true
			}
		}
		return false
	}

	return dfs(start)
}

// spmdIsWASM returns true when the compiler target is a WebAssembly target.
// On WASM, SIMD comparisons natively produce <N x i32> (all-ones/all-zeros),
// so we use i32 as the internal mask element type to avoid the redundant
// shl/shr_s sign-extension that LLVM inserts when converting <N x i1> to i32
// for v128.bitselect.
func (c *compilerContext) spmdIsWASM() bool {
	return strings.HasPrefix(c.Triple, "wasm")
}

// spmdMaskElemType returns the LLVM element type used for SPMD mask vectors.
// On WASM targets this is i32 (all-ones = active, all-zeros = inactive).
// On other targets this is i1 (the native LLVM boolean vector element type).
func (c *compilerContext) spmdMaskElemType() llvm.Type {
	if c.spmdIsWASM() {
		return c.ctx.Int32Type()
	}
	return c.ctx.Int1Type()
}

// spmdMaskType returns the LLVM mask type for an SPMD function's implicit first parameter.
// Returns zero-value llvm.Type{} if the function has no varying parameters.
func (c *compilerContext) spmdMaskType(fn *ssa.Function) llvm.Type {
	return c.spmdMaskTypeFromSig(fn.Signature)
}

// spmdMaskTypeFromSig returns the LLVM mask type for an SPMD signature's implicit mask parameter.
// On WASM the mask is <N x i32>; on other targets it is <N x i1>.
// N is determined by the first varying parameter's element type.
// Returns zero-value llvm.Type{} if the signature has no varying parameters.
func (c *compilerContext) spmdMaskTypeFromSig(sig *types.Signature) llvm.Type {
	if sig == nil {
		return llvm.Type{}
	}
	params := sig.Params()
	for i := 0; i < params.Len(); i++ {
		param := params.At(i)
		if spmdType, ok := param.Type().(*types.SPMDType); ok && spmdType.IsVarying() {
			// Found a varying parameter. Compute lane count respecting constraints.
			elemType := c.getLLVMType(spmdType.Elem())
			laneCount := c.spmdEffectiveLaneCount(spmdType, elemType)
			return llvm.VectorType(c.spmdMaskElemType(), laneCount)
		}
	}
	return llvm.Type{} // No varying parameters
}

// spmdWrapMask sign-extends an <N x i1> comparison result to <N x i32> on WASM.
// On non-WASM targets this is a no-op. LLVM's WASM backend folds sext(cmp)
// into a single WASM comparison instruction, so there is no runtime cost.
func (b *builder) spmdWrapMask(cmp llvm.Value, laneCount int) llvm.Value {
	if !b.spmdIsWASM() {
		return cmp
	}
	maskType := llvm.VectorType(b.ctx.Int32Type(), laneCount)
	return b.CreateSExt(cmp, maskType, "")
}

// spmdUnwrapMaskForIntrinsic truncates <N x i32> back to <N x i1> for LLVM masked
// memory intrinsics (masked.load, masked.store, masked.gather, masked.scatter).
// These intrinsics require an <N x i1> mask regardless of the target.
// On non-WASM targets the mask is already <N x i1> and this is a no-op.
func (b *builder) spmdUnwrapMaskForIntrinsic(mask llvm.Value, laneCount int) llvm.Value {
	if !b.spmdIsWASM() {
		return mask
	}
	i1MaskType := llvm.VectorType(b.ctx.Int1Type(), laneCount)
	return b.CreateTrunc(mask, i1MaskType, "")
}

// spmdMaskSelect emits a masked select for SPMD value merging.
// On non-WASM targets this calls CreateSelect directly (mask is <N x i1>).
// On WASM targets the mask is <N x i32> (all-ones / all-zeros), so we use
// bitwise AND/OR: result = (mask & trueVal) | (~mask & falseVal).
// Float vector types are bitcast to the matching integer type for the bitwise ops.
//
// The bitwise path is only valid when mask and data have the same total bit
// width (e.g., <4 x i32> mask with <4 x i32> or <4 x f32> data). For types
// with a different total bit width (e.g., <2 x i32> mask with <2 x i64> data),
// we fall back to truncating the mask to <N x i1> and using LLVM's native
// CreateSelect, which handles arbitrary widths correctly.
func (b *builder) spmdMaskSelect(mask, trueVal, falseVal llvm.Value) llvm.Value {
	if !b.spmdIsWASM() {
		return b.CreateSelect(mask, trueVal, falseVal, "")
	}

	valType := trueVal.Type()
	maskType := mask.Type() // <N x i32>

	// Efficient bitwise select only when mask and data have equal total bit width
	// (e.g., <4 x i32> mask with <4 x i32> or <4 x f32> data).
	// For mismatched widths (e.g., <2 x i32> mask with <2 x i64> data),
	// fall back to trunc+CreateSelect which LLVM handles correctly.
	if b.targetData.TypeAllocSize(maskType) == b.targetData.TypeAllocSize(valType) {
		var aBits, bBits llvm.Value
		needBitcast := valType != maskType
		if needBitcast {
			aBits = b.CreateBitCast(trueVal, maskType, "")
			bBits = b.CreateBitCast(falseVal, maskType, "")
		} else {
			aBits = trueVal
			bBits = falseVal
		}
		notMask := b.CreateNot(mask, "")
		and1 := b.CreateAnd(mask, aBits, "")
		and2 := b.CreateAnd(notMask, bBits, "")
		result := b.CreateOr(and1, and2, "")
		if needBitcast {
			result = b.CreateBitCast(result, valType, "")
		}
		return result
	}

	// Fallback: truncate i32 mask to i1 and use LLVM's native select.
	laneCount := maskType.VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	return b.CreateSelect(i1Mask, trueVal, falseVal, "")
}

// spmdNormalizeBoolVecToI1 converts an SPMD bool vector to <N x i1> regardless
// of whether the input is already <N x i1> or <N x i32> (WASM format).
// Used by reduce.All/Count/FindFirstSet/Mask which need a compact bit representation.
func (b *builder) spmdNormalizeBoolVecToI1(vec llvm.Value) llvm.Value {
	if !b.spmdIsWASM() {
		return vec // already <N x i1>
	}
	// WASM: vec is <N x i32> (all-ones/all-zeros). Truncate to <N x i1>.
	laneCount := vec.Type().VectorSize()
	i1Type := llvm.VectorType(b.ctx.Int1Type(), laneCount)
	return b.CreateTrunc(vec, i1Type, "")
}

// spmdCallMask returns the mask value to pass when calling an SPMD function.
// The mask is determined by the current execution context, narrowest first:
// - If inside a varying-if block: use the current narrowed mask (from mask stack)
// - If inside an SPMD loop: use the loop's tail mask
// - If inside an SPMD function: use the entry mask
// - Otherwise: all lanes active (all-ones mask)
func (b *builder) spmdCallMask(fn *ssa.Function) llvm.Value {
	maskType := b.spmdMaskType(fn)
	if maskType == (llvm.Type{}) {
		// Not an SPMD function, no mask needed.
		return llvm.Value{}
	}

	// Use current narrowed mask if inside varying-if context.
	if mask := b.spmdCurrentMask(); !mask.IsNil() {
		return mask
	}

	// Check if we're inside an SPMD loop.
	if b.spmdLoopState != nil {
		for _, loop := range b.spmdLoopState.activeLoops {
			if !loop.tailMask.IsNil() {
				return loop.tailMask
			}
		}
	}

	// Check if we're inside an SPMD function.
	if !b.spmdEntryMask.IsNil() {
		return b.spmdEntryMask
	}

	// Fallback: all lanes active.
	return llvm.ConstAllOnes(maskType)
}

// spmdVectorTypeSuffix returns the LLVM intrinsic name suffix for a vector type.
// e.g., <4 x i32> → "v4i32", <4 x float> → "v4f32", <2 x double> → "v2f64"
func spmdVectorTypeSuffix(vecType llvm.Type) string {
	n := vecType.VectorSize()
	elemType := vecType.ElementType()
	var suffix string
	switch elemType.TypeKind() {
	case llvm.IntegerTypeKind:
		suffix = "i" + strconv.Itoa(elemType.IntTypeWidth())
	case llvm.FloatTypeKind:
		suffix = "f32"
	case llvm.DoubleTypeKind:
		suffix = "f64"
	default:
		suffix = "i32" // fallback
	}
	return "v" + strconv.Itoa(n) + suffix
}

// spmdCallVectorReduce declares and calls an LLVM integer vector reduction intrinsic.
// op is the operation name: "add", "mul", "and", "or", "xor", "smax", "smin", "umax", "umin".
// Returns the scalar result.
func (b *builder) spmdCallVectorReduce(op string, vec llvm.Value) llvm.Value {
	vecType := vec.Type()
	elemType := vecType.ElementType()
	intrinsicName := "llvm.vector.reduce." + op + "." + spmdVectorTypeSuffix(vecType)
	llvmFn := b.mod.NamedFunction(intrinsicName)
	fnType := llvm.FunctionType(elemType, []llvm.Type{vecType}, false)
	if llvmFn.IsNil() {
		llvmFn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, llvmFn, []llvm.Value{vec}, "")
}

// spmdCallVectorReduceFloat declares and calls an LLVM float vector reduction intrinsic.
// op is "fadd" or "fmul". startVal is the identity element (0.0 for add, 1.0 for mul).
// Returns the scalar result.
func (b *builder) spmdCallVectorReduceFloat(op string, startVal, vec llvm.Value) llvm.Value {
	vecType := vec.Type()
	elemType := vecType.ElementType()
	intrinsicName := "llvm.vector.reduce." + op + "." + spmdVectorTypeSuffix(vecType)
	llvmFn := b.mod.NamedFunction(intrinsicName)
	fnType := llvm.FunctionType(elemType, []llvm.Type{elemType, vecType}, false)
	if llvmFn.IsNil() {
		llvmFn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, llvmFn, []llvm.Value{startVal, vec}, "")
}

// spmdIsSignedInt returns true if the Go type is a signed integer type.
func spmdIsSignedInt(t types.Type) bool {
	if basic, ok := t.Underlying().(*types.Basic); ok {
		return basic.Info()&types.IsInteger != 0 && basic.Info()&types.IsUnsigned == 0
	}
	return false
}

// spmdIsFloat returns true if the Go type is a float type.
func spmdIsFloat(t types.Type) bool {
	if basic, ok := t.Underlying().(*types.Basic); ok {
		return basic.Info()&types.IsFloat != 0
	}
	return false
}

// createLanesBuiltin handles interception of lanes.* function calls.
// Returns the LLVM value result and nil error on success.
func (b *builder) createLanesBuiltin(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	switch {
	case name == "lanes.Index":
		// lanes.Index() returns <0, 1, 2, ..., N-1> as Varying[int]
		elemType := b.getLLVMType(instr.Signature().Results().At(0).Type().(*types.SPMDType).Elem())
		laneCount := b.spmdLaneCount(elemType)
		return b.spmdLaneOffsetConst(laneCount, elemType), nil

	case strings.HasPrefix(name, "lanes.Count["):
		// lanes.Count[T](v) returns scalar int (lane count, compile-time constant)
		// Get element type from the argument's SPMDType
		argType := instr.Args[0].Type()
		var elemType llvm.Type
		if spmdType, ok := argType.(*types.SPMDType); ok {
			elemType = b.getLLVMType(spmdType.Elem())
		} else {
			elemType = b.getLLVMType(argType)
		}
		laneCount := b.spmdLaneCount(elemType)
		return llvm.ConstInt(b.intType, uint64(laneCount), false), nil

	case strings.HasPrefix(name, "lanes.Broadcast["):
		// lanes.Broadcast[T](value, lane) — extract element at lane index, splat to all lanes
		vec := b.getValue(instr.Args[0], getPos(instr))
		lane := b.getValue(instr.Args[1], getPos(instr))
		elem := b.CreateExtractElement(vec, lane, "broadcast.elem")
		return b.splatScalar(elem, vec.Type()), nil

	case strings.HasPrefix(name, "lanes.ShiftLeft["):
		// lanes.ShiftLeft[T](value, shift) — per-lane left shift
		value := b.getValue(instr.Args[0], getPos(instr))
		shift := b.getValue(instr.Args[1], getPos(instr))
		return b.CreateShl(value, shift, ""), nil

	case strings.HasPrefix(name, "lanes.ShiftRight["):
		// lanes.ShiftRight[T](value, shift) — per-lane right shift
		// Signed types use arithmetic shift right, unsigned use logical shift right
		value := b.getValue(instr.Args[0], getPos(instr))
		shift := b.getValue(instr.Args[1], getPos(instr))
		// Get element type from SPMDType to determine signedness
		argType := instr.Args[0].Type()
		if spmdType, ok := argType.(*types.SPMDType); ok && spmdIsSignedInt(spmdType.Elem()) {
			return b.CreateAShr(value, shift, ""), nil
		}
		return b.CreateLShr(value, shift, ""), nil

	case strings.HasPrefix(name, "lanes.From["):
		// lanes.From[T](data []T) — load N contiguous elements from slice as vector
		// Extract pointer from the slice value (element 0 is the data pointer)
		sliceVal := b.getValue(instr.Args[0], getPos(instr))
		ptr := b.CreateExtractValue(sliceVal, 0, "slice.ptr")
		// Determine vector type from result SPMDType
		resultType := instr.Signature().Results().At(0).Type()
		vecType := b.getLLVMType(resultType)
		// Load as vector
		return b.CreateLoad(vecType, ptr, "lanes.from"), nil

	case strings.HasPrefix(name, "lanes.FromConstrained["):
		return b.createFromConstrained(instr, name)

	case strings.HasPrefix(name, "lanes.ToConstrained["):
		return b.createToConstrained(instr, name)

	default:
		return llvm.Value{}, b.makeError(getPos(instr), "unsupported lanes builtin: "+name)
	}
}

// createReduceBuiltin handles interception of reduce.* function calls.
// Returns the LLVM value result and nil error on success.
func (b *builder) createReduceBuiltin(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	switch {
	case strings.HasPrefix(name, "reduce.Add["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		if spmdType, ok := argType.(*types.SPMDType); ok && spmdIsFloat(spmdType.Elem()) {
			// Float: ordered fadd reduction with start = 0.0
			elemType := vec.Type().ElementType()
			startVal := llvm.ConstFloat(elemType, 0.0)
			return b.spmdCallVectorReduceFloat("fadd", startVal, vec), nil
		}
		return b.spmdCallVectorReduce("add", vec), nil

	case strings.HasPrefix(name, "reduce.Mul["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		if spmdType, ok := argType.(*types.SPMDType); ok && spmdIsFloat(spmdType.Elem()) {
			// Float: ordered fmul reduction with start = 1.0
			elemType := vec.Type().ElementType()
			startVal := llvm.ConstFloat(elemType, 1.0)
			return b.spmdCallVectorReduceFloat("fmul", startVal, vec), nil
		}
		return b.spmdCallVectorReduce("mul", vec), nil

	case name == "reduce.All":
		// reduce.All(v Varying[bool]) bool — true if all lanes are true.
		// Normalize to <N x i1> first (on WASM bool vectors are <N x i32>),
		// then bitcast to iN and compare == all-ones.
		vec := b.getValue(instr.Args[0], getPos(instr))
		i1Vec := b.spmdNormalizeBoolVecToI1(vec)
		vecSize := i1Vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(i1Vec, intType, "")
		allOnes := llvm.ConstAllOnes(intType)
		return b.CreateICmp(llvm.IntEQ, intVal, allOnes, ""), nil

	case name == "reduce.Any":
		// reduce.Any(v Varying[bool]) bool — true if any lane is true.
		// spmdVectorAnyTrue handles both <N x i1> and <N x i32> (WASM) formats.
		vec := b.getValue(instr.Args[0], getPos(instr))
		return b.spmdVectorAnyTrue(vec), nil

	case strings.HasPrefix(name, "reduce.Max["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		if spmdType, ok := argType.(*types.SPMDType); ok {
			if spmdIsFloat(spmdType.Elem()) {
				return b.spmdCallVectorReduce("fmax", vec), nil
			}
			if spmdIsSignedInt(spmdType.Elem()) {
				return b.spmdCallVectorReduce("smax", vec), nil
			}
		}
		return b.spmdCallVectorReduce("umax", vec), nil

	case strings.HasPrefix(name, "reduce.Min["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		if spmdType, ok := argType.(*types.SPMDType); ok {
			if spmdIsFloat(spmdType.Elem()) {
				return b.spmdCallVectorReduce("fmin", vec), nil
			}
			if spmdIsSignedInt(spmdType.Elem()) {
				return b.spmdCallVectorReduce("smin", vec), nil
			}
		}
		return b.spmdCallVectorReduce("umin", vec), nil

	case strings.HasPrefix(name, "reduce.Or["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		return b.spmdCallVectorReduce("or", vec), nil

	case strings.HasPrefix(name, "reduce.And["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		return b.spmdCallVectorReduce("and", vec), nil

	case strings.HasPrefix(name, "reduce.Xor["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		return b.spmdCallVectorReduce("xor", vec), nil

	case strings.HasPrefix(name, "reduce.From["):
		// reduce.From[T](v Varying[T]) []T — extract all lanes into a slice.
		// NOTE: Uses stack allocation (alloca). The resulting slice is only valid
		// within the current function scope. TinyGo's escape analysis will promote
		// this to a heap allocation if the slice escapes. For the PoC this is
		// acceptable; a production implementation would use runtime.alloc directly.
		vec := b.getValue(instr.Args[0], getPos(instr))
		vecType := vec.Type()
		elemType := vecType.ElementType()
		laneCount := vecType.VectorSize()

		// Allocate stack space for the elements.
		arrType := llvm.ArrayType(elemType, laneCount)
		alloca := b.CreateAlloca(arrType, "reduce.from.arr")

		// Extract each element and store
		for i := 0; i < laneCount; i++ {
			idx := llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
			elem := b.CreateExtractElement(vec, idx, "")
			gep := b.CreateInBoundsGEP(arrType, alloca, []llvm.Value{
				llvm.ConstInt(b.ctx.Int32Type(), 0, false),
				llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false),
			}, "")
			b.CreateStore(elem, gep)
		}

		// Create slice triple {ptr, len, cap}
		ptr := b.CreateBitCast(alloca, b.dataPtrType, "")
		lenVal := llvm.ConstInt(b.uintptrType, uint64(laneCount), false)
		sliceType := b.getLLVMType(instr.Signature().Results().At(0).Type())
		slice := llvm.Undef(sliceType)
		slice = b.CreateInsertValue(slice, ptr, 0, "")
		slice = b.CreateInsertValue(slice, lenVal, 1, "")
		slice = b.CreateInsertValue(slice, lenVal, 2, "") // cap = len
		return slice, nil

	case name == "reduce.Count":
		// reduce.Count(v Varying[bool]) int — count of true lanes.
		// Normalize to <N x i1> first (on WASM bool vectors are <N x i32>),
		// then bitcast to iN and use llvm.ctpop.
		vec := b.getValue(instr.Args[0], getPos(instr))
		i1Vec := b.spmdNormalizeBoolVecToI1(vec)
		vecSize := i1Vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(i1Vec, intType, "")
		// Call llvm.ctpop to count set bits
		intrinsicName := "llvm.ctpop.i" + strconv.Itoa(vecSize)
		llvmFn := b.mod.NamedFunction(intrinsicName)
		fnType := llvm.FunctionType(intType, []llvm.Type{intType}, false)
		if llvmFn.IsNil() {
			llvmFn = llvm.AddFunction(b.mod, intrinsicName, fnType)
		}
		popcount := b.createCall(fnType, llvmFn, []llvm.Value{intVal}, "")
		return b.createZExtOrTrunc(popcount, b.intType), nil

	case name == "reduce.FindFirstSet":
		// reduce.FindFirstSet(v Varying[bool]) int — index of first true lane.
		// Normalize to <N x i1> first (on WASM bool vectors are <N x i32>),
		// then bitcast to iN and use llvm.cttz.
		vec := b.getValue(instr.Args[0], getPos(instr))
		i1Vec := b.spmdNormalizeBoolVecToI1(vec)
		vecSize := i1Vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(i1Vec, intType, "")
		// Call llvm.cttz to count trailing zeros
		intrinsicName := "llvm.cttz.i" + strconv.Itoa(vecSize)
		llvmFn := b.mod.NamedFunction(intrinsicName)
		fnType := llvm.FunctionType(intType, []llvm.Type{intType, b.ctx.Int1Type()}, false)
		if llvmFn.IsNil() {
			llvmFn = llvm.AddFunction(b.mod, intrinsicName, fnType)
		}
		cttz := b.createCall(fnType, llvmFn, []llvm.Value{
			intVal,
			llvm.ConstInt(b.ctx.Int1Type(), 0, false), // is_zero_poison = false
		}, "")
		return b.createZExtOrTrunc(cttz, b.intType), nil

	case name == "reduce.Mask":
		// reduce.Mask(v Varying[bool]) int — bitmask of active lanes.
		// Normalize to <N x i1> first (on WASM bool vectors are <N x i32>),
		// then bitcast to iN and zero-extend to int.
		vec := b.getValue(instr.Args[0], getPos(instr))
		i1Vec := b.spmdNormalizeBoolVecToI1(vec)
		vecSize := i1Vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(i1Vec, intType, "")
		return b.CreateZExt(intVal, b.intType, ""), nil

	default:
		return llvm.Value{}, b.makeError(getPos(instr), "unsupported reduce builtin: "+name)
	}
}

// spmdMaskedLoad calls llvm.masked.load.<suffix>.p0 to load a vector from a scalar pointer with a per-lane mask.
// The mask parameter may be <N x i32> (WASM format) or <N x i1>; it is truncated to <N x i1> as required
// by the LLVM masked intrinsic interface.
func (b *builder) spmdMaskedLoad(vecType llvm.Type, ptr, mask llvm.Value) llvm.Value {
	laneCount := vecType.VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)

	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.masked.load." + suffix + ".p0"

	ptrType := ptr.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(vecType, []llvm.Type{ptrType, i32Type, i1Mask.Type(), vecType}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)
	passthru := llvm.ConstNull(vecType)

	return b.createCall(fnType, fn, []llvm.Value{ptr, align, i1Mask, passthru}, "spmd.load")
}

// spmdMaskedStore calls llvm.masked.store.<suffix>.p0 to store a vector to a scalar pointer with a per-lane mask.
// The mask parameter may be <N x i32> (WASM format) or <N x i1>; it is truncated to <N x i1> as required
// by the LLVM masked intrinsic interface.
func (b *builder) spmdMaskedStore(val, ptr, mask llvm.Value) {
	vecType := val.Type()
	laneCount := vecType.VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)

	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.masked.store." + suffix + ".p0"

	ptrType := ptr.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(b.ctx.VoidType(), []llvm.Type{vecType, ptrType, i32Type, i1Mask.Type()}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)

	b.createCall(fnType, fn, []llvm.Value{val, ptr, align, i1Mask}, "")
}

// spmdMaskedGather calls llvm.masked.gather.<suffix>.v<N>p0 for non-contiguous loads from a vector of pointers.
// The mask parameter may be <N x i32> (WASM format) or <N x i1>; it is truncated to <N x i1> as required
// by the LLVM masked intrinsic interface.
func (b *builder) spmdMaskedGather(vecType llvm.Type, ptrs, mask llvm.Value) llvm.Value {
	laneCount := ptrs.Type().VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)

	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.masked.gather." + suffix + ".v" + strconv.Itoa(laneCount) + "p0"

	ptrVecType := ptrs.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(vecType, []llvm.Type{ptrVecType, i32Type, i1Mask.Type(), vecType}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)
	passthru := llvm.ConstNull(vecType)

	return b.createCall(fnType, fn, []llvm.Value{ptrs, align, i1Mask, passthru}, "spmd.gather")
}

// spmdMaskedScatter calls llvm.masked.scatter.<suffix>.v<N>p0 for non-contiguous stores to a vector of pointers.
// The mask parameter may be <N x i32> (WASM format) or <N x i1>; it is truncated to <N x i1> as required
// by the LLVM masked intrinsic interface.
func (b *builder) spmdMaskedScatter(val, ptrs, mask llvm.Value) {
	vecType := val.Type()
	laneCount := ptrs.Type().VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)

	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.masked.scatter." + suffix + ".v" + strconv.Itoa(laneCount) + "p0"

	ptrVecType := ptrs.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(b.ctx.VoidType(), []llvm.Type{vecType, ptrVecType, i32Type, i1Mask.Type()}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)

	b.createCall(fnType, fn, []llvm.Value{val, ptrs, align, i1Mask}, "")
}

// spmdBreakResult tracks a phi at rangeint.done that receives a break value.
type spmdBreakResult struct {
	phi          *ssa.Phi   // the phi at rangeint.done
	alloca       llvm.Value // alloca for accumulated break result
	breakEdge    int        // index into phi.Edges for the break edge
	breakVal     ssa.Value  // SSA value on break edge
	defaultVal   ssa.Value  // SSA value on non-break edge (entry or if.done)
	defaultEdge  int        // index of the default edge
}

// spmdForLoopInfo tracks a regular for-range loop inside an SPMD function body.
type spmdForLoopInfo struct {
	loopBlockIndex  int                  // rangeint.loop block index (or bodyBlockIndex for merged pattern)
	bodyBlockIndex  int                  // rangeint.body block index
	doneBlockIndex  int                  // rangeint.done block index
	breakMaskAlloca llvm.Value           // alloca for <N x i1> break mask (persists across iterations)
	laneCount       int                  // SIMD lane count
	breakResults    []spmdBreakResult    // phis at done block with break values
	earlyExitBlocks []llvm.BasicBlock    // blocks that jump to done on all-lanes-broken
}

// spmdBreakRedirect tracks a varying if statement where the then-branch breaks from a loop.
type spmdBreakRedirect struct {
	loop   *spmdForLoopInfo
	cond   llvm.Value      // the varying condition
	target llvm.BasicBlock // where to redirect (else/continuation block)
}

// spmdAnalyzeContiguousIndex checks if an SSA index value is a linear
// expression of a known SPMD loop iterator: base + iter_phi, where base
// is a scalar (uniform) expression. Returns the loop and scalar base offset.
// This generalizes contiguous detection beyond just the raw loop iter phi
// to cover patterns like output[j*width + i] where j*width is scalar.
func (b *builder) spmdAnalyzeContiguousIndex(index ssa.Value) (*spmdActiveLoop, llvm.Value, bool) {
	// Direct iter phi match (existing fast path).
	if loop, ok := b.spmdLoopState.activeLoops[index]; ok {
		return loop, loop.scalarIterVal, true
	}

	// Check BinOp: scalar + iter or iter + scalar.
	binop, ok := index.(*ssa.BinOp)
	if !ok || binop.Op != token.ADD {
		return nil, llvm.Value{}, false
	}

	// Try X=iter, Y=scalar.
	if loop, ok := b.spmdLoopState.activeLoops[binop.X]; ok {
		if scalarVal, ok := b.spmdUnwrapScalar(binop.Y); ok {
			scalarBase := b.CreateAdd(scalarVal, loop.scalarIterVal, "spmd.contiguous.base")
			return loop, scalarBase, true
		}
	}
	// Try X=scalar, Y=iter.
	if loop, ok := b.spmdLoopState.activeLoops[binop.Y]; ok {
		if scalarVal, ok := b.spmdUnwrapScalar(binop.X); ok {
			scalarBase := b.CreateAdd(scalarVal, loop.scalarIterVal, "spmd.contiguous.base")
			return loop, scalarBase, true
		}
	}

	return nil, llvm.Value{}, false
}

// spmdUnwrapScalar retrieves the scalar LLVM value for an SSA value that may
// be wrapped in *ssa.ChangeType chains that broadcast scalars to vectors.
// Note: only *ssa.ChangeType is unwrapped; *ssa.Convert is NOT (it may change
// the numeric value and does not correspond to a pure type annotation).
// Callers must ensure b.spmdValueOverride != nil before calling this function.
// Returns (scalarValue, true) if the underlying value is scalar, or (_, false)
// if the value is genuinely varying (in spmdValueOverride or inherently vector).
func (b *builder) spmdUnwrapScalar(v ssa.Value) (llvm.Value, bool) {
	// Unwrap ChangeType chains to find the underlying SSA value before any
	// scalar-to-vector broadcasting. This handles patterns like:
	//   j*width → ChangeType(j*width, SPMDType{int})
	// where the ChangeType splats the scalar to all lanes.
	unwrapped := v
	for {
		if ct, ok := unwrapped.(*ssa.ChangeType); ok {
			unwrapped = ct.X
		} else {
			break
		}
	}

	// Defensive check: if the unwrapped value is also in spmdValueOverride,
	// it's genuinely varying (e.g., a ChangeType wrapping the iter phi itself).
	if _, isVec := b.spmdValueOverride[unwrapped]; isVec {
		return llvm.Value{}, false
	}

	// Get the LLVM value for the unwrapped (scalar) SSA value.
	// Use the unwrapped value's position for better error location accuracy.
	scalarVal := b.getValue(unwrapped, getPos(unwrapped))
	if scalarVal.Type().TypeKind() == llvm.VectorTypeKind {
		return llvm.Value{}, false
	}
	return scalarVal, true
}

// spmdContiguousIndexAddr handles IndexAddr for contiguous SPMD access.
// Uses the loop's scalar iter value as the index.
func (b *builder) spmdContiguousIndexAddr(expr *ssa.IndexAddr, loop *spmdActiveLoop) (llvm.Value, error) {
	return b.spmdContiguousIndexAddrCore(expr, loop, loop.scalarIterVal)
}

// spmdContiguousIndexAddrCore is the shared implementation for contiguous SPMD IndexAddr.
// It generates a scalar GEP for the base element, registering the result in spmdContiguousPtr
// so that subsequent loads/stores use masked vector intrinsics.
func (b *builder) spmdContiguousIndexAddrCore(expr *ssa.IndexAddr, loop *spmdActiveLoop, scalarIndex llvm.Value) (llvm.Value, error) {
	val := b.getValue(expr.X, getPos(expr))
	scalarIndex = b.extendInteger(scalarIndex, expr.Index.Type(), b.uintptrType)

	var ptr llvm.Value
	switch ptrTyp := expr.X.Type().Underlying().(type) {
	case *types.Pointer:
		typ := ptrTyp.Elem().Underlying()
		switch typ := typ.(type) {
		case *types.Array:
			bufType := b.getLLVMType(typ)
			b.createNilCheck(expr.X, val, "gep")
			ptr = b.CreateInBoundsGEP(bufType, val, []llvm.Value{
				llvm.ConstInt(b.ctx.Int32Type(), 0, false),
				scalarIndex,
			}, "spmd.contiguous.ptr")
		default:
			return llvm.Value{}, b.makeError(expr.Pos(), "unsupported contiguous SPMD indexaddr type: "+typ.String())
		}
	case *types.Slice:
		bufptr := b.CreateExtractValue(val, 0, "indexaddr.ptr")
		bufType := b.getLLVMType(ptrTyp.Elem())
		ptr = b.CreateInBoundsGEP(bufType, bufptr, []llvm.Value{scalarIndex}, "spmd.contiguous.ptr")
	default:
		return llvm.Value{}, b.makeError(expr.Pos(), "unsupported contiguous SPMD indexaddr type: "+ptrTyp.String())
	}

	if b.spmdContiguousPtr != nil {
		b.spmdContiguousPtr[expr] = &spmdContiguousInfo{scalarPtr: ptr, loop: loop}
	}
	return ptr, nil
}

// detectSPMDForLoops scans for regular for-range loops (rangeint pattern) in an
// SPMD function body. These loops need break mask tracking when they contain
// varying if statements with break.
func (b *builder) detectSPMDForLoops() map[int]*spmdForLoopInfo {
	if !b.spmdFuncIsBody {
		return nil
	}

	result := make(map[int]*spmdForLoopInfo)

	// Scan all blocks for rangeint.body blocks (regular for loops in SPMD function bodies).
	for _, block := range b.fn.Blocks {
		if block.Comment != "rangeint.body" {
			continue
		}

		// Find the rangeint.loop predecessor.
		var loopBlock *ssa.BasicBlock
		for _, pred := range block.Preds {
			if pred.Comment == "rangeint.loop" {
				loopBlock = pred
				break
			}
		}
		if loopBlock == nil {
			// Merged body+loop pattern: rangeint.body IS the loop header.
			// Check for rangeint.iter phi and a path to rangeint.done.
			var iterPhi *ssa.Phi
			for _, instr := range block.Instrs {
				if phi, ok := instr.(*ssa.Phi); ok && phi.Comment == "rangeint.iter" {
					iterPhi = phi
					break
				}
			}
			if iterPhi == nil {
				continue // No iter phi, not a rangeint pattern
			}

			// Find rangeint.done among all blocks.
			var doneBlock *ssa.BasicBlock
			for _, blk := range b.fn.Blocks {
				if blk.Comment == "rangeint.done" {
					// Verify this done block is reachable from our body block
					// (within 2 hops — through if.then or if.done)
					for _, succ := range block.Succs {
						for _, succ2 := range succ.Succs {
							if succ2 == blk {
								doneBlock = blk
								break
							}
						}
						if succ == blk {
							doneBlock = blk
						}
						if doneBlock != nil {
							break
						}
					}
					if doneBlock != nil {
						break
					}
				}
			}
			if doneBlock == nil {
				continue
			}

			// Verify the done block is actually connected to this body block
			// by checking backward: done block's predecessors must include paths from our body.
			connected := false
			for _, pred := range doneBlock.Preds {
				if pred == block {
					connected = true
					break
				}
				// Also check 1-hop successors (through if.then or if.done).
				for _, succ := range block.Succs {
					if pred == succ {
						connected = true
						break
					}
				}
				if connected {
					break
				}
			}
			if !connected {
				continue
			}

			// Check it's not inside an SPMD go-for loop.
			insideSPMDGoFor := false
			for _, instr := range block.Instrs {
				if pos := instr.Pos(); pos.IsValid() {
					if b.isInSPMDLoop(pos) != nil {
						insideSPMDGoFor = true
						break
					}
				}
			}
			if insideSPMDGoFor {
				continue
			}

			elemType := b.getLLVMType(iterPhi.Type())
			laneCount := b.spmdLaneCount(elemType)
			maskType := llvm.VectorType(b.spmdMaskElemType(), laneCount)

			// Create break mask alloca at function entry.
			savedBlock := b.GetInsertBlock()
			entryBlock := b.llvmFn.EntryBasicBlock()
			if !entryBlock.IsNil() {
				firstInstr := entryBlock.FirstInstruction()
				if !firstInstr.IsNil() {
					b.SetInsertPointBefore(firstInstr)
				} else {
					b.SetInsertPointAtEnd(entryBlock)
				}
			}
			breakMaskAlloca := b.CreateAlloca(maskType, "spmd.break.mask")
			if !savedBlock.IsNil() {
				b.SetInsertPointAtEnd(savedBlock)
			}

			info := &spmdForLoopInfo{
				loopBlockIndex:  block.Index, // merged: same as body
				bodyBlockIndex:  block.Index,
				doneBlockIndex:  doneBlock.Index,
				breakMaskAlloca: breakMaskAlloca,
				laneCount:       laneCount,
			}

			// Scan phis at done block to find break results.
			// This needs to be done later after all allocas are created.
			result[block.Index] = info
			continue
		}

		// Check if this is inside an SPMD go-for loop. If it is, skip it —
		// SPMD go-for loops are handled by analyzeSPMDLoops(), not here.
		// We only handle regular for loops in SPMD function bodies.
		insideSPMDGoFor := false
		for _, instr := range block.Instrs {
			if pos := instr.Pos(); pos.IsValid() {
				if b.isInSPMDLoop(pos) != nil {
					insideSPMDGoFor = true
					break
				}
			}
		}
		if insideSPMDGoFor {
			continue
		}

		// Find the rangeint.done block: it's the other successor of loopBlock.
		var doneBlock *ssa.BasicBlock
		for _, succ := range loopBlock.Succs {
			if succ != block {
				doneBlock = succ
				break
			}
		}
		if doneBlock == nil {
			continue
		}

		// Determine lane count from the loop's iteration variable type.
		// Find the iter phi in the body block.
		var iterPhi *ssa.Phi
		for _, instr := range block.Instrs {
			if phi, ok := instr.(*ssa.Phi); ok && phi.Comment == "rangeint.iter" {
				iterPhi = phi
				break
			}
		}
		if iterPhi == nil {
			continue
		}

		elemType := b.getLLVMType(iterPhi.Type())
		laneCount := b.spmdLaneCount(elemType)

		// Create alloca for the break mask at the function entry block.
		// We'll initialize it to all-false later in createFunction().
		maskType := llvm.VectorType(b.spmdMaskElemType(), laneCount)

		// Save current insert point, create alloca at function entry.
		savedBlock := b.GetInsertBlock()
		entryBlock := b.llvmFn.EntryBasicBlock()
		if !entryBlock.IsNil() {
			// Insert at the beginning of the entry block.
			firstInstr := entryBlock.FirstInstruction()
			if !firstInstr.IsNil() {
				b.SetInsertPointBefore(firstInstr)
			} else {
				b.SetInsertPointAtEnd(entryBlock)
			}
		}

		breakMaskAlloca := b.CreateAlloca(maskType, "spmd.break.mask")

		// Restore insert point.
		if !savedBlock.IsNil() {
			b.SetInsertPointAtEnd(savedBlock)
		}

		info := &spmdForLoopInfo{
			loopBlockIndex:  loopBlock.Index,
			bodyBlockIndex:  block.Index,
			doneBlockIndex:  doneBlock.Index,
			breakMaskAlloca: breakMaskAlloca,
			laneCount:       laneCount,
		}

		result[block.Index] = info
	}

	// Second pass: populate breakResults for all detected loops.
	for _, loop := range result {
		doneBlock := b.fn.Blocks[loop.doneBlockIndex]

		// Scan phis at the done block.
		for _, instr := range doneBlock.Instrs {
			phi, ok := instr.(*ssa.Phi)
			if !ok {
				break // phis are always first
			}

			// Find break edge (from a then-block that jumps to done).
			// Also find a default edge (from body or entry).
			var breakEdge, defaultEdge int = -1, -1
			var breakVal, defaultVal ssa.Value

			for edgeIdx, pred := range doneBlock.Preds {
				// Check if this predecessor jumps directly to the done block
				// (a break pattern — the then-block of "if cond { break }").
				// Skip the body block itself and non-jump blocks.
				if pred.Index == loop.bodyBlockIndex {
					// This is the loop body's exit edge (not a break).
					defaultEdge = edgeIdx
					defaultVal = phi.Edges[edgeIdx]
					continue
				}
				// Check that the predecessor has a single successor (Jump → done).
				if len(pred.Succs) == 1 && pred.Succs[0].Index == loop.doneBlockIndex {
					breakEdge = edgeIdx
					breakVal = phi.Edges[edgeIdx]
				} else if defaultEdge < 0 {
					// Entry or other non-break predecessor.
					defaultEdge = edgeIdx
					defaultVal = phi.Edges[edgeIdx]
				}
			}

			// If we found a break edge, create a result alloca for this phi.
			if breakEdge >= 0 && defaultEdge >= 0 {
				phiType := b.getLLVMType(phi.Type())

				// Save insert point, create alloca at function entry.
				savedBlock := b.GetInsertBlock()
				entryBlock := b.llvmFn.EntryBasicBlock()
				if !entryBlock.IsNil() {
					firstInstr := entryBlock.FirstInstruction()
					if !firstInstr.IsNil() {
						b.SetInsertPointBefore(firstInstr)
					} else {
						b.SetInsertPointAtEnd(entryBlock)
					}
				}

				alloca := b.CreateAlloca(phiType, "spmd.break.result")

				// Restore insert point.
				if !savedBlock.IsNil() {
					b.SetInsertPointAtEnd(savedBlock)
				}

				loop.breakResults = append(loop.breakResults, spmdBreakResult{
					phi:         phi,
					alloca:      alloca,
					breakEdge:   breakEdge,
					breakVal:    breakVal,
					defaultVal:  defaultVal,
					defaultEdge: defaultEdge,
				})
			}
		}
	}

	return result
}

// spmdIsVaryingBreak checks if a varying if's then-successor jumps directly to a loop exit.
// Returns (loopInfo, true) if this is a varying-break pattern.
func (b *builder) spmdIsVaryingBreak(ifBlock *ssa.BasicBlock) (*spmdForLoopInfo, bool) {
	if b.spmdForLoops == nil {
		return nil, false
	}

	// Check if the then-branch (Succs[0]) has a single Jump instruction.
	thenBlock := ifBlock.Succs[0]
	if len(thenBlock.Instrs) == 0 {
		return nil, false
	}

	lastInstr := thenBlock.Instrs[len(thenBlock.Instrs)-1]
	if _, ok := lastInstr.(*ssa.Jump); !ok {
		return nil, false
	}

	// Check if the Jump targets a loop done block.
	if len(thenBlock.Succs) != 1 {
		return nil, false
	}

	jumpTarget := thenBlock.Succs[0]
	for _, loop := range b.spmdForLoops {
		if jumpTarget.Index == loop.doneBlockIndex {
			return loop, true
		}
	}

	return nil, false
}

// createFromConstrained handles lanes.FromConstrained[T](data Varying[T]) ([]Varying[T], []Varying[bool]).
// It decomposes a constrained vector <constraintN x T> into ceil(constraintN/platformLanes) groups
// of <platformLanes x T> vectors, plus per-group <platformLanes x maskElem> masks.
// Returns an LLVM struct of two slices: {[]Varying[T], []Varying[bool]}.
func (b *builder) createFromConstrained(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	// Get the input vector.
	vec := b.getValue(instr.Args[0], getPos(instr))

	// Get the input SPMDType from instr.Args[0].Type().
	spmdType, ok := instr.Args[0].Type().(*types.SPMDType)
	if !ok {
		return llvm.Value{}, b.makeError(getPos(instr), "lanes.FromConstrained expects Varying[T] argument")
	}

	// Get element LLVM type.
	elemLLVM := b.getLLVMType(spmdType.Elem())

	// Compute platform lane count.
	platformLanes := b.spmdLaneCount(elemLLVM)

	// Universal constraint (0) cannot be decomposed.
	if spmdType.Constraint() == 0 {
		return llvm.Value{}, b.makeError(getPos(instr), "cannot decompose universal constrained Varying[T, 0]; use type switch first")
	}

	// Determine effective constraint N.
	// Prefer the Go type's constraint when available, but fall back to the
	// actual LLVM vector width. The Go type may have been relaxed from
	// Varying[T, N] to Varying[T] by constrained-to-unconstrained assignment
	// (Commit 1), but the LLVM vector retains its original width.
	inputLanes := vec.Type().VectorSize()
	constraintN := b.spmdEffectiveLaneCount(spmdType, elemLLVM)
	if inputLanes > constraintN {
		constraintN = inputLanes
	}

	// Compute number of groups.
	numGroups := (constraintN + platformLanes - 1) / platformLanes

	// Resize input vector if needed to match constraintN.
	if inputLanes < constraintN {
		vec = b.spmdResizeVector(vec, constraintN, elemLLVM)
	}

	// Create platform vector type.
	platVecType := llvm.VectorType(elemLLVM, platformLanes)

	// Create mask elem type and mask vector type.
	maskElem := b.spmdMaskElemType()
	maskVecType := llvm.VectorType(maskElem, platformLanes)

	// Allocate arrays on stack for values and masks.
	valArrType := llvm.ArrayType(platVecType, numGroups)
	maskArrType := llvm.ArrayType(maskVecType, numGroups)
	valAlloca := b.CreateAlloca(valArrType, "fc.vals")
	maskAlloca := b.CreateAlloca(maskArrType, "fc.masks")

	// Extract each group via ShuffleVector and build masks.
	i32 := b.ctx.Int32Type()
	for g := 0; g < numGroups; g++ {
		// Build shuffle mask to extract group.
		indices := make([]llvm.Value, platformLanes)
		for lane := 0; lane < platformLanes; lane++ {
			srcIdx := g*platformLanes + lane
			if srcIdx < constraintN {
				indices[lane] = llvm.ConstInt(i32, uint64(srcIdx), false)
			} else {
				// Out-of-range: use 0 (safe), mask will mark inactive.
				indices[lane] = llvm.ConstInt(i32, 0, false)
			}
		}
		shuffleMask := llvm.ConstVector(indices, false)
		groupVec := b.CreateShuffleVector(vec, llvm.Undef(vec.Type()), shuffleMask, "fc.group")

		// Build mask: all-true for active lanes, all-false for inactive.
		maskElems := make([]llvm.Value, platformLanes)
		for lane := 0; lane < platformLanes; lane++ {
			srcIdx := g*platformLanes + lane
			if srcIdx < constraintN {
				// Active: -1 (all ones) for i32, 1 for i1.
				if maskElem == b.ctx.Int32Type() {
					maskElems[lane] = llvm.ConstInt(maskElem, 0xFFFFFFFF, false)
				} else {
					maskElems[lane] = llvm.ConstInt(maskElem, 1, false)
				}
			} else {
				maskElems[lane] = llvm.ConstInt(maskElem, 0, false)
			}
		}
		groupMask := llvm.ConstVector(maskElems, false)

		// Store to arrays.
		valGEP := b.CreateInBoundsGEP(valArrType, valAlloca, []llvm.Value{
			llvm.ConstInt(i32, 0, false),
			llvm.ConstInt(i32, uint64(g), false),
		}, "")
		b.CreateStore(groupVec, valGEP)

		maskGEP := b.CreateInBoundsGEP(maskArrType, maskAlloca, []llvm.Value{
			llvm.ConstInt(i32, 0, false),
			llvm.ConstInt(i32, uint64(g), false),
		}, "")
		b.CreateStore(groupMask, maskGEP)
	}

	// Build two slices and return as struct.
	// Get result types from signature.
	results := instr.Signature().Results()
	valSliceType := b.getLLVMType(results.At(0).Type())  // []Varying[T]
	maskSliceType := b.getLLVMType(results.At(1).Type()) // []Varying[bool]

	lenVal := llvm.ConstInt(b.uintptrType, uint64(numGroups), false)

	valPtr := b.CreateBitCast(valAlloca, b.dataPtrType, "")
	valSlice := llvm.Undef(valSliceType)
	valSlice = b.CreateInsertValue(valSlice, valPtr, 0, "")
	valSlice = b.CreateInsertValue(valSlice, lenVal, 1, "")
	valSlice = b.CreateInsertValue(valSlice, lenVal, 2, "")

	maskPtr := b.CreateBitCast(maskAlloca, b.dataPtrType, "")
	maskSlice := llvm.Undef(maskSliceType)
	maskSlice = b.CreateInsertValue(maskSlice, maskPtr, 0, "")
	maskSlice = b.CreateInsertValue(maskSlice, lenVal, 1, "")
	maskSlice = b.CreateInsertValue(maskSlice, lenVal, 2, "")

	// Multi-return: LLVM struct {valSlice, maskSlice}.
	retType := b.ctx.StructType([]llvm.Type{valSliceType, maskSliceType}, false)
	ret := llvm.Undef(retType)
	ret = b.CreateInsertValue(ret, valSlice, 0, "")
	ret = b.CreateInsertValue(ret, maskSlice, 1, "")
	return ret, nil
}

// createToConstrained handles lanes.ToConstrained[T](data []Varying[T], mask []Varying[bool], target Varying[T]) Varying[T].
// It reconstructs a constrained vector from groups of unconstrained platform vectors.
// The target parameter determines the output constraint N from its SPMDType.
func (b *builder) createToConstrained(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	// Get target type from the 3rd argument.
	targetType, ok := instr.Args[2].Type().(*types.SPMDType)
	if !ok {
		return llvm.Value{}, b.makeError(getPos(instr), "lanes.ToConstrained target parameter must be Varying[T]")
	}

	// Get element LLVM type.
	elemLLVM := b.getLLVMType(targetType.Elem())

	// Compute constraint N, platform lanes, and number of groups.
	constraintN := b.spmdEffectiveLaneCount(targetType, elemLLVM)
	platformLanes := b.spmdLaneCount(elemLLVM)
	numGroups := (constraintN + platformLanes - 1) / platformLanes

	// Universal constraint (0) cannot be constructed.
	if targetType.Constraint() == 0 {
		return llvm.Value{}, b.makeError(getPos(instr), "cannot construct universal constrained Varying[T, 0]")
	}

	// Get the data slice.
	dataSlice := b.getValue(instr.Args[0], getPos(instr))
	dataPtr := b.CreateExtractValue(dataSlice, 0, "tc.data.ptr")

	// Build output vector type.
	outVecType := llvm.VectorType(elemLLVM, constraintN)

	// Start with zeroinitializer.
	result := llvm.ConstNull(outVecType)

	// Load each group from the data slice and insert elements.
	platVecType := llvm.VectorType(elemLLVM, platformLanes)
	vecSize := b.targetData.TypeAllocSize(platVecType)
	i32 := b.ctx.Int32Type()

	for g := 0; g < numGroups; g++ {
		// Load group vector from data slice.
		offset := llvm.ConstInt(b.uintptrType, uint64(g)*vecSize, false)
		groupPtrI8 := b.CreateInBoundsGEP(b.ctx.Int8Type(), dataPtr, []llvm.Value{offset}, "")
		groupPtr := b.CreateBitCast(groupPtrI8, llvm.PointerType(platVecType, 0), "")
		groupVec := b.CreateLoad(platVecType, groupPtr, "tc.group")

		// Insert each element into result.
		for lane := 0; lane < platformLanes; lane++ {
			dstIdx := g*platformLanes + lane
			if dstIdx >= constraintN {
				break
			}
			elem := b.CreateExtractElement(groupVec, llvm.ConstInt(i32, uint64(lane), false), "")
			result = b.CreateInsertElement(result, elem, llvm.ConstInt(i32, uint64(dstIdx), false), "")
		}
	}

	return result, nil
}
