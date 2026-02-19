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
	tailMask := b.CreateICmp(llvm.IntSLT, laneIndices, boundVec, "spmd.tail.mask")

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
	cond           llvm.Value // vector condition (<N x i1>)
	ifBlockIndex   int        // block with the If instruction
	thenEntryIndex int        // Succs[0] of if-block
	elseEntryIndex int        // Succs[1] of if-block (or merge for if-without-else)
	mergeIndex     int        // common successor (merge point)
	hasElse        bool       // true if then/else are distinct from merge
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
func (b *builder) isBlockInSPMDBody(block *ssa.BasicBlock) *SPMDLoopInfo {
	if b.spmdInfo == nil {
		return nil
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

// spmdDetectVaryingIf detects and analyzes a varying if/else construct at ifBlock.
// The condition is a vector (<N x i1>), so both branches must execute for their
// respective lanes. Populates spmdVaryingIfs, spmdThenExitRedirects, and spmdMergeSelects.
func (b *builder) spmdDetectVaryingIf(ifBlock *ssa.BasicBlock, cond llvm.Value) {
	thenEntry := ifBlock.Succs[0]
	elseEntry := ifBlock.Succs[1]

	// Find the merge block (common successor).
	merge := b.spmdFindMerge(thenEntry, elseEntry)
	if merge == nil {
		// No merge found (possibly unreachable code or exit branches).
		return
	}

	// Determine if this is if-with-else or if-without-else.
	// If-without-else: elseEntry IS the merge block.
	hasElse := (elseEntry != merge)

	info := &spmdVaryingIf{
		cond:           cond,
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
		// For if-with-else: redirect then-exit blocks to else-entry.
		thenExits := b.spmdFindThenExits(thenEntry, merge)
		elseLLVMBlock := b.blockInfo[elseEntry.Index].entry
		for _, exitBlock := range thenExits {
			b.spmdThenExitRedirects[exitBlock.Index] = elseLLVMBlock
		}
	}

	// Record mask transitions for block-level mask stack management.
	if b.spmdMaskTransitions != nil {
		b.spmdMaskTransitions[thenEntry.Index] = &spmdMaskTransition{kind: "pushThen", cond: cond}
		if hasElse {
			b.spmdMaskTransitions[elseEntry.Index] = &spmdMaskTransition{kind: "swapElse", cond: cond}
		}
		b.spmdMaskTransitions[merge.Index] = &spmdMaskTransition{kind: "pop"}
	}
}

// spmdFindMerge finds the merge block (common successor) of then and else branches.
// Uses a simple approach: walk from thenEntry through Jump successors until finding
// a block that is also reachable from elseEntry.
func (b *builder) spmdFindMerge(thenEntry, elseEntry *ssa.BasicBlock) *ssa.BasicBlock {
	// Handle if-without-else: elseEntry itself is the merge.
	if b.spmdIsReachableFrom(thenEntry, elseEntry, nil) {
		return elseEntry
	}

	// Build reachable set from thenEntry (excluding elseEntry subtree).
	visited := make(map[int]bool)
	var walkThen func(*ssa.BasicBlock)
	walkThen = func(block *ssa.BasicBlock) {
		if visited[block.Index] || block == elseEntry {
			return
		}
		visited[block.Index] = true
		for _, succ := range block.Succs {
			walkThen(succ)
		}
	}
	walkThen(thenEntry)

	// Find first block reachable from elseEntry that's also in thenEntry's reachable set.
	var findIntersection func(*ssa.BasicBlock) *ssa.BasicBlock
	elseVisited := make(map[int]bool)
	findIntersection = func(block *ssa.BasicBlock) *ssa.BasicBlock {
		if elseVisited[block.Index] {
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

	// Phi must have exactly 2 edges for merge.
	if len(phi.Edges) != 2 {
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
		// At least one operand is a vector → vector select.
		return b.CreateSelect(info.cond, thenValue, elseValue, ""), true
	}

	// Both are scalars → reduce condition to scalar boolean and use scalar select.
	// Scalar phis at a varying merge represent uniform values. We use any-true
	// reduction: if any lane took the then-branch, use the then-value. This is
	// safe because the SPMD type checker forbids varying-dependent mutation of
	// uniform variables, so both edges carry the same value in practice.
	scalarCond := b.spmdVectorAnyTrue(info.cond)
	return b.CreateSelect(scalarCond, thenValue, elseValue, ""), true
}

// spmdVectorAnyTrue reduces a vector condition <N x i1> to a scalar i1.
// Returns true if any lane is true.
func (b *builder) spmdVectorAnyTrue(mask llvm.Value) llvm.Value {
	// Bitcast <N x i1> to iN (e.g., <4 x i1> → i4).
	vecSize := mask.Type().VectorSize()
	intType := b.ctx.IntType(vecSize)
	intVal := b.CreateBitCast(mask, intType, "")

	// Compare intVal != 0.
	zero := llvm.ConstNull(intType)
	return b.CreateICmp(llvm.IntNE, intVal, zero, "")
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

// spmdMaskType returns the LLVM mask type for an SPMD function's implicit first parameter.
// Returns zero-value llvm.Type{} if the function has no varying parameters.
func (c *compilerContext) spmdMaskType(fn *ssa.Function) llvm.Type {
	return c.spmdMaskTypeFromSig(fn.Signature)
}

// spmdMaskTypeFromSig returns the LLVM mask type for an SPMD signature's implicit mask parameter.
// The mask is a <N x i1> vector where N is determined by the first varying parameter's element type.
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
			return llvm.VectorType(c.ctx.Int1Type(), laneCount)
		}
	}
	return llvm.Type{} // No varying parameters
}

// spmdCallMask returns the mask value to pass when calling an SPMD function.
// The mask is determined by the current execution context:
// - If inside an SPMD loop: use the loop's tail mask
// - If inside an SPMD function: use the entry mask
// - Otherwise: all lanes active (all-ones mask)
func (b *builder) spmdCallMask(fn *ssa.Function) llvm.Value {
	maskType := b.spmdMaskType(fn)
	if maskType == (llvm.Type{}) {
		// Not an SPMD function, no mask needed.
		return llvm.Value{}
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
		// reduce.All(v Varying[bool]) bool — true if all lanes are true
		// Bitcast <N x i1> to iN, compare == -1 (all bits set)
		vec := b.getValue(instr.Args[0], getPos(instr))
		vecSize := vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(vec, intType, "")
		allOnes := llvm.ConstAllOnes(intType)
		return b.CreateICmp(llvm.IntEQ, intVal, allOnes, ""), nil

	case name == "reduce.Any":
		// reduce.Any(v Varying[bool]) bool — true if any lane is true
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
		// reduce.Count(v Varying[bool]) int — count of true lanes
		vec := b.getValue(instr.Args[0], getPos(instr))
		vecSize := vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(vec, intType, "")
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
		// reduce.FindFirstSet(v Varying[bool]) int — index of first true lane
		vec := b.getValue(instr.Args[0], getPos(instr))
		vecSize := vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(vec, intType, "")
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
		// reduce.Mask(v Varying[bool]) int — bitmask of active lanes
		vec := b.getValue(instr.Args[0], getPos(instr))
		vecSize := vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(vec, intType, "")
		return b.CreateZExt(intVal, b.intType, ""), nil

	default:
		return llvm.Value{}, b.makeError(getPos(instr), "unsupported reduce builtin: "+name)
	}
}

// spmdMaskedLoad calls llvm.masked.load.<suffix>.p0 to load a vector from a scalar pointer with a per-lane mask.
func (b *builder) spmdMaskedLoad(vecType llvm.Type, ptr, mask llvm.Value) llvm.Value {
	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.masked.load." + suffix + ".p0"

	ptrType := ptr.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(vecType, []llvm.Type{ptrType, i32Type, mask.Type(), vecType}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)
	passthru := llvm.ConstNull(vecType)

	return b.createCall(fnType, fn, []llvm.Value{ptr, align, mask, passthru}, "spmd.load")
}

// spmdMaskedStore calls llvm.masked.store.<suffix>.p0 to store a vector to a scalar pointer with a per-lane mask.
func (b *builder) spmdMaskedStore(val, ptr, mask llvm.Value) {
	vecType := val.Type()
	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.masked.store." + suffix + ".p0"

	ptrType := ptr.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(b.ctx.VoidType(), []llvm.Type{vecType, ptrType, i32Type, mask.Type()}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)

	b.createCall(fnType, fn, []llvm.Value{val, ptr, align, mask}, "")
}

// spmdMaskedGather calls llvm.masked.gather.<suffix>.v<N>p0 for non-contiguous loads from a vector of pointers.
func (b *builder) spmdMaskedGather(vecType llvm.Type, ptrs, mask llvm.Value) llvm.Value {
	suffix := spmdVectorTypeSuffix(vecType)
	laneCount := ptrs.Type().VectorSize()
	intrinsicName := "llvm.masked.gather." + suffix + ".v" + strconv.Itoa(laneCount) + "p0"

	ptrVecType := ptrs.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(vecType, []llvm.Type{ptrVecType, i32Type, mask.Type(), vecType}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)
	passthru := llvm.ConstNull(vecType)

	return b.createCall(fnType, fn, []llvm.Value{ptrs, align, mask, passthru}, "spmd.gather")
}

// spmdMaskedScatter calls llvm.masked.scatter.<suffix>.v<N>p0 for non-contiguous stores to a vector of pointers.
func (b *builder) spmdMaskedScatter(val, ptrs, mask llvm.Value) {
	vecType := val.Type()
	suffix := spmdVectorTypeSuffix(vecType)
	laneCount := ptrs.Type().VectorSize()
	intrinsicName := "llvm.masked.scatter." + suffix + ".v" + strconv.Itoa(laneCount) + "p0"

	ptrVecType := ptrs.Type()
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(b.ctx.VoidType(), []llvm.Type{vecType, ptrVecType, i32Type, mask.Type()}, false)

	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}

	elemSize := b.targetData.TypeAllocSize(vecType.ElementType())
	align := llvm.ConstInt(i32Type, elemSize, false)

	b.createCall(fnType, fn, []llvm.Value{val, ptrs, align, mask}, "")
}

// spmdContiguousIndexAddr handles IndexAddr for contiguous SPMD access.
// Returns a scalar pointer to the base element (for subsequent vector load/store via spmdMaskedLoad/Store).
// Returns an error if the container type is not supported for contiguous access.
func (b *builder) spmdContiguousIndexAddr(expr *ssa.IndexAddr, loop *spmdActiveLoop) (llvm.Value, error) {
	val := b.getValue(expr.X, getPos(expr))
	scalarIndex := loop.scalarIterVal
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
