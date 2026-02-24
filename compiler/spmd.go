package compiler

// This file extracts SPMD metadata from the typed AST for use during LLVM IR
// generation. TinyGo uses golang.org/x/tools/go/ssa which has no SPMD support,
// so we build a side-table from go/ast and go/types (which our Go fork extends).

import (
	"go/ast"
	"go/constant"
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
	ForPos    token.Pos // position of "for" keyword
	BodyStart token.Pos // start of loop body (opening brace)
	BodyEnd   token.Pos // end of loop body (closing brace)
	LaneCount int64     // SIMD lane count (from type checker)
}

// SPMDParamInfo holds info about a single varying parameter.
type SPMDParamInfo struct {
	Index    int
	ElemType types.Type
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

			// Create loop info.
			info := &SPMDLoopInfo{
				ForPos:    rangeStmt.For,
				BodyStart: rangeStmt.Body.Lbrace,
				BodyEnd:   rangeStmt.Body.Rbrace,
				LaneCount: rangeStmt.LaneCount, // set by type checker
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
					Index:    i,
					ElemType: spmdType.Elem(),
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

// spmdEffectiveLaneCount returns the lane count for an SPMDType derived from
// the SIMD register width for the element type.
func (c *compilerContext) spmdEffectiveLaneCount(spmdType *types.SPMDType, elemLLVM llvm.Type) int {
	return c.spmdLaneCount(elemLLVM)
}

// spmdRangeIndexLaneCount computes the lane count for a range-over-slice SPMD loop
// by examining the slice element type instead of the iterator's int type.
// For []byte slices, this yields 128/8=16 lanes (native v128) instead of 128/32=4.
// Falls back to the iterator type when no slice element type can be determined.
//
// Strategy 2 (IndexAddr scan) requires the index to be exactly incrBinOp — it does
// not trace through ChangeType chains. This is sufficient for the common range-over-slice
// pattern where go/ssa uses incrBinOp directly as the IndexAddr index.
func (b *builder) spmdRangeIndexLaneCount(boundValue ssa.Value, bodyBlock *ssa.BasicBlock, incrBinOp *ssa.BinOp) int {
	// Strategy 1: if boundValue is a call to builtin len, extract the slice element type.
	if call, ok := boundValue.(*ssa.Call); ok {
		if builtin, ok := call.Call.Value.(*ssa.Builtin); ok && builtin.Name() == "len" {
			if len(call.Call.Args) == 1 {
				arg := call.Call.Args[0]
				if sliceType, ok := arg.Type().Underlying().(*types.Slice); ok {
					elemLLVM := b.getLLVMType(sliceType.Elem())
					return b.spmdLaneCount(elemLLVM)
				}
			}
		}
	}

	// Strategy 2: scan the body block for an IndexAddr whose index traces back
	// to incrBinOp, and extract the slice element type from its base.
	for _, instr := range bodyBlock.Instrs {
		ia, ok := instr.(*ssa.IndexAddr)
		if !ok {
			continue
		}
		// The index must be the increment BinOp (the SPMD iterator).
		if ia.Index != ssa.Value(incrBinOp) {
			continue
		}
		if sliceType, ok := ia.X.Type().Underlying().(*types.Slice); ok {
			elemLLVM := b.getLLVMType(sliceType.Elem())
			return b.spmdLaneCount(elemLLVM)
		}
		if ptrType, ok := ia.X.Type().Underlying().(*types.Pointer); ok {
			if arrType, ok := ptrType.Elem().Underlying().(*types.Array); ok {
				elemLLVM := b.getLLVMType(arrType.Elem())
				return b.spmdLaneCount(elemLLVM)
			}
		}
	}

	// Fallback: use the iterator type (int → 4 lanes on wasm32).
	return b.spmdLaneCount(b.getLLVMType(incrBinOp.Type()))
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

	// Decomposed index fields (isDecomposed == true):
	// When laneCount > 4 on WASM (e.g., 16 for byte), a naive <16 x i32> index vector
	// would be 512-bit, exceeding WASM's 128-bit SIMD registers. Instead the index is
	// represented as scalar base (i32) + varying offset (<16 x i8>), with math
	// operations decomposed algebraically. isDecomposed is set by analyzeSPMDLoops.
	isDecomposed bool

	// Set during IR generation:
	laneIndices   llvm.Value // <iter, iter+1, ..., iter+laneCount-1> (nil when isDecomposed)
	tailMask      llvm.Value // per-lane bounds check
	scalarIterVal llvm.Value // scalar LLVM value (before override to lane indices)
}

// spmdDecomposedIndex tracks a base+offset decomposed SPMD index value.
// Used for byte-lane loops (laneCount > 4) where the full materialized vector
// (<16 x i32>) would exceed WASM's 128-bit register width.
type spmdDecomposedIndex struct {
	scalarBase    llvm.Value      // Scalar i32 component (uniform across lanes)
	varyingOffset llvm.Value      // <N x i8> component (varying per lane)
	laneCount     int             // Number of lanes (e.g., 16 for byte)
	loop          *spmdActiveLoop // The SPMD loop this index belongs to
	// fromBodyIter is true when this decomposition is the raw body iterator
	// from emitSPMDBodyPrologue (i.e., base = loop counter, offset = <0,1,...,N-1>).
	// SHR/AND/REM algebraic shortcuts require that base is aligned to the relevant
	// boundary, which is guaranteed only for the direct body iterator (whose base
	// is always a multiple of laneCount). After an ADD/SUB the base shifts by an
	// arbitrary amount, so those shortcuts are no longer valid.
	fromBodyIter bool
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

		// Compute lane count from the slice element type (if available) for optimal
		// SIMD width. For []byte slices this yields 16 lanes instead of 4.
		laneCount := b.spmdRangeIndexLaneCount(boundValue, block, incrBinOp)

		// On WASM with laneCount > 4, a naive <laneCount x i32> index vector would
		// be wider than 128 bits (e.g., <16 x i32> is 512-bit). Use decomposed
		// representation (scalar base + <N x i8> offset) to stay within 128 bits.
		isDecomposed := b.spmdIsWASM() && laneCount > 4

		loop := &spmdActiveLoop{
			info:          loopInfo,
			iterPhi:       nil, // rangeindex has no iter phi in body
			laneCount:     laneCount,
			boundValue:    boundValue,
			incrBinOp:     incrBinOp,
			isRangeIndex:  true,
			isDecomposed:  isDecomposed,
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
//
// For decomposed loops (isDecomposed == true, only possible on rangeindex with
// laneCount > 4 on WASM), the index is represented as scalar base + <N x i8>
// offset to avoid creating a <16 x i32> 512-bit vector that would exceed WASM's
// 128-bit SIMD registers. The decomposition is stored in b.spmdDecomposed.
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

	if loop.isDecomposed {
		// Decomposed path: represent the index as scalar base + <N x i8> offset.
		// This avoids creating a <16 x i32> 512-bit vector on WASM SIMD128.
		i8Type := b.ctx.Int8Type()
		laneCount := loop.laneCount

		// Create constant byte offset <0, 1, 2, ..., laneCount-1> as <N x i8>.
		varyingOffset := b.spmdLaneOffsetConst(laneCount, i8Type)

		// Register the decomposition so BinOp/IndexAddr handlers can use it.
		if b.spmdDecomposed != nil {
			b.spmdDecomposed[loop.bodyIterValue] = &spmdDecomposedIndex{
				scalarBase:    scalarPhi,
				varyingOffset: varyingOffset,
				laneCount:     laneCount,
				loop:          loop,
				fromBodyIter:  true, // raw iterator: base is always a multiple of laneCount
			}
		}

		// Compute tail mask using <N x i8> comparison to stay within 128-bit registers.
		// diff = bound - base (scalar i32); clamp to [0, laneCount]; truncate to i8.
		// Then compare: offset < clamp(diff) using unsigned <N x i8> comparison.
		boundScalar := b.getValue(loop.boundValue, token.NoPos)
		diff := b.CreateSub(boundScalar, scalarPhi, "spmd.diff")

		// Clamp diff to [0, laneCount]: max(0, min(laneCount, diff)).
		zero32 := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
		lcConst := llvm.ConstInt(b.ctx.Int32Type(), uint64(laneCount), false)
		// min(laneCount, diff): if diff > laneCount, use laneCount
		diffClamped := b.CreateSelect(
			b.CreateICmp(llvm.IntSGT, diff, lcConst, ""),
			lcConst, diff, "spmd.diff.clamped")
		// max(0, clamped): if clamped < 0, use 0
		diffClamped = b.CreateSelect(
			b.CreateICmp(llvm.IntSLT, diffClamped, zero32, ""),
			zero32, diffClamped, "spmd.diff.nonneg")

		// Truncate clamped diff to i8 (safe: value is in [0, laneCount=16]).
		diffI8 := b.CreateTrunc(diffClamped, i8Type, "spmd.diff.i8")

		// Splat diffI8 to <N x i8> for vector comparison.
		diffVec := b.splatScalar(diffI8, llvm.VectorType(i8Type, laneCount))

		// Compute: offset < clamp(diff) using unsigned comparison.
		tailMaskI1 := b.CreateICmp(llvm.IntULT, varyingOffset, diffVec, "spmd.tail.mask")
		loop.tailMask = b.spmdWrapMask(tailMaskI1, laneCount)
		// laneIndices not set for decomposed path (use spmdDecomposed map instead).
		return
	}

	// Non-decomposed path: materialize the full index vector.
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

// spmdMaterializeDecomposed converts a decomposed index (scalar base + <N x i8> offset)
// to a full <N x i32> vector by zero-extending the offset and adding the splatted base.
// This is only used as a fallback when an operation cannot be decomposed algebraically.
// On WASM SIMD128 this produces a 512-bit <16 x i32> value which LLVM must scalarize;
// it is only acceptable for non-WASM targets or when no 128-bit path is feasible.
func (b *builder) spmdMaterializeDecomposed(decomp *spmdDecomposedIndex) llvm.Value {
	i32Type := b.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, decomp.laneCount)
	baseVec := b.splatScalar(decomp.scalarBase, vecType)
	offsetExt := b.CreateZExt(decomp.varyingOffset, vecType, "")
	return b.CreateAdd(baseVec, offsetExt, "spmd.materialized.idx")
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

// spmdParentMask returns the mask one level below the top of the stack.
// This is the mask that was active before the current varying if pushed its mask.
// Returns nil Value if the stack has fewer than 2 elements.
func (b *builder) spmdParentMask() llvm.Value {
	if len(b.spmdMaskStack) >= 2 {
		return b.spmdMaskStack[len(b.spmdMaskStack)-2]
	}
	return llvm.Value{}
}

// spmdVaryingIf holds analysis results for a varying (vector) if/else construct.
type spmdVaryingIf struct {
	cond              llvm.Value // vector condition: <N x i1> on non-WASM, <N x i32> on WASM
	ifBlockIndex      int        // block with the If instruction
	thenEntryIndex    int        // Succs[0] of if-block
	elseEntryIndex    int        // Succs[1] of if-block (or merge for if-without-else)
	mergeIndex        int        // common successor (merge point)
	hasElse           bool       // true if then/else are distinct from merge
	chainDefaultPreds []int      // SSA pred indices carrying default value (for chains without else)
}

// spmdSwitchCase holds per-case mask info within a switch chain.
type spmdSwitchCase struct {
	ifBlock   int        // block index of the switch.next If instruction
	bodyBlock int        // block index of the switch.body
	caseMask  llvm.Value // computed at compile time: remainingMask & condition
}

// spmdSwitchChain holds a detected varying switch chain.
type spmdSwitchChain struct {
	cases       []spmdSwitchCase
	defaultBody int       // block index of default case body (-1 if none)
	doneBlock   int       // block index of switch.done merge
	tagValue    ssa.Value // the switch tag SSA value (for reference)
}

// spmdDeferredSwitchPhi tracks a phi at switch.done that needs deferred resolution.
// In DomPreorder, switch.done can be visited before the switch.next comparison blocks,
// so case masks may not yet be computed when the phi is first encountered. The phi is
// created as a normal LLVM phi during createExpr, and the cascaded select is built
// after all blocks have been processed (when all case masks are available).
type spmdDeferredSwitchPhi struct {
	phi      *ssa.Phi   // SSA phi at switch.done
	llvm     llvm.Value // LLVM phi (placeholder — will be replaced)
	chainIdx int        // index into spmdSwitchChains
}

// spmdCondChain describes a chain of short-circuit boolean conditions (&&/||)
// detected from cond.true/cond.false SSA patterns. Instead of treating each
// block's If as a separate varying if, the chain is collapsed into a single
// combined condition (vector AND for &&, vector OR for ||).
type spmdCondChain struct {
	outerIfBlock int         // block index of outermost If
	innerBlocks  []int       // cond.true/cond.false block indices (ordered outer→inner)
	op           token.Token // token.LAND or token.LOR
	thenTarget   int         // actual then-body entry block index
	elseTarget   int         // actual else-body entry (or merge for no-else)
	combinedCond llvm.Value  // filled during compilation: a & b [& c...]
}

// spmdMergePhiOverride tracks a phi at a varying-if merge block where then/else
// edges have been combined via a deferred select instruction.
//
// Two cases arise:
//
//  1. Multi-predecessor merge (loop header, 3+ edges): the then-branch edge is
//     skipped and the else-branch edge is replaced by select(cond, then, else).
//     skipEdgeIdx == thenEdgeIdx.
//
//  2. 2-edge merge (plain if/else or if-without-else): after CFG linearization
//     only one LLVM predecessor remains. The original "redirected" edge no longer
//     has a live LLVM predecessor, so a plain phi would be invalid.
//     skipEdgeIdx is the edge whose LLVM predecessor was redirected away.
//     elseEdgeIdx is the surviving edge; phi resolution reads blockInfo[...].exit
//     at resolution time to get the correct exit block (which may differ from the
//     block at registration time due to createRuntimeAssert inserting bounds checks).
//
// The select is always created during phi resolution (not phi creation) to avoid
// circular dependencies when edge values reference the phi itself.
type spmdMergePhiOverride struct {
	skipEdgeIdx  int            // phi edge index to skip during resolution (no LLVM pred after linearization)
	skipEdgeIdxs []int          // multiple edges to skip (for condition chains)
	thenEdgeIdx  int            // phi edge index for the then-branch value
	elseEdgeIdx  int            // phi edge index for the else-branch value; for 2-edge cases also the surviving LLVM pred
	info         *spmdVaryingIf // varying-if info (holds the vector condition)
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

// spmdShiftedLoadInfo describes a gather that can be optimized to a smaller
// contiguous load + shuffle expansion. For pattern s[(base + iter) >> shift]:
//   - The effective indices are: (base + [0,1,...,N-1]) >> shift
//   - uniqueCount = number of unique indices = ceil(laneCount / (1<<shift))
//   - shuffleMask maps each lane to its position in the loaded vector
type spmdShiftedLoadInfo struct {
	scalarPtr   llvm.Value      // scalar GEP to s[base >> shift]
	uniqueCount int             // number of unique elements to load
	shuffleMask []int           // lane i -> index in loaded vector
	elemType    llvm.Type       // element type being loaded
	loop        *spmdActiveLoop // owning SPMD loop
}

// spmdCoalescedStore represents a pair of stores in then/else branches of a varying
// if/else that write to the same destination. Instead of two masked stores, codegen
// emits select(cond, thenVal, elseVal) + one store with the parent mask.
type spmdCoalescedStore struct {
	thenStore *ssa.Store     // store in then-branch (skipped during codegen)
	elseStore *ssa.Store     // store in else-branch (emits the coalesced store)
	ifInfo    *spmdVaryingIf // the varying if that contains them
}

// spmdSameStoreAddr checks whether two SSA values represent the same store destination.
// Returns true if they are the same SSA value, or both are *ssa.IndexAddr with the
// same base (.X) and index (.Index).
func spmdSameStoreAddr(a, b ssa.Value) bool {
	if a == nil || b == nil {
		return false
	}
	if a == b {
		return true
	}
	idxA, okA := a.(*ssa.IndexAddr)
	idxB, okB := b.(*ssa.IndexAddr)
	if okA && okB && idxA.X == idxB.X && idxA.Index == idxB.Index {
		return true
	}
	return false
}

// spmdCollectBranchStores collects all *ssa.Store instructions reachable from
// entryBlock without crossing the merge block or the barrier (if-block).
// Returns a map from store destination (Addr) to the last store to that address.
func (b *builder) spmdCollectBranchStores(entryBlock, merge, barrier *ssa.BasicBlock) map[ssa.Value]*ssa.Store {
	stores := make(map[ssa.Value]*ssa.Store)
	visited := make(map[int]bool)

	var walk func(*ssa.BasicBlock)
	walk = func(block *ssa.BasicBlock) {
		if block == nil || block == merge || block == barrier || visited[block.Index] {
			return
		}
		visited[block.Index] = true
		for _, instr := range block.Instrs {
			if store, ok := instr.(*ssa.Store); ok {
				stores[store.Addr] = store
			}
		}
		for _, succ := range block.Succs {
			walk(succ)
		}
	}
	walk(entryBlock)
	return stores
}

// spmdAnalyzeCoalescedStores detects matching stores in the then and else branches
// of a varying if/else and records them in spmdCoalescedStores for codegen.
func (b *builder) spmdAnalyzeCoalescedStores(info *spmdVaryingIf) {
	if !info.hasElse {
		return // No else branch → nothing to coalesce
	}

	ifBlock := b.fn.Blocks[info.ifBlockIndex]
	thenEntry := b.fn.Blocks[info.thenEntryIndex]
	elseEntry := b.fn.Blocks[info.elseEntryIndex]
	merge := b.fn.Blocks[info.mergeIndex]

	thenStores := b.spmdCollectBranchStores(thenEntry, merge, ifBlock)
	elseStores := b.spmdCollectBranchStores(elseEntry, merge, ifBlock)

	// Match then-stores to else-stores by destination address.
	for thenAddr, thenStore := range thenStores {
		for elseAddr, elseStore := range elseStores {
			if spmdSameStoreAddr(thenAddr, elseAddr) {
				// Verify both values have the same type.
				if !types.Identical(thenStore.Val.Type(), elseStore.Val.Type()) {
					continue
				}
				coal := &spmdCoalescedStore{
					thenStore: thenStore,
					elseStore: elseStore,
					ifInfo:    info,
				}
				b.spmdCoalescedStores[thenStore] = coal
				b.spmdCoalescedStores[elseStore] = coal
				break // One match per then-store
			}
		}
	}
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

// spmdDetectCondChains scans blocks in an SPMD context for cond.true/cond.false
// patterns generated by short-circuit && and || operators. When both operands
// are varying (SPMDType), the chain is registered so that inner blocks are
// skipped during normal varying-if detection and the combined condition is used
// instead.
func (b *builder) spmdDetectCondChains() {
	// Initialize maps.
	b.spmdCondChains = make(map[int]*spmdCondChain)
	b.spmdCondChainInner = make(map[int]*spmdCondChain)

	for _, block := range b.fn.Blocks {
		if b.isBlockInSPMDBody(block) == nil {
			continue
		}
		// Look for cond.true or cond.false block comments.
		comment := block.Comment
		if comment != "cond.true" && comment != "cond.false" &&
			!strings.HasPrefix(comment, "cond.true.") && !strings.HasPrefix(comment, "cond.false.") {
			continue
		}

		// Already registered as inner?
		if _, ok := b.spmdCondChainInner[block.Index]; ok {
			continue
		}

		// Check that this block ends with an If with SPMDType condition.
		if len(block.Instrs) == 0 {
			continue
		}
		ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If)
		if !ok {
			continue
		}
		if _, ok := ifInstr.Cond.Type().(*types.SPMDType); !ok {
			continue
		}

		// Determine chain type from comment.
		isCondTrue := strings.HasPrefix(comment, "cond.true")

		// Walk back to find the outer (head) block.
		// cond.true blocks have exactly 1 predecessor (the outer block).
		// cond.false blocks also have exactly 1 predecessor.
		if len(block.Preds) != 1 {
			continue
		}
		outerBlock := block.Preds[0]

		// Outer block must also end with If and have SPMDType condition.
		if len(outerBlock.Instrs) == 0 {
			continue
		}
		outerIf, ok := outerBlock.Instrs[len(outerBlock.Instrs)-1].(*ssa.If)
		if !ok {
			continue
		}
		if _, ok := outerIf.Cond.Type().(*types.SPMDType); !ok {
			continue
		}

		// Skip if already part of a switch chain.
		if _, ok := b.spmdSwitchIfBlocks[block.Index]; ok {
			continue
		}
		if _, ok := b.spmdSwitchIfBlocks[outerBlock.Index]; ok {
			continue
		}

		// Validate structural invariant:
		// && (cond.true): outer.Succs[0]==inner, both share same false target (outer.Succs[1] == inner.Succs[1])
		// || (cond.false): outer.Succs[1]==inner, both share same true target (outer.Succs[0] == inner.Succs[0])
		var op token.Token
		if isCondTrue {
			op = token.LAND
			if outerBlock.Succs[0] != block {
				continue // not the expected structure
			}
			if outerBlock.Succs[1] != block.Succs[1] {
				continue // false targets don't match
			}
		} else {
			op = token.LOR
			if outerBlock.Succs[1] != block {
				continue
			}
			if outerBlock.Succs[0] != block.Succs[0] {
				continue // true targets don't match
			}
		}

		// Check if the outer block is already the head of a chain (for a && b && c).
		if existingChain, ok := b.spmdCondChains[outerBlock.Index]; ok {
			// Extend the existing chain.
			existingChain.innerBlocks = append(existingChain.innerBlocks, block.Index)
			// Update targets from the innermost block.
			if op == token.LAND {
				existingChain.thenTarget = block.Succs[0].Index
			} else {
				existingChain.elseTarget = block.Succs[1].Index
			}
			b.spmdCondChainInner[block.Index] = existingChain
			continue
		}

		// Check if the outer block is an inner block of an existing chain (deeper nesting).
		if existingChain, ok := b.spmdCondChainInner[outerBlock.Index]; ok {
			if existingChain.op == op { // same operator throughout
				existingChain.innerBlocks = append(existingChain.innerBlocks, block.Index)
				if op == token.LAND {
					existingChain.thenTarget = block.Succs[0].Index
				} else {
					existingChain.elseTarget = block.Succs[1].Index
				}
				b.spmdCondChainInner[block.Index] = existingChain
				continue
			}
			// Mixed operators (a && (b || c)) — skip, not a flat chain.
			continue
		}

		// Check if block is already the head of a sub-chain (e.g., for a && b && c
		// where cond.true.1 was processed before cond.true due to block ordering).
		// If so, absorb the sub-chain: move outerIfBlock up to the new outerBlock
		// and prepend block to the inner list.
		if subChain, ok := b.spmdCondChains[block.Index]; ok && subChain.op == op {
			// Absorb: the existing chain block→[...] becomes outerBlock→[block, ...]
			subChain.outerIfBlock = outerBlock.Index
			subChain.innerBlocks = append([]int{block.Index}, subChain.innerBlocks...)
			// Update targets from outerBlock for the shared side.
			if op == token.LAND {
				subChain.elseTarget = outerBlock.Succs[1].Index
			} else {
				subChain.thenTarget = outerBlock.Succs[0].Index
			}
			// Re-register: remove old head, register new head + inner.
			delete(b.spmdCondChains, block.Index)
			b.spmdCondChains[outerBlock.Index] = subChain
			b.spmdCondChainInner[block.Index] = subChain
			continue
		}

		// Create new chain.
		var thenTarget, elseTarget int
		if op == token.LAND {
			thenTarget = block.Succs[0].Index      // innermost then
			elseTarget = outerBlock.Succs[1].Index // shared false (== inner false)
		} else {
			thenTarget = outerBlock.Succs[0].Index // shared true (== inner true)
			elseTarget = block.Succs[1].Index      // innermost else
		}

		chain := &spmdCondChain{
			outerIfBlock: outerBlock.Index,
			innerBlocks:  []int{block.Index},
			op:           op,
			thenTarget:   thenTarget,
			elseTarget:   elseTarget,
		}
		b.spmdCondChains[outerBlock.Index] = chain
		b.spmdCondChainInner[block.Index] = chain
	}

}

// preDetectVaryingIfs scans all blocks in the function for If instructions
// with varying conditions and calls spmdAnalyzeVaryingIf to populate the
// spmdMergeSelects map. This must be done before compiling blocks so that
// phis at merge blocks can be converted to selects.
func (b *builder) preDetectVaryingIfs() {
	// Detect varying switch chains first, so that the individual varying-if
	// detection below can skip switch.next blocks (preventing map collisions
	// in spmdMergeSelects where all cases share the same switch.done merge).
	b.spmdDetectSwitchChains(b.fn)

	// Detect condition chains (short-circuit && / ||).
	b.spmdDetectCondChains()

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

		// Skip if this block is part of a switch chain (will be handled separately).
		if _, isSwitch := b.spmdSwitchIfBlocks[block.Index]; isSwitch {
			continue
		}

		// Skip if this block is part of a condition chain (will be handled as combined condition).
		if _, isChainInner := b.spmdCondChainInner[block.Index]; isChainInner {
			continue
		}

		// Check if condition is varying (SPMDType).
		if _, ok := ifInstr.Cond.Type().(*types.SPMDType); ok {
			b.spmdAnalyzeVaryingIf(block)
			// Detect matching stores in then/else branches for coalescing.
			if info, ok := b.spmdVaryingIfs[block.Index]; ok {
				b.spmdAnalyzeCoalescedStores(info)
			}
		}
	}
}

// spmdDetectSwitchChains scans the function for varying switch chains and populates
// spmdSwitchChains, spmdSwitchIfBlocks, and spmdSwitchBodyBlocks maps.
//
// Switch chains in go/ssa are represented as chains of If instructions with block
// comments following the pattern: "switch.next" (comparison blocks), "switch.body"
// (case bodies), and "switch.done" (merge point).
//
// go/ssa may merge the first case comparison into the parent block (e.g.,
// "rangeint.body"), so the chain head may not have a "switch.next" comment.
// The default case may be a "switch.next" block ending with Jump (not If),
// containing the default case's code with its single successor being switch.done.
func (b *builder) spmdDetectSwitchChains(fn *ssa.Function) {
	// Find all switch.next blocks that are chain heads (not pointed to by another switch.next).
	var chainStarts []*ssa.BasicBlock
	for _, block := range fn.Blocks {
		if b.isBlockInSPMDBody(block) == nil {
			continue
		}
		if !strings.HasPrefix(block.Comment, "switch.next") {
			continue
		}
		isFirst := true
		for _, otherBlock := range fn.Blocks {
			if len(otherBlock.Succs) >= 2 && otherBlock.Succs[1] == block &&
				strings.HasPrefix(otherBlock.Comment, "switch.next") {
				isFirst = false
				break
			}
		}
		if isFirst {
			chainStarts = append(chainStarts, block)
		}
	}

	// spmdExtractSwitchTag extracts the non-constant operand from a BinOp EQL
	// condition, which is the switch tag value.
	extractTag := func(block *ssa.BasicBlock) ssa.Value {
		if len(block.Instrs) == 0 {
			return nil
		}
		ifInstr, ok := block.Instrs[len(block.Instrs)-1].(*ssa.If)
		if !ok {
			return nil
		}
		binOp, ok := ifInstr.Cond.(*ssa.BinOp)
		if !ok || binOp.Op != token.EQL {
			return nil
		}
		if _, isConst := binOp.Y.(*ssa.Const); isConst {
			return binOp.X
		}
		if _, isConst := binOp.X.(*ssa.Const); isConst {
			return binOp.Y
		}
		return binOp.X
	}

	// For each chain start, check if its predecessor also compares the same tag.
	// go/ssa may merge the first comparison into the parent block (e.g., rangeint.body).
	actualChainStarts := make([]*ssa.BasicBlock, 0, len(chainStarts))
	for _, start := range chainStarts {
		actualStart := start
		startTag := extractTag(start)
		if startTag != nil {
			for _, pred := range start.Preds {
				if b.isBlockInSPMDBody(pred) == nil {
					continue
				}
				predTag := extractTag(pred)
				if predTag == startTag {
					// The predecessor compares the same tag — it's the actual first case.
					actualStart = pred
					break
				}
			}
		}
		actualChainStarts = append(actualChainStarts, actualStart)
	}

	// Process each chain.
	for _, startBlock := range actualChainStarts {
		chain := spmdSwitchChain{
			defaultBody: -1,
			doneBlock:   -1,
		}

		// Walk the chain: collect cases and extract the switch tag.
		// The walk handles three block types:
		//   1. If block (case comparison): Succs[0]=body, Succs[1]=next
		//   2. switch.next with Jump: default case body, Succs[0]=done
		//   3. Other: end of chain
		currentBlock := startBlock
		for {
			if len(currentBlock.Instrs) == 0 {
				break
			}
			lastInstr := currentBlock.Instrs[len(currentBlock.Instrs)-1]

			if ifInstr, ok := lastInstr.(*ssa.If); ok {
				// Case comparison block.
				if binOp, ok := ifInstr.Cond.(*ssa.BinOp); ok && binOp.Op == token.EQL {
					if chain.tagValue == nil {
						if _, isConst := binOp.Y.(*ssa.Const); isConst {
							chain.tagValue = binOp.X
						} else if _, isConst := binOp.X.(*ssa.Const); isConst {
							chain.tagValue = binOp.Y
						} else {
							chain.tagValue = binOp.X
						}
					} else {
						if chain.tagValue != binOp.X && chain.tagValue != binOp.Y {
							break // Inconsistent tag.
						}
					}
				}

				caseBody := currentBlock.Succs[0]
				chain.cases = append(chain.cases, spmdSwitchCase{
					ifBlock:   currentBlock.Index,
					bodyBlock: caseBody.Index,
					caseMask:  llvm.Value{},
				})

				nextBlock := currentBlock.Succs[1]
				if strings.HasPrefix(nextBlock.Comment, "switch.next") {
					currentBlock = nextBlock
				} else {
					// Last case: Succs[1] is default body or switch.done.
					if strings.HasPrefix(nextBlock.Comment, "switch.body") {
						chain.defaultBody = nextBlock.Index
					} else {
						chain.doneBlock = nextBlock.Index
					}
					break
				}
			} else if _, ok := lastInstr.(*ssa.Jump); ok && strings.HasPrefix(currentBlock.Comment, "switch.next") {
				// A switch.next block ending with Jump is the default/fallthrough path.
				// The block itself contains default case code; its successor is switch.done.
				if len(currentBlock.Succs) == 1 {
					chain.defaultBody = currentBlock.Index
					chain.doneBlock = currentBlock.Succs[0].Index
				}
				break
			} else {
				break
			}
		}

		// Check if the switch is varying. A switch is varying if either:
		// (a) the tag value has SPMDType, OR
		// (b) any comparison operand has SPMDType (go/ssa may type the tag as
		//     a plain type while case constants carry SPMDType).
		if chain.tagValue == nil {
			continue
		}
		isVarying := false
		if _, ok := chain.tagValue.Type().(*types.SPMDType); ok {
			isVarying = true
		} else {
			for _, c := range chain.cases {
				ifBlock := fn.Blocks[c.ifBlock]
				ifInstr := ifBlock.Instrs[len(ifBlock.Instrs)-1].(*ssa.If)
				if binOp, ok := ifInstr.Cond.(*ssa.BinOp); ok {
					if _, ok := binOp.X.Type().(*types.SPMDType); ok {
						isVarying = true
						break
					}
					if _, ok := binOp.Y.Type().(*types.SPMDType); ok {
						isVarying = true
						break
					}
				}
			}
		}
		if !isVarying {
			continue
		}

		// Find switch.done by looking at where body blocks jump.
		if chain.doneBlock == -1 {
			for _, caseInfo := range chain.cases {
				bodyBlock := fn.Blocks[caseInfo.bodyBlock]
				if len(bodyBlock.Succs) != 1 {
					chain.doneBlock = -1
					break
				}
				target := bodyBlock.Succs[0].Index
				if chain.doneBlock == -1 {
					chain.doneBlock = target
				} else if chain.doneBlock != target {
					chain.doneBlock = -1
					break
				}
			}
		}
		// Validate default body jumps to the same done block.
		if chain.doneBlock != -1 && chain.defaultBody != -1 {
			defBlock := fn.Blocks[chain.defaultBody]
			if len(defBlock.Succs) != 1 || defBlock.Succs[0].Index != chain.doneBlock {
				chain.doneBlock = -1
			}
		}

		// Register validated chain.
		if len(chain.cases) == 0 || chain.doneBlock == -1 {
			continue
		}
		chainIdx := len(b.spmdSwitchChains)
		b.spmdSwitchChains = append(b.spmdSwitchChains, chain)

		for i := range chain.cases {
			b.spmdSwitchIfBlocks[chain.cases[i].ifBlock] = chainIdx
			b.spmdSwitchBodyBlocks[chain.cases[i].bodyBlock] = chainIdx
		}
		if chain.defaultBody != -1 {
			b.spmdSwitchBodyBlocks[chain.defaultBody] = chainIdx
		}
	}
}

// spmdAnalyzeVaryingIf analyzes a varying if/else construct at ifBlock and populates
// spmdVaryingIfs, spmdThenExitRedirects, and spmdMergeSelects maps. This is the
// analysis-only version called during pre-detection. The condition LLVM value will
// be filled in later when the If instruction is actually compiled.
func (b *builder) spmdAnalyzeVaryingIf(ifBlock *ssa.BasicBlock) {
	// Check if this ifBlock heads a condition chain (&&/||).
	// If so, use the chain's actual then/else targets instead of the
	// block's direct successors (which point to inner chain blocks).
	thenEntry := ifBlock.Succs[0]
	elseEntry := ifBlock.Succs[1]
	if chain, ok := b.spmdCondChains[ifBlock.Index]; ok {
		thenEntry = b.fn.Blocks[chain.thenTarget]
		elseEntry = b.fn.Blocks[chain.elseTarget]
	}

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

	// For chains without else, record which preds carry the default value.
	// In a chain, the outer block and all inner blocks can jump to the merge/else target.
	if chain, ok := b.spmdCondChains[ifBlock.Index]; ok && !hasElse {
		info.chainDefaultPreds = []int{chain.outerIfBlock}
		info.chainDefaultPreds = append(info.chainDefaultPreds, chain.innerBlocks...)
	}

	// Register the varying if.
	b.spmdVaryingIfs[ifBlock.Index] = info
	b.spmdMergeSelects[merge.Index] = info

	if hasElse {
		// For if-with-else: record that then-exit blocks need to be redirected.
		// For chains, the redirect must be pre-registered because DomPreorder may
		// visit the then-exit before the last inner block compiles. The LLVM block
		// entries are pre-allocated, so we can register them now.
		thenExits := b.spmdFindThenExits(thenEntry, merge)
		elseLLVMBlock := b.blockInfo[elseEntry.Index].entry
		for _, exitBlock := range thenExits {
			b.spmdThenExitRedirects[exitBlock.Index] = elseLLVMBlock
		}
	}

	// Record mask transitions for block-level mask stack management.
	// For || (LOR) chains, the then-body is visited by DomPreorder BEFORE the
	// last inner block, so the combined condition isn't available yet. Skip
	// pushThen/swapElse for LOR chains — those rely on the merge select for
	// correctness (value-only patterns) or need deferred handling (stores).
	// For && (LAND) chains, the then-body IS dominated by the last inner block,
	// so the condition will be available before the then-body is compiled.
	isLORChain := false
	if b.spmdCondChains != nil {
		if chain, ok := b.spmdCondChains[ifBlock.Index]; ok {
			isLORChain = (chain.op == token.LOR)
		}
	}
	if b.spmdMaskTransitions != nil {
		if !isLORChain {
			// Condition will be filled in during compilation.
			b.spmdMaskTransitions[thenEntry.Index] = &spmdMaskTransition{kind: "pushThen", cond: llvm.Value{}}
			if hasElse {
				b.spmdMaskTransitions[elseEntry.Index] = &spmdMaskTransition{kind: "swapElse", cond: llvm.Value{}}
			}
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

// spmdTryGetValue is like getValue but returns (zero, false) instead of panicking
// when a local SSA value hasn't been compiled yet. DomPreorder visits blocks in
// dominance order, so a merge block can be visited before some of its predecessor
// blocks are compiled. spmdCreateMergeSelect calls this to detect that situation
// and switch to the deferred-select path instead of emitting a value immediately.
func (b *builder) spmdTryGetValue(expr ssa.Value) (llvm.Value, bool) {
	// Check overrides first (same as getValue).
	if b.spmdValueOverride != nil {
		if override, ok := b.spmdValueOverride[expr]; ok {
			return override, true
		}
	}
	switch expr := expr.(type) {
	case *ssa.Const:
		file := b.program.Fset.File(b.fn.Pos())
		pos := token.NoPos
		if file != nil {
			pos = file.Pos(0)
		}
		return b.createConst(expr, pos), true
	case *ssa.Function:
		// Functions are compiled separately; returning false here is conservative
		// (getFunction could provide the value), but function-valued phi edges at
		// varying merge blocks are extremely rare. The deferred path handles it.
		return llvm.Value{}, false
	case *ssa.Global:
		value := b.getGlobal(expr)
		if value.IsNil() {
			return llvm.Value{}, false
		}
		return value, true
	default:
		if value, ok := b.locals[expr]; ok {
			return value, true
		}
		return llvm.Value{}, false
	}
}

// spmdCreateMergeSelect converts a phi at a merge block into a select instruction
// when the phi results from a varying if/else.
//
// After CFG linearization the merge block has fewer LLVM predecessors than SSA
// predecessors:
//   - if-without-else: ifBlock no longer jumps to merge; only thenExit does.
//   - if-with-else:    thenExit no longer jumps to merge; only elseExit does.
//
// When both edge values are already available, the select is emitted immediately
// and the function returns (select, true).  When one edge value has not been
// compiled yet (DomPreorder visited the merge before that predecessor), the
// function creates a plain LLVM phi with a single incoming edge and registers a
// spmdMergePhiOverride so that phi resolution will emit the select later.
// This deferred path is the same mechanism used by spmdCreateMultiPredMergeSelect.
//
// Returns (zero, false) only when this phi is not a varying merge phi at all.
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
		// Chain without-else: merge has >2 SSA edges (outer + inner + then).
		if len(phi.Edges) > 2 && !info.hasElse && len(info.chainDefaultPreds) > 0 {
			return b.spmdCreateChainMergeSelect(phi, info)
		}
		return llvm.Value{}, false
	}

	block := phi.Block()
	preds := block.Preds

	// Classify each edge as "then" or "else/default" and try to obtain its value.
	// thenIdx is the edge from the then-branch; elseIdx is the surviving LLVM pred.
	//
	// if-without-else: ifBlock pred is "else" (pre-if default value), thenExit is "then".
	//   After linearization only thenExit remains as an LLVM pred; ifBlock is redirected.
	//   → skipEdgeIdx = ifBlock edge, llvmBlock = thenExit.exit
	//
	// if-with-else: both preds come from branches; thenExit was redirected to elseEntry.
	//   After linearization only elseExit remains as an LLVM pred; thenExit is redirected.
	//   → skipEdgeIdx = thenExit edge, llvmBlock = elseExit.exit
	var thenIdx, elseIdx int
	if !info.hasElse {
		// If-without-else: the pred matching ifBlockIndex is "else" (default).
		// For condition chains (&&/||), the outer ifBlock isn't a direct pred —
		// the last inner block (cond.false for ||) is. Use chainDefaultPreds
		// if available to identify default edges.
		defaultPredSet := make(map[int]bool)
		if len(info.chainDefaultPreds) > 0 {
			for _, idx := range info.chainDefaultPreds {
				defaultPredSet[idx] = true
			}
		} else {
			defaultPredSet[info.ifBlockIndex] = true
		}
		for i := 0; i < len(preds); i++ {
			if defaultPredSet[preds[i].Index] {
				elseIdx = i
			} else {
				thenIdx = i
			}
		}
	} else {
		// If-with-else: classify by reachability from thenEntry.
		thenEntry := b.fn.Blocks[info.thenEntryIndex]
		for i := 0; i < len(preds); i++ {
			if b.spmdIsReachableFrom(thenEntry, preds[i], block) {
				thenIdx = i
			} else {
				elseIdx = i
			}
		}
	}

	thenVal, thenOK := b.spmdTryGetValue(phi.Edges[thenIdx])
	elseVal, elseOK := b.spmdTryGetValue(phi.Edges[elseIdx])

	// For condition chains, the condition may not yet be filled in (DomPreorder
	// visits merge before last inner block). Check info.cond is valid before
	// attempting immediate select; otherwise fall through to deferred path.
	condReady := !info.cond.IsNil()
	if thenOK && elseOK && condReady {
		// Both values are ready: emit the select immediately.
		thenVal, elseVal = b.spmdBroadcastMatch(thenVal, elseVal)
		thenIsVec := thenVal.Type().TypeKind() == llvm.VectorTypeKind
		elseIsVec := elseVal.Type().TypeKind() == llvm.VectorTypeKind
		if thenIsVec || elseIsVec {
			// At least one operand is a vector → vector masked select.
			// On WASM the mask is <N x i32>, so use spmdMaskSelect instead of
			// CreateSelect (which requires an <N x i1> condition).
			return b.spmdMaskSelect(info.cond, thenVal, elseVal), true
		}
		// Both are scalars → reduce condition to scalar boolean and use scalar select.
		// Scalar phis at a varying merge represent uniform values. We use any-true
		// reduction: if any lane took the then-branch, use the then-value. This is
		// safe because the SPMD type checker forbids varying-dependent mutation of
		// uniform variables, so both edges carry the same value in practice.
		scalarCond := b.spmdVectorAnyTrue(info.cond)
		return b.CreateSelect(scalarCond, thenVal, elseVal, ""), true
	}

	// One or both values are not yet available. Defer select creation to phi
	// resolution, just like spmdCreateMultiPredMergeSelect does.
	//
	// After linearization the merge block has exactly one LLVM predecessor:
	//   if-without-else → thenExit (ifBlock was redirected away)
	//   if-with-else    → elseExit (thenExit was redirected to elseEntry)
	// Create a single-incoming phi now; the override fills in the select later.
	phiType := b.getLLVMType(phi.Type())
	llvmPhi := b.CreatePHI(phiType, "")
	b.phis = append(b.phis, phiNode{phi, llvmPhi})

	// Determine which SSA edge to skip (the one whose LLVM predecessor was redirected away).
	// The surviving LLVM exit block is read from blockInfo at phi resolution time (not here)
	// because createRuntimeAssert may insert bounds-check blocks after this point.
	var skipEdgeIdx int
	if !info.hasElse {
		skipEdgeIdx = elseIdx // ifBlock edge is redirected away; thenExit is the only LLVM pred.
	} else {
		skipEdgeIdx = thenIdx // thenExit was redirected to elseEntry; elseExit is the only LLVM pred.
	}

	b.spmdMergePhiOverrides[phi] = spmdMergePhiOverride{
		skipEdgeIdx: skipEdgeIdx,
		thenEdgeIdx: thenIdx,
		elseEdgeIdx: elseIdx,
		info:        info,
	}

	return llvmPhi, true
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

	// Record override: during phi resolution, skip the then-edge (it has no LLVM
	// predecessor after linearization) and replace the else-edge with
	// select(cond, thenVal, elseVal). The else-block's LLVM exit is read from
	// blockInfo at resolution time (not here) to handle bounds-check block insertion.
	b.spmdMergePhiOverrides[phi] = spmdMergePhiOverride{
		skipEdgeIdx: thenIdx,
		thenEdgeIdx: thenIdx,
		elseEdgeIdx: elseIdx,
		info:        info,
	}

	return llvmPhi, true
}

// spmdCreateChainMergeSelect handles phis at merge blocks of condition chains
// without else. These phis have >2 SSA edges because each block in the chain
// (outer + inner blocks) can jump to the merge. After CFG linearization,
// only the then-exit block remains as an LLVM predecessor.
//
// Edge classification:
//   - "default" edges: from outer ifBlock and all inner chain blocks — these
//     carry the pre-if default value (the phi had this value before the if)
//   - "then" edge: from the then-body exit — carries the value set in the body
//
// The select uses the combined chain condition (e.g., a AND b for &&).
func (b *builder) spmdCreateChainMergeSelect(phi *ssa.Phi, info *spmdVaryingIf) (llvm.Value, bool) {
	block := phi.Block()
	preds := block.Preds

	// Build default-pred set from chainDefaultPreds.
	defaultPredSet := make(map[int]bool)
	for _, predIdx := range info.chainDefaultPreds {
		defaultPredSet[predIdx] = true
	}

	// Find then-edge (not a default pred) and any default edge.
	thenIdx := -1
	defaultIdx := -1
	for i, pred := range preds {
		if defaultPredSet[pred.Index] {
			if defaultIdx < 0 {
				defaultIdx = i
			}
		} else {
			if thenIdx < 0 {
				thenIdx = i
			}
		}
	}

	if thenIdx < 0 || defaultIdx < 0 {
		return llvm.Value{}, false
	}

	thenVal, thenOK := b.spmdTryGetValue(phi.Edges[thenIdx])
	defaultVal, defaultOK := b.spmdTryGetValue(phi.Edges[defaultIdx])

	if thenOK && defaultOK {
		// Both values ready: emit select immediately.
		thenVal, defaultVal = b.spmdBroadcastMatch(thenVal, defaultVal)
		thenIsVec := thenVal.Type().TypeKind() == llvm.VectorTypeKind
		defaultIsVec := defaultVal.Type().TypeKind() == llvm.VectorTypeKind
		if thenIsVec || defaultIsVec {
			return b.spmdMaskSelect(info.cond, thenVal, defaultVal), true
		}
		scalarCond := b.spmdVectorAnyTrue(info.cond)
		return b.CreateSelect(scalarCond, thenVal, defaultVal, ""), true
	}

	// Deferred: create phi and register override.
	phiType := b.getLLVMType(phi.Type())
	llvmPhi := b.CreatePHI(phiType, "")
	b.phis = append(b.phis, phiNode{phi, llvmPhi})

	// After linearization only the then-exit is the LLVM pred.
	// All default edges (outer, inner blocks) are linearized away.
	// Find skipEdgeIdxs = all default edges, surviving = then-exit edge.
	var skipEdgeIdxs []int
	for i, pred := range preds {
		if defaultPredSet[pred.Index] {
			skipEdgeIdxs = append(skipEdgeIdxs, i)
		}
	}

	b.spmdMergePhiOverrides[phi] = spmdMergePhiOverride{
		skipEdgeIdx:  skipEdgeIdxs[0], // first default edge (backwards compat)
		skipEdgeIdxs: skipEdgeIdxs,    // all default edges
		thenEdgeIdx:  thenIdx,
		elseEdgeIdx:  defaultIdx,
		info:         info,
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

// spmdMaskElemType returns the LLVM element type for SPMD mask vectors.
// On WASM targets, the mask element type is sized to keep the mask in a single
// 128-bit v128 register: i32 for 4 lanes, i16 for 8, i8 for 16.
// On other targets this is always i1 (native LLVM boolean vector element).
func (c *compilerContext) spmdMaskElemType(laneCount int) llvm.Type {
	if c.spmdIsWASM() {
		return c.ctx.IntType(128 / laneCount) // 4→i32, 8→i16, 16→i8
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
			return llvm.VectorType(c.spmdMaskElemType(laneCount), laneCount)
		}
	}
	return llvm.Type{} // No varying parameters
}

// spmdWrapMask sign-extends an <N x i1> comparison result to the WASM mask type
// (e.g., <4 x i32> for 4 lanes, <8 x i16> for 8 lanes, <16 x i8> for 16 lanes).
// On non-WASM targets this is a no-op. LLVM's WASM backend folds sext(cmp)
// into a single WASM comparison instruction, so there is no runtime cost.
func (b *builder) spmdWrapMask(cmp llvm.Value, laneCount int) llvm.Value {
	if !b.spmdIsWASM() {
		return cmp
	}
	maskType := llvm.VectorType(b.spmdMaskElemType(laneCount), laneCount)
	return b.CreateSExt(cmp, maskType, "")
}

// spmdUnwrapMaskForIntrinsic truncates the WASM mask (e.g., <N x i32>, <N x i16>,
// or <N x i8>) back to <N x i1> for LLVM masked memory intrinsics
// (masked.load, masked.store, masked.gather, masked.scatter).
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
// On WASM targets the mask element type matches the data width to fit in a
// single v128 register (i32 for 4 lanes, i16 for 8, i8 for 16). The select
// uses bitwise AND/OR: result = (mask & trueVal) | (~mask & falseVal).
// Float vector types are bitcast to the matching integer type for the bitwise ops.
//
// The bitwise path is only valid when mask and data have the same total bit
// width (e.g., <16 x i8> mask with <16 x i8> data, or <4 x i32> mask with
// <4 x f32> data). For mismatched widths we fall back to truncating the mask
// to <N x i1> and using LLVM's native CreateSelect.
func (b *builder) spmdMaskSelect(mask, trueVal, falseVal llvm.Value) llvm.Value {
	if !b.spmdIsWASM() {
		return b.CreateSelect(mask, trueVal, falseVal, "")
	}

	valType := trueVal.Type()
	maskType := mask.Type() // e.g., <4 x i32>, <8 x i16>, or <16 x i8>

	// Efficient bitwise select only when mask and data have equal total bit width
	// (e.g., <16 x i8> mask with <16 x i8> data, <4 x i32> mask with <4 x f32> data).
	// For mismatched widths, fall back to trunc+CreateSelect which LLVM handles correctly.
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

	case strings.HasPrefix(name, "lanes.RotateWithin["):
		return b.createRotateWithin(instr, name)

	case strings.HasPrefix(name, "lanes.ShiftLeftWithin["):
		return b.createShiftLeftWithin(instr, name)

	case strings.HasPrefix(name, "lanes.ShiftRightWithin["):
		return b.createShiftRightWithin(instr, name)

	case strings.HasPrefix(name, "lanes.SwizzleWithin["):
		// SwizzleWithin requires variable shuffle indices which are not supported
		// as constant shufflevector masks on all LLVM targets. Deferred to a future
		// phase that can generate extractelement/insertelement sequences.
		return llvm.Value{}, b.makeError(getPos(instr), "lanes.SwizzleWithin not yet implemented")

	default:
		return llvm.Value{}, b.makeError(getPos(instr), "unsupported lanes builtin: "+name)
	}
}

// spmdExtractIntConst extracts a compile-time integer constant from an SSA value.
// Returns the int64 value and true on success, or 0 and false if not a constant.
func spmdExtractIntConst(v ssa.Value) (int64, bool) {
	c, ok := v.(*ssa.Const)
	if !ok {
		return 0, false
	}
	return c.Int64(), true
}

// spmdShuffleConst builds an LLVM <N x i32> constant vector from a slice of uint64 indices.
// This is the mask operand for CreateShuffleVector.
func (c *compilerContext) spmdShuffleConst(indices []uint64) llvm.Value {
	elts := make([]llvm.Value, len(indices))
	for i, idx := range indices {
		elts[i] = llvm.ConstInt(c.ctx.Int32Type(), idx, false)
	}
	return llvm.ConstVector(elts, false)
}

// createRotateWithin rotates values within independent groups of groupSize lanes.
//
// For a vector of totalLanes elements divided into groups of groupSize, each group
// is rotated independently by offset positions. Negative offset rotates right.
//
// Example: RotateWithin(<0,1,2,3,4,5,6,7>, offset=1, groupSize=4)
// => <1,2,3,0, 5,6,7,4>  (each group of 4 rotated left by 1)
func (b *builder) createRotateWithin(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	pos := getPos(instr)
	value := b.getValue(instr.Args[0], pos)

	offset, ok := spmdExtractIntConst(instr.Args[1])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.RotateWithin: offset must be a compile-time constant")
	}
	groupSize, ok := spmdExtractIntConst(instr.Args[2])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.RotateWithin: groupSize must be a compile-time constant")
	}

	vecType := value.Type()
	totalLanes := vecType.VectorSize()
	gs := int(groupSize)

	if gs <= 0 || totalLanes%gs != 0 {
		return llvm.Value{}, b.makeError(pos, "lanes.RotateWithin: groupSize must evenly divide lane count")
	}

	mask := spmdRotateWithinMask(totalLanes, gs, int(offset))
	shuffleMask := b.spmdShuffleConst(mask)
	return b.CreateShuffleVector(value, llvm.Undef(vecType), shuffleMask, "rotatewithin"), nil
}

// spmdRotateWithinMask computes the shufflevector index mask for RotateWithin.
// Each group of groupSize elements is rotated by offset positions (positive = left).
func spmdRotateWithinMask(totalLanes, groupSize, offset int) []uint64 {
	mask := make([]uint64, totalLanes)
	for i := 0; i < totalLanes; i++ {
		group := i / groupSize
		lane := i % groupSize
		// Positive offset rotates left: lane i gets value from lane (i+offset) % groupSize.
		// The modulo handles wrap-around and negative offsets.
		src := ((lane+offset)%groupSize + groupSize) % groupSize
		mask[i] = uint64(group*groupSize + src)
	}
	return mask
}

// createShiftLeftWithin shifts values left within independent groups of groupSize lanes.
//
// Element at position i within a group gets the value from position i+amount.
// Positions shifted in from the left (i+amount >= groupSize) become zero.
//
// Example: ShiftLeftWithin(<0,1,2,3,4,5,6,7>, amount=1, groupSize=4)
// => <1,2,3,0, 5,6,7,0>  (each group of 4 shifted left by 1, new elements = 0)
func (b *builder) createShiftLeftWithin(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	pos := getPos(instr)
	value := b.getValue(instr.Args[0], pos)

	amount, ok := spmdExtractIntConst(instr.Args[1])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.ShiftLeftWithin: amount must be a compile-time constant")
	}
	groupSize, ok := spmdExtractIntConst(instr.Args[2])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.ShiftLeftWithin: groupSize must be a compile-time constant")
	}

	vecType := value.Type()
	totalLanes := vecType.VectorSize()
	gs := int(groupSize)
	amt := int(amount)

	if gs <= 0 || totalLanes%gs != 0 {
		return llvm.Value{}, b.makeError(pos, "lanes.ShiftLeftWithin: groupSize must evenly divide lane count")
	}

	mask, zeroMask := spmdShiftLeftWithinMask(totalLanes, gs, amt)
	shuffleMask := b.spmdShuffleConst(mask)

	// Concatenate value with itself so out-of-range indices map to the second
	// copy (which we will then select away with a zero). LLVM treats indices >=
	// totalLanes in a two-operand shufflevector as elements of the second
	// operand, so passing Undef leaves those lanes undefined; we AND them to
	// zero via a select instead.
	shuffled := b.CreateShuffleVector(value, llvm.Undef(vecType), shuffleMask, "shiftleftwithin")

	// Zero out the lanes that were shifted beyond the group boundary.
	if len(zeroMask) > 0 {
		zero := llvm.ConstNull(vecType)
		// Build a per-lane boolean selector: true => keep shuffled, false => zero.
		selectorElts := make([]llvm.Value, totalLanes)
		allTrue := true
		for i := 0; i < totalLanes; i++ {
			keep := !zeroMask[i]
			if !keep {
				allTrue = false
			}
			selectorElts[i] = llvm.ConstInt(b.ctx.Int1Type(), boolToUint64(keep), false)
		}
		if !allTrue {
			selector := llvm.ConstVector(selectorElts, false)
			shuffled = b.CreateSelect(selector, shuffled, zero, "shiftleftwithin.zero")
		}
	}
	return shuffled, nil
}

// spmdShiftLeftWithinMask computes the shufflevector index mask for ShiftLeftWithin.
// Returns the mask and a per-lane boolean slice indicating which lanes should be zeroed.
// Indices that fall outside the group are set to 0 (they'll be zeroed by the caller via select).
func spmdShiftLeftWithinMask(totalLanes, groupSize, amount int) ([]uint64, []bool) {
	mask := make([]uint64, totalLanes)
	zero := make([]bool, totalLanes)
	for i := 0; i < totalLanes; i++ {
		group := i / groupSize
		lane := i % groupSize
		src := lane + amount
		if src >= groupSize {
			// Shifted out of the group; the select will zero this lane.
			mask[i] = 0
			zero[i] = true
		} else {
			mask[i] = uint64(group*groupSize + src)
		}
	}
	return mask, zero
}

// createShiftRightWithin shifts values right within independent groups of groupSize lanes.
//
// Element at position i within a group gets the value from position i-amount.
// Positions shifted in from the right (i-amount < 0) become zero.
//
// Example: ShiftRightWithin(<0,1,2,3,4,5,6,7>, amount=1, groupSize=4)
// => <0,0,1,2, 0,4,5,6>  (each group of 4 shifted right by 1, new elements = 0)
func (b *builder) createShiftRightWithin(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	pos := getPos(instr)
	value := b.getValue(instr.Args[0], pos)

	amount, ok := spmdExtractIntConst(instr.Args[1])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.ShiftRightWithin: amount must be a compile-time constant")
	}
	groupSize, ok := spmdExtractIntConst(instr.Args[2])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.ShiftRightWithin: groupSize must be a compile-time constant")
	}

	vecType := value.Type()
	totalLanes := vecType.VectorSize()
	gs := int(groupSize)
	amt := int(amount)

	if gs <= 0 || totalLanes%gs != 0 {
		return llvm.Value{}, b.makeError(pos, "lanes.ShiftRightWithin: groupSize must evenly divide lane count")
	}

	mask, zeroMask := spmdShiftRightWithinMask(totalLanes, gs, amt)
	shuffleMask := b.spmdShuffleConst(mask)

	shuffled := b.CreateShuffleVector(value, llvm.Undef(vecType), shuffleMask, "shiftrightwithin")

	// Zero out lanes that were shifted beyond the group boundary.
	if len(zeroMask) > 0 {
		zero := llvm.ConstNull(vecType)
		selectorElts := make([]llvm.Value, totalLanes)
		allTrue := true
		for i := 0; i < totalLanes; i++ {
			keep := !zeroMask[i]
			if !keep {
				allTrue = false
			}
			selectorElts[i] = llvm.ConstInt(b.ctx.Int1Type(), boolToUint64(keep), false)
		}
		if !allTrue {
			selector := llvm.ConstVector(selectorElts, false)
			shuffled = b.CreateSelect(selector, shuffled, zero, "shiftrightwithin.zero")
		}
	}
	return shuffled, nil
}

// spmdShiftRightWithinMask computes the shufflevector index mask for ShiftRightWithin.
// Returns the mask and a per-lane boolean slice indicating which lanes should be zeroed.
func spmdShiftRightWithinMask(totalLanes, groupSize, amount int) ([]uint64, []bool) {
	mask := make([]uint64, totalLanes)
	zero := make([]bool, totalLanes)
	for i := 0; i < totalLanes; i++ {
		group := i / groupSize
		lane := i % groupSize
		src := lane - amount
		if src < 0 {
			// Shifted out of the group; the select will zero this lane.
			mask[i] = 0
			zero[i] = true
		} else {
			mask[i] = uint64(group*groupSize + src)
		}
	}
	return mask, zero
}

// boolToUint64 converts a bool to 0 or 1 for use in LLVM constant construction.
func boolToUint64(b bool) uint64 {
	if b {
		return 1
	}
	return 0
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
	phi         *ssa.Phi   // the phi at rangeint.done
	alloca      llvm.Value // alloca for accumulated break result
	breakEdge   int        // index into phi.Edges for the break edge
	breakVal    ssa.Value  // SSA value on break edge
	defaultVal  ssa.Value  // SSA value on non-break edge (entry or if.done)
	defaultEdge int        // index of the default edge
}

// spmdForLoopInfo tracks a regular for-range loop inside an SPMD function body.
type spmdForLoopInfo struct {
	loopBlockIndex  int               // rangeint.loop block index (or bodyBlockIndex for merged pattern)
	bodyBlockIndex  int               // rangeint.body block index
	doneBlockIndex  int               // rangeint.done block index
	breakMaskAlloca llvm.Value        // alloca for <N x i1> break mask (persists across iterations)
	laneCount       int               // SIMD lane count
	breakResults    []spmdBreakResult // phis at done block with break values
	earlyExitBlocks []llvm.BasicBlock // blocks that jump to done on all-lanes-broken
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

// spmdAnalyzeShiftedIndex checks if an SSA index value is a right-shifted
// contiguous expression: (base + iter) >> const_k, where k > 0. This pattern
// means adjacent lanes access overlapping elements (e.g., src[i>>1] loads
// only laneCount>>k unique values). Returns info for load+shuffle optimization.
func (b *builder) spmdAnalyzeShiftedIndex(index ssa.Value) (*spmdShiftedLoadInfo, bool) {
	binop, ok := index.(*ssa.BinOp)
	if !ok || binop.Op != token.SHR {
		return nil, false
	}

	shiftAmt, ok := ssaConstUint64(binop.Y)
	if !ok || shiftAmt == 0 {
		return nil, false
	}

	// Unwrap ChangeType on the shifted operand (e.g., ChangeType(incr) in
	// range-over-slice patterns where the iter is wrapped to SPMDType).
	shiftedOperand := binop.X
	for {
		if ct, ok := shiftedOperand.(*ssa.ChangeType); ok {
			shiftedOperand = ct.X
		} else {
			break
		}
	}

	// The shifted operand (after unwrapping) must be a direct loop iterator.
	// We only accept the direct phi match (not scalar+iter ADD patterns) because
	// the fixed shuffle mask {0,0,1,1,...} assumes the scalar base is always
	// aligned to 2^shiftAmt. This is guaranteed for direct loop iterators
	// (incremented by laneCount >= 2^shiftAmt) but not for arbitrary offsets.
	loop, ok := b.spmdLoopState.activeLoops[shiftedOperand]
	if !ok {
		return nil, false
	}

	laneCount := loop.laneCount

	// Guard: shift amount must be less than the scalar integer width to avoid
	// LLVM poison values from oversized lshr.
	scalarBase := loop.scalarIterVal
	if scalarBase.IsNil() || shiftAmt >= uint64(scalarBase.Type().IntTypeWidth()) {
		return nil, false
	}

	// Build shuffle mask: lane i maps to i >> shiftAmt.
	shuffleMask := make([]int, laneCount)
	for i := 0; i < laneCount; i++ {
		shuffleMask[i] = i >> shiftAmt
	}
	uniqueCount := shuffleMask[laneCount-1] + 1

	// Compute the shifted scalar base pointer index: scalarBase >> shiftAmt.
	shiftConst := llvm.ConstInt(scalarBase.Type(), shiftAmt, false)
	shiftedBase := b.CreateLShr(scalarBase, shiftConst, "shifted.base.idx")

	return &spmdShiftedLoadInfo{
		scalarPtr:   shiftedBase, // will be completed with GEP in IndexAddr handler
		uniqueCount: uniqueCount,
		shuffleMask: shuffleMask,
		loop:        loop,
	}, true
}

// spmdShiftedLoad generates a narrow contiguous load + shufflevector expansion
// for a shifted-index gather pattern. For uniqueCount==1, emits scalar load + splat.
// For uniqueCount < laneCount, emits a narrow vector load + shuffle duplication.
// The execution mask is applied via select on the final expanded result.
func (b *builder) spmdShiftedLoad(info *spmdShiftedLoadInfo, mask llvm.Value) llvm.Value {
	laneCount := info.loop.laneCount
	elemType := info.elemType

	if info.uniqueCount == 1 {
		// Broadcast: load single scalar element, splat to all lanes.
		scalar := b.CreateLoad(elemType, info.scalarPtr, "shifted.scalar")
		vecType := llvm.VectorType(elemType, laneCount)
		result := b.splatScalar(scalar, vecType)
		return b.spmdShiftedApplyMask(result, mask)
	}

	// General case: load uniqueCount elements as a narrow vector, then shuffle.
	loadVecType := llvm.VectorType(elemType, info.uniqueCount)
	loaded := b.CreateLoad(loadVecType, info.scalarPtr, "shifted.narrow")

	// Build the shuffle mask constant vector.
	maskElems := make([]llvm.Value, laneCount)
	i32Type := b.ctx.Int32Type()
	for i := 0; i < laneCount; i++ {
		maskElems[i] = llvm.ConstInt(i32Type, uint64(info.shuffleMask[i]), false)
	}
	shuffleMaskVec := llvm.ConstVector(maskElems, false)

	// shufflevector expands the narrow loaded vector to full lane width.
	undef := llvm.Undef(loadVecType)
	result := b.CreateShuffleVector(loaded, undef, shuffleMaskVec, "shifted.expand")

	return b.spmdShiftedApplyMask(result, mask)
}

// spmdShiftedApplyMask applies the execution mask to a shifted load result.
// When the mask is not all-ones, emits select(mask, result, zeroinitializer).
func (b *builder) spmdShiftedApplyMask(result, mask llvm.Value) llvm.Value {
	if mask.IsNil() {
		return result
	}
	// For non-trivial masks, use select(mask, result, zero).
	// When mask is all-ones this is a no-op that LLVM optimizes away.
	return b.spmdMaskSelect(mask, result, llvm.ConstNull(result.Type()))
}

// spmdDecomposedBinOp handles a BinOp where one operand is a decomposed index
// (scalar base + <N x i8> offset). It applies algebraic decomposition rules to
// produce a new decomposed result or a plain vector result (for comparisons).
//
// Decomposition rules for (base + offset) OP constant:
//   - ADD scalar c:  {base+c, offset}     (new decomposed)
//   - SUB scalar c:  {base-c, offset}     (new decomposed)
//   - SHR const k:   {base>>k, offsetShifted} IF laneCount is multiple of 2^k (new decomposed)
//   - AND const mask:{0, offset & mask}   IF base is aligned (new decomposed for modulo-like ops)
//   - QUO/REM const k: {0, offset % k}    IF laneCount is divisible by k (new decomposed)
//   - EQL/NEQ/LSS/etc: compare offset against trunc(c - base) (plain <N x i1> comparison)
//   - Otherwise:      materialize to <N x i32> (fallback; produces wide vector on WASM)
//
// decompIsLHS is true when the decomposed value is expr.X (left-hand side).
func (b *builder) spmdDecomposedBinOp(expr *ssa.BinOp, decomp *spmdDecomposedIndex, decompIsLHS bool) (llvm.Value, bool) {
	laneCount := decomp.laneCount
	i8Type := b.ctx.Int8Type()

	// Get the scalar (non-decomposed) operand.
	var scalarSSA ssa.Value
	if decompIsLHS {
		scalarSSA = expr.Y
	} else {
		scalarSSA = expr.X
	}

	// Only decompose when the other operand is a scalar constant or uniform value.
	scalarLLVM := b.getValue(scalarSSA, getPos(expr))
	if scalarLLVM.Type().TypeKind() == llvm.VectorTypeKind {
		// The scalar operand was splatted into a vector (e.g., a constant of
		// Varying[int] type creates <4 x i32>). If the SSA value is a constant,
		// extract the element value to use as the scalar for decomposed arithmetic.
		if constSSA, ok := scalarSSA.(*ssa.Const); ok {
			if spmdT, ok := constSSA.Type().(*types.SPMDType); ok && spmdT.IsVarying() {
				scalarConst := ssa.NewConst(constSSA.Value, spmdT.Elem())
				scalarLLVM = b.createConst(scalarConst, getPos(expr))
			} else {
				// Both sides are truly varying; fall through to materialization.
				materialized := b.spmdMaterializeDecomposed(decomp)
				return materialized, false
			}
		} else {
			// Both sides are varying; fall through to materialization below.
			materialized := b.spmdMaterializeDecomposed(decomp)
			return materialized, false
		}
	}

	switch expr.Op {
	case token.ADD:
		if !decompIsLHS {
			// c + (base + offset) = (c + base) + offset
			// Also valid since addition is commutative.
		}
		// (base + offset) + c = (base + c) + offset
		newBase := b.CreateAdd(decomp.scalarBase, scalarLLVM, "spmd.decomp.add")
		if b.spmdDecomposed != nil {
			b.spmdDecomposed[expr] = &spmdDecomposedIndex{
				scalarBase:    newBase,
				varyingOffset: decomp.varyingOffset,
				laneCount:     laneCount,
				loop:          decomp.loop,
			}
		}
		return llvm.Value{}, true // signal: stored in spmdDecomposed

	case token.SUB:
		if !decompIsLHS {
			// c - (base + offset) is not a simple shift; fall through to materialization.
			break
		}
		// (base + offset) - c = (base - c) + offset
		newBase := b.CreateSub(decomp.scalarBase, scalarLLVM, "spmd.decomp.sub")
		if b.spmdDecomposed != nil {
			b.spmdDecomposed[expr] = &spmdDecomposedIndex{
				scalarBase:    newBase,
				varyingOffset: decomp.varyingOffset,
				laneCount:     laneCount,
				loop:          decomp.loop,
			}
		}
		return llvm.Value{}, true // signal: stored in spmdDecomposed

	case token.SHR:
		if !decompIsLHS {
			break
		}
		// (base + offset) >> k where offset = <0,1,...,N-1> and k is a constant.
		// Valid ONLY when the base is a multiple of 2^k (ensured by fromBodyIter),
		// so that (base >> k) + (offset >> k) == (base + offset) >> k with no carry.
		// After an ADD/SUB the base may be misaligned, so we skip this optimization.
		// We compute the new offset as a COMPILE-TIME constant vector to avoid
		// creating <N x i8> vector-shift-by-vector IR, which WASM cannot lower
		// (WASM i8x16.shr_u takes a scalar shift amount, not per-lane).
		if !decomp.fromBodyIter {
			break
		}
		if constVal, ok := scalarSSA.(*ssa.Const); ok {
			if k, ok := constant.Int64Val(constVal.Value); ok && k > 0 {
				power := int64(1) << uint(k)
				if int64(laneCount)%power == 0 {
					// Shift the base scalar.
					newBase := b.CreateAShr(decomp.scalarBase, scalarLLVM, "spmd.decomp.shr.base")
					// Compute the new offset as a constant vector directly.
					// For offset <0,1,...,N-1> >> k: each element i maps to i >> k.
					newOffsetElts := make([]llvm.Value, laneCount)
					for i := 0; i < laneCount; i++ {
						newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(i)>>uint(k), false)
					}
					newOffset := llvm.ConstVector(newOffsetElts, false)
					if b.spmdDecomposed != nil {
						b.spmdDecomposed[expr] = &spmdDecomposedIndex{
							scalarBase:    newBase,
							varyingOffset: newOffset,
							laneCount:     laneCount,
							loop:          decomp.loop,
						}
					}
					return llvm.Value{}, true
				}
			}
		}

	case token.AND:
		if !decompIsLHS {
			break
		}
		// (base + offset) & mask where offset = <0,1,...,N-1> and mask is a constant.
		// Valid ONLY when base & mask == 0 (ensured by fromBodyIter, since base is a
		// multiple of laneCount which is always > mask for mask < laneCount).
		// After an ADD/SUB the base may have low bits set, breaking the identity
		// (base + offset) & mask == (base & mask) + (offset & mask).
		// We compute the new offset as a COMPILE-TIME constant vector to avoid
		// creating runtime <N x i8> AND which could mismatch WASM SIMD lane widths.
		// This handles the common "i & 1" (i.e., "i % 2") pattern.
		if !decomp.fromBodyIter {
			break
		}
		if constVal, ok := scalarSSA.(*ssa.Const); ok {
			if mask, ok := constant.Int64Val(constVal.Value); ok && mask >= 0 && mask < int64(laneCount) {
				// Compute offset & mask as constant vector.
				newOffsetElts := make([]llvm.Value, laneCount)
				for i := 0; i < laneCount; i++ {
					newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(i)&uint64(mask), false)
				}
				newOffset := llvm.ConstVector(newOffsetElts, false)
				// Base contribution: base & mask (scalar).
				baseContrib := b.CreateAnd(decomp.scalarBase, scalarLLVM, "spmd.decomp.and.base")
				if b.spmdDecomposed != nil {
					b.spmdDecomposed[expr] = &spmdDecomposedIndex{
						scalarBase:    baseContrib,
						varyingOffset: newOffset,
						laneCount:     laneCount,
						loop:          decomp.loop,
					}
				}
				return llvm.Value{}, true
			}
		}

	case token.REM:
		if !decompIsLHS {
			break
		}
		// (base + offset) % k where offset = <0,1,...,N-1> and k is a constant.
		// Valid ONLY when base % k == 0 (ensured by fromBodyIter, since base is a
		// multiple of laneCount which is always divisible by k when laneCount%k==0).
		// After an ADD/SUB the base may not be aligned, so the identity
		// (base + offset) % k == (base % k) + (offset % k) may not hold.
		// We compute the new offset as a COMPILE-TIME constant vector to avoid
		// creating <N x i8> remainder IR which WASM SIMD cannot lower (no i8x16.rem).
		// Valid when laneCount is divisible by k.
		if !decomp.fromBodyIter {
			break
		}
		if constVal, ok := scalarSSA.(*ssa.Const); ok {
			if k, ok := constant.Int64Val(constVal.Value); ok && k > 0 && int64(laneCount)%k == 0 {
				// Compute offset % k as constant vector.
				newOffsetElts := make([]llvm.Value, laneCount)
				for i := 0; i < laneCount; i++ {
					newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(i)%uint64(k), false)
				}
				newOffset := llvm.ConstVector(newOffsetElts, false)
				// Base contribution: base % k (scalar).
				baseContrib := b.CreateURem(decomp.scalarBase, scalarLLVM, "spmd.decomp.rem.base")
				if b.spmdDecomposed != nil {
					b.spmdDecomposed[expr] = &spmdDecomposedIndex{
						scalarBase:    baseContrib,
						varyingOffset: newOffset,
						laneCount:     laneCount,
						loop:          decomp.loop,
					}
				}
				return llvm.Value{}, true
			}
		}

	case token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
		// Comparison: (base + offset) CMP c  →  offset CMP (c - base)  [unsigned <N x i8>]
		// The scalar operand is always on the right when decompIsLHS; when !decompIsLHS the
		// expression is  c CMP (base + offset)  and we swap the predicate below.
		//
		// We compute diff = c - base as a signed i32 scalar. Two edge cases require
		// special handling before the clamped i8 comparison:
		//
		//   diff < 0  (c < base):  every lane has base+offset > c (since offset >= 0),
		//     so (base+offset) CMP c should be: LT→all-false, LE→all-false, GT→all-true,
		//     GE→all-true, EQ→all-false, NE→all-true.
		//
		//   diff >= laneCount  (c >= base+laneCount): every lane has base+offset <= c
		//     (since offset <= laneCount-1), so:
		//     LT→all-true, LE→all-true, GT→all-false, GE→all-false, EQ→all-false,
		//     NE→all-true.
		//
		// These edge-case results are for the canonical "(base+offset) CMP c" form.
		// When !decompIsLHS the expression is "c CMP (base+offset)" and we apply the
		// complementary result (true↔false for LT/GT/LE/GE, identity for EQ/NE).

		// diff = c - base (scalar i32) using the decomposed base.
		diff := b.CreateSub(scalarLLVM, decomp.scalarBase, "spmd.decomp.cmp.diff")

		i1VecType := llvm.VectorType(b.ctx.Int1Type(), laneCount)
		allTrue := llvm.ConstAllOnes(i1VecType)
		allFalse := llvm.ConstNull(i1VecType)

		zero32 := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
		lcConst32 := llvm.ConstInt(b.ctx.Int32Type(), uint64(laneCount), false)
		// Splat the scalar edge-case conditions to <N x i1> so they can be used as
		// the condition in vector CreateSelect (LLVM select requires matching shapes).
		diffNeg := b.splatScalar(
			b.CreateICmp(llvm.IntSLT, diff, zero32, "spmd.decomp.cmp.neg"),
			i1VecType)
		diffOver := b.splatScalar(
			b.CreateICmp(llvm.IntSGE, diff, lcConst32, "spmd.decomp.cmp.over"),
			i1VecType)

		// Determine the all-true/all-false overrides for the "(base+offset) CMP c" form.
		// negResult applies when diff < 0 (c < base); overResult applies when diff >= laneCount.
		var negResult, overResult llvm.Value
		switch expr.Op {
		case token.LSS, token.LEQ:
			// (base+offset) </<= c: when c < base → all false; when c >= base+N → all true.
			negResult = allFalse
			overResult = allTrue
		case token.GTR, token.GEQ:
			// (base+offset) >/>= c: when c < base → all true; when c >= base+N → all false.
			negResult = allTrue
			overResult = allFalse
		case token.EQL:
			negResult = allFalse
			overResult = allFalse
		case token.NEQ:
			negResult = allTrue
			overResult = allTrue
		}

		// When the decomposed value is on the RHS (c CMP (base+offset)), the meaning
		// flips for ordering comparisons: LSS↔GTR, LEQ↔GEQ; EQ/NE are symmetric.
		if !decompIsLHS {
			switch expr.Op {
			case token.LSS, token.LEQ:
				// c </<= (base+offset): when c < base → all true; when c >= base+N → all false.
				negResult = allTrue
				overResult = allFalse
			case token.GTR, token.GEQ:
				// c >/>= (base+offset): when c < base → all false; when c >= base+N → all true.
				negResult = allFalse
				overResult = allTrue
				// EQL and NEQ are symmetric; negResult/overResult already set correctly above.
			}
		}

		// Clamp diff to [0, 255] for safe i8 truncation.
		max8 := llvm.ConstInt(b.ctx.Int32Type(), 255, false)
		diffClamped := b.CreateSelect(
			b.CreateICmp(llvm.IntSGT, diff, max8, ""),
			max8, diff, "spmd.decomp.cmp.hi")
		diffClamped = b.CreateSelect(
			b.CreateICmp(llvm.IntSLT, diffClamped, zero32, ""),
			zero32, diffClamped, "spmd.decomp.cmp.lo")

		// Truncate to i8 and splat for the in-range comparison.
		diffI8 := b.CreateTrunc(diffClamped, i8Type, "spmd.decomp.cmp.i8")
		diffVec := b.splatScalar(diffI8, llvm.VectorType(i8Type, laneCount))

		// Choose the comparison predicate for the in-range case.
		// offset is unsigned [0, N-1]; always use unsigned predicates.
		var pred llvm.IntPredicate
		if decompIsLHS {
			// (base + offset) CMP c → offset CMP diff
			switch expr.Op {
			case token.EQL:
				pred = llvm.IntEQ
			case token.NEQ:
				pred = llvm.IntNE
			case token.LSS:
				pred = llvm.IntULT
			case token.LEQ:
				pred = llvm.IntULE
			case token.GTR:
				pred = llvm.IntUGT
			case token.GEQ:
				pred = llvm.IntUGE
			}
		} else {
			// c CMP (base + offset) → diff CMP offset → swap predicate
			switch expr.Op {
			case token.EQL:
				pred = llvm.IntEQ
			case token.NEQ:
				pred = llvm.IntNE
			case token.LSS:
				pred = llvm.IntUGT // c < (base+off) ↔ off > (c-base)
			case token.LEQ:
				pred = llvm.IntUGE
			case token.GTR:
				pred = llvm.IntULT
			case token.GEQ:
				pred = llvm.IntULE
			}
		}
		cmpI1 := b.CreateICmp(pred, decomp.varyingOffset, diffVec, "spmd.decomp.cmp")

		// Apply edge-case overrides: diff < 0 → negResult; diff >= laneCount → overResult.
		result := b.CreateSelect(diffNeg, negResult, cmpI1, "spmd.decomp.cmp.neg.sel")
		result = b.CreateSelect(diffOver, overResult, result, "spmd.decomp.cmp.over.sel")
		return result, false
	}

	// Fallback: materialize to <N x i32> and apply the operation normally.
	// On WASM this produces a 512-bit wide vector; use only when no 128-bit path exists.
	materialized := b.spmdMaterializeDecomposed(decomp)
	return materialized, false
}

// spmdDecomposedIndexAddr handles IndexAddr where the index is a decomposed SPMD value.
// Two sub-cases:
//  1. Identity offset <0,1,...,N-1>: this is contiguous access; delegate to spmdContiguousIndexAddrCore.
//  2. Non-identity offset: build per-lane GEPs from (scalarBase + offset[lane]) for gather/scatter.
func (b *builder) spmdDecomposedIndexAddr(expr *ssa.IndexAddr, decomp *spmdDecomposedIndex) (llvm.Value, error) {
	laneCount := decomp.laneCount

	// Check if the offset is the identity permutation <0,1,...,N-1>.
	// This is true when the decomposed value is the raw bodyIterValue (i.e.,
	// its varyingOffset was initialized by emitSPMDBodyPrologue as the constant
	// lane offset sequence). We detect this by checking if the offset is a
	// constant vector equal to the initial identity.
	isIdentity := false
	identityConst := b.spmdLaneOffsetConst(laneCount, b.ctx.Int8Type())
	if decomp.varyingOffset == identityConst {
		isIdentity = true
	} else {
		// Also check structural equality: the offset may be the same constant
		// but a different Go wrapper. Compare by LLVM string representation.
		if decomp.varyingOffset.IsConstant() {
			isIdentity = (decomp.varyingOffset.String() == identityConst.String())
		}
	}

	if isIdentity {
		// Contiguous access: use the scalar base as the GEP index.
		return b.spmdContiguousIndexAddrCore(expr, decomp.loop, decomp.scalarBase)
	}

	// Non-contiguous access: build per-lane GEPs using (scalarBase + offset[lane]).
	// The offset is <N x i8>, so each lane's byte offset must be zero-extended to uintptr.
	val := b.getValue(expr.X, getPos(expr))

	var bufptr llvm.Value
	var bufType llvm.Type
	var elemType llvm.Type
	switch ptrTyp := expr.X.Type().Underlying().(type) {
	case *types.Pointer:
		typ := ptrTyp.Elem().Underlying()
		switch typ := typ.(type) {
		case *types.Array:
			bufptr = val
			bufType = b.getLLVMType(typ)
			elemType = b.getLLVMType(typ.Elem())
			b.createNilCheck(expr.X, bufptr, "gep")
		default:
			return llvm.Value{}, b.makeError(expr.Pos(), "unsupported decomposed SPMD indexaddr type: "+typ.String())
		}
	case *types.Slice:
		bufptr = b.CreateExtractValue(val, 0, "indexaddr.ptr")
		bufType = b.getLLVMType(ptrTyp.Elem())
		elemType = bufType
	default:
		return llvm.Value{}, b.makeError(expr.Pos(), "unsupported decomposed SPMD indexaddr type: "+ptrTyp.String())
	}

	// Bounds check: verify all lane indices (base + offset[lane]) are in bounds.
	if !b.info.nobounds {
		var buflen llvm.Value
		switch ptrTyp := expr.X.Type().Underlying().(type) {
		case *types.Pointer:
			typ := ptrTyp.Elem().Underlying().(*types.Array)
			buflen = llvm.ConstInt(b.uintptrType, uint64(typ.Len()), false)
		case *types.Slice:
			buflen = b.CreateExtractValue(val, 1, "indexaddr.len")
		}
		if !buflen.IsNil() {
			if decomp.varyingOffset.IsConstant() {
				// OPTIMIZED: constant offset vector → single scalar bounds check.
				// Find the maximum offset element at compile time, then check:
				//   (scalarBase + maxOffset) >= buflen → panic
				// Correctness: maxOffset >= every per-lane offset, so if
				// (base + maxOffset) is in bounds then all lanes are in bounds.
				maxOffset := uint64(0)
				for lane := 0; lane < laneCount; lane++ {
					elem := llvm.ConstExtractElement(decomp.varyingOffset,
						llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false))
					// ZExtValue: offset bytes are unsigned (zero-extended everywhere
					// in the decomposed-index pipeline).
					elemVal := elem.ZExtValue()
					if elemVal > maxOffset {
						maxOffset = elemVal
					}
				}
				maxIdx := b.CreateAdd(decomp.scalarBase,
					llvm.ConstInt(decomp.scalarBase.Type(), maxOffset, false),
					"spmd.bounds.maxidx")
				if maxIdx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
					maxIdx = b.CreateSExt(maxIdx, b.uintptrType, "")
				}
				oob := b.CreateICmp(llvm.IntUGE, maxIdx, buflen, "spmd.bounds.oob")
				b.createRuntimeAssert(oob, "lookup", "lookupPanic")
			} else {
				// FALLBACK: non-constant offset vector, per-lane bounds check.
				anyOOB := llvm.ConstInt(b.ctx.Int1Type(), 0, false)
				for lane := 0; lane < laneCount; lane++ {
					offsetByte := b.CreateExtractElement(decomp.varyingOffset,
						llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
					// Zero-extend offset byte to i32, then add scalar base.
					offsetI32 := b.CreateZExt(offsetByte, b.ctx.Int32Type(), "")
					laneIdx := b.CreateAdd(decomp.scalarBase, offsetI32, "")
					if laneIdx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
						laneIdx = b.CreateSExt(laneIdx, b.uintptrType, "")
					}
					oob := b.CreateICmp(llvm.IntUGE, laneIdx, buflen, "")
					anyOOB = b.CreateOr(anyOOB, oob, "")
				}
				b.createRuntimeAssert(anyOOB, "lookup", "lookupPanic")
			}
		}
	}

	// Build vector of pointers: each lane gets its own GEP from (base + offset[lane]).
	ptrVec := llvm.Undef(llvm.VectorType(bufptr.Type(), laneCount))
	for lane := 0; lane < laneCount; lane++ {
		offsetByte := b.CreateExtractElement(decomp.varyingOffset,
			llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
		offsetI32 := b.CreateZExt(offsetByte, b.ctx.Int32Type(), "")
		laneIdx := b.CreateAdd(decomp.scalarBase, offsetI32, "")
		if laneIdx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
			laneIdx = b.CreateSExt(laneIdx, b.uintptrType, "")
		}
		var gep llvm.Value
		switch expr.X.Type().Underlying().(type) {
		case *types.Pointer:
			gep = b.CreateInBoundsGEP(bufType, bufptr, []llvm.Value{
				llvm.ConstInt(b.ctx.Int32Type(), 0, false),
				laneIdx,
			}, "")
		case *types.Slice:
			gep = b.CreateInBoundsGEP(elemType, bufptr, []llvm.Value{laneIdx}, "")
		}
		ptrVec = b.CreateInsertElement(ptrVec, gep,
			llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
	}
	return ptrVec, nil
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
			maskType := llvm.VectorType(b.spmdMaskElemType(laneCount), laneCount)

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
		maskType := llvm.VectorType(b.spmdMaskElemType(laneCount), laneCount)

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

// spmdCompileSwitchIf compiles a switch.next If instruction. This is the
// compilation-time handler (called from the *ssa.If case) that computes the
// case mask via sequential narrowing and registers mask transitions.
//
// Algorithm (ISPC approach):
//
//	remainingMask = currentMask (for first case) or b.spmdSwitchRemainingMask (subsequent)
//	caseMask = remainingMask & cond
//	remainingMask = remainingMask & ~cond
//
// For the last case, also handles the default body by pushing the remaining mask.
func (b *builder) spmdCompileSwitchIf(block *ssa.BasicBlock, cond llvm.Value, chainIdx int) {
	chain := &b.spmdSwitchChains[chainIdx]

	// Find which case index this block is.
	caseIdx := -1
	for i := range chain.cases {
		if chain.cases[i].ifBlock == block.Index {
			caseIdx = i
			break
		}
	}
	if caseIdx == -1 {
		return
	}

	// Get the remaining mask.
	var remainingMask llvm.Value
	if caseIdx == 0 {
		// First case: use current mask from stack.
		remainingMask = b.spmdCurrentMask()
		if remainingMask.IsNil() {
			// No mask on stack — use all-ones.
			remainingMask = llvm.ConstAllOnes(cond.Type())
		}
	} else {
		// Subsequent case: use the remaining mask from previous case.
		remainingMask = b.spmdSwitchRemainingMask
	}

	// Compute case mask: remainingMask & cond.
	caseMask := b.CreateAnd(remainingMask, cond, "switch.case.mask")
	chain.cases[caseIdx].caseMask = caseMask

	// Update remaining mask: remainingMask & ~cond.
	notCond := b.CreateNot(cond, "")
	b.spmdSwitchRemainingMask = b.CreateAnd(remainingMask, notCond, "switch.remaining")

	// Register mask transition for the body block (push caseMask directly).
	// spmdMaskTransitions is always initialized by createFunction when SPMD is active.
	bodyBlockIdx := chain.cases[caseIdx].bodyBlock
	b.spmdMaskTransitions[bodyBlockIdx] = &spmdMaskTransition{kind: "pushDirect", cond: caseMask}

	// For the last case, also handle the default body (if any).
	if caseIdx == len(chain.cases)-1 && chain.defaultBody != -1 {
		defaultMask := b.spmdSwitchRemainingMask
		b.spmdMaskTransitions[chain.defaultBody] = &spmdMaskTransition{kind: "pushDirect", cond: defaultMask}
	}
}

// spmdSwitchBodyJumpTarget determines the target block for a Jump instruction
// at the end of a switch case body. Returns the next switch.next comparison block,
// the default body, or switch.done, depending on the case position in the chain.
func (b *builder) spmdSwitchBodyJumpTarget(block *ssa.BasicBlock, chain *spmdSwitchChain) llvm.BasicBlock {
	// Find which case this body belongs to.
	for i, c := range chain.cases {
		if c.bodyBlock == block.Index {
			// This body is for case i. Next target:
			if i+1 < len(chain.cases) {
				// Next case comparison block.
				nextIfIdx := chain.cases[i+1].ifBlock
				return b.blockInfo[nextIfIdx].entry
			}
			// Last case: go to default body or switch.done.
			if chain.defaultBody != -1 {
				return b.blockInfo[chain.defaultBody].entry
			}
			return b.blockInfo[chain.doneBlock].entry
		}
	}

	// Default body: go to switch.done.
	if chain.defaultBody == block.Index {
		return b.blockInfo[chain.doneBlock].entry
	}

	// Fallback: should not happen for well-formed switch chains.
	// All body blocks must be either a case body or the default body.
	return b.blockInfo[chain.doneBlock].entry
}

// spmdIsSwitchDoneBlock returns the chain index if blockIdx is a switch.done merge
// block for a detected varying switch chain, or -1 if not.
func (b *builder) spmdIsSwitchDoneBlock(blockIdx int) int {
	for i := range b.spmdSwitchChains {
		if b.spmdSwitchChains[i].doneBlock == blockIdx {
			return i
		}
	}
	return -1
}

// spmdCreateSwitchMergeSelect creates the cascaded select instructions for a
// phi at switch.done. This merges values from all case bodies using the case
// masks computed during switch compilation.
//
// Algorithm:
//
//	result = defaultValue (or zero if no default)
//	for each case i (first to last):
//	  result = select(caseMask[i], caseValue[i], result)
func (b *builder) spmdCreateSwitchMergeSelect(phi *ssa.Phi, chainIdx int) (llvm.Value, bool) {
	chain := &b.spmdSwitchChains[chainIdx]

	// Map each phi edge to its source: case body, default body, or "entry".
	var entryEdgeIdx = -1
	var defaultEdgeIdx = -1
	caseEdges := make(map[int]int) // caseIdx -> edgeIdx

	for i, pred := range phi.Block().Preds {
		if chain.defaultBody != -1 && pred.Index == chain.defaultBody {
			defaultEdgeIdx = i
		} else {
			found := false
			for ci, c := range chain.cases {
				if pred.Index == c.bodyBlock {
					caseEdges[ci] = i
					found = true
					break
				}
			}
			if !found {
				// Check if pred is reachable from a case body (multi-block case bodies).
				for ci, c := range chain.cases {
					bodyBlock := b.fn.Blocks[c.bodyBlock]
					if b.spmdIsReachableFrom(bodyBlock, pred, phi.Block()) {
						caseEdges[ci] = i
						found = true
						break
					}
				}
			}
			if !found {
				entryEdgeIdx = i // This is the pre-switch edge.
			}
		}
	}

	// Build cascaded select. Start with the entry/default value.
	var result llvm.Value
	phiType := b.getLLVMType(phi.Type())

	if defaultEdgeIdx >= 0 {
		result = b.getValue(phi.Edges[defaultEdgeIdx], getPos(phi))
	} else if entryEdgeIdx >= 0 {
		result = b.getValue(phi.Edges[entryEdgeIdx], getPos(phi))
	} else {
		result = llvm.ConstNull(phiType)
	}

	// Apply cascaded selects from first case to last.
	for ci := 0; ci < len(chain.cases); ci++ {
		edgeIdx, ok := caseEdges[ci]
		if !ok {
			continue // Case body doesn't contribute to this phi.
		}
		caseVal := b.getValue(phi.Edges[edgeIdx], getPos(phi))
		caseMask := chain.cases[ci].caseMask
		if caseMask.IsNil() {
			continue
		}

		// Broadcast match if needed.
		caseVal, result = b.spmdBroadcastMatch(caseVal, result)

		if caseVal.Type().TypeKind() == llvm.VectorTypeKind || result.Type().TypeKind() == llvm.VectorTypeKind {
			result = b.spmdMaskSelect(caseMask, caseVal, result)
		} else {
			// Scalar: reduce mask to scalar.
			scalarCond := b.spmdVectorAnyTrue(caseMask)
			result = b.CreateSelect(scalarCond, caseVal, result, "")
		}
	}

	return result, true
}

// spmdVectorIndex handles *ssa.Index when the index is a vector type (varying).
// Dispatches to string or array handlers based on the collection type.
func (b *builder) spmdVectorIndex(expr *ssa.Index, collection, index llvm.Value) (llvm.Value, error) {
	switch expr.X.Type().Underlying().(type) {
	case *types.Basic:
		return b.spmdVectorIndexString(expr, collection, index)
	case *types.Array:
		return b.spmdVectorIndexArray(expr, collection, index)
	default:
		return llvm.Value{}, b.makeError(expr.Pos(), "unsupported SPMD vector index type: "+expr.X.Type().Underlying().String())
	}
}

// spmdVectorIndexString performs per-lane byte extraction from a string using a vector index.
// Each lane extracts: string_ptr[index_lane_i] → byte result per lane.
func (b *builder) spmdVectorIndexString(expr *ssa.Index, collection, index llvm.Value) (llvm.Value, error) {
	laneCount := index.Type().VectorSize()

	// Extract {ptr, len} from string.
	buf := b.CreateExtractValue(collection, 0, "")
	length := b.CreateExtractValue(collection, 1, "len")

	// Pre-compute extended lane indices (used for both bounds check and GEP).
	laneIdxs := make([]llvm.Value, laneCount)
	for lane := 0; lane < laneCount; lane++ {
		idx := b.CreateExtractElement(index, llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
		if idx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
			idx = b.spmdExtendIndex(idx, expr.Index.Type(), b.uintptrType)
		}
		laneIdxs[lane] = idx
	}

	// Aggregate bounds check: OR per-lane OOB flags.
	if !b.info.nobounds {
		canElide := false
		if maxVal, known := spmdIndexMaxValue(expr.Index); known {
			// For constant strings, check if max index is within bounds.
			if constStr, ok := expr.X.(*ssa.Const); ok {
				if constStr.Value != nil && constStr.Value.Kind() == constant.String {
					strLen := uint64(len(constant.StringVal(constStr.Value)))
					if strLen > 0 && maxVal < strLen {
						canElide = true
					}
				}
			}
		}
		if !canElide {
			anyOOB := llvm.ConstInt(b.ctx.Int1Type(), 0, false)
			for lane := 0; lane < laneCount; lane++ {
				oob := b.CreateICmp(llvm.IntUGE, laneIdxs[lane], length, "")
				anyOOB = b.CreateOr(anyOOB, oob, "")
			}
			b.createRuntimeAssert(anyOOB, "lookup", "lookupPanic")
		}
	}

	// SPMD: on WASM, use i8x16.swizzle for const string lookups of <=16 bytes.
	if b.spmdIsWASM() {
		if constVal, ok := expr.X.(*ssa.Const); ok {
			if constVal.Value != nil && constVal.Value.Kind() == constant.String {
				strVal := constant.StringVal(constVal.Value)
				if len(strVal) <= 16 {
					return b.spmdWasmSwizzle([]byte(strVal), index, laneCount), nil
				}
			}
		}
	}

	// Per-lane: GEP, load, insert into result vector.
	bufElemType := b.ctx.Int8Type()
	result := llvm.Undef(llvm.VectorType(bufElemType, laneCount))
	for lane := 0; lane < laneCount; lane++ {
		ptr := b.CreateInBoundsGEP(bufElemType, buf, []llvm.Value{laneIdxs[lane]}, "")
		val := b.CreateLoad(bufElemType, ptr, "")
		result = b.CreateInsertElement(result, val, llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
	}
	return result, nil
}

// spmdVectorIndexArray performs per-lane element extraction from an array using a vector index.
// The array is spilled to an alloca, then each lane does GEP+load.
func (b *builder) spmdVectorIndexArray(expr *ssa.Index, collection, index llvm.Value) (llvm.Value, error) {
	laneCount := index.Type().VectorSize()
	xType := expr.X.Type().Underlying().(*types.Array)

	// Spill array to alloca (can't index a non-constant array in registers).
	arrayType := collection.Type()

	// Reject arrays of varying elements (vector element type + vector index would
	// produce a malformed vector-of-vectors result).
	if arrayType.ElementType().TypeKind() == llvm.VectorTypeKind {
		return llvm.Value{}, b.makeError(expr.Pos(), "SPMD vector index into array of varying elements is not supported")
	}

	alloca, allocaSize := b.createTemporaryAlloca(arrayType, "index.alloca")
	b.CreateStore(collection, alloca)

	// Pre-compute extended lane indices (used for both bounds check and GEP).
	laneIdxs := make([]llvm.Value, laneCount)
	for lane := 0; lane < laneCount; lane++ {
		idx := b.CreateExtractElement(index, llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
		if idx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
			idx = b.spmdExtendIndex(idx, expr.Index.Type(), b.uintptrType)
		}
		laneIdxs[lane] = idx
	}

	// Aggregate bounds check: OR per-lane OOB flags.
	arrayLen := llvm.ConstInt(b.uintptrType, uint64(xType.Len()), false)
	if !b.info.nobounds {
		canElide := false
		if maxVal, known := spmdIndexMaxValue(expr.Index); known {
			if maxVal < uint64(xType.Len()) {
				canElide = true
			}
		}
		if !canElide {
			anyOOB := llvm.ConstInt(b.ctx.Int1Type(), 0, false)
			for lane := 0; lane < laneCount; lane++ {
				oob := b.CreateICmp(llvm.IntUGE, laneIdxs[lane], arrayLen, "")
				anyOOB = b.CreateOr(anyOOB, oob, "")
			}
			b.createRuntimeAssert(anyOOB, "lookup", "lookupPanic")
		}
	}

	// Per-lane: GEP, load, insert into result vector.
	elemType := arrayType.ElementType()
	result := llvm.Undef(llvm.VectorType(elemType, laneCount))
	zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
	for lane := 0; lane < laneCount; lane++ {
		ptr := b.CreateInBoundsGEP(arrayType, alloca, []llvm.Value{zero, laneIdxs[lane]}, "index.gep")
		val := b.CreateLoad(elemType, ptr, "index.load")
		result = b.CreateInsertElement(result, val, llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
	}
	b.emitLifetimeEnd(alloca, allocaSize)
	return result, nil
}

// spmdExtendIndex extends a scalar index extracted from a vector to the target type.
// Determines signed/unsigned from the Go type (unwrapping SPMDType if needed).
func (b *builder) spmdExtendIndex(value llvm.Value, goType types.Type, targetType llvm.Type) llvm.Value {
	if value.Type().IntTypeWidth() >= targetType.IntTypeWidth() {
		return value
	}
	// Strip SPMDType before calling Underlying(), because SPMDType.Underlying()
	// delegates to its element type directly — making a post-Underlying() check unreachable.
	typ := goType
	if spmdType, ok := typ.(*types.SPMDType); ok {
		typ = spmdType.Elem()
	}
	basic, ok := typ.Underlying().(*types.Basic)
	if !ok {
		// Fallback to zero-extend for safety.
		return b.CreateZExt(value, targetType, "")
	}
	if basic.Info()&types.IsUnsigned != 0 {
		return b.CreateZExt(value, targetType, "")
	}
	return b.CreateSExt(value, targetType, "")
}

// spmdWasmSwizzle generates a WASM i8x16.swizzle instruction to look up bytes
// from a constant table using a vector of indices. The table is zero-padded to 16 bytes.
// For lane counts < 16, the result is narrowed using a shuffle extract.
func (b *builder) spmdWasmSwizzle(tableBytes []byte, index llvm.Value, laneCount int) llvm.Value {
	i8Type := b.ctx.Int8Type()
	v16i8 := llvm.VectorType(i8Type, 16)

	// Build <16 x i8> constant from table bytes, zero-pad if < 16.
	tableElems := make([]llvm.Value, 16)
	for i := 0; i < 16; i++ {
		if i < len(tableBytes) {
			tableElems[i] = llvm.ConstInt(i8Type, uint64(tableBytes[i]), false)
		} else {
			tableElems[i] = llvm.ConstInt(i8Type, 0, false)
		}
	}
	tableVec := llvm.ConstVector(tableElems, false)

	// Prepare index as <16 x i8>.
	idxVec := b.spmdSwizzlePrepareIndex(index, laneCount)

	// Call @llvm.wasm.swizzle(<16 x i8>, <16 x i8>).
	// Note: the intrinsic has no type suffix — always @llvm.wasm.swizzle.
	intrinsicName := "llvm.wasm.swizzle"
	fnType := llvm.FunctionType(v16i8, []llvm.Type{v16i8, v16i8}, false)
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	result := b.createCall(fnType, fn, []llvm.Value{tableVec, idxVec}, "spmd.swizzle")

	// If laneCount < 16, extract the first laneCount lanes.
	if laneCount < 16 {
		maskElems := make([]llvm.Value, laneCount)
		for i := 0; i < laneCount; i++ {
			maskElems[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
		}
		maskVec := llvm.ConstVector(maskElems, false)
		result = b.CreateShuffleVector(result, llvm.Undef(v16i8), maskVec, "")
	}

	return result
}

// spmdSwizzlePrepareIndex converts a vector index to <16 x i8> for i8x16.swizzle.
// If the index element type is wider than i8 (e.g., <4 x i32>), it is truncated to
// <N x i8>. If N < 16, the vector is padded to 16 lanes using a shuffle; the padding
// lanes receive undef values from the second shuffle operand, which are safe because
// spmdWasmSwizzle extracts only the first laneCount lanes from the swizzle result.
func (b *builder) spmdSwizzlePrepareIndex(index llvm.Value, laneCount int) llvm.Value {
	i8Type := b.ctx.Int8Type()

	// Truncate to i8 if wider (safe since table indices are 0-15).
	elemWidth := index.Type().ElementType().IntTypeWidth()
	if elemWidth > 8 {
		narrowType := llvm.VectorType(i8Type, laneCount)
		index = b.CreateTrunc(index, narrowType, "")
	}

	if laneCount == 16 {
		return index
	}

	// Pad to 16 lanes using a shuffle. Padding lanes pick from the undef second operand
	// and produce undef values; the swizzle result for those lanes is discarded by the
	// extract shuffle in spmdWasmSwizzle, so undef padding is safe.
	maskElems := make([]llvm.Value, 16)
	for i := 0; i < 16; i++ {
		if i < laneCount {
			maskElems[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
		} else {
			maskElems[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(laneCount), false) // pick from the undef second operand
		}
	}
	maskVec := llvm.ConstVector(maskElems, false)
	return b.CreateShuffleVector(index, llvm.Undef(llvm.VectorType(i8Type, laneCount)), maskVec, "")
}

// ssaConstUint64 extracts a uint64 value from an *ssa.Const if it holds an integer
// constant. Returns (0, false) for non-const SSA values, nil constant values, or
// constants that cannot be represented as uint64 (e.g., negative integers).
func ssaConstUint64(v ssa.Value) (uint64, bool) {
	c, ok := v.(*ssa.Const)
	if !ok || c.Value == nil || c.Value.Kind() != constant.Int {
		return 0, false
	}
	val, ok := constant.Uint64Val(c.Value)
	return val, ok
}

// typeBitWidth returns the bit width of a Go type for index range analysis.
// Returns 0 for platform-dependent sizes (int, uint) or non-integer types.
// Unwraps *types.SPMDType before inspection.
func typeBitWidth(t types.Type) int {
	// Unwrap SPMDType wrapper.
	if spmd, ok := t.(*types.SPMDType); ok {
		t = spmd.Elem()
	}
	basic, ok := t.Underlying().(*types.Basic)
	if !ok {
		return 0
	}
	switch basic.Kind() {
	case types.Uint8: // types.Byte == types.Uint8
		return 8
	case types.Uint16:
		return 16
	case types.Uint32:
		return 32
	case types.Uint64:
		return 64
	case types.Int8:
		return 8
	case types.Int16:
		return 16
	case types.Int32:
		return 32
	case types.Int64:
		return 64
	default:
		// int, uint, uintptr, float*, string, etc. — size unknown at compile time.
		return 0
	}
}

// spmdIndexMaxValue computes the maximum possible value of an SSA expression
// used as an index. Returns (maxVal, true) when the upper bound is provable,
// or (0, false) when unknown.
//
// Recognized patterns (compose recursively):
//   - BinOp SHR(x, const k): max = maxOf(x) >> k
//   - BinOp AND(x, const mask): max = mask
//   - BinOp REM(x, const k): max = k - 1  (for k > 0)
//   - Convert to narrower type: max = 2^N - 1
//   - Const: max = const value
//   - ChangeType: delegates to inner value
//
// Falls back to type-based analysis: unsigned N-bit → max 2^N - 1.
func spmdIndexMaxValue(v ssa.Value) (uint64, bool) {
	switch val := v.(type) {
	case *ssa.Const:
		return ssaConstUint64(val)

	case *ssa.ChangeType:
		// ChangeType is a no-op type coercion; delegate to the inner value.
		return spmdIndexMaxValue(val.X)

	case *ssa.Convert:
		// Conversion to a narrower integer type caps the maximum.
		bits := typeBitWidth(val.Type())
		if bits > 0 && bits < 64 {
			// Check if the type is unsigned — signed truncation can produce negative values.
			rawType := val.Type()
			if spmd, ok := rawType.(*types.SPMDType); ok {
				rawType = spmd.Elem()
			}
			if basic, ok := rawType.Underlying().(*types.Basic); ok {
				if basic.Info()&types.IsUnsigned != 0 {
					return (uint64(1) << uint(bits)) - 1, true
				}
			}
		}
		// For signed conversions or platform-size types, fall through to type fallback.

	case *ssa.BinOp:
		switch val.Op {
		case token.SHR:
			// x >> k: upper bound is maxOf(x) >> k.
			if maxX, ok := spmdIndexMaxValue(val.X); ok {
				if k, ok := ssaConstUint64(val.Y); ok && k < 64 {
					return maxX >> k, true
				}
			}

		case token.AND:
			// x & mask: the result is bounded by the mask value, regardless of x.
			if mask, ok := ssaConstUint64(val.Y); ok {
				return mask, true
			}
			if mask, ok := ssaConstUint64(val.X); ok {
				return mask, true
			}

		case token.REM:
			// x % k: result is in [0, k-1] for k > 0.
			if k, ok := ssaConstUint64(val.Y); ok && k > 0 {
				return k - 1, true
			}
		}
	}

	// Type-based fallback: unsigned N-bit type has max 2^N - 1.
	bits := typeBitWidth(v.Type())
	if bits > 0 && bits < 64 {
		rawType := v.Type()
		if spmd, ok := rawType.(*types.SPMDType); ok {
			rawType = spmd.Elem()
		}
		if basic, ok := rawType.Underlying().(*types.Basic); ok {
			if basic.Info()&types.IsUnsigned != 0 {
				return (uint64(1) << uint(bits)) - 1, true
			}
		}
	}

	return 0, false
}
