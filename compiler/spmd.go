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

// splatScalar broadcasts a scalar value to fill all lanes of a vector type.
func (b *builder) splatScalar(scalar llvm.Value, vecType llvm.Type) llvm.Value {
	undef := llvm.Undef(vecType)
	zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
	ins := b.CreateInsertElement(undef, scalar, zero, "")
	mask := llvm.ConstNull(llvm.VectorType(b.ctx.Int32Type(), vecType.VectorSize()))
	return b.CreateShuffleVector(ins, undef, mask, "splat")
}

// spmdBroadcastMatch ensures both operands have matching types for SPMD operations.
// If one operand is a vector and the other is a scalar, the scalar is splatted.
func (b *builder) spmdBroadcastMatch(x, y llvm.Value) (llvm.Value, llvm.Value) {
	xIsVec := x.Type().TypeKind() == llvm.VectorTypeKind
	yIsVec := y.Type().TypeKind() == llvm.VectorTypeKind
	if xIsVec && !yIsVec {
		y = b.splatScalar(y, x.Type())
	} else if !xIsVec && yIsVec {
		x = b.splatScalar(x, y.Type())
	}
	return x, y
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
	iterPhi    *ssa.Phi   // the "rangeint.iter" phi in body block
	laneCount  int        // e.g. 4 for int32 on WASM SIMD128
	boundValue ssa.Value  // N in "range N"
	incrBinOp  *ssa.BinOp // the ADD in the loop block

	// Set during IR generation:
	laneIndices llvm.Value // <iter, iter+1, ..., iter+laneCount-1>
	tailMask    llvm.Value // per-lane bounds check
}

// analyzeSPMDLoops performs pre-analysis of SPMD loops before block compilation.
// It identifies the SSA pattern for "go for i := range N" loops and extracts the
// key values needed for vectorization: the iter phi, bound value, and increment operation.
//
// This relies on golang.org/x/tools/go/ssa's rangeint pattern comments:
//   - "rangeint.body": loop body block containing the iteration variable phi
//   - "rangeint.iter": phi instruction for the loop counter
//   - "rangeint.loop": successor block with increment (ADD) and bounds check (LSS)
//
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

		// Check if this phi's position is inside an SPMD loop.
		loopInfo := b.isInSPMDLoop(iterPhi.Pos())
		if loopInfo == nil {
			continue
		}

		// Find the successor rangeint.loop block.
		var loopBlock *ssa.BasicBlock
		for _, succ := range block.Succs {
			if succ.Comment == "rangeint.loop" {
				loopBlock = succ
				break
			}
		}
		if loopBlock == nil {
			continue
		}

		// Find the increment BinOp (iter + 1) in the loop block.
		var incrBinOp *ssa.BinOp
		var boundValue ssa.Value
		for _, instr := range loopBlock.Instrs {
			if binOp, ok := instr.(*ssa.BinOp); ok {
				if binOp.Op == token.ADD && binOp.X == iterPhi {
					incrBinOp = binOp
				}
				// Find the bounds check (incr < N).
				if binOp.Op == token.LSS && incrBinOp != nil && binOp.X == incrBinOp {
					boundValue = binOp.Y
				}
			}
		}
		if incrBinOp == nil || boundValue == nil {
			continue
		}

		// Compute lane count based on the iter phi's element type.
		// The phi has Go's int type, which on WASM is i32.
		elemType := b.getLLVMType(iterPhi.Type())
		laneCount := b.spmdLaneCount(elemType)

		// Create the active loop entry.
		loop := &spmdActiveLoop{
			info:       loopInfo,
			iterPhi:    iterPhi,
			laneCount:  laneCount,
			boundValue: boundValue,
			incrBinOp:  incrBinOp,
		}

		// Populate the maps for quick lookup.
		state.activeLoops[iterPhi] = loop
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
func (b *builder) emitSPMDBodyPrologue(loop *spmdActiveLoop) {
	// Get the scalar phi value (already compiled as LLVM phi).
	scalarPhi := b.locals[loop.iterPhi]

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

// spmdVaryingIf holds analysis results for a varying (vector) if/else construct.
type spmdVaryingIf struct {
	cond           llvm.Value // vector condition (<N x i1>)
	ifBlockIndex   int        // block with the If instruction
	thenEntryIndex int        // Succs[0] of if-block
	elseEntryIndex int        // Succs[1] of if-block (or merge for if-without-else)
	mergeIndex     int        // common successor (merge point)
	hasElse        bool       // true if then/else are distinct from merge
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
