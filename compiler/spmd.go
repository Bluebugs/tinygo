package compiler

// This file extracts SPMD metadata from the typed AST for use during LLVM IR
// generation. TinyGo uses golang.org/x/tools/go/ssa which has no SPMD support,
// so we build a side-table from go/ast and go/types (which our Go fork extends).

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
	"sort"
	"strconv"
	"strings"

	"github.com/tinygo-org/tinygo/loader"
	"golang.org/x/tools/go/ssa"
	spmdtypes "golang.org/x/tools/go/types/spmd"
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
	// to incrBinOp, and extract the element type from its base.
	//
	// In SPMD context, the index may be wrapped in a ChangeType to lanes.Varying[int],
	// so we unwrap ChangeType chains before comparing.
	for _, instr := range bodyBlock.Instrs {
		ia, ok := instr.(*ssa.IndexAddr)
		if !ok {
			continue
		}
		// Unwrap ChangeType chains from the index (SPMD wraps int → Varying[int]).
		idx := ia.Index
		for ct, ok := idx.(*ssa.ChangeType); ok; ct, ok = idx.(*ssa.ChangeType) {
			idx = ct.X
		}
		// The unwrapped index must be the increment BinOp (the SPMD iterator).
		if idx != ssa.Value(incrBinOp) {
			continue
		}
		// Determine element type to compute lane count.
		var elemType types.Type
		if sliceType, ok := ia.X.Type().Underlying().(*types.Slice); ok {
			elemType = sliceType.Elem()
		} else if ptrType, ok := ia.X.Type().Underlying().(*types.Pointer); ok {
			if arrType, ok := ptrType.Elem().Underlying().(*types.Array); ok {
				elemType = arrType.Elem()
			}
		}
		if elemType != nil {
			elemLLVM := b.getLLVMType(elemType)
			return b.spmdLaneCount(elemLLVM)
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

// vectorToArray converts an LLVM <N x T> vector value to an [N x T] array
// by extracting each element and inserting it into an array. Inverse of arrayToVector.
// Used by MakeInterface to box lanes.Varying[T] vectors as [N]T arrays for interface packing.
func (b *builder) vectorToArray(vec llvm.Value) llvm.Value {
	vecType := vec.Type()
	n := vecType.VectorSize()
	elemType := vecType.ElementType()
	arrType := llvm.ArrayType(elemType, n)
	arr := llvm.Undef(arrType)
	for i := 0; i < n; i++ {
		idx := llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
		elem := b.CreateExtractElement(vec, idx, "")
		arr = b.CreateInsertValue(arr, elem, i, "")
	}
	return arr
}

// spmdBroadcastMatch ensures both operands have matching types for SPMD operations.
// If one operand is a vector and the other is a scalar, the scalar is splatted.
// If both are vectors with different lane counts, the wider one is resized to match the narrower.
// If both are vectors with the same lane count but different element widths (e.g., <4 x i32>
// and <4 x i8> from a zext'd byte gather), the narrower elements are zero-extended to match.
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
	// After lane-count matching, handle element-width mismatches caused by WASM
	// zero-extension. When spmdVectorIndexArray widens sub-128-bit byte gathers
	// to the mask element type (e.g., <4 x i8> -> <4 x i32>), and the paired
	// operand is still a byte-width constant (e.g., <4 x i8> splat), zero-extend
	// the narrower operand so both sides have the same integer element width.
	// Use the actual current types (x/y may have been updated above).
	xType := x.Type()
	yType := y.Type()
	if xType.TypeKind() == llvm.VectorTypeKind &&
		yType.TypeKind() == llvm.VectorTypeKind &&
		xType.VectorSize() == yType.VectorSize() &&
		xType.ElementType().TypeKind() == llvm.IntegerTypeKind &&
		yType.ElementType().TypeKind() == llvm.IntegerTypeKind &&
		xType.ElementType() != yType.ElementType() {
		xWidth := xType.ElementType().IntTypeWidth()
		yWidth := yType.ElementType().IntTypeWidth()
		if xWidth > yWidth {
			y = b.CreateZExt(y, xType, "")
		} else {
			x = b.CreateZExt(x, yType, "")
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

// spmdWASMNarrowAndPack packs a WASM-widened SPMD vector into a scalar integer
// for sub-128-bit byte stores, using only scalar operations. This avoids any
// intermediate sub-128-bit vector type (e.g., <4 x i8>), which WASM cannot
// lower. LLVM's InstCombine would otherwise reconstruct an illegal vector trunc
// from per-element scalar extract+trunc+insert patterns.
//
// val must be a <N x iWide> vector (e.g., <4 x i32>) whose elements have
// already been AND-masked to targetElemBits range (e.g., 0xFF for uint8).
// targetElemBits is the size of each stored element in bits (e.g., 8).
//
// Returns an iTotal scalar (where Total = N * targetElemBits) with elements
// packed in little-endian order: bits [0..e) = lane0, bits [e..2e) = lane1, etc.
//
// Example: <4 x i32> [v0, v1, v2, v3] with targetElemBits=8 →
//
//	i32: (v0&0xFF) | ((v1&0xFF)<<8) | ((v2&0xFF)<<16) | ((v3&0xFF)<<24)
//
// No <4 x i8> is created at any point, preventing InstCombine from
// rewriting the sequence to trunc <4 x i32> to <4 x i8>.
func (b *builder) spmdWASMNarrowAndPack(val llvm.Value, targetElemBits uint64) llvm.Value {
	laneCount := val.Type().VectorSize()
	totalBits := int(targetElemBits) * laneCount
	scalarType := b.ctx.IntType(totalBits)
	elemMask := llvm.ConstInt(val.Type().ElementType(), (1<<targetElemBits)-1, false)
	result := llvm.ConstNull(scalarType)
	for lane := 0; lane < laneCount; lane++ {
		laneIdx := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
		elem := b.CreateExtractElement(val, laneIdx, "") // iWide
		elem = b.CreateAnd(elem, elemMask, "")           // mask to targetElemBits
		elem = b.CreateZExt(elem, scalarType, "")        // zero-extend to iTotal
		shift := llvm.ConstInt(scalarType, uint64(lane)*targetElemBits, false)
		elem = b.CreateShl(elem, shift, "")   // shift to lane position
		result = b.CreateOr(result, elem, "") // accumulate
	}
	return result
}

// createSPMDConst creates a splatted vector constant for an SPMD varying type.
func (c *compilerContext) createSPMDConst(expr *ssa.Const, spmdType *types.SPMDType, pos token.Pos) llvm.Value {
	vecType := c.getLLVMType(spmdType)
	if expr.Value == nil {
		return llvm.ConstNull(vecType)
	}

	// Varying[mask] element type (MaskType) has no scalar constant form.
	// Generate all-ones (true) or all-zeros (false) mask vector directly.
	if spmdtypes.IsMask(spmdType.Elem()) {
		if expr.Value.Kind() != constant.Bool {
			panic(fmt.Sprintf("createSPMDConst: Varying[mask] constant has unexpected kind %v", expr.Value.Kind()))
		}
		if constant.BoolVal(expr.Value) {
			return llvm.ConstAllOnes(vecType)
		}
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
	info        *SPMDLoopInfo
	ssaLoopInfo *ssa.SPMDLoopInfo // x-tools-spmd loop info (non-nil when isPeeled == true)
	iterPhi     *ssa.Phi          // the "rangeint.iter" phi in body block (nil for rangeindex)
	laneCount   int               // e.g. 4 for int32 on WASM SIMD128
	boundValue  ssa.Value         // N in "range N"
	incrBinOp   *ssa.BinOp        // the ADD in the loop block

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

	// Peeling fields (only set when ssaLoopInfo.IsPeeled == true):
	isPeeled bool // loop was peeled at SSA level; main body uses all-ones mask

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

	// Pass 0: Handle SSA-peeled loops directly from metadata.
	// When peelSPMDLoops runs in go/ssa, it creates MainBodyBlock, TailBodyBlock,
	// etc. The original body block becomes unreachable but is still in fn.Blocks.
	// We claim the loopInfo here so the rangeint pass (below) skips the original
	// unreachable body block.
	for _, ssaLoop := range b.fn.SPMDLoops {
		if !ssaLoop.IsPeeled {
			continue
		}

		mainIterPhi := ssaLoop.MainIterPhi
		if mainIterPhi == nil {
			continue
		}

		// Deduplicate: map SSA loop info pointer to TinyGo loop info via position.
		// The mainIterPhi is in MainBodyBlock; find the corresponding TinyGo SPMDLoopInfo.
		// Use the ORIGINAL body block (pre-peeling) for position-based
		// SPMDLoopInfo lookup, since the peeled MainBodyBlock contains cloned
		// instructions that may lack source positions.
		var loopInfo *SPMDLoopInfo
		lookupBlock := ssaLoop.BodyBlock // original body block (unreachable after peeling)
		for _, instr := range lookupBlock.Instrs {
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
		if seenLoopInfo[loopInfo] {
			continue
		}
		seenLoopInfo[loopInfo] = true

		// The main incr BinOp is the back-edge value of mainIterPhi.
		// mainIterPhi.Edges = [zeroConst, mainIncr].
		if len(mainIterPhi.Edges) < 2 {
			continue
		}
		mainIncrBinOp, ok := mainIterPhi.Edges[1].(*ssa.BinOp)
		if !ok || mainIncrBinOp.Op != token.ADD {
			continue
		}

		// Compute lane count. For rangeindex loops (range-over-slice), the
		// iterator type is always int (i32 on WASM32) which gives 4 lanes, but
		// the actual lane count depends on the slice element type (e.g., 16 for
		// []byte). Use spmdRangeIndexLaneCount to get the correct width.
		// For rangeint loops, the iter phi type is authoritative.
		var laneCount int
		if ssaLoop.IsRangeIndex {
			laneCount = b.spmdRangeIndexLaneCount(ssaLoop.BoundValue, ssaLoop.MainBodyBlock, mainIncrBinOp)
		} else {
			elemType := b.getLLVMType(mainIterPhi.Type())
			laneCount = b.spmdLaneCount(elemType)
		}

		// On WASM, rangeindex loops with more than 4 lanes use a decomposed
		// (scalar base + <N x iW> offset) representation to stay within 128-bit SIMD.
		isDecomposed := ssaLoop.IsRangeIndex && b.spmdIsWASM() && laneCount > 4

		loop := &spmdActiveLoop{
			info:          loopInfo,
			ssaLoopInfo:   ssaLoop,
			iterPhi:       mainIterPhi,
			laneCount:     laneCount,
			boundValue:    ssaLoop.BoundValue,
			incrBinOp:     mainIncrBinOp,
			bodyIterValue: mainIterPhi,
			isPeeled:      true,
			isRangeIndex:  ssaLoop.IsRangeIndex,
			isDecomposed:  isDecomposed,
		}

		// Map main body: activeLoops[phi] triggers emitSPMDBodyPrologue via phi handler.
		state.activeLoops[mainIterPhi] = loop
		state.bodyBlocks[ssaLoop.MainBodyBlock.Index] = loop
		state.loopBlocks[ssaLoop.MainBodyBlock.Index] = loop

		// Map tail body: no iter phi in TailBodyBlock (TailIterPhi is in TailCheckBlock).
		// The prologue is triggered by body block entry code (isPeeled check).
		// Do NOT add TailIterPhi to activeLoops — TailCheckBlock is not a body block
		// and the phi handler must not call emitSPMDBodyPrologue for it.
		if ssaLoop.TailBodyBlock != nil {
			state.bodyBlocks[ssaLoop.TailBodyBlock.Index] = loop
			state.loopBlocks[ssaLoop.TailBodyBlock.Index] = loop
		}
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

// spmdRangeIndexInitOverride replaces val with -laneCount if phi is a
// rangeindex loop phi and i is its entry-edge index. The rangeindex pattern
// starts the loop phi at -1; SPMD vectorization must change this to
// -laneCount so the first iteration produces indices [0, 1, ..., N-1].
// Called from every phi-resolution path that calls b.getValue on a loop phi.
func (b *builder) spmdRangeIndexInitOverride(phi *ssa.Phi, i int, val llvm.Value) llvm.Value {
	if b.spmdLoopState == nil {
		return val
	}
	loop, ok := b.spmdLoopState.activeLoops[phi]
	if !ok || !loop.isRangeIndex {
		return val
	}
	if i == loop.initEdgeIndex {
		return llvm.ConstInt(val.Type(), uint64(int64(-loop.laneCount)), true)
	}
	return val
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
	// Peeled loop path: the SSA already contains MainBodyBlock and TailBodyBlock
	// with correct phi and control flow. We just need to set up lane indices and
	// the appropriate mask (all-ones for main, computed for tail).
	if loop.isPeeled {
		ssaLoop := loop.ssaLoopInfo
		isMainBody := b.currentBlock.Index == ssaLoop.MainBodyBlock.Index

		// Get the appropriate iter phi's scalar value.
		var scalarPhi llvm.Value
		if isMainBody {
			scalarPhi = b.locals[ssaLoop.MainIterPhi]
		} else {
			scalarPhi = b.locals[ssaLoop.TailIterPhi]
		}
		loop.scalarIterVal = scalarPhi

		// Compute lane indices: <iter, iter+1, ..., iter+laneCount-1>.
		elemType := scalarPhi.Type()
		vecType := llvm.VectorType(elemType, loop.laneCount)
		iterVec := b.splatScalar(scalarPhi, vecType)
		offsetVec := b.spmdLaneOffsetConst(loop.laneCount, elemType)
		laneIndices := b.CreateAdd(iterVec, offsetVec, "spmd.lane.idx")
		loop.laneIndices = laneIndices

		if isMainBody {
			// Main body: all-ones mask (all lanes active).
			maskType := llvm.VectorType(b.spmdMaskElemType(loop.laneCount), loop.laneCount)
			loop.tailMask = llvm.ConstAllOnes(maskType)
		} else {
			// Tail body: compute tail mask (laneIndices < bound).
			boundScalar := b.getValue(loop.boundValue, token.NoPos)
			boundVec := b.splatScalar(boundScalar, vecType)
			tailMaskI1 := b.CreateICmp(llvm.IntSLT, laneIndices, boundVec, "spmd.tail.mask")
			loop.tailMask = b.spmdWrapMask(tailMaskI1, loop.laneCount)
			// Register SSA TailMask so getValue() can resolve it directly.
			if loop.ssaLoopInfo != nil && loop.ssaLoopInfo.TailMask != nil {
				b.locals[loop.ssaLoopInfo.TailMask] = loop.tailMask
			}
		}
		return
	}

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
		// Register SSA TailMask so getValue() can resolve it directly.
		if loop.ssaLoopInfo != nil && loop.ssaLoopInfo.TailMask != nil {
			b.locals[loop.ssaLoopInfo.TailMask] = loop.tailMask
		}
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
	// Register SSA TailMask so getValue() can resolve it directly.
	if loop.ssaLoopInfo != nil && loop.ssaLoopInfo.TailMask != nil {
		b.locals[loop.ssaLoopInfo.TailMask] = loop.tailMask
	}
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

// spmdContiguousInfo tracks an IndexAddr result that was detected as contiguous SPMD access.
type spmdContiguousInfo struct {
	scalarPtr   llvm.Value      // scalar GEP result (base of contiguous access)
	loop        *spmdActiveLoop // owning loop (for lane count)
	sliceCap    llvm.Value      // cap of source slice (zero value for arrays/strings)
	scalarIndex llvm.Value      // actual scalar GEP index used (may be iter+offset)
	ssaSource   ssa.Value       // original IndexAddr.X for alloca origin tracing
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

// spmdStridePattern represents the stride decomposition of an SSA index expression.
// For example, i*2+1 has stride=2, remainder=1, where i is the SPMD loop iterator.
type spmdStridePattern struct {
	stride    int64
	remainder int64
	iterValue ssa.Value       // the body iter value matched
	loop      *spmdActiveLoop // owning SPMD loop
}

// spmdInterleavedStoreGroup represents a complete set of stride-S stores
// that write to complementary positions (remainders 0..S-1) of the same
// base slice. These can be replaced with S shufflevector interleaves +
// S contiguous masked stores.
type spmdInterleavedStoreGroup struct {
	stores    []ssa.Instruction // one store per remainder, ordered 0..S-1 (*ssa.Store or *ssa.SPMDStore)
	addrs     []*ssa.IndexAddr  // corresponding IndexAddr per remainder
	stride    int               // stride value (2, 3, or 4)
	baseSlice ssa.Value         // the common base slice (IndexAddr.X)
	loop      *spmdActiveLoop   // owning SPMD loop
	laneCount int               // SIMD lane count
}

// spmdInterleavedStoreInfo links a single store or IndexAddr to its
// interleaved group and its position within that group.
type spmdInterleavedStoreInfo struct {
	group     *spmdInterleavedStoreGroup
	remainder int
}

// spmdAnalyzeStrideIndex pattern-matches an SSA index expression to detect
// stride-S interleaved access patterns of the form iter*S+R, where iter is
// the SPMD loop body iterator, S is the stride (2, 3, or 4), and R is the
// remainder (0 to S-1).
//
// Recognized patterns (commutative MUL and ADD):
//   - iter*S           → stride=S, remainder=0
//   - iter*S + R       → stride=S, remainder=R
//
// ChangeType wrappers on the index and on iter candidates are peeled before
// matching. Only strides in [2,4] and remainders in [0, stride-1] are accepted.
// Returns nil if the pattern does not match.
func (b *builder) spmdAnalyzeStrideIndex(index ssa.Value) *spmdStridePattern {
	if b.spmdLoopState == nil {
		return nil
	}

	// Unwrap ChangeType chains on the top-level index value.
	unwrapCT := func(v ssa.Value) ssa.Value {
		for {
			if ct, ok := v.(*ssa.ChangeType); ok {
				v = ct.X
			} else {
				break
			}
		}
		return v
	}

	// checkIsIter returns the active loop if v (after ChangeType unwrap) is the
	// body iterator for some SPMD loop.
	checkIsIter := func(v ssa.Value) *spmdActiveLoop {
		core := unwrapCT(v)
		if loop, ok := b.spmdLoopState.activeLoops[core]; ok {
			return loop
		}
		return nil
	}

	// tryMul checks whether a BinOp is iter*S (commutative) and returns
	// (stride, iterValue, loop) on success.
	tryMul := func(op *ssa.BinOp) (int64, ssa.Value, *spmdActiveLoop, bool) {
		if op.Op != token.MUL {
			return 0, nil, nil, false
		}
		// Try X=iter, Y=const.
		if loop := checkIsIter(op.X); loop != nil {
			if c, ok := ssaConstInt64(op.Y); ok && c >= 2 && c <= 4 {
				return c, unwrapCT(op.X), loop, true
			}
		}
		// Try X=const, Y=iter.
		if loop := checkIsIter(op.Y); loop != nil {
			if c, ok := ssaConstInt64(op.X); ok && c >= 2 && c <= 4 {
				return c, unwrapCT(op.Y), loop, true
			}
		}
		return 0, nil, nil, false
	}

	idx := unwrapCT(index)

	// Pattern: iter*S (no remainder).
	if mul, ok := idx.(*ssa.BinOp); ok {
		if stride, iterVal, loop, ok := tryMul(mul); ok {
			return &spmdStridePattern{
				stride:    stride,
				remainder: 0,
				iterValue: iterVal,
				loop:      loop,
			}
		}
	}

	// Pattern: iter*S + R  (ADD is commutative).
	if add, ok := idx.(*ssa.BinOp); ok && add.Op == token.ADD {
		// Try add.X = mul, add.Y = const.
		if mulOp, ok := unwrapCT(add.X).(*ssa.BinOp); ok {
			if stride, iterVal, loop, ok := tryMul(mulOp); ok {
				if r, ok := ssaConstInt64(add.Y); ok && r >= 0 && r < stride {
					return &spmdStridePattern{
						stride:    stride,
						remainder: r,
						iterValue: iterVal,
						loop:      loop,
					}
				}
			}
		}
		// Try add.Y = mul, add.X = const.
		if mulOp, ok := unwrapCT(add.Y).(*ssa.BinOp); ok {
			if stride, iterVal, loop, ok := tryMul(mulOp); ok {
				if r, ok := ssaConstInt64(add.X); ok && r >= 0 && r < stride {
					return &spmdStridePattern{
						stride:    stride,
						remainder: r,
						iterValue: iterVal,
						loop:      loop,
					}
				}
			}
		}
	}

	return nil
}

// ssaConstInt64 extracts an int64 value from an *ssa.Const if it holds an
// integer constant representable as int64. Returns (0, false) otherwise.
func ssaConstInt64(v ssa.Value) (int64, bool) {
	c, ok := v.(*ssa.Const)
	if !ok || c.Value == nil || c.Value.Kind() != constant.Int {
		return 0, false
	}
	val, ok := constant.Int64Val(c.Value)
	return val, ok
}

// spmdStoreVal extracts the Val (value being stored) from either a *ssa.Store
// or a *ssa.SPMDStore instruction. Panics if instr is neither type.
func spmdStoreVal(instr ssa.Instruction) ssa.Value {
	switch s := instr.(type) {
	case *ssa.Store:
		return s.Val
	case *ssa.SPMDStore:
		return s.Val
	default:
		panic(fmt.Sprintf("spmdStoreVal: unexpected instruction type %T", instr))
	}
}

// spmdAnalyzeInterleavedStores scans all SPMD body blocks and groups stores whose
// target index follows an iter*S+R pattern (stride S in [2,4], remainder R in
// [0..S-1]) into spmdInterleavedStoreGroup records. Complete groups (all S
// remainders present) are registered in b.spmdInterleavedStores and
// b.spmdInterleavedAddrs so that the store emitter can replace the S scatter
// operations with S shufflevector+masked-store pairs.
//
// Must be called after b.spmdLoopState is populated (i.e., after analyzeSPMDLoops).
func (b *builder) spmdAnalyzeInterleavedStores() {
	if b.spmdLoopState == nil {
		return
	}

	// Key for grouping: (base-slice SSA value, stride, owning loop).
	// The loop pointer prevents cross-loop false grouping when two SPMD loops
	// write stride-S patterns to the same base slice.
	type groupKey struct {
		base   ssa.Value
		stride int64
		loop   *spmdActiveLoop
	}

	// Partial group accumulator: maps groupKey → per-remainder slot.
	type partialGroup struct {
		stores []ssa.Instruction // indexed by remainder; nil means not yet seen (*ssa.Store or *ssa.SPMDStore)
		addrs  []*ssa.IndexAddr  // parallel to stores
		loop   *spmdActiveLoop
	}

	partials := make(map[groupKey]*partialGroup)

	for _, block := range b.fn.Blocks {
		loop, ok := b.spmdLoopState.bodyBlocks[block.Index]
		if !ok {
			continue // not an SPMD body block
		}

		for _, instr := range block.Instrs {
			// Match both *ssa.Store and *ssa.SPMDStore (predicated stores).
			var storeAddr ssa.Value
			switch s := instr.(type) {
			case *ssa.SPMDStore:
				storeAddr = s.Addr
			case *ssa.Store:
				storeAddr = s.Addr
			default:
				continue
			}
			indexAddr, ok := storeAddr.(*ssa.IndexAddr)
			if !ok {
				continue
			}
			// Skip IndexAddr nodes that have other referrers besides this store.
			// Returning undef for the IndexAddr during codegen would break any
			// other use (e.g., a load in a read-modify-write pattern).
			if refs := indexAddr.Referrers(); refs == nil || len(*refs) != 1 {
				continue
			}
			pat := b.spmdAnalyzeStrideIndex(indexAddr.Index)
			if pat == nil {
				continue
			}
			if pat.loop != loop {
				continue // pattern iter belongs to a different loop
			}

			key := groupKey{base: indexAddr.X, stride: pat.stride, loop: loop}
			pg := partials[key]
			if pg == nil {
				pg = &partialGroup{
					stores: make([]ssa.Instruction, pat.stride),
					addrs:  make([]*ssa.IndexAddr, pat.stride),
					loop:   loop,
				}
				partials[key] = pg
			}
			rem := int(pat.remainder)
			if pg.stores[rem] != nil {
				continue // duplicate remainder in the same block — skip
			}
			pg.stores[rem] = instr
			pg.addrs[rem] = indexAddr
		}
	}

	// Promote complete groups (all remainders filled) to the registered maps.
	for key, pg := range partials {
		stride := int(key.stride)
		complete := true
		for r := 0; r < stride; r++ {
			if pg.stores[r] == nil {
				complete = false
				break
			}
		}
		if !complete {
			continue
		}

		// Verify all stored values have the same type.
		baseType := spmdStoreVal(pg.stores[0]).Type()
		for r := 1; r < stride; r++ {
			if !types.Identical(spmdStoreVal(pg.stores[r]).Type(), baseType) {
				complete = false
				break
			}
		}
		if !complete {
			continue
		}

		group := &spmdInterleavedStoreGroup{
			stores:    pg.stores,
			addrs:     pg.addrs,
			stride:    stride,
			baseSlice: key.base,
			loop:      pg.loop,
			laneCount: pg.loop.laneCount,
		}
		for r := 0; r < stride; r++ {
			info := &spmdInterleavedStoreInfo{group: group, remainder: r}
			b.spmdInterleavedStores[pg.stores[r]] = info
			b.spmdInterleavedAddrs[pg.addrs[r]] = info
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

// spmdVectorAnyTrue reduces an SPMD mask vector to a scalar i1.
// Returns true if any lane is active.
// On WASM targets, uses the native v128.any_true instruction via the
// @llvm.wasm.anytrue LLVM intrinsic (single WASM instruction, no bitcast).
// On other targets the mask is <N x i1>, so we bitcast to iN and compare != 0.
func (b *builder) spmdVectorAnyTrue(mask llvm.Value) llvm.Value {
	if b.spmdIsWASM() {
		// Normalize <N x i1> to WASM mask format before calling the intrinsic.
		// The llvm.wasm.anytrue intrinsic is registered by vector type; calling
		// llvm.wasm.anytrue.v4i1 with a <4 x i1> argument creates a custom
		// intrinsic call on a sub-128-bit type. The WASM backend's legalization
		// does not know how to promote <4 x i1> arguments to custom intrinsics
		// and crashes with "Do not know how to promote this operator's operand".
		// Convert to WASM mask format (<N x iW>) first so we call
		// llvm.wasm.anytrue.v4i32 (or similar), which WASM can lower natively.
		laneCount := mask.Type().VectorSize()
		if mask.Type().ElementType() == b.ctx.Int1Type() {
			mask = b.spmdWrapMask(mask, laneCount)
		}
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
		// Normalize <N x i1> to WASM mask format before calling the intrinsic.
		// See spmdVectorAnyTrue for the rationale.
		laneCount := mask.Type().VectorSize()
		if mask.Type().ElementType() == b.ctx.Int1Type() {
			mask = b.spmdWrapMask(mask, laneCount)
		}
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

// spmdIsWASM returns true when the compiler target is a WebAssembly target.
// On WASM, SIMD comparisons natively produce <N x i32> (all-ones/all-zeros),
// so we use i32 as the internal mask element type to avoid the redundant
// shl/shr_s sign-extension that LLVM inserts when converting <N x i1> to i32
// for v128.bitselect.
func (c *compilerContext) spmdIsWASM() bool {
	return strings.HasPrefix(c.Triple, "wasm")
}

// spmdIsConstAllOnesMask returns true if the mask is a compile-time constant
// with all bits set (ConstAllOnes). When true, LLVM already optimizes
// llvm.masked.load to a plain load, so cap-based optimization is unnecessary.
func (b *builder) spmdIsConstAllOnesMask(mask llvm.Value) bool {
	if !mask.IsConstant() {
		return false
	}
	allOnes := llvm.ConstAllOnes(mask.Type())
	return mask.C == allOnes.C
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
	if cmp.Type() == maskType {
		// Already in WASM mask format; SExt of same type would be invalid.
		return cmp
	}
	if cmp.Type().ElementType() != b.ctx.Int1Type() {
		panic(fmt.Sprintf("spmdWrapMask: called with non-i1 element type %s (expected <N x i1>)", cmp.Type().String()))
	}
	return b.CreateSExt(cmp, maskType, "")
}

// spmdUnwrapMaskForIntrinsic converts the WASM mask (e.g., <N x i32>, <N x i16>,
// or <N x i8>) to <laneCount x i1> for LLVM masked memory intrinsics
// (masked.load, masked.store, masked.gather, masked.scatter).
// These intrinsics require an <N x i1> mask regardless of the target.
// On non-WASM targets the mask is already <N x i1> and this is a no-op.
//
// When the mask lane count differs from laneCount (e.g., a <4 x i32> mask
// from Varying[mask] used with a 16-lane byte loop), spmdConvertMaskFormat
// first reshapes the mask to <laneCount x i1> before any truncation.
func (b *builder) spmdUnwrapMaskForIntrinsic(mask llvm.Value, laneCount int) llvm.Value {
	if !b.spmdIsWASM() {
		return mask
	}
	i1MaskType := llvm.VectorType(b.ctx.Int1Type(), laneCount)
	// If mask already has the right type, return immediately.
	if mask.Type() == i1MaskType {
		return mask
	}
	// If mask lane count differs from laneCount, reshape via spmdConvertMaskFormat.
	// Attempting CreateTrunc between vectors of different element counts is invalid.
	if mask.Type().VectorSize() != laneCount {
		return b.spmdConvertMaskFormat(mask, i1MaskType)
	}
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

// spmdMatchMaskFormat ensures that cond matches the element type of refMask.
// When cond is <N x i1> (a raw comparison) and refMask is <N x iW> (WASM format),
// this wraps cond via sign-extension to produce <N x iW>. If the types already
// match, cond is returned unchanged.
func (b *builder) spmdMatchMaskFormat(cond, refMask llvm.Value) llvm.Value {
	if cond.Type() == refMask.Type() {
		return cond
	}
	if !b.spmdIsWASM() {
		return cond // non-WASM: masks are always <N x i1>, no conversion needed
	}
	condType := cond.Type()
	if condType.TypeKind() != llvm.VectorTypeKind {
		return cond
	}
	laneCount := condType.VectorSize()
	// If cond element is i1, wrap to match refMask's element width.
	if condType.ElementType() == b.ctx.Int1Type() {
		return b.spmdWrapMask(cond, laneCount)
	}
	return cond
}

// spmdIsVaryingBoolPhi reports whether the given SSA phi node has type Varying[bool].
// Used to restrict mask-format reconciliation to only phi nodes that carry boolean
// SPMD masks, avoiding spurious conversions on unrelated vector type mismatches.
func (b *builder) spmdIsVaryingBoolPhi(phi *ssa.Phi) bool {
	spmdType, ok := phi.Type().(*types.SPMDType)
	if !ok || !spmdType.IsVarying() {
		return false
	}
	elem := spmdType.Elem().Underlying()
	basic, ok := elem.(*types.Basic)
	return ok && basic.Info()&types.IsBoolean != 0
}

// spmdFindActiveLoopForBlock returns the active SPMD loop for the given SSA
// block, or nil if the block is not inside any SPMD loop body. Checks the
// direct bodyBlocks lookup first (O(1)), then falls back to dominance to
// find loops whose body blocks dominate interior if.then/if.else/if.done blocks.
func (b *builder) spmdFindActiveLoopForBlock(block *ssa.BasicBlock) *spmdActiveLoop {
	if b.spmdLoopState == nil || block == nil {
		return nil
	}
	// Direct lookup: is this block a recognized body block?
	if loop, ok := b.spmdLoopState.bodyBlocks[block.Index]; ok {
		return loop
	}
	// Dominance fallback: find the innermost body block that dominates this block.
	// When multiple sequential loops exist in the same function, their body blocks
	// may all dominate later blocks. We select the one with the highest block index
	// (latest in the function), which corresponds to the innermost enclosing loop
	// for sequential loop layouts.
	var bestLoop *spmdActiveLoop
	bestBlockIdx := -1
	for _, loop := range b.spmdLoopState.bodyBlocks {
		var bodyBlock *ssa.BasicBlock
		if loop.isRangeIndex {
			// For rangeindex, use the loop header block (where incrBinOp lives)
			// as a dominance proxy. It dominates the body block and thus also
			// dominates any block dominated by the body block.
			bodyBlock = loop.incrBinOp.Block()
		} else {
			bodyBlock = loop.iterPhi.Block()
		}
		if bodyBlock.Dominates(block) && bodyBlock.Index > bestBlockIdx {
			bestLoop = loop
			bestBlockIdx = bodyBlock.Index
		}
	}
	return bestLoop
}

// spmdConvertMaskFormat converts a vector mask value to a different vector mask
// format without changing its logical meaning (all-ones = true, all-zeros = false).
// This handles conversions between <16 x i1>, <4 x i32>, <8 x i16>, <16 x i8>, etc.
// Used to reconcile Varying[bool] type mismatches when phi incoming values or
// constants carry a different mask format than the phi node's declared type.
func (b *builder) spmdConvertMaskFormat(mask llvm.Value, targetType llvm.Type) llvm.Value {
	if mask.Type() == targetType {
		return mask
	}
	if mask.Type().TypeKind() != llvm.VectorTypeKind || targetType.TypeKind() != llvm.VectorTypeKind {
		return mask // Not a vector-to-vector conversion; leave unchanged.
	}

	// For LLVM constants, just produce the correct constant without emitting instructions.
	if mask.IsConstant() {
		if mask.IsNull() {
			return llvm.ConstNull(targetType)
		}
		return llvm.ConstAllOnes(targetType)
	}

	srcLanes := mask.Type().VectorSize()
	srcElem := mask.Type().ElementType()
	targetLanes := targetType.VectorSize()
	targetElem := targetType.ElementType()
	i1Type := b.ctx.Int1Type()

	// Step 1: normalize source to <srcLanes x i1>.
	var i1Vec llvm.Value
	if srcElem == i1Type {
		i1Vec = mask
	} else {
		i1Vec = b.CreateTrunc(mask, llvm.VectorType(i1Type, srcLanes), "spmd.mask.cvt.trunc")
	}

	// Step 2: resize lane count if needed via shufflevector.
	// CreateShuffleVector expects an llvm.Value (constant <N x i32> vector) as mask.
	if srcLanes != targetLanes {
		if targetLanes < srcLanes {
			// Truncate: keep the first targetLanes elements.
			maskElems := make([]llvm.Value, targetLanes)
			for i := range maskElems {
				maskElems[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
			}
			shuffleMask := llvm.ConstVector(maskElems, false)
			undef := llvm.Undef(llvm.VectorType(i1Type, srcLanes))
			i1Vec = b.CreateShuffleVector(i1Vec, undef, shuffleMask, "spmd.mask.cvt.shuf")
		} else {
			// Extend: keep existing lanes, fill rest by repeating lane 0.
			maskElems := make([]llvm.Value, targetLanes)
			for i := range maskElems {
				idx := i
				if idx >= srcLanes {
					idx = 0 // repeat lane 0 as a safe placeholder
				}
				maskElems[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(idx), false)
			}
			shuffleMask := llvm.ConstVector(maskElems, false)
			undef := llvm.Undef(llvm.VectorType(i1Type, srcLanes))
			i1Vec = b.CreateShuffleVector(i1Vec, undef, shuffleMask, "spmd.mask.cvt.ext")
		}
	}

	// Step 3: widen element type if the target element is wider than i1.
	if targetElem == i1Type {
		return i1Vec
	}
	return b.CreateSExt(i1Vec, targetType, "spmd.mask.cvt.sext")
}

// spmdCallMask returns the mask value to pass when calling an SPMD function
// in cases where no SSA-level CallCommon.SPMDMask was set. This handles SPMD
// function bodies (varying-param functions) where the entry mask is the active mask.
// For go-for loop calls, SSA predication sets CallCommon.SPMDMask directly.
func (b *builder) spmdCallMask(fn *ssa.Function) llvm.Value {
	maskType := b.spmdMaskType(fn)
	if maskType == (llvm.Type{}) {
		// Not an SPMD function, no mask needed.
		return llvm.Value{}
	}

	// Entry mask for SPMD function bodies.
	if !b.spmdEntryMask.IsNil() {
		return b.spmdEntryMask
	}

	// All-ones fallback.
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
		// lanes.Count[T](v) returns the number of SIMD lanes in the current loop.
		// When called inside a go-for SPMD loop, the lane count is the loop's
		// hardware lane count, not the natural lane count of type T. For example,
		// lanes.Count(isDot) where isDot is Varying[bool] in a 4-lane loop returns
		// 4, not 16 (which would be the natural count for i1 = 1 byte in 128-bit).
		// This ensures that loop += lanes.Count(v) advances by the correct stride.
		if b.spmdLoopState != nil && b.currentBlock != nil {
			// Use the active loop's lane count when we can find one.
			// b.currentBlock is the SSA block being compiled; spmdFindActiveLoopForBlock
			// walks the loop state to find which SPMD loop this call resides in.
			if activeLoop := b.spmdFindActiveLoopForBlock(b.currentBlock); activeLoop != nil {
				return llvm.ConstInt(b.intType, uint64(activeLoop.laneCount), false), nil
			}
		}
		// Fallback: compute from the element type (for calls outside SPMD loops,
		// e.g., in SPMD function bodies where spmdFuncIsBody is true).
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
		// On non-WASM: normalize to <N x i1>, bitcast to iN, zero-extend.
		// On WASM: <N x i1> is sub-128-bit (e.g., <4 x i1> = 4 bits) and
		// cannot be lowered by the WASM backend. Use per-lane scalar extract+
		// compare+shift+OR to build the integer mask without any vector of
		// bit-type elements.
		vec := b.getValue(instr.Args[0], getPos(instr))
		if b.spmdIsWASM() {
			// WASM path: extract each lane as i32, test non-zero, shift to
			// its bit position, and OR into a scalar integer result.
			laneCount := vec.Type().VectorSize()
			result := llvm.ConstNull(b.intType)
			i32Type := b.ctx.Int32Type()
			zero := llvm.ConstNull(i32Type)
			for lane := 0; lane < laneCount; lane++ {
				laneIdx := llvm.ConstInt(i32Type, uint64(lane), false)
				elem := b.CreateExtractElement(vec, laneIdx, "")
				if elem.Type() != i32Type {
					// Widen to i32 if needed (e.g., i1 element from LLVM
					// after optimization; rare in practice).
					elem = b.CreateZExt(elem, i32Type, "")
				}
				// Non-zero element means this lane is active.
				active := b.CreateICmp(llvm.IntNE, elem, zero, "")
				bit := b.CreateZExt(active, b.intType, "")
				shift := llvm.ConstInt(b.intType, uint64(lane), false)
				bit = b.CreateShl(bit, shift, "")
				result = b.CreateOr(result, bit, "")
			}
			return result, nil
		}
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

// spmdIsAllocaOriginStatic checks if an SSA value is a stack-allocated array
// (*ssa.Alloc with Heap=false) large enough for a full vector operation.
// Conservative: returns false for anything uncertain.
func spmdIsAllocaOriginStatic(ssaVal ssa.Value, laneCount int) bool {
	if ssaVal == nil {
		return false
	}
	alloc, ok := ssaVal.(*ssa.Alloc)
	if !ok || alloc.Heap {
		return false
	}
	ptrType, ok := alloc.Type().Underlying().(*types.Pointer)
	if !ok {
		return false
	}
	arrType, ok := ptrType.Elem().Underlying().(*types.Array)
	if !ok {
		return false
	}
	return arrType.Len() >= int64(laneCount)
}

// spmdIsAllocaOrigin checks if the source of a contiguous SPMD access is a
// stack-allocated array large enough for a full vector load/store.
func (b *builder) spmdIsAllocaOrigin(ci *spmdContiguousInfo) bool {
	if ci.ssaSource == nil {
		return false
	}
	return spmdIsAllocaOriginStatic(ci.ssaSource, ci.loop.laneCount)
}

// spmdFullLoadWithSelect emits a runtime cap check and, when safe, replaces a
// scalarized llvm.masked.load with a plain v128.load + select(mask, loaded, zero).
// On WASM, llvm.masked.load scalarizes to 4-16 conditional scalar loads. A full
// v128.load + select is only 2 instructions when the slice backing array has
// enough capacity (scalarIter + laneCount <= sliceCap).
//
// Emits:
//
//	iterPlusLanes = scalarIter + laneCount
//	canFullLoad = iterPlusLanes ule sliceCap
//	br canFullLoad, fullBB, maskedBB
//	fullBB: raw = load <N x T>, ptr; result = select(mask, raw, zero); br mergeBB
//	maskedBB: result = masked.load(ptr, mask); br mergeBB
//	mergeBB: phi [fullBB, maskedBB]
func (b *builder) spmdFullLoadWithSelect(vecType llvm.Type, ci *spmdContiguousInfo, mask llvm.Value) llvm.Value {
	// Fast path: alloca origin — stack memory is always fully accessible.
	if b.spmdIsAllocaOrigin(ci) {
		raw := b.CreateLoad(vecType, ci.scalarPtr, "spmd.alloca.load")
		return b.spmdMaskSelect(mask, raw, llvm.ConstNull(vecType))
	}

	laneCount := ci.loop.laneCount

	// Defensive: sliceCap must be valid when the alloca fast-path was not taken.
	if ci.sliceCap.IsNil() {
		panic("spmdFullLoadWithSelect: sliceCap is nil and source is not alloca")
	}

	// Compute scalarIndex + laneCount using the actual GEP index (not raw iter,
	// which may differ when an offset is applied, e.g. output[j*width + i]).
	iterType := ci.scalarIndex.Type()
	laneCountVal := llvm.ConstInt(iterType, uint64(laneCount), false)
	iterPlusLanes := b.CreateAdd(ci.scalarIndex, laneCountVal, "spmd.iter.plus.lanes")

	// Normalize cap to same width as index for comparison.
	capVal := ci.sliceCap
	if capVal.Type() != iterType {
		capWidth := capVal.Type().IntTypeWidth()
		iterWidth := iterType.IntTypeWidth()
		if capWidth < iterWidth {
			capVal = b.CreateZExt(capVal, iterType, "spmd.cap.ext")
		} else {
			capVal = b.CreateTrunc(capVal, iterType, "spmd.cap.trunc")
		}
	}

	// Runtime check: scalarIter + laneCount <= sliceCap (unsigned).
	canFullLoad := b.CreateICmp(llvm.IntULE, iterPlusLanes, capVal, "spmd.can.fullload")

	// Create basic blocks.
	fullBB := b.insertBasicBlock("spmd.fullload")
	maskedBB := b.insertBasicBlock("spmd.maskedload")
	mergeBB := b.insertBasicBlock("spmd.load.merge")

	b.CreateCondBr(canFullLoad, fullBB, maskedBB)

	// Full load path: plain v128.load + select.
	b.SetInsertPointAtEnd(fullBB)
	rawLoad := b.CreateLoad(vecType, ci.scalarPtr, "spmd.fullload.raw")
	zeroinit := llvm.ConstNull(vecType)
	fullResult := b.spmdMaskSelect(mask, rawLoad, zeroinit)
	b.CreateBr(mergeBB)
	fullExitBB := b.GetInsertBlock()

	// Masked load path: existing scalarized fallback.
	b.SetInsertPointAtEnd(maskedBB)
	maskedResult := b.spmdMaskedLoad(vecType, ci.scalarPtr, mask)
	b.CreateBr(mergeBB)
	maskedExitBB := b.GetInsertBlock()

	// Merge with phi.
	b.SetInsertPointAtEnd(mergeBB)
	phi := b.CreatePHI(vecType, "spmd.load.result")
	phi.AddIncoming([]llvm.Value{fullResult, maskedResult}, []llvm.BasicBlock{fullExitBB, maskedExitBB})

	return phi
}

// spmdNarrowLoadElemBits returns the actual in-memory element bit width for a
// narrow contiguous SPMD load on WASM, or 0 if no narrowing is required.
// Narrowing is needed when the SSA element type (e.g., bool = i1 = 1 byte) is
// narrower than the WASM-legal vector element (e.g., i32 = 4 bytes), causing a
// `<N x i32>` load to read N×4 bytes instead of N×1 bytes.
// Returns the per-element memory width in bits (e.g., 8 for bool, 8 for uint8).
func (b *builder) spmdNarrowLoadElemBits(ssaElemType types.Type, laneCount int) uint64 {
	if !b.spmdIsWASM() {
		return 0
	}
	targetLLVMElem := b.getLLVMType(ssaElemType)
	if targetLLVMElem.TypeKind() != llvm.IntegerTypeKind {
		return 0
	}
	// Memory element size (bytes × 8). For i1 (bool): TypeAllocSize = 1 byte.
	targetElemBits := b.targetData.TypeAllocSize(targetLLVMElem) * 8
	if targetElemBits == 0 {
		// Fallback for unusual data layouts where i1 alloc size is 0.
		targetElemBits = uint64(targetLLVMElem.IntTypeWidth())
	}
	// WASM mask element is i32 (32 bits). Only narrow if target is sub-32-bit.
	if targetElemBits >= 32 {
		return 0
	}
	// Total memory footprint must be sub-128-bit for this path to apply.
	totalBits := targetElemBits * uint64(laneCount)
	if totalBits >= 128 {
		return 0
	}
	return targetElemBits
}

// spmdMaskedLoadNarrow loads N elements of targetElemBits bits each from ptr
// and returns a <N x iWide> vector (where iWide = spmdMaskElemType = i32 on WASM)
// with each element zero-extended. This is the inverse of spmdMaskedStoreNarrow.
//
// For example, loading 4 booleans (1 byte each) from a [16]bool array:
//   - Load i32 (4 bytes) from ptr
//   - Extract byte N: (i32 >> (N*8)) & 0xFF
//   - Return <4 x i32> with lane N = extracted byte (0 or non-zero for bool)
//
// The mask is used to gate per-lane loads from arbitrary locations.
// For alloca-origin contiguous accesses, all lanes are always valid (mask = all-ones).
func (b *builder) spmdMaskedLoadNarrow(targetElemBits uint64, ptr llvm.Value, laneCount int, mask llvm.Value) llvm.Value {
	scalarBits := int(targetElemBits) * laneCount
	scalarType := b.ctx.IntType(scalarBits)
	// Load the packed bytes as a single scalar integer.
	packed := b.CreateLoad(scalarType, ptr, "spmd.narrow.load")
	// Unpack per lane: extract targetElemBits-wide slice and zero-extend to i32.
	wideElemType := b.spmdMaskElemType(laneCount) // i32 on WASM
	result := llvm.ConstNull(llvm.VectorType(wideElemType, laneCount))
	elemMask := llvm.ConstInt(scalarType, (1<<targetElemBits)-1, false)
	for lane := 0; lane < laneCount; lane++ {
		shift := llvm.ConstInt(scalarType, uint64(lane)*targetElemBits, false)
		elem := b.CreateLShr(packed, shift, "")
		elem = b.CreateAnd(elem, elemMask, "")
		elem = b.CreateZExt(elem, wideElemType, "")
		laneIdx := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
		result = b.CreateInsertElement(result, elem, laneIdx, "")
	}
	return result
}

// spmdNarrowStoreElemBits returns the target element bit width if val needs
// narrowing before a masked store on WASM, or 0 if no narrowing is required.
// Narrowing is needed when val is a <N x iWide> vector (e.g., <4 x i32> from
// an AND-masked byte in createConvert) but the SSA target element type is
// narrower (e.g., uint8 = 8 bits). This mismatch arises because WASM cannot
// lower <4 x i8> vectors, so createConvert keeps byte values in <4 x i32> and
// relies on the store path to pack them correctly.
func (b *builder) spmdNarrowStoreElemBits(val llvm.Value, ssaElemType types.Type) uint64 {
	if !b.spmdIsWASM() {
		return 0
	}
	if val.Type().TypeKind() != llvm.VectorTypeKind {
		return 0
	}
	// Get the LLVM type the SSA element type would naturally map to.
	targetLLVMElem := b.getLLVMType(ssaElemType)
	if targetLLVMElem.TypeKind() != llvm.IntegerTypeKind {
		return 0
	}
	// Use allocation size (bytes × 8) rather than IntTypeWidth(). For i1 (bool),
	// IntTypeWidth() = 1 bit but TypeAllocSize() = 1 byte = 8 bits — the actual
	// in-memory granularity. This ensures that a [16]bool store correctly uses
	// 8 bits per lane rather than 1 bit per lane.
	targetElemBits := b.targetData.TypeAllocSize(targetLLVMElem) * 8
	if targetElemBits == 0 {
		// Fallback: some unusual data layouts may return 0 for i1.
		targetElemBits = uint64(targetLLVMElem.IntTypeWidth())
	}
	valElemBits := uint64(val.Type().ElementType().IntTypeWidth())
	if valElemBits <= targetElemBits {
		return 0 // no narrowing needed
	}
	// Check that the sub-128-bit constraint applies (targetElemBits < 32).
	totalBits := targetElemBits * uint64(val.Type().VectorSize())
	if totalBits >= 128 {
		return 0 // not a sub-128-bit target; use normal store
	}
	return targetElemBits
}

// spmdMaskedStoreNarrow stores val (a <N x iWide> vector) as N elements of
// targetElemBits bits each to ptr, applying mask per lane.
//
// This is the WASM-safe path for narrowing stores where val's element type is
// wider than the target memory element (e.g., <4 x i32> from an AND-masked
// byte value stored to *uint8 memory). We use spmdWASMNarrowAndPack to convert
// <N x iWide> → i(N*targetElemBits) via scalar extract+AND+shift+OR, avoiding
// any intermediate sub-128-bit vector type that WASM cannot lower.
//
// The mask is applied by loading the existing memory value, blending at the
// packed scalar level, and storing back.
func (b *builder) spmdMaskedStoreNarrow(val llvm.Value, targetElemBits uint64, ptr, mask llvm.Value) {
	laneCount := val.Type().VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	totalBits := int(targetElemBits) * laneCount
	scalarType := b.ctx.IntType(totalBits)

	// Pack the wide vector into a scalar integer using per-element scalar ops.
	scalarVal := b.spmdWASMNarrowAndPack(val, targetElemBits)

	// Load the existing bytes as a scalar integer.
	oldScalar := b.CreateLoad(scalarType, ptr, "spmd.old.narrow")

	// Build a per-lane scalar mask: for lane i, fill bits [i*e..(i+1)*e) with
	// 0xFF...FF if the lane is active, 0x00...00 otherwise.
	scalarMaskVal := llvm.ConstNull(scalarType)
	allBits := llvm.ConstAllOnes(b.ctx.IntType(int(targetElemBits)))
	for lane := 0; lane < laneCount; lane++ {
		laneIdx := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
		laneMask1 := b.CreateExtractElement(i1Mask, laneIdx, "")
		laneMaskElem := b.CreateSelect(laneMask1, allBits,
			llvm.ConstNull(b.ctx.IntType(int(targetElemBits))), "")
		shift := llvm.ConstInt(scalarType, uint64(lane)*targetElemBits, false)
		laneMaskWide := b.CreateZExt(laneMaskElem, scalarType, "")
		laneMaskShifted := b.CreateShl(laneMaskWide, shift, "")
		scalarMaskVal = b.CreateOr(scalarMaskVal, laneMaskShifted, "")
	}

	// Blend: (newVal & mask) | (oldVal & ~mask).
	notMask := b.CreateNot(scalarMaskVal, "")
	blended := b.CreateOr(
		b.CreateAnd(scalarVal, scalarMaskVal, ""),
		b.CreateAnd(oldScalar, notMask, ""), "")
	b.CreateStore(blended, ptr)
}

// spmdMaskedStore calls llvm.masked.store.<suffix>.p0 to store a vector to a scalar pointer with a per-lane mask.
// The mask parameter may be <N x i32> (WASM format) or <N x i1>; it is truncated to <N x i1> as required
// by the LLVM masked intrinsic interface.
//
// On WASM, sub-128-bit vector stores (e.g., <4 x i8> = 32 bits) are handled by packing the
// sub-element-width vector into a scalar integer (bitcast) and storing it as one unit.
// The mask is applied by loading the existing value, blending per-element, and storing back.
func (b *builder) spmdMaskedStore(val, ptr, mask llvm.Value) {
	vecType := val.Type()
	laneCount := vecType.VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)

	// WASM: sub-128-bit vectors (e.g., <4 x i8> = 32 bits) cannot be stored with a
	// SIMD masked store intrinsic. Pack the sub-vector into a scalar integer and use
	// a load-blend-store pattern instead.
	if b.spmdIsWASM() {
		vecBits := uint64(b.targetData.TypeAllocSize(vecType)) * 8
		if vecBits < 128 {
			scalarType := b.ctx.IntType(int(vecBits))
			scalarVal := b.CreateBitCast(val, scalarType, "spmd.pack")
			oldScalar := b.CreateLoad(scalarType, ptr, "spmd.old.pack")
			// Build a scalar mask by OR-ing each active lane bit into the full mask.
			// For 4 lanes of i8: mask is 0x000000FF, 0x0000FF00, etc. for each lane.
			scalarMaskVal := llvm.ConstNull(scalarType)
			elemBits := uint64(b.targetData.TypeAllocSize(vecType.ElementType())) * 8
			allBits := llvm.ConstAllOnes(b.ctx.IntType(int(elemBits)))
			for lane := 0; lane < laneCount; lane++ {
				laneIdx := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
				laneMask1 := b.CreateExtractElement(i1Mask, laneIdx, "")
				// Extend lane mask bit to element width, then shift to position.
				laneMaskElem := b.CreateSelect(laneMask1, allBits, llvm.ConstNull(b.ctx.IntType(int(elemBits))), "")
				shift := llvm.ConstInt(scalarType, uint64(lane)*elemBits, false)
				laneMaskWide := b.CreateZExt(laneMaskElem, scalarType, "")
				laneMaskShifted := b.CreateShl(laneMaskWide, shift, "")
				scalarMaskVal = b.CreateOr(scalarMaskVal, laneMaskShifted, "")
			}
			// Blend: (newVal & mask) | (oldVal & ~mask)
			notMask := b.CreateNot(scalarMaskVal, "")
			blended := b.CreateOr(b.CreateAnd(scalarVal, scalarMaskVal, ""), b.CreateAnd(oldScalar, notMask, ""), "")
			b.CreateStore(blended, ptr)
			return
		}
	}

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

// spmdFullStoreWithBlend emits a runtime cap check and, when safe, replaces a
// scalarized llvm.masked.store with a load-blend-store pattern:
//
//	v128.load(ptr) → v128.bitselect(mask, newVal, oldVal) → v128.store(ptr)
//
// On WASM, llvm.masked.store scalarizes to 4-16 conditional scalar stores. The
// load-blend-store pattern is only 3 instructions when the slice backing array
// has enough capacity (scalarIndex + laneCount <= sliceCap).
//
// Emits:
//
//	iterPlusLanes = scalarIndex + laneCount
//	canFullStore = iterPlusLanes ule sliceCap
//	br canFullStore, blendBB, maskedBB
//	blendBB: old = load ptr; blended = select(mask, val, old); store blended, ptr; br mergeBB
//	maskedBB: masked.store(val, ptr, mask); br mergeBB
//	mergeBB: (void, no phi)
func (b *builder) spmdFullStoreWithBlend(val llvm.Value, ci *spmdContiguousInfo, mask llvm.Value) {
	vecType := val.Type()

	// Fast path: alloca origin — stack memory is always fully accessible.
	if b.spmdIsAllocaOrigin(ci) {
		// WASM: sub-128-bit vectors (e.g., <4 x i8>) cannot use SIMD load/store.
		// Pack to scalar, blend as integer, store back.
		if b.spmdIsWASM() {
			vecBits := uint64(b.targetData.TypeAllocSize(vecType)) * 8
			if vecBits < 128 {
				scalarType := b.ctx.IntType(int(vecBits))
				scalarVal := b.CreateBitCast(val, scalarType, "spmd.pack")
				oldScalar := b.CreateLoad(scalarType, ci.scalarPtr, "spmd.alloca.old.pack")
				// Build scalar mask: OR per-lane bits.
				i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, vecType.VectorSize())
				scalarMaskVal := llvm.ConstNull(scalarType)
				elemBits := uint64(b.targetData.TypeAllocSize(vecType.ElementType())) * 8
				allBits := llvm.ConstAllOnes(b.ctx.IntType(int(elemBits)))
				for lane := 0; lane < vecType.VectorSize(); lane++ {
					laneIdx := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
					laneMask1 := b.CreateExtractElement(i1Mask, laneIdx, "")
					laneMaskElem := b.CreateSelect(laneMask1, allBits, llvm.ConstNull(b.ctx.IntType(int(elemBits))), "")
					shift := llvm.ConstInt(scalarType, uint64(lane)*elemBits, false)
					laneMaskWide := b.CreateZExt(laneMaskElem, scalarType, "")
					laneMaskShifted := b.CreateShl(laneMaskWide, shift, "")
					scalarMaskVal = b.CreateOr(scalarMaskVal, laneMaskShifted, "")
				}
				notMask := b.CreateNot(scalarMaskVal, "")
				blended := b.CreateOr(b.CreateAnd(scalarVal, scalarMaskVal, ""), b.CreateAnd(oldScalar, notMask, ""), "")
				b.CreateStore(blended, ci.scalarPtr)
				return
			}
		}
		old := b.CreateLoad(vecType, ci.scalarPtr, "spmd.alloca.old")
		blended := b.spmdMaskSelect(mask, val, old)
		b.CreateStore(blended, ci.scalarPtr)
		return
	}

	laneCount := ci.loop.laneCount

	// Defensive: sliceCap must be valid when the alloca fast-path was not taken.
	if ci.sliceCap.IsNil() {
		panic("spmdFullStoreWithBlend: sliceCap is nil and source is not alloca")
	}

	// Compute scalarIndex + laneCount using the actual GEP index.
	iterType := ci.scalarIndex.Type()
	laneCountVal := llvm.ConstInt(iterType, uint64(laneCount), false)
	iterPlusLanes := b.CreateAdd(ci.scalarIndex, laneCountVal, "spmd.iter.plus.lanes")

	// Normalize cap to same width as index for comparison.
	capVal := ci.sliceCap
	if capVal.Type() != iterType {
		capWidth := capVal.Type().IntTypeWidth()
		iterWidth := iterType.IntTypeWidth()
		if capWidth < iterWidth {
			capVal = b.CreateZExt(capVal, iterType, "spmd.cap.ext")
		} else {
			capVal = b.CreateTrunc(capVal, iterType, "spmd.cap.trunc")
		}
	}

	// Runtime check: scalarIndex + laneCount <= sliceCap (unsigned).
	canFullStore := b.CreateICmp(llvm.IntULE, iterPlusLanes, capVal, "spmd.can.fullstore")

	// Create basic blocks.
	blendBB := b.insertBasicBlock("spmd.blend")
	maskedBB := b.insertBasicBlock("spmd.maskedstore")
	mergeBB := b.insertBasicBlock("spmd.store.merge")

	b.CreateCondBr(canFullStore, blendBB, maskedBB)

	// Blend path: load existing → select → store.
	b.SetInsertPointAtEnd(blendBB)
	oldVal := b.CreateLoad(vecType, ci.scalarPtr, "spmd.blend.old")
	blended := b.spmdMaskSelect(mask, val, oldVal)
	b.CreateStore(blended, ci.scalarPtr)
	b.CreateBr(mergeBB)

	// Masked store path: existing scalarized fallback.
	b.SetInsertPointAtEnd(maskedBB)
	b.spmdMaskedStore(val, ci.scalarPtr, mask)
	b.CreateBr(mergeBB)

	// Merge (void return, no phi).
	b.SetInsertPointAtEnd(mergeBB)
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
		sliceCap := b.CreateExtractValue(val, 2, "indexaddr.cap")
		bufType := b.getLLVMType(ptrTyp.Elem())
		ptr = b.CreateInBoundsGEP(bufType, bufptr, []llvm.Value{scalarIndex}, "spmd.contiguous.ptr")
		if b.spmdContiguousPtr != nil {
			b.spmdContiguousPtr[expr] = &spmdContiguousInfo{scalarPtr: ptr, loop: loop, sliceCap: sliceCap, scalarIndex: scalarIndex, ssaSource: expr.X}
		}
		return ptr, nil
	default:
		return llvm.Value{}, b.makeError(expr.Pos(), "unsupported contiguous SPMD indexaddr type: "+ptrTyp.String())
	}

	if b.spmdContiguousPtr != nil {
		b.spmdContiguousPtr[expr] = &spmdContiguousInfo{scalarPtr: ptr, loop: loop, scalarIndex: scalarIndex, ssaSource: expr.X}
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

	case token.MUL:
		// (base + offset) * c = base*c + offset*c  (MUL is commutative, so both LHS and RHS work)
		//
		// We compute the new offset as a COMPILE-TIME constant vector to avoid creating
		// <N x i8> multiply IR (WASM has no i8x16.mul). This requires that (N-1)*c fits
		// in i8 (<= 255) so the scaled offsets stay within the byte range.
		//
		// fromBodyIter is NOT propagated: the scaled offset <0, c, 2c, ..., (N-1)*c>
		// breaks the carry-free shift identity used by downstream SHR/AND/REM unless
		// c is a power of two. Conservatively set to false to prevent incorrect code.
		if constVal, ok := scalarSSA.(*ssa.Const); ok {
			if c, ok := constant.Int64Val(constVal.Value); ok && c > 0 {
				maxOffset := int64(laneCount-1) * c
				if maxOffset <= 255 {
					// Scale the scalar base.
					newBase := b.CreateMul(decomp.scalarBase, scalarLLVM, "spmd.decomp.mul.base")
					// Compute scaled offset as a constant vector: <0, c, 2c, ..., (N-1)*c>.
					newOffsetElts := make([]llvm.Value, laneCount)
					for i := 0; i < laneCount; i++ {
						newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(int64(i)*c), false)
					}
					newOffset := llvm.ConstVector(newOffsetElts, false)
					if b.spmdDecomposed != nil {
						b.spmdDecomposed[expr] = &spmdDecomposedIndex{
							scalarBase:    newBase,
							varyingOffset: newOffset,
							laneCount:     laneCount,
							loop:          decomp.loop,
							fromBodyIter:  false,
						}
					}
					return llvm.Value{}, true
				}
			}
		}

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

	case token.QUO:
		if !decompIsLHS {
			break
		}
		// (base + offset) / k where offset = <0,1,...,N-1> and k is a constant.
		// Requires fromBodyIter (base is a multiple of laneCount).
		if !decomp.fromBodyIter {
			break
		}
		if constVal, ok := scalarSSA.(*ssa.Const); ok {
			if k, ok := constant.Int64Val(constVal.Value); ok && k > 0 {
				baseDiv := b.CreateUDiv(decomp.scalarBase, scalarLLVM, "spmd.decomp.quo.base")
				if int64(laneCount)%k == 0 {
					// Fast path: laneCount is a multiple of k, so base is always a
					// multiple of k. The offset vector is a compile-time constant.
					newOffsetElts := make([]llvm.Value, laneCount)
					for i := 0; i < laneCount; i++ {
						newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(i)/uint64(k), false)
					}
					newOffset := llvm.ConstVector(newOffsetElts, false)
					if b.spmdDecomposed != nil {
						b.spmdDecomposed[expr] = &spmdDecomposedIndex{
							scalarBase:    baseDiv,
							varyingOffset: newOffset,
							laneCount:     laneCount,
							loop:          decomp.loop,
						}
					}
					return llvm.Value{}, true
				}
				// General path: laneCount % k != 0 (e.g., 16-lane byte loop with k=3).
				// base may not be a multiple of k, so (base+lane)/k != base/k + lane/k.
				// Use: (base+lane)/k = base/k + (base%k + lane)/k
				// Since base%k ∈ [0, k-1], precompute k constant offset vectors
				// (one per remainder) and select at runtime.
				// Guard: fall through to materialization for very large k to avoid
				// generating k pattern vectors and k-1 cascading selects.
				if k > int64(laneCount) {
					break
				}
				baseRem := b.CreateURem(decomp.scalarBase, scalarLLVM, "spmd.decomp.quo.rem")
				// Precompute offset vectors for each remainder r ∈ [0, k-1]:
				//   pattern_r[lane] = (r + lane) / k
				patterns := make([]llvm.Value, k)
				for r := int64(0); r < k; r++ {
					elts := make([]llvm.Value, laneCount)
					for lane := 0; lane < laneCount; lane++ {
						elts[lane] = llvm.ConstInt(i8Type, uint64((r+int64(lane))/k), false)
					}
					patterns[r] = llvm.ConstVector(elts, false)
				}
				// Select the right pattern based on base%k using cascading selects.
				newOffset := patterns[0]
				i1VecType := llvm.VectorType(b.ctx.Int1Type(), laneCount)
				for r := int64(1); r < k; r++ {
					cmpVal := llvm.ConstInt(baseRem.Type(), uint64(r), false)
					cmp := b.CreateICmp(llvm.IntEQ, baseRem, cmpVal, "spmd.decomp.quo.cmp")
					cmpVec := b.splatScalar(cmp, i1VecType)
					newOffset = b.CreateSelect(cmpVec, patterns[r], newOffset, "spmd.decomp.quo.sel")
				}
				if b.spmdDecomposed != nil {
					b.spmdDecomposed[expr] = &spmdDecomposedIndex{
						scalarBase:    baseDiv,
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
		// We compute the new offset as a COMPILE-TIME constant vector to avoid
		// creating <N x i8> remainder IR which WASM SIMD cannot lower (no i8x16.rem).
		if !decomp.fromBodyIter {
			break
		}
		if constVal, ok := scalarSSA.(*ssa.Const); ok {
			if k, ok := constant.Int64Val(constVal.Value); ok && k > 0 {
				// Base contribution: base % k (scalar).
				baseRem := b.CreateURem(decomp.scalarBase, scalarLLVM, "spmd.decomp.rem.base")
				if int64(laneCount)%k == 0 {
					// Fast path: laneCount is a multiple of k. The offset pattern
					// repeats cleanly and base%k is always 0 (since base is a multiple
					// of laneCount which is divisible by k).
					newOffsetElts := make([]llvm.Value, laneCount)
					for i := 0; i < laneCount; i++ {
						newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(i)%uint64(k), false)
					}
					newOffset := llvm.ConstVector(newOffsetElts, false)
					if b.spmdDecomposed != nil {
						b.spmdDecomposed[expr] = &spmdDecomposedIndex{
							scalarBase:    baseRem,
							varyingOffset: newOffset,
							laneCount:     laneCount,
							loop:          decomp.loop,
						}
					}
					return llvm.Value{}, true
				}
				// General path: laneCount % k != 0 (e.g., 16-lane byte loop with k=3).
				// Use: (base+lane) % k = (base%k + lane) % k
				// Since base%k ∈ [0, k-1], precompute k constant offset vectors
				// (one per remainder) and select at runtime.
				// Guard: fall through to materialization for very large k.
				if k > int64(laneCount) {
					break
				}
				patterns := make([]llvm.Value, k)
				for r := int64(0); r < k; r++ {
					elts := make([]llvm.Value, laneCount)
					for lane := 0; lane < laneCount; lane++ {
						elts[lane] = llvm.ConstInt(i8Type, uint64((r+int64(lane))%k), false)
					}
					patterns[r] = llvm.ConstVector(elts, false)
				}
				newOffset := patterns[0]
				i1VecType := llvm.VectorType(b.ctx.Int1Type(), laneCount)
				for r := int64(1); r < k; r++ {
					cmpVal := llvm.ConstInt(baseRem.Type(), uint64(r), false)
					cmp := b.CreateICmp(llvm.IntEQ, baseRem, cmpVal, "spmd.decomp.rem.cmp")
					cmpVec := b.splatScalar(cmp, i1VecType)
					newOffset = b.CreateSelect(cmpVec, patterns[r], newOffset, "spmd.decomp.rem.sel")
				}
				// REM result has scalarBase=0 since all information is in the offset.
				zeroBase := llvm.ConstInt(decomp.scalarBase.Type(), 0, false)
				if b.spmdDecomposed != nil {
					b.spmdDecomposed[expr] = &spmdDecomposedIndex{
						scalarBase:    zeroBase,
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

	// Fallback: materialize to <N x i32> and return without applying the operation.
	// On WASM this produces a 512-bit wide vector that may exceed the target's native
	// vector width. Arithmetic operations on <16 x i32> cause LLVM legalization errors,
	// so the specific BinOp cases above should handle all operations that need computation.
	// This fallback is only reached for operations that are handled upstream (comparisons
	// return via the comparison case above) or should not occur in practice.
	//
	// NOTE: If a new BinOp reaches this fallback and produces incorrect results,
	// add a dedicated case above (like QUO/REM) that stays in the narrow <N x i8> lane width.
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

	// SPMD: clamp inactive lane offsets to 0 to prevent out-of-bounds access from
	// masked-out lanes (e.g., tail iterations where laneCount > remaining elements).
	// This ensures both bounds checks and InBoundsGEP only see valid indices.
	safeOffset := decomp.varyingOffset
	if expr.SPMDMask != nil {
		mask := b.getValue(expr.SPMDMask, getPos(expr))
		if !b.spmdIsConstAllOnesMask(mask) {
			maskI1 := b.CreateTrunc(mask, llvm.VectorType(b.ctx.Int1Type(), laneCount), "spmd.offset.mask")
			zeros := llvm.ConstNull(decomp.varyingOffset.Type())
			safeOffset = b.CreateSelect(maskI1, decomp.varyingOffset, zeros, "spmd.offset.clamp")
		}
	}

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
			if safeOffset.IsConstant() {
				// OPTIMIZED: constant offset vector → single scalar bounds check.
				// Find the maximum offset element at compile time, then check:
				//   (scalarBase + maxOffset) >= buflen → panic
				// Correctness: maxOffset >= every per-lane offset, so if
				// (base + maxOffset) is in bounds then all lanes are in bounds.
				maxOffset := uint64(0)
				for lane := 0; lane < laneCount; lane++ {
					elem := llvm.ConstExtractElement(safeOffset,
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
					offsetByte := b.CreateExtractElement(safeOffset,
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
		offsetByte := b.CreateExtractElement(safeOffset,
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

	// SPMD: clamp inactive lane indices to 0 using SSA-level mask.
	if expr.SPMDMask != nil {
		mask := b.getValue(expr.SPMDMask, getPos(expr))
		if !b.spmdIsConstAllOnesMask(mask) {
			maskI1 := b.CreateTrunc(mask, llvm.VectorType(b.ctx.Int1Type(), laneCount), "spmd.idx.mask")
			zeros := llvm.ConstNull(index.Type())
			index = b.CreateSelect(maskI1, index, zeros, "spmd.idx.clamp")
		}
	}

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
	// On WASM, build the result in the mask element type (e.g., <4 x i32>) to avoid
	// sub-128-bit vectors (e.g., <4 x i8> = 32 bits) which WASM cannot lower.
	// Each loaded i8 byte is zero-extended to the wider element type during insertion.
	bufElemType := b.ctx.Int8Type()
	resultElemType := bufElemType
	if b.spmdIsWASM() {
		vecBits := uint64(b.targetData.TypeAllocSize(llvm.VectorType(bufElemType, laneCount))) * 8
		if vecBits < 128 {
			resultElemType = b.spmdMaskElemType(laneCount)
		}
	}
	result := llvm.Undef(llvm.VectorType(resultElemType, laneCount))
	for lane := 0; lane < laneCount; lane++ {
		ptr := b.CreateInBoundsGEP(bufElemType, buf, []llvm.Value{laneIdxs[lane]}, "")
		val := b.CreateLoad(bufElemType, ptr, "")
		if resultElemType != bufElemType {
			val = b.CreateZExt(val, resultElemType, "")
		}
		result = b.CreateInsertElement(result, val, llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
	}
	return result, nil
}

// spmdVectorIndexArray performs per-lane element extraction from an array using a vector index.
// The array is spilled to an alloca, then each lane does GEP+load.
func (b *builder) spmdVectorIndexArray(expr *ssa.Index, collection, index llvm.Value) (llvm.Value, error) {
	laneCount := index.Type().VectorSize()
	xType := expr.X.Type().Underlying().(*types.Array)

	// SPMD: clamp inactive lane indices to 0 using SSA-level mask.
	if expr.SPMDMask != nil {
		mask := b.getValue(expr.SPMDMask, getPos(expr))
		if !b.spmdIsConstAllOnesMask(mask) {
			maskI1 := b.CreateTrunc(mask, llvm.VectorType(b.ctx.Int1Type(), laneCount), "spmd.idx.mask")
			zeros := llvm.ConstNull(index.Type())
			index = b.CreateSelect(maskI1, index, zeros, "spmd.idx.clamp")
		}
	}

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
	// On WASM, build the result in the mask element type (e.g., <4 x i32>) to avoid
	// sub-128-bit vectors (e.g., <4 x i8> = 32 bits) which WASM cannot lower.
	// Each loaded byte is zero-extended to the wider element type during insertion
	// rather than after, so no intermediate sub-128-bit vector is ever created.
	elemType := arrayType.ElementType()
	resultElemType := elemType
	if b.spmdIsWASM() {
		vecBits := uint64(b.targetData.TypeAllocSize(llvm.VectorType(elemType, laneCount))) * 8
		if vecBits < 128 {
			resultElemType = b.spmdMaskElemType(laneCount)
		}
	}
	result := llvm.Undef(llvm.VectorType(resultElemType, laneCount))
	zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
	for lane := 0; lane < laneCount; lane++ {
		ptr := b.CreateInBoundsGEP(arrayType, alloca, []llvm.Value{zero, laneIdxs[lane]}, "index.gep")
		val := b.CreateLoad(elemType, ptr, "index.load")
		if resultElemType != elemType {
			val = b.CreateZExt(val, resultElemType, "")
		}
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

// spmdEmitInterleavedStoreMasked handles the last store (remainder == stride-1) in a
// stride-S interleaved store group. It collects values already saved for
// remainders 0..S-2 from spmdInterleavedValues, shuffles them into S output
// vectors in interleaved order, computes the base pointer into the destination
// slice, and emits S masked stores.
//
// lastVal is the SSA value being stored (from Val field of the last store).
// lastAddr is the SSA address (from Addr field of the last store).
// mask is the execution mask for the store; if nil/zero, all-ones is used.
func (b *builder) spmdEmitInterleavedStoreMasked(lastVal ssa.Value, lastAddr ssa.Value, info *spmdInterleavedStoreInfo, mask llvm.Value) {
	group := info.group
	stride := group.stride
	N := group.laneCount

	// Collect all S values: 0..S-2 were saved earlier; S-1 comes from lastVal.
	vals := make([]llvm.Value, stride)
	saved := b.spmdInterleavedValues[group]
	for r := 0; r < stride-1; r++ {
		if saved == nil || saved[r].IsNil() {
			// Earlier remainder was never saved — fall back gracefully.
			return
		}
		vals[r] = saved[r]
	}
	vals[stride-1] = b.getValue(lastVal, token.NoPos)

	// Shuffle values into S interleaved output vectors.
	var outVecs []llvm.Value
	switch stride {
	case 2:
		ov := b.spmdInterleaveStride2(vals[0], vals[1], N)
		outVecs = ov[:]
	case 3:
		ov := b.spmdInterleaveStride3(vals[0], vals[1], vals[2], N)
		outVecs = ov[:]
	case 4:
		ov := b.spmdInterleaveStride4(vals[0], vals[1], vals[2], vals[3], N)
		outVecs = ov[:]
	default:
		return // unsupported stride
	}

	// Compute the base pointer for the output slice.
	// decomp.scalarBase for remainder-0 addr is already iter*stride (the scalar
	// part of the decomposed index). GEP from the slice data pointer at that offset.
	addr0 := group.addrs[0]
	baseSliceVal := b.getValue(addr0.X, getPos(addr0))

	var bufptr llvm.Value
	var elemType llvm.Type
	var buflen llvm.Value

	switch ptrTyp := addr0.X.Type().Underlying().(type) {
	case *types.Slice:
		bufptr = b.CreateExtractValue(baseSliceVal, 0, "interleaved.ptr")
		buflen = b.CreateExtractValue(baseSliceVal, 1, "interleaved.len")
		elemType = b.getLLVMType(ptrTyp.Elem())
	case *types.Pointer:
		typ := ptrTyp.Elem().Underlying()
		switch arr := typ.(type) {
		case *types.Array:
			bufptr = baseSliceVal
			buflen = llvm.ConstInt(b.uintptrType, uint64(arr.Len()), false)
			elemType = b.getLLVMType(arr.Elem())
		default:
			return
		}
	default:
		return
	}

	// Retrieve the scalar base index (iter*stride) from the decomposed index map.
	// This is the scalar component of the decomposed index for addr0.Index.
	// Fall back to zero if not found (shouldn't happen for well-formed groups).
	scalarBase := llvm.Value{}
	if b.spmdDecomposed != nil {
		if decomp, ok := b.spmdDecomposed[addr0.Index]; ok {
			scalarBase = decomp.scalarBase
		}
	}
	if scalarBase.IsNil() {
		scalarBase = llvm.ConstInt(b.uintptrType, 0, false)
	}
	scalarBase = b.extendInteger(scalarBase, addr0.Index.Type(), b.uintptrType)

	// Bounds check: ensure scalarBase + stride*laneCount <= len(slice).
	if !b.info.nobounds && !buflen.IsNil() {
		endIdx := b.CreateAdd(scalarBase,
			llvm.ConstInt(b.uintptrType, uint64(stride*N), false),
			"interleaved.end")
		oob := b.CreateICmp(llvm.IntUGT, endIdx, buflen, "interleaved.oob")
		b.createRuntimeAssert(oob, "lookup", "lookupPanic")
	}

	// Get the execution mask and expand it for each of the S output stores.
	if mask.IsNil() {
		mask = llvm.ConstAllOnes(llvm.VectorType(b.spmdMaskElemType(N), N))
	}
	expandedMasks := b.spmdExpandMaskForStride(mask, stride, N)

	// Emit S masked stores: outVecs[k] → bufptr[scalarBase + k*N].
	for k := 0; k < stride; k++ {
		offset := llvm.ConstInt(b.uintptrType, uint64(k*N), false)
		ptr := b.CreateInBoundsGEP(elemType, bufptr,
			[]llvm.Value{b.CreateAdd(scalarBase, offset, "interleaved.off")},
			"interleaved.store.ptr")
		b.spmdMaskedStore(outVecs[k], ptr, expandedMasks[k])
	}
}

// spmdInterleaveStride2 builds two <N x T> output vectors by interleaving
// val0 (even positions) and val1 (odd positions).
//
// The N source lanes produce 2*N output bytes laid out as:
//
//	[v0[0], v1[0], v0[1], v1[1], ..., v0[N-1], v1[N-1]]
//
// Split into two output vectors of N elements each:
//
//	out[0] = [v0[0], v1[0], v0[1], v1[1], ..., v0[N/2-1], v1[N/2-1]]
//	out[1] = [v0[N/2], v1[N/2], v0[N/2+1], v1[N/2+1], ..., v0[N-1], v1[N-1]]
//
// Each output vector is produced by a single CreateShuffleVector where indices
// >= N refer to elements of the second input operand.
func (b *builder) spmdInterleaveStride2(val0, val1 llvm.Value, N int) [2]llvm.Value {
	// N must be even; all WASM SIMD lane counts (4, 8, 16) satisfy this.
	if N%2 != 0 {
		panic(fmt.Sprintf("spmdInterleaveStride2: N must be even, got %d", N))
	}
	// lo output: interleave first halves.
	// lo[j] = v0[j/2]  if j is even, v1[j/2]  if j is odd  (j in [0, N/2*2))
	// Out index j: even → val0[j/2], odd → val1[(j-1)/2]
	// In shufflevector terms: val0 occupies indices [0, N) and val1 occupies [N, 2N).
	loMask := make([]uint64, N)
	hiMask := make([]uint64, N)
	for j := 0; j < N/2; j++ {
		loMask[2*j] = uint64(j)       // val0[j]
		loMask[2*j+1] = uint64(N + j) // val1[j]
	}
	for j := 0; j < N/2; j++ {
		hiMask[2*j] = uint64(N/2 + j)       // val0[N/2+j]
		hiMask[2*j+1] = uint64(N + N/2 + j) // val1[N/2+j]
	}
	lo := b.CreateShuffleVector(val0, val1, b.spmdShuffleConst(loMask), "interleave2.lo")
	hi := b.CreateShuffleVector(val0, val1, b.spmdShuffleConst(hiMask), "interleave2.hi")
	return [2]llvm.Value{lo, hi}
}

// spmdInterleaveStride3 builds three <N x T> output vectors by interleaving
// val0, val1, val2 (positions 0, 1, 2 within each triplet respectively).
//
// The N source lanes produce 3*N output bytes:
//
//	[v0[0],v1[0],v2[0], v0[1],v1[1],v2[1], ..., v0[N-1],v1[N-1],v2[N-1]]
//
// This is split into three N-element output vectors. For each output k (0..2)
// and position j (0..N-1): global position = k*N+j, triplet = (k*N+j)/3,
// slot = (k*N+j)%3 selects from val0/val1/val2.
//
// Each output is produced in two shuffle steps: first merge val0 and val1 for
// the positions they occupy, leaving val2 positions as undef; then overlay val2.
func (b *builder) spmdInterleaveStride3(val0, val1, val2 llvm.Value, N int) [3]llvm.Value {
	var out [3]llvm.Value

	for k := 0; k < 3; k++ {
		// Step 1: shuffle val0 and val1 into an N-element intermediate; val2 positions
		// use index 0 as placeholder (they will be overwritten in step 2).
		mask01 := make([]uint64, N)
		// Step 2: mask to overlay val2 values onto the result of step 1.
		mask2 := make([]uint64, N)

		for j := 0; j < N; j++ {
			globalPos := k*N + j
			triplet := globalPos / 3
			slot := globalPos % 3
			switch slot {
			case 0:
				// val0[triplet]: shufflevector index triplet (first input = val0 → indices 0..N-1)
				mask01[j] = uint64(triplet)
				mask2[j] = uint64(j) // keep from step1 (first input in step2 = step1 result)
			case 1:
				// val1[triplet]: shufflevector index N+triplet (second input = val1 → indices N..2N-1)
				mask01[j] = uint64(N + triplet)
				mask2[j] = uint64(j) // keep from step1
			case 2:
				// val2[triplet]: not present in step1; use placeholder, then override in step2.
				mask01[j] = 0                  // placeholder (will be replaced)
				mask2[j] = uint64(N + triplet) // second input in step2 = val2 → indices N..2N-1
			}
		}

		step1 := b.CreateShuffleVector(val0, val1, b.spmdShuffleConst(mask01),
			fmt.Sprintf("interleave3.k%d.step1", k))
		out[k] = b.CreateShuffleVector(step1, val2, b.spmdShuffleConst(mask2),
			fmt.Sprintf("interleave3.k%d", k))
	}
	return out
}

// spmdInterleaveStride4 builds four <N x T> output vectors by interleaving
// val0, val1, val2, val3 (positions 0, 1, 2, 3 within each group of four).
//
// Uses a butterfly (two-level stride-2) approach:
//
//	Level 1:
//	  pair02[lo,hi] = stride2(val0, val2)   — interleaves 0-indexed and 2-indexed
//	  pair13[lo,hi] = stride2(val1, val3)   — interleaves 1-indexed and 3-indexed
//	Level 2:
//	  out[0,1] = stride2(pair02.lo, pair13.lo)
//	  out[2,3] = stride2(pair02.hi, pair13.hi)
func (b *builder) spmdInterleaveStride4(val0, val1, val2, val3 llvm.Value, N int) [4]llvm.Value {
	pair02 := b.spmdInterleaveStride2(val0, val2, N)
	pair13 := b.spmdInterleaveStride2(val1, val3, N)
	out01 := b.spmdInterleaveStride2(pair02[0], pair13[0], N)
	out23 := b.spmdInterleaveStride2(pair02[1], pair13[1], N)
	return [4]llvm.Value{out01[0], out01[1], out23[0], out23[1]}
}

// spmdExpandMaskForStride expands a source execution mask of N lanes into
// S output masks, each N lanes wide, for stride-S interleaved stores.
//
// For stride S, output k, position j: the corresponding source lane is
// (k*N + j) / S. The expanded mask for output k has its j-th element equal to
// the source mask at lane (k*N+j)/S.
//
// The source mask is <N x maskElemType> (i32 on WASM). Each output mask has the
// same element type and N elements.
func (b *builder) spmdExpandMaskForStride(mask llvm.Value, stride, N int) []llvm.Value {
	out := make([]llvm.Value, stride)
	undef := llvm.Undef(mask.Type())

	for k := 0; k < stride; k++ {
		expandIdx := make([]uint64, N)
		for j := 0; j < N; j++ {
			srcLane := (k*N + j) / stride
			expandIdx[j] = uint64(srcLane)
		}
		out[k] = b.CreateShuffleVector(mask, undef, b.spmdShuffleConst(expandIdx),
			fmt.Sprintf("interleaved.mask.k%d", k))
	}
	return out
}

// createSPMDSelect emits LLVM IR for an SPMDSelect instruction.
// SPMDSelect yields X where Mask is active, Y where inactive, per SIMD lane.
// This is the predicated replacement for Phi at varying merge points.
//
// The mask is in platform-native Varying[mask] format (e.g., <4 x i32> on WASM).
// spmdBroadcastMatch aligns x and y; spmdMaskSelect handles mask-vs-data
// width mismatches internally (falls back to trunc+CreateSelect on WASM
// when bitwise select conditions are not met).
func (b *builder) createSPMDSelect(instr *ssa.SPMDSelect) llvm.Value {
	mask := b.getValue(instr.Mask, token.NoPos)
	x := b.getValue(instr.X, token.NoPos)
	y := b.getValue(instr.Y, token.NoPos)

	// Broadcast scalar operands to vector when needed (e.g., uniform constants).
	x, y = b.spmdBroadcastMatch(x, y)

	return b.spmdMaskSelect(mask, x, y)
}

// createSPMDLoad emits LLVM IR for an SPMDLoad instruction.
// SPMDLoad loads from Addr only for lanes where Mask is active.
// Inactive lanes receive a zero value. It is the predicated replacement
// for UnOp{MUL} (pointer dereference) in varying paths.
//
// For scalar addresses (uniform pointer), the load is speculative (executed
// unconditionally) and inactive-lane results are zeroed via mask-select.
// For vector addresses (varying pointers), a masked gather is used.
func (b *builder) createSPMDLoad(instr *ssa.SPMDLoad) llvm.Value {
	addr := b.getValue(instr.Addr, instr.Pos())
	mask := b.getValue(instr.Mask, instr.Pos())

	// Determine the result type from the SSA result type.
	resultType := b.getLLVMType(instr.Type())

	// Derive lane count from the mask vector, which always has the correct
	// target-specific width. instr.Lanes may use host int sizes.
	laneCount := mask.Type().VectorSize()

	// Shifted-contiguous access: e.g. src[i>>1] in a 16-lane rangeindex loop.
	// TinyGo detects the shift pattern during IndexAddr compilation and records
	// it in spmdShiftedPtr. Use spmdShiftedLoad which performs a narrow
	// contiguous load + shufflevector expansion rather than a full gather.
	if b.spmdShiftedPtr != nil {
		if info, ok := b.spmdShiftedPtr[instr.Addr]; ok {
			return b.spmdShiftedLoad(info, mask)
		}
	}

	// Contiguous access: use vector load instead of scalar load+broadcast.
	// Two sources of contiguity info:
	//   1. SSA-level: instr.Contiguous (set by spmdMaskMemOps in go/ssa)
	//   2. TinyGo-level: spmdContiguousPtr map (populated during IndexAddr compilation)
	// Both agree for go-for loops. The SSA field provides visibility in SSA dumps
	// and serves as a safety net for future backends. The map provides the LLVM
	// values (scalarPtr, sliceCap) needed for actual codegen.
	if b.spmdContiguousPtr != nil {
		if ci, ok := b.spmdContiguousPtr[instr.Addr]; ok {
			ssaElemType := instr.Addr.Type().Underlying().(*types.Pointer).Elem()
			elemType := b.getLLVMType(ssaElemType)

			// Narrow load path (WASM byte/bool elements).
			if narrowBits := b.spmdNarrowLoadElemBits(ssaElemType, laneCount); narrowBits > 0 {
				return b.spmdMaskedLoadNarrow(narrowBits, ci.scalarPtr, laneCount, mask)
			}
			// Bool load fix: load as <N x i8>, truncate to <N x i1>.
			isBoolLoad := elemType == b.ctx.Int1Type()
			if isBoolLoad {
				elemType = b.ctx.Int8Type()
			}
			// WASM sub-128-bit widening.
			if b.spmdIsWASM() {
				vecBits := uint64(b.targetData.TypeAllocSize(elemType)) * 8 * uint64(laneCount)
				if vecBits < 128 {
					elemType = b.spmdMaskElemType(laneCount)
				}
			}
			vecType := llvm.VectorType(elemType, laneCount)
			// Cap-based optimization: full load + select when safe.
			var result llvm.Value
			if !b.spmdIsConstAllOnesMask(mask) && (b.spmdIsAllocaOrigin(ci) || !ci.sliceCap.IsNil()) {
				result = b.spmdFullLoadWithSelect(vecType, ci, mask)
				b.currentBlockInfo.exit = b.GetInsertBlock()
			} else {
				result = b.spmdMaskedLoad(vecType, ci.scalarPtr, mask)
			}
			if isBoolLoad {
				result = b.CreateTrunc(result, llvm.VectorType(b.ctx.Int1Type(), laneCount), "")
			}
			return result
		}
	}

	// If the address is a vector (varying pointers), emit a masked gather.
	// spmdMaskedGather expects a vector result type, not a scalar. Build the
	// vector type from the scalar resultType and the lane count derived from
	// the address vector (which has the correct LLVM lane count for this target).
	if addr.Type().TypeKind() == llvm.VectorTypeKind {
		addrLaneCount := addr.Type().VectorSize()
		vecResultType := resultType
		if resultType.TypeKind() != llvm.VectorTypeKind {
			vecResultType = llvm.VectorType(resultType, addrLaneCount)
		}
		return b.spmdMaskedGather(vecResultType, addr, mask)
	}

	// Scalar address: load speculatively, then broadcast and mask.
	// Safety: the predication pass only generates SPMDLoad inside blocks that
	// were statically reachable in the pre-predication CFG, so the pointer was
	// valid for all lanes in that block. This assumes flat (non-nested) if
	// linearization; nested ifs require mask threading to remain safe.
	loaded := b.CreateLoad(resultType, addr, "spmd.load")

	// If the result is scalar, broadcast to a vector then mask-select.
	if resultType.TypeKind() != llvm.VectorTypeKind {
		vecType := llvm.VectorType(resultType, laneCount)
		splatted := b.splatScalar(loaded, vecType)
		zero := llvm.ConstNull(vecType)
		return b.spmdMaskSelect(mask, splatted, zero)
	}

	// Result is already a vector (loaded a vector from memory).
	zero := llvm.ConstNull(resultType)
	return b.spmdMaskSelect(mask, loaded, zero)
}

// createSPMDStore emits LLVM IR for an SPMDStore instruction.
// SPMDStore stores Val to Addr only for lanes where Mask is active.
// It is the predicated replacement for Store in varying paths.
//
// For scalar addresses (uniform pointer), a masked store is emitted; spmdMaskedStore
// handles mask unwrapping internally. For vector addresses (varying pointers),
// a masked scatter is used; spmdMaskedScatter handles mask unwrapping internally.
func (b *builder) createSPMDStore(instr *ssa.SPMDStore) {
	// SPMD: interleaved stride-S store handling.
	// First S-1 remainders save their values; last remainder emits interleaved stores.
	if b.spmdInterleavedStores != nil {
		if info, ok := b.spmdInterleavedStores[instr]; ok {
			// Skip interleaved store in tail body phase of SSA-peeled loops — let
			// normal scatter path handle it. The interleaved scalarBase is wrong
			// for the tail.
			inTail := false
			if b.spmdLoopState != nil {
				if loop, ok := b.spmdLoopState.bodyBlocks[b.currentBlock.Index]; ok && loop.isPeeled {
					inTail = b.currentBlock.Index == loop.ssaLoopInfo.TailBodyBlock.Index
				}
			}
			if !inTail {
				mask := b.getValue(instr.Mask, instr.Pos())
				if info.remainder < info.group.stride-1 {
					vals := b.spmdInterleavedValues[info.group]
					if vals == nil {
						vals = make([]llvm.Value, info.group.stride)
						b.spmdInterleavedValues[info.group] = vals
					}
					vals[info.remainder] = b.getValue(instr.Val, instr.Pos())
					return
				}
				b.spmdEmitInterleavedStoreMasked(instr.Val, instr.Addr, info, mask)
				return
			}
		}
	}

	addr := b.getValue(instr.Addr, instr.Pos())
	val := b.getValue(instr.Val, instr.Pos())
	mask := b.getValue(instr.Mask, instr.Pos())

	// The predicated SSA pass copies Store operands directly.
	// When the original Store was inside a varying if/else, the SSA types
	// may still be scalar (e.g., int32, *int32). SPMDStore needs vector
	// operands, so splat scalar values using the active loop's lane count.
	// Derive lane count from the mask vector, which always has the correct
	// target-specific width. instr.Lanes may use host int sizes (e.g., 8 bytes
	// on amd64 host) rather than target sizes (4 bytes on WASM).
	laneCount := mask.Type().VectorSize()
	if val.Type().TypeKind() != llvm.VectorTypeKind {
		// Scalar value: splat to vector.
		val = b.splatScalar(val, llvm.VectorType(val.Type(), laneCount))
	}

	// Contiguous access: use vector store instead of scatter.
	// See createSPMDLoad for contiguity info sources (SSA-level + TinyGo-level).
	if b.spmdContiguousPtr != nil {
		if ci, ok := b.spmdContiguousPtr[instr.Addr]; ok {
			// Bool store fix.
			if val.Type().TypeKind() == llvm.VectorTypeKind &&
				val.Type().ElementType() == b.ctx.Int1Type() {
				val = b.CreateZExt(val, llvm.VectorType(b.ctx.Int8Type(), laneCount), "")
			}
			// Narrowing detection.
			var narrowBits uint64
			if addrPtrType, ok := instr.Addr.Type().Underlying().(*types.Pointer); ok {
				narrowBits = b.spmdNarrowStoreElemBits(val, addrPtrType.Elem())
			}
			if narrowBits == 0 {
				if spmdVal, ok := instr.Val.Type().(*types.SPMDType); ok {
					narrowBits = b.spmdNarrowStoreElemBits(val, spmdVal.Elem())
				}
			}
			if narrowBits > 0 {
				b.spmdMaskedStoreNarrow(val, narrowBits, ci.scalarPtr, mask)
			} else if !b.spmdIsConstAllOnesMask(mask) && (b.spmdIsAllocaOrigin(ci) || !ci.sliceCap.IsNil()) {
				b.spmdFullStoreWithBlend(val, ci, mask)
				b.currentBlockInfo.exit = b.GetInsertBlock()
			} else {
				b.spmdMaskedStore(val, ci.scalarPtr, mask)
			}
			return
		}
	}

	// If the address is a vector (varying pointers), emit a masked scatter.
	// spmdMaskedScatter handles mask format unwrapping internally.
	if addr.Type().TypeKind() == llvm.VectorTypeKind {
		b.spmdMaskedScatter(val, addr, mask)
		return
	}

	// Scalar address: use a masked store. spmdMaskedStore handles mask
	// format unwrapping (WASM <N x i32> → <N x i1>) internally.
	b.spmdMaskedStore(val, addr, mask)
}

// createSPMDIndex emits LLVM IR for an SPMDIndex instruction.
// SPMDIndex produces consecutive lane indices [0, 1, ..., Lanes-1]
// in the loop's natural element type. It is the predicated replacement
// for lanes.Index() calls inside SPMD loops.
//
// The result is a constant vector <0, 1, 2, ..., Lanes-1>.
func (b *builder) createSPMDIndex(instr *ssa.SPMDIndex) llvm.Value {
	lanes := instr.Lanes
	elemType := b.getLLVMType(instr.ElemType)
	// Reuse the existing spmdLaneOffsetConst helper for consistency.
	return b.spmdLaneOffsetConst(lanes, elemType)
}

// createTypeAssertSPMD handles type assertions to lanes.Varying[T].
// The boxed representation uses [N]T array (matching getTypeCode), but the
// SSA result type is <N x T> vector. This function compares against the array
// type code, extracts the array value, and converts it to a vector.
func (b *builder) createTypeAssertSPMD(itf llvm.Value, expr *ssa.TypeAssert, spmdType *types.SPMDType, vecType llvm.Type) llvm.Value {
	// Build the array type that matches the type code used in boxing.
	elemLLVM := b.getLLVMType(spmdType.Elem())
	laneCount := b.spmdEffectiveLaneCount(spmdType, elemLLVM)
	arrGoType := types.NewArray(spmdType.Elem(), int64(laneCount))
	arrLLVMType := b.getLLVMType(arrGoType)

	// Compare type codes using the array type (same as boxing path in MakeInterface).
	actualTypeNum := b.CreateExtractValue(itf, 0, "interface.type")
	name, _ := getTypeCodeName(arrGoType)
	globalName := "reflect/types.typeid:" + name
	assertedTypeCodeGlobal := b.mod.NamedGlobal(globalName)
	if assertedTypeCodeGlobal.IsNil() {
		assertedTypeCodeGlobal = llvm.AddGlobal(b.mod, b.ctx.Int8Type(), globalName)
		assertedTypeCodeGlobal.SetGlobalConstant(true)
	}
	commaOk := b.createRuntimeCall("typeAssert", []llvm.Value{actualTypeNum, assertedTypeCodeGlobal}, "typecode")

	// Branch on type match to avoid speculative extraction before the check.
	prevBlock := b.GetInsertBlock()
	okBlock := b.insertBasicBlock("typeassert.spmd.ok")
	nextBlock := b.insertBasicBlock("typeassert.spmd.next")
	b.currentBlockInfo.exit = nextBlock
	b.CreateCondBr(commaOk, okBlock, nextBlock)

	// OK block: extract the [N]T array from the interface, then convert to <N x T> vector.
	b.SetInsertPointAtEnd(okBlock)
	arrValue := b.extractValueFromInterface(itf, arrLLVMType)
	valueOk := b.arrayToVector(arrValue, vecType)
	b.CreateBr(nextBlock)

	// Merge block: phi between zero-vector (failed assert) and extracted vector (ok).
	b.SetInsertPointAtEnd(nextBlock)
	phi := b.CreatePHI(vecType, "typeassert.spmd.value")
	phi.AddIncoming([]llvm.Value{llvm.ConstNull(vecType), valueOk}, []llvm.BasicBlock{prevBlock, okBlock})

	if expr.CommaOk {
		tuple := b.ctx.ConstStruct([]llvm.Value{llvm.Undef(vecType), llvm.Undef(b.ctx.Int1Type())}, false)
		tuple = b.CreateInsertValue(tuple, phi, 0, "")
		tuple = b.CreateInsertValue(tuple, commaOk, 1, "")
		return tuple
	}
	b.createRuntimeCall("interfaceTypeAssert", []llvm.Value{commaOk}, "")
	return phi
}
