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

// spmdGatherCacheKey identifies a coalesced gather group under a specific mask context.
// Two IndexAddr instructions in the same SPMDGatherGroup but under different SPMD masks
// (e.g., from different varying-switch cases) must emit separate merged swizzles.
type spmdGatherCacheKey struct {
	group *ssa.SPMDGatherGroup
	mask  ssa.Value // nil when the access has no SPMDMask (all-lanes-active context)
}

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

// spmdRegisterBytes returns the SIMD register width in bytes.
// 16 for SSE/WASM SIMD128, 32 for AVX2, 64 for AVX-512.
// Defaults to 16 if not configured.
func (c *compilerContext) spmdRegisterBytes() int {
	if c.SIMDRegisterBytes > 0 {
		return c.SIMDRegisterBytes
	}
	return 16
}

// spmdLaneCount returns the number of SIMD lanes for a given LLVM element type.
// Register width in bytes divided by element size in bytes.
// Returns 1 in scalar fallback mode (-simd=false) so that Varying[T] maps to
// the scalar T LLVM type rather than a vector type.
func (c *compilerContext) spmdLaneCount(elemType llvm.Type) int {
	if !c.simdEnabled {
		return 1
	}
	elemSize := c.targetData.TypeAllocSize(elemType)
	if elemSize == 0 {
		return 1
	}
	return c.spmdRegisterBytes() / int(elemSize)
}

// spmdEffectiveLaneCount returns the lane count for an SPMDType derived from
// the SIMD register width for the element type.
func (c *compilerContext) spmdEffectiveLaneCount(spmdType *types.SPMDType, elemLLVM llvm.Type) int {
	return c.spmdLaneCount(elemLLVM)
}

// spmdMinLaneCountForSig returns the minimum lane count across all varying
// parameters and results in a function signature. On architectures where
// different element sizes produce different lane counts (e.g., AVX2: float32→8,
// int→4), functions with mixed varying types must operate at the minimum width
// so all Varying[T] operands are consistent and no spurious undef lanes arise.
// Returns 0 if the signature has no varying parameters or results.
func (c *compilerContext) spmdMinLaneCountForSig(sig *types.Signature) int {
	if sig == nil {
		return 0
	}
	minLC := 0
	scan := func(t types.Type) {
		spmdType, ok := t.(*types.SPMDType)
		if !ok || !spmdType.IsVarying() {
			return
		}
		elemType := c.getLLVMType(spmdType.Elem())
		lc := c.spmdLaneCount(elemType)
		if lc <= 0 {
			return
		}
		if minLC == 0 || lc < minLC {
			minLC = lc
		}
	}
	params := sig.Params()
	for i := 0; i < params.Len(); i++ {
		scan(params.At(i).Type())
	}
	if sig.Results() != nil {
		results := sig.Results()
		for i := 0; i < results.Len(); i++ {
			scan(results.At(i).Type())
		}
	}
	return minLC
}

// spmdRangeIndexArrayLenCap returns the fixed array length if the rangeindex
// loop iterates over a fixed-size array, or -1 for slices/unknown.
//
// Used to cap lane count for rangeindex loops over fixed-size arrays: the SIMD
// width may exceed the array length (e.g., [4]uint16: registerBits/16=8 lanes but only 4
// elements), which would cause out-of-bounds access at lanes 4-7.
//
// Detection strategy (in order):
//  1. BoundValue has array type [N]T (pre-peel case, before BoundValue updated).
//  2. Body block contains an IndexAddr over *[N]T (non-predicated body).
//  3. Body block contains an SPMDLoad with Contiguous Source of type *[N]T
//     (post-predication: IndexAddr+Load was replaced with SPMDLoad, but Source
//     retains the original IndexAddr.X pointer with type *[N]T).
func spmdRangeIndexArrayLenCap(boundValue ssa.Value, bodyBlock *ssa.BasicBlock, incrBinOp *ssa.BinOp) int64 {
	// Strategy 1: BoundValue has array type — set before predication.
	if arrType, ok := boundValue.Type().Underlying().(*types.Array); ok {
		return arrType.Len()
	}

	if bodyBlock == nil {
		return -1
	}

	for _, instr := range bodyBlock.Instrs {
		switch v := instr.(type) {
		case *ssa.Index:
			// Strategy 2: raw Index (array value access, e.g. arr[i] where arr is [N]T).
			// For array value access (not pointer), X.Type() is directly [N]T.
			if arrType, ok := v.X.Type().Underlying().(*types.Array); ok {
				return arrType.Len()
			}
			// Found Index over string/slice — no cap needed.
			return -1
		case *ssa.IndexAddr:
			// Strategy 2b: raw IndexAddr still present (non-predicated body with pointer access).
			if ptrType, ok := v.X.Type().Underlying().(*types.Pointer); ok {
				if arrType, ok := ptrType.Elem().Underlying().(*types.Array); ok {
					return arrType.Len()
				}
			}
			// Found IndexAddr over slice — no cap needed.
			return -1
		case *ssa.SPMDLoad:
			// Strategy 3: post-predication, Source = IndexAddr.X retains the
			// original array pointer type.
			if !v.Contiguous || v.Source == nil {
				continue
			}
			if ptrType, ok := v.Source.Type().Underlying().(*types.Pointer); ok {
				if arrType, ok := ptrType.Elem().Underlying().(*types.Array); ok {
					return arrType.Len()
				}
			}
			// Found SPMDLoad over slice — no cap needed.
			return -1
		}
	}
	return -1
}

// spmdRangeIndexLaneCount computes the lane count for a range-over-slice SPMD loop
// by examining the slice element type instead of the iterator's int type.
// For []byte slices, this yields registerBits/8 lanes (native v128) instead of registerBits/32.
// Falls back to the iterator type when no slice element type can be determined.
//
// For fixed-length arrays, the lane count is capped at the array length to prevent
// OOB access. For example, [4]uint16 has elem size 2 → registerBits/16 lanes from SIMD width,
// but only 4 elements exist, so laneCount is capped to 4 → <4 x i16> = 64-bit vector.
// The sub-128-bit widening in spmdMaskElemType handles the resulting narrow vector.
//
// Strategy 2 (IndexAddr scan) requires the index to be exactly incrBinOp — it does
// not trace through ChangeType chains. This is sufficient for the common range-over-slice
// pattern where go/ssa uses incrBinOp directly as the IndexAddr index.
func (b *builder) spmdRangeIndexLaneCount(boundValue ssa.Value, bodyBlock *ssa.BasicBlock, incrBinOp *ssa.BinOp) int {
	// Strategy 0: boundValue has array type (pre-peeling, before BoundValue is
	// replaced with a typed int constant). For "go for i, v := range arr" where
	// arr is [N]T, BoundValue.Type() = [N]T. Extract N (array length) and T
	// (element type) directly. This is the most reliable strategy.
	if arrType, ok := boundValue.Type().Underlying().(*types.Array); ok {
		elemLLVM := b.getLLVMType(arrType.Elem())
		laneCount := b.spmdLaneCount(elemLLVM)
		if arrayLen := arrType.Len(); arrayLen > 0 && int64(laneCount) > arrayLen {
			laneCount = int(arrayLen)
		}
		return laneCount
	}

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

	if bodyBlock == nil {
		// Fallback: use the iterator type (int → 4 lanes on wasm32).
		return b.spmdLaneCount(b.getLLVMType(incrBinOp.Type()))
	}

	// Strategy 2: scan the body block for an Index or IndexAddr whose index
	// traces back to incrBinOp, and extract the element type from its base.
	//
	// Two access patterns:
	//   - *ssa.Index: x[i] where x is an array VALUE (e.g. [4]uint16); X.Type() = [N]T.
	//   - *ssa.IndexAddr: &x[i] where x is a slice or *array; X.Type() = []T or *[N]T.
	//
	// In SPMD context, the index may be wrapped in a ChangeType to lanes.Varying[int],
	// so we unwrap ChangeType chains before comparing.
	for _, instr := range bodyBlock.Instrs {
		var xVal ssa.Value
		var idxVal ssa.Value
		switch v := instr.(type) {
		case *ssa.Index:
			xVal, idxVal = v.X, v.Index
		case *ssa.IndexAddr:
			xVal, idxVal = v.X, v.Index
		default:
			continue
		}
		// Unwrap ChangeType chains from the index (SPMD wraps int → Varying[int]).
		idx := idxVal
		for ct, ok := idx.(*ssa.ChangeType); ok; ct, ok = idx.(*ssa.ChangeType) {
			idx = ct.X
		}
		// The unwrapped index must be the increment BinOp (the SPMD iterator).
		if idx != ssa.Value(incrBinOp) {
			continue
		}
		// Determine element type to compute lane count.
		var elemType types.Type
		var arrayLen int64 = -1 // -1 means no cap (slice or unknown)
		xType := xVal.Type().Underlying()
		if arrType, ok := xType.(*types.Array); ok {
			// *ssa.Index with array value: X.Type() = [N]T.
			elemType = arrType.Elem()
			arrayLen = arrType.Len()
		} else if sliceType, ok := xType.(*types.Slice); ok {
			// *ssa.IndexAddr with slice: X.Type() = []T.
			elemType = sliceType.Elem()
		} else if ptrType, ok := xType.(*types.Pointer); ok {
			if arrType, ok := ptrType.Elem().Underlying().(*types.Array); ok {
				// *ssa.IndexAddr with *[N]T.
				elemType = arrType.Elem()
				arrayLen = arrType.Len()
			}
		}
		if elemType != nil {
			elemLLVM := b.getLLVMType(elemType)
			laneCount := b.spmdLaneCount(elemLLVM)
			// Cap lane count at array length for fixed-size arrays.
			// Prevents OOB access when simd_width/elem_size > array_length
			// (e.g., [4]uint16: registerBits/16=8 lanes but only 4 elements exist).
			if arrayLen > 0 && int64(laneCount) > arrayLen {
				laneCount = int(arrayLen)
			}
			return laneCount
		}
	}

	// Strategy 3: post-predication scan for SPMDLoad whose Source retains the
	// original array/slice pointer type. Mirrors spmdRangeIndexArrayLenCap Strategy 3.
	for _, instr := range bodyBlock.Instrs {
		v, ok := instr.(*ssa.SPMDLoad)
		if !ok || !v.Contiguous || v.Source == nil {
			continue
		}
		srcType := v.Source.Type().Underlying()
		if ptrType, ok := srcType.(*types.Pointer); ok {
			if arrType, ok := ptrType.Elem().Underlying().(*types.Array); ok {
				elemLLVM := b.getLLVMType(arrType.Elem())
				laneCount := b.spmdLaneCount(elemLLVM)
				if arrayLen := arrType.Len(); arrayLen > 0 && int64(laneCount) > arrayLen {
					laneCount = int(arrayLen)
				}
				return laneCount
			}
			if sliceType, ok := ptrType.Elem().Underlying().(*types.Slice); ok {
				elemLLVM := b.getLLVMType(sliceType.Elem())
				return b.spmdLaneCount(elemLLVM)
			}
		}
		// SPMDLoad over slice (Source.Type() = []T pointer) — no cap needed.
		return b.spmdLaneCount(b.getLLVMType(incrBinOp.Type()))
	}

	// Fallback: use the iterator type (int → 4 lanes on wasm32).
	return b.spmdLaneCount(b.getLLVMType(incrBinOp.Type()))
}

// spmdConvertScalarToElem converts a scalar integer to match the target element
// type. For bool/mask values (useSExt=true), sign-extension preserves the
// all-ones/all-zeros pattern (i1 true → i32 -1). For data values, zero-extension
// is used. This is needed when splatting a scalar to a vector with a different
// element width (e.g., i1 bool scalar into <4 x i32> mask vector).
func (b *builder) spmdConvertScalarToElem(scalar llvm.Value, elemType llvm.Type, useSExt bool) llvm.Value {
	if scalar.Type() == elemType {
		return scalar
	}
	if scalar.Type().TypeKind() != llvm.IntegerTypeKind || elemType.TypeKind() != llvm.IntegerTypeKind {
		return scalar
	}
	scalarWidth := scalar.Type().IntTypeWidth()
	elemWidth := elemType.IntTypeWidth()
	if scalarWidth < elemWidth {
		if useSExt {
			return b.CreateSExt(scalar, elemType, "")
		}
		return b.CreateZExt(scalar, elemType, "")
	}
	return b.CreateTrunc(scalar, elemType, "")
}

// splatScalar broadcasts a scalar value to fill all lanes of a vector type.
// In scalar fallback mode (laneCount=1), vecType is a plain scalar type rather
// than a vector, so there is nothing to broadcast — return the scalar directly.
func (b *builder) splatScalar(scalar llvm.Value, vecType llvm.Type) llvm.Value {
	if vecType.TypeKind() != llvm.VectorTypeKind {
		// Scalar fallback: vecType is scalar T (laneCount=1). No splat needed.
		return scalar
	}
	undef := llvm.Undef(vecType)
	zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
	ins := b.CreateInsertElement(undef, scalar, zero, "")
	mask := llvm.ConstNull(llvm.VectorType(b.ctx.Int32Type(), vecType.VectorSize()))
	return b.CreateShuffleVector(ins, undef, mask, "splat")
}

// arrayToVector converts an LLVM [N x T] array value to a <M x T> vector
// by extracting each element and inserting it into a vector. If arrayLen < M,
// the remaining lanes are zero-initialized (safe padding for partial arrays).
// In scalar fallback mode (laneCount=1), vecType is a plain scalar type rather
// than a vector; extract element 0 from the array and return it as a scalar.
func (b *builder) arrayToVector(arr llvm.Value, vecType llvm.Type) llvm.Value {
	if vecType.TypeKind() != llvm.VectorTypeKind {
		// Scalar fallback: vecType is scalar T (laneCount=1).
		// Extract the single element from the [1 x T] array.
		if arr.Type().ArrayLength() > 0 {
			return b.CreateExtractValue(arr, 0, "")
		}
		return llvm.ConstNull(vecType)
	}
	n := vecType.VectorSize()
	arrayLen := arr.Type().ArrayLength()
	vec := llvm.ConstNull(vecType) // zero-init instead of Undef for safe padding
	for i := 0; i < arrayLen && i < n; i++ {
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
	// Aggregate varying types (strings, structs) are already [N x T] arrays.
	if vecType.TypeKind() == llvm.ArrayTypeKind {
		return vec
	}
	// Scalar fallback: laneCount=1 produces scalar T, not <1 x T>.
	// Wrap into [1 x T] for interface boxing.
	if vecType.TypeKind() != llvm.VectorTypeKind {
		arrType := llvm.ArrayType(vecType, 1)
		arr := llvm.Undef(arrType)
		return b.CreateInsertValue(arr, vec, 0, "")
	}
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

// spmdBoxedVaryingGoType returns the Go struct type used to box a varying value
// with its mask: struct{ Value [N]T; Mask [N]int32 }.
func (c *compilerContext) spmdBoxedVaryingGoType(spmdType *types.SPMDType, laneCount int) *types.Struct {
	arrayType := types.NewArray(spmdType.Elem(), int64(laneCount))
	// Mask element type must match spmdMaskElemType: SIMD targets use regBits/laneCount
	// bits per lane (2→int64, 4→int32, 8→int16, 16→int8), non-SIMD uses int8 (for i1).
	var maskElemGoType types.Type
	if c.spmdUsesSIMD() {
		switch c.spmdRegisterBytes() * 8 / laneCount {
		case 64:
			maskElemGoType = types.Typ[types.Int64]
		case 32:
			maskElemGoType = types.Typ[types.Int32]
		case 16:
			maskElemGoType = types.Typ[types.Int16]
		default:
			maskElemGoType = types.Typ[types.Int8]
		}
	} else {
		maskElemGoType = types.Typ[types.Int8]
	}
	maskArrayType := types.NewArray(maskElemGoType, int64(laneCount))
	return types.NewStruct([]*types.Var{
		types.NewVar(token.NoPos, nil, "Value", arrayType),
		types.NewVar(token.NoPos, nil, "Mask", maskArrayType),
	}, []string{`spmd:"varying"`, ""})
}

// spmdBroadcastMatch ensures both operands have matching types for SPMD operations.
// If one operand is a vector and the other is a scalar, the scalar is splatted.
// If both are vectors with different lane counts, the wider one is resized to match the narrower.
// If both are vectors with the same lane count but different element widths (e.g., <4 x i32>
// and <4 x i8> from a zext'd byte gather), the narrower elements are extended to match.
// When signExtend is true (for boolean/mask values), sign-extension is used to preserve
// the all-ones/all-zeros pattern; otherwise zero-extension is used for unsigned data.
func (b *builder) spmdBroadcastMatch(x, y llvm.Value, signExtend ...bool) (llvm.Value, llvm.Value) {
	useSExt := len(signExtend) > 0 && signExtend[0]
	xIsVec := x.Type().TypeKind() == llvm.VectorTypeKind
	yIsVec := y.Type().TypeKind() == llvm.VectorTypeKind
	if xIsVec && !yIsVec {
		// Convert scalar element type to match vector element type before splatting.
		// This handles cases like splatting an i1 bool into a <4 x i32> mask vector.
		y = b.spmdConvertScalarToElem(y, x.Type().ElementType(), useSExt)
		y = b.splatScalar(y, x.Type())
	} else if !xIsVec && yIsVec {
		x = b.spmdConvertScalarToElem(x, y.Type().ElementType(), useSExt)
		x = b.splatScalar(x, y.Type())
	} else if xIsVec && yIsVec && x.Type().VectorSize() != y.Type().VectorSize() {
		if useSExt {
			// Mask/boolean operations: expand the narrower to the wider lane count
			// using spmdConvertMaskFormat, which normalizes through i1 and preserves
			// the all-ones/all-zeros pattern. This ensures 16-lane byte-loop masks
			// are not truncated to 4 lanes when combined with a <4 x i32> active mask.
			xSize := x.Type().VectorSize()
			ySize := y.Type().VectorSize()
			if xSize < ySize {
				x = b.spmdConvertMaskFormat(x, y.Type())
			} else {
				y = b.spmdConvertMaskFormat(y, x.Type())
			}
		} else {
			// Data operations: when one operand is a splat constant (all lanes equal),
			// resize it to match the other operand's lane count. This handles the case
			// where a Varying[int] constant is splatted at the register-natural lane count
			// (e.g., <2 x i64> on SSE) but the loop index has the correct effective lane
			// count (e.g., <4 x i32>). Without this, the constant's narrower lane count
			// would be treated as authoritative, truncating the loop index.
			xSize := x.Type().VectorSize()
			ySize := y.Type().VectorSize()
			xSplat := x.IsConstant() && spmdIsConstSplat(x)
			ySplat := y.IsConstant() && spmdIsConstSplat(y)
			if xSplat && !ySplat {
				// x is a splat constant — resize it to match y.
				x = b.spmdResizeVector(x, ySize, y.Type().ElementType())
			} else if ySplat && !xSplat {
				// y is a splat constant — resize it to match x.
				y = b.spmdResizeVector(y, xSize, x.Type().ElementType())
			} else if xSize < ySize {
				// Neither is a splat constant: narrower is authoritative.
				y = b.spmdResizeVector(y, xSize, x.Type().ElementType())
			} else {
				x = b.spmdResizeVector(x, ySize, y.Type().ElementType())
			}
		}
	}
	// After lane-count matching, handle element-width mismatches caused by WASM
	// extension. When spmdVectorIndexArray widens sub-128-bit byte gathers
	// to the mask element type (e.g., <4 x i8> -> <4 x i32>), and the paired
	// operand is still a byte-width constant (e.g., <4 x i8> splat), extend
	// the narrower operand so both sides have the same integer element width.
	// For boolean/mask values (useSExt=true), sign-extension preserves the
	// all-ones/all-zeros pattern (e.g., i8 0xFF → i32 0xFFFFFFFF).
	// For unsigned data values, zero-extension is used.
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
		if useSExt {
			if xWidth > yWidth {
				y = b.CreateSExt(y, xType, "")
			} else {
				x = b.CreateSExt(x, yType, "")
			}
		} else {
			if xWidth > yWidth {
				y = b.CreateZExt(y, xType, "")
			} else {
				x = b.CreateZExt(x, yType, "")
			}
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
	// Handle before the scalar early-return since MaskType.Elem() is not
	// a valid constant type for createConst.
	if spmdtypes.IsMask(spmdType.Elem()) {
		if expr.Value.Kind() != constant.Bool {
			panic(fmt.Sprintf("createSPMDConst: Varying[mask] constant has unexpected kind %v", expr.Value.Kind()))
		}
		if constant.BoolVal(expr.Value) {
			return llvm.ConstAllOnes(vecType)
		}
		return llvm.ConstNull(vecType)
	}

	// Scalar fallback: laneCount=1, vecType is scalar T. Create scalar constant directly.
	if !c.simdEnabled {
		scalarConst := ssa.NewConst(expr.Value, spmdType.Elem())
		return c.createConst(scalarConst, pos)
	}

	// Varying[bool] constants use the WASM mask format: all-ones for true,
	// all-zeros for false — matching comparison results (sext from i1).
	// Using ConstAllOnes ensures correct bitwise-select behavior when the
	// boolean is used as a select operand or mask input.
	if c.spmdUsesSIMD() && expr.Value.Kind() == constant.Bool {
		if basic, ok := spmdType.Elem().Underlying().(*types.Basic); ok && basic.Info()&types.IsBoolean != 0 {
			if constant.BoolVal(expr.Value) {
				return llvm.ConstAllOnes(vecType)
			}
			return llvm.ConstNull(vecType)
		}
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
	laneIndices     llvm.Value // <iter, iter+1, ..., iter+laneCount-1> (nil when isDecomposed)
	tailMask        llvm.Value // per-lane bounds check
	scalarIterVal   llvm.Value // scalar LLVM value (before override to lane indices)
	prologueEmitted bool       // true after emitSPMDBodyPrologue ran for this loop
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

// spmdVecShadowState tracks register-based vector values for allocas initialized
// by spmdPromoteByteArrayCopyToVector. When scalar byte stores at constant indices
// partially modify the alloca, insertelement keeps the shadow in sync. At identity-load
// time the shadow is returned directly, avoiding the alloca round-trip that LLVM
// decomposes into per-byte loads on WASM (v128.load8_splat + 15× v128.load8_lane).
// spmdVecShadowPendingPhi records an LLVM phi that was created at a merge block
// before all SSA predecessors had been processed (DomPreorder may visit merge
// blocks before some of their predecessors). spmdVecShadowFinalize completes
// these phis after the full block-processing loop finishes.
type spmdVecShadowPendingPhi struct {
	phi      llvm.Value
	ssaBlock *ssa.BasicBlock // the merge block that owns the phi
	alloc    *ssa.Alloc      // the promoted alloca this phi tracks
}

type spmdVecShadowState struct {
	current     map[*ssa.Alloc]llvm.Value                     // alloc → current shadow vector in this block
	blockOut    map[llvm.BasicBlock]map[*ssa.Alloc]llvm.Value // LLVM block (exit) → shadow snapshot at block end
	pendingPhis []spmdVecShadowPendingPhi                     // phis awaiting finalization after all blocks compile
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

		// Compute lane count. For rangeindex loops (range-over-slice/array), the
		// iterator type is always int (i32 on WASM32) which gives 4 lanes, but
		// the actual lane count depends on the element type (e.g., 16 for byte).
		//
		// Priority: (1) SSA metadata LaneCount from type checker (accounts for
		// decomposed index optimization), (2) heuristic IndexAddr scan,
		// (3) iterator type fallback.
		var laneCount int
		if ssaLoop.IsRangeIndex {
			laneCount = b.spmdRangeIndexLaneCount(ssaLoop.BoundValue, ssaLoop.MainBodyBlock, mainIncrBinOp)
			// Only use the type checker's LaneCount when the heuristic fell back to
			// the iterator type (indicating it found no reliable element type info).
			// When the heuristic found the correct value (Strategies 0-3), it may
			// already account for array-length capping; overriding it with the type
			// checker's wider value (which ignores array length) would cause OOB access.
			// Compare against the fallback to detect whether the heuristic succeeded.
			iterFallback := b.spmdLaneCount(b.getLLVMType(mainIncrBinOp.Type()))
			if ssaLoop.LaneCount > laneCount && laneCount == iterFallback {
				// Heuristic fell back to iterator type; try the type checker's value.
				// Cap against the array length so we never exceed it for fixed-size arrays
				// (e.g., [4]uint16 → LaneCount=8 but only 4 elements exist).
				arrayLenCap := spmdRangeIndexArrayLenCap(ssaLoop.BoundValue, ssaLoop.MainBodyBlock, mainIncrBinOp)
				if arrayLenCap < 0 || ssaLoop.LaneCount <= int(arrayLenCap) {
					// No cap needed, or type checker is within array bounds: use it.
					laneCount = ssaLoop.LaneCount
				} else if int(arrayLenCap) > laneCount {
					// Type checker exceeds the array length, and iterator fallback is
					// narrower than array length (e.g., [4]uint16 on x86-64: checker=8,
					// array len=4, iterator fallback=2). Use array length as the lane count.
					laneCount = int(arrayLenCap)
				}
				// If ssaLoop.LaneCount > arrayLenCap and heuristic already at or
				// above arrayLenCap, keep the heuristic value (already correct).
			}
		} else {
			elemType := b.getLLVMType(mainIterPhi.Type())
			laneCount = b.spmdLaneCount(elemType)
		}

		// Scalar fallback: override to 1 lane when SIMD is disabled.
		// The type checker sets LaneCount=1 for rangeint (via spmdLaneCount
		// returning 1), but for rangeindex the heuristic may compute laneCount>1
		// — the simdEnabled check ensures we stay scalar.
		if !b.simdEnabled {
			laneCount = 1
		}

		// Rangeindex loops with more than 4 lanes use a decomposed (scalar base +
		// <N x i8> offset) representation on all targets. A naive <laneCount x i32>
		// index vector would be wider than the natural SIMD register (e.g., 16 lanes
		// × 32 bits = 512 bits on SSE), causing index corruption from narrowing. The
		// decomposed path keeps the base as a scalar and tracks only byte offsets in
		// the vector, which fits in any register width.
		isDecomposed := ssaLoop.IsRangeIndex && laneCount > 4

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
		state.loopBlocks[ssaLoop.MainBodyBlock.Index] = loop

		// Register all blocks reachable from the main body within the loop scope.
		// After predication, body may span multiple blocks (body → if.done).
		peeledStopBlocks := map[int]bool{ssaLoop.MainBodyBlock.Index: true}
		if ssaLoop.DoneBlock != nil {
			peeledStopBlocks[ssaLoop.DoneBlock.Index] = true
		}
		if ssaLoop.TailCheckBlock != nil {
			peeledStopBlocks[ssaLoop.TailCheckBlock.Index] = true
		}
		spmdRegisterBodyBlocks(state, ssaLoop.MainBodyBlock, loop, peeledStopBlocks)

		// Map tail body: no iter phi in TailBodyBlock (TailIterPhi is in TailCheckBlock).
		// The prologue is triggered by body block entry code (isPeeled check).
		// Do NOT add TailIterPhi to activeLoops — TailCheckBlock is not a body block
		// and the phi handler must not call emitSPMDBodyPrologue for it.
		if ssaLoop.TailBodyBlock != nil {
			state.loopBlocks[ssaLoop.TailBodyBlock.Index] = loop
			// Register all blocks reachable from tail body too.
			tailStopBlocks := map[int]bool{ssaLoop.TailBodyBlock.Index: true}
			if ssaLoop.DoneBlock != nil {
				tailStopBlocks[ssaLoop.DoneBlock.Index] = true
			}
			if ssaLoop.TrampolineBlock != nil {
				tailStopBlocks[ssaLoop.TrampolineBlock.Index] = true
			}
			// Don't cross into main body.
			tailStopBlocks[ssaLoop.MainBodyBlock.Index] = true
			spmdRegisterBodyBlocks(state, ssaLoop.TailBodyBlock, loop, tailStopBlocks)
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

		// Scalar fallback: with laneCount=1 there is no vectorization. Skip
		// SPMD loop registration so the loop executes as a plain scalar loop.
		if laneCount <= 1 {
			continue
		}

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

		// Find matching x-tools-spmd SPMDLoopInfo for non-peeled rangeindex loops.
		// Do this BEFORE computing lane count so we can use ssaLoop.BoundValue
		// (which retains the original array type [N]T) for the array-length cap.
		// This enables TailMask registration so SSA-level masked loads use the
		// correct tail mask instead of all-ones.
		// Match by BodyBlock pointer first, then fall back to comment matching
		// (blocks may be restructured after predication/optimization).
		var ssaLoop *ssa.SPMDLoopInfo
		for _, sl := range b.fn.SPMDLoops {
			if sl.IsRangeIndex && !sl.IsPeeled {
				if sl.BodyBlock == block || (sl.BodyBlock != nil && sl.BodyBlock.Comment == block.Comment) {
					ssaLoop = sl
					break
				}
			}
		}

		// Compute lane count from the slice element type (if available) for optimal
		// SIMD width. For []byte slices this yields 16 lanes instead of 4.
		laneCount := b.spmdRangeIndexLaneCount(boundValue, block, incrBinOp)
		// Only use the type checker's LaneCount when the heuristic fell back to
		// the iterator type (indicating it found no reliable element type info).
		// When the heuristic found the correct value (Strategies 0-3), it may already
		// account for array-length capping; overriding with the type checker's wider
		// value (which ignores array length) would cause OOB access.
		// Compare against the fallback to detect whether the heuristic succeeded.
		iterFallback := b.spmdLaneCount(b.getLLVMType(incrBinOp.Type()))
		if loopInfo != nil && int(loopInfo.LaneCount) > laneCount && laneCount == iterFallback {
			// Heuristic fell back to iterator type; try type checker's value.
			// Scan the body block directly for array type info. This works even after
			// predication replaces IndexAddr+Load with SPMDLoad (Strategy 3 in
			// spmdRangeIndexArrayLenCap uses SPMDLoad.Source which retains array type).
			// Do NOT use ssaLoop.BoundValue here: comment-based ssaLoop matching may
			// return a wrong loop (same "rangeindex.body" comment for multiple loops).
			arrayLenCap := spmdRangeIndexArrayLenCap(boundValue, block, incrBinOp)
			if arrayLenCap < 0 || int(loopInfo.LaneCount) <= int(arrayLenCap) {
				// No cap needed, or type checker is within array bounds: use it.
				laneCount = int(loopInfo.LaneCount)
			} else if int(arrayLenCap) > laneCount {
				// Type checker exceeds array length, and iterator fallback is narrower
				// than array length (e.g., [4]uint16 on x86-64: checker=8, cap=4,
				// fallback=2). Use array length as the correct lane count.
				laneCount = int(arrayLenCap)
			}
			// If loopInfo.LaneCount > arrayLenCap and heuristic already at or
			// above arrayLenCap, keep the heuristic value (already correct).
		}

		// Scalar fallback: with laneCount=1 there is no vectorization. Skip
		// SPMD loop registration so the loop executes as a plain scalar loop.
		if !b.simdEnabled {
			continue
		}

		// Use decomposed representation (scalar base + <N x i8> offset) on all
		// targets when laneCount > 4. A naive <laneCount x i32> index vector would
		// exceed the natural SIMD register width on targets like SSE (e.g., 16 lanes
		// × 32 bits = 512 bits, which would be narrowed to 128 bits and corrupt the
		// upper lane indices). The decomposed path is width-independent and correct
		// on both WASM SIMD128 and x86-64 SSE/AVX/AVX-512.
		isDecomposed := laneCount > 4

		loop := &spmdActiveLoop{
			info:          loopInfo,
			ssaLoopInfo:   ssaLoop,
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
		state.loopBlocks[loopBlock.Index] = loop

		// Register body block AND all reachable successor blocks within the loop.
		// After varying-if predication, the body may span multiple blocks (e.g.,
		// body → if.done). All such blocks need the loop's tail mask for reduces.
		stopBlocks := map[int]bool{loopBlock.Index: true}
		// Also stop at DoneBlock to prevent post-loop blocks from being
		// registered as body blocks (e.g., when break redirects to DoneBlock).
		if loop.ssaLoopInfo != nil && loop.ssaLoopInfo.DoneBlock != nil {
			stopBlocks[loop.ssaLoopInfo.DoneBlock.Index] = true
		}
		spmdRegisterBodyBlocks(state, block, loop, stopBlocks)
	}

	if len(state.activeLoops) == 0 {
		return nil
	}

	return state
}

// spmdRegisterBodyBlocks BFS-walks from startBlock and registers all reachable
// blocks as body blocks for the given loop, stopping at stopBlocks. This ensures
// that multi-block loop bodies (e.g., after varying-if predication creates
// if.done merge blocks) get the loop's tail mask applied to reduce builtins.
func spmdRegisterBodyBlocks(state *spmdLoopState, startBlock *ssa.BasicBlock, loop *spmdActiveLoop, stopBlocks map[int]bool) {
	state.bodyBlocks[startBlock.Index] = loop
	queue := []*ssa.BasicBlock{startBlock}
	visited := map[int]bool{startBlock.Index: true}
	for len(queue) > 0 {
		b := queue[0]
		queue = queue[1:]
		for _, succ := range b.Succs {
			if visited[succ.Index] || stopBlocks[succ.Index] {
				continue
			}
			visited[succ.Index] = true
			state.bodyBlocks[succ.Index] = loop
			queue = append(queue, succ)
		}
	}
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

// spmdBoundScalar returns the integer scalar bound for a SPMD loop.
// For peeled rangeindex loops, loop.boundValue is the original range expression
// (e.g., the array value [4]uint16) rather than an integer length. In that case
// we extract the array length directly from the type as a constant. For all other
// cases we call getValue and the result is already an integer.
// The returned value has LLVM integer type compatible with intType (may differ in
// width — caller must extend or truncate as needed).
func (b *builder) spmdBoundScalar(loop *spmdActiveLoop, intType llvm.Type) llvm.Value {
	if arrType, ok := loop.boundValue.Type().Underlying().(*types.Array); ok {
		// Peeled rangeindex over [N]T: bound is the array length as a constant.
		return llvm.ConstInt(intType, uint64(arrType.Len()), false)
	}
	return b.getValue(loop.boundValue, token.NoPos)
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

		if loop.isDecomposed {
			// Decomposed path: represent the index as scalar base + <N x i8> offset.
			// The base is always stored as i32 regardless of target pointer width.
			// On x86-64 the loop iterator phi is i64, but array/slice lengths always
			// fit in 32 bits, and all decomposed arithmetic uses i32 constants. Storing
			// a narrowed i32 base avoids per-site truncations throughout the pipeline.
			i8Type := b.ctx.Int8Type()
			i32Type := b.ctx.Int32Type()
			laneCount := loop.laneCount

			varyingOffset := b.spmdLaneOffsetConst(laneCount, i8Type)

			// Normalize base to i32 (safe: indices into Go slices/arrays fit in 32 bits).
			decompBase := scalarPhi
			if decompBase.Type() != i32Type {
				decompBase = b.CreateTrunc(decompBase, i32Type, "spmd.base.narrow")
			}

			if b.spmdDecomposed != nil {
				// Use the appropriate iter value as the body iterator for decomposition.
				var bodyIter ssa.Value
				if isMainBody {
					bodyIter = ssaLoop.MainIterPhi
				} else {
					bodyIter = ssaLoop.TailIterPhi
				}
				b.spmdDecomposed[bodyIter] = &spmdDecomposedIndex{
					scalarBase:    decompBase,
					varyingOffset: varyingOffset,
					laneCount:     laneCount,
					loop:          loop,
					fromBodyIter:  true,
				}
			}

			if isMainBody {
				maskType := llvm.VectorType(b.spmdMaskElemType(laneCount), laneCount)
				loop.tailMask = llvm.ConstAllOnes(maskType)
			} else {
				// Tail mask via <N x i8> comparison: offset < clamp(bound - base).
				// All arithmetic is done in i32 regardless of target pointer width.
				// On x86-64 the iterator phi is i64, so truncate both operands to i32
				// before the subtraction. The result is always in [0, laneCount] which
				// fits safely in i8 after clamping.
				i32Type := b.ctx.Int32Type()
				boundScalar := b.spmdBoundScalar(loop, i32Type)
				// Ensure boundScalar is i32 for the sub/clamp arithmetic.
				if boundScalar.Type() != i32Type {
					boundScalar = b.CreateTrunc(boundScalar, i32Type, "spmd.bound.narrow")
				}
				scalarPhiI32 := scalarPhi
				if scalarPhiI32.Type() != i32Type {
					scalarPhiI32 = b.CreateTrunc(scalarPhiI32, i32Type, "spmd.base.narrow")
				}
				diff := b.CreateSub(boundScalar, scalarPhiI32, "spmd.diff")
				zero32 := llvm.ConstInt(i32Type, 0, false)
				lcConst := llvm.ConstInt(i32Type, uint64(laneCount), false)
				diffClamped := b.CreateSelect(
					b.CreateICmp(llvm.IntSGT, diff, lcConst, ""),
					lcConst, diff, "spmd.diff.clamped")
				diffClamped = b.CreateSelect(
					b.CreateICmp(llvm.IntSLT, diffClamped, zero32, ""),
					zero32, diffClamped, "spmd.diff.nonneg")
				diffI8 := b.CreateTrunc(diffClamped, i8Type, "spmd.diff.i8")
				diffVec := b.splatScalar(diffI8, llvm.VectorType(i8Type, laneCount))
				tailMaskI1 := b.CreateICmp(llvm.IntULT, varyingOffset, diffVec, "spmd.tail.mask")
				loop.tailMask = b.spmdWrapMask(tailMaskI1, laneCount)
				if ssaLoop.TailMask != nil {
					b.locals[ssaLoop.TailMask] = loop.tailMask
				}
			}
			return
		}

		// Non-decomposed peeled path: full <laneCount x elemType> lane indices.
		elemType := scalarPhi.Type()
		// Narrow the index element type when it would produce a vector wider than
		// the SIMD register width. For example, range over [4]uint16 with int index
		// on x86-64: laneCount=4, elemType=i64 → <4 x i64>=256 bits, too wide.
		// Truncate to i32 → <4 x i32>=128 bits (for 128-bit registers).
		// Use separate narrowedElemType/narrowedPhi for vector construction so that
		// loop.scalarIterVal (used by contiguous detection and extendInteger for GEP
		// indexing) always keeps the original width.
		narrowedElemType := elemType
		narrowedPhi := scalarPhi
		elemBits := uint64(b.targetData.TypeAllocSize(elemType)) * 8
		regBits := uint64(b.spmdRegisterBytes()) * 8
		if uint64(loop.laneCount)*elemBits > regBits {
			narrowBits := regBits / uint64(loop.laneCount)
			narrowedElemType = b.ctx.IntType(int(narrowBits))
			narrowedPhi = b.CreateTrunc(scalarPhi, narrowedElemType, "spmd.iter.narrow")
			// Do NOT update loop.scalarIterVal — it must stay at the original width
			// so spmdAnalyzeContiguousIndex and extendInteger see the correct type.
		}
		vecType := llvm.VectorType(narrowedElemType, loop.laneCount)
		iterVec := b.splatScalar(narrowedPhi, vecType)
		offsetVec := b.spmdLaneOffsetConst(loop.laneCount, narrowedElemType)
		laneIndices := b.CreateAdd(iterVec, offsetVec, "spmd.lane.idx")
		loop.laneIndices = laneIndices

		if isMainBody {
			// Main body: all-ones mask (all lanes active).
			maskType := llvm.VectorType(b.spmdMaskElemType(loop.laneCount), loop.laneCount)
			loop.tailMask = llvm.ConstAllOnes(maskType)

			// Peeled rangeindex: the incrBinOp (mainIterPhi + laneCount) is compiled
			// inside this body block where spmdValueOverride makes it produce a vector.
			// But it is also the phi back-edge and must stay scalar for LLVM phi validity.
			// Pre-compute the scalar next-iteration value and register it in
			// spmdPeeledScalarIncr so the phi resolution loop can use it instead.
			// Use the ORIGINAL elemType and scalarPhi so the phi back-edge type matches.
			if loop.incrBinOp != nil {
				laneCountScalar := llvm.ConstInt(elemType, uint64(loop.laneCount), false)
				scalarIncr := b.CreateAdd(scalarPhi, laneCountScalar, "spmd.iter.incr")
				if b.spmdPeeledScalarIncr == nil {
					b.spmdPeeledScalarIncr = make(map[ssa.Value]llvm.Value)
				}
				b.spmdPeeledScalarIncr[loop.incrBinOp] = scalarIncr
			}
		} else {
			// Tail body: compute tail mask (laneIndices < bound).
			boundScalar := b.spmdBoundScalar(loop, narrowedElemType)
			// Narrow or extend bound to match the (possibly narrowed) narrowedElemType.
			if boundScalar.Type() != narrowedElemType {
				if b.targetData.TypeAllocSize(boundScalar.Type()) > b.targetData.TypeAllocSize(narrowedElemType) {
					boundScalar = b.CreateTrunc(boundScalar, narrowedElemType, "spmd.bound.narrow")
				} else {
					boundScalar = b.CreateZExt(boundScalar, narrowedElemType, "spmd.bound.zext")
				}
			}
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
		i32Type := b.ctx.Int32Type()
		laneCount := loop.laneCount

		// Create constant byte offset <0, 1, 2, ..., laneCount-1> as <N x i8>.
		varyingOffset := b.spmdLaneOffsetConst(laneCount, i8Type)

		// Normalize base to i32 (safe: indices into Go slices/arrays fit in 32 bits).
		// On x86-64 the loop iterator is i64; storing i32 avoids per-site truncations.
		decompBase := scalarPhi
		if decompBase.Type() != i32Type {
			decompBase = b.CreateTrunc(decompBase, i32Type, "spmd.base.narrow")
		}

		// Register the decomposition so BinOp/IndexAddr handlers can use it.
		if b.spmdDecomposed != nil {
			b.spmdDecomposed[loop.bodyIterValue] = &spmdDecomposedIndex{
				scalarBase:    decompBase,
				varyingOffset: varyingOffset,
				laneCount:     laneCount,
				loop:          loop,
				fromBodyIter:  true, // raw iterator: base is always a multiple of laneCount
			}
		}

		// Compute tail mask using <N x i8> comparison to stay within the natural
		// SIMD register. diff = bound - base (scalar i32); clamp to [0, laneCount];
		// truncate to i8. Then compare: offset < clamp(diff) using unsigned <N x i8>.
		// All arithmetic is done in i32 regardless of target pointer width. On
		// x86-64 the iterator is i64, so both operands are truncated to i32 first.
		// The result is always in [0, laneCount] which fits in i8 after clamping.
		boundScalar := b.spmdBoundScalar(loop, i32Type)
		// Ensure boundScalar is i32 for the sub/clamp arithmetic.
		if boundScalar.Type() != i32Type {
			boundScalar = b.CreateTrunc(boundScalar, i32Type, "spmd.bound.narrow")
		}
		scalarPhiI32 := scalarPhi
		if scalarPhiI32.Type() != i32Type {
			scalarPhiI32 = b.CreateTrunc(scalarPhiI32, i32Type, "spmd.base.narrow")
		}
		diff := b.CreateSub(boundScalar, scalarPhiI32, "spmd.diff")

		// Clamp diff to [0, laneCount]: max(0, min(laneCount, diff)).
		zero32 := llvm.ConstInt(i32Type, 0, false)
		lcConst := llvm.ConstInt(i32Type, uint64(laneCount), false)
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

	// Narrow the index element type when it would produce a vector wider than
	// the SIMD register width. For example, range over [4]uint16 with int index
	// on x86-64: laneCount=4, elemType=i64 → <4 x i64>=256 bits, too wide.
	// Truncate to i32 → <4 x i32>=128 bits (for 128-bit registers).
	// Use separate narrowedElemType/narrowedPhi for vector construction so that
	// loop.scalarIterVal (already set above) keeps the original width for use by
	// spmdAnalyzeContiguousIndex and extendInteger in GEP indexing.
	narrowedElemType := elemType
	narrowedPhi := scalarPhi
	elemBits := uint64(b.targetData.TypeAllocSize(elemType)) * 8
	regBits := uint64(b.spmdRegisterBytes()) * 8
	if uint64(loop.laneCount)*elemBits > regBits {
		narrowBits := regBits / uint64(loop.laneCount)
		narrowedElemType = b.ctx.IntType(int(narrowBits))
		narrowedPhi = b.CreateTrunc(scalarPhi, narrowedElemType, "spmd.iter.narrow")
		// Do NOT update loop.scalarIterVal — it must stay at the original width
		// so spmdAnalyzeContiguousIndex and extendInteger see the correct type.
	}

	// Create vector type for the lane count.
	vecType := llvm.VectorType(narrowedElemType, loop.laneCount)

	// Splat the scalar iterator across all lanes.
	iterVec := b.splatScalar(narrowedPhi, vecType)

	// Create the offset constant <0, 1, 2, ..., laneCount-1>.
	offsetVec := b.spmdLaneOffsetConst(loop.laneCount, narrowedElemType)

	// Compute lane indices: <iter, iter+1, iter+2, ..., iter+laneCount-1>.
	laneIndices := b.CreateAdd(iterVec, offsetVec, "spmd.lane.idx")

	// Get the bound value and splat it.
	boundScalar := b.spmdBoundScalar(loop, narrowedElemType)
	// Narrow or extend bound to match the (possibly narrowed) narrowedElemType.
	if boundScalar.Type() != narrowedElemType {
		if b.targetData.TypeAllocSize(boundScalar.Type()) > b.targetData.TypeAllocSize(narrowedElemType) {
			boundScalar = b.CreateTrunc(boundScalar, narrowedElemType, "spmd.bound.narrow")
		} else {
			boundScalar = b.CreateZExt(boundScalar, narrowedElemType, "spmd.bound.zext")
		}
	}
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
	// decomp.scalarBase is always i32 (normalized at creation in emitSPMDBodyPrologue).
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
	// Scalar fallback: mask is a scalar integer (i1 or i32). Non-zero = any true.
	if mask.Type().TypeKind() != llvm.VectorTypeKind {
		if mask.Type() == b.ctx.Int1Type() {
			return mask
		}
		zero := llvm.ConstNull(mask.Type())
		return b.CreateICmp(llvm.IntNE, mask, zero, "")
	}
	if b.spmdUsesSIMD() {
		// Normalize <N x i1> to mask format before calling the intrinsic.
		// The llvm.wasm.anytrue intrinsic is registered by vector type; calling
		// llvm.wasm.anytrue.v4i1 with a <4 x i1> argument creates a custom
		// intrinsic call on a sub-128-bit type. The WASM backend's legalization
		// does not know how to promote <4 x i1> arguments to custom intrinsics
		// and crashes with "Do not know how to promote this operator's operand".
		// Convert to mask format (<N x iW>) first so we call
		// llvm.wasm.anytrue.v4i32 (or similar), which WASM can lower natively.
		// On x86, spmdAnyTrue bitcasts to <16 x i8> internally via pmovmskb.
		laneCount := mask.Type().VectorSize()
		if mask.Type().ElementType() == b.ctx.Int1Type() {
			mask = b.spmdWrapMask(mask, laneCount)
		}
		i32Result := b.spmdAnyTrue(mask)
		zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
		return b.CreateICmp(llvm.IntNE, i32Result, zero, "")
	}
	// Non-SIMD: <N x i1> → iN bitcast, compare != 0.
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
	// Scalar fallback: mask is a scalar integer (i1 or i32). All-true = non-zero.
	if mask.Type().TypeKind() != llvm.VectorTypeKind {
		if mask.Type() == b.ctx.Int1Type() {
			return mask
		}
		zero := llvm.ConstNull(mask.Type())
		return b.CreateICmp(llvm.IntNE, mask, zero, "")
	}
	if b.spmdUsesSIMD() {
		// Normalize <N x i1> to mask format before calling the intrinsic.
		// See spmdVectorAnyTrue for the rationale.
		// On x86, spmdAllTrue bitcasts to <16 x i8> internally via pmovmskb.
		laneCount := mask.Type().VectorSize()
		if mask.Type().ElementType() == b.ctx.Int1Type() {
			mask = b.spmdWrapMask(mask, laneCount)
		}
		// Use native all_true (WASM) or pmovmskb+icmp (x86).
		i32Result := b.spmdAllTrue(mask)
		zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
		return b.CreateICmp(llvm.IntNE, i32Result, zero, "")
	}
	// Non-SIMD: <N x i1> → iN bitcast, compare == all-ones.
	vecSize := mask.Type().VectorSize()
	intType := b.ctx.IntType(vecSize)
	intVal := b.CreateBitCast(mask, intType, "")
	allOnes := llvm.ConstAllOnes(intType)
	return b.CreateICmp(llvm.IntEQ, intVal, allOnes, "")
}

// spmdVectorReduceUmax reduces a vector of unsigned integers to the scalar
// maximum across all lanes using the @llvm.vector.reduce.umax intrinsic.
// This is used to collapse a vector of per-lane indices into a single scalar
// for a unified bounds check: if max(indices) < length, all lanes are in bounds.
func (b *builder) spmdVectorReduceUmax(vec llvm.Value) llvm.Value {
	vecType := vec.Type()
	elemType := vecType.ElementType()
	laneCount := vecType.VectorSize()
	elemBits := elemType.IntTypeWidth()
	intrinsicName := fmt.Sprintf("llvm.vector.reduce.umax.v%di%d", laneCount, elemBits)
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fnType := llvm.FunctionType(elemType, []llvm.Type{vecType}, false)
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.CreateCall(fn.GlobalValueType(), fn, []llvm.Value{vec}, "spmd.reduce.umax")
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

// spmdWasmBitmask calls @llvm.wasm.bitmask on a vector.
// Returns i32 with the high bit of each lane packed into a bitmask.
// The input must be in WASM mask format (<N x iW>, not <N x i1>).
// Only valid for WASM targets.
func (b *builder) spmdWasmBitmask(vec llvm.Value) llvm.Value {
	vecType := vec.Type()
	suffix := spmdVectorTypeSuffix(vecType)
	intrinsicName := "llvm.wasm.bitmask." + suffix
	i32Type := b.ctx.Int32Type()
	fnType := llvm.FunctionType(i32Type, []llvm.Type{vecType}, false)
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{vec}, "")
}

// spmdIsWASM returns true when the compiler target is a WebAssembly target.
// On WASM, SIMD comparisons natively produce <N x i32> (all-ones/all-zeros),
// so we use i32 as the internal mask element type to avoid the redundant
// shl/shr_s sign-extension that LLVM inserts when converting <N x i1> to i32
// for v128.bitselect.
func (c *compilerContext) spmdIsWASM() bool {
	return strings.HasPrefix(c.Triple, "wasm")
}

// spmdIsX86 returns true when the compiler target is an x86 or x86-64 target.
func (c *compilerContext) spmdIsX86() bool {
	return strings.HasPrefix(c.Triple, "x86_64") || strings.HasPrefix(c.Triple, "i386")
}

// spmdHasSSSE3 returns true when the target is x86 with SSSE3 enabled.
// SSSE3 provides pshufb (i8x16.swizzle equivalent) used for fast byte permutation.
func (c *compilerContext) spmdHasSSSE3() bool {
	return c.spmdIsX86() && strings.Contains(c.Features, "+ssse3")
}

// spmdUsesSIMD returns true when SIMD vector instructions should be emitted.
// Returns false in scalar fallback mode (-simd=false).
// WASM requires the -simd flag to be explicitly enabled; x86-64 has SSE2 as baseline.
func (c *compilerContext) spmdUsesSIMD() bool {
	if c.spmdIsWASM() {
		return c.simdEnabled
	}
	if c.spmdIsX86() {
		return c.simdEnabled // SSE2 is baseline for x86-64
	}
	return false
}

// spmdHasRelaxedSIMD returns true when the target has the WebAssembly
// relaxed-simd feature enabled. Relaxed SIMD unlocks instructions like
// i8x16.relaxed_swizzle (undefined out-of-range behaviour instead of zero)
// and i32x4.relaxed_dot_i8x16_i7x16_add_s.
func (c *compilerContext) spmdHasRelaxedSIMD() bool {
	return c.spmdIsWASM() && strings.Contains(c.Features, "+relaxed-simd")
}

// spmdRelaxedDotI8x16Add emits an i32x4.relaxed_dot_i8x16_i7x16_add_s
// intrinsic call: result[i] = sum(a[4i+j]*b[4i+j] for j=0..3) + acc[i].
// The second operand (bVec) must hold signed 7-bit values [-64, 63].
func (b *builder) spmdRelaxedDotI8x16Add(a, bVec, acc llvm.Value) llvm.Value {
	i8Type := b.ctx.Int8Type()
	i32Type := b.ctx.Int32Type()
	v16i8 := llvm.VectorType(i8Type, 16)
	v4i32 := llvm.VectorType(i32Type, 4)

	const intrinsicName = "llvm.wasm.relaxed.dot.i8x16.i7x16.add.signed"
	fnType := llvm.FunctionType(v4i32, []llvm.Type{v16i8, v16i8, v4i32}, false)
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{a, bVec, acc}, "spmd.relaxed.dot")
}

// spmdSwizzle emits a byte-permute (i8x16.swizzle equivalent) for the current target.
// On WASM: llvm.wasm.swizzle or llvm.wasm.relaxed.swizzle (always 128-bit / <16 x i8>).
// On x86 with SSSE3: spmdX86Pshufb, which handles both <16 x i8> (pshufb) and
// <32 x i8> (vpshufb ymm for AVX2) based on the table vector width.
// Fallback: per-lane extractelement/insertelement loop (16 lanes only).
// Table and indices must be the same vector type. Returns the same vector type.
func (b *builder) spmdSwizzle(table, indices llvm.Value) llvm.Value {
	if b.spmdIsWASM() {
		// WASM always operates on 128-bit (16 lanes).
		v16i8 := llvm.VectorType(b.ctx.Int8Type(), 16)
		intrinsicName := "llvm.wasm.swizzle"
		if b.spmdHasRelaxedSIMD() {
			intrinsicName = "llvm.wasm.relaxed.swizzle"
		}
		fnType := llvm.FunctionType(v16i8, []llvm.Type{v16i8, v16i8}, false)
		fn := b.mod.NamedFunction(intrinsicName)
		if fn.IsNil() {
			fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
		}
		return b.createCall(fnType, fn, []llvm.Value{table, indices}, "spmd.swizzle")
	}
	if b.spmdHasSSSE3() {
		// spmdX86Pshufb dispatches to pshufb (128-bit) or vpshufb (256-bit) based on
		// the vector width — no hardcoded <16 x i8> needed here.
		return b.spmdX86Pshufb(table, indices)
	}
	return b.spmdSwizzleScalarFallback(table, indices)
}

// spmdSwizzleScalarFallback emits a per-lane byte-permute via extractelement/insertelement.
// Used when neither WASM swizzle nor x86 pshufb is available.
// Indices with value >= 16 or bit 7 set produce 0 (matching i8x16.swizzle semantics).
func (b *builder) spmdSwizzleScalarFallback(table, indices llvm.Value) llvm.Value {
	i8Type := b.ctx.Int8Type()
	i32Type := b.ctx.Int32Type()
	v16i8 := llvm.VectorType(i8Type, 16)
	result := llvm.ConstNull(v16i8)
	for i := 0; i < 16; i++ {
		laneConst := llvm.ConstInt(i32Type, uint64(i), false)
		idx := b.CreateExtractElement(indices, laneConst, "")
		// Indices >= 16 or with bit 7 set produce 0 per swizzle semantics.
		oob := b.CreateICmp(llvm.IntUGE, idx, llvm.ConstInt(i8Type, 16, false), "")
		elem := b.CreateExtractElement(table, idx, "")
		elem = b.CreateSelect(oob, llvm.ConstInt(i8Type, 0, false), elem, "")
		result = b.CreateInsertElement(result, elem, laneConst, "")
	}
	return result
}

// spmdX86BitmaskI32 extracts one bit per logical lane from vec into a scalar i32.
// vec must be a mask vector in the all-ones/all-zeros lane format (<N x iW>).
// For <N x i8> (byte elements, 1 byte per lane), pmovmskb maps directly: 1 bit
// per byte = 1 bit per lane. For wider elements (<N x i16>, <N x i32>), pmovmskb
// on the raw bytes gives multiple bits per lane (garbage). Instead we reduce to
// <N x i1> (icmp ne 0) then bitcast to iN: this always yields exactly 1 bit per
// logical lane regardless of element width.
func (b *builder) spmdX86BitmaskI32(vec llvm.Value) llvm.Value {
	laneCount := vec.Type().VectorSize()
	elemType := vec.Type().ElementType()
	var i32bitmask llvm.Value
	if elemType == b.ctx.Int8Type() {
		// Fast path: <N x i8> — pmovmskb is a direct 1-bit-per-lane extraction.
		i32bitmask = b.spmdX86Pmovmskb(vec)
	} else {
		// General path: collapse to <N x i1> (one true/false per lane), then
		// bitcast to iN to get a compact bitmask. LLVM typically lowers this
		// to a comparison + movmskb or similar without extra memory traffic.
		zero := llvm.ConstNull(vec.Type())
		i1Vec := b.CreateICmp(llvm.IntNE, vec, zero, "bitmask.i1")
		intType := b.ctx.IntType(laneCount)
		iN := b.CreateBitCast(i1Vec, intType, "bitmask.iN")
		i32bitmask = b.CreateZExt(iN, b.ctx.Int32Type(), "bitmask.i32")
	}
	return i32bitmask
}

// spmdBitmask extracts one bit per logical lane into a scalar i32 bitmask.
// On WASM: llvm.wasm.bitmask on the vector in WASM mask format.
// On x86: for <N x i8> uses pmovmskb (1 bit per byte = 1 bit per lane).
// For wider elements (<N x i16>, <N x i32>) uses icmp+bitcast to avoid
// pmovmskb producing multiple bits per lane.
func (b *builder) spmdBitmask(vec llvm.Value) llvm.Value {
	if b.spmdIsWASM() {
		return b.spmdWasmBitmask(vec)
	}
	return b.spmdX86BitmaskI32(vec)
}

// spmdAnyTrue tests if any lane in the vector is nonzero.
// On WASM: llvm.wasm.anytrue returning i32 (0 or 1).
// On x86: bitmask + icmp ne 0, returning i32.
// Supports any element width: <N x i8>, <N x i16>, <N x i32>.
func (b *builder) spmdAnyTrue(vec llvm.Value) llvm.Value {
	if b.spmdIsWASM() {
		return b.spmdWasmAnyTrue(vec)
	}
	mask := b.spmdX86BitmaskI32(vec)
	zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
	ne := b.CreateICmp(llvm.IntNE, mask, zero, "anytrue")
	return b.CreateZExt(ne, b.ctx.Int32Type(), "anytrue.i32")
}

// spmdAllTrue tests if all lanes in the vector are nonzero.
// On WASM: llvm.wasm.alltrue returning i32 (0 or 1).
// On x86: bitmask + icmp eq allOnesMask, returning i32.
// allOnesMask is (1<<laneCount)-1: one bit per logical lane.
// Supports any element width: <N x i8>, <N x i16>, <N x i32>.
func (b *builder) spmdAllTrue(vec llvm.Value) llvm.Value {
	if b.spmdIsWASM() {
		return b.spmdWasmAllTrue(vec)
	}
	laneCount := vec.Type().VectorSize()
	mask := b.spmdX86BitmaskI32(vec)
	// One bit per logical lane; all-ones means laneCount bits are set.
	allOnesMask := uint64((1 << laneCount) - 1)
	allOnes := llvm.ConstInt(b.ctx.Int32Type(), allOnesMask, false)
	eq := b.CreateICmp(llvm.IntEQ, mask, allOnes, "alltrue")
	return b.CreateZExt(eq, b.ctx.Int32Type(), "alltrue.i32")
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
// On SIMD targets, the mask element type is sized to fill the register:
// regBits/laneCount bits per lane (e.g., 256-bit AVX2 with 8 lanes → i32,
// 128-bit SIMD128 with 4 lanes → i32, 8 lanes → i16, 16 lanes → i8).
// On other targets this is always i1 (native LLVM boolean vector element).
func (c *compilerContext) spmdMaskElemType(laneCount int) llvm.Type {
	if c.spmdUsesSIMD() {
		regBits := c.spmdRegisterBytes() * 8
		return c.ctx.IntType(regBits / laneCount)
	}
	return c.ctx.Int1Type()
}

// spmdMaskType returns the LLVM mask type for an SPMD function's implicit first parameter.
// Returns zero-value llvm.Type{} if the function has no varying parameters.
func (c *compilerContext) spmdMaskType(fn *ssa.Function) llvm.Type {
	return c.spmdMaskTypeFromSig(fn.Signature)
}

// spmdMaskTypeFromSig returns the LLVM mask type for an SPMD signature's implicit mask parameter.
// On SIMD targets the mask is <N x iW> (e.g., <4 x i32>); on non-SIMD targets it is <N x i1>.
// N is the minimum lane count across all varying parameters and results. Using the minimum
// ensures consistency on architectures where different element sizes yield different lane counts
// (e.g., AVX2: float32→8 lanes, int→4 lanes). A function with both Varying[float32] and
// Varying[int] must use 4 lanes throughout.
// Returns zero-value llvm.Type{} if the signature has no varying parameters.
func (c *compilerContext) spmdMaskTypeFromSig(sig *types.Signature) llvm.Type {
	if sig == nil {
		return llvm.Type{}
	}
	laneCount := c.spmdMinLaneCountForSig(sig)
	if laneCount == 0 {
		return llvm.Type{} // No varying parameters
	}
	// Scalar fallback: laneCount=1 means no SIMD. Return scalar mask (i32 on
	// WASM, i1 elsewhere) instead of <1 x maskElem> which LLVM rejects as a
	// branch condition.
	if laneCount <= 1 {
		return c.getLLVMType(spmdtypes.NewVaryingMask())
	}
	return llvm.VectorType(c.spmdMaskElemType(laneCount), laneCount)
}

// spmdWrapMask sign-extends an <N x i1> comparison result to the SIMD mask type
// (e.g., <4 x i32> for 4 lanes, <8 x i16> for 8 lanes, <16 x i8> for 16 lanes).
// On non-SIMD targets this is a no-op. LLVM folds sext(cmp) into the comparison
// instruction on both WASM and x86, so there is no runtime cost.
func (b *builder) spmdWrapMask(cmp llvm.Value, laneCount int) llvm.Value {
	if !b.spmdUsesSIMD() {
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
	if !b.spmdUsesSIMD() {
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
	if !b.spmdUsesSIMD() {
		return b.CreateSelect(mask, trueVal, falseVal, "")
	}

	valType := trueVal.Type()
	maskType := mask.Type() // e.g., <4 x i32>, <8 x i16>, or <16 x i8>

	// Reconcile lane count mismatch: if mask has more lanes than the data, narrow it.
	// This happens when getLLVMType(Varying[mask]) produces a wider vector than the
	// data type (e.g., <32 x i1> mask for <16 x i8> data on AVX2 with i1 size=1).
	if maskType.TypeKind() == llvm.VectorTypeKind && valType.TypeKind() == llvm.VectorTypeKind {
		maskLanes := maskType.VectorSize()
		dataLanes := valType.VectorSize()
		if maskLanes != dataLanes {
			maskElem := b.spmdMaskElemType(dataLanes)
			targetMaskType := llvm.VectorType(maskElem, dataLanes)
			mask = b.spmdConvertMaskFormat(mask, targetMaskType)
			maskType = mask.Type()
		}
	}

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
	if !b.spmdUsesSIMD() {
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
	if !b.spmdUsesSIMD() {
		return cond // non-WASM-SIMD: masks are always <N x i1>, no conversion needed
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

// spmdFuncBodyLaneCount returns the lane count for an SPMD function body
// (spmdFuncIsBody=true), derived from the entry mask vector type. Returns 0
// if the entry mask is not a vector (scalar fallback or not an SPMD function).
// The entry mask is set from the first varying parameter's element type at
// function entry, so its VectorSize() gives the correct hardware lane count.
func (b *builder) spmdFuncBodyLaneCount() int {
	if b.spmdEntryMask.IsNil() {
		return 0
	}
	if b.spmdEntryMask.Type().TypeKind() != llvm.VectorTypeKind {
		return 0
	}
	return b.spmdEntryMask.Type().VectorSize()
}

// spmdWidenVector extends a vector from srcLanes to dstLanes by appending
// undef lanes at the end. Used when passing a narrow-lane vector to an SPMD
// function declared with more lanes (e.g., calling an 8-lane float32 function
// from a 4-lane int loop on AVX2 x86-64). The callee's entry mask constrains
// which lanes are used, so undef lanes in the extra positions are safe.
func (b *builder) spmdWidenVector(vec llvm.Value, dstLanes int) llvm.Value {
	srcType := vec.Type()
	if srcType.TypeKind() != llvm.VectorTypeKind {
		return vec
	}
	srcLanes := srcType.VectorSize()
	if srcLanes >= dstLanes {
		return vec
	}
	// Build a shufflevector mask: first srcLanes from vec, rest as undef (0xFFFFFFFF).
	shuffleElems := make([]llvm.Value, dstLanes)
	i32Type := b.ctx.Int32Type()
	for i := 0; i < srcLanes; i++ {
		shuffleElems[i] = llvm.ConstInt(i32Type, uint64(i), false)
	}
	undefIdx := llvm.ConstInt(i32Type, 0xFFFFFFFF, false) // undef lane index
	for i := srcLanes; i < dstLanes; i++ {
		shuffleElems[i] = undefIdx
	}
	shuffleMask := llvm.ConstVector(shuffleElems, false)
	undef := llvm.Undef(srcType)
	return b.CreateShuffleVector(vec, undef, shuffleMask, "spmd.widen")
}

// spmdNarrowVector extracts the first dstLanes from a wider vector via
// shufflevector. Used when the call site produces narrower vectors than the
// callee expects and must widen, but also for the reverse (narrowing a result).
func (b *builder) spmdNarrowVector(vec llvm.Value, dstLanes int) llvm.Value {
	srcType := vec.Type()
	if srcType.TypeKind() != llvm.VectorTypeKind {
		return vec
	}
	srcLanes := srcType.VectorSize()
	if srcLanes <= dstLanes {
		return vec
	}
	i32Type := b.ctx.Int32Type()
	shuffleElems := make([]llvm.Value, dstLanes)
	for i := 0; i < dstLanes; i++ {
		shuffleElems[i] = llvm.ConstInt(i32Type, uint64(i), false)
	}
	shuffleMask := llvm.ConstVector(shuffleElems, false)
	undef := llvm.Undef(srcType)
	return b.CreateShuffleVector(vec, undef, shuffleMask, "spmd.narrow")
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

// spmdDeferMask returns the execution mask to pack into a defer struct for an
// SPMD function call. Uses the SSA-level mask from CallCommon.SPMDMask if set
// by the predication pass, otherwise falls back to spmdCallMask.
func (b *builder) spmdDeferMask(instr *ssa.Defer, fn *ssa.Function) llvm.Value {
	if instr.Call.SPMDMask != nil {
		return b.getValue(instr.Call.SPMDMask, getPos(instr))
	}
	mask := b.spmdCallMask(fn)
	if mask.IsNil() {
		return llvm.ConstAllOnes(b.spmdMaskType(fn))
	}
	return mask
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
	case llvm.PointerTypeKind:
		suffix = "p0"
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
	// Scalar fallback: with laneCount=1, lanes builtins degenerate to scalar ops.
	if !b.simdEnabled {
		switch {
		case name == "lanes.Index":
			elemType := b.getLLVMType(instr.Signature().Results().At(0).Type().(*types.SPMDType).Elem())
			return llvm.ConstInt(elemType, 0, false), nil // single lane = index 0
		case strings.HasPrefix(name, "lanes.Count["):
			return llvm.ConstInt(b.intType, 1, false), nil // 1 lane
		case strings.HasPrefix(name, "lanes.From["):
			// lanes.From[T](data []T) in scalar mode: load element 0 from the slice.
			// The result type is Varying[T] = T in scalar mode.
			slice := b.getValue(instr.Args[0], getPos(instr))
			resultSSAType := instr.Signature().Results().At(0).Type().(*types.SPMDType)
			elemType := b.getLLVMType(resultSSAType.Elem())
			ptr := b.CreateExtractValue(slice, 0, "lanes.from.ptr")
			gep := b.CreateInBoundsGEP(elemType, ptr, []llvm.Value{
				llvm.ConstInt(b.ctx.Int32Type(), 0, false),
			}, "lanes.from.gep")
			return b.CreateLoad(elemType, gep, "lanes.from.val"), nil
		case strings.HasPrefix(name, "lanes.Broadcast["),
			strings.HasPrefix(name, "lanes.Rotate["),
			strings.HasPrefix(name, "lanes.Swizzle["),
			strings.HasPrefix(name, "lanes.ShiftLeft["),
			strings.HasPrefix(name, "lanes.ShiftRight["),
			strings.HasPrefix(name, "lanes.RotateWithin["),
			strings.HasPrefix(name, "lanes.ShiftLeftWithin["),
			strings.HasPrefix(name, "lanes.ShiftRightWithin["),
			strings.HasPrefix(name, "lanes.SwizzleWithin["):
			// Single lane: all cross-lane ops are identity.
			return b.getValue(instr.Args[0], getPos(instr)), nil
		case name == "lanes.DotProductI8x16Add":
			// Scalar fallback for DotProductI8x16Add(a, b [16]byte, acc [4]int) [4]int.
			// acc[i] += sum(int8(a[i*4+j]) * int8(b[i*4+j]) for j in 0..3)
			pos := getPos(instr)
			aVal := b.getValue(instr.Args[0], pos) // [16 x i8]
			bVal := b.getValue(instr.Args[1], pos) // [16 x i8]
			acc := b.getValue(instr.Args[2], pos)  // [4 x i32]
			i32Type := b.ctx.Int32Type()
			for i := 0; i < 4; i++ {
				sum := llvm.ConstInt(i32Type, 0, false)
				for j := 0; j < 4; j++ {
					idx := i*4 + j
					aElem := b.CreateExtractValue(aVal, idx, "")
					bElem := b.CreateExtractValue(bVal, idx, "")
					// Sign-extend i8 to i32 for signed multiply.
					aExt := b.CreateSExt(aElem, i32Type, "")
					bExt := b.CreateSExt(bElem, i32Type, "")
					prod := b.CreateMul(aExt, bExt, "")
					sum = b.CreateAdd(sum, prod, "")
				}
				accElem := b.CreateExtractValue(acc, i, "")
				accElem = b.CreateAdd(accElem, sum, "")
				acc = b.CreateInsertValue(acc, accElem, i, "")
			}
			return acc, nil
		}
	}

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
		if vec.Type().TypeKind() == llvm.ArrayTypeKind {
			// Aggregate types use [N x T] arrays, not LLVM vectors.
			// Extract via alloca + variable-index GEP, then fill all slots.
			laneCount := vec.Type().ArrayLength()
			elemType := vec.Type().ElementType()
			alloca := b.CreateAlloca(vec.Type(), "broadcast.arr")
			b.CreateStore(vec, alloca)
			idx := lane
			if idx.Type() != b.ctx.Int32Type() {
				idx = b.CreateTrunc(idx, b.ctx.Int32Type(), "")
			}
			gep := b.CreateInBoundsGEP(vec.Type(), alloca, []llvm.Value{
				llvm.ConstInt(b.ctx.Int32Type(), 0, false),
				idx,
			}, "broadcast.gep")
			elem := b.CreateLoad(elemType, gep, "broadcast.elem")
			result := llvm.Undef(vec.Type())
			for i := 0; i < laneCount; i++ {
				result = b.CreateInsertValue(result, elem, i, "")
			}
			return result, nil
		}
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
		// lanes.From[T](data []T) — load N contiguous elements from slice as vector.
		// Extract pointer from the slice value (element 0 is the data pointer).
		sliceVal := b.getValue(instr.Args[0], getPos(instr))
		ptr := b.CreateExtractValue(sliceVal, 0, "slice.ptr")
		// Determine vector type from result SPMDType.
		resultType := instr.Signature().Results().At(0).Type()
		vecType := b.getLLVMType(resultType)
		// Cap lane count to the SPMD context's effective lane count.
		// In an SPMD function body, spmdFuncBodyLaneCount gives the minimum across
		// all varying parameter/return types. In a go-for loop, the active loop's
		// laneCount is authoritative. This prevents mixed-width mismatches where
		// e.g. lanes.From[float32] produces 8 lanes but Varying[int] uses 4 on AVX2.
		if spmdType, ok := resultType.(*types.SPMDType); ok && spmdType.IsVarying() {
			effectiveLC := 0
			if b.spmdFuncIsBody {
				effectiveLC = b.spmdFuncBodyLaneCount()
			}
			if effectiveLC == 0 {
				if activeLoop := b.spmdFindActiveLoopForBlock(b.currentBlock); activeLoop != nil {
					effectiveLC = activeLoop.laneCount
				}
			}
			// Also cap to slice length if known at compile time (e.g., lanes.From(literal_slice)).
			if effectiveLC == 0 {
				sliceLen := b.CreateExtractValue(sliceVal, 1, "slice.len")
				if sliceLen.IsConstant() {
					constLen := int(sliceLen.ZExtValue())
					if constLen > 0 && constLen < vecType.VectorSize() {
						effectiveLC = constLen
					}
				}
			}
			if effectiveLC > 0 && vecType.VectorSize() != effectiveLC {
				elemType := b.getLLVMType(spmdType.Elem())
				vecType = llvm.VectorType(elemType, effectiveLC)
			}
		}
		// Load as vector.
		return b.CreateLoad(vecType, ptr, "lanes.from"), nil

	case strings.HasPrefix(name, "lanes.Rotate["):
		// lanes.Rotate[T](value Varying[T], offset int) Varying[T]
		// Full-width rotation: equivalent to RotateWithin with groupSize = laneCount.
		// offset must be a compile-time constant.
		return b.createRotate(instr)

	case strings.HasPrefix(name, "lanes.Swizzle["):
		// lanes.Swizzle[T](value Varying[T], indices Varying[int]) Varying[T]
		// Full-width swizzle with runtime varying indices.
		// Uses per-lane extractelement/insertelement since indices are not
		// compile-time constants and cannot use shufflevector.
		return b.createSwizzle(instr)

	case strings.HasPrefix(name, "lanes.RotateWithin["):
		return b.createRotateWithin(instr, name)

	case strings.HasPrefix(name, "lanes.ShiftLeftWithin["):
		return b.createShiftLeftWithin(instr, name)

	case strings.HasPrefix(name, "lanes.ShiftRightWithin["):
		return b.createShiftRightWithin(instr, name)

	case strings.HasPrefix(name, "lanes.SwizzleWithin["):
		return b.createSwizzleWithin(instr, name)

	case name == "lanes.DotProductI8x16Add":
		// lanes.DotProductI8x16Add(a, b [16]byte, acc [4]int) [4]int
		// Maps to i32x4.relaxed_dot_i8x16_i7x16_add_s on WASM Relaxed SIMD,
		// or pmaddubsw + pmaddwd on x86 with SSSE3.
		pos := getPos(instr)
		aVal := b.getValue(instr.Args[0], pos)
		bVal := b.getValue(instr.Args[1], pos)
		accVal := b.getValue(instr.Args[2], pos)

		v16i8 := llvm.VectorType(b.ctx.Int8Type(), 16)
		v4i32 := llvm.VectorType(b.ctx.Int32Type(), 4)

		// Convert a and b aggregates to <16 x i8> vectors.
		// Direct bitcast between aggregate and vector types is illegal in LLVM IR.
		if aVal.Type().TypeKind() == llvm.ArrayTypeKind {
			aVal = b.spmdAggregateToVector(aVal, v16i8, 16, "dot.a")
		}
		if bVal.Type().TypeKind() == llvm.ArrayTypeKind {
			bVal = b.spmdAggregateToVector(bVal, v16i8, 16, "dot.b")
		}

		// Determine the native result type from the function's return type.
		// On WASM32 int=i32; on x86-64 int=i64. The computation is always i32,
		// so we must adapt: truncate acc elements to i32, compute, then sign-extend.
		resultGoType := instr.Signature().Results().At(0).Type()
		resultLLVMType := b.getLLVMType(resultGoType) // [4 x i32] on WASM32, [4 x i64] on x86-64
		accElemType := b.intType                      // i32 on WASM32, i64 on x86-64

		// Convert acc to <4 x i32> for the computation.
		var accI32 llvm.Value
		if accVal.Type().TypeKind() == llvm.ArrayTypeKind {
			if accElemType == b.ctx.Int32Type() {
				// WASM32: [4 x i32] aggregate → <4 x i32> vector.
				accI32 = b.spmdAggregateToVector(accVal, v4i32, 4, "dot.acc")
			} else {
				// x86-64: [4 x i64] aggregate; extract and truncate each element to i32.
				accI32 = llvm.ConstNull(v4i32)
				for i := 0; i < 4; i++ {
					elem := b.CreateExtractValue(accVal, i, "dot.acc.elem")
					elem32 := b.CreateTrunc(elem, b.ctx.Int32Type(), "dot.acc.trunc")
					accI32 = b.CreateInsertElement(accI32, elem32,
						llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false), "dot.acc.i32")
				}
			}
		} else if accVal.Type().TypeKind() == llvm.VectorTypeKind && accVal.Type().ElementType() != b.ctx.Int32Type() {
			// Varying acc (already a vector) with non-i32 elements: trunc to i32.
			accI32 = b.CreateTrunc(accVal, v4i32, "dot.acc.trunc")
		} else {
			accI32 = accVal
		}

		var result llvm.Value
		if b.spmdIsWASM() && b.spmdHasRelaxedSIMD() {
			result = b.spmdRelaxedDotI8x16Add(aVal, bVal, accI32)
		} else if b.spmdHasSSSE3() {
			// x86: pmaddubsw(a_u8, b_i8) → <8 x i16>, then pmaddwd(result, ones) → <4 x i32>.
			// Weight 100 (as used in the IPv4 parser) fits in u8 without decomposition.
			halfResult := b.spmdX86Pmaddubsw(aVal, bVal) // <8 x i16>
			i16Type := b.ctx.Int16Type()
			ones := llvm.ConstVector([]llvm.Value{
				llvm.ConstInt(i16Type, 1, false), llvm.ConstInt(i16Type, 1, false),
				llvm.ConstInt(i16Type, 1, false), llvm.ConstInt(i16Type, 1, false),
				llvm.ConstInt(i16Type, 1, false), llvm.ConstInt(i16Type, 1, false),
				llvm.ConstInt(i16Type, 1, false), llvm.ConstInt(i16Type, 1, false),
			}, false)
			result = b.spmdX86Pmaddwd(halfResult, ones) // <4 x i32>
			result = b.CreateAdd(result, accI32, "dot.add.acc")
		} else {
			return llvm.Value{}, b.makeError(pos, "lanes.DotProductI8x16Add requires WASM +relaxed-simd or x86 +ssse3")
		}

		// On x86-64 (int=i64), sign-extend the <4 x i32> result back to [4 x i64].
		if accElemType != b.ctx.Int32Type() {
			// Convert <4 x i32> → [4 x i64]: extract each lane, sext, insert into aggregate.
			finalResult := llvm.Undef(resultLLVMType)
			for i := 0; i < 4; i++ {
				elem := b.CreateExtractElement(result,
					llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false), "dot.res.elem")
				elem64 := b.CreateSExt(elem, accElemType, "dot.res.sext")
				finalResult = b.CreateInsertValue(finalResult, elem64, i, "dot.res")
			}
			return finalResult, nil
		}
		return result, nil

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

// spmdExtractConstVectorInt64s extracts compile-time constant int64 values from an LLVM vector.
// Returns nil if the value is not a constant vector or has unexpected length.
// Uses signed extraction (SExtValue); indices are expected to be non-negative.
func (b *builder) spmdExtractConstVectorInt64s(v llvm.Value, expectedLen int) []int64 {
	if !v.IsConstant() {
		return nil
	}
	if v.Type().TypeKind() != llvm.VectorTypeKind {
		return nil
	}
	n := v.Type().VectorSize()
	if n != expectedLen {
		return nil
	}
	result := make([]int64, n)
	i32Type := b.ctx.Int32Type()
	for i := 0; i < n; i++ {
		idx := llvm.ConstInt(i32Type, uint64(i), false)
		elem := llvm.ConstExtractElement(v, idx)
		if elem.IsUndef() || !elem.IsConstant() {
			return nil
		}
		result[i] = int64(elem.SExtValue())
	}
	return result
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

// spmdSwizzleWithinMask computes the shufflevector index mask for SwizzleWithin.
// Each group of groupSize elements is permuted according to indices array.
func spmdSwizzleWithinMask(totalLanes, groupSize int, indices []int64) []uint64 {
	mask := make([]uint64, totalLanes)
	for i := 0; i < totalLanes; i++ {
		group := i / groupSize
		laneInGroup := i % groupSize
		srcLane := group*groupSize + int(indices[laneInGroup])
		mask[i] = uint64(srcLane)
	}
	return mask
}

// createRotate rotates all lanes of a vector by a compile-time constant offset.
//
// Positive offset rotates left (each lane i gets the value from lane (i+offset) % N).
// Negative offset rotates right.
//
// Example: Rotate(<0,1,2,3>, offset=1) => <1,2,3,0>
// Example: Rotate(<0,1,2,3>, offset=-1) => <3,0,1,2>
func (b *builder) createRotate(instr *ssa.CallCommon) (llvm.Value, error) {
	pos := getPos(instr)
	value := b.getValue(instr.Args[0], pos)

	offset, ok := spmdExtractIntConst(instr.Args[1])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.Rotate: offset must be a compile-time constant")
	}

	vecType := value.Type()
	totalLanes := vecType.VectorSize()
	if totalLanes == 0 {
		// Array aggregate type: fall back to extractvalue/insertvalue
		totalLanes = vecType.ArrayLength()
		off := int(offset)
		result := llvm.Undef(vecType)
		for i := 0; i < totalLanes; i++ {
			srcIdx := ((i+off)%totalLanes + totalLanes) % totalLanes
			elem := b.CreateExtractValue(value, srcIdx, "rotate.elem")
			result = b.CreateInsertValue(result, elem, i, "rotate.res")
		}
		return result, nil
	}

	// Vector type: use shufflevector with a constant mask.
	mask := spmdRotateWithinMask(totalLanes, totalLanes, int(offset))
	shuffleMask := b.spmdShuffleConst(mask)
	return b.CreateShuffleVector(value, llvm.Undef(vecType), shuffleMask, "spmd.rotate"), nil
}

// createSwizzle reorders lanes of a vector according to a runtime varying index vector.
//
// For lane i, the output is value[indices[i] % laneCount].
// Since indices are runtime values, this cannot use shufflevector; it uses
// per-lane extractelement/insertelement instead.
//
// Example: Swizzle(<10,20,30,40>, <3,0,2,1>) => <40,10,30,20>
func (b *builder) createSwizzle(instr *ssa.CallCommon) (llvm.Value, error) {
	pos := getPos(instr)
	value := b.getValue(instr.Args[0], pos)
	indices := b.getValue(instr.Args[1], pos)

	i32Type := b.ctx.Int32Type()

	if value.Type().TypeKind() == llvm.VectorTypeKind {
		laneCount := value.Type().VectorSize()
		laneCountVal := llvm.ConstInt(i32Type, uint64(laneCount), false)
		result := llvm.Undef(value.Type())
		for i := 0; i < laneCount; i++ {
			laneIdx := llvm.ConstInt(i32Type, uint64(i), false)
			// Extract the source index for this output lane.
			srcIdx := b.CreateExtractElement(indices, laneIdx, "swizzle.idx")
			// Ensure srcIdx is i32 (indices vector may be i64 on 64-bit targets).
			if srcIdx.Type().IntTypeWidth() != 32 {
				srcIdx = b.CreateTrunc(srcIdx, i32Type, "swizzle.idx.trunc")
			}
			// Wrap to valid range [0, laneCount).
			srcIdx = b.CreateURem(srcIdx, laneCountVal, "swizzle.idx.mod")
			elem := b.CreateExtractElement(value, srcIdx, "swizzle.elem")
			result = b.CreateInsertElement(result, elem, laneIdx, "swizzle.res")
		}
		return result, nil
	}

	if value.Type().TypeKind() == llvm.ArrayTypeKind {
		// Aggregate [N x T]: store to alloca so we can index with variable indices.
		laneCount := value.Type().ArrayLength()
		laneCountVal := llvm.ConstInt(i32Type, uint64(laneCount), false)
		elemType := value.Type().ElementType()
		alloca := b.CreateAlloca(value.Type(), "swizzle.arr")
		b.CreateStore(value, alloca)
		result := llvm.Undef(value.Type())
		for i := 0; i < laneCount; i++ {
			laneIdx := llvm.ConstInt(i32Type, uint64(i), false)
			srcIdx := b.CreateExtractElement(indices, laneIdx, "swizzle.idx")
			if srcIdx.Type().IntTypeWidth() != 32 {
				srcIdx = b.CreateTrunc(srcIdx, i32Type, "swizzle.idx.trunc")
			}
			srcIdx = b.CreateURem(srcIdx, laneCountVal, "swizzle.idx.mod")
			zero := llvm.ConstInt(i32Type, 0, false)
			gep := b.CreateInBoundsGEP(value.Type(), alloca, []llvm.Value{zero, srcIdx}, "swizzle.gep")
			elem := b.CreateLoad(elemType, gep, "swizzle.elem")
			result = b.CreateInsertValue(result, elem, i, "swizzle.res")
		}
		return result, nil
	}

	return llvm.Value{}, b.makeError(pos, "lanes.Swizzle: unsupported value type")
}

// createSwizzleWithin permutes values within independent groups using compile-time constant indices.
//
// For a vector of totalLanes elements divided into groups of groupSize, each group
// is permuted according to the indices array. indices[i] specifies the source lane
// within the group for output lane i.
//
// Example: SwizzleWithin(<0,1,2,3,4,5,6,7>, indices=<1,0,2,3>, groupSize=4)
// => <1,0,2,3,5,4,6,7>  (each group of 4 permuted by indices)
func (b *builder) createSwizzleWithin(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	pos := getPos(instr)
	value := b.getValue(instr.Args[0], pos)
	indices := b.getValue(instr.Args[1], pos)

	groupSize, ok := spmdExtractIntConst(instr.Args[2])
	if !ok {
		return llvm.Value{}, b.makeError(pos, "lanes.SwizzleWithin: groupSize must be a compile-time constant")
	}

	vecType := value.Type()
	totalLanes := vecType.VectorSize()
	gs := int(groupSize)

	if gs <= 0 || totalLanes%gs != 0 {
		return llvm.Value{}, b.makeError(pos, "lanes.SwizzleWithin: groupSize must evenly divide lane count")
	}

	// indices is Varying[int], always totalLanes-wide. Extract constant values
	// and verify the permutation pattern repeats across all groups.
	fullIndices := b.spmdExtractConstVectorInt64s(indices, totalLanes)
	if fullIndices == nil {
		return llvm.Value{}, b.makeError(pos, "lanes.SwizzleWithin: indices must be a compile-time constant")
	}
	// Validate that every group carries the same permutation.
	constIndices := fullIndices[:gs]
	for g := 1; g < totalLanes/gs; g++ {
		for k := 0; k < gs; k++ {
			if fullIndices[g*gs+k] != constIndices[k] {
				return llvm.Value{}, b.makeError(pos, fmt.Sprintf(
					"lanes.SwizzleWithin: indices must repeat every %d lanes (group 0 lane %d = %d, group %d lane %d = %d)",
					gs, k, constIndices[k], g, k, fullIndices[g*gs+k]))
			}
		}
	}

	// Verify all indices are in valid range.
	for i, idx := range constIndices {
		if idx < 0 || idx >= int64(gs) {
			return llvm.Value{}, b.makeError(pos, fmt.Sprintf("lanes.SwizzleWithin: indices[%d] out of range [0, %d]", i, gs-1))
		}
	}

	mask := spmdSwizzleWithinMask(totalLanes, gs, constIndices)
	shuffleMask := b.spmdShuffleConst(mask)
	return b.CreateShuffleVector(value, llvm.Undef(vecType), shuffleMask, "swizzlewithin"), nil
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

// spmdGetReduceMask returns the LLVM execution mask for a reduce builtin call.
// The predication pass sets CallCommon.SPMDMask when the call site is inside a
// varying control flow region.
//
// For go-for loops, the SSA predication assigns a constant all-ones mask for the
// main body and a tail mask SSA value for peeled tail bodies. For non-peeled loops
// (rangeindex/range-over-slice), the SSA mask is always all-ones even though only
// some lanes are active — the actual tail mask is computed at runtime by TinyGo's
// loop body prologue. We detect this case and return the loop's runtime tailMask
// so that reduce operations only see values from active lanes.
//
// Returns a zero llvm.Value when no masking is needed (all lanes guaranteed active).
func (b *builder) spmdGetReduceMask(instr *ssa.CallCommon) llvm.Value {
	// First: get the SSA-level mask (may be nil or a constant all-ones).
	var ssaMask llvm.Value
	if instr.SPMDMask != nil {
		ssaMask = b.getValue(instr.SPMDMask, getPos(instr))
	}

	// Second: check whether we are inside a loop body whose tailMask is a
	// runtime value. Use direct bodyBlocks lookup only (no dominance fallback)
	// to avoid matching done blocks after the loop — tailMask doesn't dominate
	// those. Predication sub-blocks (varying if/else within the loop body)
	// get their mask via SSA-level SPMDMask from predication.
	var loopMask llvm.Value
	if b.spmdLoopState != nil && b.currentBlock != nil {
		if loop, ok := b.spmdLoopState.bodyBlocks[b.currentBlock.Index]; ok {
			lm := loop.tailMask
			if !lm.IsNil() {
				if loop.isPeeled {
					// For peeled loops, only use tailMask in the tail body block.
					// Main body has all lanes active — no masking needed.
					if loop.ssaLoopInfo != nil &&
						loop.ssaLoopInfo.TailBodyBlock != nil &&
						b.currentBlock.Index == loop.ssaLoopInfo.TailBodyBlock.Index {
						if !lm.IsConstant() || ssaMask.IsNil() {
							loopMask = lm
						}
					}
				} else {
					// Non-peeled: tailMask is in body prologue, dominates body.
					if !lm.IsConstant() || ssaMask.IsNil() {
						loopMask = lm
					}
				}
			}
		}
	}

	// Combine: if both masks are present, AND them so both varying-if and
	// loop-tail constraints are applied. If only one is present, use it.
	switch {
	case ssaMask.IsNil() && loopMask.IsNil():
		return llvm.Value{} // no masking needed
	case ssaMask.IsNil():
		return loopMask
	case loopMask.IsNil():
		// SSA mask may be a constant all-ones (predication assigned it for the
		// main body); return zero so callers skip masking for the trivial case.
		if ssaMask.IsConstant() && !ssaMask.IsNull() {
			return llvm.Value{}
		}
		return ssaMask
	default:
		// Both non-nil: AND them. But first check if either is a trivial
		// all-ones constant — if so, the other mask alone is sufficient.
		// This avoids lane-count mismatches when the SSA mask (e.g., <4 x i32>
		// from Varying[mask]) has fewer lanes than the loop mask (e.g., <16 x i8>
		// for a uint8 rangeindex loop).
		if ssaMask.IsConstant() && !ssaMask.IsNull() {
			return loopMask
		}
		if loopMask.IsConstant() && !loopMask.IsNull() {
			return ssaMask
		}
		// Both are non-trivial runtime masks: normalize to the larger lane count
		// to avoid truncating tail mask information.
		ssaLanes := ssaMask.Type().VectorSize()
		loopLanes := loopMask.Type().VectorSize()
		laneCount := ssaLanes
		if loopLanes > laneCount {
			laneCount = loopLanes
		}
		i1SSA := b.spmdUnwrapMaskForIntrinsic(ssaMask, laneCount)
		i1Loop := b.spmdUnwrapMaskForIntrinsic(loopMask, laneCount)
		combined := b.CreateAnd(i1SSA, i1Loop, "spmd.reduce.combined.mask")
		return combined
	}
}

// spmdMaskBoolVecForAny applies the execution mask to a boolean vector so that
// inactive lanes appear false. Used by reduce.Any, reduce.Mask, reduce.FindFirstSet,
// and reduce.Count: ANDs the bool vector with the mask so inactive lanes become 0.
// Both inputs are normalized to <N x i1> before the AND; the result is <N x i1>.
// When mask is a zero llvm.Value (no masking needed) vec is returned unchanged.
func (b *builder) spmdMaskBoolVecForAny(vec, mask llvm.Value) llvm.Value {
	if mask.IsNil() {
		return vec
	}
	// Normalize both to <N x i1> so CreateAnd is valid.
	i1Vec := b.spmdNormalizeBoolVecToI1(vec)
	laneCount := i1Vec.Type().VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	return b.CreateAnd(i1Vec, i1Mask, "spmd.reduce.mask.and")
}

// spmdMaskBoolVecForAll applies the execution mask to a boolean vector so that
// inactive lanes appear true. Used by reduce.All: ORs the bool vector with NOT mask
// so inactive lanes become 1 and do not prevent "all true".
// Both inputs are normalized to <N x i1> before the OR; the result is <N x i1>.
// When mask is a zero llvm.Value (no masking needed) vec is returned unchanged.
func (b *builder) spmdMaskBoolVecForAll(vec, mask llvm.Value) llvm.Value {
	if mask.IsNil() {
		return vec
	}
	// Normalize both to <N x i1>.
	i1Vec := b.spmdNormalizeBoolVecToI1(vec)
	laneCount := i1Vec.Type().VectorSize()
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	notMask := b.CreateNot(i1Mask, "spmd.reduce.notmask")
	return b.CreateOr(i1Vec, notMask, "spmd.reduce.mask.or")
}

// spmdMaskArithVec replaces inactive-lane values with an identity element so that
// they do not affect the reduction result. Used by arithmetic reduce builtins.
// identity must be a scalar constant of the vector element type; inactive lanes
// are set to identity via a per-lane select driven by the execution mask.
// When mask is a zero llvm.Value (no masking needed) vec is returned unchanged.
func (b *builder) spmdMaskArithVec(vec, mask llvm.Value, identity llvm.Value) llvm.Value {
	if mask.IsNil() {
		return vec
	}
	// Build a constant splat of the identity value matching vec's vector type.
	vecType := vec.Type()
	laneCount := vecType.VectorSize()
	elems := make([]llvm.Value, laneCount)
	for i := range elems {
		elems[i] = identity
	}
	identityVec := llvm.ConstVector(elems, false)
	// Use spmdMaskSelect: active lanes keep vec, inactive lanes get identity.
	// spmdMaskSelect accepts either WASM mask format or <N x i1>.
	normalizedMask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	return b.CreateSelect(normalizedMask, vec, identityVec, "spmd.reduce.masked")
}

// createReduceBuiltin handles interception of reduce.* function calls.
// Returns the LLVM value result and nil error on success.
func (b *builder) createReduceBuiltin(instr *ssa.CallCommon, name string) (llvm.Value, error) {
	// Scalar fallback: with laneCount=1, the input is a scalar (not a vector).
	// Reduction of a single element is identity — return the value directly.
	// reduce.From wraps the scalar in a 1-element stack slice.
	if !b.simdEnabled {
		val := b.getValue(instr.Args[0], getPos(instr))
		if strings.HasPrefix(name, "reduce.From[") {
			// Build a 1-element []T slice from the scalar value.
			elemType := val.Type()
			arrType := llvm.ArrayType(elemType, 1)
			alloca := b.CreateAlloca(arrType, "reduce.from.scalar")
			gep := b.CreateInBoundsGEP(arrType, alloca, []llvm.Value{
				llvm.ConstInt(b.ctx.Int32Type(), 0, false),
				llvm.ConstInt(b.ctx.Int32Type(), 0, false),
			}, "")
			b.CreateStore(val, gep)
			ptr := b.CreateBitCast(alloca, b.dataPtrType, "")
			lenVal := llvm.ConstInt(b.uintptrType, 1, false)
			sliceType := b.getLLVMType(instr.Signature().Results().At(0).Type())
			slice := llvm.Undef(sliceType)
			slice = b.CreateInsertValue(slice, ptr, 0, "")
			slice = b.CreateInsertValue(slice, lenVal, 1, "")
			slice = b.CreateInsertValue(slice, lenVal, 2, "") // cap = len
			return slice, nil
		}
		// Boolean reductions (Any, All) return the bool directly.
		// Numeric reductions (Add, Mul, Min, Max) return the scalar.
		// FindFirstSet returns 0 (only lane 0 exists, always index 0).
		// Count returns 1 if true, 0 if false.
		if name == "reduce.FindFirstSet" {
			return llvm.ConstInt(b.intType, 0, false), nil
		}
		if name == "reduce.Count" {
			return b.CreateZExt(val, b.intType, ""), nil
		}
		// reduce.Mask returns int bitmask — single lane = bit 0.
		if name == "reduce.Mask" {
			return b.CreateZExt(val, b.intType, ""), nil
		}
		return val, nil
	}

	switch {
	case strings.HasPrefix(name, "reduce.Add["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		if spmdType, ok := argType.(*types.SPMDType); ok && spmdIsFloat(spmdType.Elem()) {
			// Float: ordered fadd reduction with start = 0.0. Identity for add is 0.0.
			elemType := vec.Type().ElementType()
			identity := llvm.ConstFloat(elemType, 0.0)
			vec = b.spmdMaskArithVec(vec, b.spmdGetReduceMask(instr), identity)
			startVal := llvm.ConstFloat(elemType, 0.0)
			return b.spmdCallVectorReduceFloat("fadd", startVal, vec), nil
		}
		// Integer add: identity is 0.
		elemType := vec.Type().ElementType()
		identity := llvm.ConstInt(elemType, 0, false)
		vec = b.spmdMaskArithVec(vec, b.spmdGetReduceMask(instr), identity)
		return b.spmdCallVectorReduce("add", vec), nil

	case strings.HasPrefix(name, "reduce.Mul["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		if spmdType, ok := argType.(*types.SPMDType); ok && spmdIsFloat(spmdType.Elem()) {
			// Float: ordered fmul reduction with start = 1.0. Identity for mul is 1.0.
			elemType := vec.Type().ElementType()
			identity := llvm.ConstFloat(elemType, 1.0)
			vec = b.spmdMaskArithVec(vec, b.spmdGetReduceMask(instr), identity)
			startVal := llvm.ConstFloat(elemType, 1.0)
			return b.spmdCallVectorReduceFloat("fmul", startVal, vec), nil
		}
		// Integer mul: identity is 1.
		elemType := vec.Type().ElementType()
		identity := llvm.ConstInt(elemType, 1, false)
		vec = b.spmdMaskArithVec(vec, b.spmdGetReduceMask(instr), identity)
		return b.spmdCallVectorReduce("mul", vec), nil

	case name == "reduce.All":
		// reduce.All(v Varying[bool]) bool — true if all active lanes are true.
		// Inactive lanes must appear true so they don't prevent "all true".
		vec := b.getValue(instr.Args[0], getPos(instr))
		masked := b.spmdMaskBoolVecForAll(vec, b.spmdGetReduceMask(instr))
		// Normalize to <N x i1> then bitcast to iN and compare == all-ones.
		i1Vec := b.spmdNormalizeBoolVecToI1(masked)
		vecSize := i1Vec.Type().VectorSize()
		intType := b.ctx.IntType(vecSize)
		intVal := b.CreateBitCast(i1Vec, intType, "")
		allOnes := llvm.ConstAllOnes(intType)
		return b.CreateICmp(llvm.IntEQ, intVal, allOnes, ""), nil

	case name == "reduce.Any":
		// reduce.Any(v Varying[bool]) bool — true if any active lane is true.
		// Inactive lanes must appear false so they don't falsely trigger "any".
		vec := b.getValue(instr.Args[0], getPos(instr))
		reduceMask := b.spmdGetReduceMask(instr)
		masked := b.spmdMaskBoolVecForAny(vec, reduceMask)
		// spmdVectorAnyTrue handles both <N x i1> and <N x i32> (WASM) formats.
		return b.spmdVectorAnyTrue(masked), nil

	case strings.HasPrefix(name, "reduce.Max["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		execMask := b.spmdGetReduceMask(instr)
		if spmdType, ok := argType.(*types.SPMDType); ok {
			if spmdIsFloat(spmdType.Elem()) {
				// Identity for fmax is -Inf: inactive lanes contribute -Inf and never win.
				elemType := vec.Type().ElementType()
				identity := llvm.ConstFloatFromString(elemType, "-Inf")
				vec = b.spmdMaskArithVec(vec, execMask, identity)
				return b.spmdCallVectorReduce("fmax", vec), nil
			}
			if spmdIsSignedInt(spmdType.Elem()) {
				// Identity for smax is INT_MIN: inactive lanes contribute the minimum
				// signed value and never beat any active lane's value.
				elemType := vec.Type().ElementType()
				bits := elemType.IntTypeWidth()
				identity := llvm.ConstInt(elemType, uint64(1)<<(uint(bits)-1), false)
				vec = b.spmdMaskArithVec(vec, execMask, identity)
				return b.spmdCallVectorReduce("smax", vec), nil
			}
		}
		// Unsigned max: identity is 0 (inactive lanes contribute 0 and never win).
		elemType := vec.Type().ElementType()
		identity := llvm.ConstInt(elemType, 0, false)
		vec = b.spmdMaskArithVec(vec, execMask, identity)
		return b.spmdCallVectorReduce("umax", vec), nil

	case strings.HasPrefix(name, "reduce.Min["):
		vec := b.getValue(instr.Args[0], getPos(instr))
		argType := instr.Args[0].Type()
		execMask := b.spmdGetReduceMask(instr)
		if spmdType, ok := argType.(*types.SPMDType); ok {
			if spmdIsFloat(spmdType.Elem()) {
				// Identity for fmin is +Inf: inactive lanes contribute +Inf and never win.
				elemType := vec.Type().ElementType()
				identity := llvm.ConstFloatFromString(elemType, "+Inf")
				vec = b.spmdMaskArithVec(vec, execMask, identity)
				return b.spmdCallVectorReduce("fmin", vec), nil
			}
			if spmdIsSignedInt(spmdType.Elem()) {
				// Identity for smin is INT_MAX: inactive lanes contribute the maximum
				// signed value and never beat any active lane's value.
				elemType := vec.Type().ElementType()
				bits := elemType.IntTypeWidth()
				identity := llvm.ConstInt(elemType, (uint64(1)<<(uint(bits)-1))-1, false)
				vec = b.spmdMaskArithVec(vec, execMask, identity)
				return b.spmdCallVectorReduce("smin", vec), nil
			}
		}
		// Unsigned min: identity is UINT_MAX (inactive lanes contribute max, never win).
		elemType := vec.Type().ElementType()
		identity := llvm.ConstAllOnes(elemType)
		vec = b.spmdMaskArithVec(vec, execMask, identity)
		return b.spmdCallVectorReduce("umin", vec), nil

	case strings.HasPrefix(name, "reduce.Or["):
		// Identity for OR is 0: inactive lanes contribute 0, preserving other bits.
		vec := b.getValue(instr.Args[0], getPos(instr))
		elemType := vec.Type().ElementType()
		identity := llvm.ConstInt(elemType, 0, false)
		vec = b.spmdMaskArithVec(vec, b.spmdGetReduceMask(instr), identity)
		return b.spmdCallVectorReduce("or", vec), nil

	case strings.HasPrefix(name, "reduce.And["):
		// Identity for AND is all-ones: inactive lanes contribute all-ones, preserving other bits.
		vec := b.getValue(instr.Args[0], getPos(instr))
		elemType := vec.Type().ElementType()
		identity := llvm.ConstAllOnes(elemType)
		vec = b.spmdMaskArithVec(vec, b.spmdGetReduceMask(instr), identity)
		return b.spmdCallVectorReduce("and", vec), nil

	case strings.HasPrefix(name, "reduce.Xor["):
		// Identity for XOR is 0: inactive lanes contribute 0, preserving other bits.
		vec := b.getValue(instr.Args[0], getPos(instr))
		elemType := vec.Type().ElementType()
		identity := llvm.ConstInt(elemType, 0, false)
		vec = b.spmdMaskArithVec(vec, b.spmdGetReduceMask(instr), identity)
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

		// Varying[aggregateType] uses [N x T] arrays (LLVM vectors require scalar
		// elements); Varying[scalarType] uses <N x T> vectors.
		var laneCount int
		if vecType.TypeKind() == llvm.ArrayTypeKind {
			laneCount = vecType.ArrayLength()
		} else {
			laneCount = vecType.VectorSize()
		}

		// Allocate stack space for the elements.
		arrType := llvm.ArrayType(elemType, laneCount)
		alloca := b.CreateAlloca(arrType, "reduce.from.arr")

		// Extract each element and store. Use extractvalue for [N x T] array
		// types and extractelement for <N x T> vector types.
		for i := 0; i < laneCount; i++ {
			var elem llvm.Value
			if vecType.TypeKind() == llvm.ArrayTypeKind {
				elem = b.CreateExtractValue(vec, i, "")
			} else {
				idx := llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
				elem = b.CreateExtractElement(vec, idx, "")
			}
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
		// reduce.Count(v Varying[bool]) int — count of true active lanes.
		// Inactive lanes are cleared to false before counting so they don't inflate the result.
		vec := b.getValue(instr.Args[0], getPos(instr))
		masked := b.spmdMaskBoolVecForAny(vec, b.spmdGetReduceMask(instr))
		// Normalize to <N x i1> then bitcast to iN and use llvm.ctpop.
		i1Vec := b.spmdNormalizeBoolVecToI1(masked)
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
		// reduce.FindFirstSet(v Varying[bool]) int — index of first true active lane.
		// Inactive lanes are cleared to false before searching so they don't appear first.
		vec := b.getValue(instr.Args[0], getPos(instr))
		masked := b.spmdMaskBoolVecForAny(vec, b.spmdGetReduceMask(instr))
		// Normalize to <N x i1> then bitcast to iN and use llvm.cttz.
		i1Vec := b.spmdNormalizeBoolVecToI1(masked)
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
		// reduce.Mask(v Varying[bool]) int — bitmask of active lanes that are true.
		// Inactive lanes are cleared to false before computing the bitmask.
		// On non-WASM: normalize to <N x i1>, bitcast to iN, zero-extend.
		// On WASM: use native i8x16.bitmask / i16x8.bitmask / i32x4.bitmask
		// / i64x2.bitmask intrinsic which extracts the high bit of each lane.
		vec := b.getValue(instr.Args[0], getPos(instr))
		masked := b.spmdMaskBoolVecForAny(vec, b.spmdGetReduceMask(instr))
		if b.spmdUsesSIMD() {
			// Ensure masked is in WASM mask format (<N x iW>, not <N x i1>).
			// spmdMaskBoolVecForAny returns <N x i1>; sign-extend to WASM format.
			if masked.Type().ElementType() == b.ctx.Int1Type() {
				laneCount := masked.Type().VectorSize()
				maskType := llvm.VectorType(b.spmdMaskElemType(laneCount), laneCount)
				masked = b.CreateSExt(masked, maskType, "")
			}
			// spmdBitmask extracts the high bit of each lane element into a
			// scalar i32 bitmask. For all-ones mask elements (e.g., 0xFF or
			// 0xFFFFFFFF), the high bit is 1; for all-zeros, it's 0.
			// Dispatches to llvm.wasm.bitmask on WASM or pmovmskb on x86.
			result := b.spmdBitmask(masked)
			return b.createZExtOrTrunc(result, b.intType), nil
		}
		i1Vec := b.spmdNormalizeBoolVecToI1(masked)
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
	elemAlign := int(b.targetData.TypeAllocSize(vecType.ElementType()))
	rawLoad.SetAlignment(elemAlign)
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
	if !b.spmdUsesSIMD() {
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
		// Adapt elem to wideElemType. The packed scalar may be wider than
		// wideElemType (e.g., scalarType=i64 for 4×uint16=64 bits, but
		// wideElemType=i32 for 4 lanes on WASM). After the AND, elem holds at
		// most targetElemBits significant bits, so truncation is value-preserving.
		// When narrower, ZExt; when wider, Trunc (the AND already zeroed the high bits).
		scalarBitWidth := uint64(scalarType.IntTypeWidth())
		wideBitWidth := uint64(wideElemType.IntTypeWidth())
		if scalarBitWidth < wideBitWidth {
			elem = b.CreateZExt(elem, wideElemType, "")
		} else if scalarBitWidth > wideBitWidth {
			elem = b.CreateTrunc(elem, wideElemType, "")
		}
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
	if !b.spmdUsesSIMD() {
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
	if b.spmdUsesSIMD() {
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
	// Fast path: all-ones mask — direct store, no blend needed.
	if b.spmdIsConstAllOnesMask(mask) {
		elemAlign := int(b.targetData.TypeAllocSize(val.Type().ElementType()))
		st := b.CreateStore(val, ci.scalarPtr)
		st.SetAlignment(elemAlign)
		return
	}

	vecType := val.Type()

	// Fast path: alloca origin — stack memory is always fully accessible.
	if b.spmdIsAllocaOrigin(ci) {
		// WASM: sub-128-bit vectors (e.g., <4 x i8>) cannot use SIMD load/store.
		// Pack to scalar, blend as integer, store back.
		if b.spmdUsesSIMD() {
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
		// Full-array optimization: when the alloca holds exactly laneCount elements
		// and WASM codegen is active, use the alloca base pointer for both the old
		// load and the blended store. Without this, the store goes through a GEP
		// with a runtime iter offset (e.g. gep(digits, 0, %iter)), preventing LLVM
		// from forwarding the value to a subsequent "load <N x i8>, ptr %digits".
		// Shadow tracking is NOT updated here — this path targets SPMD loop output
		// allocas, which are distinct from the byte-array copy allocas tracked by
		// spmdVecShadow.
		if b.spmdUsesSIMD() {
			if alloc, ok := ci.ssaSource.(*ssa.Alloc); ok {
				ptrType, ptrOK := alloc.Type().Underlying().(*types.Pointer)
				if ptrOK {
					if arrType, arrOK := ptrType.Elem().Underlying().(*types.Array); arrOK {
						if arrType.Len() == int64(ci.loop.laneCount) {
							allocaPtr := b.getValue(alloc, getPos(alloc))
							old := b.CreateLoad(vecType, allocaPtr, "spmd.alloca.old")
							old.SetAlignment(1)
							blended := b.spmdMaskSelect(mask, val, old)
							// Store with align 1 to match the alignment of subsequent
							// identity loads (e.g., in spmdSwizzleFromPtr). LLVM's
							// store-to-load forwarding requires consistent alignment;
							// a store with the default vector alignment (16 for <16 x i8>)
							// and a load with align 1 from the same Go [N]byte alloca
							// prevents EarlyCSE/GVN forwarding, causing the WASM backend
							// to emit load8_splat + load8_lane × (N-1) instead of v128.load.
							st := b.CreateStore(blended, allocaPtr)
							st.SetAlignment(1)
							return
						}
					}
				}
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
	blendElemAlign := int(b.targetData.TypeAllocSize(vecType.ElementType()))
	oldVal.SetAlignment(blendElemAlign)
	blended := b.spmdMaskSelect(mask, val, oldVal)
	st := b.CreateStore(blended, ci.scalarPtr)
	st.SetAlignment(blendElemAlign)
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

// spmdFieldAddrPerLane computes per-lane field addresses when the FieldAddr
// receiver is *Varying[Struct] (i.e., ptrVec is a <N x ptr> vector, each lane
// pointing to a different struct instance). Returns a <N x ptr> vector where
// element i is a pointer to fieldIndex within lane i's struct.
func (b *builder) spmdFieldAddrPerLane(ptrVec llvm.Value, structType llvm.Type, fieldIndex, laneCount int) llvm.Value {
	result := llvm.Undef(ptrVec.Type())
	for i := 0; i < laneCount; i++ {
		idx := llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
		lanePtr := b.CreateExtractElement(ptrVec, idx, "fieldaddr.base")
		fieldPtr := b.CreateInBoundsGEP(structType, lanePtr, []llvm.Value{
			llvm.ConstInt(b.ctx.Int32Type(), 0, false),
			llvm.ConstInt(b.ctx.Int32Type(), uint64(fieldIndex), false),
		}, "fieldaddr.lane")
		result = b.CreateInsertElement(result, fieldPtr, idx, "")
	}
	return result
}

// spmdFieldAddrForVaryingPtr checks if the FieldAddr's base (expr.X) has a
// contiguous access entry (from a prior IndexAddr in a SPMD loop), and if so,
// propagates it to the FieldAddr result with an updated scalarPtr pointing to
// the field's GEP.
//
// This enables SPMDLoad/SPMDStore on the field to use masked vector load/store
// instead of falling back to per-lane scatter/gather.
func (b *builder) spmdFieldAddrForVaryingPtr(expr *ssa.FieldAddr, fieldGEP llvm.Value) {
	if b.spmdContiguousPtr == nil {
		return
	}
	baseCI, ok := b.spmdContiguousPtr[expr.X]
	if !ok {
		return
	}
	// Propagate: the field is contiguous because the base struct is contiguous.
	// Inherit all metadata from the base entry but point scalarPtr at the field GEP.
	fieldCI := *baseCI
	fieldCI.scalarPtr = fieldGEP
	b.spmdContiguousPtr[expr] = &fieldCI
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
	loaded.SetAlignment(1) // Element-aligned: source slice may not be vector-aligned.

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

	// decomp.scalarBase is always i32 (normalized at creation). Ensure scalarLLVM
	// matches so all arithmetic ops (ADD, SUB, MUL, SHR, etc.) stay in i32.
	// On x86-64 int constants and values are i64; truncating to i32 is safe because
	// decomposed indices represent positions within arrays (max 2^31-1 elements).
	i32Type := b.ctx.Int32Type()
	if scalarLLVM.Type().TypeKind() == llvm.IntegerTypeKind && scalarLLVM.Type().IntTypeWidth() > i32Type.IntTypeWidth() {
		scalarLLVM = b.CreateTrunc(scalarLLVM, i32Type, "spmd.scalar.narrow")
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
			// Vectorized prelude: reduce all lane indices to a single scalar max,
			// then do one scalar bounds check. Inactive lanes were clamped to 0
			// above, so they cannot cause a false OOB. This replaces N
			// extract+compare+OR scalar ops with a single vector intrinsic.
			maxIdx := b.spmdVectorReduceUmax(index)
			if maxIdx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
				maxIdx = b.CreateZExt(maxIdx, b.uintptrType, "")
			}
			oob := b.CreateICmp(llvm.IntUGE, maxIdx, length, "spmd.bounds.oob")
			b.createRuntimeAssert(oob, "lookup", "lookupPanic")
		}
	}

	// SPMD: use swizzle/pshufb for const string lookups of <=16 bytes.
	// WASM i8x16.swizzle is always 128-bit so laneCount is capped at 16.
	// On x86 with AVX2, vpshufb operates on 256-bit registers (32 lanes) so
	// the cap is widened to spmdRegisterBytes(). The table content must still
	// fit in 16 bytes because swizzle indices address a 16-entry lookup table
	// (spmdWasmSwizzle duplicates the table for wider registers).
	maxSwizzleLanes := 16
	if !b.spmdIsWASM() && b.spmdRegisterBytes() > 16 {
		maxSwizzleLanes = b.spmdRegisterBytes()
	}
	if b.spmdUsesSIMD() && laneCount <= maxSwizzleLanes {
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
	if b.spmdUsesSIMD() {
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

	// WASM fast paths.
	elemType := arrayType.ElementType()
	if b.spmdUsesSIMD() {
		elemSize := b.targetData.TypeAllocSize(elemType)
		totalSize := int(xType.Len()) * int(elemSize)

		// Identity load elimination: when the array has exactly laneCount elements,
		// the total array fits in v128, and the index is the loop's sequential lane
		// index, the per-lane gather would just return each element in order.
		// Skip the gather and load the array directly as a <N x T> vector.
		if totalSize <= 16 && int64(xType.Len()) == int64(laneCount) && b.spmdIsLoopLaneIndex(index, expr.Index) {
			// Determine result element type: widen sub-128-bit vectors to avoid
			// WASM lowering failures (e.g., <4 x i8>=32-bit or <4 x i16>=64-bit).
			// Matches the same widening logic used in the GEP fallback path below.
			resultElemType := elemType
			vecBits := uint64(b.targetData.TypeAllocSize(llvm.VectorType(elemType, laneCount))) * 8
			if vecBits < 128 {
				resultElemType = b.spmdMaskElemType(laneCount)
			}

			if unop, ok := expr.X.(*ssa.UnOp); ok && unop.Op == token.MUL {
				// Shadow vector bypass: when the alloca has a tracked shadow (kept
				// in sync by insertelement on scalar stores), return it directly
				// instead of loading from the alloca. Avoids the alloca round-trip
				// that LLVM decomposes into per-lane loads.
				if alloc, ok := unop.X.(*ssa.Alloc); ok && b.spmdVecShadow != nil {
					if shadow, ok := b.spmdVecShadow.current[alloc]; ok {
						if shadow.Type() == llvm.VectorType(resultElemType, laneCount) {
							return shadow, nil
						}
						// Shadow has different element width; fall through to load path.
					}
				}

				srcPtr := b.getValue(unop.X, getPos(expr))
				if totalSize == 16 {
					// Load directly as <N x T>. v128.load reads exactly 16 bytes
					// regardless of element type; alignment from the element type.
					vecType := llvm.VectorType(elemType, laneCount)
					align := b.targetData.ABITypeAlignment(elemType)
					load := b.CreateLoad(vecType, srcPtr, "spmd.identity.load")
					load.SetAlignment(align)
					return load, nil
				}
				// Sub-128-bit: load per-lane via GEP+load and zero-extend to resultElemType.
				// We cannot load the entire array as a v128 (it would read garbage padding),
				// and sub-128-bit LLVM vector types cannot be lowered on WASM.
				arrayLoadType := collection.Type()
				result := llvm.Undef(llvm.VectorType(resultElemType, laneCount))
				zero := llvm.ConstInt(b.ctx.Int32Type(), 0, false)
				for lane := 0; lane < laneCount; lane++ {
					laneConst := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
					ptr := b.CreateInBoundsGEP(arrayLoadType, srcPtr, []llvm.Value{zero, laneConst}, "spmd.identity.gep")
					val := b.CreateLoad(elemType, ptr, "spmd.identity.lane")
					if resultElemType != elemType {
						val = b.CreateZExt(val, resultElemType, "")
					}
					result = b.CreateInsertElement(result, val, laneConst, "")
				}
				return result, nil
			}
			// Register-based: convert aggregate to vector via ExtractValue+InsertElement,
			// then zero-extend to resultElemType if widening is required.
			rawVec := b.spmdAggregateToVector(collection, llvm.VectorType(elemType, laneCount), laneCount, "spmd.identity.cast")
			if resultElemType != elemType {
				// Widen each element from elemType to resultElemType.
				result := llvm.Undef(llvm.VectorType(resultElemType, laneCount))
				for lane := 0; lane < laneCount; lane++ {
					laneConst := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
					val := b.CreateExtractElement(rawVec, laneConst, "")
					val = b.CreateZExt(val, resultElemType, "")
					result = b.CreateInsertElement(result, val, laneConst, "")
				}
				return result, nil
			}
			return rawVec, nil
		}

		// pshufb/swizzle fast path: byte arrays ≤ 16 elements.
		// Bounds safety: swizzle returns 0 for indices >= 16 (no memory access,
		// purely register-based). WASM i8x16.swizzle is always 128-bit so laneCount
		// is capped at 16. On x86 with AVX2, vpshufb covers 256-bit registers (32
		// lanes) so the cap is widened to spmdRegisterBytes(). The array length must
		// still be ≤ 16 because the table occupies a single 128-bit row (spmdWasmSwizzle
		// duplicates the row for wider registers).
		maxSwizzleLanes := 16
		if !b.spmdIsWASM() && b.spmdRegisterBytes() > 16 {
			maxSwizzleLanes = b.spmdRegisterBytes()
		}
		if elemType == b.ctx.Int8Type() && int(xType.Len()) <= 16 && laneCount <= maxSwizzleLanes {
			// If the collection was loaded from memory (SSA *UnOp dereference),
			// use the source pointer directly — avoids aggregate→vector conversion
			// that LLVM decomposes into per-byte loads.
			if unop, ok := expr.X.(*ssa.UnOp); ok && unop.Op == token.MUL {
				// Shadow vector bypass: when the alloca has a tracked shadow (kept
				// in sync by spmdFullStoreWithBlend), use it directly as the swizzle
				// table instead of reloading from memory. Avoids LLVM's failure to
				// forward through a GEP-indexed store (v128.load8_splat + N-1 load8_lane).
				if alloc, ok := unop.X.(*ssa.Alloc); ok && b.spmdVecShadow != nil {
					if shadow, ok := b.spmdVecShadow.current[alloc]; ok {
						return b.spmdSwizzleWithTable(shadow, index, laneCount)
					}
				}
				srcPtr := b.getValue(unop.X, getPos(expr))
				return b.spmdSwizzleFromPtr(srcPtr, index, int(xType.Len()), laneCount)
			}
			return b.spmdSwizzleArrayBytes(collection, index, int(xType.Len()), laneCount)
		}
	}

	// Bounds check for the GEP fallback path (actual memory access).
	arrayLen := llvm.ConstInt(b.uintptrType, uint64(xType.Len()), false)
	if !b.info.nobounds {
		canElide := false
		if maxVal, known := spmdIndexMaxValue(expr.Index); known {
			if maxVal < uint64(xType.Len()) {
				canElide = true
			}
		}
		if !canElide {
			maxIdx := b.spmdVectorReduceUmax(index)
			if maxIdx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
				maxIdx = b.CreateZExt(maxIdx, b.uintptrType, "")
			}
			oob := b.CreateICmp(llvm.IntUGE, maxIdx, arrayLen, "spmd.bounds.oob")
			b.createRuntimeAssert(oob, "lookup", "lookupPanic")
		}
	}

	// Fallback: alloca + per-lane GEP + load.
	alloca, allocaSize := b.createTemporaryAlloca(arrayType, "index.alloca")
	b.CreateStore(collection, alloca)

	// Pre-compute extended lane indices for GEP.
	laneIdxs := make([]llvm.Value, laneCount)
	for lane := 0; lane < laneCount; lane++ {
		idx := b.CreateExtractElement(index, llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false), "")
		if idx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
			idx = b.spmdExtendIndex(idx, expr.Index.Type(), b.uintptrType)
		}
		laneIdxs[lane] = idx
	}

	// Per-lane: GEP, load, insert into result vector.
	// On WASM, build the result in the mask element type (e.g., <4 x i32>) to avoid
	// sub-128-bit vectors (e.g., <4 x i8> = 32 bits) which WASM cannot lower.
	// Each loaded byte is zero-extended to the wider element type during insertion
	// rather than after, so no intermediate sub-128-bit vector is ever created.
	resultElemType := elemType
	if b.spmdUsesSIMD() {
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

// spmdPromoteByteArrayCopyToVector detects a [N]T copy to a local alloc and
// returns a <N x T> vector loaded directly from the source pointer. This
// prevents LLVM's store-forwarding from decomposing the subsequent identity
// load <N x T> (emitted by spmdVectorIndexArray) into N individual element loads.
//
// Pattern detected:
//
//	*t72 = t74        -- SSA Store: copy [N]T to local alloc t72
//	t74 = *t73        -- SSA UnOp MUL: t74 is loaded from pointer t73
//
// When this pattern is matched, instead of storing the pre-loaded [N x T]
// aggregate (which LLVM would decompose via store-forwarding), we load <N x T>
// directly from t73's LLVM pointer value. LLVM then sees a clean vector store
// followed by only the scalar patches, and the subsequent identity load
// load <N x T> from the alloca is store-forwarded as:
//
//	<N x T> from_t73_ptr + insert(patch12) + insert(patch13) ...
//
// which produces a single v128.load + N_patches replace_lane instructions
// instead of N individual element loads + N replace_lane ops.
//
// Guards: WASM + SPMD loops present + dest is local *ssa.Alloc + val is a
// *ssa.UnOp{MUL, srcPtr} + srcPtr points to [N]T with N*sizeof(T) <= 16 bytes.
// Zero-total-size arrays are excluded (zero-length types are handled elsewhere).
//
// Returns llvm.Value{} (nil) when the pattern does not match.
func (b *builder) spmdPromoteByteArrayCopyToVector(instr *ssa.Store) llvm.Value {
	if !b.spmdUsesSIMD() {
		return llvm.Value{}
	}
	// Destination must be a local stack alloc.
	alloc, isAlloc := instr.Addr.(*ssa.Alloc)
	if !isAlloc {
		return llvm.Value{}
	}

	// spmdPromoteByteArrayCopyToVector handles two SSA patterns:
	//
	// Pattern A — direct pointer dereference:
	//   Store(alloc, *ptr)  where ptr: *[N]byte
	//
	// Pattern B — struct field extraction from a dereferenced struct:
	//   Store(alloc, Field(structVal, idx))  where structVal = *structPtr
	//   and the field type is [N]byte.
	//
	// In both cases we load <N x i8> directly from the source address (the
	// struct pointer GEP'd to the field, or the original pointer) instead of
	// copying through the alloca as an aggregate. This prevents LLVM's
	// store-forwarding from decomposing the subsequent identity load into N
	// scalar byte loads (which would produce N replace_lane ops in WASM).

	// checkReferrers returns false if val has real referrers beyond instr.
	// DebugRef instructions are skipped — they generate no LLVM IR.
	checkReferrers := func(val ssa.Value) bool {
		refs := val.Referrers()
		if refs == nil {
			return true
		}
		for _, r := range *refs {
			if _, isDebugRef := r.(*ssa.DebugRef); isDebugRef {
				continue
			}
			if r == instr {
				continue
			}
			return false
		}
		return true
	}

	registerShadow := func(load llvm.Value) {
		if b.spmdVecShadow == nil {
			b.spmdVecShadow = &spmdVecShadowState{
				current:  make(map[*ssa.Alloc]llvm.Value),
				blockOut: make(map[llvm.BasicBlock]map[*ssa.Alloc]llvm.Value),
			}
		}
		b.spmdVecShadow.current[alloc] = load
	}

	// Pattern A: Store(alloc, *ptr) where ptr: *[N]T and N*sizeof(T) <= 16.
	if unop, ok := instr.Val.(*ssa.UnOp); ok && unop.Op == token.MUL {
		srcPtrType, ok := unop.X.Type().Underlying().(*types.Pointer)
		if !ok {
			return llvm.Value{}
		}
		arr, ok := srcPtrType.Elem().Underlying().(*types.Array)
		if !ok {
			return llvm.Value{}
		}
		// Allow any element type where the total array fits in v128 (16 bytes).
		elemLLVM := b.getLLVMType(arr.Elem())
		elemSize := b.targetData.TypeAllocSize(elemLLVM)
		n := int(arr.Len())
		totalSize := n * int(elemSize)
		if totalSize == 0 || totalSize > 16 {
			return llvm.Value{}
		}
		// Only promote if no other real instruction uses the dereference result;
		// otherwise getValue would be called with the aggregate type and a vector
		// store would cause a type mismatch.
		if !checkReferrers(unop) {
			return llvm.Value{}
		}

		// Resolve the source pointer. When unop.X is a FieldAddr into a local
		// alloca (i.e. `&entry.shuffleMask` where `entry` is a local copy of a
		// struct from a table), LLVM's SROA pass decomposes the whole-struct
		// store (table→alloca) into 16 individual byte stores. The subsequent
		// `load <16 x i8>` from the alloca is then converted by wasm-ld's LTO
		// SROA into an insertelement chain, which the WASM backend lowers as
		// v128.load8_splat + 15× v128.load8_lane.
		//
		// To avoid this, when unop.X is FieldAddr(structAlloc, fieldIdx) and
		// structAlloc was populated by Store(structAlloc, *srcPtr), compute the
		// source directly: GEP(srcPtr, 0, fieldIdx). This bypasses the alloca
		// and loads directly from the original source memory.
		srcPtr := b.getValue(unop.X, getPos(unop))
		if fa, isFAddr := unop.X.(*ssa.FieldAddr); isFAddr {
			if structAlloc, isAlloc := fa.X.(*ssa.Alloc); isAlloc {
				refs := structAlloc.Referrers()
				if refs != nil {
					for _, ref := range *refs {
						refStore, isStore := ref.(*ssa.Store)
						if !isStore || refStore.Addr != structAlloc {
							continue
						}
						// The stored value must be a dereference: *srcStructPtr.
						srcUnop, isMUL := refStore.Val.(*ssa.UnOp)
						if !isMUL || srcUnop.Op != token.MUL {
							continue
						}
						// srcStructPtr must point to the same struct type as structAlloc.
						srcStructPtrType, ok := srcUnop.X.Type().Underlying().(*types.Pointer)
						if !ok {
							continue
						}
						allocPtrType, ok := structAlloc.Type().Underlying().(*types.Pointer)
						if !ok {
							continue
						}
						if !types.Identical(srcStructPtrType.Elem(), allocPtrType.Elem()) {
							continue
						}
						// Found: bypass the alloca and GEP into the source pointer.
						structSrcPtr := b.getValue(srcUnop.X, getPos(unop))
						structLLVMType := b.getLLVMType(srcStructPtrType.Elem())
						srcPtr = b.CreateGEP(structLLVMType, structSrcPtr, []llvm.Value{
							llvm.ConstInt(b.ctx.Int32Type(), 0, false),
							llvm.ConstInt(b.ctx.Int32Type(), uint64(fa.Field), false),
						}, "spmd.field.ptr")
						break
					}
				}
			}
		}

		vecType := llvm.VectorType(elemLLVM, n)
		align := b.targetData.ABITypeAlignment(elemLLVM)
		load := b.CreateLoad(vecType, srcPtr, "spmd.alloc.vec.copy")
		load.SetAlignment(align)
		registerShadow(load)
		return load
	}

	// Pattern B: Store(alloc, Field(structVal, idx)) where structVal = *structPtr
	// and the field type is [N]T with N*sizeof(T) <= 16.
	if field, ok := instr.Val.(*ssa.Field); ok {
		// The field's parent must be a pointer dereference: structVal = *structPtr.
		structUnop, ok := field.X.(*ssa.UnOp)
		if !ok || structUnop.Op != token.MUL {
			return llvm.Value{}
		}
		// Field type must be [N]T with N*sizeof(T) <= 16.
		arr, ok := field.Type().Underlying().(*types.Array)
		if !ok {
			return llvm.Value{}
		}
		// Allow any element type where the total array fits in v128 (16 bytes).
		elemLLVM := b.getLLVMType(arr.Elem())
		elemSize := b.targetData.TypeAllocSize(elemLLVM)
		n := int(arr.Len())
		totalSize := n * int(elemSize)
		if totalSize == 0 || totalSize > 16 {
			return llvm.Value{}
		}
		// Only promote if no other real instruction uses the Field result.
		if !checkReferrers(field) {
			return llvm.Value{}
		}

		// Determine the struct pointer to GEP from.
		// When structUnop.X is a local *ssa.Alloc (i.e. a value copy like
		// `entry := table[hash]`), loading from the alloca's field runs into a
		// SROA hazard: the alloca is filled by a whole-struct store that LLVM's
		// SROA pass decomposes into 16 individual byte stores. The subsequent
		// `load <16 x i8>` from that alloca is then converted by wasm-ld's LTO
		// SROA into a 16-element insertelement chain, which the WASM backend
		// lowers as v128.load8_splat + 15× v128.load8_lane.
		//
		// To avoid this, when structUnop.X is an alloca, look through it to
		// find the original source pointer via a Store(alloca, *sourcePtr) in
		// the same function. Use that source pointer + field GEP instead.
		structSrcPtr := structUnop.X
		if structAlloc, isAlloc := structUnop.X.(*ssa.Alloc); isAlloc {
			// Search referrers of structAlloc for a Store(structAlloc, *srcPtr).
			refs := structAlloc.Referrers()
			if refs != nil {
				for _, ref := range *refs {
					refStore, isStore := ref.(*ssa.Store)
					if !isStore || refStore.Addr != structAlloc {
						continue
					}
					// The stored value must be a dereference: *srcPtr.
					srcUnop, isMUL := refStore.Val.(*ssa.UnOp)
					if !isMUL || srcUnop.Op != token.MUL {
						continue
					}
					// srcPtr must point to the same struct type.
					srcPtrType, ok := srcUnop.X.Type().Underlying().(*types.Pointer)
					if !ok {
						continue
					}
					structType, ok := structUnop.X.Type().Underlying().(*types.Pointer)
					if !ok {
						continue
					}
					if !types.Identical(srcPtrType.Elem(), structType.Elem()) {
						continue
					}
					// Found: use srcUnop.X (the original pointer) instead of
					// the local alloca, bypassing SROA decomposition.
					structSrcPtr = srcUnop.X
					break
				}
			}
		}

		// Compute a GEP into the struct to get a pointer to the field, then load
		// <N x T> directly — avoiding the aggregate alloca round-trip.
		structPtr := b.getValue(structSrcPtr, getPos(field))
		structLLVMType := b.getLLVMType(structSrcPtr.Type().Underlying().(*types.Pointer).Elem())
		fieldGEP := b.CreateGEP(structLLVMType, structPtr, []llvm.Value{
			llvm.ConstInt(b.ctx.Int32Type(), 0, false),
			llvm.ConstInt(b.ctx.Int32Type(), uint64(field.Field), false),
		}, "spmd.field.ptr")
		vecType := llvm.VectorType(elemLLVM, n)
		align := b.targetData.ABITypeAlignment(elemLLVM)
		load := b.CreateLoad(vecType, fieldGEP, "spmd.alloc.vec.copy")
		load.SetAlignment(align)
		registerShadow(load)
		return load
	}

	return llvm.Value{}
}

// spmdVecShadowUpdate keeps a shadow vector in sync when a scalar byte store writes
// to a constant index of a tracked alloca. Called from the *ssa.Store handler after
// the underlying CreateStore. Emits an insertelement into the shadow.
func (b *builder) spmdVecShadowUpdate(instr *ssa.Store) {
	if b.spmdVecShadow == nil || len(b.spmdVecShadow.current) == 0 {
		return
	}
	// Direct store to a tracked alloc (not through IndexAddr) invalidates the
	// shadow: the alloca content changed in a way we don't track via insertelement.
	// The initial promote path (spmdPromoteByteArrayCopyToVector) sets the shadow;
	// any subsequent direct store means re-initialization we haven't captured.
	if alloc, ok := instr.Addr.(*ssa.Alloc); ok {
		if _, tracked := b.spmdVecShadow.current[alloc]; tracked {
			delete(b.spmdVecShadow.current, alloc)
		}
		return
	}
	ia, ok := instr.Addr.(*ssa.IndexAddr)
	if !ok {
		return
	}
	alloc, ok := ia.X.(*ssa.Alloc)
	if !ok {
		return
	}
	shadow, ok := b.spmdVecShadow.current[alloc]
	if !ok {
		return
	}
	constIdx, ok := ia.Index.(*ssa.Const)
	if !ok {
		// Non-constant index: drop tracking — shadow can no longer be forwarded.
		delete(b.spmdVecShadow.current, alloc)
		return
	}
	idxVal, ok := constant.Int64Val(constIdx.Value)
	if !ok || idxVal < 0 || int(idxVal) >= shadow.Type().VectorSize() {
		delete(b.spmdVecShadow.current, alloc)
		return
	}

	val := b.getValue(instr.Val, getPos(instr))
	elemType := shadow.Type().ElementType()
	if val.Type() != elemType {
		val = b.CreateTrunc(val, elemType, "spmd.shadow.trunc")
	}
	idx := llvm.ConstInt(b.ctx.Int32Type(), uint64(idxVal), false)
	newShadow := b.CreateInsertElement(shadow, val, idx, "spmd.shadow.insert")
	b.spmdVecShadow.current[alloc] = newShadow
}

// spmdVecShadowSaveBlock snapshots the current shadow state for the given LLVM
// exit block so that successor blocks can inherit or merge it via phi nodes.
func (b *builder) spmdVecShadowSaveBlock(exitBlock llvm.BasicBlock) {
	if b.spmdVecShadow == nil || len(b.spmdVecShadow.current) == 0 {
		return
	}
	out := make(map[*ssa.Alloc]llvm.Value, len(b.spmdVecShadow.current))
	for k, v := range b.spmdVecShadow.current {
		out[k] = v
	}
	b.spmdVecShadow.blockOut[exitBlock] = out
}

// spmdVecShadowBlockEntry restores or merges shadow state at the start of a new
// basic block. For single-predecessor blocks the shadow is inherited directly.
// For merge blocks (multiple preds) an LLVM phi is inserted at the block entry.
// When a predecessor has not yet been processed (DomPreorder may visit merge
// blocks before some of their predecessors), a partial phi is created and
// recorded in pendingPhis for completion by spmdVecShadowFinalize.
func (b *builder) spmdVecShadowBlockEntry(entryBlock llvm.BasicBlock, ssaBlock *ssa.BasicBlock) {
	if b.spmdVecShadow == nil {
		return
	}
	preds := ssaBlock.Preds
	if len(preds) == 0 {
		return
	}

	// Collect allocs tracked by any already-processed predecessor.
	allocs := make(map[*ssa.Alloc]bool)
	for _, pred := range preds {
		predLLVM := b.blockInfo[pred.Index].exit
		if out, ok := b.spmdVecShadow.blockOut[predLLVM]; ok {
			for alloc := range out {
				allocs[alloc] = true
			}
		}
	}
	if len(allocs) == 0 {
		return
	}

	// Save insert point; phi nodes must be at block entry before any instruction.
	savedIP := b.GetInsertBlock()
	firstInstr := entryBlock.FirstInstruction()
	if firstInstr.IsNil() {
		b.SetInsertPointAtEnd(entryBlock)
	} else {
		b.SetInsertPointBefore(firstInstr)
	}

	for alloc := range allocs {
		var vals []llvm.Value
		var predBlocks []llvm.BasicBlock
		allSame := true
		var first llvm.Value
		hasMissing := false

		for _, pred := range preds {
			predLLVM := b.blockInfo[pred.Index].exit
			var v llvm.Value
			if out, ok := b.spmdVecShadow.blockOut[predLLVM]; ok {
				v = out[alloc]
			}
			if v.IsNil() {
				// Predecessor has no shadow yet (not yet processed, or never tracks this alloc).
				hasMissing = true
				continue
			}
			vals = append(vals, v)
			predBlocks = append(predBlocks, predLLVM)
			if first.IsNil() {
				first = v
			} else if v.C != first.C {
				allSame = false
			}
		}

		if len(vals) == 0 {
			// No predecessor has a shadow — skip.
			delete(b.spmdVecShadow.current, alloc)
			continue
		}

		if !hasMissing && allSame {
			// All predecessors agree on the same value — no phi needed.
			b.spmdVecShadow.current[alloc] = first
			continue
		}

		// Create a phi (possibly partial if hasMissing). Partial phis are completed
		// by spmdVecShadowFinalize after all blocks are processed.
		phi := b.CreatePHI(vals[0].Type(), "spmd.shadow.phi")
		b.spmdVecShadow.current[alloc] = phi
		if hasMissing {
			// Defer filling in predecessor edges until all blocks are compiled.
			b.spmdVecShadow.pendingPhis = append(b.spmdVecShadow.pendingPhis, spmdVecShadowPendingPhi{
				phi:      phi,
				ssaBlock: ssaBlock,
				alloc:    alloc,
			})
		} else {
			phi.AddIncoming(vals, predBlocks)
		}
	}

	b.SetInsertPointAtEnd(savedIP)
}

// spmdVecShadowFinalize completes any LLVM phi nodes that were created with
// partial predecessors during spmdVecShadowBlockEntry. Called once after all
// SSA blocks have been compiled (so all blockOut entries are populated).
//
// For each pending phi, we walk all SSA predecessors and look up the shadow
// value for the alloc in blockOut. Missing predecessors (alloc not tracked on
// that path) get llvm.Undef, which LLVM will propagate/simplify. Phis with
// all-same predecessors are left as phis (never erased) so that blockOut
// entries referencing them remain valid throughout finalization.
func (b *builder) spmdVecShadowFinalize() {
	if b.spmdVecShadow == nil || len(b.spmdVecShadow.pendingPhis) == 0 {
		return
	}
	for _, pending := range b.spmdVecShadow.pendingPhis {
		for _, pred := range pending.ssaBlock.Preds {
			predLLVM := b.blockInfo[pred.Index].exit
			var v llvm.Value
			if out, ok := b.spmdVecShadow.blockOut[predLLVM]; ok {
				v = out[pending.alloc]
			}
			if v.IsNil() {
				// Predecessor doesn't track this alloc — use undef so the phi
				// remains valid LLVM IR. LLVM will simplify paths that never
				// contribute to the final swizzle result.
				v = llvm.Undef(pending.phi.Type())
			}
			pending.phi.AddIncoming([]llvm.Value{v}, []llvm.BasicBlock{predLLVM})
		}
	}
	b.spmdVecShadow.pendingPhis = nil
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

// spmdWasmSwizzle generates a byte-permute lookup using a constant table and a
// vector of indices. The table is automatically duplicated when the swizzle width
// exceeds the table size (e.g., a 16-byte hextable becomes [table, table] for
// AVX2 32-lane swizzle, because vpshufb ymm shuffles each 128-bit half independently).
//
// For WASM (always 128-bit) and SSE (laneCount ≤ 16): swizzleWidth = 16, table
// zero-padded to 16 bytes. For AVX2 (laneCount > 16, non-WASM): swizzleWidth =
// laneCount; table duplicated to fill the register.
//
// For lane counts < swizzleWidth, the result is narrowed to laneCount lanes.
func (b *builder) spmdWasmSwizzle(tableBytes []byte, index llvm.Value, laneCount int) llvm.Value {
	i8Type := b.ctx.Int8Type()
	tableLen := len(tableBytes)

	// Determine the swizzle width: use the native register width for the target.
	// WASM always stays at 128-bit (16 lanes); AVX2 uses the full 256-bit (32 lanes).
	swizzleWidth := 16
	if laneCount > 16 && !b.spmdIsWASM() {
		swizzleWidth = laneCount
	}

	// Build the constant table vector. When swizzleWidth > tableLen, duplicate the
	// table to fill all positions (required by AVX2 vpshufb which shuffles each
	// 128-bit half independently — each half needs its own copy of the table).
	tableElems := make([]llvm.Value, swizzleWidth)
	for i := 0; i < swizzleWidth; i++ {
		srcIdx := i % tableLen
		if srcIdx < tableLen {
			tableElems[i] = llvm.ConstInt(i8Type, uint64(tableBytes[srcIdx]), false)
		} else {
			tableElems[i] = llvm.ConstInt(i8Type, 0, false)
		}
	}
	tableVec := llvm.ConstVector(tableElems, false)

	// Prepare index to match swizzle width.
	idxVec := b.spmdSwizzlePrepareIndex(index, laneCount, swizzleWidth)

	// Identity swizzle optimization: if the index is [0, 1, 2, ..., N-1],
	// the swizzle is a no-op — use the table directly.
	var result llvm.Value
	if spmdIsIdentitySwizzleIndex(idxVec) {
		result = tableVec
	} else {
		// Dispatch to the appropriate swizzle implementation for the target.
		result = b.spmdSwizzle(tableVec, idxVec)
	}

	// If laneCount < swizzleWidth, extract the first laneCount lanes.
	if laneCount < swizzleWidth {
		swizzleVecType := llvm.VectorType(i8Type, swizzleWidth)
		maskElems := make([]llvm.Value, laneCount)
		for i := 0; i < laneCount; i++ {
			maskElems[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(i), false)
		}
		maskVec := llvm.ConstVector(maskElems, false)
		result = b.CreateShuffleVector(result, llvm.Undef(swizzleVecType), maskVec, "")
	}

	return result
}

// spmdAggregateToVector converts an aggregate [N x T] value to a <N x T> LLVM
// vector using ExtractValue + InsertElement. Direct bitcast between aggregate
// and vector types is illegal in LLVM IR, so this register-only sequence is
// required. name is used as the InsertElement instruction name prefix.
func (b *builder) spmdAggregateToVector(agg llvm.Value, vecType llvm.Type, n int, name string) llvm.Value {
	i32Type := b.ctx.Int32Type()
	vec := llvm.ConstNull(vecType)
	for i := 0; i < n; i++ {
		elem := b.CreateExtractValue(agg, i, "")
		vec = b.CreateInsertElement(vec, elem,
			llvm.ConstInt(i32Type, uint64(i), false), name)
	}
	return vec
}

// spmdSwizzleArrayBytes uses i8x16.swizzle to perform vectorized byte lookup
// from a runtime [N]byte array (N ≤ 16) on WASM. The array is loaded into a
// <16 x i8> register and swizzled with the index vector. For sub-128-bit
// results (laneCount < 16), each byte is widened to the mask element type.
func (b *builder) spmdSwizzleArrayBytes(collection, index llvm.Value, arrayLen, laneCount int) (llvm.Value, error) {
	i8Type := b.ctx.Int8Type()
	i32Type := b.ctx.Int32Type()
	v16i8 := llvm.VectorType(i8Type, 16)

	// Convert [N x i8] aggregate to <16 x i8> vector entirely in registers.
	// Aggregates are first-class SSA values — no memory round-trip needed.
	tableVec := llvm.ConstNull(v16i8)
	for i := 0; i < arrayLen; i++ {
		val := b.CreateExtractValue(collection, i, "")
		tableVec = b.CreateInsertElement(tableVec, val,
			llvm.ConstInt(i32Type, uint64(i), false), "")
	}

	return b.spmdSwizzleWithTable(tableVec, index, laneCount)
}

// spmdSwizzleFromPtr loads a byte array (≤ 16 bytes) directly from a pointer
// as a <16 x i8> vector and swizzles it with the index vector. For N=16, this
// is a single v128.load — no alloca round-trip needed. For N<16, loads the
// array and zero-pads to 16 bytes.
func (b *builder) spmdSwizzleFromPtr(ptr, index llvm.Value, arrayLen, laneCount int) (llvm.Value, error) {
	i8Type := b.ctx.Int8Type()
	i32Type := b.ctx.Int32Type()
	v16i8 := llvm.VectorType(i8Type, 16)

	var tableVec llvm.Value
	if arrayLen == 16 {
		// Load directly as <16 x i8> from the pointer — one v128.load.
		// Use alignment 1: Go [16]byte has no alignment guarantee beyond 1.
		load := b.CreateLoad(v16i8, ptr, "swizzle.table")
		load.SetAlignment(1)
		tableVec = load
	} else {
		// Load the smaller array, then pad to 16 bytes.
		arrType := llvm.ArrayType(i8Type, arrayLen)
		arrVal := b.CreateLoad(arrType, ptr, "swizzle.arr")
		tableVec = llvm.ConstNull(v16i8)
		for i := 0; i < arrayLen; i++ {
			val := b.CreateExtractValue(arrVal, i, "")
			tableVec = b.CreateInsertElement(tableVec, val,
				llvm.ConstInt(i32Type, uint64(i), false), "")
		}
	}

	return b.spmdSwizzleWithTable(tableVec, index, laneCount)
}

// spmdCoalescedGather emits a single i8x16.swizzle for the entire SPMDGatherGroup
// that expr belongs to, then extracts the per-member column for this particular
// IndexAddr. On subsequent calls for the same group the cached merged result is
// used directly, so only one swizzle instruction is emitted per group.
//
// The merged <16 x i8> swizzle result layout (stride = 16/laneCount):
//
//	[ m0_l0, m1_l0, …, pad,  m0_l1, m1_l1, …, pad,  … ]
//	  ←── stride ──→          ←── stride ──→
//
// Each stride group holds one byte per group member (at positions 0…maxOffset)
// and padding bytes (set to 0xFF so swizzle returns 0) for unused positions.
func (b *builder) spmdCoalescedGather(expr *ssa.IndexAddr, arrayPtr, index llvm.Value, laneCount int) (llvm.Value, error) {
	group := expr.SPMDGatherGroup
	// Include the SSA mask value in the cache key: two members of the same gather
	// group that live in different varying-switch cases have different SPMDMask values
	// (e.g., case-3 mask vs case-2 mask). Using only the group pointer would reuse a
	// merged swizzle built under the wrong mask, producing incorrect inactive-lane indices.
	cacheKey := spmdGatherCacheKey{group: group, mask: expr.SPMDMask}

	// Return cached merged result if the group was already processed under this mask.
	if merged, ok := b.spmdGatherCache[cacheKey]; ok {
		return b.spmdExtractGatherColumn(merged, expr.SPMDGatherPos, laneCount)
	}

	// First member: build and emit the merged swizzle for the whole group.

	i8Type := b.ctx.Int8Type()
	i32Type := b.ctx.Int32Type()
	v16i8 := llvm.VectorType(i8Type, 16)

	// Load the source array as <16 x i8>.
	ptrTyp := expr.X.Type().Underlying().(*types.Pointer)
	arrTyp := ptrTyp.Elem().Underlying().(*types.Array)
	arrLen := int(arrTyp.Len())

	var tableVec llvm.Value
	if arrLen == 16 {
		load := b.CreateLoad(v16i8, arrayPtr, "gather.table")
		load.SetAlignment(1)
		tableVec = load
	} else {
		arrType := llvm.ArrayType(i8Type, arrLen)
		arrVal := b.CreateLoad(arrType, arrayPtr, "gather.arr")
		tableVec = llvm.ConstNull(v16i8)
		for i := 0; i < arrLen; i++ {
			elem := b.CreateExtractValue(arrVal, i, "")
			tableVec = b.CreateInsertElement(tableVec, elem,
				llvm.ConstInt(i32Type, uint64(i), false), "")
		}
	}

	// Determine the maximum offset across all group members.
	maxOffset := 0
	for _, m := range group.Members {
		if m.Offset > maxOffset {
			maxOffset = m.Offset
		}
	}

	stride := group.Stride // bytes per lane in the merged result (= 16 / laneCount)

	// Build the base index: this member's index minus its offset gives base[0].
	// The base is a <laneCount x i32> vector of the starting element for each lane.
	baseIndex := index
	if expr.SPMDGatherPos > 0 {
		offsetScalar := llvm.ConstInt(i32Type, uint64(expr.SPMDGatherPos), false)
		offsetVec := b.splatScalar(offsetScalar, index.Type())
		baseIndex = b.CreateSub(index, offsetVec, "gather.base")
	}

	// Truncate base from <laneCount x i32> to <laneCount x i8> (indices fit in a byte).
	baseI8x4 := b.CreateTrunc(baseIndex, llvm.VectorType(i8Type, laneCount), "gather.base.i8")

	// Replicate each lane's base byte across its stride group in a <16 x i8>.
	// Shuffle mask: lane L occupies positions [L*stride … L*stride+stride-1];
	// all positions in that group pick from element L of the first operand.
	// Positions beyond laneCount*stride pick from a zero second operand (undef-safe).
	shuffleMask1 := make([]llvm.Value, 16)
	for i := 0; i < 16; i++ {
		lane := i / stride
		if lane < laneCount {
			shuffleMask1[i] = llvm.ConstInt(i32Type, uint64(lane), false)
		} else {
			// Pick element 0 of the zero second operand (laneCount + 0).
			shuffleMask1[i] = llvm.ConstInt(i32Type, uint64(laneCount), false)
		}
	}
	zeros4 := llvm.ConstNull(llvm.VectorType(i8Type, laneCount))
	replicated := b.CreateShuffleVector(baseI8x4, zeros4,
		llvm.ConstVector(shuffleMask1, false), "gather.rep")
	// replicated = [s0,s0,s0,…, s1,s1,s1,…, s2,s2,s2,…, s3,s3,s3,…] (stride groups)

	// Add per-position offsets within each stride group.
	// Position (lane*stride + posInStride) gets offset posInStride when posInStride ≤ maxOffset,
	// or 0 for padding positions (the padding index will be masked to 0xFF below).
	offsetElems := make([]llvm.Value, 16)
	for i := 0; i < 16; i++ {
		posInStride := i % stride
		if posInStride <= maxOffset {
			offsetElems[i] = llvm.ConstInt(i8Type, uint64(posInStride), false)
		} else {
			offsetElems[i] = llvm.ConstInt(i8Type, 0, false)
		}
	}
	offsetVec := llvm.ConstVector(offsetElems, false)
	indexed := b.CreateAdd(replicated, offsetVec, "gather.idx")
	// indexed = [s0+0, s0+1, …, s1+0, s1+1, …, …]

	// For positions outside the valid member range (posInStride > maxOffset, or
	// lane >= laneCount) set the index to 0xFF so the swizzle returns 0 there.
	// These positions are extracted by spmdExtractGatherColumn only as padding.
	validElems := make([]llvm.Value, 16)
	for i := 0; i < 16; i++ {
		posInStride := i % stride
		lane := i / stride
		if lane < laneCount && posInStride <= maxOffset {
			validElems[i] = llvm.ConstInt(i8Type, 0xFF, false) // keep indexed
		} else {
			validElems[i] = llvm.ConstInt(i8Type, 0, false) // will be replaced by 0xFF
		}
	}
	validMaskVec := llvm.ConstVector(validElems, false)
	invalidFill := b.splatScalar(llvm.ConstInt(i8Type, 0xFF, false), v16i8)
	// Use icmp ne to get <16 x i1>, then CreateSelect to choose indexed or 0xFF.
	isValid := b.CreateICmp(llvm.IntNE, validMaskVec, llvm.ConstNull(v16i8), "gather.valid")
	mergedIdx := b.CreateSelect(isValid, indexed, invalidFill, "gather.merged.idx")

	// Emit a single swizzle for the whole group, dispatched to target.
	merged := b.spmdSwizzle(tableVec, mergedIdx)

	// Cache the merged result for the remaining members under the same mask context.
	b.spmdGatherCache[cacheKey] = merged

	return b.spmdExtractGatherColumn(merged, expr.SPMDGatherPos, laneCount)
}

// spmdExtractGatherColumn extracts one byte per lane from a merged <16 x i8>
// swizzle result (produced by spmdCoalescedGather) and widens it to <laneCount x i32>
// to match the output format of individual spmdSwizzleFromPtr/spmdSwizzleWithTable calls
// on WASM (which zero-extend each byte to i32).
//
// The merged layout uses stride = 16/laneCount bytes per lane. For the member at
// memberPos, the relevant byte for lane L is at index L*stride + memberPos in the
// merged <16 x i8>.
func (b *builder) spmdExtractGatherColumn(merged llvm.Value, memberPos, laneCount int) (llvm.Value, error) {
	i8Type := b.ctx.Int8Type()
	i32Type := b.ctx.Int32Type()
	stride := 16 / laneCount
	v16i8 := llvm.VectorType(i8Type, 16)

	// Combined column extraction + zero-extend via single i8x16.shuffle + bitcast.
	//
	// For laneCount=4, stride=4, memberPos=0: column bytes are at merged[0,4,8,12].
	// We build a <16 x i8> where each 4-byte group has the column byte in the
	// little-endian LSB (byte 0) and zeros in bytes 1-3. Bitcasting to <4 x i32>
	// gives us the zero-extended i32 values directly.
	//
	// Shuffle mask (memberPos=0): [0, 16, 16, 16, 4, 16, 16, 16, 8, 16, 16, 16, 12, 16, 16, 16]
	// where index >= 16 refers to the second operand (zeroinitializer), giving 0 bytes.
	//
	// Cost: 1 i8x16.shuffle + 1 free bitcast = 2 ops total (vs 12 extract+zext+insert).
	zero := llvm.ConstNull(v16i8)
	shuffleMask := make([]llvm.Value, 16)
	for lane := 0; lane < laneCount; lane++ {
		srcIdx := lane*stride + memberPos // byte position in merged result
		for bytePos := 0; bytePos < (16 / laneCount); bytePos++ {
			outPos := lane*(16/laneCount) + bytePos
			if bytePos == 0 {
				// LSB: take the column byte from merged (first operand)
				shuffleMask[outPos] = llvm.ConstInt(i32Type, uint64(srcIdx), false)
			} else {
				// High bytes: take zero from second operand (index 16+)
				shuffleMask[outPos] = llvm.ConstInt(i32Type, uint64(16+bytePos), false)
			}
		}
	}

	packed := b.CreateShuffleVector(merged, zero,
		llvm.ConstVector(shuffleMask, false), "gather.col.shuf")

	// Bitcast <16 x i8> to <laneCount x i32> — free on all targets.
	v4i32 := llvm.VectorType(i32Type, laneCount)
	result := b.CreateBitCast(packed, v4i32, "gather.col.i32")

	return result, nil
}

// spmdSwizzleWithTable performs a byte-permute lookup using a pre-built <16 x i8>
// table vector and a vector of indices. Handles AVX2 by duplicating the 16-byte
// table across both 128-bit halves (vpshufb shuffles each half independently).
func (b *builder) spmdSwizzleWithTable(tableVec, index llvm.Value, laneCount int) (llvm.Value, error) {
	i32Type := b.ctx.Int32Type()

	// Determine swizzle width: WASM stays 16; AVX2 uses full register width.
	// AVX2 vpshufb shuffles each 128-bit half independently, so the 16-byte
	// table must be duplicated to fill both halves.
	swizzleWidth := 16
	if laneCount > 16 && !b.spmdIsWASM() {
		swizzleWidth = laneCount
	}

	// Duplicate table if swizzle is wider than 16 bytes.
	// tableVec is <16 x i8>; shuffle it to <swizzleWidth x i8> by repeating
	// indices [0..15, 0..15, ...].
	if swizzleWidth > 16 {
		maskElems := make([]llvm.Value, swizzleWidth)
		for i := 0; i < swizzleWidth; i++ {
			maskElems[i] = llvm.ConstInt(i32Type, uint64(i%16), false)
		}
		maskVec := llvm.ConstVector(maskElems, false)
		tableVec = b.CreateShuffleVector(tableVec, llvm.Undef(tableVec.Type()), maskVec, "swizzle.dup")
	}

	// Prepare index to match swizzle width.
	idxVec := b.spmdSwizzlePrepareIndex(index, laneCount, swizzleWidth)

	// Identity swizzle optimization: if the index is [0, 1, 2, ..., N-1],
	// the swizzle is a no-op — use the table directly.
	var swizzled llvm.Value
	if spmdIsIdentitySwizzleIndex(idxVec) {
		swizzled = tableVec
	} else {
		swizzled = b.spmdSwizzle(tableVec, idxVec)
	}

	// Fast path: result already has the right width.
	if laneCount == swizzleWidth {
		return swizzled, nil
	}

	// Extract laneCount bytes from wider swizzle result via shufflevector.
	if laneCount < swizzleWidth {
		maskElems := make([]llvm.Value, laneCount)
		for i := 0; i < laneCount; i++ {
			maskElems[i] = llvm.ConstInt(i32Type, uint64(i), false)
		}
		maskVec := llvm.ConstVector(maskElems, false)
		swizzled = b.CreateShuffleVector(swizzled, llvm.Undef(swizzled.Type()), maskVec, "swizzle.narrow")
	}

	return swizzled, nil
}

// spmdIsLoopLaneIndex checks whether the given LLVM vector value is the
// sequential lane index of the active SPMD loop (e.g., <iter, iter+1, ...,
// iter+N-1>). Used to detect identity swizzles: when an array index equals
// the loop's lane index and the array length equals the lane count, the
// swizzle is a no-op and can be elided.
// For decomposed loops (laneIndices is nil), accepts the SSA index value
// to check against the loop's bodyIterValue.
func (b *builder) spmdIsLoopLaneIndex(index llvm.Value, ssaIndex ...ssa.Value) bool {
	if b.spmdLoopState == nil || b.currentBlock == nil {
		return false
	}
	loop := b.spmdFindActiveLoopForBlock(b.currentBlock)
	if loop == nil {
		return false
	}
	// Non-decomposed: compare LLVM values directly.
	if !loop.laneIndices.IsNil() {
		return index.C == loop.laneIndices.C
	}
	// Decomposed: compare SSA values (the LLVM value is materialized fresh each time).
	if len(ssaIndex) > 0 && loop.bodyIterValue != nil && ssaIndex[0] != nil {
		return ssaIndex[0] == loop.bodyIterValue
	}
	return false
}

// spmdIsIdentitySwizzleIndex checks whether a <16 x i8> swizzle index vector is
// the identity permutation [0, 1, 2, ..., 15]. When true, the swizzle is a no-op
// and can be elided — the table vector IS the result.
// spmdIsConstSplat returns true if v is a constant vector where all elements
// have the same value (a splat constant like <4 x i32> splat(4)).
func spmdIsConstSplat(v llvm.Value) bool {
	if v.Type().TypeKind() != llvm.VectorTypeKind {
		return false
	}
	n := v.Type().VectorSize()
	if n == 0 {
		return false
	}
	first := llvm.ConstExtractElement(v, llvm.ConstInt(v.Type().Context().Int32Type(), 0, false))
	for i := 1; i < n; i++ {
		elem := llvm.ConstExtractElement(v, llvm.ConstInt(v.Type().Context().Int32Type(), uint64(i), false))
		if elem != first {
			return false
		}
	}
	return true
}

func spmdIsIdentitySwizzleIndex(v llvm.Value) bool {
	if !v.IsConstant() || v.Type().TypeKind() != llvm.VectorTypeKind {
		return false
	}
	n := v.Type().VectorSize()
	i32Type := v.Type().Context().Int32Type()
	for i := 0; i < n; i++ {
		elem := llvm.ConstExtractElement(v, llvm.ConstInt(i32Type, uint64(i), false))
		if elem.IsUndef() {
			// Undef padding lanes (from spmdSwizzlePrepareIndex for laneCount < 16)
			// are acceptable — they correspond to don't-care positions.
			continue
		}
		if elem.ZExtValue() != uint64(i) {
			return false
		}
	}
	return true
}

// spmdSwizzlePrepareIndex converts a vector index to <targetWidth x i8> for use
// with a swizzle instruction. If the index element type is wider than i8 (e.g.,
// <4 x i32>), it is truncated to <laneCount x i8>. If laneCount == targetWidth,
// the index is returned as-is. If laneCount < targetWidth, the index is padded to
// targetWidth lanes using a shuffle; padding lanes pick from an undef second operand
// and are safe because the caller discards those lanes from the swizzle result.
func (b *builder) spmdSwizzlePrepareIndex(index llvm.Value, laneCount, targetWidth int) llvm.Value {
	i8Type := b.ctx.Int8Type()

	// Truncate to i8 if wider (safe since table indices are 0-15).
	elemWidth := index.Type().ElementType().IntTypeWidth()
	if elemWidth > 8 {
		narrowType := llvm.VectorType(i8Type, laneCount)
		index = b.CreateTrunc(index, narrowType, "")
	}

	if laneCount == targetWidth {
		return index
	}

	// Pad to targetWidth lanes using a shuffle. Padding lanes pick from the undef
	// second operand and produce undef; the swizzle result for those lanes is discarded
	// by the caller's extraction step, so undef padding is safe.
	maskElems := make([]llvm.Value, targetWidth)
	for i := 0; i < targetWidth; i++ {
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

		case token.ADD:
			// x + y: upper bound is maxOf(x) + maxOf(y), guarding against overflow.
			if maxX, okX := spmdIndexMaxValue(val.X); okX {
				if maxY, okY := spmdIndexMaxValue(val.Y); okY {
					sum := maxX + maxY
					if sum >= maxX && sum >= maxY { // overflow guard
						return sum, true
					}
				}
			}

		case token.SUB:
			// x - y: upper bound is maxOf(x) (subtraction can only decrease).
			// Only safe when y is a non-negative constant — for signed types,
			// x - (-y) = x + y which can exceed maxOf(x).
			if maxX, okX := spmdIndexMaxValue(val.X); okX {
				if _, okY := ssaConstUint64(val.Y); okY {
					return maxX, true
				}
			}

		case token.MUL:
			// x * y: upper bound is maxOf(x) * maxOf(y), guarding against overflow.
			if maxX, okX := spmdIndexMaxValue(val.X); okX {
				if maxY, okY := spmdIndexMaxValue(val.Y); okY {
					if maxX == 0 || maxY == 0 {
						return 0, true
					}
					prod := maxX * maxY
					if prod/maxX == maxY { // overflow guard
						return prod, true
					}
				}
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

	// Clamp N to the actual vector lane count. On x86-64, the group laneCount may
	// be larger than the actual LLVM vector (e.g., laneCount=32 for bytes on AVX2
	// but x86 pshufb always produces <16 x i8>). Use the actual lane count so
	// shufflevector indices remain within bounds.
	if vals[0].Type().TypeKind() == llvm.VectorTypeKind {
		if actualN := vals[0].Type().VectorSize(); actualN < N {
			N = actualN
		}
	}

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
	// On x86-64 with laneCount > 4 (decomposed path), the scalar base is in
	// spmdDecomposed. On WASM (non-decomposed), extract lane 0 from the LLVM
	// vector value — lane 0 always holds the smallest index which equals the
	// scalar base for the current SIMD group (e.g., g*stride+0 for g=0,4,8,...).
	scalarBase := llvm.Value{}
	if b.spmdDecomposed != nil {
		if decomp, ok := b.spmdDecomposed[addr0.Index]; ok {
			scalarBase = decomp.scalarBase
		}
	}
	if scalarBase.IsNil() {
		// Non-decomposed path (WASM, laneCount <= 4): get the LLVM value of the
		// varying index vector and extract element 0, which is the scalar base.
		idxVec := b.getValue(addr0.Index, getPos(addr0))
		if idxVec.Type().TypeKind() == llvm.VectorTypeKind {
			scalarBase = b.CreateExtractElement(idxVec,
				llvm.ConstInt(b.ctx.Int32Type(), 0, false), "interleaved.base")
		} else {
			scalarBase = idxVec
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

	// Narrow output vectors to match the destination element type when needed.
	// This handles the case where values were widened from bytes to i32 by the
	// swizzle fast path (spmdSwizzleWithTable widens <N x i8> to <N x i32> on
	// WASM to avoid sub-128-bit vectors). If the outVecs have wider element types
	// than the slice element type, truncate them now so the masked store emits
	// byte stores rather than i32 stores.
	if elemType.TypeKind() == llvm.IntegerTypeKind {
		destWidth := elemType.IntTypeWidth()
		for k, v := range outVecs {
			if v.Type().TypeKind() == llvm.VectorTypeKind {
				srcElem := v.Type().ElementType()
				if srcElem.TypeKind() == llvm.IntegerTypeKind && srcElem.IntTypeWidth() > destWidth {
					outVecs[k] = b.CreateTrunc(v, llvm.VectorType(elemType, v.Type().VectorSize()), "interleave.trunc")
				}
			}
		}
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

// spmdWidenMaskToOperandLanes widens a mask from M lanes to N lanes (N > M,
// N % M == 0) by replicating each lane (N/M) times using a shufflevector.
// This is needed when a Varying[bool] accumulator (16 lanes on WASM128) is
// selected by a loop mask derived from a narrower type (e.g., 4-lane int32).
//
// The output mask uses the same element type as the input. For WASM i32 masks
// the result is <N x i32>; for i1 masks the result is <N x i1>.
func (b *builder) spmdWidenMaskToOperandLanes(mask llvm.Value, targetLanes int) llvm.Value {
	maskType := mask.Type()
	srcLanes := maskType.VectorSize()
	if srcLanes == targetLanes {
		return mask
	}
	if targetLanes%srcLanes != 0 {
		panic(fmt.Sprintf("spmdWidenMaskToOperandLanes: targetLanes %d is not a multiple of srcLanes %d", targetLanes, srcLanes))
	}
	ratio := targetLanes / srcLanes

	// Handle constant masks directly to avoid emitting runtime instructions.
	if mask.IsConstant() {
		if mask.IsNull() {
			return llvm.ConstNull(llvm.VectorType(maskType.ElementType(), targetLanes))
		}
		return llvm.ConstAllOnes(llvm.VectorType(maskType.ElementType(), targetLanes))
	}

	// Normalize to <srcLanes x i1> so we can work with plain bits.
	i1Type := b.ctx.Int1Type()
	srcElem := maskType.ElementType()
	var i1Mask llvm.Value
	if srcElem == i1Type {
		i1Mask = mask
	} else {
		i1Mask = b.CreateTrunc(mask, llvm.VectorType(i1Type, srcLanes), "spmd.widen.trunc")
	}

	// Build a shuffle that replicates each source lane `ratio` times:
	// [0,0,...,0, 1,1,...,1, ..., srcLanes-1,...,srcLanes-1]
	shuffleElems := make([]llvm.Value, targetLanes)
	for i := 0; i < targetLanes; i++ {
		srcIdx := i / ratio
		shuffleElems[i] = llvm.ConstInt(b.ctx.Int32Type(), uint64(srcIdx), false)
	}
	shuffleMask := llvm.ConstVector(shuffleElems, false)
	undef := llvm.Undef(llvm.VectorType(i1Type, srcLanes))
	widened := b.CreateShuffleVector(i1Mask, undef, shuffleMask, "spmd.widen.shuf")

	// Restore original element width if the input was not i1 (e.g., i32 WASM mask).
	if srcElem == i1Type {
		return widened
	}
	targetType := llvm.VectorType(srcElem, targetLanes)
	return b.CreateSExt(widened, targetType, "spmd.widen.sext")
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
	// Scalar fallback: plain select on scalar bool mask.
	if !b.simdEnabled {
		mask := b.getValue(instr.Mask, token.NoPos)
		x := b.getValue(instr.X, token.NoPos)
		y := b.getValue(instr.Y, token.NoPos)
		// Mask is scalar i32 (0 or -1) or i1. Truncate to i1 for select.
		if mask.Type() != b.ctx.Int1Type() {
			mask = b.CreateICmp(llvm.IntNE, mask, llvm.ConstNull(mask.Type()), "")
		}
		return b.CreateSelect(mask, x, y, "")
	}

	mask := b.getValue(instr.Mask, token.NoPos)
	x := b.getValue(instr.X, token.NoPos)
	y := b.getValue(instr.Y, token.NoPos)

	// Determine if the result type is boolean/mask. For such types, operands
	// use the all-ones/all-zeros pattern and sign-extension must be used when
	// aligning element widths (e.g., i8 0xFF → i32 0xFFFFFFFF, not 0x000000FF).
	isBoolOrMask := false
	if spmdType, ok := instr.Type().(*types.SPMDType); ok && spmdType.IsVarying() {
		elem := spmdType.Elem().Underlying()
		if basic, ok := elem.(*types.Basic); ok && basic.Info()&types.IsBoolean != 0 {
			isBoolOrMask = true
		}
		if spmdtypes.IsMask(spmdType.Elem()) {
			isBoolOrMask = true
		}
	}

	// Broadcast scalar operands to vector when needed (e.g., uniform constants).
	x, y = b.spmdBroadcastMatch(x, y, isBoolOrMask)

	// When both operands are scalar but the mask is a vector (produced by the
	// store-merge optimisation where uniform values like clamp bounds are
	// selected per-lane), splat both to a vector of the mask's lane count.
	// spmdBroadcastMatch only handles the one-scalar/one-vector case; we must
	// handle the all-scalar case separately.
	if x.Type().TypeKind() != llvm.VectorTypeKind &&
		y.Type().TypeKind() != llvm.VectorTypeKind &&
		mask.Type().TypeKind() == llvm.VectorTypeKind {
		laneCount := mask.Type().VectorSize()
		vecType := llvm.VectorType(x.Type(), laneCount)
		x = b.splatScalar(x, vecType)
		y = b.splatScalar(y, vecType)
	}

	// Handle lane count mismatch between mask and operands. This occurs when a
	// Varying[bool] accumulator (16 lanes on WASM128) is used inside a 4-lane
	// int32 go for loop — the loop mask has 4 lanes but the bool operands have 16.
	//
	// When operands have more lanes than the mask (widening), replicate each mask
	// bit to cover the corresponding operand lanes using shufflevector. For example,
	// a <4 x i32> mask for a 16-lane bool select becomes <16 x i1> with each bit
	// replicated 4 times: [0,0,0,0, 1,1,1,1, 2,2,2,2, 3,3,3,3].
	//
	// When operands have fewer lanes than the mask, narrow the mask to match.
	// This can happen when getLLVMType(Varying[mask]) computes a different lane
	// count than the data type (e.g., <32 x i1> for Varying[mask] on AVX2 vs
	// <8 x i32> for float32 data). The comparison result has the correct lane
	// count; we narrow the over-counted mask to match.
	if x.Type().TypeKind() == llvm.VectorTypeKind {
		maskLanes := mask.Type().VectorSize()
		operandLanes := x.Type().VectorSize()
		if maskLanes != operandLanes {
			if maskLanes > operandLanes {
				// Narrow: mask has more lanes than operands. Convert to the correct
				// operand-appropriate mask type (e.g., <8 x i32> for 8 float32 lanes).
				maskElem := b.spmdMaskElemType(operandLanes)
				targetMaskType := llvm.VectorType(maskElem, operandLanes)
				mask = b.spmdConvertMaskFormat(mask, targetMaskType)
			} else {
				// Widen: replicate each mask lane (operandLanes/maskLanes) times.
				mask = b.spmdWidenMaskToOperandLanes(mask, operandLanes)
			}
		}
	}

	return b.spmdMaskSelect(mask, x, y)
}

// spmdIsVectorizableElemType returns true if the LLVM type can be used as a
// vector element type. LLVM vectors only support integer, floating-point, and
// pointer element types — NOT structs or arrays. When this returns false,
// createSPMDStore and createSPMDLoad must use per-lane scalar fallbacks instead
// of LLVM vector intrinsics.
func spmdIsVectorizableElemType(t llvm.Type) bool {
	switch t.TypeKind() {
	case llvm.IntegerTypeKind, llvm.FloatTypeKind, llvm.DoubleTypeKind, llvm.PointerTypeKind:
		return true
	default:
		return false
	}
}

// spmdConditionalStore stores a scalar value to a scalar address only if any
// lane in the mask is active. Used for non-vectorizable types (structs,
// interfaces, slice headers) at scalar (uniform) addresses where all active
// lanes share the same destination.
func (b *builder) spmdConditionalStore(val, ptr, mask llvm.Value) {
	laneCount := mask.Type().VectorSize()
	// Unwrap to <N x i1> so spmdVectorAnyTrue can handle both WASM and non-WASM.
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	anyActive := b.spmdVectorAnyTrue(i1Mask)

	storeBB := b.insertBasicBlock("spmd.struct.store")
	mergeBB := b.insertBasicBlock("spmd.struct.done")
	b.CreateCondBr(anyActive, storeBB, mergeBB)
	b.SetInsertPointAtEnd(storeBB)
	b.CreateStore(val, ptr)
	b.CreateBr(mergeBB)
	b.SetInsertPointAtEnd(mergeBB)
}

// spmdPerLaneScatterStore stores a scalar value to varying addresses (a vector
// of pointers), one per lane, conditional on the mask. Used for non-vectorizable
// element types (structs, interfaces) with varying pointer operands.
func (b *builder) spmdPerLaneScatterStore(val, ptrs, mask llvm.Value, laneCount int) {
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	for lane := 0; lane < laneCount; lane++ {
		laneIdx := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
		maskBit := b.CreateExtractElement(i1Mask, laneIdx, "spmd.lane.mask")
		lanePtr := b.CreateExtractElement(ptrs, laneIdx, "spmd.lane.ptr")

		storeBB := b.insertBasicBlock("spmd.struct.scatter")
		mergeBB := b.insertBasicBlock("spmd.struct.scatter.done")
		b.CreateCondBr(maskBit, storeBB, mergeBB)
		b.SetInsertPointAtEnd(storeBB)
		b.CreateStore(val, lanePtr)
		b.CreateBr(mergeBB)
		b.SetInsertPointAtEnd(mergeBB)
	}
}

// spmdPerLaneGather loads a scalar value from each of the varying addresses
// (a vector of pointers), conditional on the mask. Returns an array of loaded
// values as an LLVM array type [N x T]. Used for non-vectorizable element types
// (structs, interfaces) with varying pointer operands.
func (b *builder) spmdPerLaneGather(elemType llvm.Type, ptrs, mask llvm.Value, laneCount int) llvm.Value {
	i1Mask := b.spmdUnwrapMaskForIntrinsic(mask, laneCount)
	arrType := llvm.ArrayType(elemType, laneCount)
	result := llvm.Undef(arrType)
	for lane := 0; lane < laneCount; lane++ {
		laneIdx := llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false)
		maskBit := b.CreateExtractElement(i1Mask, laneIdx, "spmd.lane.mask")
		lanePtr := b.CreateExtractElement(ptrs, laneIdx, "spmd.lane.ptr")

		// condBB is the block containing the conditional branch.
		// The false edge goes directly from condBB to mergeBB.
		condBB := b.GetInsertBlock()
		loadBB := b.insertBasicBlock("spmd.struct.gather")
		mergeBB := b.insertBasicBlock("spmd.struct.gather.done")
		b.CreateCondBr(maskBit, loadBB, mergeBB)

		b.SetInsertPointAtEnd(loadBB)
		loaded := b.CreateLoad(elemType, lanePtr, "spmd.lane.loaded")
		b.CreateBr(mergeBB)
		loadExitBB := b.GetInsertBlock()

		b.SetInsertPointAtEnd(mergeBB)
		zero := llvm.ConstNull(elemType)
		phi := b.CreatePHI(elemType, "spmd.lane.val")
		// loadExitBB → loaded value; condBB → zero (mask bit was false).
		phi.AddIncoming([]llvm.Value{loaded, zero}, []llvm.BasicBlock{loadExitBB, condBB})

		result = b.CreateInsertValue(result, phi, lane, "")
	}
	return result
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
	// Scalar fallback: plain load, no masking.
	if !b.simdEnabled {
		addr := b.getValue(instr.Addr, instr.Pos())
		resultType := b.getLLVMType(instr.Type())
		return b.CreateLoad(resultType, addr, "")
	}

	addr := b.getValue(instr.Addr, instr.Pos())
	mask := b.getValue(instr.Mask, instr.Pos())

	// Determine the result type from the SSA result type.
	resultType := b.getLLVMType(instr.Type())

	// Derive lane count from the mask vector, which always has the correct
	// target-specific width. instr.Lanes may use host int sizes.
	laneCount := mask.Type().VectorSize()

	// WASM swizzle fast path: IndexAddr detected a byte array ≤ 16 on WASM,
	// loaded it as <16 x i8>, and ran i8x16.swizzle eagerly. Return the
	// cached result directly — no per-lane loads needed.
	if b.spmdSwizzleResult != nil {
		if result, ok := b.spmdSwizzleResult[instr.Addr]; ok {
			return result
		}
	}

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
			_ = ci
			ssaElemType := instr.Addr.Type().Underlying().(*types.Pointer).Elem()
			elemType := b.getLLVMType(ssaElemType)

			// If the element type is already a vector (e.g., Varying[T] as a slice
			// element), do a plain load rather than creating an illegal nested vector
			// <N x <M x T>>. This handles "go for over []Varying[T]" where laneCount=1
			// but each element is a full SIMD vector (<M x T>).
			if elemType.TypeKind() == llvm.VectorTypeKind {
				return b.CreateLoad(elemType, ci.scalarPtr, "spmd.load.result")
			}

			// Contiguous aggregate load: the element is a struct/string/slice
			// that can't form an LLVM vector. Emit N scalar loads at
			// consecutive GEP offsets and pack into [N x elemType].
			if !spmdIsVectorizableElemType(elemType) {
				arrType := llvm.ArrayType(elemType, laneCount)
				result := llvm.Undef(arrType)
				for lane := 0; lane < laneCount; lane++ {
					gep := b.CreateInBoundsGEP(elemType, ci.scalarPtr, []llvm.Value{
						llvm.ConstInt(b.ctx.Int32Type(), uint64(lane), false),
					}, "spmd.agg.gep")
					loaded := b.CreateLoad(elemType, gep, "spmd.agg.lane")
					result = b.CreateInsertValue(result, loaded, lane, "")
				}
				return result
			}

			// Narrow load path (WASM byte/bool elements).
			narrowBits := b.spmdNarrowLoadElemBits(ssaElemType, laneCount)
			if narrowBits > 0 {
				return b.spmdMaskedLoadNarrow(narrowBits, ci.scalarPtr, laneCount, mask)
			}
			// Bool load fix: load as <N x i8>, truncate to <N x i1>.
			isBoolLoad := elemType == b.ctx.Int1Type()
			if isBoolLoad {
				elemType = b.ctx.Int8Type()
			}
			// WASM sub-128-bit widening.
			if b.spmdUsesSIMD() {
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
		// Non-vectorizable types (structs, interfaces) cannot use LLVM gather.
		// Fall back to per-lane conditional scalar loads.
		// Returns [N x T] array (NOT <N x T> vector). Struct values must not
		// flow into vector operations — they are only passed through to stores.
		// Exception: if resultType is already a vector (e.g., Varying[int] → <4 x i32>),
		// it IS vectorizable — fall through to spmdMaskedGather with vecResultType=resultType.
		if !spmdIsVectorizableElemType(resultType) && resultType.TypeKind() != llvm.VectorTypeKind {
			return b.spmdPerLaneGather(resultType, addr, mask, addrLaneCount)
		}
		vecResultType := resultType
		if resultType.TypeKind() != llvm.VectorTypeKind {
			vecResultType = llvm.VectorType(resultType, addrLaneCount)
		} else if resultType.VectorSize() != addrLaneCount {
			// getLLVMType(Varying[T]) returns the register-natural lane count (e.g., <16 x i8>
			// for Varying[byte] on SSE). But the gather must return exactly addrLaneCount
			// elements — one per address pointer. Rebuild with the correct lane count.
			vecResultType = llvm.VectorType(resultType.ElementType(), addrLaneCount)
		}
		return b.spmdMaskedGather(vecResultType, addr, mask)
	}

	// Scalar address: load speculatively, then broadcast and mask.
	// Safety: the predication pass only generates SPMDLoad inside blocks that
	// were statically reachable in the pre-predication CFG, so the pointer was
	// valid for all lanes in that block. This assumes flat (non-nested) if
	// linearization; nested ifs require mask threading to remain safe.
	loaded := b.CreateLoad(resultType, addr, "spmd.load")

	// Non-vectorizable types (structs, interfaces, slice headers) cannot be
	// put in LLVM vector types. Return the scalar value directly — all active
	// lanes share the same scalar address so the single loaded value is correct.
	if !spmdIsVectorizableElemType(resultType) {
		return loaded
	}

	// N=1 serial path: laneCount=1 means serial execution (e.g., Varying[[]int]
	// where the slice element is 12 bytes). Return the scalar value directly —
	// broadcasting to <1 x T> would produce type mismatches with phi nodes in
	// the inner loop, since the phi was typed as T (scalar), not <1 x T>.
	if laneCount == 1 && resultType.TypeKind() != llvm.VectorTypeKind {
		return loaded
	}

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
	// Scalar fallback: plain store, no masking.
	if !b.simdEnabled {
		val := b.getValue(instr.Val, instr.Pos())
		addr := b.getValue(instr.Addr, instr.Pos())
		b.CreateStore(val, addr)
		return
	}

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

	// Derive lane count from the mask vector, which always has the correct
	// target-specific width. instr.Lanes may use host int sizes (e.g., 8 bytes
	// on amd64 host) rather than target sizes (4 bytes on WASM).
	laneCount := mask.Type().VectorSize()

	// Non-vectorizable types (structs, interfaces, slice headers, closures)
	// cannot be LLVM vector elements. LLVM only supports integer, float, and
	// pointer element types in vectors. Use per-lane scalar fallbacks instead
	// of vector intrinsics.
	valElemType := val.Type()
	if valElemType.TypeKind() == llvm.VectorTypeKind {
		valElemType = valElemType.ElementType()
	}
	if !spmdIsVectorizableElemType(valElemType) {
		if addr.Type().TypeKind() == llvm.VectorTypeKind {
			// Varying address: per-lane conditional scatter.
			b.spmdPerLaneScatterStore(val, addr, mask, laneCount)
		} else {
			// Scalar address: store if any lane is active. All active lanes
			// share the same destination, so a single conditional store suffices.
			b.spmdConditionalStore(val, addr, mask)
		}
		return
	}

	// The predicated SSA pass copies Store operands directly.
	// When the original Store was inside a varying if/else, the SSA types
	// may still be scalar (e.g., int32, *int32). SPMDStore needs vector
	// operands, so splat scalar values using the active loop's lane count.
	if val.Type().TypeKind() != llvm.VectorTypeKind {
		// Scalar value: splat to vector.
		val = b.splatScalar(val, llvm.VectorType(val.Type(), laneCount))
	}

	// Contiguous access: use vector store instead of scatter.
	// See createSPMDLoad for contiguity info sources (SSA-level + TinyGo-level).
	if b.spmdContiguousPtr != nil {
		if ci, ok := b.spmdContiguousPtr[instr.Addr]; ok {
			// If the element type is already a full SIMD vector (e.g., Varying[T] as a
			// slice element in "go for over []Varying[T]"), do a plain store.
			// Wrapping in another vector would produce <N x <M x T>> which LLVM rejects.
			// The outer loop's mask (laneCount=1) already controls execution at the
			// block level, so unconditional store to the scalar pointer is correct.
			if addrPtrType, ok := instr.Addr.Type().Underlying().(*types.Pointer); ok {
				if b.getLLVMType(addrPtrType.Elem()).TypeKind() == llvm.VectorTypeKind {
					b.CreateStore(val, ci.scalarPtr)
					return
				}
			}
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
			} else if b.spmdIsConstAllOnesMask(mask) {
				// All lanes active — direct contiguous store, no blend needed.
				elemAlign := int(b.targetData.TypeAllocSize(val.Type().ElementType()))
				st := b.CreateStore(val, ci.scalarPtr)
				st.SetAlignment(elemAlign)
			} else if b.spmdIsAllocaOrigin(ci) || !ci.sliceCap.IsNil() {
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
		// Narrow widened values before scatter: when the value is <N x i32> but the
		// destination pointer type implies a narrower element (e.g., *uint8 → i8), truncate
		// so each lane scatter-writes only the correct number of bytes.
		if val.Type().TypeKind() == llvm.VectorTypeKind {
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
				narrowType := b.ctx.IntType(int(narrowBits))
				val = b.CreateTrunc(val, llvm.VectorType(narrowType, laneCount), "scatter.trunc")
			}
		}
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
// The boxed representation uses struct{[N]T, [N]maskElem} where maskElem is
// i32/i16/i8/i1 depending on lane count and platform. The type-code name is
// derived from boxedGoType (which uses int32) so the canonical name is stable
// across platforms. The LLVM struct type is built directly to match the actual
// mask element width. The SSA result type is <N x T> vector.
func (b *builder) createTypeAssertSPMD(itf llvm.Value, expr *ssa.TypeAssert, spmdType *types.SPMDType, vecType llvm.Type) llvm.Value {
	// Build the struct type that matches the type code used in boxing.
	elemLLVM := b.getLLVMType(spmdType.Elem())
	laneCount := b.spmdEffectiveLaneCount(spmdType, elemLLVM)
	boxedGoType := b.spmdBoxedVaryingGoType(spmdType, laneCount)
	// Build the LLVM struct type directly so the mask array element type matches
	// the platform-specific mask element (i32 for 4-lane, i16 for 8-lane, i8 for
	// 16-lane, i1 for non-WASM). getLLVMType(boxedGoType) always gives [N]i32
	// because spmdBoxedVaryingGoType hardcodes int32 for the mask field.
	maskElemType := b.spmdMaskElemType(laneCount)
	maskArrLLVM := llvm.ArrayType(maskElemType, laneCount)
	valArrLLVM := llvm.ArrayType(elemLLVM, laneCount)
	boxedLLVMType := b.ctx.StructType([]llvm.Type{valArrLLVM, maskArrLLVM}, false)

	// Compare type codes using the struct type (same as boxing path).
	actualTypeNum := b.CreateExtractValue(itf, 0, "interface.type")
	name, _ := getTypeCodeName(boxedGoType)
	globalName := "reflect/types.typeid:" + name
	assertedTypeCodeGlobal := b.mod.NamedGlobal(globalName)
	if assertedTypeCodeGlobal.IsNil() {
		assertedTypeCodeGlobal = llvm.AddGlobal(b.mod, b.ctx.Int8Type(), globalName)
		assertedTypeCodeGlobal.SetGlobalConstant(true)
	}
	commaOk := b.createRuntimeCall("typeAssert", []llvm.Value{actualTypeNum, assertedTypeCodeGlobal}, "typecode")

	// Branch on type match.
	prevBlock := b.GetInsertBlock()
	okBlock := b.insertBasicBlock("typeassert.spmd.ok")
	nextBlock := b.insertBasicBlock("typeassert.spmd.next")
	b.currentBlockInfo.exit = nextBlock
	b.CreateCondBr(commaOk, okBlock, nextBlock)

	// OK block: extract struct, get value array (field 0), convert to vector.
	b.SetInsertPointAtEnd(okBlock)
	boxedStruct := b.extractValueFromInterface(itf, boxedLLVMType)
	valArr := b.CreateExtractValue(boxedStruct, 0, "typeassert.spmd.valarr")
	valueOk := b.arrayToVector(valArr, vecType)
	b.CreateBr(nextBlock)

	// Merge block.
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

// createSPMDExtractMask extracts the embedded lane mask from a boxed
// Varying[T] interface value. The interface holds struct{[N]T, [N]maskElem}
// where maskElem is i32/i16/i8/i1 depending on lane count and platform;
// this extracts field 1 (mask array) and converts to a mask vector.
func (b *builder) createSPMDExtractMask(instr *ssa.SPMDExtractMask) llvm.Value {
	itf := b.getValue(instr.X, token.NoPos)
	laneCount := instr.Lanes

	maskElemType := b.spmdMaskElemType(laneCount)
	maskVecType := llvm.VectorType(maskElemType, laneCount)

	// Build mask array type using the platform-specific element type.
	maskArrType := llvm.ArrayType(maskElemType, laneCount)

	// Find the SPMDType from a sibling TypeAssert on the same interface.
	var boxedLLVMType llvm.Type
	if refs := instr.X.Referrers(); refs != nil {
		for _, ref := range *refs {
			if ta, ok := ref.(*ssa.TypeAssert); ok {
				if spmdType, ok2 := ta.AssertedType.(*types.SPMDType); ok2 {
					// Build the LLVM struct type directly so the mask array element
					// type matches the platform-specific mask element width, not the
					// int32 hardcoded in spmdBoxedVaryingGoType.
					elemLLVM := b.getLLVMType(spmdType.Elem())
					valArrLLVM := llvm.ArrayType(elemLLVM, laneCount)
					boxedLLVMType = b.ctx.StructType([]llvm.Type{valArrLLVM, maskArrType}, false)
					break
				}
			}
		}
	}
	if boxedLLVMType.IsNil() {
		// Fallback: cannot determine struct type. Return all-ones mask.
		return llvm.ConstAllOnes(maskVecType)
	}

	// Extract the struct from the interface, then extract mask array (field 1).
	boxedStruct := b.extractValueFromInterface(itf, boxedLLVMType)
	maskArr := b.CreateExtractValue(boxedStruct, 1, "spmd.extract.maskarr")
	return b.arrayToVector(maskArr, maskVecType)
}

// createSPMDVectorFromMemoryMasked emits the overread+mask sequence that loads
// a 16-byte vector from dataPtr and zeroes bytes at index >= length. This is
// the safe path used on WASM (guard zone guarantees no trap) and as the slow
// path on x86 when the pointer is within 16 bytes of a page boundary.
//
// Sequence: raw load → index const → splat(len) → icmp ult → sext → and.
// The caller guarantees length <= lanes <= 16.
func (b *builder) createSPMDVectorFromMemoryMasked(dataPtr, length llvm.Value, lanes int) llvm.Value {
	i8Type := b.ctx.Int8Type()
	vecType := llvm.VectorType(i8Type, lanes)

	// Load lanes bytes. On WASM the 16-byte guard zone ensures no trap; on
	// x86 the caller already checked the page boundary.
	rawLoad := b.CreateLoad(vecType, dataPtr, "vfm.raw")
	rawLoad.SetAlignment(1)

	// Build lane index constant [0, 1, ..., lanes-1].
	indices := make([]llvm.Value, lanes)
	for i := 0; i < lanes; i++ {
		indices[i] = llvm.ConstInt(i8Type, uint64(i), false)
	}
	indicesVec := llvm.ConstVector(indices, false)

	// Splat length as i8. Values 0-16 always fit without truncation hazard.
	lenI8 := b.CreateTrunc(length, i8Type, "vfm.len.i8")
	lenSplat := b.splatScalar(lenI8, vecType)

	// icmp ult + sext: active lanes get 0xFF, inactive lanes get 0x00.
	// LLVM may combine sext+and into v128.andnot or v128.bitselect.
	mask := b.CreateICmp(llvm.IntULT, indicesVec, lenSplat, "vfm.mask")
	maskExt := b.CreateSExt(mask, vecType, "vfm.mask.ext")

	return b.CreateAnd(rawLoad, maskExt, "vfm.masked")
}

// createSPMDVectorFromMemory lowers an SSA SPMDVectorFromMemory instruction to
// LLVM IR. It loads elements from a string or slice source into a vector,
// zeroing lanes whose index >= len(src).
//
// The source (instr.Ptr) is a string or []byte header, NOT a raw pointer.
// Field 0 is the data pointer, field 1 is the length.
//
// On x86-64 a page-safe fast path emits a single vmovdqu when the pointer is
// not within the last 16 bytes of a 4096-byte page (covers ~99.6% of cases).
// Garbage bytes beyond len(src) are acceptable because callers use scalar
// bitmask trimming, not zero-padding, to discard inactive lanes.
// The WASM path (and the x86 slow path) use createSPMDVectorFromMemoryMasked
// which overreads and ANDs with a lane-index mask.
func (b *builder) createSPMDVectorFromMemory(instr *ssa.SPMDVectorFromMemory) llvm.Value {
	src := b.getValue(instr.Ptr, instr.Pos())
	length := b.getValue(instr.Len, instr.Pos())

	lanes := instr.Lanes
	dataPtr := b.CreateExtractValue(src, 0, "vfm.ptr")

	// x86-64 fast path: emit a single vmovdqu when the pointer cannot straddle
	// a page boundary. Page safety: a 16-byte load starting at ptr is safe when
	// (ptr & 0xFFF) <= 0xFF0. For the ~0.4% of pointers in the last 16 bytes of
	// a page we fall through to the masked slow path.
	if b.spmdIsX86() && lanes == 16 {
		i8Type := b.ctx.Int8Type()
		v16i8 := llvm.VectorType(i8Type, 16)

		// Compute page offset and test whether we are near a page boundary.
		ptrInt := b.CreatePtrToInt(dataPtr, b.uintptrType, "vfm.page.ptr")
		pageOff := b.CreateAnd(ptrInt, llvm.ConstInt(b.uintptrType, 0xFFF, false), "vfm.page.off")
		nearEnd := b.CreateICmp(llvm.IntUGT, pageOff, llvm.ConstInt(b.uintptrType, 0xFF0, false), "vfm.page.near")

		curBlock := b.GetInsertBlock()
		fastBlock := b.ctx.AddBasicBlock(curBlock.Parent(), "vfm.fast")
		slowBlock := b.ctx.AddBasicBlock(curBlock.Parent(), "vfm.slow")
		mergeBlock := b.ctx.AddBasicBlock(curBlock.Parent(), "vfm.merge")

		b.CreateCondBr(nearEnd, slowBlock, fastBlock)

		// Fast path: raw unaligned load, no masking needed.
		// Bytes beyond len(src) contain arbitrary data; callers discard them via
		// the active-lane mask computed from the execution mask, not from the
		// vector content itself.
		b.SetInsertPointAtEnd(fastBlock)
		fastLoad := b.CreateLoad(v16i8, dataPtr, "vfm.raw.fast")
		fastLoad.SetAlignment(1)
		b.CreateBr(mergeBlock)

		// Slow path: overread + mask (identical to the WASM path).
		b.SetInsertPointAtEnd(slowBlock)
		slowResult := b.createSPMDVectorFromMemoryMasked(dataPtr, length, lanes)
		slowBlock = b.GetInsertBlock() // createSPMDVectorFromMemoryMasked may not add blocks, but be safe
		b.CreateBr(mergeBlock)

		// Merge: pick the result from whichever path executed.
		b.SetInsertPointAtEnd(mergeBlock)
		phi := b.CreatePHI(v16i8, "vfm.result")
		phi.AddIncoming([]llvm.Value{fastLoad, slowResult}, []llvm.BasicBlock{fastBlock, slowBlock})
		return phi
	}

	// Non-x86 path (WASM and others): overread + mask, relying on the runtime
	// guard zone or platform memory layout for safety.
	return b.createSPMDVectorFromMemoryMasked(dataPtr, length, lanes)
}

// spmdPmaddSide holds the decomposed components of one operand in a
// stride-2 widen-multiply-add expression: MUL(Convert(load(src[i*2+r])), C)
// or just Convert(load(src[i*2+r])) where C defaults to 1.
type spmdPmaddSide struct {
	load      *ssa.SPMDLoad // the underlying SPMDLoad
	indexAddr *ssa.IndexAddr
	constVal  int64 // the multiply constant (1 if bare Convert)
	remainder int64 // 0 for i*2, 1 for i*2+1
}

// spmdExtractPmaddSide attempts to parse an SSA value as one side of a
// stride-2 widen-multiply-add:
//   - MUL(Convert(SPMDLoad(IndexAddr)), constVal)  → remainder from index
//   - Convert(SPMDLoad(IndexAddr))                 → constVal=1, remainder from index
//
// It validates that the IndexAddr index follows an iter*2+r stride pattern.
// Returns nil if the pattern does not match.
func (b *builder) spmdExtractPmaddSide(v ssa.Value) *spmdPmaddSide {
	var cvt *ssa.Convert
	var constVal int64 = 1

	switch side := v.(type) {
	case *ssa.BinOp:
		// MUL(Convert, Const) or MUL(Const, Convert)
		if side.Op != token.MUL {
			return nil
		}
		if c, ok := side.Y.(*ssa.Const); ok {
			if cv, ok2 := side.X.(*ssa.Convert); ok2 {
				cvt = cv
				if iv, ok3 := ssaConstInt64(c); ok3 && iv > 0 {
					constVal = iv
				} else {
					return nil
				}
			}
		} else if c, ok := side.X.(*ssa.Const); ok {
			if cv, ok2 := side.Y.(*ssa.Convert); ok2 {
				cvt = cv
				if iv, ok3 := ssaConstInt64(c); ok3 && iv > 0 {
					constVal = iv
				} else {
					return nil
				}
			}
		}
		if cvt == nil {
			return nil
		}
	case *ssa.Convert:
		cvt = side
		constVal = 1
	default:
		return nil
	}

	// The Convert must load from an SPMDLoad.
	load, ok := cvt.X.(*ssa.SPMDLoad)
	if !ok {
		return nil
	}

	// The load address must be an IndexAddr.
	ia, ok := load.Addr.(*ssa.IndexAddr)
	if !ok {
		return nil
	}

	// The index must follow a stride-2 pattern (iter*2 or iter*2+1).
	// Try spmdAnalyzeStrideIndex first (works for main body via activeLoops).
	// Fall back to spmdMatchStride2Index for tail body (TailIterPhi not in activeLoops).
	pat := b.spmdAnalyzeStrideIndex(ia.Index)
	if pat != nil && pat.stride == 2 {
		return &spmdPmaddSide{
			load:      load,
			indexAddr: ia,
			constVal:  constVal,
			remainder: pat.remainder,
		}
	}

	// Fallback: directly match iter*2 or iter*2+1 patterns where iter may be
	// in spmdValueOverride (tail body) rather than activeLoops.
	if rem, ok2 := b.spmdMatchStride2Index(ia.Index); ok2 {
		return &spmdPmaddSide{
			load:      load,
			indexAddr: ia,
			constVal:  constVal,
			remainder: rem,
		}
	}

	return nil
}

// spmdMatchStride2Index matches an index expression of the form iter*2 or iter*2+1,
// where iter (after ChangeType unwrapping) is registered in spmdValueOverride.
// Returns (remainder, true) where remainder is 0 for iter*2 and 1 for iter*2+1.
// This handles tail body blocks where TailIterPhi is in spmdValueOverride but
// not in activeLoops.
func (b *builder) spmdMatchStride2Index(index ssa.Value) (int64, bool) {
	if b.spmdValueOverride == nil {
		return 0, false
	}

	// Unwrap ChangeType wrappers.
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

	// isIter returns true if v (after ChangeType peel) is in spmdValueOverride.
	isIter := func(v ssa.Value) bool {
		core := unwrapCT(v)
		_, ok := b.spmdValueOverride[core]
		return ok
	}

	idx := unwrapCT(index)

	// Pattern: iter*2 (remainder 0)
	if mul, ok := idx.(*ssa.BinOp); ok && mul.Op == token.MUL {
		cX, isX := ssaConstInt64(mul.X)
		cY, isY := ssaConstInt64(mul.Y)
		if isX && cX == 2 && isIter(mul.Y) {
			return 0, true
		}
		if isY && cY == 2 && isIter(mul.X) {
			return 0, true
		}
	}

	// Pattern: iter*2+1 (remainder 1)
	if add, ok := idx.(*ssa.BinOp); ok && add.Op == token.ADD {
		// add.X = iter*2, add.Y = 1
		if c, ok2 := ssaConstInt64(add.Y); ok2 && c == 1 {
			if mul, ok3 := unwrapCT(add.X).(*ssa.BinOp); ok3 && mul.Op == token.MUL {
				cX, isX := ssaConstInt64(mul.X)
				cY, isY := ssaConstInt64(mul.Y)
				if isX && cX == 2 && isIter(mul.Y) {
					return 1, true
				}
				if isY && cY == 2 && isIter(mul.X) {
					return 1, true
				}
			}
		}
		// add.Y = iter*2, add.X = 1
		if c, ok2 := ssaConstInt64(add.X); ok2 && c == 1 {
			if mul, ok3 := unwrapCT(add.Y).(*ssa.BinOp); ok3 && mul.Op == token.MUL {
				cX, isX := ssaConstInt64(mul.X)
				cY, isY := ssaConstInt64(mul.Y)
				if isX && cX == 2 && isIter(mul.Y) {
					return 1, true
				}
				if isY && cY == 2 && isIter(mul.X) {
					return 1, true
				}
			}
		}
	}

	return 0, false
}

// spmdTryEmitPmadd attempts to detect and emit a vpmaddubsw (byte→int16) or
// vpmaddwd (int16→int32) instruction for a BinOp ADD in an SPMD body block.
//
// Recognized pattern:
//
//	Convert(src[i*2], wide)*C_even + Convert(src[i*2+1], wide)*C_odd
//
// where src is a []byte (for pmaddubsw) or []int16 (for pmaddwd),
// Convert widens to int16 or int32 respectively, and C_even/C_odd are integer constants.
// One side may omit the MUL (implicit multiply by 1).
//
// On x86 with SSSE3: emits a double-width contiguous load + pmaddubsw/pmaddwd.
// On other targets: returns (zero, false) so normal codegen handles it.
func (b *builder) spmdTryEmitPmadd(addInstr *ssa.BinOp) (llvm.Value, bool) {
	if !b.spmdHasSSSE3() {
		return llvm.Value{}, false
	}

	// Parse both sides of the ADD.
	sideX := b.spmdExtractPmaddSide(addInstr.X)
	sideY := b.spmdExtractPmaddSide(addInstr.Y)
	if sideX == nil || sideY == nil {
		return llvm.Value{}, false
	}

	// One side must be remainder 0 (even index) and the other remainder 1 (odd index).
	var evenSide, oddSide *spmdPmaddSide
	switch {
	case sideX.remainder == 0 && sideY.remainder == 1:
		evenSide, oddSide = sideX, sideY
	case sideX.remainder == 1 && sideY.remainder == 0:
		evenSide, oddSide = sideY, sideX
	default:
		return llvm.Value{}, false
	}

	// Both IndexAddr must reference the same source slice.
	if evenSide.indexAddr.X != oddSide.indexAddr.X {
		return llvm.Value{}, false
	}

	// Determine the pattern kind from the source element type.
	srcElemType := evenSide.load.Addr.Type().Underlying().(*types.Pointer).Elem()
	srcBasic, ok := srcElemType.Underlying().(*types.Basic)
	if !ok {
		return llvm.Value{}, false
	}
	kind := srcBasic.Kind()

	// Classify: byte→int16 (pmaddubsw) or int16→int32 (pmaddwd).
	isPmaddubsw := (kind == types.Uint8 || kind == types.Byte)
	isPmaddwd := (kind == types.Int16)
	if !isPmaddubsw && !isPmaddwd {
		return llvm.Value{}, false
	}

	constEven := evenSide.constVal
	constOdd := oddSide.constVal

	if isPmaddubsw {
		// vpmaddubsw stores weights as signed i8 (-128..127). Constants > 127
		// would be truncated to negative values by llvm.ConstInt(i8, uint64(v)),
		// producing wrong multiply results. Reject them.
		if constEven > 127 || constOdd > 127 {
			return llvm.Value{}, false
		}
		// vpmaddubsw multiplies each unsigned byte by a signed byte weight and
		// adds adjacent pairs into a signed int16, with saturation. Go int16
		// arithmetic wraps on overflow, so the pattern is only safe when the
		// maximum possible adjacent sum cannot saturate: with uint8 source values
		// (max 255), the worst case is 255*constEven + 255*constOdd. If that
		// exceeds 32767 the hardware saturates but Go would wrap — diverging
		// results — so bail out and let normal codegen handle it.
		if 255*(constEven+constOdd) > 32767 {
			return llvm.Value{}, false
		}
	}
	// pmaddwd (int16→int32) multiplies adjacent int16 pairs by int16 weights and
	// adds them into int32 with two's-complement wrapping — identical to Go
	// int32 wrapping semantics — so no saturation guard is needed.

	// Determine the SPMD lane count from the loop that owns the even-side load.
	// The loop is identified via the stride pattern in evenSide. We use the
	// even-side load's SPMDLoad.Lanes field as a fallback.
	laneCount := evenSide.load.Lanes
	if laneCount <= 0 {
		return llvm.Value{}, false
	}
	srcLaneCount := laneCount * 2 // 2N bytes/int16 in the source vector

	// Get the source slice and compute the scalar base pointer.
	// We use the even-index IndexAddr to get the source.
	srcIndexAddr := evenSide.indexAddr
	srcSliceVal := b.getValue(srcIndexAddr.X, getPos(srcIndexAddr))
	if srcSliceVal.IsNil() {
		return llvm.Value{}, false
	}

	var bufptr llvm.Value
	var buflen llvm.Value
	var srcElemLLVM llvm.Type

	switch ptrTyp := srcIndexAddr.X.Type().Underlying().(type) {
	case *types.Slice:
		bufptr = b.CreateExtractValue(srcSliceVal, 0, "pmadd.src.ptr")
		buflen = b.CreateExtractValue(srcSliceVal, 1, "pmadd.src.len")
		srcElemLLVM = b.getLLVMType(ptrTyp.Elem())
	case *types.Pointer:
		arrType, ok2 := ptrTyp.Elem().Underlying().(*types.Array)
		if !ok2 {
			return llvm.Value{}, false
		}
		bufptr = srcSliceVal
		buflen = llvm.ConstInt(b.uintptrType, uint64(arrType.Len()), false)
		srcElemLLVM = b.getLLVMType(arrType.Elem())
	default:
		return llvm.Value{}, false
	}

	// Get the scalar base index (iter*2) from the even-index IndexAddr.
	// Same strategy as interleaved stores: try spmdDecomposed first, then
	// extract lane 0 from the varying index vector.
	scalarBase := llvm.Value{}
	if b.spmdDecomposed != nil {
		if decomp, ok2 := b.spmdDecomposed[srcIndexAddr.Index]; ok2 {
			scalarBase = decomp.scalarBase
		}
	}
	if scalarBase.IsNil() {
		idxVec := b.getValue(srcIndexAddr.Index, getPos(srcIndexAddr))
		if idxVec.Type().TypeKind() == llvm.VectorTypeKind {
			scalarBase = b.CreateExtractElement(idxVec,
				llvm.ConstInt(b.ctx.Int32Type(), 0, false), "pmadd.base.elem0")
		} else {
			scalarBase = idxVec
		}
	}
	scalarBase = b.extendInteger(scalarBase, srcIndexAddr.Index.Type(), b.uintptrType)

	// Bounds check: scalarBase + srcLaneCount <= buflen.
	if !b.info.nobounds && !buflen.IsNil() {
		endIdx := b.CreateAdd(scalarBase,
			llvm.ConstInt(b.uintptrType, uint64(srcLaneCount), false),
			"pmadd.end")
		oob := b.CreateICmp(llvm.IntUGT, endIdx, buflen, "pmadd.oob")
		b.createRuntimeAssert(oob, "lookup", "lookupPanic")
	}

	// Compute pointer to src[scalarBase].
	srcPtr := b.CreateInBoundsGEP(srcElemLLVM, bufptr, []llvm.Value{scalarBase}, "pmadd.srcptr")

	// Dispatch based on element type.
	var result llvm.Value
	if isPmaddubsw {
		result = b.spmdEmitPmaddubsw(srcPtr, srcLaneCount, laneCount, constEven, constOdd)
	} else {
		result = b.spmdEmitPmaddwd(srcPtr, srcLaneCount, laneCount, constEven, constOdd)
	}
	if result.IsNil() {
		return llvm.Value{}, false
	}

	// Apply execution mask to zero out inactive lanes.
	// Get the mask from the even-side SPMDLoad (both loads have the same mask).
	maskVal := b.getValue(evenSide.load.Mask, getPos(evenSide.load))
	if !b.spmdIsConstAllOnesMask(maskVal) {
		// Normalize mask to <N x i1> for select.
		maskI1 := b.CreateTrunc(maskVal, llvm.VectorType(b.ctx.Int1Type(), laneCount), "pmadd.mask.i1")
		zero := llvm.ConstNull(result.Type())
		result = b.CreateSelect(maskI1, result, zero, "pmadd.masked")
	}

	return result, true
}

// spmdEmitPmaddubsw emits the byte→int16 pmaddubsw pattern:
// load <srcLaneCount x i8> from srcPtr, build interleaved weight vector
// [constEven, constOdd, ...], emit pmaddubsw → <laneCount x i16>.
func (b *builder) spmdEmitPmaddubsw(srcPtr llvm.Value, srcLaneCount, laneCount int, constEven, constOdd int64) llvm.Value {
	i8Type := b.ctx.Int8Type()

	// Load srcLaneCount bytes contiguously from srcPtr.
	srcVecType := llvm.VectorType(i8Type, srcLaneCount)
	rawLoad := b.CreateLoad(srcVecType, srcPtr, "pmadd.src.vec")
	rawLoad.SetAlignment(1)

	// Build constant weight vector [constEven, constOdd, constEven, constOdd, ...].
	constElems := make([]llvm.Value, srcLaneCount)
	for j := 0; j < srcLaneCount; j += 2 {
		constElems[j] = llvm.ConstInt(i8Type, uint64(constEven), false)
		constElems[j+1] = llvm.ConstInt(i8Type, uint64(constOdd), false)
	}
	constVec := llvm.ConstVector(constElems, false)

	return b.spmdX86Pmaddubsw(rawLoad, constVec)
}

// spmdEmitPmaddwd emits the int16→int32 pmaddwd pattern:
// load <srcLaneCount x i16> from srcPtr, build interleaved weight vector
// [constEven, constOdd, ...], emit pmaddwd → <laneCount x i32>.
func (b *builder) spmdEmitPmaddwd(srcPtr llvm.Value, srcLaneCount, laneCount int, constEven, constOdd int64) llvm.Value {
	i16Type := b.ctx.Int16Type()

	// Load srcLaneCount int16s contiguously from srcPtr.
	srcVecType := llvm.VectorType(i16Type, srcLaneCount)
	rawLoad := b.CreateLoad(srcVecType, srcPtr, "pmadd.src.vec")
	rawLoad.SetAlignment(1)

	// Build constant weight vector [constEven, constOdd, constEven, constOdd, ...].
	constElems := make([]llvm.Value, srcLaneCount)
	for j := 0; j < srcLaneCount; j += 2 {
		constElems[j] = llvm.ConstInt(i16Type, uint64(constEven), false)
		constElems[j+1] = llvm.ConstInt(i16Type, uint64(constOdd), false)
	}
	constVec := llvm.ConstVector(constElems, false)

	return b.spmdX86Pmaddwd(rawLoad, constVec)
}
