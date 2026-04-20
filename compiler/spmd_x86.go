package compiler

import "tinygo.org/x/go-llvm"

// spmdX86Pshufb emits a byte-permute intrinsic appropriate for the vector width.
// For <16 x i8> (128-bit): llvm.x86.ssse3.pshuf.b.128 (SSSE3 pshufb).
// For <32 x i8> (256-bit): llvm.x86.avx2.pshuf.b (AVX2 vpshufb ymm).
// Indices with bit 7 set produce 0 (matching i8x16.swizzle semantics).
func (b *builder) spmdX86Pshufb(table, indices llvm.Value) llvm.Value {
	vecType := table.Type()
	laneCount := vecType.VectorSize()

	switch laneCount {
	case 32:
		// AVX2 256-bit: vpshufb ymm — shuffles each 128-bit half independently.
		v32i8 := llvm.VectorType(b.ctx.Int8Type(), 32)
		fnType := llvm.FunctionType(v32i8, []llvm.Type{v32i8, v32i8}, false)
		fn := b.mod.NamedFunction("llvm.x86.avx2.pshuf.b")
		if fn.IsNil() {
			fn = llvm.AddFunction(b.mod, "llvm.x86.avx2.pshuf.b", fnType)
		}
		return b.createCall(fnType, fn, []llvm.Value{table, indices}, "x86.vpshufb")
	default:
		// SSE/SSSE3 128-bit: pshufb xmm.
		v16i8 := llvm.VectorType(b.ctx.Int8Type(), 16)
		fnType := llvm.FunctionType(v16i8, []llvm.Type{v16i8, v16i8}, false)
		fn := b.mod.NamedFunction("llvm.x86.ssse3.pshuf.b.128")
		if fn.IsNil() {
			fn = llvm.AddFunction(b.mod, "llvm.x86.ssse3.pshuf.b.128", fnType)
		}
		return b.createCall(fnType, fn, []llvm.Value{table, indices}, "x86.pshufb")
	}
}

// spmdX86Pmovmskb extracts the MSB of each byte lane into a scalar i32 bitmask.
// For <16 x i8> (128-bit) emits llvm.x86.sse2.pmovmskb.128 (SSE2); low 16 bits
// significant. For <32 x i8> (256-bit) emits llvm.x86.avx2.pmovmskb (AVX2); all
// 32 bits significant. Input vector must already be the correct byte-element width.
func (b *builder) spmdX86Pmovmskb(vec llvm.Value) llvm.Value {
	i32Type := b.ctx.Int32Type()
	vecType := vec.Type()
	fnType := llvm.FunctionType(i32Type, []llvm.Type{vecType}, false)
	var intrinsicName string
	switch vecType.VectorSize() {
	case 32:
		intrinsicName = "llvm.x86.avx2.pmovmskb"
	default: // 16 bytes (SSE2/128-bit)
		intrinsicName = "llvm.x86.sse2.pmovmskb.128"
	}
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{vec}, "x86.pmovmskb")
}

// spmdX86Movmskps extracts the sign bit of each float/double lane into a scalar
// i32 bitmask. Dispatches based on element bit-width and lane count:
//
//   - <4 x i32>  → bitcast to <4 x float>  → llvm.x86.sse.movmsk.ps       (SSE)
//   - <8 x i32>  → bitcast to <8 x float>  → llvm.x86.avx.movmsk.ps.256   (AVX)
//   - <2 x i64>  → bitcast to <2 x double> → llvm.x86.sse2.movmsk.pd      (SSE2)
//   - <4 x i64>  → bitcast to <4 x double> → llvm.x86.avx.movmsk.pd.256   (AVX)
//
// The input must be a mask vector in all-ones-or-all-zeros lane format. Because
// 0xFFFFFFFF and 0xFFFFFFFFFFFFFFFF both have their MSB set, the sign bit of each
// reinterpreted float/double lane equals the lane's truth value, so movmskps/pd
// is a correct 1-instruction bitmask extraction without any comparison.
func (b *builder) spmdX86Movmskps(vec llvm.Value) llvm.Value {
	i32Type := b.ctx.Int32Type()
	vecType := vec.Type()
	elemType := vecType.ElementType()
	laneCount := vecType.VectorSize()

	var intrinsicName string
	var floatVecType llvm.Type

	switch {
	case elemType == b.ctx.Int32Type() && laneCount == 8:
		// AVX 256-bit: vmovmskps ymm — 8 float lanes → 8-bit mask in i32.
		floatVecType = llvm.VectorType(b.ctx.FloatType(), 8)
		intrinsicName = "llvm.x86.avx.movmsk.ps.256"
	case elemType == b.ctx.Int32Type():
		// SSE 128-bit: movmskps xmm — 4 float lanes → 4-bit mask in i32.
		floatVecType = llvm.VectorType(b.ctx.FloatType(), 4)
		intrinsicName = "llvm.x86.sse.movmsk.ps"
	case elemType == b.ctx.Int64Type() && laneCount == 4:
		// AVX 256-bit: vmovmskpd ymm — 4 double lanes → 4-bit mask in i32.
		floatVecType = llvm.VectorType(b.ctx.DoubleType(), 4)
		intrinsicName = "llvm.x86.avx.movmsk.pd.256"
	default:
		// SSE2 128-bit: movmskpd xmm — 2 double lanes → 2-bit mask in i32.
		floatVecType = llvm.VectorType(b.ctx.DoubleType(), 2)
		intrinsicName = "llvm.x86.sse2.movmsk.pd"
	}

	floatVec := b.CreateBitCast(vec, floatVecType, "movmsk.cast")
	fnType := llvm.FunctionType(i32Type, []llvm.Type{floatVecType}, false)
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, intrinsicName, fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{floatVec}, "x86.movmskps")
}

// spmdX86Pmaddubsw emits pmaddubsw: multiply unsigned×signed bytes, horizontal
// add adjacent pairs → i16. Dispatches to SSSE3 (128-bit) or AVX2 (256-bit).
// Input: <N x i8> × <N x i8> where N=16 (SSE) or N=32 (AVX2)
// Output: <N/2 x i16>
func (b *builder) spmdX86Pmaddubsw(a, bVec llvm.Value) llvm.Value {
	laneCount := a.Type().VectorSize()
	switch laneCount {
	case 32:
		// AVX2 256-bit: vpmaddubsw ymm — 32 bytes → 16 int16s.
		v16i16 := llvm.VectorType(b.ctx.Int16Type(), 16)
		v32i8 := llvm.VectorType(b.ctx.Int8Type(), 32)
		fnType := llvm.FunctionType(v16i16, []llvm.Type{v32i8, v32i8}, false)
		fn := b.mod.NamedFunction("llvm.x86.avx2.pmadd.ub.sw")
		if fn.IsNil() {
			fn = llvm.AddFunction(b.mod, "llvm.x86.avx2.pmadd.ub.sw", fnType)
		}
		return b.createCall(fnType, fn, []llvm.Value{a, bVec}, "x86.vpmaddubsw")
	default:
		// SSE/SSSE3 128-bit: pmaddubsw xmm — 16 bytes → 8 int16s.
		v8i16 := llvm.VectorType(b.ctx.Int16Type(), 8)
		v16i8 := llvm.VectorType(b.ctx.Int8Type(), 16)
		fnType := llvm.FunctionType(v8i16, []llvm.Type{v16i8, v16i8}, false)
		fn := b.mod.NamedFunction("llvm.x86.ssse3.pmadd.ub.sw.128")
		if fn.IsNil() {
			fn = llvm.AddFunction(b.mod, "llvm.x86.ssse3.pmadd.ub.sw.128", fnType)
		}
		return b.createCall(fnType, fn, []llvm.Value{a, bVec}, "x86.pmaddubsw")
	}
}

// spmdX86Pmaddwd emits pmaddwd: multiply i16 pairs, horizontal add → i32.
// Dispatches to SSE2 (128-bit) or AVX2 (256-bit).
// Input: <N x i16> × <N x i16> where N=8 (SSE) or N=16 (AVX2)
// Output: <N/2 x i32>
func (b *builder) spmdX86Pmaddwd(a, bVec llvm.Value) llvm.Value {
	laneCount := a.Type().VectorSize()
	switch laneCount {
	case 16:
		// AVX2 256-bit: vpmaddwd ymm — 16 int16s → 8 int32s.
		v8i32 := llvm.VectorType(b.ctx.Int32Type(), 8)
		v16i16 := llvm.VectorType(b.ctx.Int16Type(), 16)
		fnType := llvm.FunctionType(v8i32, []llvm.Type{v16i16, v16i16}, false)
		fn := b.mod.NamedFunction("llvm.x86.avx2.pmadd.wd")
		if fn.IsNil() {
			fn = llvm.AddFunction(b.mod, "llvm.x86.avx2.pmadd.wd", fnType)
		}
		return b.createCall(fnType, fn, []llvm.Value{a, bVec}, "x86.vpmaddwd")
	default:
		// SSE2 128-bit: pmaddwd xmm — 8 int16s → 4 int32s.
		// Note: intrinsic name has no .128 suffix, unlike other SSE2 intrinsics.
		v4i32 := llvm.VectorType(b.ctx.Int32Type(), 4)
		v8i16 := llvm.VectorType(b.ctx.Int16Type(), 8)
		fnType := llvm.FunctionType(v4i32, []llvm.Type{v8i16, v8i16}, false)
		fn := b.mod.NamedFunction("llvm.x86.sse2.pmadd.wd")
		if fn.IsNil() {
			fn = llvm.AddFunction(b.mod, "llvm.x86.sse2.pmadd.wd", fnType)
		}
		return b.createCall(fnType, fn, []llvm.Value{a, bVec}, "x86.pmaddwd")
	}
}
