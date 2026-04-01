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

// spmdX86Pmaddubsw emits llvm.x86.ssse3.pmadd.ub.sw.128.
// Multiplies u8×i8 pairs with saturation, horizontally adds adjacent products → i16.
// Input: a <16 x i8> (unsigned), b <16 x i8> (signed). Output: <8 x i16>.
func (b *builder) spmdX86Pmaddubsw(a, bVec llvm.Value) llvm.Value {
	v8i16 := llvm.VectorType(b.ctx.Int16Type(), 8)
	v16i8 := llvm.VectorType(b.ctx.Int8Type(), 16)
	fnType := llvm.FunctionType(v8i16, []llvm.Type{v16i8, v16i8}, false)
	fn := b.mod.NamedFunction("llvm.x86.ssse3.pmadd.ub.sw.128")
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, "llvm.x86.ssse3.pmadd.ub.sw.128", fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{a, bVec}, "x86.pmaddubsw")
}

// spmdX86Pmaddwd emits llvm.x86.sse2.pmadd.wd.
// Multiplies i16 pairs, horizontally adds adjacent products → i32.
// Input: a <8 x i16>, b <8 x i16>. Output: <4 x i32>.
// Note: the LLVM intrinsic name is "llvm.x86.sse2.pmadd.wd" (no .128 suffix),
// unlike the 128-bit variants of other SSE2/SSSE3 intrinsics.
func (b *builder) spmdX86Pmaddwd(a, bVec llvm.Value) llvm.Value {
	v4i32 := llvm.VectorType(b.ctx.Int32Type(), 4)
	v8i16 := llvm.VectorType(b.ctx.Int16Type(), 8)
	fnType := llvm.FunctionType(v4i32, []llvm.Type{v8i16, v8i16}, false)
	fn := b.mod.NamedFunction("llvm.x86.sse2.pmadd.wd")
	if fn.IsNil() {
		fn = llvm.AddFunction(b.mod, "llvm.x86.sse2.pmadd.wd", fnType)
	}
	return b.createCall(fnType, fn, []llvm.Value{a, bVec}, "x86.pmaddwd")
}
