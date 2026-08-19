// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/types"
	"strings"
	"testing"

	"tinygo.org/x/go-llvm"
)

// Regression tests for the mask-vs-data conversion split.
//
// Before the fix, createConvert routed EVERY already-vectorized value through
// spmdConvertMaskFormat, including plain data conversions such as
// `lanes.Varying[float32](i)` where `i` is a varying int derived from the loop
// index. spmdConvertMaskFormat normalizes through <N x i1> and finishes with a
// sext, so a float32 target produced the invalid instruction
//
//	sext <8 x i1> %... to <8 x float>
//
// and the build failed with "SExt only produces an integer". The 4-lane default
// on linux/amd64 (Go `int` is 8 bytes, so the loop index capped lane count at
// SIMDRegisterSize/8) hid this: the 8-wide varying-float32 path was never taken.

// nonConstVector returns a non-constant vector value of the given type, so that
// LLVM does not constant-fold the conversion under test away.
func nonConstVector(b *builder, typ llvm.Type, name string) llvm.Value {
	g := llvm.AddGlobal(b.mod, typ, "src_"+name)
	g.SetInitializer(llvm.ConstNull(typ))
	return b.CreateLoad(typ, g, "in")
}

// spmdConvertVectorElem must emit a real element-wise conversion for data
// vectors, and the result must verify.
func TestSPMDConvertVectorElemDataConversions(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Builder.Dispose()

	i64x8 := llvm.VectorType(c.ctx.Int64Type(), 8)
	f32x8 := llvm.VectorType(c.ctx.FloatType(), 8)
	i32x8 := llvm.VectorType(c.ctx.Int32Type(), 8)

	tests := []struct {
		name     string
		src      llvm.Type
		dst      llvm.Type
		typeFrom types.Type
		typeTo   types.Type
		wantOp   string
	}{
		// The exact case from mandelbrotFlat at 8 lanes: Varying[int] → Varying[float32].
		{"int64_to_float32", i64x8, f32x8, types.Typ[types.Int], types.Typ[types.Float32], "sitofp"},
		{"uint64_to_float32", i64x8, f32x8, types.Typ[types.Uint64], types.Typ[types.Float32], "uitofp"},
		{"float32_to_int32", f32x8, i32x8, types.Typ[types.Float32], types.Typ[types.Int32], "fptosi"},
		{"int64_to_int32", i64x8, i32x8, types.Typ[types.Int], types.Typ[types.Int32], "trunc"},
		{"int32_to_int64", i32x8, i64x8, types.Typ[types.Int32], types.Typ[types.Int], "sext"},
		{"uint32_to_int64", i32x8, i64x8, types.Typ[types.Uint32], types.Typ[types.Int], "zext"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			in := nonConstVector(b, tt.src, tt.name)
			got, err := b.spmdConvertVectorElem(in, tt.dst, tt.typeFrom, tt.typeTo, 0)
			if err != nil {
				t.Fatalf("spmdConvertVectorElem: %v", err)
			}
			if got.Type() != tt.dst {
				t.Fatalf("result type = %s, want %s", got.Type().String(), tt.dst.String())
			}
			if ir := got.String(); !strings.Contains(ir, tt.wantOp) {
				t.Errorf("expected %q in %q", tt.wantOp, ir)
			}
			if strings.Contains(got.String(), "i1") {
				t.Errorf("data conversion went through the i1 mask path: %s", got.String())
			}
		})
	}
}

// spmdConvertMaskFormat must refuse a non-integer target rather than silently
// emitting `sext <N x i1> to <N x float>`, which does not verify.
func TestSPMDConvertMaskFormatRejectsFloatTarget(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Builder.Dispose()

	i32x8 := llvm.VectorType(c.ctx.Int32Type(), 8)
	f32x8 := llvm.VectorType(c.ctx.FloatType(), 8)
	mask := nonConstVector(b, i32x8, "mask")

	defer func() {
		r := recover()
		if r == nil {
			t.Fatal("spmdConvertMaskFormat accepted a float target; it must reject it " +
				"instead of emitting `sext <N x i1> to <N x float>`")
		}
		if msg, ok := r.(string); !ok || !strings.Contains(msg, "non-integer mask target") {
			t.Fatalf("unexpected panic value: %v", r)
		}
	}()
	b.spmdConvertMaskFormat(mask, f32x8)
}

// spmdIsBoolOrMaskType is the predicate that keeps data off the mask path.
func TestSPMDIsBoolOrMaskType(t *testing.T) {
	tests := []struct {
		typ  types.Type
		want bool
	}{
		{types.Typ[types.Bool], true},
		{types.NewVarying(types.Typ[types.Bool]), true},
		{types.Typ[types.Float32], false},
		{types.Typ[types.Int], false},
		{types.NewVarying(types.Typ[types.Float32]), false},
		{types.NewVarying(types.Typ[types.Int32]), false},
	}
	for _, tt := range tests {
		if got := spmdIsBoolOrMaskType(tt.typ); got != tt.want {
			t.Errorf("spmdIsBoolOrMaskType(%s) = %v, want %v", tt.typ, got, tt.want)
		}
	}
}

// Regression test for the SPMD signature/body layout width guard.
//
// builder.getLLVMType and spmdSigTypeFor re-lay-out Varying[T] at the function's
// single lane width. That is only valid for vectorizable element kinds: LLVM has
// no vector-of-struct or vector-of-array type, and compilerContext.getLLVMType
// lays those Varying[T] out as [Lanes x T] instead.
//
// The original guard was `minLC < naturalLC`, which excluded 8-16 byte aggregates
// only by accident (Varying[interface{}] is 16 bytes: naturalLC 2 on AVX2 vs
// minLC 8, so `8 < 2` was false). Widening in both directions with `!=` admits
// them, and without spmdIsVectorizableElemKind it produced <8 x {ptr, ptr}> --
// invalid IR. Verified by hand with a `func(Varying[float32], Varying[pair])`
// signature, which emitted `<8 x %main.pair>` before this guard.
func TestSPMDIsVectorizableElemKind(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	i32 := c.ctx.Int32Type()
	vectorizable := []llvm.Type{
		i32,
		c.ctx.Int8Type(),
		c.ctx.FloatType(),
		c.ctx.DoubleType(),
		c.dataPtrType,
	}
	for _, typ := range vectorizable {
		if !spmdIsVectorizableElemKind(typ) {
			t.Errorf("spmdIsVectorizableElemKind(%s) = false, want true", spmdLLVMTypeString(typ))
		}
	}

	// Aggregates: interface{} / complex128 / string / [2]int64 all lower to one
	// of these shapes.
	aggregates := []llvm.Type{
		c.ctx.StructType([]llvm.Type{c.dataPtrType, c.dataPtrType}, false), // interface{}
		c.ctx.StructType([]llvm.Type{c.ctx.DoubleType(), c.ctx.DoubleType()}, false), // complex128
		llvm.ArrayType(c.ctx.Int64Type(), 2),                               // [2]int64
		llvm.VectorType(i32, 4),                                            // never nest vectors
	}
	for _, typ := range aggregates {
		if spmdIsVectorizableElemKind(typ) {
			t.Errorf("spmdIsVectorizableElemKind(%s) = true, want false: LLVM has no vector of this kind",
				spmdLLVMTypeString(typ))
		}
	}
}
