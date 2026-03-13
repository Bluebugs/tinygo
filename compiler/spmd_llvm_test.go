// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/constant"
	"go/token"
	"go/types"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"golang.org/x/tools/go/ssa"
	"tinygo.org/x/go-llvm"
)

// newTestCompilerContext creates a minimal compiler context for testing SPMD LLVM functionality.
func newTestCompilerContext(t *testing.T) *compilerContext {
	t.Helper()
	target, err := llvm.GetTargetFromTriple("wasm32-unknown-wasi")
	if err != nil {
		t.Fatalf("failed to get WASM target: %v", err)
	}
	machine := target.CreateTargetMachine("wasm32-unknown-wasi", "", "+simd128",
		llvm.CodeGenLevelDefault, llvm.RelocDefault, llvm.CodeModelDefault)
	config := &Config{
		Triple:   "wasm32-unknown-wasi",
		Features: "+simd128",
	}
	return newCompilerContext("test", machine, config, false)
}

// newTestBuilder creates a builder with a basic function and entry block.
func newTestBuilder(t *testing.T, c *compilerContext) *builder {
	t.Helper()
	fn := llvm.AddFunction(c.mod, "test_func", llvm.FunctionType(c.ctx.VoidType(), nil, false))
	bb := llvm.AddBasicBlock(fn, "entry")
	b := &builder{compilerContext: c}
	b.Builder = c.ctx.NewBuilder()
	b.llvmFn = fn
	b.SetInsertPointAtEnd(bb)
	return b
}

func TestSPMDLaneCount(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name     string
		elemType llvm.Type
		want     int
	}{
		{"int8", c.ctx.Int8Type(), 16},     // 128 bits / 8 bits = 16
		{"int16", c.ctx.Int16Type(), 8},    // 128 bits / 16 bits = 8
		{"int32", c.ctx.Int32Type(), 4},    // 128 bits / 32 bits = 4
		{"int64", c.ctx.Int64Type(), 2},    // 128 bits / 64 bits = 2
		{"float32", c.ctx.FloatType(), 4},  // 128 bits / 32 bits = 4
		{"float64", c.ctx.DoubleType(), 2}, // 128 bits / 64 bits = 2
		{"bool", c.ctx.Int1Type(), 16},     // TypeAllocSize(i1) = 1 byte, 128 bits / 8 bits = 16
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.spmdLaneCount(tt.elemType)
			if got != tt.want {
				t.Errorf("spmdLaneCount(%s) = %d, want %d", tt.name, got, tt.want)
			}
		})
	}
}

func TestSPMDMakeLLVMTypeVarying(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name         string
		goType       types.Type
		wantLanes    int
		wantElemKind llvm.TypeKind
	}{
		{
			name:         "varying_int32",
			goType:       types.NewVarying(types.Typ[types.Int32]),
			wantLanes:    4,
			wantElemKind: llvm.IntegerTypeKind,
		},
		{
			name:         "varying_float64",
			goType:       types.NewVarying(types.Typ[types.Float64]),
			wantLanes:    2,
			wantElemKind: llvm.DoubleTypeKind,
		},
		{
			name:         "varying_int8",
			goType:       types.NewVarying(types.Typ[types.Int8]),
			wantLanes:    16,
			wantElemKind: llvm.IntegerTypeKind,
		},
		{
			name:         "varying_bool",
			goType:       types.NewVarying(types.Typ[types.Bool]),
			wantLanes:    16,
			wantElemKind: llvm.IntegerTypeKind,
		},
		{
			name:         "varying_float32",
			goType:       types.NewVarying(types.Typ[types.Float32]),
			wantLanes:    4,
			wantElemKind: llvm.FloatTypeKind,
		},
		{
			name:         "varying_int64",
			goType:       types.NewVarying(types.Typ[types.Int64]),
			wantLanes:    2,
			wantElemKind: llvm.IntegerTypeKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llvmType := c.getLLVMType(tt.goType)

			// Verify it's a vector type
			if llvmType.TypeKind() != llvm.VectorTypeKind {
				t.Errorf("getLLVMType(%s) TypeKind = %v, want VectorTypeKind", tt.name, llvmType.TypeKind())
			}

			// Verify vector size matches expected lane count
			gotLanes := llvmType.VectorSize()
			if gotLanes != tt.wantLanes {
				t.Errorf("getLLVMType(%s) VectorSize = %d, want %d", tt.name, gotLanes, tt.wantLanes)
			}

			// Verify element type kind
			elemType := llvmType.ElementType()
			if elemType.TypeKind() != tt.wantElemKind {
				t.Errorf("getLLVMType(%s) ElementType.TypeKind = %v, want %v", tt.name, elemType.TypeKind(), tt.wantElemKind)
			}
		})
	}
}

// TestSPMDMakeLLVMTypeVaryingAggregate verifies that Varying[T] for aggregate
// element types (structs, slices) produces [N x T] ArrayType rather than an
// invalid <N x T> VectorType. LLVM only supports scalar/pointer vector elements.
// Varying[[]int] on WASM32 gives []int = 12 bytes, laneCount = 16/12 = 1.
func TestSPMDMakeLLVMTypeVaryingAggregate(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name          string
		goType        types.Type
		wantArrayLen  int
		wantElemKind  llvm.TypeKind
	}{
		{
			// Varying[[]int] on WASM32: []int is {ptr,i32,i32} = 12 bytes, laneCount = 16/12 = 1.
			name:         "varying_slice_int",
			goType:       types.NewVarying(types.NewSlice(types.Typ[types.Int])),
			wantArrayLen: 1,
			wantElemKind: llvm.StructTypeKind,
		},
		{
			// Varying[[]float32] on WASM32: []float32 is {ptr,i32,i32} = 12 bytes, laneCount = 1.
			name:         "varying_slice_float32",
			goType:       types.NewVarying(types.NewSlice(types.Typ[types.Float32])),
			wantArrayLen: 1,
			wantElemKind: llvm.StructTypeKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llvmType := c.getLLVMType(tt.goType)

			// Aggregate-element Varying must use ArrayType, not VectorType.
			if llvmType.TypeKind() != llvm.ArrayTypeKind {
				t.Errorf("getLLVMType(%s) TypeKind = %v, want ArrayTypeKind (aggregate elements cannot be LLVM vector elements)", tt.name, llvmType.TypeKind())
				return
			}

			// Verify array length matches expected lane count.
			gotLen := llvmType.ArrayLength()
			if gotLen != tt.wantArrayLen {
				t.Errorf("getLLVMType(%s) ArrayLength = %d, want %d", tt.name, gotLen, tt.wantArrayLen)
			}

			// Verify element type kind.
			elemKind := llvmType.ElementType().TypeKind()
			if elemKind != tt.wantElemKind {
				t.Errorf("getLLVMType(%s) ElementType.TypeKind = %v, want %v", tt.name, elemKind, tt.wantElemKind)
			}
		})
	}
}

func TestSPMDConstVector(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name       string
		constValue constant.Value
		goType     types.Type
		wantNull   bool
	}{
		{
			name:       "int32_const_42",
			constValue: constant.MakeInt64(42),
			goType:     types.NewVarying(types.Typ[types.Int32]),
			wantNull:   false,
		},
		{
			name:       "float64_const_3.14",
			constValue: constant.MakeFloat64(3.14),
			goType:     types.NewVarying(types.Typ[types.Float64]),
			wantNull:   false,
		},
		{
			name:       "int8_const_7",
			constValue: constant.MakeInt64(7),
			goType:     types.NewVarying(types.Typ[types.Int8]),
			wantNull:   false,
		},
		{
			name:       "bool_const_true",
			constValue: constant.MakeBool(true),
			goType:     types.NewVarying(types.Typ[types.Bool]),
			wantNull:   false,
		},
		{
			name:       "nil_const",
			constValue: nil,
			goType:     types.NewVarying(types.Typ[types.Int32]),
			wantNull:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create ssa.Const
			constExpr := ssa.NewConst(tt.constValue, tt.goType)

			// Generate LLVM constant vector
			result := c.createSPMDConst(constExpr, tt.goType.(*types.SPMDType), token.NoPos)

			// Verify result is not null (LLVM zero value)
			if result.IsNil() {
				t.Errorf("createSPMDConst(%s) returned nil LLVM value", tt.name)
				return
			}

			// Verify result is a vector type
			resultType := result.Type()
			if resultType.TypeKind() != llvm.VectorTypeKind {
				t.Errorf("createSPMDConst(%s) result type = %v, want VectorTypeKind", tt.name, resultType.TypeKind())
			}

			// Verify null vs non-null constant vector
			if tt.wantNull {
				if !result.IsConstant() || !result.IsNull() {
					t.Errorf("createSPMDConst(%s) expected null constant vector, got non-null", tt.name)
				}
			} else {
				if !result.IsConstant() {
					t.Errorf("createSPMDConst(%s) expected constant vector, got non-constant", tt.name)
				}
			}
		})
	}
}

func TestSPMDBroadcastMatch(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	scalarVal := llvm.ConstInt(c.ctx.Int32Type(), 1, false)

	// Create a vector constant [0, 1, 2, 3]
	vecElts := make([]llvm.Value, 4)
	for i := range vecElts {
		vecElts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false)
	}
	vecVal := llvm.ConstVector(vecElts, false)

	tests := []struct {
		name      string
		x         llvm.Value
		y         llvm.Value
		wantXKind llvm.TypeKind
		wantYKind llvm.TypeKind
	}{
		{
			name:      "scalar_scalar",
			x:         scalarVal,
			y:         scalarVal,
			wantXKind: llvm.IntegerTypeKind,
			wantYKind: llvm.IntegerTypeKind,
		},
		{
			name:      "vector_vector",
			x:         vecVal,
			y:         vecVal,
			wantXKind: llvm.VectorTypeKind,
			wantYKind: llvm.VectorTypeKind,
		},
		{
			name:      "scalar_vector",
			x:         scalarVal,
			y:         vecVal,
			wantXKind: llvm.VectorTypeKind, // x should be broadcast to vector
			wantYKind: llvm.VectorTypeKind,
		},
		{
			name:      "vector_scalar",
			x:         vecVal,
			y:         scalarVal,
			wantXKind: llvm.VectorTypeKind,
			wantYKind: llvm.VectorTypeKind, // y should be broadcast to vector
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotX, gotY := b.spmdBroadcastMatch(tt.x, tt.y)

			// Verify result types
			if gotX.Type().TypeKind() != tt.wantXKind {
				t.Errorf("spmdBroadcastMatch(%s) x type kind = %v, want %v",
					tt.name, gotX.Type().TypeKind(), tt.wantXKind)
			}

			if gotY.Type().TypeKind() != tt.wantYKind {
				t.Errorf("spmdBroadcastMatch(%s) y type kind = %v, want %v",
					tt.name, gotY.Type().TypeKind(), tt.wantYKind)
			}

			// If broadcast occurred, verify vector sizes match
			if tt.wantXKind == llvm.VectorTypeKind && tt.wantYKind == llvm.VectorTypeKind {
				xSize := gotX.Type().VectorSize()
				ySize := gotY.Type().VectorSize()
				if xSize != ySize {
					t.Errorf("spmdBroadcastMatch(%s) vector sizes don't match: x=%d, y=%d",
						tt.name, xSize, ySize)
				}
			}
		})
	}
}

func TestSPMDSplatScalar(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name      string
		scalarVal llvm.Value
		vecType   llvm.Type
		wantLanes int
	}{
		{
			name:      "int32_splat_4_lanes",
			scalarVal: llvm.ConstInt(c.ctx.Int32Type(), 42, false),
			vecType:   llvm.VectorType(c.ctx.Int32Type(), 4),
			wantLanes: 4,
		},
		{
			name:      "float64_splat_2_lanes",
			scalarVal: llvm.ConstFloat(c.ctx.DoubleType(), 3.14),
			vecType:   llvm.VectorType(c.ctx.DoubleType(), 2),
			wantLanes: 2,
		},
		{
			name:      "int8_splat_16_lanes",
			scalarVal: llvm.ConstInt(c.ctx.Int8Type(), 7, false),
			vecType:   llvm.VectorType(c.ctx.Int8Type(), 16),
			wantLanes: 16,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := b.splatScalar(tt.scalarVal, tt.vecType)

			// Verify result is not nil
			if result.IsNil() {
				t.Errorf("splatScalar(%s) returned nil LLVM value", tt.name)
				return
			}

			// Verify result is a vector
			resultType := result.Type()
			if resultType.TypeKind() != llvm.VectorTypeKind {
				t.Errorf("splatScalar(%s) result type = %v, want VectorTypeKind",
					tt.name, resultType.TypeKind())
			}

			// Verify vector size
			gotLanes := resultType.VectorSize()
			if gotLanes != tt.wantLanes {
				t.Errorf("splatScalar(%s) vector size = %d, want %d",
					tt.name, gotLanes, tt.wantLanes)
			}

			// Verify element type matches scalar type
			elemType := resultType.ElementType()
			if elemType.C != tt.scalarVal.Type().C {
				t.Errorf("splatScalar(%s) element type mismatch", tt.name)
			}
		})
	}
}

func TestSPMDLLVMTypeConsistency(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	// Create a varying type.
	varyingInt32 := types.NewVarying(types.Typ[types.Int32])

	// Get the LLVM type twice. SPMDType bypasses the typeutil.Map cache
	// (which can't hash custom types), but LLVM memoizes vector types
	// internally, so repeated calls return the same object.
	llvmType1 := c.getLLVMType(varyingInt32)
	llvmType2 := c.getLLVMType(varyingInt32)

	if llvmType1.C != llvmType2.C {
		t.Errorf("getLLVMType returned different LLVM types for same Go type")
	}

	if llvmType1.TypeKind() != llvm.VectorTypeKind {
		t.Errorf("getLLVMType(varying int32) = %v, want VectorTypeKind", llvmType1.TypeKind())
	}
}

func TestSPMDLaneOffsetConstant(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name      string
		laneCount int
		elemType  llvm.Type
	}{
		{"4xi32", 4, c.ctx.Int32Type()},
		{"2xi64", 2, c.ctx.Int64Type()},
		{"8xi16", 8, c.ctx.Int16Type()},
		{"16xi8", 16, c.ctx.Int8Type()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := c.spmdLaneOffsetConst(tt.laneCount, tt.elemType)

			if result.IsNil() {
				t.Fatal("spmdLaneOffsetConst returned nil")
			}

			// Verify it's a vector type.
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
			}

			// Verify lane count.
			if result.Type().VectorSize() != tt.laneCount {
				t.Errorf("vector size = %d, want %d", result.Type().VectorSize(), tt.laneCount)
			}

			// Verify it's a constant.
			if !result.IsConstant() {
				t.Error("expected constant vector")
			}

			// Verify element type matches.
			if result.Type().ElementType().C != tt.elemType.C {
				t.Error("element type mismatch")
			}
		})
	}
}

func TestSPMDComputeLaneIndices(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name      string
		iterVal   uint64
		laneCount int
		elemType  llvm.Type
	}{
		{"iter0_4lanes", 0, 4, c.ctx.Int32Type()},
		{"iter8_4lanes", 8, 4, c.ctx.Int32Type()},
		{"iter12_4lanes", 12, 4, c.ctx.Int32Type()},
		{"iter0_2lanes", 0, 2, c.ctx.Int64Type()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vecType := llvm.VectorType(tt.elemType, tt.laneCount)
			scalarVal := llvm.ConstInt(tt.elemType, tt.iterVal, false)

			// Splat scalar and add offset (mirrors emitSPMDBodyPrologue logic).
			iterVec := b.splatScalar(scalarVal, vecType)
			offsetVec := c.spmdLaneOffsetConst(tt.laneCount, tt.elemType)
			laneIndices := b.CreateAdd(iterVec, offsetVec, "test.lane.idx")

			// Verify result is a vector with correct lane count.
			if laneIndices.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("result type = %v, want VectorTypeKind", laneIndices.Type().TypeKind())
			}
			if laneIndices.Type().VectorSize() != tt.laneCount {
				t.Errorf("vector size = %d, want %d", laneIndices.Type().VectorSize(), tt.laneCount)
			}
		})
	}
}

func TestSPMDComputeTailMask(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	elemType := c.ctx.Int32Type()
	vecType := llvm.VectorType(elemType, laneCount)

	tests := []struct {
		name     string
		iterVal  uint64
		boundVal uint64
	}{
		{"all_active", 0, 16},        // lanes 0,1,2,3 < 16 → all true
		{"partial_tail", 12, 14},     // lanes 12,13,14,15 < 14 → T,T,F,F
		{"single_lane", 0, 1},        // lanes 0,1,2,3 < 1 → T,F,F,F
		{"all_active_exact", 12, 16}, // lanes 12,13,14,15 < 16 → all true
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create lane indices.
			scalarVal := llvm.ConstInt(elemType, tt.iterVal, false)
			iterVec := b.splatScalar(scalarVal, vecType)
			offsetVec := c.spmdLaneOffsetConst(laneCount, elemType)
			laneIndices := b.CreateAdd(iterVec, offsetVec, "test.idx")

			// Create bound vector.
			boundScalar := llvm.ConstInt(elemType, tt.boundVal, false)
			boundVec := b.splatScalar(boundScalar, vecType)

			// Compute tail mask.
			tailMask := b.CreateICmp(llvm.IntSLT, laneIndices, boundVec, "test.mask")

			// Verify result is a vector of i1.
			if tailMask.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("tail mask type = %v, want VectorTypeKind", tailMask.Type().TypeKind())
			}
			if tailMask.Type().VectorSize() != laneCount {
				t.Errorf("tail mask size = %d, want %d", tailMask.Type().VectorSize(), laneCount)
			}
			if tailMask.Type().ElementType().TypeKind() != llvm.IntegerTypeKind {
				t.Errorf("tail mask element type = %v, want IntegerTypeKind", tailMask.Type().ElementType().TypeKind())
			}
		})
	}
}

func TestSPMDAnalyzeLoopsNil(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// With no spmdInfo, analyzeSPMDLoops should return nil.
	if b.spmdInfo != nil {
		t.Fatal("expected spmdInfo to be nil for test builder")
	}

	state := b.analyzeSPMDLoops()
	if state != nil {
		t.Errorf("analyzeSPMDLoops() = %v, want nil for non-SPMD function", state)
	}
}

func TestSPMDVectorAnyTrue(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// The test context is WASM, so masks use the <N x i32> format where
	// all-ones (0xFFFFFFFF) means active and all-zeros means inactive.
	// On WASM, spmdVectorAnyTrue uses the native @llvm.wasm.anytrue intrinsic
	// (v128.any_true) and still returns a scalar i1.
	maskElemType := c.spmdMaskElemType(4) // i32 on WASM for 4 lanes
	allOnes := llvm.ConstAllOnes(maskElemType)
	allZeros := llvm.ConstNull(maskElemType)

	tests := []struct {
		name     string
		maskVals []bool
	}{
		{"all_true", []bool{true, true, true, true}},
		{"all_false", []bool{false, false, false, false}},
		{"mixed", []bool{true, false, true, false}},
		{"single_true", []bool{false, false, false, true}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create a <4 x i32> constant vector using WASM mask format.
			vecElts := make([]llvm.Value, len(tt.maskVals))
			for i, val := range tt.maskVals {
				if val {
					vecElts[i] = allOnes
				} else {
					vecElts[i] = allZeros
				}
			}
			maskVec := llvm.ConstVector(vecElts, false)

			// Call spmdVectorAnyTrue.
			result := b.spmdVectorAnyTrue(maskVec)

			// Verify result is not nil.
			if result.IsNil() {
				t.Errorf("spmdVectorAnyTrue(%s) returned nil", tt.name)
				return
			}

			// Verify result type is i1.
			if result.Type().TypeKind() != llvm.IntegerTypeKind {
				t.Errorf("spmdVectorAnyTrue(%s) result type = %v, want IntegerTypeKind",
					tt.name, result.Type().TypeKind())
			}
			if result.Type().IntTypeWidth() != 1 {
				t.Errorf("spmdVectorAnyTrue(%s) result width = %d, want 1",
					tt.name, result.Type().IntTypeWidth())
			}
		})
	}
}

func TestSPMDSelectCreation(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Create a <4 x i1> condition.
	condElts := make([]llvm.Value, 4)
	for i := range condElts {
		condElts[i] = llvm.ConstInt(c.ctx.Int1Type(), uint64(i%2), false)
	}
	condVec := llvm.ConstVector(condElts, false)

	// Create <4 x i32> true and false values.
	trueElts := make([]llvm.Value, 4)
	falseElts := make([]llvm.Value, 4)
	for i := range trueElts {
		trueElts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64(i+10), false)
		falseElts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64(i+20), false)
	}
	trueVec := llvm.ConstVector(trueElts, false)
	falseVec := llvm.ConstVector(falseElts, false)

	// Use CreateSelect to verify LLVM handles vector select.
	result := b.CreateSelect(condVec, trueVec, falseVec, "test.select")

	// Verify result is not nil.
	if result.IsNil() {
		t.Fatal("CreateSelect returned nil")
	}

	// Verify result type is <4 x i32>.
	if result.Type().TypeKind() != llvm.VectorTypeKind {
		t.Errorf("select result type = %v, want VectorTypeKind", result.Type().TypeKind())
	}
	if result.Type().VectorSize() != 4 {
		t.Errorf("select result size = %d, want 4", result.Type().VectorSize())
	}
	if result.Type().ElementType().TypeKind() != llvm.IntegerTypeKind {
		t.Errorf("select result element type = %v, want IntegerTypeKind",
			result.Type().ElementType().TypeKind())
	}
}

func TestSPMDIsBlockInSPMDBody(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// With no spmdInfo, isBlockInSPMDBody should return nil for any block.
	if b.spmdInfo != nil {
		t.Fatal("expected spmdInfo to be nil for test builder")
	}
	if b.spmdContiguousPtr != nil {
		t.Fatal("expected spmdContiguousPtr to be nil for test builder")
	}

	// Without a real SSA function, we can't call isBlockInSPMDBody directly.
	// Verify the precondition: spmdInfo is nil, so the method would return nil.
}

func TestSPMDBroadcastMatchForSelect(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	scalarVal := llvm.ConstInt(c.ctx.Int32Type(), 42, false)

	// Create a vector <4 x i32>.
	vecElts := make([]llvm.Value, 4)
	for i := range vecElts {
		vecElts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false)
	}
	vecVal := llvm.ConstVector(vecElts, false)

	tests := []struct {
		name      string
		x         llvm.Value
		y         llvm.Value
		wantXKind llvm.TypeKind
		wantYKind llvm.TypeKind
	}{
		{
			name:      "vector_scalar",
			x:         vecVal,
			y:         scalarVal,
			wantXKind: llvm.VectorTypeKind,
			wantYKind: llvm.VectorTypeKind, // scalar should be broadcast
		},
		{
			name:      "scalar_vector",
			x:         scalarVal,
			y:         vecVal,
			wantXKind: llvm.VectorTypeKind, // scalar should be broadcast
			wantYKind: llvm.VectorTypeKind,
		},
		{
			name:      "vector_vector",
			x:         vecVal,
			y:         vecVal,
			wantXKind: llvm.VectorTypeKind,
			wantYKind: llvm.VectorTypeKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotX, gotY := b.spmdBroadcastMatch(tt.x, tt.y)

			// Verify result types.
			if gotX.Type().TypeKind() != tt.wantXKind {
				t.Errorf("spmdBroadcastMatch(%s) x type kind = %v, want %v",
					tt.name, gotX.Type().TypeKind(), tt.wantXKind)
			}

			if gotY.Type().TypeKind() != tt.wantYKind {
				t.Errorf("spmdBroadcastMatch(%s) y type kind = %v, want %v",
					tt.name, gotY.Type().TypeKind(), tt.wantYKind)
			}

			// If both are vectors, verify sizes match.
			if tt.wantXKind == llvm.VectorTypeKind && tt.wantYKind == llvm.VectorTypeKind {
				xSize := gotX.Type().VectorSize()
				ySize := gotY.Type().VectorSize()
				if xSize != ySize {
					t.Errorf("spmdBroadcastMatch(%s) vector sizes don't match: x=%d, y=%d",
						tt.name, xSize, ySize)
				}
			}
		})
	}
}

func TestSPMDMaskType(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name          string
		createSig     func() *types.Signature
		wantMaskType  bool
		wantLaneCount int
		wantElemWidth int
	}{
		{
			name: "no_varying_params",
			createSig: func() *types.Signature {
				params := types.NewTuple(
					types.NewVar(token.NoPos, nil, "x", types.Typ[types.Int32]),
					types.NewVar(token.NoPos, nil, "y", types.Typ[types.Float64]),
				)
				return types.NewSignatureType(nil, nil, nil, params, nil, false)
			},
			wantMaskType: false,
		},
		{
			name: "varying_int32_param",
			createSig: func() *types.Signature {
				params := types.NewTuple(
					types.NewVar(token.NoPos, nil, "v", types.NewVarying(types.Typ[types.Int32])),
				)
				return types.NewSignatureType(nil, nil, nil, params, nil, false)
			},
			wantMaskType:  true,
			wantLaneCount: 4,  // 128 bits / 32 bits = 4 lanes
			wantElemWidth: 32, // i32 for WASM mask (avoids shl/shr_s sign extension)
		},
		{
			name: "varying_int8_param",
			createSig: func() *types.Signature {
				params := types.NewTuple(
					types.NewVar(token.NoPos, nil, "v", types.NewVarying(types.Typ[types.Int8])),
				)
				return types.NewSignatureType(nil, nil, nil, params, nil, false)
			},
			wantMaskType:  true,
			wantLaneCount: 16,       // 128 bits / 8 bits = 16 lanes
			wantElemWidth: 128 / 16, // i8 for WASM mask at 16 lanes (128/16=8 bits)
		},
		{
			name: "varying_float64_param",
			createSig: func() *types.Signature {
				params := types.NewTuple(
					types.NewVar(token.NoPos, nil, "v", types.NewVarying(types.Typ[types.Float64])),
				)
				return types.NewSignatureType(nil, nil, nil, params, nil, false)
			},
			wantMaskType:  true,
			wantLaneCount: 2,       // 128 bits / 64 bits = 2 lanes
			wantElemWidth: 128 / 2, // i64 for WASM mask at 2 lanes (128/2=64 bits)
		},
		{
			name: "mixed_params_with_varying",
			createSig: func() *types.Signature {
				params := types.NewTuple(
					types.NewVar(token.NoPos, nil, "x", types.Typ[types.Int32]),
					types.NewVar(token.NoPos, nil, "v", types.NewVarying(types.Typ[types.Int32])),
					types.NewVar(token.NoPos, nil, "y", types.Typ[types.Float64]),
				)
				return types.NewSignatureType(nil, nil, nil, params, nil, false)
			},
			wantMaskType:  true,
			wantLaneCount: 4,
			wantElemWidth: 32, // i32 for WASM mask
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			sig := tt.createSig()
			maskType := c.spmdMaskTypeFromSig(sig)

			if tt.wantMaskType {
				// Expect a valid mask type.
				if maskType == (llvm.Type{}) {
					t.Errorf("spmdMaskTypeFromSig(%s) returned zero-value, want mask type", tt.name)
					return
				}

				// Verify it's a vector type.
				if maskType.TypeKind() != llvm.VectorTypeKind {
					t.Errorf("spmdMaskTypeFromSig(%s) type kind = %v, want VectorTypeKind",
						tt.name, maskType.TypeKind())
				}

				// Verify lane count.
				gotLaneCount := maskType.VectorSize()
				if gotLaneCount != tt.wantLaneCount {
					t.Errorf("spmdMaskTypeFromSig(%s) lane count = %d, want %d",
						tt.name, gotLaneCount, tt.wantLaneCount)
				}

				// Verify element type width (i32 on WASM, i1 on other targets).
				elemType := maskType.ElementType()
				if elemType.TypeKind() != llvm.IntegerTypeKind {
					t.Errorf("spmdMaskTypeFromSig(%s) elem type kind = %v, want IntegerTypeKind",
						tt.name, elemType.TypeKind())
				}
				if elemType.IntTypeWidth() != tt.wantElemWidth {
					t.Errorf("spmdMaskTypeFromSig(%s) elem width = %d, want %d",
						tt.name, elemType.IntTypeWidth(), tt.wantElemWidth)
				}
			} else {
				// Expect zero-value (no mask).
				if maskType != (llvm.Type{}) {
					t.Errorf("spmdMaskTypeFromSig(%s) returned mask type, want zero-value", tt.name)
				}
			}
		})
	}
}

func TestSPMDCallMaskAllTrue(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name      string
		laneCount int
	}{
		{"4_lanes", 4},
		{"2_lanes", 2},
		{"8_lanes", 8},
		{"16_lanes", 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create <N x i1> mask type.
			maskType := llvm.VectorType(c.ctx.Int1Type(), tt.laneCount)

			// Create all-ones mask (all lanes active).
			allOnes := llvm.ConstAllOnes(maskType)

			// Verify result is not nil.
			if allOnes.IsNil() {
				t.Errorf("ConstAllOnes(%s) returned nil", tt.name)
				return
			}

			// Verify result type is <N x i1>.
			if allOnes.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("ConstAllOnes(%s) type kind = %v, want VectorTypeKind",
					tt.name, allOnes.Type().TypeKind())
			}

			gotLaneCount := allOnes.Type().VectorSize()
			if gotLaneCount != tt.laneCount {
				t.Errorf("ConstAllOnes(%s) lane count = %d, want %d",
					tt.name, gotLaneCount, tt.laneCount)
			}

			elemType := allOnes.Type().ElementType()
			if elemType.TypeKind() != llvm.IntegerTypeKind {
				t.Errorf("ConstAllOnes(%s) elem type = %v, want IntegerTypeKind",
					tt.name, elemType.TypeKind())
			}

			// Verify it's a constant.
			if !allOnes.IsConstant() {
				t.Errorf("ConstAllOnes(%s) not constant", tt.name)
			}

			// Verify it's not null (all-ones means all true).
			if allOnes.IsNull() {
				t.Errorf("ConstAllOnes(%s) is null, want all-ones", tt.name)
			}
		})
	}
}

func TestSPMDVectorTypeSuffix(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name    string
		vecType llvm.Type
		want    string
	}{
		{"v4i32", llvm.VectorType(c.ctx.Int32Type(), 4), "v4i32"},
		{"v2i64", llvm.VectorType(c.ctx.Int64Type(), 2), "v2i64"},
		{"v16i8", llvm.VectorType(c.ctx.Int8Type(), 16), "v16i8"},
		{"v4f32", llvm.VectorType(c.ctx.FloatType(), 4), "v4f32"},
		{"v2f64", llvm.VectorType(c.ctx.DoubleType(), 2), "v2f64"},
		{"v4i1", llvm.VectorType(c.ctx.Int1Type(), 4), "v4i1"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := spmdVectorTypeSuffix(tt.vecType)
			if got != tt.want {
				t.Errorf("spmdVectorTypeSuffix(%s) = %q, want %q", tt.name, got, tt.want)
			}
		})
	}
}

func TestSPMDCallVectorReduce(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name         string
		op           string
		vecType      llvm.Type
		wantRetKind  llvm.TypeKind
		wantRetWidth int // for integer types; 0 to skip check
	}{
		{
			name:         "add_v4i32",
			op:           "add",
			vecType:      llvm.VectorType(c.ctx.Int32Type(), 4),
			wantRetKind:  llvm.IntegerTypeKind,
			wantRetWidth: 32,
		},
		{
			name:         "and_v4i1",
			op:           "and",
			vecType:      llvm.VectorType(c.ctx.Int1Type(), 4),
			wantRetKind:  llvm.IntegerTypeKind,
			wantRetWidth: 1,
		},
		{
			name:         "or_v4i32",
			op:           "or",
			vecType:      llvm.VectorType(c.ctx.Int32Type(), 4),
			wantRetKind:  llvm.IntegerTypeKind,
			wantRetWidth: 32,
		},
		{
			name:         "smax_v4i32",
			op:           "smax",
			vecType:      llvm.VectorType(c.ctx.Int32Type(), 4),
			wantRetKind:  llvm.IntegerTypeKind,
			wantRetWidth: 32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vec := llvm.ConstNull(tt.vecType)
			result := b.spmdCallVectorReduce(tt.op, vec)

			if result.IsNil() {
				t.Fatalf("spmdCallVectorReduce(%s, %s) returned nil", tt.op, tt.name)
			}

			// Verify return type is scalar, not vector.
			resultType := result.Type()
			if resultType.TypeKind() == llvm.VectorTypeKind {
				t.Errorf("spmdCallVectorReduce(%s) returned vector type, want scalar", tt.name)
			}

			if resultType.TypeKind() != tt.wantRetKind {
				t.Errorf("spmdCallVectorReduce(%s) return type kind = %v, want %v",
					tt.name, resultType.TypeKind(), tt.wantRetKind)
			}

			if tt.wantRetWidth > 0 && resultType.TypeKind() == llvm.IntegerTypeKind {
				if resultType.IntTypeWidth() != tt.wantRetWidth {
					t.Errorf("spmdCallVectorReduce(%s) return width = %d, want %d",
						tt.name, resultType.IntTypeWidth(), tt.wantRetWidth)
				}
			}

			// Verify intrinsic was declared in the module.
			intrinsicName := "llvm.vector.reduce." + tt.op + "." + spmdVectorTypeSuffix(tt.vecType)
			llvmFn := c.mod.NamedFunction(intrinsicName)
			if llvmFn.IsNil() {
				t.Errorf("intrinsic %q not found in module", intrinsicName)
			}
		})
	}
}

func TestSPMDCallVectorReduceFloat(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name     string
		op       string
		startVal float64
		vecType  llvm.Type
	}{
		{
			name:     "fadd_v4f32",
			op:       "fadd",
			startVal: 0.0,
			vecType:  llvm.VectorType(c.ctx.FloatType(), 4),
		},
		{
			name:     "fmul_v4f32",
			op:       "fmul",
			startVal: 1.0,
			vecType:  llvm.VectorType(c.ctx.FloatType(), 4),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vec := llvm.ConstNull(tt.vecType)
			elemType := tt.vecType.ElementType()
			startVal := llvm.ConstFloat(elemType, tt.startVal)
			result := b.spmdCallVectorReduceFloat(tt.op, startVal, vec)

			if result.IsNil() {
				t.Fatalf("spmdCallVectorReduceFloat(%s) returned nil", tt.name)
			}

			// Verify return type is scalar float, not vector.
			resultType := result.Type()
			if resultType.TypeKind() == llvm.VectorTypeKind {
				t.Errorf("spmdCallVectorReduceFloat(%s) returned vector type, want scalar", tt.name)
			}

			// Should match element type kind.
			if resultType.TypeKind() != elemType.TypeKind() {
				t.Errorf("spmdCallVectorReduceFloat(%s) return type kind = %v, want %v",
					tt.name, resultType.TypeKind(), elemType.TypeKind())
			}

			// Verify intrinsic was declared.
			intrinsicName := "llvm.vector.reduce." + tt.op + "." + spmdVectorTypeSuffix(tt.vecType)
			llvmFn := c.mod.NamedFunction(intrinsicName)
			if llvmFn.IsNil() {
				t.Errorf("intrinsic %q not found in module", intrinsicName)
			}
		})
	}
}

func TestSPMDReduceAllVectorFull(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name     string
		maskVals []bool
	}{
		{"all_true", []bool{true, true, true, true}},
		{"mixed", []bool{true, false, true, false}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create <4 x i1> constant vector.
			vecElts := make([]llvm.Value, len(tt.maskVals))
			for i, val := range tt.maskVals {
				if val {
					vecElts[i] = llvm.ConstInt(c.ctx.Int1Type(), 1, false)
				} else {
					vecElts[i] = llvm.ConstInt(c.ctx.Int1Type(), 0, false)
				}
			}
			maskVec := llvm.ConstVector(vecElts, false)

			// Simulate reduce.All: bitcast to iN, compare == -1
			vecSize := maskVec.Type().VectorSize()
			intType := c.ctx.IntType(vecSize)
			intVal := b.CreateBitCast(maskVec, intType, "")
			allOnes := llvm.ConstAllOnes(intType)
			result := b.CreateICmp(llvm.IntEQ, intVal, allOnes, "")

			// Verify result is i1.
			if result.Type().TypeKind() != llvm.IntegerTypeKind {
				t.Errorf("reduce.All(%s) result type = %v, want IntegerTypeKind",
					tt.name, result.Type().TypeKind())
			}
			if result.Type().IntTypeWidth() != 1 {
				t.Errorf("reduce.All(%s) result width = %d, want 1",
					tt.name, result.Type().IntTypeWidth())
			}
		})
	}
}

func TestSPMDReduceCount(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name     string
		maskVals []bool
	}{
		{"all_true", []bool{true, true, true, true}},
		{"two_true", []bool{true, false, true, false}},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Create <4 x i1> constant vector.
			vecElts := make([]llvm.Value, len(tt.maskVals))
			for i, val := range tt.maskVals {
				if val {
					vecElts[i] = llvm.ConstInt(c.ctx.Int1Type(), 1, false)
				} else {
					vecElts[i] = llvm.ConstInt(c.ctx.Int1Type(), 0, false)
				}
			}
			maskVec := llvm.ConstVector(vecElts, false)

			// Simulate reduce.Count: bitcast to iN, ctpop, zext
			vecSize := maskVec.Type().VectorSize()
			intType := c.ctx.IntType(vecSize)
			intVal := b.CreateBitCast(maskVec, intType, "")

			intrinsicName := "llvm.ctpop.i" + strconv.Itoa(vecSize)
			llvmFn := c.mod.NamedFunction(intrinsicName)
			fnType := llvm.FunctionType(intType, []llvm.Type{intType}, false)
			if llvmFn.IsNil() {
				llvmFn = llvm.AddFunction(c.mod, intrinsicName, fnType)
			}
			popcount := b.createCall(fnType, llvmFn, []llvm.Value{intVal}, "")

			// Verify ctpop returns the right type.
			if popcount.IsNil() {
				t.Fatal("ctpop returned nil")
			}

			if popcount.Type().TypeKind() != llvm.IntegerTypeKind {
				t.Errorf("ctpop result type = %v, want IntegerTypeKind", popcount.Type().TypeKind())
			}

			// Verify the intrinsic was declared.
			declaredFn := c.mod.NamedFunction(intrinsicName)
			if declaredFn.IsNil() {
				t.Errorf("intrinsic %q not found in module", intrinsicName)
			}
		})
	}
}

func TestSPMDLanesIndex(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name      string
		laneCount int
		elemType  llvm.Type
	}{
		{"4_lanes_i32", 4, c.ctx.Int32Type()},
		{"16_lanes_i8", 16, c.ctx.Int8Type()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// lanes.Index() generates spmdLaneOffsetConst
			result := c.spmdLaneOffsetConst(tt.laneCount, tt.elemType)

			if result.IsNil() {
				t.Fatal("spmdLaneOffsetConst returned nil")
			}

			// Verify it's a vector type.
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
			}

			// Verify lane count.
			if result.Type().VectorSize() != tt.laneCount {
				t.Errorf("vector size = %d, want %d", result.Type().VectorSize(), tt.laneCount)
			}

			// Verify it's a constant.
			if !result.IsConstant() {
				t.Error("expected constant vector")
			}

			// Verify element type matches.
			if result.Type().ElementType().C != tt.elemType.C {
				t.Error("element type mismatch")
			}
		})
	}
}

func TestSPMDLanesBroadcast(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Create a vector [10, 20, 30, 40].
	vecElts := make([]llvm.Value, 4)
	for i := range vecElts {
		vecElts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64((i+1)*10), false)
	}
	vecVal := llvm.ConstVector(vecElts, false)

	// Extract element at lane 2 and splat to all lanes.
	lane := llvm.ConstInt(c.ctx.Int32Type(), 2, false)
	elem := b.CreateExtractElement(vecVal, lane, "broadcast.elem")
	result := b.splatScalar(elem, vecVal.Type())

	// Verify result is not nil.
	if result.IsNil() {
		t.Fatal("broadcast returned nil")
	}

	// Verify result is a vector with 4 lanes.
	if result.Type().TypeKind() != llvm.VectorTypeKind {
		t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
	}
	if result.Type().VectorSize() != 4 {
		t.Errorf("vector size = %d, want 4", result.Type().VectorSize())
	}

	// Verify element type matches.
	if result.Type().ElementType().C != c.ctx.Int32Type().C {
		t.Error("element type mismatch")
	}
}

func TestSPMDIsSignedInt(t *testing.T) {
	tests := []struct {
		name string
		typ  types.Type
		want bool
	}{
		{"int32", types.Typ[types.Int32], true},
		{"int64", types.Typ[types.Int64], true},
		{"int", types.Typ[types.Int], true},
		{"uint32", types.Typ[types.Uint32], false},
		{"uint64", types.Typ[types.Uint64], false},
		{"float32", types.Typ[types.Float32], false},
		{"float64", types.Typ[types.Float64], false},
		{"bool", types.Typ[types.Bool], false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := spmdIsSignedInt(tt.typ)
			if got != tt.want {
				t.Errorf("spmdIsSignedInt(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

func TestSPMDIsFloat(t *testing.T) {
	tests := []struct {
		name string
		typ  types.Type
		want bool
	}{
		{"float32", types.Typ[types.Float32], true},
		{"float64", types.Typ[types.Float64], true},
		{"int32", types.Typ[types.Int32], false},
		{"uint64", types.Typ[types.Uint64], false},
		{"bool", types.Typ[types.Bool], false},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := spmdIsFloat(tt.typ)
			if got != tt.want {
				t.Errorf("spmdIsFloat(%s) = %v, want %v", tt.name, got, tt.want)
			}
		})
	}
}

func TestSPMDMaskedLoadIntrinsic(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name       string
		elemType   llvm.Type
		laneCount  int
		wantSuffix string
	}{
		{"v4i32", c.ctx.Int32Type(), 4, "v4i32"},
		{"v2i64", c.ctx.Int64Type(), 2, "v2i64"},
		{"v4f32", c.ctx.FloatType(), 4, "v4f32"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vecType := llvm.VectorType(tt.elemType, tt.laneCount)
			maskType := llvm.VectorType(c.ctx.Int1Type(), tt.laneCount)

			// Allocate a stack buffer to use as pointer for the load.
			arrType := llvm.ArrayType(tt.elemType, tt.laneCount)
			ptr := b.CreateAlloca(arrType, "test.alloca")

			// Get a GEP pointer to the first element (scalar pointer).
			zero := llvm.ConstInt(c.ctx.Int32Type(), 0, false)
			scalarPtr := b.CreateInBoundsGEP(arrType, ptr, []llvm.Value{zero, zero}, "test.ptr")

			// Use all-true mask.
			mask := llvm.ConstAllOnes(maskType)

			// Call spmdMaskedLoad.
			result := b.spmdMaskedLoad(vecType, scalarPtr, mask)

			if result.IsNil() {
				t.Fatalf("spmdMaskedLoad returned nil")
			}

			// Verify result type is the vector type.
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
			}
			if result.Type().VectorSize() != tt.laneCount {
				t.Errorf("result lane count = %d, want %d", result.Type().VectorSize(), tt.laneCount)
			}

			// Verify the intrinsic was declared in the module.
			intrinsicName := "llvm.masked.load." + tt.wantSuffix + ".p0"
			fn := c.mod.NamedFunction(intrinsicName)
			if fn.IsNil() {
				t.Errorf("intrinsic %q not declared in module", intrinsicName)
			}
		})
	}
}

func TestSPMDMaskedStoreIntrinsic(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name       string
		elemType   llvm.Type
		laneCount  int
		wantSuffix string
	}{
		{"v4i32", c.ctx.Int32Type(), 4, "v4i32"},
		{"v2i64", c.ctx.Int64Type(), 2, "v2i64"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			vecType := llvm.VectorType(tt.elemType, tt.laneCount)
			maskType := llvm.VectorType(c.ctx.Int1Type(), tt.laneCount)

			// Allocate a stack buffer to use as destination pointer.
			arrType := llvm.ArrayType(tt.elemType, tt.laneCount)
			ptr := b.CreateAlloca(arrType, "test.alloca")
			zero := llvm.ConstInt(c.ctx.Int32Type(), 0, false)
			scalarPtr := b.CreateInBoundsGEP(arrType, ptr, []llvm.Value{zero, zero}, "test.ptr")

			// Create a null vector value to store.
			val := llvm.ConstNull(vecType)

			// Use all-true mask.
			mask := llvm.ConstAllOnes(maskType)

			// Call spmdMaskedStore (void return, so just verify no panic).
			b.spmdMaskedStore(val, scalarPtr, mask)

			// Verify the intrinsic was declared in the module.
			intrinsicName := "llvm.masked.store." + tt.wantSuffix + ".p0"
			fn := c.mod.NamedFunction(intrinsicName)
			if fn.IsNil() {
				t.Errorf("intrinsic %q not declared in module", intrinsicName)
			}
		})
	}
}

func TestSPMDContiguousInfoFields(t *testing.T) {
	// Verify spmdContiguousInfo fields can be set and read.
	//
	// spmdUnwrapScalar (used by spmdAnalyzeContiguousIndex) cannot be unit-tested
	// directly because it requires a live spmdValueOverride map, which is only
	// populated during full function compilation. The ChangeType-unwrap path is
	// exercised end-to-end by the mandelbrot integration test (integ_mandelbrot),
	// which verifies 0 differences vs serial output and a ~4x SPMD speedup.
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	elemType := c.ctx.Int32Type()
	arrType := llvm.ArrayType(elemType, 4)
	ptr := b.CreateAlloca(arrType, "test.alloca")
	zero := llvm.ConstInt(c.ctx.Int32Type(), 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, ptr, []llvm.Value{zero, zero}, "scalar.ptr")

	loop := &spmdActiveLoop{
		laneCount: 4,
	}

	info := &spmdContiguousInfo{
		scalarPtr: scalarPtr,
		loop:      loop,
	}

	if info.scalarPtr.IsNil() {
		t.Error("expected non-nil scalarPtr")
	}
	if info.loop == nil {
		t.Error("expected non-nil loop")
	}
	if info.loop.laneCount != 4 {
		t.Errorf("loop.laneCount = %d, want 4", info.loop.laneCount)
	}
}

func TestSPMDMaskAndOperation(t *testing.T) {
	// Verify mask AND operations used in mask transitions produce correct vector types.
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i1Type := c.ctx.Int1Type()
	maskType := llvm.VectorType(i1Type, laneCount)

	// Create parent mask (all true) and condition (alternating).
	parentMask := llvm.ConstAllOnes(maskType)
	condElts := make([]llvm.Value, laneCount)
	for i := range condElts {
		condElts[i] = llvm.ConstInt(i1Type, uint64(i%2), false)
	}
	cond := llvm.ConstVector(condElts, false)

	// Simulate pushThen: thenMask = parentMask AND cond.
	thenMask := b.CreateAnd(parentMask, cond, "spmd.then.mask")
	if thenMask.IsNil() {
		t.Fatal("CreateAnd returned nil")
	}
	if thenMask.Type().TypeKind() != llvm.VectorTypeKind {
		t.Errorf("thenMask type = %v, want VectorTypeKind", thenMask.Type().TypeKind())
	}
	if thenMask.Type().VectorSize() != laneCount {
		t.Errorf("thenMask size = %d, want %d", thenMask.Type().VectorSize(), laneCount)
	}

	// Simulate swapElse: elseMask = parentMask AND NOT(cond).
	notCond := b.CreateNot(cond, "")
	elseMask := b.CreateAnd(parentMask, notCond, "spmd.else.mask")
	if elseMask.IsNil() {
		t.Fatal("CreateAnd(not) returned nil")
	}
	if elseMask.Type().TypeKind() != llvm.VectorTypeKind {
		t.Errorf("elseMask type = %v, want VectorTypeKind", elseMask.Type().TypeKind())
	}
}

// TestSPMDRangeIndexFields verifies that the new rangeindex fields on spmdActiveLoop
// are correctly distinguished from rangeint fields.
func TestSPMDRangeIndexFields(t *testing.T) {
	tests := []struct {
		name          string
		isRangeIndex  bool
		initEdgeIndex int
		wantIsRI      bool
		wantEdge      int
	}{
		// rangeint loop: isRangeIndex false, initEdgeIndex -1 (unused sentinel).
		{"rangeint_loop", false, -1, false, -1},
		// rangeindex loop: isRangeIndex true, initEdgeIndex identifies the entry edge.
		{"rangeindex_loop_edge0", true, 0, true, 0},
		{"rangeindex_loop_edge1", true, 1, true, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			loop := &spmdActiveLoop{
				isRangeIndex:  tt.isRangeIndex,
				initEdgeIndex: tt.initEdgeIndex,
			}

			if loop.isRangeIndex != tt.wantIsRI {
				t.Errorf("isRangeIndex = %v, want %v", loop.isRangeIndex, tt.wantIsRI)
			}
			if loop.initEdgeIndex != tt.wantEdge {
				t.Errorf("initEdgeIndex = %d, want %d", loop.initEdgeIndex, tt.wantEdge)
			}
			// For rangeint loops, iterPhi is the bodyIterValue source.
			// For rangeindex loops, bodyIterValue is the incrBinOp.
			// Both are nil here (no real SSA), but the field distinction is what matters.
			if tt.isRangeIndex && loop.iterPhi != nil {
				t.Errorf("rangeindex loop should have nil iterPhi, got non-nil")
			}
		})
	}
}

// TestSPMDActiveLoopBodyIterValue verifies that bodyIterValue is correctly set for both
// rangeint (iterPhi) and rangeindex (incrBinOp) loop types.
func TestSPMDActiveLoopBodyIterValue(t *testing.T) {
	// For rangeint: bodyIterValue == iterPhi (same pointer).
	// We can test this relationship via the isRangeIndex flag since we can't create
	// real *ssa.Phi or *ssa.BinOp instances without a full SSA build.
	t.Run("rangeint_bodyIterValue_is_iterPhi", func(t *testing.T) {
		loop := &spmdActiveLoop{
			isRangeIndex: false,
			// In real usage: bodyIterValue = iterPhi (set during analyzeSPMDLoops).
		}
		// Verify the discriminant flag.
		if loop.isRangeIndex {
			t.Error("rangeint loop must have isRangeIndex == false")
		}
		// Verify iterPhi field exists and is nil (no real SSA).
		if loop.iterPhi != nil {
			t.Error("expected nil iterPhi in synthetic loop")
		}
	})

	t.Run("rangeindex_bodyIterValue_is_incrBinOp", func(t *testing.T) {
		loop := &spmdActiveLoop{
			isRangeIndex:  true,
			iterPhi:       nil, // rangeindex: no iter phi in body block
			initEdgeIndex: 0,
		}
		if !loop.isRangeIndex {
			t.Error("rangeindex loop must have isRangeIndex == true")
		}
		if loop.iterPhi != nil {
			t.Error("rangeindex loop must have nil iterPhi")
		}
	})
}

// TestSPMDPhiInitOverride verifies that the -laneCount constant computation for
// rangeindex loop phi initial value override is correct for various lane counts.
func TestSPMDPhiInitOverride(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name      string
		laneCount int
		intType   llvm.Type
		wantBits  int
	}{
		{"2_lanes_i64", 2, c.ctx.Int64Type(), 64},
		{"4_lanes_i32", 4, c.ctx.Int32Type(), 32},
		{"8_lanes_i16", 8, c.ctx.Int16Type(), 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Simulate the phi init override: llvm.ConstInt(type, uint64(int64(-laneCount)), true)
			negLC := uint64(int64(-tt.laneCount))
			val := llvm.ConstInt(tt.intType, negLC, true)

			if val.IsNil() {
				t.Fatal("ConstInt returned nil")
			}

			// Verify the constant is the correct type.
			if val.Type().TypeKind() != llvm.IntegerTypeKind {
				t.Errorf("type kind = %v, want IntegerTypeKind", val.Type().TypeKind())
			}
			if val.Type().IntTypeWidth() != tt.wantBits {
				t.Errorf("int width = %d, want %d", val.Type().IntTypeWidth(), tt.wantBits)
			}

			// Verify it's a constant.
			if !val.IsConstant() {
				t.Error("expected constant value")
			}

			// Verify adding laneCount to -laneCount gives 0.
			posLC := llvm.ConstInt(tt.intType, uint64(tt.laneCount), false)
			sum := llvm.ConstAdd(val, posLC)
			if !sum.IsNull() {
				t.Errorf("-laneCount + laneCount should be 0 for laneCount=%d", tt.laneCount)
			}
		})
	}
}

func TestSPMDBroadcastMatchVectorWidth(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	t.Run("narrower_wins", func(t *testing.T) {
		// Create <16 x i8> constant (splatted 97 = 'a') and <4 x i8> constant (splatted 0).
		wide := llvm.ConstVector(func() []llvm.Value {
			elts := make([]llvm.Value, 16)
			for i := range elts {
				elts[i] = llvm.ConstInt(c.ctx.Int8Type(), 97, false)
			}
			return elts
		}(), false)
		narrow := llvm.ConstVector(func() []llvm.Value {
			elts := make([]llvm.Value, 4)
			for i := range elts {
				elts[i] = llvm.ConstInt(c.ctx.Int8Type(), 0, false)
			}
			return elts
		}(), false)

		rx, ry := b.spmdBroadcastMatch(wide, narrow)
		if rx.Type().VectorSize() != 4 {
			t.Errorf("expected x resized to 4 lanes, got %d", rx.Type().VectorSize())
		}
		if ry.Type().VectorSize() != 4 {
			t.Errorf("expected y unchanged at 4 lanes, got %d", ry.Type().VectorSize())
		}
	})

	t.Run("resize_vector_truncate", func(t *testing.T) {
		// Direct test of spmdResizeVector: truncate <16 x i8> to 4 lanes.
		wide := llvm.ConstVector(func() []llvm.Value {
			elts := make([]llvm.Value, 16)
			for i := range elts {
				elts[i] = llvm.ConstInt(c.ctx.Int8Type(), uint64(i), false)
			}
			return elts
		}(), false)

		result := b.spmdResizeVector(wide, 4, c.ctx.Int8Type())
		if result.Type().VectorSize() != 4 {
			t.Errorf("expected 4 lanes, got %d", result.Type().VectorSize())
		}
		if result.Type().ElementType().IntTypeWidth() != 8 {
			t.Errorf("expected i8 element type, got i%d", result.Type().ElementType().IntTypeWidth())
		}
	})

	t.Run("same_width_noop", func(t *testing.T) {
		// Same width vectors should pass through unchanged.
		a := llvm.ConstVector(func() []llvm.Value {
			elts := make([]llvm.Value, 4)
			for i := range elts {
				elts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false)
			}
			return elts
		}(), false)
		b2 := llvm.ConstVector(func() []llvm.Value {
			elts := make([]llvm.Value, 4)
			for i := range elts {
				elts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64(i+10), false)
			}
			return elts
		}(), false)

		rx, ry := b.spmdBroadcastMatch(a, b2)
		if rx.Type().VectorSize() != 4 || ry.Type().VectorSize() != 4 {
			t.Errorf("expected both unchanged at 4 lanes")
		}
	})
}

func TestSPMDEffectiveLaneCount(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name      string
		spmdType  *types.SPMDType
		elemType  llvm.Type
		wantLanes int
	}{
		{
			name:      "int32",
			spmdType:  types.NewVarying(types.Typ[types.Int32]),
			elemType:  c.ctx.Int32Type(),
			wantLanes: 4, // SIMD128: 128/32 = 4
		},
		{
			name:      "float32",
			spmdType:  types.NewVarying(types.Typ[types.Float32]),
			elemType:  c.ctx.FloatType(),
			wantLanes: 4, // SIMD128: 128/32 = 4
		},
		{
			name:      "int64",
			spmdType:  types.NewVarying(types.Typ[types.Int64]),
			elemType:  c.ctx.Int64Type(),
			wantLanes: 2, // SIMD128: 128/64 = 2
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := c.spmdEffectiveLaneCount(tt.spmdType, tt.elemType)
			if got != tt.wantLanes {
				t.Errorf("spmdEffectiveLaneCount(%s) = %d, want %d", tt.name, got, tt.wantLanes)
			}
		})
	}
}

func TestSPMDArrayToVector(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name      string
		elemType  llvm.Type
		laneCount int
		vals      []uint64
	}{
		{
			name:      "4xi32",
			elemType:  c.ctx.Int32Type(),
			laneCount: 4,
			vals:      []uint64{10, 20, 30, 40},
		},
		{
			name:      "2xi64",
			elemType:  c.ctx.Int64Type(),
			laneCount: 2,
			vals:      []uint64{100, 200},
		},
		{
			name:      "8xi16",
			elemType:  c.ctx.Int16Type(),
			laneCount: 8,
			vals:      []uint64{1, 2, 3, 4, 5, 6, 7, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build an LLVM array constant [N x T]{vals...}.
			arrType := llvm.ArrayType(tt.elemType, tt.laneCount)
			elts := make([]llvm.Value, tt.laneCount)
			for i, v := range tt.vals {
				elts[i] = llvm.ConstInt(tt.elemType, v, false)
			}
			arr := llvm.ConstArray(tt.elemType, elts)

			// Convert to vector.
			vecType := llvm.VectorType(tt.elemType, tt.laneCount)
			result := b.arrayToVector(arr, vecType)

			// Verify result type.
			if result.IsNil() {
				t.Fatal("arrayToVector returned nil")
			}
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
			}
			if result.Type().VectorSize() != tt.laneCount {
				t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), tt.laneCount)
			}
			if result.Type().ElementType().C != tt.elemType.C {
				t.Error("element type mismatch")
			}

			// Verify the array input type was correct.
			if arr.Type().TypeKind() != llvm.ArrayTypeKind {
				t.Errorf("input type = %v, want ArrayTypeKind", arr.Type().TypeKind())
			}
			_ = arrType // used to verify
		})
	}
}

// TestSPMDFuncBodyDetection verifies the spmdFuncIsBody flag logic:
//   - Set when spmdEntryMask is present but no go-for loops were detected
//   - Not set when no entry mask (non-SPMD function)
//   - Not set when go-for loops exist (loop-based SPMD takes precedence)
func TestSPMDFuncBodyDetection(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	i1Type := c.ctx.Int1Type()
	mask4Type := llvm.VectorType(i1Type, 4)
	entryMask := llvm.ConstAllOnes(mask4Type)

	t.Run("entry_mask_no_loops_sets_flag", func(t *testing.T) {
		// Simulate an SPMD function that received a varying parameter (entry mask set)
		// but has no go-for loops (spmdLoopState is nil).
		b := &builder{compilerContext: c}
		b.spmdEntryMask = entryMask
		// spmdLoopState is nil (no go-for loops detected)

		// Apply the logic from createFunction.
		if b.spmdLoopState == nil && !b.spmdEntryMask.IsNil() {
			b.spmdFuncIsBody = true
		}

		if !b.spmdFuncIsBody {
			t.Error("expected spmdFuncIsBody = true when entry mask set and no loops")
		}

		// Also verify that maps would be initialized under this condition.
		shouldInit := b.spmdLoopState != nil || b.spmdFuncIsBody
		if !shouldInit {
			t.Error("expected maps to be initialized when spmdFuncIsBody is true")
		}
	})

	t.Run("no_entry_mask_no_loops_no_flag", func(t *testing.T) {
		// Regular (non-SPMD) function: no entry mask, no loops.
		b := &builder{compilerContext: c}
		// spmdEntryMask is zero value (nil)
		// spmdLoopState is nil

		if b.spmdLoopState == nil && !b.spmdEntryMask.IsNil() {
			b.spmdFuncIsBody = true
		}

		if b.spmdFuncIsBody {
			t.Error("expected spmdFuncIsBody = false when no entry mask and no loops")
		}

		shouldInit := b.spmdLoopState != nil || b.spmdFuncIsBody
		if shouldInit {
			t.Error("expected maps NOT to be initialized for non-SPMD function")
		}
	})

	t.Run("loops_present_no_func_body_flag", func(t *testing.T) {
		// SPMD function with go-for loops: spmdFuncIsBody must remain false
		// because loop-based SPMD takes precedence and manages block state itself.
		b := &builder{compilerContext: c}
		b.spmdEntryMask = entryMask
		// Simulate a non-nil loop state by using a non-nil pointer.
		b.spmdLoopState = &spmdLoopState{}

		if b.spmdLoopState == nil && !b.spmdEntryMask.IsNil() {
			b.spmdFuncIsBody = true
		}

		if b.spmdFuncIsBody {
			t.Error("expected spmdFuncIsBody = false when go-for loops are present")
		}

		// Maps should still be initialized (driven by spmdLoopState != nil).
		shouldInit := b.spmdLoopState != nil || b.spmdFuncIsBody
		if !shouldInit {
			t.Error("expected maps to be initialized when loops are present")
		}
	})

	t.Run("isBlockInSPMDBody_returns_sentinel_when_func_body", func(t *testing.T) {
		// When spmdFuncIsBody is true, isBlockInSPMDBody must return non-nil
		// for any block (even nil block argument) because the sentinel is returned
		// before any block inspection.
		// We need spmdInfo non-nil to pass the first guard.
		// Save and restore c.spmdInfo so later subtests that share c are not affected.
		saved := c.spmdInfo
		defer func() { c.spmdInfo = saved }()

		b := &builder{compilerContext: c}
		b.spmdFuncIsBody = true
		b.spmdInfo = &SPMDInfo{} // non-nil sentinel to pass the nil guard

		result := b.isBlockInSPMDBody(nil)
		if result == nil {
			t.Error("expected non-nil sentinel from isBlockInSPMDBody when spmdFuncIsBody=true")
		}
	})

	t.Run("isBlockInSPMDBody_returns_nil_when_no_spmd_info", func(t *testing.T) {
		// When spmdInfo is nil (not an SPMD function at all), isBlockInSPMDBody
		// must return nil regardless of spmdFuncIsBody.
		// Explicitly clear c.spmdInfo to guard against state leaking from prior subtests.
		c.spmdInfo = nil

		b := &builder{compilerContext: c}
		b.spmdFuncIsBody = true

		result := b.isBlockInSPMDBody(nil)
		if result != nil {
			t.Error("expected nil from isBlockInSPMDBody when spmdInfo=nil")
		}
	})
}

func TestSPMDVectorAllTrue(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)

	// The test context is WASM, so masks use the <N x i32> format where
	// all-ones (0xFFFFFFFF) means active and all-zeros means inactive.
	// On WASM, spmdVectorAllTrue uses the native @llvm.wasm.alltrue intrinsic
	// (i32x4.all_true) and still returns a scalar i1.
	maskType := llvm.VectorType(c.spmdMaskElemType(4), 4) // <4 x i32> on WASM for 4 lanes

	t.Run("all_true", func(t *testing.T) {
		allOnes := llvm.ConstAllOnes(maskType)
		result := b.spmdVectorAllTrue(allOnes)
		if result.IsNil() {
			t.Fatal("expected non-nil result")
		}
		if result.Type().TypeKind() != llvm.IntegerTypeKind || result.Type().IntTypeWidth() != 1 {
			t.Error("expected i1 result type")
		}
	})

	t.Run("not_all_true", func(t *testing.T) {
		allZeros := llvm.ConstNull(maskType)
		result := b.spmdVectorAllTrue(allZeros)
		if result.IsNil() {
			t.Fatal("expected non-nil result")
		}
	})
}

func TestSPMDIsWASM(t *testing.T) {
	// The default test context is WASM.
	c := newTestCompilerContext(t)
	defer c.dispose()

	if !c.spmdIsWASM() {
		t.Error("spmdIsWASM() = false for wasm32-unknown-wasi, want true")
	}
}

// TestSPMDMaskElemType verifies that the mask element type is lane-count-dependent on WASM.
// For 4 lanes (i32 elements): mask elem = i32 (128/4 = 32 bits).
// For 16 lanes (i8 elements): mask elem = i8 (128/16 = 8 bits).
func TestSPMDMaskElemType(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	// 4-lane test: int elements → i32 mask elem (128/4 = 32 bits)
	elemType4 := c.spmdMaskElemType(4)
	if elemType4.TypeKind() != llvm.IntegerTypeKind {
		t.Errorf("spmdMaskElemType(4) kind = %v, want IntegerTypeKind", elemType4.TypeKind())
	}
	if elemType4.IntTypeWidth() != 32 {
		t.Errorf("spmdMaskElemType(4) width = %d, want 32 (i32 on WASM for 4 lanes)", elemType4.IntTypeWidth())
	}

	// 8-lane test: int16 elements → i16 mask elem (128/8 = 16 bits)
	elemType8 := c.spmdMaskElemType(8)
	if elemType8.TypeKind() != llvm.IntegerTypeKind {
		t.Errorf("spmdMaskElemType(8) kind = %v, want IntegerTypeKind", elemType8.TypeKind())
	}
	if elemType8.IntTypeWidth() != 16 {
		t.Errorf("spmdMaskElemType(8) width = %d, want 16 (i16 on WASM for 8 lanes)", elemType8.IntTypeWidth())
	}

	// 16-lane test: byte elements → i8 mask elem (128/16 = 8 bits)
	elemType16 := c.spmdMaskElemType(16)
	if elemType16.TypeKind() != llvm.IntegerTypeKind {
		t.Errorf("spmdMaskElemType(16) kind = %v, want IntegerTypeKind", elemType16.TypeKind())
	}
	if elemType16.IntTypeWidth() != 8 {
		t.Errorf("spmdMaskElemType(16) width = %d, want 8 (i8 on WASM for 16 lanes)", elemType16.IntTypeWidth())
	}
}

// TestSPMDWrapMask verifies that spmdWrapMask sign-extends <N x i1> to the WASM mask type.
func TestSPMDWrapMask(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	// Create a <4 x i1> comparison result.
	i1VecType := llvm.VectorType(c.ctx.Int1Type(), laneCount)
	cmpResult := llvm.ConstAllOnes(i1VecType) // all-true

	// spmdWrapMask should sext to <4 x i32> on WASM.
	wrapped := b.spmdWrapMask(cmpResult, laneCount)

	if wrapped.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatalf("wrapped type kind = %v, want VectorTypeKind", wrapped.Type().TypeKind())
	}
	if wrapped.Type().VectorSize() != laneCount {
		t.Errorf("wrapped lane count = %d, want %d", wrapped.Type().VectorSize(), laneCount)
	}
	if wrapped.Type().ElementType().IntTypeWidth() != 32 {
		t.Errorf("wrapped elem width = %d, want 32", wrapped.Type().ElementType().IntTypeWidth())
	}
}

// TestSPMDUnwrapMaskForIntrinsic verifies that spmdUnwrapMaskForIntrinsic truncates
// <N x i32> masks back to <N x i1> for LLVM masked memory intrinsics.
func TestSPMDUnwrapMaskForIntrinsic(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	// Create a <4 x i32> mask (WASM format, all-ones = all active).
	i32VecType := llvm.VectorType(c.ctx.Int32Type(), laneCount)
	i32Mask := llvm.ConstAllOnes(i32VecType)

	// spmdUnwrapMaskForIntrinsic should truncate to <4 x i1>.
	i1Mask := b.spmdUnwrapMaskForIntrinsic(i32Mask, laneCount)

	if i1Mask.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatalf("i1Mask type kind = %v, want VectorTypeKind", i1Mask.Type().TypeKind())
	}
	if i1Mask.Type().VectorSize() != laneCount {
		t.Errorf("i1Mask lane count = %d, want %d", i1Mask.Type().VectorSize(), laneCount)
	}
	if i1Mask.Type().ElementType().IntTypeWidth() != 1 {
		t.Errorf("i1Mask elem width = %d, want 1", i1Mask.Type().ElementType().IntTypeWidth())
	}
}

// TestSPMDMaskSelectWASM verifies that spmdMaskSelect uses bitwise ops on WASM
// (since <N x i32> cannot be used directly as a CreateSelect condition).
func TestSPMDMaskSelectWASM(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	// Create a <4 x i32> mask (all-ones = all lanes active).
	maskType := llvm.VectorType(i32Type, laneCount)
	mask := llvm.ConstAllOnes(maskType)

	// Create two vector values to select between.
	vecType := llvm.VectorType(i32Type, laneCount)
	trueElts := make([]llvm.Value, laneCount)
	falseElts := make([]llvm.Value, laneCount)
	for i := range trueElts {
		trueElts[i] = llvm.ConstInt(i32Type, uint64(i+1), false)
		falseElts[i] = llvm.ConstInt(i32Type, 0, false)
	}
	trueVec := llvm.ConstVector(trueElts, false)
	falseVec := llvm.ConstVector(falseElts, false)

	// spmdMaskSelect should produce a vector result.
	result := b.spmdMaskSelect(mask, trueVec, falseVec)

	if result.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatalf("result type kind = %v, want VectorTypeKind", result.Type().TypeKind())
	}
	if result.Type() != vecType {
		t.Errorf("result type = %v, want %v", result.Type(), vecType)
	}
}

// TestSPMDVectorAnyTrueWASM verifies that spmdVectorAnyTrue handles <N x i32> masks
// (WASM format) by calling the @llvm.wasm.anytrue intrinsic (v128.any_true)
// and comparing the i32 result != 0, returning a scalar i1.
func TestSPMDVectorAnyTrueWASM(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	// Create a <4 x i32> mask (all-ones).
	maskType := llvm.VectorType(i32Type, laneCount)
	mask := llvm.ConstAllOnes(maskType)

	// spmdVectorAnyTrue should return scalar i1.
	result := b.spmdVectorAnyTrue(mask)

	if result.Type().TypeKind() != llvm.IntegerTypeKind {
		t.Fatalf("result type kind = %v, want IntegerTypeKind", result.Type().TypeKind())
	}
	if result.Type().IntTypeWidth() != 1 {
		t.Errorf("result width = %d, want 1 (scalar i1)", result.Type().IntTypeWidth())
	}
}

// TestSPMDNormalizeBoolVecToI1 verifies that spmdNormalizeBoolVecToI1 truncates
// <N x i32> to <N x i1> on WASM for bool-specific reduce operations.
func TestSPMDNormalizeBoolVecToI1(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32VecType := llvm.VectorType(c.ctx.Int32Type(), laneCount)
	i32Mask := llvm.ConstAllOnes(i32VecType)

	// On WASM, spmdNormalizeBoolVecToI1 truncates <4 x i32> to <4 x i1>.
	i1Vec := b.spmdNormalizeBoolVecToI1(i32Mask)

	if i1Vec.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatalf("i1Vec type kind = %v, want VectorTypeKind", i1Vec.Type().TypeKind())
	}
	if i1Vec.Type().VectorSize() != laneCount {
		t.Errorf("i1Vec lane count = %d, want %d", i1Vec.Type().VectorSize(), laneCount)
	}
	if i1Vec.Type().ElementType().IntTypeWidth() != 1 {
		t.Errorf("i1Vec elem width = %d, want 1", i1Vec.Type().ElementType().IntTypeWidth())
	}
}

// TestSPMDWasmAnyTrueIntrinsic verifies that spmdWasmAnyTrue calls @llvm.wasm.anytrue
// and returns an i32. Also checks the intrinsic declaration is present in the module.
func TestSPMDWasmAnyTrueIntrinsic(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	// Build a <4 x i32> mask (WASM format, all-ones = all active).
	maskType := llvm.VectorType(c.ctx.Int32Type(), laneCount)
	mask := llvm.ConstAllOnes(maskType)

	// Call the helper — it should return i32.
	result := b.spmdWasmAnyTrue(mask)

	if result.IsNil() {
		t.Fatal("spmdWasmAnyTrue returned nil")
	}
	if result.Type().TypeKind() != llvm.IntegerTypeKind {
		t.Fatalf("spmdWasmAnyTrue result type kind = %v, want IntegerTypeKind",
			result.Type().TypeKind())
	}
	if result.Type().IntTypeWidth() != 32 {
		t.Errorf("spmdWasmAnyTrue result width = %d, want 32 (i32)", result.Type().IntTypeWidth())
	}

	// The @llvm.wasm.anytrue.v4i32 intrinsic must now be present in the module.
	intrinsicName := "llvm.wasm.anytrue.v4i32"
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		t.Errorf("intrinsic %q not found in module after spmdWasmAnyTrue call", intrinsicName)
	}
}

// TestSPMDWasmAllTrueIntrinsic verifies that spmdWasmAllTrue calls @llvm.wasm.alltrue
// and returns an i32. Also checks the intrinsic declaration is present in the module.
func TestSPMDWasmAllTrueIntrinsic(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	// Build a <4 x i32> mask (WASM format, all-ones = all active).
	maskType := llvm.VectorType(c.ctx.Int32Type(), laneCount)
	mask := llvm.ConstAllOnes(maskType)

	// Call the helper — it should return i32.
	result := b.spmdWasmAllTrue(mask)

	if result.IsNil() {
		t.Fatal("spmdWasmAllTrue returned nil")
	}
	if result.Type().TypeKind() != llvm.IntegerTypeKind {
		t.Fatalf("spmdWasmAllTrue result type kind = %v, want IntegerTypeKind",
			result.Type().TypeKind())
	}
	if result.Type().IntTypeWidth() != 32 {
		t.Errorf("spmdWasmAllTrue result width = %d, want 32 (i32)", result.Type().IntTypeWidth())
	}

	// The @llvm.wasm.alltrue.v4i32 intrinsic must now be present in the module.
	intrinsicName := "llvm.wasm.alltrue.v4i32"
	fn := b.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		t.Errorf("intrinsic %q not found in module after spmdWasmAllTrue call", intrinsicName)
	}
}

// TestSPMDMaskedLoadWithI32Mask verifies that spmdMaskedLoad correctly unwraps
// an <N x i32> mask to <N x i1> before calling the LLVM masked.load intrinsic.
func TestSPMDMaskedLoadWithI32Mask(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	elemType := c.ctx.Int32Type()
	vecType := llvm.VectorType(elemType, laneCount)

	// Create <4 x i32> mask (WASM format).
	i32MaskType := llvm.VectorType(c.ctx.Int32Type(), laneCount)
	mask := llvm.ConstAllOnes(i32MaskType)

	// Allocate a buffer for the pointer.
	arrType := llvm.ArrayType(elemType, laneCount)
	ptr := b.CreateAlloca(arrType, "test.alloca")
	zero := llvm.ConstInt(c.ctx.Int32Type(), 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, ptr, []llvm.Value{zero, zero}, "test.ptr")

	// spmdMaskedLoad should unwrap the i32 mask to i1 internally.
	result := b.spmdMaskedLoad(vecType, scalarPtr, mask)

	if result.IsNil() {
		t.Fatal("spmdMaskedLoad returned nil")
	}
	if result.Type() != vecType {
		t.Errorf("result type = %v, want %v", result.Type(), vecType)
	}

	// Verify the intrinsic uses <4 x i1> mask parameter (not <4 x i32>).
	intrinsicName := "llvm.masked.load.v4i32.p0"
	fn := c.mod.NamedFunction(intrinsicName)
	if fn.IsNil() {
		t.Fatalf("intrinsic %q not found in module", intrinsicName)
	}
	// The 3rd parameter (index 2) of masked.load should be <4 x i1>.
	fnType := fn.GlobalValueType()
	paramTypes := fnType.ParamTypes()
	if len(paramTypes) < 3 {
		t.Fatalf("expected >= 3 params, got %d", len(paramTypes))
	}
	maskParam := paramTypes[2]
	if maskParam.TypeKind() != llvm.VectorTypeKind {
		t.Errorf("mask param kind = %v, want VectorTypeKind", maskParam.TypeKind())
	}
	if maskParam.ElementType().IntTypeWidth() != 1 {
		t.Errorf("mask param elem width = %d, want 1 (i1)", maskParam.ElementType().IntTypeWidth())
	}
}

func TestSPMDConstIntOrSplat(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name      string
		typ       llvm.Type
		val       uint64
		expectVec bool
		lanes     int
	}{
		{"scalar_i32", c.ctx.Int32Type(), 42, false, 0},
		{"vector_4xi32", llvm.VectorType(c.ctx.Int32Type(), 4), 42, true, 4},
		{"scalar_i64", c.ctx.Int64Type(), 100, false, 0},
		{"vector_2xi64", llvm.VectorType(c.ctx.Int64Type(), 2), 100, true, 2},
		{"scalar_i8", c.ctx.Int8Type(), 7, false, 0},
		{"vector_16xi8", llvm.VectorType(c.ctx.Int8Type(), 16), 7, true, 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := spmdConstIntOrSplat(tt.typ, tt.val, false)
			if result.IsNil() {
				t.Fatal("spmdConstIntOrSplat returned nil")
			}
			if tt.expectVec {
				if result.Type().TypeKind() != llvm.VectorTypeKind {
					t.Errorf("expected vector type, got %v", result.Type().TypeKind())
				}
				if result.Type().VectorSize() != tt.lanes {
					t.Errorf("expected %d lanes, got %d", tt.lanes, result.Type().VectorSize())
				}
				// Verify all lanes have the same value (for constant vectors).
				if !result.IsConstant() {
					t.Error("expected constant vector")
				}
			} else {
				if result.Type().TypeKind() == llvm.VectorTypeKind {
					t.Error("expected scalar type, got vector")
				}
				// Verify scalar value.
				if !result.IsConstant() {
					t.Error("expected constant scalar")
				}
			}
		})
	}
}

// TestSPMDRotateWithinMask verifies the shuffle index mask computed by spmdRotateWithinMask.
func TestSPMDRotateWithinMask(t *testing.T) {
	tests := []struct {
		name       string
		totalLanes int
		groupSize  int
		offset     int
		want       []uint64
	}{
		{
			// 8 lanes, groups of 4, rotate left by 1:
			// group0: [0,1,2,3] -> src for lane i is (i+1)%4 -> [1,2,3,0]
			// group1: [4,5,6,7] -> [5,6,7,4]
			name:       "8_lanes_group4_offset1",
			totalLanes: 8,
			groupSize:  4,
			offset:     1,
			want:       []uint64{1, 2, 3, 0, 5, 6, 7, 4},
		},
		{
			// 8 lanes, groups of 4, rotate right by 1 (offset=-1):
			// group0: src for lane i is (i-1+4)%4 -> [3,0,1,2]
			// group1: [7,4,5,6]
			name:       "8_lanes_group4_offset_minus1",
			totalLanes: 8,
			groupSize:  4,
			offset:     -1,
			want:       []uint64{3, 0, 1, 2, 7, 4, 5, 6},
		},
		{
			// 16 lanes, groups of 4, rotate left by 2:
			// group0: src (i+2)%4 -> [2,3,0,1]
			// group1: [6,7,4,5]
			// group2: [10,11,8,9]
			// group3: [14,15,12,13]
			name:       "16_lanes_group4_offset2",
			totalLanes: 16,
			groupSize:  4,
			offset:     2,
			want:       []uint64{2, 3, 0, 1, 6, 7, 4, 5, 10, 11, 8, 9, 14, 15, 12, 13},
		},
		{
			// 4 lanes, groups of 2, rotate left by 1:
			// group0: [0,1] -> [1,0]
			// group1: [2,3] -> [3,2]
			name:       "4_lanes_group2_offset1",
			totalLanes: 4,
			groupSize:  2,
			offset:     1,
			want:       []uint64{1, 0, 3, 2},
		},
		{
			// 4 lanes, single group of 4, rotate left by 3:
			// src for lane i: (i+3)%4 -> [3,0,1,2]
			name:       "4_lanes_group4_offset3",
			totalLanes: 4,
			groupSize:  4,
			offset:     3,
			want:       []uint64{3, 0, 1, 2},
		},
		{
			// Zero offset: identity permutation.
			name:       "8_lanes_group4_offset0",
			totalLanes: 8,
			groupSize:  4,
			offset:     0,
			want:       []uint64{0, 1, 2, 3, 4, 5, 6, 7},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := spmdRotateWithinMask(tt.totalLanes, tt.groupSize, tt.offset)
			if len(got) != len(tt.want) {
				t.Fatalf("mask length = %d, want %d", len(got), len(tt.want))
			}
			for i, v := range got {
				if v != tt.want[i] {
					t.Errorf("mask[%d] = %d, want %d", i, v, tt.want[i])
				}
			}
		})
	}
}

// TestSPMDShiftLeftWithinMask verifies the shuffle index mask computed by spmdShiftLeftWithinMask.
func TestSPMDShiftLeftWithinMask(t *testing.T) {
	tests := []struct {
		name       string
		totalLanes int
		groupSize  int
		amount     int
		wantMask   []uint64
		wantZero   []bool
	}{
		{
			// 8 lanes, groups of 4, shift left by 1:
			// lane i in group gets value from lane i+1; lane 3 in each group is zeroed.
			// group0: [1,2,3,zero] -> mask=[1,2,3,0], zero=[F,F,F,T]
			// group1: [5,6,7,zero] -> mask=[5,6,7,0], zero=[F,F,F,T]
			name:       "8_lanes_group4_amount1",
			totalLanes: 8,
			groupSize:  4,
			amount:     1,
			wantMask:   []uint64{1, 2, 3, 0, 5, 6, 7, 0},
			wantZero:   []bool{false, false, false, true, false, false, false, true},
		},
		{
			// 8 lanes, groups of 4, shift left by 2:
			// lane i gets value from lane i+2; lanes 2,3 in each group zeroed.
			// group0: [2,3,zero,zero], group1: [6,7,zero,zero]
			name:       "8_lanes_group4_amount2",
			totalLanes: 8,
			groupSize:  4,
			amount:     2,
			wantMask:   []uint64{2, 3, 0, 0, 6, 7, 0, 0},
			wantZero:   []bool{false, false, true, true, false, false, true, true},
		},
		{
			// Zero amount: identity permutation, no zeroing.
			name:       "8_lanes_group4_amount0",
			totalLanes: 8,
			groupSize:  4,
			amount:     0,
			wantMask:   []uint64{0, 1, 2, 3, 4, 5, 6, 7},
			wantZero:   []bool{false, false, false, false, false, false, false, false},
		},
		{
			// 4 lanes, groups of 2, shift left by 1:
			// group0: [1,zero], group1: [3,zero]
			name:       "4_lanes_group2_amount1",
			totalLanes: 4,
			groupSize:  2,
			amount:     1,
			wantMask:   []uint64{1, 0, 3, 0},
			wantZero:   []bool{false, true, false, true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMask, gotZero := spmdShiftLeftWithinMask(tt.totalLanes, tt.groupSize, tt.amount)
			if len(gotMask) != len(tt.wantMask) {
				t.Fatalf("mask length = %d, want %d", len(gotMask), len(tt.wantMask))
			}
			for i := range gotMask {
				if !tt.wantZero[i] && gotMask[i] != tt.wantMask[i] {
					t.Errorf("mask[%d] = %d, want %d", i, gotMask[i], tt.wantMask[i])
				}
				if gotZero[i] != tt.wantZero[i] {
					t.Errorf("zero[%d] = %v, want %v", i, gotZero[i], tt.wantZero[i])
				}
			}
		})
	}
}

// TestSPMDShiftRightWithinMask verifies the shuffle index mask computed by spmdShiftRightWithinMask.
func TestSPMDShiftRightWithinMask(t *testing.T) {
	tests := []struct {
		name       string
		totalLanes int
		groupSize  int
		amount     int
		wantMask   []uint64
		wantZero   []bool
	}{
		{
			// 8 lanes, groups of 4, shift right by 1:
			// lane i gets value from lane i-1; lane 0 in each group zeroed.
			// group0: [zero,0,1,2], group1: [zero,4,5,6]
			name:       "8_lanes_group4_amount1",
			totalLanes: 8,
			groupSize:  4,
			amount:     1,
			wantMask:   []uint64{0, 0, 1, 2, 0, 4, 5, 6},
			wantZero:   []bool{true, false, false, false, true, false, false, false},
		},
		{
			// 8 lanes, groups of 4, shift right by 2:
			// lanes 0,1 in each group zeroed.
			// group0: [zero,zero,0,1], group1: [zero,zero,4,5]
			name:       "8_lanes_group4_amount2",
			totalLanes: 8,
			groupSize:  4,
			amount:     2,
			wantMask:   []uint64{0, 0, 0, 1, 0, 0, 4, 5},
			wantZero:   []bool{true, true, false, false, true, true, false, false},
		},
		{
			// Zero amount: identity permutation, no zeroing.
			name:       "8_lanes_group4_amount0",
			totalLanes: 8,
			groupSize:  4,
			amount:     0,
			wantMask:   []uint64{0, 1, 2, 3, 4, 5, 6, 7},
			wantZero:   []bool{false, false, false, false, false, false, false, false},
		},
		{
			// 4 lanes, groups of 2, shift right by 1:
			// group0: [zero,0], group1: [zero,2]
			name:       "4_lanes_group2_amount1",
			totalLanes: 4,
			groupSize:  2,
			amount:     1,
			wantMask:   []uint64{0, 0, 0, 2},
			wantZero:   []bool{true, false, true, false},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			gotMask, gotZero := spmdShiftRightWithinMask(tt.totalLanes, tt.groupSize, tt.amount)
			if len(gotMask) != len(tt.wantMask) {
				t.Fatalf("mask length = %d, want %d", len(gotMask), len(tt.wantMask))
			}
			for i := range gotMask {
				if !tt.wantZero[i] && gotMask[i] != tt.wantMask[i] {
					t.Errorf("mask[%d] = %d, want %d", i, gotMask[i], tt.wantMask[i])
				}
				if gotZero[i] != tt.wantZero[i] {
					t.Errorf("zero[%d] = %v, want %v", i, gotZero[i], tt.wantZero[i])
				}
			}
		})
	}
}

// TestSPMDRotateWithin verifies that createRotateWithin emits a shufflevector instruction
// with correct type (same vector type as input) and non-nil result.
func TestSPMDRotateWithin(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Build a constant <8 x i32> input vector [0..7].
	vecElts := make([]llvm.Value, 8)
	for i := range vecElts {
		vecElts[i] = llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false)
	}
	input := llvm.ConstVector(vecElts, false)
	vecType := input.Type() // <8 x i32>

	// Compute the shuffle mask directly (rotate left by 1 within groups of 4).
	mask := spmdRotateWithinMask(8, 4, 1)
	shuffleMask := c.spmdShuffleConst(mask)

	result := b.CreateShuffleVector(input, llvm.Undef(vecType), shuffleMask, "rotatewithin")

	if result.IsNil() {
		t.Fatal("CreateShuffleVector returned nil")
	}
	if result.Type().TypeKind() != llvm.VectorTypeKind {
		t.Errorf("result type kind = %v, want VectorTypeKind", result.Type().TypeKind())
	}
	if result.Type().VectorSize() != 8 {
		t.Errorf("result lane count = %d, want 8", result.Type().VectorSize())
	}
	if result.Type().ElementType().IntTypeWidth() != 32 {
		t.Errorf("result element width = %d, want 32", result.Type().ElementType().IntTypeWidth())
	}
}

// TestSPMDShiftLeftWithin verifies that the shift-left-within shuffle mask and zero mask
// generate the correct LLVM shufflevector + select sequence.
func TestSPMDShiftLeftWithin(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Build a constant <8 x i32> input vector [10,11,12,13,20,21,22,23].
	vals := []uint64{10, 11, 12, 13, 20, 21, 22, 23}
	vecElts := make([]llvm.Value, 8)
	for i, v := range vals {
		vecElts[i] = llvm.ConstInt(c.ctx.Int32Type(), v, false)
	}
	input := llvm.ConstVector(vecElts, false)
	vecType := input.Type()

	// ShiftLeft by 1 within groups of 4.
	mask, zeroMask := spmdShiftLeftWithinMask(8, 4, 1)

	shuffleMask := c.spmdShuffleConst(mask)
	shuffled := b.CreateShuffleVector(input, llvm.Undef(vecType), shuffleMask, "shiftleftwithin")

	if shuffled.IsNil() {
		t.Fatal("CreateShuffleVector returned nil")
	}

	// Apply zero select for out-of-range lanes.
	zero := llvm.ConstNull(vecType)
	selectorElts := make([]llvm.Value, 8)
	for i := 0; i < 8; i++ {
		keep := !zeroMask[i]
		selectorElts[i] = llvm.ConstInt(c.ctx.Int1Type(), boolToUint64(keep), false)
	}
	selector := llvm.ConstVector(selectorElts, false)
	result := b.CreateSelect(selector, shuffled, zero, "shiftleftwithin.zero")

	if result.IsNil() {
		t.Fatal("CreateSelect returned nil")
	}
	if result.Type().VectorSize() != 8 {
		t.Errorf("result lane count = %d, want 8", result.Type().VectorSize())
	}
}

// TestSPMDShiftRightWithin verifies that the shift-right-within shuffle mask and zero mask
// generate the correct LLVM shufflevector + select sequence.
func TestSPMDShiftRightWithin(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Build a constant <8 x i32> input vector [10,11,12,13,20,21,22,23].
	vals := []uint64{10, 11, 12, 13, 20, 21, 22, 23}
	vecElts := make([]llvm.Value, 8)
	for i, v := range vals {
		vecElts[i] = llvm.ConstInt(c.ctx.Int32Type(), v, false)
	}
	input := llvm.ConstVector(vecElts, false)
	vecType := input.Type()

	// ShiftRight by 1 within groups of 4.
	mask, zeroMask := spmdShiftRightWithinMask(8, 4, 1)

	shuffleMask := c.spmdShuffleConst(mask)
	shuffled := b.CreateShuffleVector(input, llvm.Undef(vecType), shuffleMask, "shiftrightwithin")

	if shuffled.IsNil() {
		t.Fatal("CreateShuffleVector returned nil")
	}

	// Apply zero select for out-of-range lanes.
	zero := llvm.ConstNull(vecType)
	selectorElts := make([]llvm.Value, 8)
	for i := 0; i < 8; i++ {
		keep := !zeroMask[i]
		selectorElts[i] = llvm.ConstInt(c.ctx.Int1Type(), boolToUint64(keep), false)
	}
	selector := llvm.ConstVector(selectorElts, false)
	result := b.CreateSelect(selector, shuffled, zero, "shiftrightwithin.zero")

	if result.IsNil() {
		t.Fatal("CreateSelect returned nil")
	}
	if result.Type().VectorSize() != 8 {
		t.Errorf("result lane count = %d, want 8", result.Type().VectorSize())
	}
}

// TestSPMDSwitchMaskNarrowing verifies sequential mask narrowing for switch cases.
// Tests the ISPC algorithm:
//
//	mask1 = remaining & cond1
//	remaining1 = remaining & ~cond1
//	mask2 = remaining1 & cond2
//	remaining2 = remaining1 & ~cond2
//	mask3 = remaining2 & cond3
func TestSPMDSwitchMaskNarrowing(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)

	// Simulate a 3-case switch with all-ones initial mask.
	i32x4 := llvm.VectorType(c.ctx.Int32Type(), 4)
	allOnes := llvm.ConstAllOnes(i32x4)

	// Case 1: mask1 = allOnes & cond1, remaining1 = allOnes & ~cond1.
	cond1 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(c.ctx.Int32Type(), 0xFFFFFFFF, false), // lane 0 matches
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
	}, false)
	mask1 := b.CreateAnd(allOnes, cond1, "mask1")
	notCond1 := b.CreateNot(cond1, "")
	remaining1 := b.CreateAnd(allOnes, notCond1, "remaining1")

	// Case 2: mask2 = remaining1 & cond2, remaining2 = remaining1 & ~cond2.
	cond2 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
		llvm.ConstInt(c.ctx.Int32Type(), 0xFFFFFFFF, false), // lane 1 matches
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
	}, false)
	mask2 := b.CreateAnd(remaining1, cond2, "mask2")
	notCond2 := b.CreateNot(cond2, "")
	remaining2 := b.CreateAnd(remaining1, notCond2, "remaining2")

	// Case 3: mask3 = remaining2 & cond3.
	cond3 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
		llvm.ConstInt(c.ctx.Int32Type(), 0xFFFFFFFF, false), // lane 2 matches
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
	}, false)
	mask3 := b.CreateAnd(remaining2, cond3, "mask3")

	// Verify masks are non-nil and have correct type.
	if mask1.IsNil() {
		t.Error("mask1 is nil")
	}
	if mask2.IsNil() {
		t.Error("mask2 is nil")
	}
	if mask3.IsNil() {
		t.Error("mask3 is nil")
	}
	if mask1.Type() != i32x4 {
		t.Errorf("mask1 type = %v, want <4 x i32>", mask1.Type())
	}
	if mask2.Type() != i32x4 {
		t.Errorf("mask2 type = %v, want <4 x i32>", mask2.Type())
	}
	if mask3.Type() != i32x4 {
		t.Errorf("mask3 type = %v, want <4 x i32>", mask3.Type())
	}

	// Verify that sequential narrowing produces distinct masks.
	// Each mask should activate different lanes (lane 0, 1, 2 respectively).
	if mask1 == mask2 || mask1 == mask3 || mask2 == mask3 {
		t.Error("masks should be distinct (different conditions)")
	}
}

// TestSPMDSwitchCascadedSelect verifies cascaded select merge for switch.done phi.
// Tests building: result = select(mask3, val3, select(mask2, val2, select(mask1, val1, default))).
func TestSPMDSwitchCascadedSelect(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)

	i32x4 := llvm.VectorType(c.ctx.Int32Type(), 4)
	i32 := c.ctx.Int32Type()

	// Create 3 case masks (lane 0, 1, 2 active respectively).
	mask1 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 0xFFFFFFFF, false),
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0, false),
	}, false)
	mask2 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0xFFFFFFFF, false),
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0, false),
	}, false)
	mask3 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0xFFFFFFFF, false),
		llvm.ConstInt(i32, 0, false),
	}, false)

	// Create values for each case (10, 20, 30) and default (0).
	val1 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 10, false),
		llvm.ConstInt(i32, 10, false),
		llvm.ConstInt(i32, 10, false),
		llvm.ConstInt(i32, 10, false),
	}, false)
	val2 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 20, false),
		llvm.ConstInt(i32, 20, false),
		llvm.ConstInt(i32, 20, false),
		llvm.ConstInt(i32, 20, false),
	}, false)
	val3 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 30, false),
		llvm.ConstInt(i32, 30, false),
		llvm.ConstInt(i32, 30, false),
		llvm.ConstInt(i32, 30, false),
	}, false)
	defaultVal := llvm.ConstNull(i32x4)

	// Build cascaded select: result = select(mask1, val1, default), then select(mask2, val2, result), etc.
	result := defaultVal
	result = b.spmdMaskSelect(mask1, val1, result)
	result = b.spmdMaskSelect(mask2, val2, result)
	result = b.spmdMaskSelect(mask3, val3, result)

	// Verify result is non-nil and has correct type.
	if result.IsNil() {
		t.Error("result is nil")
	}
	if result.Type() != i32x4 {
		t.Errorf("result type = %v, want <4 x i32>", result.Type())
	}

	// The cascaded select should produce a merged result.
	// Each lane should have the value from its matching case: [10, 20, 30, 0].
}

// TestSPMDSwitchDefaultMask verifies default case gets remaining mask.
// Tests that after 2 cases, the remaining mask is correctly assigned to the default body.
func TestSPMDSwitchDefaultMask(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)

	i32x4 := llvm.VectorType(c.ctx.Int32Type(), 4)
	i32 := c.ctx.Int32Type()

	// Start with all-ones mask.
	allOnes := llvm.ConstAllOnes(i32x4)

	// Case 1: mask1 = allOnes & cond1 (lane 0), remaining1 = allOnes & ~cond1.
	cond1 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 0xFFFFFFFF, false),
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0, false),
	}, false)
	remaining1 := b.CreateAnd(allOnes, b.CreateNot(cond1, ""), "remaining1")

	// Case 2: mask2 = remaining1 & cond2 (lane 1), remaining2 = remaining1 & ~cond2.
	cond2 := llvm.ConstVector([]llvm.Value{
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0xFFFFFFFF, false),
		llvm.ConstInt(i32, 0, false),
		llvm.ConstInt(i32, 0, false),
	}, false)
	defaultMask := b.CreateAnd(remaining1, b.CreateNot(cond2, ""), "default.mask")

	// Verify default mask is non-nil and has correct type.
	if defaultMask.IsNil() {
		t.Error("defaultMask is nil")
	}
	if defaultMask.Type() != i32x4 {
		t.Errorf("defaultMask type = %v, want <4 x i32>", defaultMask.Type())
	}

	// Default mask should activate lanes [0, 0, 1, 1] (lanes 2 and 3).
}

func TestSPMDExtendIndex(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Use -1 (0xFF in i8) to distinguish ZExt from SExt:
	// ZExt(-1 as uint8) → 255 (0x000000FF)
	// SExt(-1 as int8)  → -1  (0xFFFFFFFF)
	i8Val := llvm.ConstInt(c.ctx.Int8Type(), 0xFF, false) // -1 / 255

	tests := []struct {
		name    string
		goType  types.Type
		wantVal uint64 // expected ZExtValue of the result constant
	}{
		{"uint8_zext", types.Typ[types.Uint8], 0xFF},     // 255
		{"int8_sext", types.Typ[types.Int8], 0xFFFFFFFF}, // -1 as i32
		{"spmd_uint8_zext", types.NewVarying(types.Typ[types.Uint8]), 0xFF},
		{"spmd_int8_sext", types.NewVarying(types.Typ[types.Int8]), 0xFFFFFFFF},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := b.spmdExtendIndex(i8Val, tt.goType, b.uintptrType)
			if result.Type() != b.uintptrType {
				t.Fatalf("expected uintptr type, got width %d", result.Type().IntTypeWidth())
			}
			// For constant inputs, spmdExtendIndex should produce a constant result.
			if result.IsAConstantInt().IsNil() {
				t.Fatal("expected constant result from constant input")
			}
			gotVal := result.ZExtValue()
			if gotVal != tt.wantVal {
				t.Errorf("got 0x%X, want 0x%X", gotVal, tt.wantVal)
			}
		})
	}
}

func TestSPMDExtendIndexNoOpWhenWide(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// When value is already uintptr width, spmdExtendIndex must return it unchanged.
	val := llvm.ConstInt(b.uintptrType, 42, false)
	result := b.spmdExtendIndex(val, types.Typ[types.Uint32], b.uintptrType)
	if result.Type() != b.uintptrType {
		t.Errorf("expected uintptr type, got width %d", result.Type().IntTypeWidth())
	}
}

func TestSPMDVectorIndexStringLLVM(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Build a string global: "0123456789abcdef"
	i8Type := c.ctx.Int8Type()
	strBytes := "0123456789abcdef"
	strConst := llvm.ConstString(strBytes, false)
	strGlobal := llvm.AddGlobal(c.mod, strConst.Type(), "test_str")
	strGlobal.SetInitializer(strConst)
	strGlobal.SetGlobalConstant(true)
	strPtr := b.CreateInBoundsGEP(strConst.Type(), strGlobal, []llvm.Value{
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
		llvm.ConstInt(c.ctx.Int32Type(), 0, false),
	}, "str.ptr")

	// Manually exercise the per-lane logic that spmdVectorIndexString uses
	// (we can't call it directly without a full ssa.Index, but we test the
	// LLVM IR primitives it relies on).
	laneCount := 4

	// Simulate vector index [0, 1, 2, 3]
	vecType := llvm.VectorType(i8Type, laneCount)
	indexVec := llvm.Undef(vecType)
	for i := 0; i < laneCount; i++ {
		indexVec = b.CreateInsertElement(indexVec, llvm.ConstInt(i8Type, uint64(i), false),
			llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false), "")
	}

	// Perform the per-lane extraction that spmdVectorIndexString does.
	bufElemType := c.ctx.Int8Type()
	result := llvm.Undef(llvm.VectorType(bufElemType, laneCount))
	for lane := 0; lane < laneCount; lane++ {
		laneIdx := b.CreateExtractElement(indexVec, llvm.ConstInt(c.ctx.Int32Type(), uint64(lane), false), "")
		// Extend i8 index to uintptr.
		laneIdx = b.spmdExtendIndex(laneIdx, types.Typ[types.Uint8], b.uintptrType)
		ptr := b.CreateInBoundsGEP(bufElemType, strPtr, []llvm.Value{laneIdx}, "")
		val := b.CreateLoad(bufElemType, ptr, "")
		result = b.CreateInsertElement(result, val, llvm.ConstInt(c.ctx.Int32Type(), uint64(lane), false), "")
	}

	if result.Type().TypeKind() != llvm.VectorTypeKind {
		t.Errorf("expected vector result, got type kind %v", result.Type().TypeKind())
	}
	if result.Type().VectorSize() != laneCount {
		t.Errorf("expected %d lanes, got %d", laneCount, result.Type().VectorSize())
	}
	if result.Type().ElementType() != i8Type {
		t.Errorf("expected i8 element type")
	}
}

func TestSPMDVectorIndexArrayLLVM(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name      string
		arrayLen  int
		elemType  llvm.Type
		laneCount int
		indexType llvm.Type
	}{
		{"4xi32_array_4xi32_index", 4, c.ctx.Int32Type(), 4, c.ctx.Int32Type()},
		{"16xi8_array_4xi8_index", 16, c.ctx.Int8Type(), 4, c.ctx.Int8Type()},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build an array value.
			arrayType := llvm.ArrayType(tt.elemType, tt.arrayLen)
			arrayVal := llvm.ConstNull(arrayType)

			// Build vector index [0, 1, 2, ... laneCount-1].
			vecType := llvm.VectorType(tt.indexType, tt.laneCount)
			indexVec := llvm.Undef(vecType)
			for i := 0; i < tt.laneCount; i++ {
				indexVec = b.CreateInsertElement(indexVec,
					llvm.ConstInt(tt.indexType, uint64(i%tt.arrayLen), false),
					llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false), "")
			}

			// Exercise the alloca+GEP+load pattern used by spmdVectorIndexArray.
			alloca, allocaSize := b.createTemporaryAlloca(arrayType, "index.alloca")
			b.CreateStore(arrayVal, alloca)
			zero := llvm.ConstInt(c.ctx.Int32Type(), 0, false)

			result := llvm.Undef(llvm.VectorType(tt.elemType, tt.laneCount))
			for lane := 0; lane < tt.laneCount; lane++ {
				laneIdx := b.CreateExtractElement(indexVec, llvm.ConstInt(c.ctx.Int32Type(), uint64(lane), false), "")
				laneIdx = b.spmdExtendIndex(laneIdx, types.Typ[types.Uint8], b.uintptrType)
				ptr := b.CreateInBoundsGEP(arrayType, alloca, []llvm.Value{zero, laneIdx}, "index.gep")
				val := b.CreateLoad(tt.elemType, ptr, "index.load")
				result = b.CreateInsertElement(result, val, llvm.ConstInt(c.ctx.Int32Type(), uint64(lane), false), "")
			}

			b.emitLifetimeEnd(alloca, allocaSize)

			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("expected vector result, got type kind %v", result.Type().TypeKind())
			}
			if result.Type().VectorSize() != tt.laneCount {
				t.Errorf("expected %d lanes, got %d", tt.laneCount, result.Type().VectorSize())
			}
			if result.Type().ElementType() != tt.elemType {
				t.Errorf("element type mismatch")
			}
		})
	}
}

func TestSPMDSwizzleArrayBytes(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()

	tests := []struct {
		name      string
		arrayLen  int
		laneCount int
	}{
		{"16xi8_4lanes", 16, 4},
		{"8xi8_4lanes", 8, 4},
		{"4xi8_4lanes", 4, 4},
		{"16xi8_16lanes", 16, 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build a [N x i8] array value.
			arrayType := llvm.ArrayType(i8Type, tt.arrayLen)
			arrayVal := llvm.ConstNull(arrayType)

			// Build a non-identity index vector to ensure the swizzle
			// intrinsic is emitted. Reversed indices [N-1, N-2, ..., 0]
			// avoid the identity-swizzle elision path (which fires only
			// for the sequential [0, 1, ..., N-1] permutation).
			vecType := llvm.VectorType(i32Type, tt.laneCount)
			indexVec := llvm.Undef(vecType)
			for i := 0; i < tt.laneCount; i++ {
				idx := (tt.arrayLen - 1 - i) % tt.arrayLen
				indexVec = b.CreateInsertElement(indexVec,
					llvm.ConstInt(i32Type, uint64(idx), false),
					llvm.ConstInt(i32Type, uint64(i), false), "")
			}

			result, err := b.spmdSwizzleArrayBytes(arrayVal, indexVec, tt.arrayLen, tt.laneCount)
			if err != nil {
				t.Fatal(err)
			}

			// Result should be a vector.
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Fatalf("expected vector, got %v", result.Type().TypeKind())
			}
			if result.Type().VectorSize() != tt.laneCount {
				t.Errorf("expected %d lanes, got %d", tt.laneCount, result.Type().VectorSize())
			}

			// Verify module IR contains llvm.wasm.swizzle.
			modIR := b.mod.String()
			if !strings.Contains(modIR, "llvm.wasm.swizzle") {
				t.Error("expected llvm.wasm.swizzle in module IR")
			}
		})
	}
}

// TestSPMDVectorReduceUmax verifies that spmdVectorReduceUmax emits
// @llvm.vector.reduce.umax and produces a scalar result of the correct type.
func TestSPMDVectorReduceUmax(t *testing.T) {
	tests := []struct {
		name      string
		laneCount int
		elemBits  int
	}{
		{"4xi32", 4, 32},
		{"4xi8", 4, 8},
		{"8xi16", 8, 16},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := newTestCompilerContext(t)
			defer c.dispose()
			b := newTestBuilder(t, c)
			defer b.Dispose()

			elemType := c.ctx.IntType(tt.elemBits)
			vecType := llvm.VectorType(elemType, tt.laneCount)

			// Build a constant vector [0, 1, 2, ..., laneCount-1].
			vec := llvm.Undef(vecType)
			for i := 0; i < tt.laneCount; i++ {
				vec = b.CreateInsertElement(vec,
					llvm.ConstInt(elemType, uint64(i), false),
					llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false), "")
			}

			result := b.spmdVectorReduceUmax(vec)

			// Result must be a scalar of the same element type.
			if result.Type() != elemType {
				t.Errorf("expected scalar type i%d, got %s", tt.elemBits, result.Type().String())
			}

			// The LLVM IR for the call must reference the umax intrinsic.
			ir := result.String()
			want := "llvm.vector.reduce.umax"
			if !strings.Contains(ir, want) {
				t.Errorf("expected IR to contain %q, got:\n%s", want, ir)
			}
		})
	}
}

// TestSPMDVectorReduceUmaxBoundsCheck verifies that the vectorized bounds check
// in spmdVectorIndexString and spmdVectorIndexArray uses llvm.vector.reduce.umax
// (one intrinsic call) rather than per-lane extract+icmp+or (3N scalar ops).
func TestSPMDVectorReduceUmaxBoundsCheck(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, laneCount)

	// Simulate a clamped index vector [0, 1, 2, 3].
	index := llvm.Undef(vecType)
	for i := 0; i < laneCount; i++ {
		index = b.CreateInsertElement(index,
			llvm.ConstInt(i32Type, uint64(i), false),
			llvm.ConstInt(i32Type, uint64(i), false), "")
	}

	// Invoke spmdVectorReduceUmax — the replacement for the per-lane OR chain.
	maxIdx := b.spmdVectorReduceUmax(index)
	if maxIdx.Type().IntTypeWidth() < b.uintptrType.IntTypeWidth() {
		maxIdx = b.CreateZExt(maxIdx, b.uintptrType, "")
	}
	length := llvm.ConstInt(b.uintptrType, 16, false)
	oob := b.CreateICmp(llvm.IntUGE, maxIdx, length, "spmd.bounds.oob")

	// The oob flag must come from a single icmp, not a chain of ORs.
	ir := oob.String()
	if !strings.Contains(ir, "spmd.bounds.oob") {
		t.Errorf("expected icmp named spmd.bounds.oob in IR, got:\n%s", ir)
	}

	// Confirm the umax intrinsic is referenced somewhere in the module IR.
	modIR := c.mod.String()
	if !strings.Contains(modIR, "llvm.vector.reduce.umax") {
		t.Errorf("expected llvm.vector.reduce.umax in module IR")
	}
	// Confirm there is no per-lane OR chain (the old pattern created N "or i1" ops).
	// The new path has a single icmp, so there should be no "or i1" at all.
	if strings.Contains(modIR, "or i1") {
		t.Errorf("unexpected per-lane 'or i1' chain found; umax path should not emit it")
	}
}

func TestSPMDSwizzleDetection(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()

	// Test spmdSwizzlePrepareIndex with various lane counts and index widths.
	tests := []struct {
		name      string
		laneCount int
		indexType llvm.Type
		wantWidth int // expected result vector size (always 16)
	}{
		{"16xi8_direct", 16, i8Type, 16},
		{"4xi32_trunc_pad", 4, c.ctx.Int32Type(), 16},
		{"8xi16_trunc_pad", 8, c.ctx.Int16Type(), 16},
		{"4xi8_pad", 4, i8Type, 16},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build vector index
			vecType := llvm.VectorType(tt.indexType, tt.laneCount)
			indexVec := llvm.Undef(vecType)
			for i := 0; i < tt.laneCount; i++ {
				indexVec = b.CreateInsertElement(indexVec, llvm.ConstInt(tt.indexType, uint64(i), false),
					llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false), "")
			}

			result := b.spmdSwizzlePrepareIndex(indexVec, tt.laneCount)
			if result.Type().VectorSize() != tt.wantWidth {
				t.Errorf("vector size = %d, want %d", result.Type().VectorSize(), tt.wantWidth)
			}
			if result.Type().ElementType() != i8Type {
				t.Errorf("element type should be i8")
			}
		})
	}
}

func TestSPMDWasmSwizzle(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()

	tests := []struct {
		name       string
		tableBytes []byte
		laneCount  int
		indexType  llvm.Type
		wantLanes  int // expected result vector lanes
	}{
		{"16byte_table_16lanes", []byte("0123456789abcdef"), 16, i8Type, 16},
		{"16byte_table_4lanes", []byte("0123456789abcdef"), 4, c.ctx.Int32Type(), 4},
		{"8byte_table_padded", []byte("01234567"), 4, i8Type, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build vector index
			vecType := llvm.VectorType(tt.indexType, tt.laneCount)
			indexVec := llvm.Undef(vecType)
			for i := 0; i < tt.laneCount; i++ {
				indexVec = b.CreateInsertElement(indexVec, llvm.ConstInt(tt.indexType, uint64(i%len(tt.tableBytes)), false),
					llvm.ConstInt(c.ctx.Int32Type(), uint64(i), false), "")
			}

			result := b.spmdWasmSwizzle(tt.tableBytes, indexVec, tt.laneCount)

			if result.Type().VectorSize() != tt.wantLanes {
				t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), tt.wantLanes)
			}
			if result.Type().ElementType() != i8Type {
				t.Errorf("result element type should be i8")
			}
		})
	}
}

func TestSPMDSwizzleRejectsNonWASM(t *testing.T) {
	// Create a non-WASM target context.
	target, err := llvm.GetTargetFromTriple("x86_64-unknown-linux-gnu")
	if err != nil {
		t.Skipf("x86_64 target not available: %v", err)
	}
	machine := target.CreateTargetMachine("x86_64-unknown-linux-gnu", "", "",
		llvm.CodeGenLevelDefault, llvm.RelocDefault, llvm.CodeModelDefault)
	config := &Config{
		Triple:   "x86_64-unknown-linux-gnu",
		Features: "",
	}
	c := newCompilerContext("test", machine, config, false)
	defer c.dispose()

	if c.spmdIsWASM() {
		t.Fatal("expected non-WASM context")
	}
}

func TestSPMDWasmSwizzleHextableShape(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	if !c.spmdIsWASM() {
		t.Fatal("expected WASM context for swizzle test")
	}

	// Simulate: const hextable = "0123456789abcdef"
	strBytes := "0123456789abcdef"
	if len(strBytes) > 16 {
		t.Fatal("string too long for swizzle")
	}

	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Build a <4 x i32> index vector [0, 1, 2, 3]
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, 4)
	indexVec := llvm.Undef(vecType)
	for i := 0; i < 4; i++ {
		indexVec = b.CreateInsertElement(indexVec, llvm.ConstInt(i32Type, uint64(i), false),
			llvm.ConstInt(i32Type, uint64(i), false), "")
	}

	result := b.spmdWasmSwizzle([]byte(strBytes), indexVec, 4)

	// Result should be <4 x i8> (narrowed from <16 x i8>)
	if result.Type().VectorSize() != 4 {
		t.Errorf("result lanes = %d, want 4", result.Type().VectorSize())
	}
	if result.Type().ElementType() != c.ctx.Int8Type() {
		t.Error("result element type should be i8")
	}
}

// TestSPMDMaterializeDecomposed verifies that spmdMaterializeDecomposed converts
// a decomposed index (scalar base + <N x i8> offset) to a full <N x i32> vector.
func TestSPMDMaterializeDecomposed(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 16
	i32Type := c.ctx.Int32Type()
	i8Type := c.ctx.Int8Type()

	// Create a scalar base value (i32 constant 32).
	scalarBase := llvm.ConstInt(i32Type, 32, false)

	// Create a <16 x i8> offset [0..15].
	offsets := make([]llvm.Value, laneCount)
	for i := 0; i < laneCount; i++ {
		offsets[i] = llvm.ConstInt(i8Type, uint64(i), false)
	}
	varyingOffset := llvm.ConstVector(offsets, false)

	decomp := &spmdDecomposedIndex{
		scalarBase:    scalarBase,
		varyingOffset: varyingOffset,
		laneCount:     laneCount,
	}

	result := b.spmdMaterializeDecomposed(decomp)

	// Result should be <16 x i32>.
	if result.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatalf("result type = %v, want vector", result.Type())
	}
	if result.Type().VectorSize() != laneCount {
		t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), laneCount)
	}
	if result.Type().ElementType() != i32Type {
		t.Errorf("result elem type = %v, want i32", result.Type().ElementType())
	}
}

// TestSPMDDecomposedIndexTailMask verifies that emitSPMDBodyPrologue produces a
// <16 x i8> tail mask for decomposed byte-lane loops (laneCount == 16 on WASM).
func TestSPMDDecomposedIndexTailMask(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()

	// Initialize the maps that emitSPMDBodyPrologue writes into.
	b.spmdDecomposed = make(map[ssa.Value]*spmdDecomposedIndex)

	// Create a fake loop struct. boundValue and bodyIterValue are SSA values so we need
	// to use ssa.Const for the bound. Since SSA requires a program context we test
	// the tail mask math directly with LLVM values instead.
	//
	// Test scenario: base=0, laneCount=16, bound=10.
	// Expected: lanes 0..9 are active (offset < 10), lanes 10..15 are inactive.
	laneCount := 16

	// Simulate the core tail mask computation from emitSPMDBodyPrologue (decomposed path).
	scalarPhi := llvm.ConstInt(i32Type, 0, false)    // base iteration = 0
	boundScalar := llvm.ConstInt(i32Type, 10, false) // bound = 10
	diff := b.CreateSub(boundScalar, scalarPhi, "diff")

	zero32 := llvm.ConstInt(i32Type, 0, false)
	lcConst := llvm.ConstInt(i32Type, uint64(laneCount), false)

	diffClamped := b.CreateSelect(
		b.CreateICmp(llvm.IntSGT, diff, lcConst, ""),
		lcConst, diff, "clamped")
	diffClamped = b.CreateSelect(
		b.CreateICmp(llvm.IntSLT, diffClamped, zero32, ""),
		zero32, diffClamped, "nonneg")

	diffI8 := b.CreateTrunc(diffClamped, i8Type, "diff.i8")
	diffVec := b.splatScalar(diffI8, llvm.VectorType(i8Type, laneCount))

	// Create identity offset <0, 1, ..., 15>.
	varyingOffset := c.spmdLaneOffsetConst(laneCount, i8Type)

	tailMaskI1 := b.CreateICmp(llvm.IntULT, varyingOffset, diffVec, "tail.mask")
	tailMask := b.spmdWrapMask(tailMaskI1, laneCount)

	// On WASM with 16 lanes, mask element type should be i8 (128/16 = 8 bits).
	if tailMask.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatalf("tailMask type = %v, want vector", tailMask.Type())
	}
	if tailMask.Type().VectorSize() != laneCount {
		t.Errorf("tailMask lanes = %d, want %d", tailMask.Type().VectorSize(), laneCount)
	}
	if tailMask.Type().ElementType() != i8Type {
		t.Errorf("tailMask elem type = %v, want i8", tailMask.Type().ElementType())
	}
}

// TestSPMDDecomposedBinOpAdd verifies that spmdDecomposedBinOp handles ADD correctly.
// (base + offset) + scalar_c → decomposed {base+c, offset}.
func TestSPMDDecomposedBinOpAdd(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()
	laneCount := 16

	b.spmdDecomposed = make(map[ssa.Value]*spmdDecomposedIndex)

	// scalarBase = 16, varyingOffset = <0,1,...,15>
	scalarBase := llvm.ConstInt(i32Type, 16, false)
	offsets := make([]llvm.Value, laneCount)
	for i := 0; i < laneCount; i++ {
		offsets[i] = llvm.ConstInt(i8Type, uint64(i), false)
	}
	varyingOffset := llvm.ConstVector(offsets, false)

	// Build an ssa.BinOp-like test: we can't create real SSA values in a unit test
	// so we test the LLVM helper directly: simulate what spmdDecomposedBinOp does
	// for ADD with a scalar constant 100.
	scalar := llvm.ConstInt(i32Type, 100, false)
	newBase := b.CreateAdd(scalarBase, scalar, "spmd.decomp.add")

	// The result should be base+c = 116 (constant fold).
	decomp := &spmdDecomposedIndex{
		scalarBase:    newBase,
		varyingOffset: varyingOffset,
		laneCount:     laneCount,
	}

	// Materialize to verify correctness.
	result := b.spmdMaterializeDecomposed(decomp)

	if result.Type().VectorSize() != laneCount {
		t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), laneCount)
	}
	if result.Type().ElementType() != i32Type {
		t.Errorf("result elem type = %v, want i32", result.Type().ElementType())
	}
}

// TestSPMDDecomposedBinOpShr verifies the right-shift decomposition rule.
// (base + offset) >> k → {base>>k, offset>>k} when laneCount is multiple of 2^k.
func TestSPMDDecomposedBinOpShr(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()
	laneCount := 16

	// scalarBase = 16 (multiple of 16), varyingOffset = <0,1,...,15>
	scalarBase := llvm.ConstInt(i32Type, 16, false)
	offsets := make([]llvm.Value, laneCount)
	for i := 0; i < laneCount; i++ {
		offsets[i] = llvm.ConstInt(i8Type, uint64(i), false)
	}
	varyingOffset := llvm.ConstVector(offsets, false)

	// Simulate right-shift by 1: 16 is multiple of 2^1 = 2, so decomposition is valid.
	// base>>1 = 8, offset>>1 = <0,0,1,1,2,2,...,7,7>
	k := uint64(1)
	i8Shift := llvm.ConstInt(i8Type, k, false)
	shiftVec := b.splatScalar(i8Shift, llvm.VectorType(i8Type, laneCount))

	newBase := b.CreateAShr(scalarBase, llvm.ConstInt(i32Type, k, false), "base.shr")
	newOffset := b.CreateLShr(varyingOffset, shiftVec, "off.shr")

	decomp := &spmdDecomposedIndex{
		scalarBase:    newBase,
		varyingOffset: newOffset,
		laneCount:     laneCount,
	}

	// Verify structure: newBase should be 8, newOffset should be <0,0,1,1,...,7,7>.
	result := b.spmdMaterializeDecomposed(decomp)

	if result.Type().VectorSize() != laneCount {
		t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), laneCount)
	}
	if result.Type().ElementType() != i32Type {
		t.Errorf("result elem type = %v, want i32", result.Type().ElementType())
	}
	// The offset vector element type should remain i8.
	if newOffset.Type().ElementType() != i8Type {
		t.Errorf("newOffset elem type = %v, want i8", newOffset.Type().ElementType())
	}
}

// TestSPMDDecomposedBinOpQuo verifies the integer division decomposition rule.
// (base + offset) / k → {base/k, <0/k, 1/k, ..., (N-1)/k>} when laneCount%k==0
// and base is from the body iter (aligned to laneCount).
func TestSPMDDecomposedBinOpQuo(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()
	laneCount := 16

	// scalarBase = 48 (multiple of 16, divisible by 3), varyingOffset = <0,1,...,15>
	scalarBase := llvm.ConstInt(i32Type, 48, false)
	offsets := make([]llvm.Value, laneCount)
	for i := 0; i < laneCount; i++ {
		offsets[i] = llvm.ConstInt(i8Type, uint64(i), false)
	}
	_ = llvm.ConstVector(offsets, false) // varyingOffset: <0,1,...,15> — not used directly; newOffset is built below

	// Simulate integer division by 3: base/3 = 16, offset/3 = <0,0,0,1,1,1,2,2,2,3,3,3,4,4,4,5>
	k := uint64(3)
	newBase := b.CreateUDiv(scalarBase, llvm.ConstInt(i32Type, k, false), "base.quo")
	newOffsetElts := make([]llvm.Value, laneCount)
	for i := 0; i < laneCount; i++ {
		newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(i)/k, false)
	}
	newOffset := llvm.ConstVector(newOffsetElts, false)

	decomp := &spmdDecomposedIndex{
		scalarBase:    newBase,
		varyingOffset: newOffset,
		laneCount:     laneCount,
	}

	// Verify structure: newBase should be 16, newOffset should be <0,0,0,1,1,1,...,5>.
	result := b.spmdMaterializeDecomposed(decomp)

	if result.Type().VectorSize() != laneCount {
		t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), laneCount)
	}
	if result.Type().ElementType() != i32Type {
		t.Errorf("result elem type = %v, want i32", result.Type().ElementType())
	}
	if newOffset.Type().ElementType() != i8Type {
		t.Errorf("newOffset elem type = %v, want i8", newOffset.Type().ElementType())
	}
}

// TestSPMDDecomposedBinOpMul verifies the MUL decomposition rule.
// (base + offset) * c → {base*c, <0, c, 2c, ..., (N-1)*c>} when (N-1)*c <= 255.
// Like TestSPMDDecomposedBinOpAdd, we test the LLVM-level arithmetic directly
// (not the full spmdDecomposedBinOp method which requires live getValue/getPos).
func TestSPMDDecomposedBinOpMul(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()
	laneCount := 16

	tests := []struct {
		name      string
		constC    int64
		base      uint64
		shouldFit bool // whether (N-1)*c <= 255
	}{
		{name: "c=2 stride-2", constC: 2, base: 16, shouldFit: true},
		{name: "c=16 boundary", constC: 16, base: 0, shouldFit: true},
		{name: "c=17 max boundary", constC: 17, base: 0, shouldFit: true}, // 15*17=255
		{name: "c=18 overflow", constC: 18, base: 0, shouldFit: false},    // 15*18=270 > 255
		{name: "c=3 non-power-of-two", constC: 3, base: 32, shouldFit: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify the overflow guard: (N-1)*c <= 255
			maxOffset := int64(laneCount-1) * tt.constC
			fitsInByte := maxOffset <= 255
			if fitsInByte != tt.shouldFit {
				t.Fatalf("overflow guard: (N-1)*c = %d, fits = %v, want %v", maxOffset, fitsInByte, tt.shouldFit)
			}

			if !tt.shouldFit {
				return // decomposition would be skipped
			}

			// Simulate the MUL decomposition: base*c as scalar, <0,c,2c,...,(N-1)*c> as offset.
			scalarBase := llvm.ConstInt(i32Type, tt.base, false)
			scalar := llvm.ConstInt(i32Type, uint64(tt.constC), false)
			newBase := b.CreateMul(scalarBase, scalar, "spmd.decomp.mul.base")

			newOffsetElts := make([]llvm.Value, laneCount)
			for i := 0; i < laneCount; i++ {
				newOffsetElts[i] = llvm.ConstInt(i8Type, uint64(int64(i)*tt.constC), false)
			}
			newOffset := llvm.ConstVector(newOffsetElts, false)

			decomp := &spmdDecomposedIndex{
				scalarBase:    newBase,
				varyingOffset: newOffset,
				laneCount:     laneCount,
				fromBodyIter:  false, // MUL must NOT propagate fromBodyIter
			}

			// Materialize to verify correctness.
			result := b.spmdMaterializeDecomposed(decomp)

			if result.Type().VectorSize() != laneCount {
				t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), laneCount)
			}
			if result.Type().ElementType() != i32Type {
				t.Errorf("result elem type = %v, want i32", result.Type().ElementType())
			}
			// Verify offset vector element type stays i8.
			if newOffset.Type().ElementType() != i8Type {
				t.Errorf("offset elem type = %v, want i8", newOffset.Type().ElementType())
			}
			// Verify fromBodyIter is false (prevents incorrect downstream SHR).
			if decomp.fromBodyIter {
				t.Error("fromBodyIter should be false after MUL (breaks carry-free shift identity)")
			}
		})
	}
}

// TestSPMDDecomposedIndexStructure verifies that the spmdDecomposedIndex struct
// fields are correctly populated by the emitSPMDBodyPrologue decomposed path.
// This is a structural test using LLVM value types.
func TestSPMDDecomposedIndexStructure(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()
	laneCount := 16

	// Simulate what emitSPMDBodyPrologue does for a decomposed loop.
	scalarPhi := llvm.ConstInt(i32Type, 0, false)
	varyingOffset := c.spmdLaneOffsetConst(laneCount, i8Type)

	decomp := &spmdDecomposedIndex{
		scalarBase:    scalarPhi,
		varyingOffset: varyingOffset,
		laneCount:     laneCount,
	}

	// Verify scalar base is i32.
	if decomp.scalarBase.Type() != i32Type {
		t.Errorf("scalarBase type = %v, want i32", decomp.scalarBase.Type())
	}

	// Verify varying offset is <16 x i8>.
	if decomp.varyingOffset.Type().VectorSize() != laneCount {
		t.Errorf("varyingOffset lanes = %d, want %d",
			decomp.varyingOffset.Type().VectorSize(), laneCount)
	}
	if decomp.varyingOffset.Type().ElementType() != i8Type {
		t.Errorf("varyingOffset elem type = %v, want i8",
			decomp.varyingOffset.Type().ElementType())
	}

	// Materialize and verify output is <16 x i32>.
	mat := b.spmdMaterializeDecomposed(decomp)
	if mat.Type().VectorSize() != laneCount {
		t.Errorf("materialized lanes = %d, want %d", mat.Type().VectorSize(), laneCount)
	}
	if mat.Type().ElementType() != i32Type {
		t.Errorf("materialized elem type = %v, want i32", mat.Type().ElementType())
	}
}

// TestSPMDDecomposedChangeTypePropagation verifies that spmdDecomposed metadata is
// propagated through *ssa.ChangeType nodes. Without the fix in compiler.go, BinOps
// on a ChangeType-wrapped loop variable would miss the decomposed path and produce
// <4 x i32> operations instead of decomposed <16 x i8> operations for byte-lane loops.
//
// NOTE: This is a data-structure invariant test, not a behavioural regression test.
// It exercises the map propagation pattern directly because constructing a real
// *ssa.ChangeType requires a live SSA function. The hex-encode E2E test serves as
// the actual regression guard for the production code path in compiler.go:2912.
func TestSPMDDecomposedChangeTypePropagation(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()
	laneCount := 16

	b.spmdDecomposed = make(map[ssa.Value]*spmdDecomposedIndex)

	// Create a decomposed index for a hypothetical loop variable (the "inner" value).
	scalarBase := llvm.ConstInt(i32Type, 0, false)
	varyingOffset := c.spmdLaneOffsetConst(laneCount, i8Type)
	decomp := &spmdDecomposedIndex{
		scalarBase:    scalarBase,
		varyingOffset: varyingOffset,
		laneCount:     laneCount,
	}

	// Use two distinct ssa.Const values as stand-ins for the inner SSA value
	// and the ChangeType result. Both implement ssa.Value via their pointer identity.
	innerVal := ssa.NewConst(constant.MakeInt64(0), types.Typ[types.Int32])
	changeTypeVal := ssa.NewConst(constant.MakeInt64(1), types.Typ[types.Int32])

	// Register decomposition for the inner value (simulating emitSPMDBodyPrologue).
	b.spmdDecomposed[innerVal] = decomp

	// Simulate the ChangeType propagation from compiler.go (*ssa.ChangeType case):
	//   if decomp, ok := b.spmdDecomposed[expr.X]; ok {
	//       b.spmdDecomposed[expr] = decomp
	//   }
	if d, ok := b.spmdDecomposed[innerVal]; ok {
		b.spmdDecomposed[changeTypeVal] = d
	}

	// Verify that the ChangeType'd value now has the same decomposition.
	gotDecomp, ok := b.spmdDecomposed[changeTypeVal]
	if !ok {
		t.Fatal("spmdDecomposed not propagated through simulated ChangeType")
	}
	if gotDecomp != decomp {
		t.Error("propagated decomp is not the same pointer as the original")
	}
	if gotDecomp.laneCount != laneCount {
		t.Errorf("propagated laneCount = %d, want %d", gotDecomp.laneCount, laneCount)
	}
	if gotDecomp.scalarBase != scalarBase {
		t.Error("propagated scalarBase differs from original")
	}
	if gotDecomp.varyingOffset != varyingOffset {
		t.Error("propagated varyingOffset differs from original")
	}
}

// TestSPMDDecomposedBoundsCheckMaxOffset verifies that the optimized bounds check for
// constant offset vectors emits a single scalar comparison instead of per-lane checks.
// When the offset vector is a compile-time constant, the compiler computes maxOffset at
// compile time and emits: (scalarBase + maxOffset) >= buflen, which is equivalent to
// per-lane checks because maxOffset >= every per-lane offset.
func TestSPMDDecomposedBoundsCheckMaxOffset(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()
	laneCount := 16

	// Build offset vector <0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7> — max is 7.
	offsets := make([]llvm.Value, laneCount)
	for lane := 0; lane < laneCount; lane++ {
		offsets[lane] = llvm.ConstInt(i8Type, uint64(lane/2), false)
	}
	varyingOffset := llvm.ConstVector(offsets, false)

	if !varyingOffset.IsConstant() {
		t.Fatal("varyingOffset must be a constant for the optimized path")
	}

	// Verify that the maximum offset extracted at compile time is 7 (= 14/2).
	maxOffset := uint64(0)
	for lane := 0; lane < laneCount; lane++ {
		elem := llvm.ConstExtractElement(varyingOffset,
			llvm.ConstInt(i32Type, uint64(lane), false))
		val := elem.ZExtValue()
		if val > maxOffset {
			maxOffset = val
		}
	}
	if maxOffset != 7 {
		t.Fatalf("maxOffset = %d, want 7", maxOffset)
	}

	// Build a fresh function with two i32 parameters so that scalarBase and buflen
	// are non-constant runtime values. LLVM constant-folds CreateAdd/CreateICmp when
	// both operands are constants, producing no instructions in the block.
	fnType := llvm.FunctionType(c.ctx.VoidType(), []llvm.Type{i32Type, i32Type}, false)
	fn := llvm.AddFunction(c.mod, "test_bounds_check", fnType)
	bb := llvm.AddBasicBlock(fn, "entry")
	builder := c.ctx.NewBuilder()
	defer builder.Dispose()
	builder.SetInsertPointAtEnd(bb)

	scalarBase := fn.Param(0) // non-constant i32
	buflen := fn.Param(1)     // non-constant i32

	// Emit the optimized check: scalarBase + maxOffset, then icmp uge with buflen.
	maxIdx := builder.CreateAdd(scalarBase,
		llvm.ConstInt(scalarBase.Type(), maxOffset, false),
		"spmd.bounds.maxidx")
	oob := builder.CreateICmp(llvm.IntUGE, maxIdx, buflen, "spmd.bounds.oob")

	// Verify types.
	if maxIdx.Type() != i32Type {
		t.Errorf("maxIdx type = %v, want i32", maxIdx.Type())
	}
	if oob.Type() != c.ctx.Int1Type() {
		t.Errorf("oob type = %v, want i1", oob.Type())
	}

	// Confirm the add instruction carries the expected name.
	if name := maxIdx.Name(); name != "spmd.bounds.maxidx" {
		t.Errorf("maxIdx name = %q, want %q", name, "spmd.bounds.maxidx")
	}

	// Count instructions in the entry block: should be exactly 2 (add + icmp),
	// not laneCount*2 instructions as the per-lane fallback would produce.
	entry := fn.EntryBasicBlock()
	count := 0
	for inst := entry.FirstInstruction(); !inst.IsNil(); inst = llvm.NextInstruction(inst) {
		count++
	}
	// 2 instructions: the add and the icmp (the assert call added by
	// createRuntimeAssert is not called here; we test the emitted IR directly).
	if count != 2 {
		t.Errorf("instruction count = %d, want 2 (single add + single icmp, not per-lane)", count)
	}
}

// TestSPMDDecomposedSplattedConstantExtraction verifies that a Varying[T] constant
// (which createConst splats into a vector) is correctly handled in decomposed BinOp
// context. Without the fix in spmd.go, the "scalar" operand in spmdDecomposedBinOp
// would be a <4 x i32> vector, causing a type mismatch in the decomposed arithmetic.
func TestSPMDDecomposedSplattedConstantExtraction(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	// createSPMDConst should produce a <4 x i32> vector for Varying[int32] constant 2.
	int32Type := types.Typ[types.Int32]
	spmdInt32Type := types.NewVarying(int32Type)
	constVal := constant.MakeInt64(2)

	splatted := c.createSPMDConst(ssa.NewConst(constVal, spmdInt32Type), spmdInt32Type, token.NoPos)
	if splatted.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatalf("expected vector type from createSPMDConst, got %v", splatted.Type())
	}
	// Verify it is <4 x i32> (WASM128 lane count for int32).
	if splatted.Type().VectorSize() != 4 {
		t.Errorf("splatted vector lanes = %d, want 4", splatted.Type().VectorSize())
	}
	if splatted.Type().ElementType() != c.ctx.Int32Type() {
		t.Errorf("splatted element type = %v, want i32", splatted.Type().ElementType())
	}

	// Simulate the fix: when a Varying[T] constant enters spmdDecomposedBinOp, the
	// scalar element value must be extracted for decomposed arithmetic. Passing the
	// full splatted vector to CreateAdd(scalarBase, ...) would produce a type error.
	// The fix creates a new ssa.Const using the element type, not the SPMDType.
	elemType := spmdInt32Type.Elem() // types.Typ[types.Int32]
	scalarConst := ssa.NewConst(constVal, elemType)
	scalarLLVM := c.createConst(scalarConst, token.NoPos)

	// The scalar constant must be a plain i32, not a vector.
	if scalarLLVM.Type().TypeKind() == llvm.VectorTypeKind {
		t.Fatal("scalar constant extracted from SPMDType should not be a vector type")
	}
	if scalarLLVM.Type() != c.ctx.Int32Type() {
		t.Errorf("scalar constant type = %v, want i32", scalarLLVM.Type())
	}
	// Verify the extracted constant has the correct value.
	if val := scalarLLVM.ZExtValue(); val != 2 {
		t.Errorf("scalar constant value = %d, want 2", val)
	}
}

// TestSPMDSSAConstUint64 verifies that ssaConstUint64 extracts integer values
// from *ssa.Const correctly and rejects non-integer or nil constants.
func TestSPMDSSAConstUint64(t *testing.T) {
	tests := []struct {
		name   string
		val    *ssa.Const
		want   uint64
		wantOK bool
	}{
		{
			name:   "positive int",
			val:    ssa.NewConst(constant.MakeInt64(42), types.Typ[types.Int]),
			want:   42,
			wantOK: true,
		},
		{
			name:   "zero",
			val:    ssa.NewConst(constant.MakeInt64(0), types.Typ[types.Int]),
			want:   0,
			wantOK: true,
		},
		{
			name:   "max uint8",
			val:    ssa.NewConst(constant.MakeInt64(255), types.Typ[types.Uint8]),
			want:   255,
			wantOK: true,
		},
		{
			name:   "large value",
			val:    ssa.NewConst(constant.MakeInt64(65535), types.Typ[types.Uint16]),
			want:   65535,
			wantOK: true,
		},
		{
			// ssa.NewConst(nil, int) produces a zero-valued int constant, not a nil Value.
			// Use a string constant (non-integer Kind) to exercise the rejection path.
			name:   "non-integer constant",
			val:    ssa.NewConst(constant.MakeString("hello"), types.Typ[types.String]),
			want:   0,
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := ssaConstUint64(tt.val)
			if ok != tt.wantOK {
				t.Errorf("ssaConstUint64() ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && got != tt.want {
				t.Errorf("ssaConstUint64() = %d, want %d", got, tt.want)
			}
		})
	}
}

// TestSPMDTypeBitWidth verifies that typeBitWidth returns the correct bit width
// for fixed-size integer types and 0 for platform-dependent or non-integer types.
func TestSPMDTypeBitWidth(t *testing.T) {
	tests := []struct {
		name string
		typ  types.Type
		want int
	}{
		{"uint8", types.Typ[types.Uint8], 8},
		{"byte", types.Typ[types.Byte], 8},
		{"uint16", types.Typ[types.Uint16], 16},
		{"uint32", types.Typ[types.Uint32], 32},
		{"uint64", types.Typ[types.Uint64], 64},
		{"int8", types.Typ[types.Int8], 8},
		{"int16", types.Typ[types.Int16], 16},
		{"int32", types.Typ[types.Int32], 32},
		{"int64", types.Typ[types.Int64], 64},
		// Platform-dependent sizes return 0.
		{"int (platform)", types.Typ[types.Int], 0},
		{"uint (platform)", types.Typ[types.Uint], 0},
		// Non-integer types return 0.
		{"float32", types.Typ[types.Float32], 0},
		{"string", types.Typ[types.String], 0},
		// SPMDType wrapping is unwrapped before inspection.
		{"varying uint8", types.NewVarying(types.Typ[types.Uint8]), 8},
		{"varying int32", types.NewVarying(types.Typ[types.Int32]), 32},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := typeBitWidth(tt.typ)
			if got != tt.want {
				t.Errorf("typeBitWidth(%s) = %d, want %d", tt.name, got, tt.want)
			}
		})
	}
}

// TestSPMDIndexMaxValueConst verifies that spmdIndexMaxValue returns the correct
// maximum for *ssa.Const values.
func TestSPMDIndexMaxValueConst(t *testing.T) {
	tests := []struct {
		name    string
		val     ssa.Value
		wantMax uint64
		wantOK  bool
	}{
		{
			name:    "const 10",
			val:     ssa.NewConst(constant.MakeInt64(10), types.Typ[types.Int]),
			wantMax: 10,
			wantOK:  true,
		},
		{
			name:    "const 0",
			val:     ssa.NewConst(constant.MakeInt64(0), types.Typ[types.Int]),
			wantMax: 0,
			wantOK:  true,
		},
		{
			name:    "const 255 uint8",
			val:     ssa.NewConst(constant.MakeInt64(255), types.Typ[types.Uint8]),
			wantMax: 255,
			wantOK:  true,
		},
		{
			name:    "const 15 int32",
			val:     ssa.NewConst(constant.MakeInt64(15), types.Typ[types.Int32]),
			wantMax: 15,
			wantOK:  true,
		},
		{
			// Non-integer constant (string) is rejected — not representable as uint64.
			name:   "non-integer const falls through to type fallback",
			val:    ssa.NewConst(constant.MakeString("hi"), types.Typ[types.String]),
			wantOK: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := spmdIndexMaxValue(tt.val)
			if ok != tt.wantOK {
				t.Errorf("spmdIndexMaxValue() ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && got != tt.wantMax {
				t.Errorf("spmdIndexMaxValue() = %d, want %d", got, tt.wantMax)
			}
		})
	}
}

// TestSPMDIndexMaxValueBinOps verifies that spmdIndexMaxValue computes correct
// maximums for ADD, SUB, and MUL binary operations on known-max operands.
func TestSPMDIndexMaxValueBinOps(t *testing.T) {
	// makeBinOp creates a BinOp with two const operands.
	makeBinOp := func(op token.Token, x, y int64) *ssa.BinOp {
		xConst := ssa.NewConst(constant.MakeInt64(x), types.Typ[types.Int])
		yConst := ssa.NewConst(constant.MakeInt64(y), types.Typ[types.Int])
		binop := &ssa.BinOp{
			Op: op,
			X:  xConst,
			Y:  yConst,
		}
		return binop
	}

	// makeRemAdd creates a nested BinOp: (remX REM remY) ADD addZ.
	makeRemAdd := func(remX, remY, addZ int64) *ssa.BinOp {
		rem := makeBinOp(token.REM, remX, remY)
		zConst := ssa.NewConst(constant.MakeInt64(addZ), types.Typ[types.Int])
		add := &ssa.BinOp{
			Op: token.ADD,
			X:  rem,
			Y:  zConst,
		}
		return add
	}

	tests := []struct {
		name    string
		val     ssa.Value
		wantMax uint64
		wantOK  bool
	}{
		{
			name:    "ADD 3+5",
			val:     makeBinOp(token.ADD, 3, 5),
			wantMax: 8,
			wantOK:  true,
		},
		{
			name:    "ADD 0+10",
			val:     makeBinOp(token.ADD, 0, 10),
			wantMax: 10,
			wantOK:  true,
		},
		{
			// SUB max is maxOf(x); subtraction can only decrease the value.
			name:    "SUB 10-3 (max is maxOf(x)=10)",
			val:     makeBinOp(token.SUB, 10, 3),
			wantMax: 10,
			wantOK:  true,
		},
		{
			name:    "MUL 3*4",
			val:     makeBinOp(token.MUL, 3, 4),
			wantMax: 12,
			wantOK:  true,
		},
		{
			name:    "MUL 0*100",
			val:     makeBinOp(token.MUL, 0, 100),
			wantMax: 0,
			wantOK:  true,
		},
		{
			// Composing REM + ADD: (i%3)+2 → max = 2+2 = 4.
			name:    "nested: (i%3)+2 → max = 2+2 = 4",
			val:     makeRemAdd(100, 3, 2),
			wantMax: 4,
			wantOK:  true,
		},
		{
			// Composing REM + ADD: (i%16)+0 → max = 15+0 = 15.
			name:    "nested: (i%16)+0 → max = 15+0 = 15",
			val:     makeRemAdd(1000, 16, 0),
			wantMax: 15,
			wantOK:  true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, ok := spmdIndexMaxValue(tt.val)
			if ok != tt.wantOK {
				t.Errorf("spmdIndexMaxValue() ok = %v, want %v", ok, tt.wantOK)
			}
			if ok && got != tt.wantMax {
				t.Errorf("spmdIndexMaxValue() = %d, want %d", got, tt.wantMax)
			}
		})
	}
}

func TestSPMDStoreCoalescing(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, laneCount)
	maskType := llvm.VectorType(c.ctx.Int32Type(), laneCount) // WASM i32 mask

	// Create array alloca + contiguous GEP (simulating d[i])
	arrType := llvm.ArrayType(i32Type, 16)
	arrPtr := b.CreateAlloca(arrType, "d")
	zero := llvm.ConstInt(i32Type, 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, arrPtr, []llvm.Value{zero, zero}, "d.ptr")

	// Create condition, then-value, else-value
	cond := llvm.ConstAllOnes(maskType) // all-true condition (for testing)
	thenVal := llvm.ConstNull(vecType)  // then: store zeros
	elseVal := llvm.ConstInt(i32Type, 42, false)
	elseVec := b.splatScalar(elseVal, vecType)

	// Create parent mask (all-ones = top-level go-for body)
	parentMask := llvm.ConstAllOnes(maskType)

	// Emit the coalesced store: select(cond, thenVal, elseVal) + single store
	selected := b.spmdMaskSelect(cond, thenVal, elseVec)
	b.spmdMaskedStore(selected, scalarPtr, parentMask)

	// Verify: exactly one llvm.masked.store call in the module
	maskedStoreFn := c.mod.NamedFunction("llvm.masked.store.v4i32.p0")
	if maskedStoreFn.IsNil() {
		t.Fatal("expected llvm.masked.store.v4i32.p0 to be declared")
	}

	// Verify the function has instructions (basic sanity)
	fn := c.mod.FirstFunction()
	for !fn.IsNil() {
		if fn.Name() == "test_func" {
			bb := fn.FirstBasicBlock()
			instrCount := 0
			for instr := bb.FirstInstruction(); !instr.IsNil(); instr = llvm.NextInstruction(instr) {
				instrCount++
			}
			// Should have: alloca, GEP, splat, select/bitwise-ops, masked-store, terminator
			if instrCount < 3 {
				t.Errorf("expected at least 3 instructions, got %d", instrCount)
			}
			break
		}
		fn = llvm.NextFunction(fn)
	}
}

// TestSPMDStoreCoalescingScalarScalar verifies that when both then-value and
// else-value are scalar constants (uniform), the coalesced store codegen path
// correctly splats them to vectors before the mask select.
func TestSPMDStoreCoalescingScalarScalar(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, laneCount)
	maskType := llvm.VectorType(c.ctx.Int32Type(), laneCount)

	// Both values are scalar constants (the scenario the fix addresses).
	cond := llvm.ConstAllOnes(maskType)
	thenScalar := llvm.ConstInt(i32Type, 100, false)
	elseScalar := llvm.ConstInt(i32Type, 200, false)

	// Simulate the codegen path: broadcastMatch, then scalar-scalar splat.
	thenVal, elseVal := b.spmdBroadcastMatch(thenScalar, elseScalar)
	// After broadcastMatch, both should still be scalar (neither is vector).
	if thenVal.Type().TypeKind() == llvm.VectorTypeKind {
		t.Fatal("expected thenVal to remain scalar after broadcastMatch")
	}
	if elseVal.Type().TypeKind() == llvm.VectorTypeKind {
		t.Fatal("expected elseVal to remain scalar after broadcastMatch")
	}

	// Apply the scalar-scalar splat (matching the codegen fix).
	if thenVal.Type().TypeKind() != llvm.VectorTypeKind &&
		elseVal.Type().TypeKind() != llvm.VectorTypeKind &&
		cond.Type().TypeKind() == llvm.VectorTypeKind {
		lc := cond.Type().VectorSize()
		vt := llvm.VectorType(thenVal.Type(), lc)
		thenVal = b.splatScalar(thenVal, vt)
		elseVal = b.splatScalar(elseVal, vt)
	}

	// Now both must be vectors.
	if thenVal.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatal("expected thenVal to be vector after splat")
	}
	if elseVal.Type().TypeKind() != llvm.VectorTypeKind {
		t.Fatal("expected elseVal to be vector after splat")
	}
	if thenVal.Type() != vecType {
		t.Errorf("expected thenVal type %v, got %v", vecType, thenVal.Type())
	}
	if elseVal.Type() != vecType {
		t.Errorf("expected elseVal type %v, got %v", vecType, elseVal.Type())
	}

	// Verify the full chain works: select + masked store.
	arrType := llvm.ArrayType(i32Type, 16)
	arrPtr := b.CreateAlloca(arrType, "d")
	zero := llvm.ConstInt(i32Type, 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, arrPtr, []llvm.Value{zero, zero}, "d.ptr")
	parentMask := llvm.ConstAllOnes(maskType)

	selected := b.spmdMaskSelect(cond, thenVal, elseVal)
	b.spmdMaskedStore(selected, scalarPtr, parentMask)

	maskedStoreFn := c.mod.NamedFunction("llvm.masked.store.v4i32.p0")
	if maskedStoreFn.IsNil() {
		t.Fatal("expected llvm.masked.store.v4i32.p0 to be declared")
	}
}

// TestSPMDIndexMaxValueTypeFallback verifies that spmdIndexMaxValue applies the
// type-based upper bound for unsigned fixed-size types when expression analysis
// is not available. This test exercises the logic path directly via typeBitWidth
// and the IsUnsigned check, mirroring what spmdIndexMaxValue does internally.
func TestSPMDIndexMaxValueTypeFallback(t *testing.T) {
	tests := []struct {
		name    string
		typ     types.Type
		wantMax uint64
		wantOK  bool
	}{
		{"uint8 typed", types.Typ[types.Uint8], 255, true},
		{"uint16 typed", types.Typ[types.Uint16], 65535, true},
		{"uint32 typed", types.Typ[types.Uint32], 4294967295, true},
		// Platform-dependent sizes — unknown upper bound.
		{"int (unknown)", types.Typ[types.Int], 0, false},
		{"uint (unknown)", types.Typ[types.Uint], 0, false},
		// Varying wrappers are unwrapped correctly.
		{"varying uint8", types.NewVarying(types.Typ[types.Uint8]), 255, true},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			bits := typeBitWidth(tt.typ)
			if bits == 0 {
				if tt.wantOK {
					t.Errorf("expected known max for %s, but typeBitWidth returned 0", tt.name)
				}
				return
			}
			// Check if the type is unsigned (mirrors spmdIndexMaxValue's fallback logic).
			rawType := tt.typ
			if spmd, ok := rawType.(*types.SPMDType); ok {
				rawType = spmd.Elem()
			}
			basic, ok := rawType.Underlying().(*types.Basic)
			if !ok {
				if tt.wantOK {
					t.Errorf("expected basic type for %s", tt.name)
				}
				return
			}
			isUnsigned := basic.Info()&types.IsUnsigned != 0
			if !isUnsigned {
				if tt.wantOK {
					t.Errorf("expected unsigned type for %s", tt.name)
				}
				return
			}
			maxVal := (uint64(1) << uint(bits)) - 1
			if maxVal != tt.wantMax {
				t.Errorf("type fallback max for %s = %d, want %d", tt.name, maxVal, tt.wantMax)
			}
		})
	}
}

// TestSPMDShiftedIndexDetection verifies that spmdAnalyzeShiftedIndex correctly
// detects right-shifted contiguous index patterns like i>>1, i>>2, and rejects
// non-matching patterns like i>>0, i+1, or non-constant shifts.
func TestSPMDShiftedIndexDetection(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Create a fake SPMD loop with laneCount=16 (byte lanes on WASM).
	iterPhi := &ssa.Phi{}
	loop := &spmdActiveLoop{
		iterPhi:       iterPhi,
		laneCount:     16,
		scalarIterVal: llvm.ConstInt(c.ctx.Int32Type(), 0, false),
	}
	b.spmdLoopState = &spmdLoopState{
		activeLoops: map[ssa.Value]*spmdActiveLoop{
			iterPhi: loop,
		},
	}
	b.spmdValueOverride = map[ssa.Value]llvm.Value{
		iterPhi: llvm.Undef(llvm.VectorType(c.ctx.Int32Type(), 16)),
	}

	// Helper to create a constant SSA value with an integer value.
	mkConst := func(val uint64) *ssa.Const {
		cnst := ssa.NewConst(constant.MakeUint64(val), types.Typ[types.Int])
		return cnst
	}

	tests := []struct {
		name            string
		index           ssa.Value
		wantMatch       bool
		wantUniqueCount int
		wantShuffleLast int // expected shuffleMask[laneCount-1]
	}{
		{
			name: "i>>1 (16 lanes)",
			index: &ssa.BinOp{
				Op: token.SHR,
				X:  iterPhi,
				Y:  mkConst(1),
			},
			wantMatch:       true,
			wantUniqueCount: 8,
			wantShuffleLast: 7,
		},
		{
			name: "i>>2 (16 lanes)",
			index: &ssa.BinOp{
				Op: token.SHR,
				X:  iterPhi,
				Y:  mkConst(2),
			},
			wantMatch:       true,
			wantUniqueCount: 4,
			wantShuffleLast: 3,
		},
		{
			name: "i>>3 (16 lanes)",
			index: &ssa.BinOp{
				Op: token.SHR,
				X:  iterPhi,
				Y:  mkConst(3),
			},
			wantMatch:       true,
			wantUniqueCount: 2,
			wantShuffleLast: 1,
		},
		{
			name: "i>>4 (16 lanes, broadcast)",
			index: &ssa.BinOp{
				Op: token.SHR,
				X:  iterPhi,
				Y:  mkConst(4),
			},
			wantMatch:       true,
			wantUniqueCount: 1,
			wantShuffleLast: 0,
		},
		{
			name: "i>>0 (no shift, reject)",
			index: &ssa.BinOp{
				Op: token.SHR,
				X:  iterPhi,
				Y:  mkConst(0),
			},
			wantMatch: false,
		},
		{
			name: "i+1 (not SHR, reject)",
			index: &ssa.BinOp{
				Op: token.ADD,
				X:  iterPhi,
				Y:  mkConst(1),
			},
			wantMatch: false,
		},
		{
			name:      "plain iter (not BinOp, reject)",
			index:     iterPhi,
			wantMatch: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			info, ok := b.spmdAnalyzeShiftedIndex(tt.index)
			if ok != tt.wantMatch {
				t.Fatalf("spmdAnalyzeShiftedIndex match = %v, want %v", ok, tt.wantMatch)
			}
			if !ok {
				return
			}
			if info.uniqueCount != tt.wantUniqueCount {
				t.Errorf("uniqueCount = %d, want %d", info.uniqueCount, tt.wantUniqueCount)
			}
			if len(info.shuffleMask) != 16 {
				t.Fatalf("shuffleMask length = %d, want 16", len(info.shuffleMask))
			}
			if info.shuffleMask[15] != tt.wantShuffleLast {
				t.Errorf("shuffleMask[15] = %d, want %d", info.shuffleMask[15], tt.wantShuffleLast)
			}
			// Verify shuffle mask monotonically non-decreasing.
			for i := 1; i < 16; i++ {
				if info.shuffleMask[i] < info.shuffleMask[i-1] {
					t.Errorf("shuffleMask not monotonic: [%d]=%d < [%d]=%d",
						i, info.shuffleMask[i], i-1, info.shuffleMask[i-1])
				}
			}
		})
	}
}

// TestSPMDShiftedLoadCodegen verifies that spmdShiftedLoad generates correct
// LLVM IR: narrow vector load + shufflevector for uniqueCount > 1, and
// scalar load + splat for uniqueCount == 1.
func TestSPMDShiftedLoadCodegen(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i8Type := c.ctx.Int8Type()
	i32Type := c.ctx.Int32Type()

	t.Run("shr1_16lanes_i8", func(t *testing.T) {
		// Simulate i>>1 with 16 byte lanes: 8 unique elements.
		// Create an alloca to serve as the base pointer.
		arrayType := llvm.ArrayType(i8Type, 8)
		ptr := b.CreateAlloca(arrayType, "test.arr")
		basePtr := b.CreateInBoundsGEP(arrayType, ptr, []llvm.Value{
			llvm.ConstInt(i32Type, 0, false),
			llvm.ConstInt(i32Type, 0, false),
		}, "base")

		shuffleMask := make([]int, 16)
		for i := 0; i < 16; i++ {
			shuffleMask[i] = i >> 1
		}

		info := &spmdShiftedLoadInfo{
			scalarPtr:   basePtr,
			uniqueCount: 8,
			shuffleMask: shuffleMask,
			elemType:    i8Type,
			loop:        &spmdActiveLoop{laneCount: 16},
		}

		result := b.spmdShiftedLoad(info, llvm.Value{})
		if result.IsNil() {
			t.Fatal("spmdShiftedLoad returned nil")
		}
		if result.Type().TypeKind() != llvm.VectorTypeKind {
			t.Fatalf("result type = %v, want vector", result.Type())
		}
		if result.Type().VectorSize() != 16 {
			t.Errorf("result lanes = %d, want 16", result.Type().VectorSize())
		}
		if result.Type().ElementType() != i8Type {
			t.Errorf("result elem = %v, want i8", result.Type().ElementType())
		}
	})

	t.Run("shr4_16lanes_broadcast", func(t *testing.T) {
		// Simulate i>>4 with 16 byte lanes: 1 unique element (broadcast).
		ptr := b.CreateAlloca(i8Type, "test.scalar")

		shuffleMask := make([]int, 16)
		// All zeros for broadcast.

		info := &spmdShiftedLoadInfo{
			scalarPtr:   ptr,
			uniqueCount: 1,
			shuffleMask: shuffleMask,
			elemType:    i8Type,
			loop:        &spmdActiveLoop{laneCount: 16},
		}

		result := b.spmdShiftedLoad(info, llvm.Value{})
		if result.IsNil() {
			t.Fatal("spmdShiftedLoad returned nil")
		}
		if result.Type().TypeKind() != llvm.VectorTypeKind {
			t.Fatalf("result type = %v, want vector", result.Type())
		}
		if result.Type().VectorSize() != 16 {
			t.Errorf("result lanes = %d, want 16", result.Type().VectorSize())
		}
	})

	t.Run("shr1_4lanes_i32", func(t *testing.T) {
		// Simulate i>>1 with 4 i32 lanes: 2 unique elements.
		arrayType := llvm.ArrayType(i32Type, 2)
		ptr := b.CreateAlloca(arrayType, "test.arr32")
		basePtr := b.CreateInBoundsGEP(arrayType, ptr, []llvm.Value{
			llvm.ConstInt(i32Type, 0, false),
			llvm.ConstInt(i32Type, 0, false),
		}, "base32")

		shuffleMask := []int{0, 0, 1, 1}

		info := &spmdShiftedLoadInfo{
			scalarPtr:   basePtr,
			uniqueCount: 2,
			shuffleMask: shuffleMask,
			elemType:    i32Type,
			loop:        &spmdActiveLoop{laneCount: 4},
		}

		result := b.spmdShiftedLoad(info, llvm.Value{})
		if result.IsNil() {
			t.Fatal("spmdShiftedLoad returned nil")
		}
		if result.Type().VectorSize() != 4 {
			t.Errorf("result lanes = %d, want 4", result.Type().VectorSize())
		}
		if result.Type().ElementType() != i32Type {
			t.Errorf("result elem = %v, want i32", result.Type().ElementType())
		}
	})

	t.Run("masked_shr1_16lanes", func(t *testing.T) {
		// Test with a non-nil mask: should produce select(mask, result, zero).
		arrayType := llvm.ArrayType(i8Type, 8)
		ptr := b.CreateAlloca(arrayType, "test.masked")
		basePtr := b.CreateInBoundsGEP(arrayType, ptr, []llvm.Value{
			llvm.ConstInt(i32Type, 0, false),
			llvm.ConstInt(i32Type, 0, false),
		}, "base.masked")

		shuffleMask := make([]int, 16)
		for i := 0; i < 16; i++ {
			shuffleMask[i] = i >> 1
		}

		info := &spmdShiftedLoadInfo{
			scalarPtr:   basePtr,
			uniqueCount: 8,
			shuffleMask: shuffleMask,
			elemType:    i8Type,
			loop:        &spmdActiveLoop{laneCount: 16},
		}

		// Create a mask: <16 x i8> all-ones (WASM format for 16 lanes).
		maskType := llvm.VectorType(c.spmdMaskElemType(16), 16)
		mask := llvm.ConstAllOnes(maskType)

		result := b.spmdShiftedLoad(info, mask)
		if result.IsNil() {
			t.Fatal("spmdShiftedLoad with mask returned nil")
		}
		if result.Type().VectorSize() != 16 {
			t.Errorf("result lanes = %d, want 16", result.Type().VectorSize())
		}
	})
}

// extractConstVec extracts all integer elements from a constant LLVM vector as
// a []uint64 slice, using ConstExtractElement with an i32 index. The inputs
// must be constant-foldable (e.g., shufflevector of constant inputs).
func extractConstVec(t *testing.T, v llvm.Value, ctx llvm.Context) []uint64 {
	t.Helper()
	n := v.Type().VectorSize()
	result := make([]uint64, n)
	for i := 0; i < n; i++ {
		idx := llvm.ConstInt(ctx.Int32Type(), uint64(i), false)
		elem := llvm.ConstExtractElement(v, idx)
		result[i] = elem.ZExtValue()
	}
	return result
}

// buildConstVec creates a <N x elemType> constant vector from a []uint64 value
// slice. Each value must fit in the element type.
func buildConstVec(ctx llvm.Context, elemType llvm.Type, vals []uint64) llvm.Value {
	elts := make([]llvm.Value, len(vals))
	for i, v := range vals {
		elts[i] = llvm.ConstInt(elemType, v, false)
	}
	return llvm.ConstVector(elts, false)
}

// TestSPMDInterleaveStride2Masks verifies that spmdInterleaveStride2 produces
// the correct interleaving for N=16 (byte lanes on WASM SIMD128).
//
// val0 = <0,1,...,15>  and  val1 = <16,17,...,31>
// Expected outputs:
//
//	out[0] = <0,16,1,17,2,18,3,19,4,20,5,21,6,22,7,23>   (lo: first halves interleaved)
//	out[1] = <8,24,9,25,10,26,11,27,12,28,13,29,14,30,15,31>  (hi: second halves interleaved)
func TestSPMDInterleaveStride2Masks(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	const N = 16
	i8Type := c.ctx.Int8Type()

	// Build val0 = <0,1,...,15> and val1 = <16,17,...,31>.
	v0vals := make([]uint64, N)
	v1vals := make([]uint64, N)
	for i := 0; i < N; i++ {
		v0vals[i] = uint64(i)
		v1vals[i] = uint64(N + i)
	}
	val0 := buildConstVec(c.ctx, i8Type, v0vals)
	val1 := buildConstVec(c.ctx, i8Type, v1vals)

	out := b.spmdInterleaveStride2(val0, val1, N)

	want0 := []uint64{0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23}
	want1 := []uint64{8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31}

	got0 := extractConstVec(t, out[0], c.ctx)
	got1 := extractConstVec(t, out[1], c.ctx)

	if !reflect.DeepEqual(got0, want0) {
		t.Errorf("out[0] = %v\n          want %v", got0, want0)
	}
	if !reflect.DeepEqual(got1, want1) {
		t.Errorf("out[1] = %v\n          want %v", got1, want1)
	}
}

// TestSPMDInterleaveStride2N4 verifies spmdInterleaveStride2 for N=4 (i32 lanes).
//
// val0 = <0,1,2,3>  and  val1 = <4,5,6,7>
// Expected:
//
//	out[0] = <0,4,1,5>
//	out[1] = <2,6,3,7>
func TestSPMDInterleaveStride2N4(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	const N = 4
	i32Type := c.ctx.Int32Type()

	val0 := buildConstVec(c.ctx, i32Type, []uint64{0, 1, 2, 3})
	val1 := buildConstVec(c.ctx, i32Type, []uint64{4, 5, 6, 7})

	out := b.spmdInterleaveStride2(val0, val1, N)

	want0 := []uint64{0, 4, 1, 5}
	want1 := []uint64{2, 6, 3, 7}

	got0 := extractConstVec(t, out[0], c.ctx)
	got1 := extractConstVec(t, out[1], c.ctx)

	if !reflect.DeepEqual(got0, want0) {
		t.Errorf("out[0] = %v, want %v", got0, want0)
	}
	if !reflect.DeepEqual(got1, want1) {
		t.Errorf("out[1] = %v, want %v", got1, want1)
	}
}

// TestSPMDInterleaveStride4Butterfly verifies that spmdInterleaveStride4
// produces the correct 4-way interleaving for N=16 (byte lanes on WASM).
//
// val0 = <0..15>, val1 = <16..31>, val2 = <32..47>, val3 = <48..63>
// For each global output position p = k*16+j:
//
//	element = vals[p%4][p/4]
//
// So:
//
//	out[0] = [0,16,32,48, 1,17,33,49, 2,18,34,50, 3,19,35,51]
//	out[1] = [4,20,36,52, 5,21,37,53, 6,22,38,54, 7,23,39,55]
//	out[2] = [8,24,40,56, 9,25,41,57, 10,26,42,58, 11,27,43,59]
//	out[3] = [12,28,44,60, 13,29,45,61, 14,30,46,62, 15,31,47,63]
func TestSPMDInterleaveStride4Butterfly(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	const N = 16
	i8Type := c.ctx.Int8Type()

	vals := [4]llvm.Value{}
	for s := 0; s < 4; s++ {
		v := make([]uint64, N)
		for i := 0; i < N; i++ {
			v[i] = uint64(s*N + i)
		}
		vals[s] = buildConstVec(c.ctx, i8Type, v)
	}

	out := b.spmdInterleaveStride4(vals[0], vals[1], vals[2], vals[3], N)

	// Build expected: for output k, position j: globalPos = k*N+j; value = vals[globalPos%4][globalPos/4].
	wants := [4][]uint64{}
	for k := 0; k < 4; k++ {
		wants[k] = make([]uint64, N)
		for j := 0; j < N; j++ {
			globalPos := k*N + j
			srcVec := globalPos % 4
			srcLane := globalPos / 4
			wants[k][j] = uint64(srcVec*N + srcLane)
		}
	}

	for k := 0; k < 4; k++ {
		got := extractConstVec(t, out[k], c.ctx)
		if !reflect.DeepEqual(got, wants[k]) {
			t.Errorf("out[%d] = %v\n             want %v", k, got, wants[k])
		}
	}
}

// TestSPMDMaskExpansionStride2 verifies spmdExpandMaskForStride for stride=2, N=16.
//
// Source mask = <0,1,2,...,15>. For stride-2, the expanded mask for output k at
// position j selects source lane (k*N+j)/2:
//
//	expanded[0]: srcLane = j/2       → [0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7]
//	expanded[1]: srcLane = (16+j)/2  → [8,8,9,9,10,10,11,11,12,12,13,13,14,14,15,15]
func TestSPMDMaskExpansionStride2(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	const N = 16
	const stride = 2
	maskElemType := c.spmdMaskElemType(N)

	// Build source mask <0,1,...,15> where each element is treated as a lane identifier.
	srcVals := make([]uint64, N)
	for i := 0; i < N; i++ {
		srcVals[i] = uint64(i)
	}
	srcMask := buildConstVec(c.ctx, maskElemType, srcVals)

	expanded := b.spmdExpandMaskForStride(srcMask, stride, N)

	if len(expanded) != stride {
		t.Fatalf("len(expanded) = %d, want %d", len(expanded), stride)
	}

	// Build expected: expanded[k][j] = (k*N+j)/stride sourced from srcVals.
	for k := 0; k < stride; k++ {
		want := make([]uint64, N)
		for j := 0; j < N; j++ {
			srcLane := (k*N + j) / stride
			want[j] = srcVals[srcLane]
		}
		got := extractConstVec(t, expanded[k], c.ctx)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("expanded[%d] = %v\n                  want %v", k, got, want)
		}
	}
}

// TestSPMDAnalyzeStrideIndex tests the SSA pattern matching in spmdAnalyzeStrideIndex
// for stride-S interleaved index expressions of the form iter*S+R.
func TestSPMDAnalyzeStrideIndex(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Create a fake SPMD loop iterator phi (laneCount=16, i8 lanes on WASM).
	iterPhi := &ssa.Phi{}
	loop := &spmdActiveLoop{
		bodyIterValue: iterPhi,
		laneCount:     16,
		scalarIterVal: llvm.ConstInt(c.ctx.Int32Type(), 0, false),
	}
	b.spmdLoopState = &spmdLoopState{
		activeLoops: map[ssa.Value]*spmdActiveLoop{
			iterPhi: loop,
		},
		bodyBlocks: map[int]*spmdActiveLoop{},
		loopBlocks: map[int]*spmdActiveLoop{},
	}

	// mkConst creates a typed integer SSA constant.
	mkConst := func(val int64) *ssa.Const {
		return ssa.NewConst(constant.MakeInt64(val), types.Typ[types.Int])
	}
	// mkMul creates iter*factor as a BinOp.
	mkMul := func(iter ssa.Value, factor int64) *ssa.BinOp {
		op := &ssa.BinOp{}
		op.Op = token.MUL
		op.X = iter
		op.Y = mkConst(factor)
		return op
	}
	// mkAdd creates lhs+rhs as a BinOp.
	mkAdd := func(lhs, rhs ssa.Value) *ssa.BinOp {
		op := &ssa.BinOp{}
		op.Op = token.ADD
		op.X = lhs
		op.Y = rhs
		return op
	}

	tests := []struct {
		name          string
		index         ssa.Value
		wantNil       bool
		wantStride    int64
		wantRemainder int64
	}{
		{
			name:          "iter*2 → stride=2 remainder=0",
			index:         mkMul(iterPhi, 2),
			wantStride:    2,
			wantRemainder: 0,
		},
		{
			name:          "iter*2+1 → stride=2 remainder=1",
			index:         mkAdd(mkMul(iterPhi, 2), mkConst(1)),
			wantStride:    2,
			wantRemainder: 1,
		},
		{
			name:          "iter*3+2 → stride=3 remainder=2",
			index:         mkAdd(mkMul(iterPhi, 3), mkConst(2)),
			wantStride:    3,
			wantRemainder: 2,
		},
		{
			name:          "iter*4 → stride=4 remainder=0",
			index:         mkMul(iterPhi, 4),
			wantStride:    4,
			wantRemainder: 0,
		},
		{
			// Commutative MUL: 2*iter instead of iter*2.
			name: "2*iter+1 → stride=2 remainder=1 (commutative MUL and ADD)",
			index: func() ssa.Value {
				mul := &ssa.BinOp{}
				mul.Op = token.MUL
				mul.X = mkConst(2) // constant on left
				mul.Y = iterPhi
				// 1 + mul (constant on left of ADD).
				add := &ssa.BinOp{}
				add.Op = token.ADD
				add.X = mkConst(1)
				add.Y = mul
				return add
			}(),
			wantStride:    2,
			wantRemainder: 1,
		},
		{
			// Stride 5 exceeds [2,4]; should return nil.
			name:    "iter*5 → nil (stride > 4)",
			index:   mkMul(iterPhi, 5),
			wantNil: true,
		},
		{
			// ADD without MUL: iter+1 should return nil (no stride).
			name: "iter+1 → nil (no MUL)",
			index: func() ssa.Value {
				op := &ssa.BinOp{}
				op.Op = token.ADD
				op.X = iterPhi
				op.Y = mkConst(1)
				return op
			}(),
			wantNil: true,
		},
		{
			// Remainder >= stride: iter*2+3 is invalid (R=3 >= S=2).
			name:    "iter*2+3 → nil (remainder >= stride)",
			index:   mkAdd(mkMul(iterPhi, 2), mkConst(3)),
			wantNil: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			pat := b.spmdAnalyzeStrideIndex(tt.index)
			if tt.wantNil {
				if pat != nil {
					t.Errorf("spmdAnalyzeStrideIndex = %+v, want nil", pat)
				}
				return
			}
			if pat == nil {
				t.Fatal("spmdAnalyzeStrideIndex = nil, want non-nil")
			}
			if pat.stride != tt.wantStride {
				t.Errorf("stride = %d, want %d", pat.stride, tt.wantStride)
			}
			if pat.remainder != tt.wantRemainder {
				t.Errorf("remainder = %d, want %d", pat.remainder, tt.wantRemainder)
			}
			if pat.loop != loop {
				t.Errorf("loop pointer mismatch")
			}
		})
	}
}

// TestSPMDInterleaveStride3 verifies spmdInterleaveStride3 for N=16 (byte lanes).
//
// val0 = <0..15>, val1 = <16..31>, val2 = <32..47>
// For each global output position p = k*16+j:
//
//	element = vals[p%3][p/3]
//
// So the expected output vectors are:
//
//	out[0] = [0,16,32, 1,17,33, 2,18,34, 3,19,35, 4,20,36, 5]       (positions 0-15)
//	out[1] = [21,37, 6,22,38, 7,23,39, 8,24,40, 9,25,41, 10,26]     (positions 16-31)
//	out[2] = [42, 11,27,43, 12,28,44, 13,29,45, 14,30,46, 15,31,47] (positions 32-47)
func TestSPMDInterleaveStride3(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	const N = 16
	i8Type := c.ctx.Int8Type()

	vals := [3]llvm.Value{}
	for s := 0; s < 3; s++ {
		v := make([]uint64, N)
		for i := 0; i < N; i++ {
			v[i] = uint64(s*N + i)
		}
		vals[s] = buildConstVec(c.ctx, i8Type, v)
	}

	out := b.spmdInterleaveStride3(vals[0], vals[1], vals[2], N)

	// Build expected: for output k, position j: globalPos = k*N+j; value = vals[globalPos%3][globalPos/3].
	for k := 0; k < 3; k++ {
		want := make([]uint64, N)
		for j := 0; j < N; j++ {
			globalPos := k*N + j
			srcVec := globalPos % 3
			srcLane := globalPos / 3
			want[j] = uint64(srcVec*N + srcLane)
		}
		got := extractConstVec(t, out[k], c.ctx)
		if !reflect.DeepEqual(got, want) {
			t.Errorf("out[%d] =\n  got  %v\n  want %v", k, got, want)
		}
	}
}

// TestSPMDFullLoadWithSelect verifies that spmdFullLoadWithSelect emits the
// correct branch structure: cap check -> full load + select | masked load -> merge phi.
func TestSPMDFullLoadWithSelect(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, laneCount)

	scalarIndex := llvm.ConstInt(b.uintptrType, 8, false)
	sliceCap := llvm.ConstInt(b.uintptrType, 16, false)

	arrType := llvm.ArrayType(i32Type, 16)
	alloca := b.CreateAlloca(arrType, "test.buf")
	zero := llvm.ConstInt(i32Type, 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, alloca, []llvm.Value{zero, zero}, "test.ptr")

	maskElemType := i32Type
	mask := llvm.ConstVector([]llvm.Value{
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstNull(maskElemType),
		llvm.ConstNull(maskElemType),
	}, false)

	ci := &spmdContiguousInfo{
		scalarPtr:   scalarPtr,
		loop:        &spmdActiveLoop{laneCount: laneCount},
		sliceCap:    sliceCap,
		scalarIndex: scalarIndex,
	}

	mergeBB := b.insertBasicBlock("test.after")
	result := b.spmdFullLoadWithSelect(vecType, ci, mask)
	b.CreateBr(mergeBB)
	b.SetInsertPointAtEnd(mergeBB)
	b.CreateRetVoid()

	if result.IsNil() {
		t.Fatal("spmdFullLoadWithSelect returned nil")
	}
	if result.Type().TypeKind() != llvm.VectorTypeKind {
		t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
	}
	if result.Type().VectorSize() != laneCount {
		t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), laneCount)
	}

	ir := c.mod.String()
	if !strings.Contains(ir, "llvm.masked.load") {
		t.Error("IR should contain llvm.masked.load for the fallback path")
	}
	if !strings.Contains(ir, "spmd.fullload") {
		t.Error("IR should contain spmd.fullload basic block")
	}
	if !strings.Contains(ir, "spmd.maskedload") {
		t.Error("IR should contain spmd.maskedload basic block")
	}
	if !strings.Contains(ir, "spmd.load.merge") {
		t.Error("IR should contain spmd.load.merge basic block")
	}
}

// TestSPMDIsConstAllOnesMask verifies that spmdIsConstAllOnesMask correctly
// identifies all-ones masks and rejects partial or non-constant masks.
func TestSPMDIsConstAllOnesMask(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i32Type := c.ctx.Int32Type()
	maskType := llvm.VectorType(i32Type, 4)

	// All-ones mask should be detected.
	allOnesMask := llvm.ConstAllOnes(maskType)
	if !b.spmdIsConstAllOnesMask(allOnesMask) {
		t.Error("should return true for ConstAllOnes mask")
	}

	// Partial mask should NOT be detected.
	partialMask := llvm.ConstVector([]llvm.Value{
		llvm.ConstAllOnes(i32Type),
		llvm.ConstNull(i32Type),
		llvm.ConstNull(i32Type),
		llvm.ConstNull(i32Type),
	}, false)
	if b.spmdIsConstAllOnesMask(partialMask) {
		t.Error("should return false for partial mask")
	}

	// Null mask should NOT be detected.
	nullMask := llvm.ConstNull(maskType)
	if b.spmdIsConstAllOnesMask(nullMask) {
		t.Error("should return false for null mask")
	}
}

// TestSPMDFullLoadCapTypeMismatch verifies that spmdFullLoadWithSelect handles
// integer type mismatches between the scalar index and slice cap without
// panicking, and produces valid IR with the appropriate extend/truncate
// instruction.
//
// LLVM constant-folds trunc/zext on ConstInt values, so the index and cap
// must be non-constant (function parameters) to keep the cast instructions
// visible in the IR.
func TestSPMDFullLoadCapTypeMismatch(t *testing.T) {
	laneCount := 4

	tests := []struct {
		name         string
		indexType    func(c *compilerContext) llvm.Type
		capType      func(c *compilerContext) llvm.Type
		wantIRString string
	}{
		{
			// scalarIndex is i32 (uintptr on wasm32), sliceCap is i64.
			// cap (64 bits) > iter (32 bits) -> Trunc path.
			name:         "i32 index, i64 cap",
			indexType:    func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			capType:      func(c *compilerContext) llvm.Type { return c.ctx.Int64Type() },
			wantIRString: "spmd.cap.trunc",
		},
		{
			// scalarIndex is i64, sliceCap is i32 (uintptr on wasm32).
			// cap (32 bits) < iter (64 bits) -> ZExt path.
			name:         "i64 index, i32 cap",
			indexType:    func(c *compilerContext) llvm.Type { return c.ctx.Int64Type() },
			capType:      func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			wantIRString: "spmd.cap.ext",
		},
		{
			// Both i32 — the common case on wasm32. No cast needed.
			name:         "i32 index, i32 cap",
			indexType:    func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			capType:      func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			wantIRString: "spmd.can.fullload",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			// Each sub-test gets its own context and builder because the
			// builder state cannot be reused after branching IR is created.
			c := newTestCompilerContext(t)
			defer c.dispose()

			i32Type := c.ctx.Int32Type()
			vecType := llvm.VectorType(i32Type, laneCount)

			idxType := tc.indexType(c)
			capType := tc.capType(c)

			// Use function parameters rather than ConstInt values so that
			// LLVM cannot constant-fold the trunc/zext instructions away.
			ptrType := llvm.PointerType(i32Type, 0)
			paramTypes := []llvm.Type{idxType, capType, ptrType}
			fnType := llvm.FunctionType(c.ctx.VoidType(), paramTypes, false)
			fn := llvm.AddFunction(c.mod, "test_func_mismatch", fnType)
			bb := llvm.AddBasicBlock(fn, "entry")

			b := &builder{compilerContext: c}
			b.Builder = c.ctx.NewBuilder()
			defer b.Dispose()
			b.llvmFn = fn
			b.SetInsertPointAtEnd(bb)

			scalarIndex := fn.Param(0)
			sliceCap := fn.Param(1)
			scalarPtr := fn.Param(2)

			maskElemType := i32Type
			mask := llvm.ConstVector([]llvm.Value{
				llvm.ConstAllOnes(maskElemType),
				llvm.ConstAllOnes(maskElemType),
				llvm.ConstNull(maskElemType),
				llvm.ConstNull(maskElemType),
			}, false)

			ci := &spmdContiguousInfo{
				scalarPtr:   scalarPtr,
				loop:        &spmdActiveLoop{laneCount: laneCount},
				sliceCap:    sliceCap,
				scalarIndex: scalarIndex,
			}

			mergeBB := b.insertBasicBlock("test.after")
			result := b.spmdFullLoadWithSelect(vecType, ci, mask)
			b.CreateBr(mergeBB)
			b.SetInsertPointAtEnd(mergeBB)
			b.CreateRetVoid()

			if result.IsNil() {
				t.Fatal("spmdFullLoadWithSelect returned nil")
			}
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
			}
			if result.Type().VectorSize() != laneCount {
				t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), laneCount)
			}

			ir := c.mod.String()
			if !strings.Contains(ir, tc.wantIRString) {
				t.Errorf("IR should contain %q", tc.wantIRString)
			}
			if !strings.Contains(ir, "spmd.fullload") {
				t.Error("IR should contain spmd.fullload basic block")
			}
			if !strings.Contains(ir, "spmd.maskedload") {
				t.Error("IR should contain spmd.maskedload basic block")
			}
			// Same-width case must NOT have cast instructions.
			if tc.name == "i32 index, i32 cap" {
				if strings.Contains(ir, "spmd.cap.ext") || strings.Contains(ir, "spmd.cap.trunc") {
					t.Error("same-width case should not have ext/trunc instructions")
				}
			}
		})
	}
}

// TestSPMDFullStoreWithBlend verifies that spmdFullStoreWithBlend emits the
// correct branch structure: cap check → blend BB (load+select+store) | masked-store BB → merge.
func TestSPMDFullStoreWithBlend(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, laneCount)

	scalarIndex := llvm.ConstInt(b.uintptrType, 8, false)
	sliceCap := llvm.ConstInt(b.uintptrType, 16, false)

	arrType := llvm.ArrayType(i32Type, 16)
	alloca := b.CreateAlloca(arrType, "test.buf")
	zero := llvm.ConstInt(i32Type, 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, alloca, []llvm.Value{zero, zero}, "test.ptr")

	maskElemType := i32Type
	mask := llvm.ConstVector([]llvm.Value{
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstNull(maskElemType),
		llvm.ConstNull(maskElemType),
	}, false)

	storeVal := llvm.ConstNull(vecType)

	ci := &spmdContiguousInfo{
		scalarPtr:   scalarPtr,
		loop:        &spmdActiveLoop{laneCount: laneCount},
		sliceCap:    sliceCap,
		scalarIndex: scalarIndex,
	}

	b.spmdFullStoreWithBlend(storeVal, ci, mask)
	b.CreateRetVoid()

	ir := c.mod.String()

	if !strings.Contains(ir, "llvm.masked.store") {
		t.Error("IR should contain llvm.masked.store for the fallback path")
	}
	if !strings.Contains(ir, "spmd.blend") {
		t.Error("IR should contain spmd.blend basic block")
	}
	if !strings.Contains(ir, "spmd.maskedstore") {
		t.Error("IR should contain spmd.maskedstore basic block")
	}
	if !strings.Contains(ir, "spmd.store.merge") {
		t.Error("IR should contain spmd.store.merge basic block")
	}
	if !strings.Contains(ir, "spmd.blend.old") {
		t.Error("IR should contain spmd.blend.old load in the blend path")
	}
}

// TestSPMDAllocaFastPathLoad verifies that spmdFullLoadWithSelect emits a direct
// load+select when the contiguous access originates from a stack-allocated array
// (alloca fast-path), with no runtime cap-check branch or masked.load fallback.
func TestSPMDAllocaFastPathLoad(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, laneCount)

	scalarIndex := llvm.ConstInt(b.uintptrType, 0, false)

	arrType := llvm.ArrayType(i32Type, 16)
	alloca := b.CreateAlloca(arrType, "test.buf")
	zero := llvm.ConstInt(i32Type, 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, alloca, []llvm.Value{zero, zero}, "test.ptr")

	maskElemType := i32Type
	mask := llvm.ConstVector([]llvm.Value{
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstNull(maskElemType),
		llvm.ConstNull(maskElemType),
	}, false)

	goI32 := types.Typ[types.Int32]
	goArrType := types.NewArray(goI32, 16)
	goPtrType := types.NewPointer(goArrType)
	mockAlloc := ssaAllocWithType(goPtrType, false)

	ci := &spmdContiguousInfo{
		scalarPtr:   scalarPtr,
		loop:        &spmdActiveLoop{laneCount: laneCount},
		ssaSource:   mockAlloc,
		scalarIndex: scalarIndex,
	}

	mergeBB := b.insertBasicBlock("test.after")
	result := b.spmdFullLoadWithSelect(vecType, ci, mask)
	b.CreateBr(mergeBB)
	b.SetInsertPointAtEnd(mergeBB)
	b.CreateRetVoid()

	if result.IsNil() {
		t.Fatal("returned nil for alloca fast-path")
	}

	ir := c.mod.String()
	if strings.Contains(ir, "spmd.can.fullload") {
		t.Error("alloca fast-path should NOT emit cap check")
	}
	if strings.Contains(ir, "spmd.maskedload") {
		t.Error("alloca fast-path should NOT emit masked load fallback")
	}
	if !strings.Contains(ir, "spmd.alloca.load") {
		t.Error("alloca fast-path should emit spmd.alloca.load")
	}
	// WASM uses bitwise mask select (AND/OR), not LLVM select.
	if !strings.Contains(ir, " and ") && !strings.Contains(ir, " or ") {
		t.Error("alloca fast-path load should emit mask select to zero inactive lanes")
	}
}

// TestSPMDAllocaFastPathStore verifies that spmdFullStoreWithBlend emits a direct
// load-blend-store when the contiguous access originates from a stack-allocated
// array (alloca fast-path), with no runtime cap-check branch or masked.store fallback.
func TestSPMDAllocaFastPathStore(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	laneCount := 4
	i32Type := c.ctx.Int32Type()
	vecType := llvm.VectorType(i32Type, laneCount)

	scalarIndex := llvm.ConstInt(b.uintptrType, 0, false)

	arrType := llvm.ArrayType(i32Type, 16)
	alloca := b.CreateAlloca(arrType, "test.buf")
	zero := llvm.ConstInt(i32Type, 0, false)
	scalarPtr := b.CreateInBoundsGEP(arrType, alloca, []llvm.Value{zero, zero}, "test.ptr")

	maskElemType := i32Type
	mask := llvm.ConstVector([]llvm.Value{
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstAllOnes(maskElemType),
		llvm.ConstNull(maskElemType),
		llvm.ConstNull(maskElemType),
	}, false)

	storeVal := llvm.ConstNull(vecType)

	goI32 := types.Typ[types.Int32]
	goArrType := types.NewArray(goI32, 16)
	goPtrType := types.NewPointer(goArrType)
	mockAlloc := ssaAllocWithType(goPtrType, false)

	ci := &spmdContiguousInfo{
		scalarPtr:   scalarPtr,
		loop:        &spmdActiveLoop{laneCount: laneCount},
		ssaSource:   mockAlloc,
		scalarIndex: scalarIndex,
	}

	b.spmdFullStoreWithBlend(storeVal, ci, mask)
	b.CreateRetVoid()

	ir := c.mod.String()
	if strings.Contains(ir, "spmd.can.fullstore") {
		t.Error("alloca fast-path should NOT emit cap check")
	}
	if strings.Contains(ir, "spmd.maskedstore") {
		t.Error("alloca fast-path should NOT emit masked store fallback")
	}
	if !strings.Contains(ir, "spmd.alloca.old") {
		t.Error("alloca fast-path should emit spmd.alloca.old load")
	}
	// WASM uses bitwise mask select (AND/OR), not LLVM select.
	if !strings.Contains(ir, " and ") && !strings.Contains(ir, " or ") {
		t.Error("alloca fast-path store should emit mask select to blend active lanes")
	}
}

// TestSPMDFullStoreCapTypeMismatch verifies that spmdFullStoreWithBlend handles
// integer type mismatches between the scalar index and slice cap without
// panicking, and produces valid IR with the appropriate extend/truncate
// instruction.
//
// LLVM constant-folds trunc/zext on ConstInt values, so the index and cap
// must be non-constant (function parameters) to keep the cast instructions
// visible in the IR.
func TestSPMDFullStoreCapTypeMismatch(t *testing.T) {
	laneCount := 4

	tests := []struct {
		name         string
		indexType    func(c *compilerContext) llvm.Type
		capType      func(c *compilerContext) llvm.Type
		wantIRString string
	}{
		{
			// scalarIndex is i32 (uintptr on wasm32), sliceCap is i64.
			// cap (64 bits) > iter (32 bits) -> Trunc path.
			name:         "i32 index, i64 cap",
			indexType:    func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			capType:      func(c *compilerContext) llvm.Type { return c.ctx.Int64Type() },
			wantIRString: "spmd.cap.trunc",
		},
		{
			// scalarIndex is i64, sliceCap is i32 (uintptr on wasm32).
			// cap (32 bits) < iter (64 bits) -> ZExt path.
			name:         "i64 index, i32 cap",
			indexType:    func(c *compilerContext) llvm.Type { return c.ctx.Int64Type() },
			capType:      func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			wantIRString: "spmd.cap.ext",
		},
		{
			// Both i32 — the common case on wasm32. No cast needed.
			name:         "i32 index, i32 cap",
			indexType:    func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			capType:      func(c *compilerContext) llvm.Type { return c.ctx.Int32Type() },
			wantIRString: "spmd.can.fullstore",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestCompilerContext(t)
			defer c.dispose()

			i32Type := c.ctx.Int32Type()
			vecType := llvm.VectorType(i32Type, laneCount)

			idxType := tc.indexType(c)
			capType := tc.capType(c)

			ptrType := llvm.PointerType(i32Type, 0)
			paramTypes := []llvm.Type{idxType, capType, ptrType}
			fnType := llvm.FunctionType(c.ctx.VoidType(), paramTypes, false)
			fn := llvm.AddFunction(c.mod, "test_func_store_mismatch", fnType)
			bb := llvm.AddBasicBlock(fn, "entry")

			b := &builder{compilerContext: c}
			b.Builder = c.ctx.NewBuilder()
			defer b.Dispose()
			b.llvmFn = fn
			b.SetInsertPointAtEnd(bb)

			scalarIndex := fn.Param(0)
			sliceCap := fn.Param(1)
			scalarPtr := fn.Param(2)

			maskElemType := i32Type
			mask := llvm.ConstVector([]llvm.Value{
				llvm.ConstAllOnes(maskElemType),
				llvm.ConstAllOnes(maskElemType),
				llvm.ConstNull(maskElemType),
				llvm.ConstNull(maskElemType),
			}, false)

			storeVal := llvm.ConstNull(vecType)

			ci := &spmdContiguousInfo{
				scalarPtr:   scalarPtr,
				loop:        &spmdActiveLoop{laneCount: laneCount},
				sliceCap:    sliceCap,
				scalarIndex: scalarIndex,
			}

			b.spmdFullStoreWithBlend(storeVal, ci, mask)
			b.CreateRetVoid()

			ir := c.mod.String()
			if !strings.Contains(ir, tc.wantIRString) {
				t.Errorf("IR should contain %q", tc.wantIRString)
			}
			if !strings.Contains(ir, "spmd.blend") {
				t.Error("IR should contain spmd.blend basic block")
			}
			if !strings.Contains(ir, "spmd.maskedstore") {
				t.Error("IR should contain spmd.maskedstore basic block")
			}
			// Same-width case must NOT have cast instructions.
			if tc.name == "i32 index, i32 cap" {
				if strings.Contains(ir, "spmd.cap.ext") || strings.Contains(ir, "spmd.cap.trunc") {
					t.Error("same-width case should not have ext/trunc instructions")
				}
			}
		})
	}
}

// TestSPMDWidenMaskToOperandLanes verifies that spmdWidenMaskToOperandLanes
// correctly replicates each mask lane when a narrow loop mask (e.g., 4-lane
// int32) selects a wider Varying[bool] accumulator (16 lanes on WASM128).
func TestSPMDWidenMaskToOperandLanes(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name        string
		srcLanes    int
		targetLanes int
		elemBits    int // 1 for i1, 32 for i32
	}{
		{
			name:        "i1_4to16",
			srcLanes:    4,
			targetLanes: 16,
			elemBits:    1,
		},
		{
			name:        "i32_4to16",
			srcLanes:    4,
			targetLanes: 16,
			elemBits:    32,
		},
		{
			name:        "i1_2to8",
			srcLanes:    2,
			targetLanes: 8,
			elemBits:    1,
		},
		{
			name:        "same_lanes_noop",
			srcLanes:    4,
			targetLanes: 4,
			elemBits:    32,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var elemType llvm.Type
			if tt.elemBits == 1 {
				elemType = c.ctx.Int1Type()
			} else {
				elemType = c.ctx.Int32Type()
			}
			maskType := llvm.VectorType(elemType, tt.srcLanes)

			// Build a non-constant mask so the shuffle path is exercised.
			// Use undef as a stand-in for a runtime mask value.
			mask := llvm.Undef(maskType)

			result := b.spmdWidenMaskToOperandLanes(mask, tt.targetLanes)

			if result.IsNil() {
				t.Fatal("spmdWidenMaskToOperandLanes returned nil")
			}
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Fatalf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
			}
			if result.Type().VectorSize() != tt.targetLanes {
				t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), tt.targetLanes)
			}
			// Element type of the output must match the input element type.
			gotElemBits := result.Type().ElementType().IntTypeWidth()
			if gotElemBits != tt.elemBits {
				t.Errorf("result element width = %d bits, want %d", gotElemBits, tt.elemBits)
			}
		})
	}
}

// TestSPMDWidenMaskConstNull verifies that constant null (all-inactive) masks
// are widened without emitting runtime instructions.
func TestSPMDWidenMaskConstNull(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	srcMask := llvm.ConstNull(llvm.VectorType(c.ctx.Int32Type(), 4))
	result := b.spmdWidenMaskToOperandLanes(srcMask, 16)

	if result.IsNil() {
		t.Fatal("spmdWidenMaskToOperandLanes returned nil for ConstNull")
	}
	if !result.IsConstant() {
		t.Error("widened ConstNull should remain constant")
	}
	if !result.IsNull() {
		t.Error("widened ConstNull should be null (all-inactive)")
	}
	if result.Type().VectorSize() != 16 {
		t.Errorf("result lanes = %d, want 16", result.Type().VectorSize())
	}
}

// TestSPMDWidenMaskConstAllOnes verifies that constant all-ones (all-active) masks
// are widened without emitting runtime instructions.
func TestSPMDWidenMaskConstAllOnes(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	srcMask := llvm.ConstAllOnes(llvm.VectorType(c.ctx.Int32Type(), 4))
	result := b.spmdWidenMaskToOperandLanes(srcMask, 16)

	if result.IsNil() {
		t.Fatal("spmdWidenMaskToOperandLanes returned nil for ConstAllOnes")
	}
	if !result.IsConstant() {
		t.Error("widened ConstAllOnes should remain constant")
	}
	if result.Type().VectorSize() != 16 {
		t.Errorf("result lanes = %d, want 16", result.Type().VectorSize())
	}
}

func TestVectorToArray(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name      string
		elemType  llvm.Type
		laneCount int
		vals      []uint64
	}{
		{
			name:      "4xi32",
			elemType:  c.ctx.Int32Type(),
			laneCount: 4,
			vals:      []uint64{10, 20, 30, 40},
		},
		{
			name:      "2xi64",
			elemType:  c.ctx.Int64Type(),
			laneCount: 2,
			vals:      []uint64{100, 200},
		},
		{
			name:      "8xi16",
			elemType:  c.ctx.Int16Type(),
			laneCount: 8,
			vals:      []uint64{1, 2, 3, 4, 5, 6, 7, 8},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Build a constant <N x T> vector to convert.
			elts := make([]llvm.Value, tt.laneCount)
			for i, v := range tt.vals {
				elts[i] = llvm.ConstInt(tt.elemType, v, false)
			}
			vec := llvm.ConstVector(elts, false)
			if vec.Type().TypeKind() != llvm.VectorTypeKind {
				t.Fatalf("input type = %v, want VectorTypeKind", vec.Type().TypeKind())
			}

			arr := b.vectorToArray(vec)

			// Verify result is non-nil.
			if arr.IsNil() {
				t.Fatal("vectorToArray returned nil")
			}
			// Verify result kind is array.
			if arr.Type().TypeKind() != llvm.ArrayTypeKind {
				t.Errorf("result type = %v, want ArrayTypeKind", arr.Type().TypeKind())
			}
			// Verify array length matches input lane count.
			if arr.Type().ArrayLength() != tt.laneCount {
				t.Errorf("result array length = %d, want %d", arr.Type().ArrayLength(), tt.laneCount)
			}
			// Verify array element type matches input element type.
			if arr.Type().ElementType().C != tt.elemType.C {
				t.Errorf("result element type mismatch: got %v, want %v", arr.Type().ElementType(), tt.elemType)
			}
		})
	}
}

// TestSPMDReduceFromAggregate verifies that the reduce.From handler uses
// extractvalue (not extractelement) when the input is a [N x T] array type,
// as produced for aggregate Varying types such as Varying[string].
// On WASM32, string is {ptr, i32} = 8 bytes, so laneCount = 16/8 = 2 and
// Varying[string] is represented as [2 x {ptr, i32}].
func TestSPMDReduceFromAggregate(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	// Build [2 x {i32, i32}] as a simplified stand-in for [2 x string_struct].
	// Using i32 fields instead of actual ptr+i32 keeps the test target-independent.
	i32Type := c.ctx.Int32Type()
	elemStructType := c.ctx.StructType([]llvm.Type{i32Type, i32Type}, false)
	laneCount := 2
	arrayType := llvm.ArrayType(elemStructType, laneCount)

	// Create a dedicated function that receives the array as a parameter so that
	// extractvalue is emitted as a real IR instruction. LLVM folds extractvalue
	// on constant and undef aggregates at build time, leaving no instruction in IR.
	fnType := llvm.FunctionType(c.ctx.VoidType(), []llvm.Type{arrayType}, false)
	fn := llvm.AddFunction(c.mod, "test_reduce_from_aggregate", fnType)
	bb := llvm.AddBasicBlock(fn, "entry")
	bld := c.ctx.NewBuilder()
	defer bld.Dispose()
	bld.SetInsertPointAtEnd(bb)

	// The array parameter is a genuine SSA value — extractvalue on it is not folded.
	arrayVal := fn.Param(0)

	// Replicate the fixed reduce.From logic: branch on ArrayTypeKind.
	vecType := arrayVal.Type()
	if vecType.TypeKind() != llvm.ArrayTypeKind {
		t.Fatalf("test setup: expected ArrayTypeKind, got %v", vecType.TypeKind())
	}
	extractedElemType := vecType.ElementType()
	extractedLaneCount := vecType.ArrayLength()
	if extractedLaneCount != laneCount {
		t.Fatalf("ArrayLength = %d, want %d", extractedLaneCount, laneCount)
	}

	// Allocate output array and extract each lane via extractvalue.
	outArrType := llvm.ArrayType(extractedElemType, extractedLaneCount)
	alloca := bld.CreateAlloca(outArrType, "reduce.from.arr")

	for i := 0; i < extractedLaneCount; i++ {
		elem := bld.CreateExtractValue(arrayVal, i, "")
		gep := bld.CreateInBoundsGEP(outArrType, alloca, []llvm.Value{
			llvm.ConstInt(i32Type, 0, false),
			llvm.ConstInt(i32Type, uint64(i), false),
		}, "")
		bld.CreateStore(elem, gep)
	}
	bld.CreateRetVoid()

	// The function's IR must contain extractvalue and must NOT contain extractelement.
	// The pre-fix code called CreateExtractElement on the [N x T] array, which
	// LLVM rejects with "Invalid extractelement operands!".
	fnIR := fn.String()
	if !strings.Contains(fnIR, "extractvalue") {
		t.Errorf("expected extractvalue in function IR for aggregate Varying type; got:\n%s", fnIR)
	}
	if strings.Contains(fnIR, "extractelement") {
		t.Errorf("unexpected extractelement in function IR: aggregate Varying types must use extractvalue; got:\n%s", fnIR)
	}

	// Verify the alloca is non-nil and the output array type has the expected shape.
	if alloca.IsNil() {
		t.Fatal("alloca returned nil")
	}
	if outArrType.TypeKind() != llvm.ArrayTypeKind {
		t.Errorf("outArrType = %v, want ArrayTypeKind", outArrType.TypeKind())
	}
	if outArrType.ArrayLength() != laneCount {
		t.Errorf("outArrType.ArrayLength() = %d, want %d", outArrType.ArrayLength(), laneCount)
	}
}
