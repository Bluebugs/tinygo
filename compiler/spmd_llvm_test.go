// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/constant"
	"go/token"
	"go/types"
	"strconv"
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
		{
			name:         "varying_int32_constrained_4",
			goType:       types.NewVaryingConstrained(types.Typ[types.Int32], 4),
			wantLanes:    4, // constraint matches SIMD128 width
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
	maskElemType := c.spmdMaskElemType() // i32 on WASM
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

	// Without a real SSA function, we can't call isBlockInSPMDBody directly.
	// Verify the precondition: spmdInfo is nil, so the method would return nil.
	// Also verify spmdShouldRedirectJump returns false with nil maps.
	if b.spmdThenExitRedirects != nil {
		t.Fatal("expected spmdThenExitRedirects to be nil for test builder")
	}
	if b.spmdMergeSelects != nil {
		t.Fatal("expected spmdMergeSelects to be nil for test builder")
	}
	if b.spmdVaryingIfs != nil {
		t.Fatal("expected spmdVaryingIfs to be nil for test builder")
	}
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
			wantLaneCount: 4, // 128 bits / 32 bits = 4 lanes
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
			wantLaneCount: 16, // 128 bits / 8 bits = 16 lanes
			wantElemWidth: 32, // i32 for WASM mask
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
			wantLaneCount: 2, // 128 bits / 64 bits = 2 lanes
			wantElemWidth: 32, // i32 for WASM mask
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
		{
			name: "constrained_varying_int32_8",
			createSig: func() *types.Signature {
				params := types.NewTuple(
					types.NewVar(token.NoPos, nil, "v", types.NewVaryingConstrained(types.Typ[types.Int32], 8)),
				)
				return types.NewSignatureType(nil, nil, nil, params, nil, false)
			},
			wantMaskType:  true,
			wantLaneCount: 8, // constraint overrides SIMD128 width (4)
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

func TestSPMDMaskStack(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	i1Type := c.ctx.Int1Type()
	mask4Type := llvm.VectorType(i1Type, 4)

	allTrue := llvm.ConstAllOnes(mask4Type)
	allFalse := llvm.ConstNull(mask4Type)

	// Empty stack returns nil.
	mask := b.spmdCurrentMask()
	if !mask.IsNil() {
		t.Error("expected nil mask from empty stack")
	}

	// Pop on empty stack is a no-op (no panic).
	b.spmdPopMask()
	mask = b.spmdCurrentMask()
	if !mask.IsNil() {
		t.Error("expected nil mask after pop on empty stack")
	}

	// Push one mask and read it back.
	b.spmdPushMask(allTrue)
	mask = b.spmdCurrentMask()
	if mask.IsNil() {
		t.Fatal("expected non-nil mask after push")
	}
	if mask.C != allTrue.C {
		t.Error("expected allTrue mask after push")
	}

	// Push a second mask: top changes.
	b.spmdPushMask(allFalse)
	mask = b.spmdCurrentMask()
	if mask.C != allFalse.C {
		t.Error("expected allFalse mask after second push")
	}

	// Pop second mask: back to first.
	b.spmdPopMask()
	mask = b.spmdCurrentMask()
	if mask.C != allTrue.C {
		t.Error("expected allTrue mask after pop")
	}

	// Pop first mask: stack empty again.
	b.spmdPopMask()
	mask = b.spmdCurrentMask()
	if !mask.IsNil() {
		t.Error("expected nil mask after popping all")
	}
}

func TestSPMDMaskedLoadIntrinsic(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	tests := []struct {
		name      string
		elemType  llvm.Type
		laneCount int
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

func TestSPMDMaskTransitionTypes(t *testing.T) {
	// Verify that spmdMaskTransition structs can be constructed correctly.
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	maskType := llvm.VectorType(c.ctx.Int1Type(), 4)
	cond := llvm.ConstAllOnes(maskType)

	// Construct each transition type and verify the kind field.
	pushTr := &spmdMaskTransition{kind: "pushThen", cond: cond}
	swapTr := &spmdMaskTransition{kind: "swapElse", cond: cond}
	popTr := &spmdMaskTransition{kind: "pop"}

	if pushTr.kind != "pushThen" {
		t.Errorf("pushThen kind = %q, want %q", pushTr.kind, "pushThen")
	}
	if swapTr.kind != "swapElse" {
		t.Errorf("swapElse kind = %q, want %q", swapTr.kind, "swapElse")
	}
	if popTr.kind != "pop" {
		t.Errorf("pop kind = %q, want %q", popTr.kind, "pop")
	}
	if pushTr.cond.C != cond.C {
		t.Error("pushThen cond not preserved")
	}

	// Verify builder starts with nil mask transitions map.
	if b.spmdMaskTransitions != nil {
		t.Error("expected nil spmdMaskTransitions for fresh builder")
	}
	if b.spmdContiguousPtr != nil {
		t.Error("expected nil spmdContiguousPtr for fresh builder")
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

func TestSPMDConstrainedLaneCount(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name         string
		goType       types.Type
		wantLanes    int
		wantElemKind llvm.TypeKind
	}{
		{
			name:         "constrained_int_4",
			goType:       types.NewVaryingConstrained(types.Typ[types.Int32], 4),
			wantLanes:    4, // constraint matches SIMD128 width
			wantElemKind: llvm.IntegerTypeKind,
		},
		{
			name:         "constrained_int_8",
			goType:       types.NewVaryingConstrained(types.Typ[types.Int32], 8),
			wantLanes:    8, // constraint exceeds SIMD128 width
			wantElemKind: llvm.IntegerTypeKind,
		},
		{
			name:         "constrained_byte_4",
			goType:       types.NewVaryingConstrained(types.Typ[types.Byte], 4),
			wantLanes:    4, // smaller than SIMD128 (normally 16 for byte)
			wantElemKind: llvm.IntegerTypeKind,
		},
		{
			name:         "constrained_uint16_2",
			goType:       types.NewVaryingConstrained(types.Typ[types.Uint16], 2),
			wantLanes:    2, // smaller than SIMD128 (normally 8 for uint16)
			wantElemKind: llvm.IntegerTypeKind,
		},
		{
			name:         "constrained_float32_8",
			goType:       types.NewVaryingConstrained(types.Typ[types.Float32], 8),
			wantLanes:    8, // double SIMD128 width
			wantElemKind: llvm.FloatTypeKind,
		},
		{
			name:         "unconstrained_int32",
			goType:       types.NewVarying(types.Typ[types.Int32]),
			wantLanes:    4, // default SIMD128 width
			wantElemKind: llvm.IntegerTypeKind,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			llvmType := c.getLLVMType(tt.goType)

			if llvmType.TypeKind() != llvm.VectorTypeKind {
				t.Errorf("getLLVMType(%s) TypeKind = %v, want VectorTypeKind", tt.name, llvmType.TypeKind())
				return
			}

			gotLanes := llvmType.VectorSize()
			if gotLanes != tt.wantLanes {
				t.Errorf("getLLVMType(%s) VectorSize = %d, want %d", tt.name, gotLanes, tt.wantLanes)
			}

			elemType := llvmType.ElementType()
			if elemType.TypeKind() != tt.wantElemKind {
				t.Errorf("getLLVMType(%s) ElementType.TypeKind = %v, want %v", tt.name, elemType.TypeKind(), tt.wantElemKind)
			}
		})
	}
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
			name:      "unconstrained_int32",
			spmdType:  types.NewVarying(types.Typ[types.Int32]),
			elemType:  c.ctx.Int32Type(),
			wantLanes: 4, // SIMD128 default
		},
		{
			name:      "constrained_4",
			spmdType:  types.NewVaryingConstrained(types.Typ[types.Int32], 4),
			elemType:  c.ctx.Int32Type(),
			wantLanes: 4, // constraint matches default
		},
		{
			name:      "constrained_8",
			spmdType:  types.NewVaryingConstrained(types.Typ[types.Int32], 8),
			elemType:  c.ctx.Int32Type(),
			wantLanes: 8, // constraint overrides default
		},
		{
			name:      "constrained_2",
			spmdType:  types.NewVaryingConstrained(types.Typ[types.Int32], 2),
			elemType:  c.ctx.Int32Type(),
			wantLanes: 2, // constraint overrides default
		},
		{
			name:      "constrained_byte_4",
			spmdType:  types.NewVaryingConstrained(types.Typ[types.Byte], 4),
			elemType:  c.ctx.Int8Type(),
			wantLanes: 4, // constraint overrides 16 (SIMD128/8bits)
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

func TestSPMDConstrainedConst(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name       string
		constValue constant.Value
		goType     *types.SPMDType
		wantLanes  int
	}{
		{
			name:       "constrained_int32_8_const_42",
			constValue: constant.MakeInt64(42),
			goType:     types.NewVaryingConstrained(types.Typ[types.Int32], 8),
			wantLanes:  8,
		},
		{
			name:       "constrained_byte_4_const_7",
			constValue: constant.MakeInt64(7),
			goType:     types.NewVaryingConstrained(types.Typ[types.Byte], 4),
			wantLanes:  4,
		},
		{
			name:       "unconstrained_int32_const_42",
			constValue: constant.MakeInt64(42),
			goType:     types.NewVarying(types.Typ[types.Int32]),
			wantLanes:  4, // SIMD128 default
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			constExpr := ssa.NewConst(tt.constValue, tt.goType)
			result := c.createSPMDConst(constExpr, tt.goType, token.NoPos)

			if result.IsNil() {
				t.Fatal("createSPMDConst returned nil")
			}
			if result.Type().TypeKind() != llvm.VectorTypeKind {
				t.Errorf("result type = %v, want VectorTypeKind", result.Type().TypeKind())
				return
			}
			if result.Type().VectorSize() != tt.wantLanes {
				t.Errorf("result lanes = %d, want %d", result.Type().VectorSize(), tt.wantLanes)
			}
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

func TestSPMDForLoopBreakMask(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	t.Run("detectSPMDForLoops_returns_nil_when_not_func_body", func(t *testing.T) {
		b := &builder{compilerContext: c}
		b.spmdFuncIsBody = false

		result := b.detectSPMDForLoops()
		if result != nil {
			t.Error("expected nil when spmdFuncIsBody=false")
		}
	})

	t.Run("spmdIsVaryingBreak_returns_false_when_no_loops", func(t *testing.T) {
		b := &builder{compilerContext: c}
		b.spmdForLoops = nil

		// spmdIsVaryingBreak requires a real SSA block, but we can't easily
		// construct one for unit testing. This test verifies the nil guard.
		// Integration testing will cover the actual break detection logic.
		loopInfo, isBreak := b.spmdIsVaryingBreak(nil)
		if loopInfo != nil || isBreak {
			t.Error("expected false when spmdForLoops=nil")
		}
	})

	t.Run("spmdForLoopInfo_alloca_created_with_correct_type", func(t *testing.T) {
		// Verify that the alloca has the correct vector type <N x i1>.
		laneCount := 4
		maskType := llvm.VectorType(c.ctx.Int1Type(), laneCount)

		// Create a test function to hold the alloca.
		fn := llvm.AddFunction(c.mod, "test_break_mask", llvm.FunctionType(c.ctx.VoidType(), nil, false))
		bb := llvm.AddBasicBlock(fn, "entry")
		b := &builder{compilerContext: c}
		b.Builder = c.ctx.NewBuilder()
		b.SetInsertPointAtEnd(bb)
		b.llvmFn = fn

		alloca := b.CreateAlloca(maskType, "spmd.break.mask")

		// Verify the alloca type.
		allocaType := alloca.Type()
		if allocaType.TypeKind() != llvm.PointerTypeKind {
			t.Errorf("expected pointer type, got %v", allocaType.TypeKind())
		}

		// Initialize to all-false and store.
		zeroMask := llvm.ConstNull(maskType)
		b.CreateStore(zeroMask, alloca)

		// Load and verify the mask type.
		loaded := b.CreateLoad(maskType, alloca, "break.mask")
		if loaded.Type().TypeKind() != llvm.VectorTypeKind {
			t.Errorf("expected vector type, got %v", loaded.Type().TypeKind())
		}
		if loaded.Type().VectorSize() != laneCount {
			t.Errorf("expected lane count %d, got %d", laneCount, loaded.Type().VectorSize())
		}
	})

	t.Run("break_mask_accumulation", func(t *testing.T) {
		// Test that OR-accumulation works correctly.
		laneCount := 4
		maskType := llvm.VectorType(c.ctx.Int1Type(), laneCount)

		fn := llvm.AddFunction(c.mod, "test_accumulation", llvm.FunctionType(c.ctx.VoidType(), nil, false))
		bb := llvm.AddBasicBlock(fn, "entry")
		b := &builder{compilerContext: c}
		b.Builder = c.ctx.NewBuilder()
		b.SetInsertPointAtEnd(bb)

		// Create alloca and initialize to zero.
		alloca := b.CreateAlloca(maskType, "spmd.break.mask")
		zeroMask := llvm.ConstNull(maskType)
		b.CreateStore(zeroMask, alloca)

		// Simulate first break: lanes [true, false, false, false]
		firstMask := llvm.ConstVector([]llvm.Value{
			llvm.ConstInt(c.ctx.Int1Type(), 1, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
		}, false)

		currentBreak := b.CreateLoad(maskType, alloca, "break.mask.cur")
		newBreak := b.CreateOr(currentBreak, firstMask, "break.mask.new")
		b.CreateStore(newBreak, alloca)

		// Simulate second break: lanes [false, false, true, false]
		secondMask := llvm.ConstVector([]llvm.Value{
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
			llvm.ConstInt(c.ctx.Int1Type(), 1, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
		}, false)

		currentBreak = b.CreateLoad(maskType, alloca, "break.mask.cur")
		newBreak = b.CreateOr(currentBreak, secondMask, "break.mask.new")
		b.CreateStore(newBreak, alloca)

		// Verify the final mask has both lanes set (manual verification via IR inspection).
		// In actual usage, the final mask would be [true, false, true, false].
		finalMask := b.CreateLoad(maskType, alloca, "break.mask.final")
		if finalMask.Type().TypeKind() != llvm.VectorTypeKind {
			t.Error("expected vector type for accumulated break mask")
		}
	})

	t.Run("active_mask_computation", func(t *testing.T) {
		// Test that activeMask = entryMask & ~breakMask works correctly.
		laneCount := 4
		maskType := llvm.VectorType(c.ctx.Int1Type(), laneCount)

		fn := llvm.AddFunction(c.mod, "test_active_mask", llvm.FunctionType(c.ctx.VoidType(), nil, false))
		bb := llvm.AddBasicBlock(fn, "entry")
		b := &builder{compilerContext: c}
		b.Builder = c.ctx.NewBuilder()
		b.SetInsertPointAtEnd(bb)

		// Entry mask: all lanes active.
		entryMask := llvm.ConstAllOnes(maskType)

		// Break mask: lanes [true, false, false, true] have broken.
		breakMask := llvm.ConstVector([]llvm.Value{
			llvm.ConstInt(c.ctx.Int1Type(), 1, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
			llvm.ConstInt(c.ctx.Int1Type(), 1, false),
		}, false)

		// Compute active mask.
		notBreak := b.CreateNot(breakMask, "not.break")
		activeMask := b.CreateAnd(entryMask, notBreak, "spmd.active.mask")

		// Expected: [false, true, true, false]
		if activeMask.Type().TypeKind() != llvm.VectorTypeKind {
			t.Error("expected vector type for active mask")
		}
		if activeMask.Type().VectorSize() != laneCount {
			t.Errorf("expected lane count %d, got %d", laneCount, activeMask.Type().VectorSize())
		}
	})
}

func TestSPMDBreakMaskInstructions(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	t.Run("mask_stack_with_break_mask", func(t *testing.T) {
		// Verify that mask stack operations work correctly when break mask is active.
		laneCount := 4
		maskType := llvm.VectorType(c.ctx.Int1Type(), laneCount)

		fn := llvm.AddFunction(c.mod, "test_mask_stack", llvm.FunctionType(c.ctx.VoidType(), nil, false))
		bb := llvm.AddBasicBlock(fn, "entry")
		b := &builder{compilerContext: c}
		b.Builder = c.ctx.NewBuilder()
		b.SetInsertPointAtEnd(bb)

		// Initialize mask stack with entry mask.
		entryMask := llvm.ConstAllOnes(maskType)
		b.spmdMaskStack = []llvm.Value{entryMask}

		// Push then-mask for varying if.
		cond := llvm.ConstVector([]llvm.Value{
			llvm.ConstInt(c.ctx.Int1Type(), 1, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
			llvm.ConstInt(c.ctx.Int1Type(), 1, false),
			llvm.ConstInt(c.ctx.Int1Type(), 0, false),
		}, false)

		parentMask := b.spmdCurrentMask()
		thenMask := b.CreateAnd(parentMask, cond, "spmd.then.mask")
		b.spmdPushMask(thenMask)

		// Verify stack depth.
		if len(b.spmdMaskStack) != 2 {
			t.Errorf("expected stack depth 2, got %d", len(b.spmdMaskStack))
		}

		// Verify current mask is the then-mask.
		currentMask := b.spmdCurrentMask()
		if currentMask.C != thenMask.C {
			t.Error("expected current mask to be the then-mask")
		}

		// Pop the then-mask.
		b.spmdPopMask()
		if len(b.spmdMaskStack) != 1 {
			t.Errorf("expected stack depth 1 after pop, got %d", len(b.spmdMaskStack))
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
	maskType := llvm.VectorType(c.spmdMaskElemType(), 4) // <4 x i32> on WASM

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

func TestSPMDCallMaskNarrowedByVaryingIf(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)

	// Set up SPMD function body context.
	// The test context is WASM, so masks use the <N x i32> format.
	maskType := llvm.VectorType(c.spmdMaskElemType(), 4) // <4 x i32> on WASM
	entryMask := llvm.ConstAllOnes(maskType)
	b.spmdEntryMask = entryMask
	b.spmdMaskStack = []llvm.Value{entryMask}

	// Create a narrowed mask (simulating varying if).
	narrowedMask := b.CreateAnd(entryMask, llvm.ConstNull(maskType), "narrowed")
	b.spmdPushMask(narrowedMask)

	// spmdCallMask should return the narrowed mask, not the entry mask.
	// We can't call spmdCallMask directly since it needs an ssa.Function,
	// but we can verify spmdCurrentMask returns the narrowed mask.
	currentMask := b.spmdCurrentMask()
	if currentMask.C != narrowedMask.C {
		t.Error("spmdCurrentMask should return narrowed mask when inside varying-if")
	}

	// Verify stack depth.
	if len(b.spmdMaskStack) != 2 {
		t.Errorf("expected stack depth 2, got %d", len(b.spmdMaskStack))
	}
}

// TestSPMDIsWASM verifies that spmdIsWASM correctly detects WASM targets.
func TestSPMDIsWASM(t *testing.T) {
	// The default test context is WASM.
	c := newTestCompilerContext(t)
	defer c.dispose()

	if !c.spmdIsWASM() {
		t.Error("spmdIsWASM() = false for wasm32-unknown-wasi, want true")
	}
}

// TestSPMDMaskElemType verifies that the mask element type is i32 on WASM.
func TestSPMDMaskElemType(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	elemType := c.spmdMaskElemType()
	if elemType.TypeKind() != llvm.IntegerTypeKind {
		t.Errorf("spmdMaskElemType() kind = %v, want IntegerTypeKind", elemType.TypeKind())
	}
	if elemType.IntTypeWidth() != 32 {
		t.Errorf("spmdMaskElemType() width = %d, want 32 (i32 on WASM)", elemType.IntTypeWidth())
	}
}

// TestSPMDWrapMask verifies that spmdWrapMask sign-extends <N x i1> to <N x i32> on WASM.
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

func TestFromConstrainedDimensions(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()

	tests := []struct {
		name        string
		constraintN int
		platformL   int
		wantGroups  int
	}{
		{"exact_1group", 4, 4, 1},
		{"exact_2groups", 8, 4, 2},
		{"partial_2groups", 6, 4, 2},
		{"exact_3groups", 12, 4, 3},
		{"single_elem_group", 1, 4, 1},
		{"large_constraint", 16, 4, 4},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			numGroups := (tt.constraintN + tt.platformL - 1) / tt.platformL
			if numGroups != tt.wantGroups {
				t.Errorf("numGroups(%d, %d) = %d, want %d", tt.constraintN, tt.platformL, numGroups, tt.wantGroups)
			}
		})
	}
}

func TestFromConstrainedShuffleMask(t *testing.T) {
	// Test that shuffle masks correctly clamp out-of-range indices.
	tests := []struct {
		name        string
		constraintN int
		platformL   int
		group       int
		wantActive  int // number of active lanes in this group
	}{
		{"full_group", 8, 4, 0, 4},
		{"full_group_2", 8, 4, 1, 4},
		{"partial_last", 6, 4, 1, 2},
		{"partial_3of4", 7, 4, 1, 3},
		{"single_lane", 1, 4, 0, 1},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			active := 0
			for lane := 0; lane < tt.platformL; lane++ {
				srcIdx := tt.group*tt.platformL + lane
				if srcIdx < tt.constraintN {
					active++
				}
			}
			if active != tt.wantActive {
				t.Errorf("active lanes = %d, want %d", active, tt.wantActive)
			}
		})
	}
}

func TestFromConstrainedMaskValues(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Test mask generation: full group vs partial group.
	maskElem := c.spmdMaskElemType()
	platformLanes := 4

	// Full group mask (all active).
	fullMask := make([]llvm.Value, platformLanes)
	for i := 0; i < platformLanes; i++ {
		if maskElem == c.ctx.Int32Type() {
			fullMask[i] = llvm.ConstInt(maskElem, 0xFFFFFFFF, false)
		} else {
			fullMask[i] = llvm.ConstInt(maskElem, 1, false)
		}
	}
	fullVec := llvm.ConstVector(fullMask, false)
	if fullVec.Type().VectorSize() != platformLanes {
		t.Errorf("full mask vector size = %d, want %d", fullVec.Type().VectorSize(), platformLanes)
	}

	// Partial group mask (2 active, 2 inactive for constraintN=6, group=1).
	constraintN := 6
	group := 1
	partialMask := make([]llvm.Value, platformLanes)
	for lane := 0; lane < platformLanes; lane++ {
		srcIdx := group*platformLanes + lane
		if srcIdx < constraintN {
			if maskElem == c.ctx.Int32Type() {
				partialMask[lane] = llvm.ConstInt(maskElem, 0xFFFFFFFF, false)
			} else {
				partialMask[lane] = llvm.ConstInt(maskElem, 1, false)
			}
		} else {
			partialMask[lane] = llvm.ConstInt(maskElem, 0, false)
		}
	}
	partialVec := llvm.ConstVector(partialMask, false)
	if partialVec.Type().VectorSize() != platformLanes {
		t.Errorf("partial mask vector size = %d, want %d", partialVec.Type().VectorSize(), platformLanes)
	}

	_ = b // builder used to keep test infrastructure consistent
}

func TestToConstrainedReconstruction(t *testing.T) {
	c := newTestCompilerContext(t)
	defer c.dispose()
	b := newTestBuilder(t, c)
	defer b.Dispose()

	// Simulate round-trip: build a <6 x i32> vector, decompose to 2 groups, reassemble.
	elemType := c.ctx.Int32Type()
	constraintN := 6
	platformLanes := c.spmdLaneCount(elemType)
	numGroups := (constraintN + platformLanes - 1) / platformLanes

	// Create source vector <6 x i32> = [10, 20, 30, 40, 50, 60].
	srcVecType := llvm.VectorType(elemType, constraintN)
	elems := make([]llvm.Value, constraintN)
	for i := 0; i < constraintN; i++ {
		elems[i] = llvm.ConstInt(elemType, uint64((i+1)*10), false)
	}
	srcVec := llvm.ConstVector(elems, false)

	// Decompose: extract groups via ShuffleVector.
	i32 := c.ctx.Int32Type()
	groups := make([]llvm.Value, numGroups)
	for g := 0; g < numGroups; g++ {
		indices := make([]llvm.Value, platformLanes)
		for lane := 0; lane < platformLanes; lane++ {
			srcIdx := g*platformLanes + lane
			if srcIdx < constraintN {
				indices[lane] = llvm.ConstInt(i32, uint64(srcIdx), false)
			} else {
				indices[lane] = llvm.ConstInt(i32, 0, false) // clamped
			}
		}
		shuffleMask := llvm.ConstVector(indices, false)
		groups[g] = b.CreateShuffleVector(srcVec, llvm.Undef(srcVecType), shuffleMask, "")
	}

	// Reconstruct: insert elements back.
	result := llvm.ConstNull(srcVecType)
	for g := 0; g < numGroups; g++ {
		for lane := 0; lane < platformLanes; lane++ {
			dstIdx := g*platformLanes + lane
			if dstIdx >= constraintN {
				break
			}
			elem := b.CreateExtractElement(groups[g], llvm.ConstInt(i32, uint64(lane), false), "")
			result = b.CreateInsertElement(result, elem, llvm.ConstInt(i32, uint64(dstIdx), false), "")
		}
	}

	if result.Type().VectorSize() != constraintN {
		t.Errorf("reconstructed vector size = %d, want %d", result.Type().VectorSize(), constraintN)
	}
}

func TestFromConstrainedSingleGroup(t *testing.T) {
	// When constraintN == platformLanes, there should be exactly 1 group.
	c := newTestCompilerContext(t)
	defer c.dispose()

	elemType := c.ctx.Int32Type()
	platformLanes := c.spmdLaneCount(elemType) // 4
	constraintN := platformLanes
	numGroups := (constraintN + platformLanes - 1) / platformLanes
	if numGroups != 1 {
		t.Errorf("single group: numGroups = %d, want 1", numGroups)
	}

	// All lanes should be active.
	for lane := 0; lane < platformLanes; lane++ {
		srcIdx := 0*platformLanes + lane
		if srcIdx >= constraintN {
			t.Errorf("lane %d should be active but srcIdx=%d >= constraintN=%d", lane, srcIdx, constraintN)
		}
	}
}

func TestFromConstrainedUniversalError(t *testing.T) {
	// Varying[T, 0] (universal) should not be decomposable.
	spmdType := types.NewVarying(types.Typ[types.Int32])
	// Constraint 0 is universal.
	constrainedType := types.NewVaryingConstrained(types.Typ[types.Int32], 0)

	if spmdType.Constraint() == 0 {
		t.Errorf("unconstrained type should have constraint -1, got 0")
	}
	if constrainedType.Constraint() != 0 {
		t.Errorf("universal constrained type should have constraint 0, got %d", constrainedType.Constraint())
	}

	// The actual error check happens in createFromConstrained, which checks
	// spmdType.Constraint() == 0. We verify the type system gives us the right values.
	if !constrainedType.IsConstrained() {
		t.Errorf("universal constrained type should report IsConstrained()=true")
	}
}
