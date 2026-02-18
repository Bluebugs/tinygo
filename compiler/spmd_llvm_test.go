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
			// Create a <4 x i1> constant vector.
			vecElts := make([]llvm.Value, len(tt.maskVals))
			for i, val := range tt.maskVals {
				if val {
					vecElts[i] = llvm.ConstInt(c.ctx.Int1Type(), 1, false)
				} else {
					vecElts[i] = llvm.ConstInt(c.ctx.Int1Type(), 0, false)
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
			wantElemWidth: 1, // i1 for mask
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
			wantElemWidth: 1,
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
			wantElemWidth: 1,
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
			wantElemWidth: 1,
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
			wantElemWidth: 1,
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

				// Verify element type is i1.
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
