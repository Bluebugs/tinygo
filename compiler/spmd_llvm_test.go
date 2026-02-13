// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/constant"
	"go/token"
	"go/types"
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
			name:         "varying_int32_constrained",
			goType:       types.NewVaryingConstrained(types.Typ[types.Int32], 4),
			wantLanes:    4, // WASM SIMD128 width, constraint ignored at LLVM level
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
