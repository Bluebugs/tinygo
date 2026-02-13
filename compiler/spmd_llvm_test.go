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
		{"int8", c.ctx.Int8Type(), 16},   // 128 bits / 8 bits = 16
		{"int16", c.ctx.Int16Type(), 8},  // 128 bits / 16 bits = 8
		{"int32", c.ctx.Int32Type(), 4},  // 128 bits / 32 bits = 4
		{"int64", c.ctx.Int64Type(), 2},  // 128 bits / 64 bits = 2
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
