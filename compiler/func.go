package compiler

// This file implements function values and closures. It may need some lowering
// in a later step, see func-lowering.go.

import (
	"go/types"

	"golang.org/x/tools/go/ssa"
	"tinygo.org/x/go-llvm"
)

// createFuncValue creates a function value from a raw function pointer with no
// context.
func (b *builder) createFuncValue(funcPtr, context llvm.Value, sig *types.Signature) llvm.Value {
	// Closure is: {context, function pointer}
	funcValueType := b.getFuncType(sig)
	funcValue := llvm.Undef(funcValueType)
	funcValue = b.CreateInsertValue(funcValue, context, 0, "")
	funcValue = b.CreateInsertValue(funcValue, funcPtr, 1, "")
	return funcValue
}

// extractFuncScalar returns some scalar that can be used in comparisons. It is
// a cheap operation.
func (b *builder) extractFuncScalar(funcValue llvm.Value) llvm.Value {
	return b.CreateExtractValue(funcValue, 1, "")
}

// extractFuncContext extracts the context pointer from this function value. It
// is a cheap operation.
func (b *builder) extractFuncContext(funcValue llvm.Value) llvm.Value {
	return b.CreateExtractValue(funcValue, 0, "")
}

// decodeFuncValue extracts the context and the function pointer from this func
// value.
func (b *builder) decodeFuncValue(funcValue llvm.Value) (funcPtr, context llvm.Value) {
	context = b.CreateExtractValue(funcValue, 0, "")
	funcPtr = b.CreateExtractValue(funcValue, 1, "")
	return
}

// getFuncType returns the type of a func value given a signature.
func (c *compilerContext) getFuncType(typ *types.Signature) llvm.Type {
	return c.ctx.StructType([]llvm.Type{c.dataPtrType, c.funcPtrType}, false)
}

// getLLVMFunctionType returns a LLVM function type for a given signature.
func (c *compilerContext) getLLVMFunctionType(typ *types.Signature) llvm.Type {
	// For SPMD signatures with mixed-width varying types (e.g., Varying[float32]
	// and Varying[int] on AVX2 where float32→8 lanes and int→4 lanes), all
	// Varying[T] parameters and results are capped to the minimum lane count so
	// the function uses a single, consistent lane width throughout.
	minLC := c.spmdMinLaneCountForSig(typ)

	// getLLVMTypeForSig returns the LLVM type for a Go type, capping Varying[T]
	// at minLC when minLC > 0.
	getLLVMTypeForSig := func(goType types.Type) llvm.Type {
		if minLC <= 0 {
			return c.getLLVMType(goType)
		}
		if spmdType, ok := goType.(*types.SPMDType); ok && spmdType.IsVarying() {
			elemType := c.getLLVMType(spmdType.Elem())
			naturalLC := c.spmdLaneCount(elemType)
			if naturalLC <= 1 {
				return elemType // scalar fallback
			}
			if minLC < naturalLC {
				// Cap to minimum: e.g., Varying[float32] on AVX2 → <4 x float> not <8 x float>
				return llvm.VectorType(elemType, minLC)
			}
		}
		return c.getLLVMType(goType)
	}

	// Get the return type.
	var returnType llvm.Type
	switch typ.Results().Len() {
	case 0:
		// No return values.
		returnType = c.ctx.VoidType()
	case 1:
		// Just one return value.
		returnType = getLLVMTypeForSig(typ.Results().At(0).Type())
	default:
		// Multiple return values. Put them together in a struct.
		// This appears to be the common way to handle multiple return values in
		// LLVM.
		members := make([]llvm.Type, typ.Results().Len())
		for i := 0; i < typ.Results().Len(); i++ {
			members[i] = getLLVMTypeForSig(typ.Results().At(i).Type())
		}
		returnType = c.ctx.StructType(members, false)
	}

	// Get the parameter types.
	var paramTypes []llvm.Type

	// SPMD: insert execution mask type as first parameter for SPMD signatures.
	// This is used for function pointer types. Exported SPMD functions are
	// forbidden by the type checker, so all SPMD function values have a mask.
	if maskType := c.spmdMaskTypeFromSig(typ); maskType != (llvm.Type{}) {
		paramTypes = append(paramTypes, maskType)
	}

	if typ.Recv() != nil {
		recv := c.getLLVMType(typ.Recv().Type())
		if recv.StructName() == "runtime._interface" {
			// This is a call on an interface, not a concrete type.
			// The receiver is not an interface, but a i8* type.
			recv = c.dataPtrType
		}
		for _, info := range c.expandFormalParamType(recv, "", nil) {
			paramTypes = append(paramTypes, info.llvmType)
		}
	}
	for i := 0; i < typ.Params().Len(); i++ {
		subType := getLLVMTypeForSig(typ.Params().At(i).Type())
		for _, info := range c.expandFormalParamType(subType, "", nil) {
			paramTypes = append(paramTypes, info.llvmType)
		}
	}
	// All functions take these parameters at the end.
	paramTypes = append(paramTypes, c.dataPtrType) // context

	// Make a func type out of the signature.
	return llvm.FunctionType(returnType, paramTypes, false)
}

// parseMakeClosure makes a function value (with context) from the given
// closure expression.
func (b *builder) parseMakeClosure(expr *ssa.MakeClosure) (llvm.Value, error) {
	if len(expr.Bindings) == 0 {
		panic("unexpected: MakeClosure without bound variables")
	}
	f := expr.Fn.(*ssa.Function)

	// Collect all bound variables.
	boundVars := make([]llvm.Value, len(expr.Bindings))
	for i, binding := range expr.Bindings {
		// The context stores the bound variables.
		llvmBoundVar := b.getValue(binding, getPos(expr))
		boundVars[i] = llvmBoundVar
	}

	// Store the bound variables in a single object, allocating it on the heap
	// if necessary.
	context := b.emitPointerPack(boundVars)

	// Create the closure.
	_, fn := b.getFunction(f)
	return b.createFuncValue(fn, context, f.Signature), nil
}
