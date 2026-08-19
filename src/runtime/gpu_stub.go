//go:build !(tinygo.wasm && js)

// SPMD GPU offload runtime ABI stub for all targets other than wasm/js
// (native, wasip1, other tinygo.wasm variants, etc). WebGPU is only
// reachable from a browser host, so these targets never have a GPU backend
// available. This file exists so that compiler-generated IR (Task 6) can
// unconditionally reference spmdGPU* by name on every target; the
// spmdGPUAvailable() == false check compiled into that IR ensures the
// panicking bodies below are never actually reached at runtime.
package runtime

import "unsafe"

// gpuBufferDesc mirrors the layout in gpu_wasm.go so callers can build the
// same type regardless of target.
type gpuBufferDesc struct {
	dataPtr uint32
	byteLen uint32
	mode    uint32
}

// spmdGPUAvailable always reports false: no target other than wasm/js has a
// WebGPU backend.
func spmdGPUAvailable() bool {
	return false
}

func spmdGPURegister(kernelID int32, wgsl string, entry string) {
	runtimePanic("spmd_gpu: GPU offload is not available on this target")
}

func spmdGPULaunch(kernelID int32, n uint32, params unsafe.Pointer, paramsLen uint32, bufsPtr unsafe.Pointer, bufsLen uint32) {
	runtimePanic("spmd_gpu: GPU offload is not available on this target")
}

// spmdGPUDone has no //go:wasmexport directive here: that pragma is only
// supported on wasm targets ("//go:wasmexport is only supported on wasm" is
// a compiler error otherwise, see compiler/symbol.go). It is kept as a
// plain Go function so any target-independent caller can still reference it
// by name, though it is never invoked from a host on non-wasm targets.
func spmdGPUDone(seq uint32) {
	runtimePanic("spmd_gpu: GPU offload is not available on this target")
}
