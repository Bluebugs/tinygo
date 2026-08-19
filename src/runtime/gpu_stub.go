//go:build !(tinygo.wasm && (js || spmd.gpu.host.browser) && spmd.gpu.webgpu) && !(!tinygo.wasm && linux && amd64 && spmd.gpu.webgpu)

// SPMD GPU offload runtime ABI stub for every build that is not a
// wasm/js (-target=wasm) build with -gpu=webgpu and is not a native
// linux/amd64 build with -gpu=webgpu (gpu_native.go, Task 10) -- i.e. all other targets
// (native, wasip1, other tinygo.wasm variants) AND ordinary wasm/js builds
// without -gpu=webgpu, which must stay byte-identical to a pre-feature
// build (I6). WebGPU is only
// reachable from a browser host, so these targets never have a GPU backend
// available. This file exists so that compiler-generated IR (Task 6) can
// unconditionally reference spmdGPU* by name on every target; the
// spmdGPUAvailable() == false check compiled into that IR ensures the
// panicking bodies below are never actually reached at runtime.
package runtime

import "unsafe"

// gpuBufferDesc mirrors the layout in gpu_wasm.go so callers can build the
// same type regardless of target.
// mode selects the host-side data movement for this buffer:
//
//	0  read-only:  upload before dispatch, no readback.
//	1  read_write: upload before dispatch, read back after.
//	2  write-only: do NOT upload, read back after. Emitted only when the
//	   compiler proved the kernel writes every element and never reads the
//	   buffer (gpu_eligible.go writeOnlySliceObjs) AND a runtime
//	   n == len(slice) check passed (gpu_offload.go gpuBuildBuffers), so
//	   the same kernel can launch with 2 or 1 on different calls.
//
// Any mode other than 0 is read back.
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
func spmdGPUDone(seq uint32, status uint32) {
	runtimePanic("spmd_gpu: GPU offload is not available on this target")
}
