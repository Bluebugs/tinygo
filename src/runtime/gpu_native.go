//go:build !tinygo.wasm && linux && amd64 && spmd.gpu.webgpu

// SPMD GPU offload runtime ABI for native targets (Task 10): wgpu-native +
// Vulkan instead of the browser's WebGPU.
//
// This file provides exactly the same Go-visible API the compiler emits
// calls to as gpu_wasm.go does -- spmdGPUAvailable / spmdGPURegister /
// spmdGPULaunch / spmdGPUDone -- so compiler/gpu_offload.go is completely
// unaware of which host backend is underneath. The WGSL transpiler and the
// kernel descriptors are unchanged; the same generated shader runs on both.
//
// The one structural difference is that spmdGPULaunch here is
// SYNCHRONOUS. A native host can block on the GPU (wgpuDevicePoll with
// wait=true), so the seq counter, the pending-completion map, the
// completion channel and the //go:wasmexport spmd_gpu_done callback that
// gpu_wasm.go needs are all absent rather than emulated: the C shim has
// already finished the readback by the time spmd_gpu_launch returns.
//
// Build tag notes. The three GPU runtime files must partition every
// possible build exactly, because createRuntimeCall panics if any of these
// names is missing on some target:
//
//	gpu_wasm.go    tinygo.wasm && js && spmd.gpu.webgpu
//	gpu_native.go  !tinygo.wasm && linux && amd64 && spmd.gpu.webgpu
//	gpu_stub.go    everything else (the negation of both of the above)
package runtime

import "unsafe"

// The C side lives in gpu_native.c, which compileopts/target.go appends to
// ExtraFiles when -gpu=webgpu is passed on a native target. These are
// body-less Go declarations bound to C symbols by //export, the same
// mechanism the rest of the runtime uses for libc entry points (see
// os_linux.go). The runtime package cannot use cgo, so the WebGPU
// descriptor structs stay entirely on the C side and the Go/C contract is
// just these three flat calls plus gpuBufferDesc.

//export spmd_gpu_available
func spmdGPUAvailableC() int32

//export spmd_gpu_register
func spmdGPURegisterC(kernelID int32, wgsl *byte, wgslLen uint32, entry *byte, entryLen uint32) int32

//export spmd_gpu_launch
func spmdGPULaunchC(kernelID int32, n uint32, params unsafe.Pointer, paramsLen uint32, bufs unsafe.Pointer, bufCount uint32) int32

// gpuBufferDesc describes one GPU buffer argument for a kernel launch.
//
// Field order and size are load-bearing and must agree EXACTLY with three
// other places, or the C shim reads garbage pointers:
//
//   - compiler/gpu_offload.go gpuBuildBuffers, which emits the LLVM struct
//     {i64, i32, i32} on an 8-byte-pointer target and asserts its size,
//   - spmd_gpu_buffer_desc in gpu_native.c, which has a _Static_assert on
//     the same 16 bytes,
//   - this struct: uintptr + uint32 + uint32 = 16 bytes on amd64, no
//     padding (the two uint32s exactly fill the 8-byte alignment slot).
//
// The wasm32 layout in gpu_wasm.go is deliberately DIFFERENT ({u32, u32,
// u32}, 12 bytes) because the JS glue reads the descriptor array as a flat
// stride-3 Uint32Array. That layout must not be disturbed.
//
// mode selects the host-side data movement for this buffer:
//
//	0  read-only:  upload before dispatch, no readback.
//	1  read_write: upload before dispatch, read back after.
//	2  write-only: do NOT upload, read back after.
type gpuBufferDesc struct {
	dataPtr uintptr
	byteLen uint32
	mode    uint32
}

// spmdGPUAvailable reports whether a wgpu-native Vulkan device was
// obtained. The first call performs instance/adapter/device creation
// (~90 ms on the measured machine); the result is cached in C, so the
// per-loop guard the compiler emits costs a load after that.
func spmdGPUAvailable() bool {
	return spmdGPUAvailableC() != 0
}

// spmdGPURegister compiles a WGSL compute shader into a pipeline under
// kernelID, once per kernel. A failure here is fatal for the same reason it
// is on the wasm path: the compiler has already branched away from the CPU
// path, so there is nothing to fall back to.
func spmdGPURegister(kernelID int32, wgsl string, entry string) {
	if spmdGPURegisterC(kernelID, stringData(wgsl), uint32(len(wgsl)),
		stringData(entry), uint32(len(entry))) == 0 {
		runtimePanic("spmd_gpu: GPU kernel registration failed (see stderr for the underlying WebGPU error)")
	}
}

func stringData(s string) *byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.StringData(s)
}

// spmdGPULaunch dispatches a GPU kernel and returns only once the dispatch
// AND the readback of every output buffer have completed. Unlike the wasm
// path there is no goroutine parking involved -- the C shim blocks the
// calling thread in wgpuDevicePoll.
//
// A failed launch is a hard panic rather than a silent wrong answer,
// matching gpu_wasm.go: the output buffers were not written, and the
// compiler already branched away from the CPU path, so there is no correct
// way to continue.
func spmdGPULaunch(kernelID int32, n uint32, params unsafe.Pointer, paramsLen uint32, bufsPtr unsafe.Pointer, bufsLen uint32) {
	if spmdGPULaunchC(kernelID, n, params, paramsLen, bufsPtr, bufsLen) == 0 {
		runtimePanic("spmd_gpu: GPU kernel launch failed (see stderr for the underlying WebGPU error); output buffers were not written")
	}
}

// spmdGPUDone exists only so that every target defines the same set of
// spmdGPU* names. The native backend is synchronous and never has an
// outstanding launch to complete, so nothing can legitimately call this.
func spmdGPUDone(seq uint32, status uint32) {
	runtimePanic("spmd_gpu: spmdGPUDone is not used by the native GPU backend")
}

// spmdGPULaunchesInFlight always reports false. Unlike gpu_wasm.go's
// version, this is not merely the "no GPU backend" stub value (that's
// gpu_stub.go's job) -- it is correct FOR THIS BACKEND SPECIFICALLY:
// spmdGPULaunch above is synchronous (it blocks the calling thread in
// wgpuDevicePoll via the C shim) and never parks a goroutine on a channel,
// so there is never a launch "in flight" from scheduler_cooperative.go's
// point of view. Needed because this file, gpu_wasm.go, and gpu_stub.go
// must each define every spmdGPU* name the compiler/runtime can reference
// (see the build-tag partition note above) -- omitting this one is a
// working default build (native defaults to scheduler.threads, which does
// not compile scheduler_cooperative.go at all) but a compile error the
// moment someone passes -scheduler=tasks. Do not "fix" this into a real
// counter -- there is nothing to count here.
func spmdGPULaunchesInFlight() bool {
	return false
}
