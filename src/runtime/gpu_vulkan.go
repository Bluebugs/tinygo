//go:build !tinygo.wasm && linux && amd64 && spmd.gpu.webgpu && spmd.gpu.host.vulkan

// SPMD GPU offload runtime ABI for native targets using a direct Vulkan
// compute host (-gpu=webgpu -gpu-host=vulkan) instead of wgpu-native.
//
// Same Go-visible API as gpu_native.go, except spmdGPURegister takes the
// SPIR-V module (compiled at build time by naga from the transpiler's WGSL)
// plus the kernel's storage buffer count, which the C side needs to build an
// explicit descriptor set layout (Vulkan has no `layout: 'auto'`).
//
// Launches are synchronous, exactly like gpu_native.go: the C shim waits on
// a fence and copies the outputs back before returning.
//
// Failure after a fence timeout or VK_ERROR_DEVICE_LOST: the launch that hit
// it panics, and the Vulkan host is then disabled for the rest of the
// process (spmdGPUAvailable returns false), so later loops run on the CPU.
//
// Build tag notes. The four GPU runtime files partition every build exactly
// (see compileopts/gpu_tags_test.go, which proves it):
//
//	gpu_wasm.go    tinygo.wasm && (js || spmd.gpu.host.browser) && spmd.gpu.webgpu
//	gpu_native.go  !tinygo.wasm && linux && amd64 && spmd.gpu.webgpu && !spmd.gpu.host.vulkan
//	gpu_vulkan.go  !tinygo.wasm && linux && amd64 && spmd.gpu.webgpu && spmd.gpu.host.vulkan
//	gpu_stub.go    everything else
package runtime

import "unsafe"

// Implemented in gpu_vulkan.c, which compileopts/target.go appends to
// ExtraFiles for -gpu-host=vulkan builds.

//export spmd_vk_available
func spmdVKAvailableC() int32

//export spmd_vk_register
func spmdVKRegisterC(kernelID int32, spirv *byte, spirvLen uint32, entry *byte, entryLen uint32, bufCount int32) int32

//export spmd_vk_launch
func spmdVKLaunchC(kernelID int32, n uint32, params unsafe.Pointer, paramsLen uint32, bufs unsafe.Pointer, bufCount uint32) int32

// gpuBufferDesc must match gpu_native.go, spmd_vk_buffer_desc in
// gpu_vulkan.c (which has _Static_asserts on it) and gpuBuildBuffers in
// compiler/gpu_offload.go: {u64 dataPtr; u32 byteLen; u32 mode}, 16 bytes.
//
//	mode 0  read-only:  upload before dispatch, no readback.
//	mode 1  read_write: upload before dispatch, read back after.
//	mode 2  write-only: do NOT upload, read back after.
type gpuBufferDesc struct {
	dataPtr uintptr
	byteLen uint32
	mode    uint32
}

// spmdGPUAvailable reports whether a non-CPU Vulkan device with a compute
// queue was obtained. Initialization happens once and is cached in C.
func spmdGPUAvailable() bool {
	return spmdVKAvailableC() != 0
}

// spmdGPURegister creates the compute pipeline for kernelID from SPIR-V
// bytes, once per kernel. Failure is fatal: the compiler has already
// branched away from the CPU path.
func spmdGPURegister(kernelID int32, shader string, entry string, bufCount int32) {
	if spmdVKRegisterC(kernelID, stringData(shader), uint32(len(shader)),
		stringData(entry), uint32(len(entry)), bufCount) == 0 {
		runtimePanic("spmd_gpu: Vulkan host kernel registration failed (see stderr for the underlying Vulkan error)")
	}
}

func stringData(s string) *byte {
	if len(s) == 0 {
		return nil
	}
	return unsafe.StringData(s)
}

// spmdGPULaunch dispatches a kernel and returns once every output buffer has
// been copied back. A failure panics rather than continuing over unwritten
// output memory.
func spmdGPULaunch(kernelID int32, n uint32, params unsafe.Pointer, paramsLen uint32, bufsPtr unsafe.Pointer, bufsLen uint32) {
	if spmdVKLaunchC(kernelID, n, params, paramsLen, bufsPtr, bufsLen) == 0 {
		runtimePanic("spmd_gpu: Vulkan host kernel launch failed (see stderr for the underlying Vulkan error); output buffers were not written")
	}
}

// spmdGPUDone exists only so every target defines the same spmdGPU* names;
// this backend is synchronous and never has an outstanding launch.
func spmdGPUDone(seq uint32, status uint32) {
	runtimePanic("spmd_gpu: spmdGPUDone is not used by the Vulkan GPU backend")
}

// spmdGPULaunchesInFlight is always false: launches are synchronous and
// never park a goroutine (see gpu_native.go for why the name must exist).
func spmdGPULaunchesInFlight() bool {
	return false
}
