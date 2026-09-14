//go:build tinygo.wasm && (js || spmd.gpu.host.browser) && spmd.gpu.webgpu

// SPMD GPU offload runtime ABI (browser/JS target only).
//
// This file declares the imports the JS glue (Task 7) must implement under
// the "spmd_gpu" wasm import module, plus the Go-level wrapper functions
// that compiler-generated IR (Task 6) calls to register and launch GPU
// kernels for "go for" loops offloaded to WebGPU.
//
// The spmd.gpu.webgpu build tag (from Config.BuildTags, set only when
// -gpu=webgpu is passed) is load-bearing: without it this file compiled into
// EVERY tinygo.wasm && js build, so //go:wasmexport spmd_gpu_done added an
// export to every plain `-target=wasm` binary -- measurably contradicting the
// "nothing changes without -gpu" claim (I6). With the tag, a default build is
// byte-identical to one from before this feature existed.
//
// This is additionally restricted to tinygo.wasm && js (i.e. -target=wasm)
// rather than all tinygo.wasm targets: wasip1 also carries the tinygo.wasm
// build tag, but wasmtime/wasi hosts have no WebGPU and refuse to
// instantiate a module with unresolved imports from unknown modules. Only
// the browser/JS target should carry the spmd_gpu imports; wasi (and every
// other target) uses the no-op stub in gpu_stub.go instead.
package runtime

import "unsafe"

//go:wasmimport spmd_gpu available
func gpuAvailable() uint32

//go:wasmimport spmd_gpu register
func gpuRegister(kernelID int32, wgslPtr unsafe.Pointer, wgslLen uint32, entryPtr unsafe.Pointer, entryLen uint32)

//go:wasmimport spmd_gpu launch
func gpuLaunch(kernelID int32, n uint32, paramsPtr unsafe.Pointer, paramsLen uint32, buffersPtr unsafe.Pointer, buffersLen uint32, seq uint32)

// gpuBufferDesc describes one GPU buffer argument for a kernel launch.
//
// Field order and size are load-bearing: the JS glue reads an array of
// these as a flat Uint32Array with stride 3 (dataPtr, byteLen, mode).
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
	// byteLen is the Go byte length; the device buffer is rounded up to a
	// multiple of 4.
	byteLen uint32
	mode    uint32
}

// spmdGPUAvailable reports whether a WebGPU device was successfully
// obtained by the JS host. Compiler-generated code calls this to decide
// whether to offload a "go for" loop to the GPU or fall back to the
// SIMD/scalar CPU path.
func spmdGPUAvailable() bool {
	return gpuAvailable() != 0
}

// spmdGPURegister registers (compiles) a WGSL compute shader under kernelID
// with the JS host, once per kernel.
func spmdGPURegister(kernelID int32, wgsl string, entry string) {
	wgslPtr, wgslLen := stringToPtr(wgsl)
	entryPtr, entryLen := stringToPtr(entry)
	gpuRegister(kernelID, wgslPtr, wgslLen, entryPtr, entryLen)
}

func stringToPtr(s string) (unsafe.Pointer, uint32) {
	if len(s) == 0 {
		return nil, 0
	}
	return unsafe.Pointer(unsafe.StringData(s)), uint32(len(s))
}

// spmdGPUSeq and spmdGPUPending implement a monotonically increasing
// seq -> completion-channel map used to correlate asynchronous GPU launches
// (dispatched from JS, potentially completing on a JS microtask/callback)
// with the goroutine blocked waiting for the result.
//
// No mutex guards this map. The wasm/js target always runs with TinyGo's
// single-threaded cooperative "asyncify" scheduler (see targets/wasm.json),
// so there is never more than one Go-level goroutine actually executing at
// a time: spmdGPULaunch runs to the point where it blocks on <-ch, and
// spmdGPUDone (a //go:wasmexport entry point) runs to completion before any
// other Go code resumes. There is no preemption and no real parallelism
// that could race two map accesses against each other. A mutex would only
// add overhead here, not safety.
var (
	spmdGPUSeq     uint32
	spmdGPUPending = make(map[uint32]chan struct{})
	// spmdGPUStatus records the completion status the JS host reported for
	// a launch (see spmdGPUDone's ok argument). Read and deleted by the
	// launching goroutine once it wakes up.
	spmdGPUStatus = make(map[uint32]uint32)
	// spmdGPUInFlightCount tracks the number of GPU launches currently
	// blocked waiting on a host response. scheduler_cooperative.go reads
	// this (via spmdGPULaunchesInFlight, below) to distinguish "the run
	// queue is empty because we're genuinely waiting on an outstanding
	// async host launch" (safe to return early and let the host resume us
	// later) from "the run queue is empty because a goroutine deadlocked
	// on an ordinary channel with nothing outstanding" (must still panic,
	// exactly as a plain, non-GPU wasi build would) -- see
	// hostResumesScheduler in scheduler_hostresume_browser.go. No mutex:
	// same single-goroutine-at-a-time reasoning as the maps above.
	spmdGPUInFlightCount uint32
)

// spmdGPULaunchesInFlight reports whether any goroutine is currently
// blocked waiting on a GPU launch to complete. See spmdGPUInFlightCount.
func spmdGPULaunchesInFlight() bool {
	return spmdGPUInFlightCount > 0
}

// spmdGPULaunch dispatches a GPU kernel launch and blocks the calling
// goroutine until the JS host reports completion via spmdGPUDone.
func spmdGPULaunch(kernelID int32, n uint32, params unsafe.Pointer, paramsLen uint32, bufsPtr unsafe.Pointer, bufsLen uint32) {
	spmdGPUSeq++
	seq := spmdGPUSeq

	ch := make(chan struct{})
	spmdGPUPending[seq] = ch

	gpuLaunch(kernelID, n, params, paramsLen, bufsPtr, bufsLen, seq)

	// The increment/decrement pair below is currently correct as a simple
	// bracket (nothing between them can return early, and every return in
	// spmdGPULaunch happens after the decrement). But the failure mode of
	// getting this wrong is a MASKED panic that depends on execution
	// history -- a future edit that adds an early return inside this window
	// would silently turn a later genuine deadlock into what looks like a
	// clean exit (see hostResumesScheduler / spmdGPULaunchesInFlight in
	// scheduler_cooperative.go). defer insures against that regardless of
	// what gets added between here and the receive.
	spmdGPUInFlightCount++
	defer func() { spmdGPUInFlightCount-- }()
	<-ch

	// I2: the launch is only "done" if the host says it SUCCEEDED. A shader
	// that failed to compile (createShaderModule does not throw -- the
	// failure surfaces later from createComputePipeline), a dispatch refused
	// by the workgroup-count ceiling, or any exception caught by the JS glue
	// all used to unblock this goroutine with the output buffers NEVER
	// WRITTEN, and execution simply continued over stale/garbage data. The
	// compile-time path fails closed; the runtime must too. There is no
	// CPU-path re-entry available from here (the compiler already branched
	// away from it), so a failed launch is a hard, clearly-labelled panic
	// rather than a silent wrong answer.
	status, haveStatus := spmdGPUStatus[seq]
	delete(spmdGPUStatus, seq)
	if !haveStatus || status == 0 {
		runtimePanic("spmd_gpu: GPU kernel launch failed (see host console for the underlying WebGPU error); output buffers were not written")
	}
}

// spmdGPUDone is called by the JS host (via a wasm export call) when an
// asynchronous GPU launch identified by seq has completed. status is 1 when
// the dispatch AND the readback of every read_write buffer completed
// successfully, and 0 when anything went wrong (no device, unknown kernel,
// shader compile/pipeline failure, refused dispatch, mapAsync rejection,
// ...). It unblocks the goroutine parked in spmdGPULaunch, which panics on
// a 0 status rather than proceeding over unwritten output buffers.
//
//go:wasmexport spmd_gpu_done
func spmdGPUDone(seq uint32, status uint32) {
	ch, ok := spmdGPUPending[seq]
	if !ok {
		return
	}
	spmdGPUStatus[seq] = status
	delete(spmdGPUPending, seq)
	close(ch)
}
