//go:build tinygo.wasm && js

// SPMD GPU offload runtime ABI (browser/JS target only).
//
// This file declares the imports the JS glue (Task 7) must implement under
// the "spmd_gpu" wasm import module, plus the Go-level wrapper functions
// that compiler-generated IR (Task 6) calls to register and launch GPU
// kernels for "go for" loops offloaded to WebGPU.
//
// This is intentionally restricted to tinygo.wasm && js (i.e. -target=wasm)
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
type gpuBufferDesc struct {
	dataPtr uint32
	byteLen uint32
	mode    uint32 // 0 = read, 1 = read_write
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
)

// spmdGPULaunch dispatches a GPU kernel launch and blocks the calling
// goroutine until the JS host reports completion via spmdGPUDone.
func spmdGPULaunch(kernelID int32, n uint32, params unsafe.Pointer, paramsLen uint32, bufsPtr unsafe.Pointer, bufsLen uint32) {
	spmdGPUSeq++
	seq := spmdGPUSeq

	ch := make(chan struct{})
	spmdGPUPending[seq] = ch

	gpuLaunch(kernelID, n, params, paramsLen, bufsPtr, bufsLen, seq)

	<-ch
}

// spmdGPUDone is called by the JS host (via a wasm export call) when an
// asynchronous GPU launch identified by seq has completed. It unblocks the
// goroutine parked in spmdGPULaunch.
//
//go:wasmexport spmd_gpu_done
func spmdGPUDone(seq uint32) {
	ch, ok := spmdGPUPending[seq]
	if !ok {
		return
	}
	delete(spmdGPUPending, seq)
	close(ch)
}
