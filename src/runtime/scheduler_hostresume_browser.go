//go:build spmd.gpu.host.browser

package runtime

// hostResumesScheduler is true only for browser-hosted builds (built with
// -gpu-host=browser, which sets the spmd.gpu.host.browser build tag
// regardless of GOOS/target -- notably including -target=wasi, where
// asyncScheduler (GOOS=="js") is false).
//
// Such builds can genuinely park a goroutine on a channel receive that is
// unblocked by an async host operation (e.g. spmd_gpu.launch's WebGPU
// dispatch, completed later via the exported spmd_gpu_done), and rely on the
// host explicitly calling the exported go_scheduler (see
// scheduler_wasi_hostresume.go) to resume it. Without this, the scheduler's
// deadlock-on-empty-runqueue path (waitForEvents, see wait_other.go) fires
// the instant such a goroutine parks, since a plain wasi build has no way to
// be resumed by the host and correctly treats "nothing runnable" as a real
// deadlock.
//
// This is INTENTIONALLY separate from asyncScheduler (scheduler_asyncjs.go
// et al.): asyncScheduler also governs the sleepTicks early-return in
// scheduler() (see scheduler_cooperative.go), which must stay JS-only.
// Under wasi (browser-hosted or not), sleepTicks genuinely blocks on the
// WASI clock; making it return early there would break time.Sleep for every
// wasi program, browser-hosted or not.
const hostResumesScheduler = true
