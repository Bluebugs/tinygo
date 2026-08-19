//go:build tinygo.wasm && wasip1 && spmd.gpu.host.browser

package runtime

// go_scheduler is exported for browser-hosted wasi builds only (-target=wasi
// or -target=wasip1, combined with -gpu-host=browser, which sets the
// spmd.gpu.host.browser build tag -- see hostResumesScheduler in
// scheduler_hostresume_browser.go).
//
// This lets the host resume a goroutine parked in scheduler() after it
// returned early because hostResumesScheduler made the deadlock-on-empty-
// runqueue path return instead of panicking (see scheduler_cooperative.go).
// The canonical caller is worker/runner.js's spmd_gpu glue, after handling
// spmd_gpu_done: it calls spmd_gpu_done(seq, status) to mark the goroutine
// runnable, then go_scheduler() to actually run it (mirroring the existing
// runtime.sleepTicks / go_scheduler pattern used on GOOS=js, adapted to
// wasi: see runtime_wasm_js_scheduler.go, whose own //go:build excludes
// !wasip1 -- that file and this one are deliberately disjoint, not
// overlapping, on the wasip1 tag).
//
// Deliberately NOT exporting `resume` here: that entry point re-fires the JS
// `syscall/js` event queue (handleEvent), which has no wasi equivalent and
// is not part of the spmd_gpu ABI.
//
//export go_scheduler
func go_scheduler() {
	scheduler(false)
}
