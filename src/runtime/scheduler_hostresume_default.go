//go:build !spmd.gpu.host.browser

package runtime

// hostResumesScheduler is false outside of -gpu-host=browser builds. See
// scheduler_hostresume_browser.go for the full rationale. This keeps every
// existing target (including a plain -target=wasi build with no GPU host
// flags) byte-for-byte unaffected: the deadlock-on-empty-runqueue path in
// scheduler() still calls waitForEvents() exactly as before, and
// go_scheduler is not exported (see scheduler_wasi_hostresume.go's build
// tag).
const hostResumesScheduler = false
