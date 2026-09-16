//go:build gc.boehm && spmd.gpu.host.vulkan

package runtime

import (
	"internal/gclayout"
	"unsafe"
)

//export GC_malloc_gpu
func libgc_malloc_gpu(uintptr) unsafe.Pointer

//export GC_gpu_pool_enabled
func libgc_gpu_pool_enabled() int32

//export GC_gpu_lookup
func libgc_gpu_lookup(uintptr) unsafe.Pointer

//export GC_gpu_chunk_base
func libgc_gpu_chunk_base(unsafe.Pointer) uintptr

//export spmd_vk_pool_register
func spmdVKPoolRegister() int32

// gpuPoolInit registers the Vulkan chunk provider with bdwgc. It runs once,
// from initHeap, before any goroutine other than the main one exists: the
// provider must never be entered while the world is stopped, so the Vulkan
// device is created here rather than lazily.
func gpuPoolInit() {
	spmdVKPoolRegister()
}

// gpuPoolEnabled reports whether GPU-pool allocation is active.
//
// GC_gpu_pool_enabled may create the GPU allocation kind on demand, which
// takes bdwgc's (non-recursive) allocation lock and allocates, so it must
// never be called with that lock held -- see the locking contract in
// lib/bdwgc-gpu/gc_gpu.h.
func gpuPoolEnabled() bool { return libgc_gpu_pool_enabled() != 0 }

// gpuPoolLookup returns a per-chunk identity (the chunk's base address) so
// tests can tell "same chunk" from "different chunk" and "not pooled".
func gpuPoolLookup(p uintptr) (int, bool) {
	c := libgc_gpu_lookup(p)
	if c == nil {
		return 0, false
	}
	return int(libgc_gpu_chunk_base(c)), true
}

// allocGPU allocates size bytes for a pointer-free object from the GPU pool.
// layout must be gclayout.NoPtrs; anything else, and any pool failure, falls
// back to alloc(size, layout). The result is always zeroed.
//
//go:noinline
func allocGPU(size uintptr, layout unsafe.Pointer) unsafe.Pointer {
	if size == 0 {
		return unsafe.Pointer(&zeroSizedAlloc)
	}
	if layout != gclayout.NoPtrs.AsPtr() {
		// Defensive: the marking pass never marks pointer-containing types.
		return alloc(size, layout)
	}
	gcLock.Lock()
	ptr := libgc_malloc_gpu(size)
	gcResumeWorld()
	gcLock.Unlock()
	if ptr == nil {
		return alloc(size, layout) // pool disabled or chunk growth failed
	}
	// GC_malloc_gpu does not zero, so do it here (like libgc_malloc_atomic).
	memzero(ptr, size)
	return ptr
}

// spmdAllocGPUBytes is the //go:linkname-able test hook: allocGPU with NoPtrs.
func spmdAllocGPUBytes(size uintptr) unsafe.Pointer {
	return allocGPU(size, gclayout.NoPtrs.AsPtr())
}
