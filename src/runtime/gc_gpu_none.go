//go:build !gc.boehm || !spmd.gpu.host.vulkan

package runtime

import (
	"internal/gclayout"
	"unsafe"
)

// allocGPU without a GPU pool is exactly alloc: the compiler only marks
// allocation sites for -gpu-host=vulkan builds, but the symbol must exist
// everywhere so the runtime package compiles for every target.
func allocGPU(size uintptr, layout unsafe.Pointer) unsafe.Pointer { return alloc(size, layout) }

func gpuPoolEnabled() bool { return false }

func gpuPoolLookup(p uintptr) (int, bool) { return 0, false }

func gpuPoolInit() {}

// spmdAllocGPUBytes must pass the SAME layout as the Boehm path, so a program
// using the test hook allocates pointer-free memory on every target rather
// than conservatively-scanned memory on some of them.
func spmdAllocGPUBytes(size uintptr) unsafe.Pointer {
	return allocGPU(size, gclayout.NoPtrs.AsPtr())
}
