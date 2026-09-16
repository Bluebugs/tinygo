/* lib/bdwgc-gpu/gc_gpu.h (copied into the patched tree as include/gc/gc_gpu.h) */
#ifndef GC_GPU_H
#define GC_GPU_H
#include <stddef.h>
#include <stdint.h>
#include "gc.h"

/* A chunk of foreign (Vulkan) memory donated to the GPU pool. */
typedef struct {
    uintptr_t base; /* HBLKSIZE-aligned mapped address */
    size_t    size; /* multiple of HBLKSIZE */
    uint64_t  tag;  /* opaque provider tag; the Vulkan host stores its VkBuffer here */
} GC_gpu_chunk;

/* Returns 1 and fills *out on success, 0 on failure (pool is then disabled). */
typedef int (*GC_gpu_chunk_provider_proc)(size_t bytes, GC_gpu_chunk *out);

/* Chunk size the pool asks for by default: 64 MiB. */
#define GC_GPU_CHUNK_BYTES ((size_t)64 << 20)

/* LOCKING CONTRACTS -- bdwgc's LOCK() is NOT recursive, so these matter:
   - GC_set_gpu_chunk_provider and GC_gpu_pool_enabled may CREATE the GPU
     allocation kind on demand, which calls GC_init and GC_new_free_list; both
     take the allocation lock themselves and allocate. Neither function may be
     called with the allocation lock held -- doing so (for example from
     GC_allochblk_nth) deadlocks. Call them from ordinary, unlocked context.
   - GC_gpu_expand (internal, declared in gpu_pool.c) is the mirror image: it
     reaches GC_scratch_alloc and therefore MUST be called WITH the allocation
     lock held, which is what its GC_allochblk_nth caller already holds. */
GC_API void GC_CALL GC_set_gpu_chunk_provider(GC_gpu_chunk_provider_proc fn);
GC_API int  GC_CALL GC_gpu_pool_enabled(void);
GC_API void GC_CALL GC_gpu_disable(void);
/* Allocates lb bytes (rounded up to a multiple of 4) from the GPU pool.
   Returns NULL when the pool is disabled or a chunk cannot be obtained;
   the caller then falls back to GC_malloc_atomic. Memory is NOT zeroed. */
GC_API void * GC_CALL GC_malloc_gpu(size_t lb);
/* Lock-free lookup: the chunk containing p, or NULL. */
GC_API const GC_gpu_chunk * GC_CALL GC_gpu_lookup(uintptr_t p);
GC_API size_t GC_CALL GC_gpu_chunk_count(void);
/* Test hook (implemented in the patched allchblk.c): walks both block free
   lists and returns the number of pool/chunk invariant violations -- a block
   carries GPU_POOL_BLK iff it lies inside a chunk, sits on the matching
   pool's list, and does not extend past its chunk. Returns 0 when healthy.
   Walks the lists without the allocation lock: single-threaded test use. */
GC_API size_t GC_CALL GC_gpu_verify_pools(void);
/* Accessors, so callers that cannot spell GC_gpu_chunk (the TinyGo runtime
   reaches these through //export declarations) can still read a chunk. */
GC_API uintptr_t GC_CALL GC_gpu_chunk_base(const GC_gpu_chunk *c);
GC_API size_t    GC_CALL GC_gpu_chunk_size(const GC_gpu_chunk *c);
GC_API uint64_t  GC_CALL GC_gpu_chunk_tag(const GC_gpu_chunk *c);
#endif
