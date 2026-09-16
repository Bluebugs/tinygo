/* lib/bdwgc-gpu/gpu_pool.c (new; compiled only with -DTINYGO_GPU_POOL) */
#include "private/gc_priv.h"
/* gc_priv.h includes this only under THREADS, and the TinyGo/SPMD build is
   single-threaded, but the lock-free chunk-table publish/read below needs the
   GC_cptr_* acquire/release helpers regardless. With GC_BUILTIN_ATOMIC (set by
   both the TinyGo build and the harness) these are compiler intrinsics. */
#include "private/gc_atomic_ops.h"
#include "gc_gpu.h"

#ifdef TINYGO_GPU_POOL

GC_INNER int GC_gpu_kind = -1;
STATIC GC_gpu_chunk_provider_proc GC_gpu_provider = 0;
STATIC int GC_gpu_disabled = 0;

/* Copy-on-write sorted table. Readers take the pointer once; writers build a
   new array under the allocation lock and publish it with a release store, so
   a launch-time lookup needs no lock. Old arrays are intentionally leaked
   (a handful of pointer-sized arrays for the life of the process). */
typedef struct { size_t n; GC_gpu_chunk c[1]; } GC_gpu_table;
STATIC GC_gpu_table *GC_gpu_tab = 0;

GC_API void GC_CALL GC_gpu_disable(void) { GC_gpu_disabled = 1; }

/* Has the GPU pool been permanently disabled?  A pure read of a plain int:
   it neither allocates nor takes the allocation lock, so unlike
   GC_gpu_pool_enabled() it is safe to call FROM the allocator, with the
   allocation lock held.  That is exactly where it is needed: once the pool is
   disabled the GPU block free list can never be grown again, so GC_allochblk
   returns NULL forever.  GC_alloc_large and GC_allocobj answer a persistent
   NULL by retrying a bounded number of times and then ABORT()ing the process.
   The patched copies of both consult this predicate and return NULL instead,
   which turns a dead pool into a fallback to the normal heap. */
GC_INNER GC_bool GC_gpu_pool_broken(void) {
    return GC_gpu_disabled ? TRUE : FALSE;
}

/* Creates the GPU allocation kind. MUST be called without the allocation lock
   held: GC_new_free_list takes the lock itself, and it allocates, so the
   collector must already be initialized. */
STATIC void GC_gpu_ensure_kind(void) {
    if (GC_gpu_disabled || GC_gpu_provider == 0 || GC_gpu_kind >= 0) return;
    /* GC_new_free_list allocates, so GC_init must have run. The TinyGo runtime
       registers its provider after gcInit(), but a standalone caller (the
       stress harness) may register before GC_INIT(); GC_init is idempotent. */
    if (!GC_is_init_called()) GC_init();
    /* descr 0 = GC_DS_LENGTH, no pointers. adjust MUST be FALSE here: bdwgc
       asserts `1 == clear || (0 == descr && !adjust && !clear)` in
       GC_new_kind_inner, i.e. a pointer-free non-clearing kind must not
       relocate its descriptor. This mirrors bdwgc's own PTRFREE kind. */
    GC_gpu_kind = (int)GC_new_kind(GC_new_free_list(), 0 /* no pointers */,
                                   FALSE /* no descr adjust */,
                                   FALSE /* do not clear */);
}

GC_API int  GC_CALL GC_gpu_pool_enabled(void) {
    if (GC_gpu_disabled || GC_gpu_provider == 0) return 0;
    if (GC_gpu_kind < 0) GC_gpu_ensure_kind();
    return !GC_gpu_disabled && GC_gpu_provider != 0 && GC_gpu_kind >= 0;
}
GC_API size_t GC_CALL GC_gpu_chunk_count(void) {
    GC_gpu_table *t = (GC_gpu_table *)GC_cptr_load_acquire((volatile ptr_t *)&GC_gpu_tab);
    return t == 0 ? 0 : t->n;
}
GC_API const GC_gpu_chunk * GC_CALL GC_gpu_lookup(uintptr_t p) {
    GC_gpu_table *t = (GC_gpu_table *)GC_cptr_load_acquire((volatile ptr_t *)&GC_gpu_tab);
    if (t == 0) return 0;
    size_t lo = 0, hi = t->n;
    while (lo < hi) {
        size_t mid = (lo + hi) / 2;
        if (p < t->c[mid].base) hi = mid;
        else if (p >= t->c[mid].base + t->c[mid].size) lo = mid + 1;
        else return &t->c[mid];
    }
    return 0;
}
GC_API uintptr_t GC_CALL GC_gpu_chunk_base(const GC_gpu_chunk *c) { return c->base; }
GC_API size_t    GC_CALL GC_gpu_chunk_size(const GC_gpu_chunk *c) { return c->size; }
GC_API uint64_t  GC_CALL GC_gpu_chunk_tag(const GC_gpu_chunk *c)  { return c->tag; }

/* Called with the allocation lock held. Builds a new sorted table containing
   the old entries plus ch, then publishes it with a release store so
   GC_gpu_lookup readers never observe a half-built array. The old table is
   intentionally leaked (one small array per chunk, for the process lifetime):
   a reader may still hold a pointer into it. */
STATIC GC_bool GC_gpu_add_chunk(const GC_gpu_chunk *ch) {
    GC_gpu_table *old = GC_gpu_tab;
    size_t n = (old == 0 ? 0 : old->n);
    GC_gpu_table *fresh = (GC_gpu_table *)GC_scratch_alloc(
        sizeof(GC_gpu_table) + n * sizeof(GC_gpu_chunk));
    if (fresh == 0) return FALSE;
    size_t i = 0, j = 0;
    while (i < n && old->c[i].base < ch->base) { fresh->c[j++] = old->c[i++]; }
    fresh->c[j++] = *ch;
    while (i < n) { fresh->c[j++] = old->c[i++]; }
    fresh->n = n + 1;
    GC_cptr_store_release((volatile ptr_t *)&GC_gpu_tab, (ptr_t)fresh);
    return TRUE;
}

/* The chunk most recently added by GC_gpu_expand. Used only by the
   harness-only bump allocator below, which carves directly out of it.
   Written under the lock. */
STATIC GC_gpu_chunk GC_gpu_last_chunk;

/* Grows the GPU pool by at least need_hblks blocks. Asks the provider for a
   whole 64 MiB chunk, or for the rounded-up request when it is larger, so an
   object never spans two chunks. Called from GC_allochblk_nth when the GPU
   free list cannot satisfy a request.
   Must be called with the allocation lock held (GC_scratch_alloc asserts it). */
GC_INNER GC_bool GC_gpu_expand(size_t need_hblks) {
    if (GC_gpu_disabled || GC_gpu_provider == 0 || GC_gpu_kind < 0) return FALSE;
    size_t need = need_hblks * HBLKSIZE;
    size_t want = need > GC_GPU_CHUNK_BYTES ? need : GC_GPU_CHUNK_BYTES;
    want = (want + (HBLKSIZE - 1)) & ~(size_t)(HBLKSIZE - 1);
    GC_gpu_chunk ch;
    if (!GC_gpu_provider(want, &ch)) { GC_gpu_disabled = 1; return FALSE; }
    if (ch.base % HBLKSIZE != 0 || ch.size < want) { GC_gpu_disabled = 1; return FALSE; }
    ch.size &= ~(size_t)(HBLKSIZE - 1);
    if (!GC_gpu_add_chunk(&ch)) { GC_gpu_disabled = 1; return FALSE; }
    GC_gpu_last_chunk = ch;
#ifdef TINYGO_GPU_POOL_BDWGC
    GC_add_to_heap_pool((struct hblk *)ch.base, ch.size, GC_POOL_GPU);
#endif
    return TRUE;
}

/* GC_gpu_kind is created LAZILY, on provider registration, not inside
   GC_init: the TinyGo runtime registers the provider from initHeap AFTER
   gcInit() has already run GC_init, so a GC_init-time creation would never
   happen and the pool would stay permanently disabled. GC_new_kind is public
   API and is safe to call after GC_init. */
GC_API void GC_CALL GC_set_gpu_chunk_provider(GC_gpu_chunk_provider_proc fn) {
    GC_gpu_provider = fn;
    if (fn != 0 && GC_gpu_kind < 0) GC_gpu_ensure_kind();
}

#ifndef TINYGO_GPU_POOL_BDWGC
/* A chunk-local bump allocator with no dependency on the bdwgc block pool.
   DEAD IN EVERY PRODUCT BUILD: builder/bdwgc.go always passes
   -DTINYGO_GPU_POOL_BDWGC, so TinyGo always takes the GC_malloc_kind path
   below, where Boehm owns lifetime, sweeping and reuse. This path survives
   only for the standalone C stress harness, which links gpu_pool.c against
   mmap'd fake chunks without the rest of bdwgc and uses it to exercise the
   chunk table, the pointer->chunk lookup and the no-cross-chunk invariant in
   isolation. It never reclaims, by construction. */
STATIC ptr_t GC_gpu_bump_ptr = 0;
STATIC ptr_t GC_gpu_bump_end = 0;

/* Runs under the allocation lock (GC_gpu_expand requires it) and rebinds the
   bump range to the freshly obtained chunk. The tail of the previous chunk is
   abandoned, which is exactly what keeps objects from spanning chunks. */
STATIC void * GC_CALLBACK GC_gpu_expand_locked(void *client_data) {
    size_t need_hblks = *(size_t *)client_data;
    if (!GC_gpu_expand(need_hblks)) return 0;
    GC_gpu_bump_ptr = (ptr_t)GC_gpu_last_chunk.base;
    GC_gpu_bump_end = GC_gpu_bump_ptr + GC_gpu_last_chunk.size;
    return (void *)GC_gpu_bump_ptr;
}

STATIC void * GC_gpu_bump_alloc(size_t lb) {
    /* Guard the hblk round-up below against wrapping. */
    if (lb > (size_t)(-1) - HBLKSIZE) return 0;
    if ((size_t)(GC_gpu_bump_end - GC_gpu_bump_ptr) < lb) {
        size_t need_hblks = (lb + (HBLKSIZE - 1)) / HBLKSIZE;
        if (GC_call_with_alloc_lock(GC_gpu_expand_locked, &need_hblks) == 0) return 0;
        if ((size_t)(GC_gpu_bump_end - GC_gpu_bump_ptr) < lb) return 0;
    }
    ptr_t result = GC_gpu_bump_ptr;
    /* Chunk bases are HBLKSIZE-aligned and every lb is a multiple of 4, so the
       bump pointer stays 4-byte aligned by construction. */
    GC_ASSERT(((uintptr_t)result & 3) == 0);
    GC_gpu_bump_ptr = result + lb;
    return (void *)result;
}
#endif /* !TINYGO_GPU_POOL_BDWGC */

GC_API void * GC_CALL GC_malloc_gpu(size_t lb) {
    if (!GC_gpu_pool_enabled()) return 0;
    /* Reject oversize requests BEFORE rounding: for lb > SIZE_MAX - 3 the
       `lb + 3` below wraps to a tiny value (and `lb == 0` then bumps it to 4),
       which would hand back a 4-byte object for a huge request. No real GPU
       buffer comes anywhere near this size, so rejecting is the safe answer;
       the caller falls back to GC_malloc_atomic, which fails honestly. */
    if (lb > (size_t)(-1) - 3) return 0;
    lb = (lb + 3) & ~(size_t)3;              /* object sizes are multiples of 4 */
    if (lb == 0) lb = 4;
#ifdef TINYGO_GPU_POOL_BDWGC
    return GC_malloc_kind(lb, GC_gpu_kind);  /* real dual-pool path (all product builds) */
#else
    return GC_gpu_bump_alloc(lb);            /* harness-only chunk-local allocator */
#endif
}
#endif /* TINYGO_GPU_POOL */
