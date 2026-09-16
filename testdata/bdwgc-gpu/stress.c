/* Standalone GPU-pool stress harness (Spike 1 exit gate, kept as a regression test).
   Fake chunks come from mmap, so no Vulkan is needed. */
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <time.h>

#include "gc.h"
#include "gc_inline.h" /* GC_malloc_kind is declared here, not in gc.h */
#include "gc_gpu.h"

#define KEEP 4096

/* Allocation rounds. With the real dual-pool allocator (Task 1b) Boehm
   reclaims, so the full 10^7 rounds run in bounded memory. The Task 1a
   chunk-local bump allocator has no reclamation at all, so the same round
   count would demand terabytes of resident chunk memory; the default is
   therefore reduced when TINYGO_GPU_POOL_BDWGC is off. Override with
   -DALLOC_ROUNDS=<n>. */
#ifndef ALLOC_ROUNDS
#  ifdef TINYGO_GPU_POOL_BDWGC
#    define ALLOC_ROUNDS 10000000
#  else
#    define ALLOC_ROUNDS 10000
#  endif
#endif

/* Lookups allocate nothing, so the lookup-latency measurement always runs the
   full 10^7 iterations regardless of ALLOC_ROUNDS. */
#ifndef LOOKUP_ROUNDS
#  define LOOKUP_ROUNDS 10000000
#endif

/* Collect ~100 times over the run whatever ALLOC_ROUNDS is; for the 10^7
   default this is exactly every 100000 iterations. */
#define GC_EVERY (ALLOC_ROUNDS / 100 + 1)

static size_t g_chunks;

static int fake_chunk(size_t bytes, GC_gpu_chunk *out) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return 0;
    out->base = (uintptr_t)p;
    out->size = bytes;
    out->tag = ++g_chunks;
    return 1;
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

struct kept { void *p; size_t n; size_t off; unsigned char pat; };
static struct kept keep[KEEP];
static void *normal_head;

static size_t rnd_size(unsigned *s) {
    *s = *s * 1103515245u + 12345u;
    unsigned k = (*s >> 16) % 100;
    if (k < 70) return 16 + (*s % 240);          /* small */
    if (k < 95) return 4096 + (*s % 60000);      /* medium */
    return (size_t)1 << (20 + (*s % 4));         /* 1..8 MB large */
}

/* Like-for-like timing for the <=1.5x STOP GATE.
   IMPORTANT: this ratio is only STOP-GATE evidence when built with
   -DTINYGO_GPU_POOL_BDWGC. Without it, GC_malloc_gpu is the Task 1a
   chunk-local bump allocator, which is trivially faster than GC_malloc_atomic
   (it never reclaims and never collects), so the ratio passes vacuously. The
   gate is therefore only enforced under that guard; see the printed NOTE.
   Two rules make the ratio mean something:
     1. SAME SIZES -- every timed GC_malloc_gpu(n) is paired with a
        GC_malloc_atomic(n) of exactly the same n, so the two totals cover
        one identical size distribution.
     2. INDEPENDENT TIMERS -- each allocator has its own start stamp, taken
        immediately before its own call and accumulated immediately after it.
        A single shared stamp would make each total include the other
        allocator plus the check/memset/retention work between them, which
        is exactly the mistake that makes such a ratio meaningless.
   The clock cost is bounded by only timing one paired sample per batch, so
   clock_gettime does not dominate. The batch shrinks with ALLOC_ROUNDS so a
   short run still collects ~1000 paired samples. */
#ifndef TIME_BATCH
#  if ALLOC_ROUNDS >= 1000000
#    define TIME_BATCH 1000
#  else
#    define TIME_BATCH 10
#  endif
#endif

static void check_object(void *p, size_t n) {
    const GC_gpu_chunk *c = GC_gpu_lookup((uintptr_t)p);
    /* NB: check_object is also called on deliberately misaligned INTERIOR
       pointers (keep[].off), so it must not check alignment. Alignment is
       checked at the allocation site instead. */
    if (c == NULL) { fprintf(stderr, "FAIL: GPU object %p outside every chunk\n", p); exit(1); }
    if ((uintptr_t)p + n > c->base + c->size) {
        fprintf(stderr, "FAIL: GPU object %p size %zu spans past chunk %llu\n", p, n,
                (unsigned long long)c->tag);
        exit(1);
    }
}

int main(void) {
    GC_set_gpu_chunk_provider(fake_chunk);
    GC_INIT();
    /* We build with -DGC_DONT_REGISTER_MAIN_STATIC_DATA, exactly like TinyGo
       (which scans its roots itself), so this program's static data is NOT a
       root by default. keep[] and normal_head are the only statics holding
       live references, so register them explicitly. Without this the
       retention checks below do not test retention at all: every kept object
       is unreachable, gets reclaimed, and its memory is handed out again.
       (Task 1a could not notice: chunk memory was not GC-managed at all.) */
    GC_add_roots((char *)keep, (char *)keep + sizeof(keep));
    GC_add_roots((char *)&normal_head, (char *)&normal_head + sizeof(normal_head));
    if (!GC_gpu_pool_enabled()) { fprintf(stderr, "FAIL: pool not enabled\n"); return 1; }

    /* Oversize requests must be REJECTED, not silently truncated: the
       multiple-of-4 rounding inside GC_malloc_gpu (lb + 3) wraps for
       lb > SIZE_MAX - 3, which would otherwise return a 4-byte object for a
       huge request. */
    if (GC_malloc_gpu(SIZE_MAX) != NULL) {
        fprintf(stderr, "FAIL: GC_malloc_gpu(SIZE_MAX) did not return NULL\n"); return 1;
    }
    if (GC_malloc_gpu(SIZE_MAX - 3) != NULL) {
        fprintf(stderr, "FAIL: GC_malloc_gpu(SIZE_MAX-3) did not return NULL\n"); return 1;
    }
    if (GC_malloc_gpu(SIZE_MAX - 4) != NULL) {
        fprintf(stderr, "FAIL: GC_malloc_gpu(SIZE_MAX-4) did not return NULL\n"); return 1;
    }

    unsigned s = 1;
    /* OR of every base pointer returned by GC_malloc_gpu. The lowest set bit
       of the result is the strongest alignment every result satisfied, which
       is what the launch-time binding rule needs to know (a bound buffer's
       offset must be a multiple of minStorageBufferOffsetAlignment). */
    uintptr_t align_or = 0;
    double gpu_ns = 0, atomic_ns = 0;
    size_t gpu_ops = 0, atomic_ops = 0;

    for (long i = 0; i < ALLOC_ROUNDS; i++) {
        size_t n = rnd_size(&s);
        /* Timed sample: one paired GPU/atomic allocation per batch. Each
           timer wraps ONLY its own call. */
        int timed = (i % TIME_BATCH == 0);
        double t0_gpu = 0, t0_atomic = 0;

        if (timed) t0_gpu = now_s();
        void *p = GC_malloc_gpu(n);
        if (timed) { gpu_ns += (now_s() - t0_gpu) * 1e9; gpu_ops++; }
        if (p == NULL) { fprintf(stderr, "FAIL: GC_malloc_gpu(%zu) returned NULL\n", n); return 1; }
        /* Base pointers are 4-byte aligned (sizes are rounded up to a
           multiple of 4 and chunk bases are HBLKSIZE-aligned). */
        if (((uintptr_t)p & 3u) != 0) {
            fprintf(stderr, "FAIL: GC_malloc_gpu(%zu) returned unaligned %p\n", n, p); return 1;
        }
        align_or |= (uintptr_t)p;
        check_object(p, n);
        unsigned char pat = (unsigned char)(i & 0xff);
        memset(p, pat, n);

        /* The like-for-like control: an atomic allocation of the SAME size n,
           on the same timed iterations, with its OWN timer. */
        if (timed) t0_atomic = now_s();
        void **q = GC_malloc_atomic(n);
        if (timed) { atomic_ns += (now_s() - t0_atomic) * 1e9; atomic_ops++; }
        if (q == NULL) { fprintf(stderr, "FAIL: GC_malloc_atomic returned NULL\n"); return 1; }
        if (GC_gpu_lookup((uintptr_t)q) != NULL) {
            fprintf(stderr, "FAIL: normal object %p landed in a GPU chunk\n", (void *)q); return 1;
        }
        if ((i % 16) == 0) { void *pp = GC_malloc(64); *(void **)pp = normal_head; normal_head = pp; }

        /* Retain some objects through an INTERIOR pointer ONLY: the base
           pointer is deliberately dropped. Under -DTINYGO_GPU_POOL_BDWGC this
           is a real test: the object survives a collection only if
           ALL_INTERIOR_POINTERS marking works on GPU-pool blocks.
           WITHOUT that guard the check is VACUOUS and proves nothing about
           retention: GC_add_to_heap_pool is compiled out, so chunk memory is
           never handed to the collector, and GC_gcollect can neither reclaim
           nor corrupt these objects. The checks are kept in place so Task 1b
           gets them for free the moment the pool is really GC-managed.
           keep[].off records how far in the stored pointer is. */
        size_t slot = (size_t)(s >> 8) % KEEP;
        if (keep[slot].p != NULL) {
            unsigned char *base = (unsigned char *)keep[slot].p - keep[slot].off;
            for (size_t k = 0; k < keep[slot].n; k++) {
                if (base[k] != keep[slot].pat) {
                    fprintf(stderr, "FAIL: kept object %p corrupted at %zu\n", (void *)base, k);
                    return 1;
                }
            }
        }
        keep[slot].off = (n > 64 ? 33 : 0);
        keep[slot].p = (unsigned char *)p + keep[slot].off; /* interior pointer only */
        keep[slot].n = n; keep[slot].pat = pat;

        if ((i % GC_EVERY) == 0) {
            GC_gcollect();
            /* Structural check of the block free lists themselves, not just of
               the objects we happen to hold. This catches a block that lost or
               gained GPU_POOL_BLK and therefore sits on the wrong pool's free
               list -- e.g. a GPU block stranded on the main list, which
               GC_unmap_old would then unmap even though it is a Vulkan
               mapping, or a main block on the GPU list that could be served to
               a GPU allocation from outside every chunk. */
            size_t bad = GC_gpu_verify_pools();
            if (bad != 0) {
                fprintf(stderr, "FAIL: %zu pool/chunk invariant violation(s) "
                        "on the block free lists at round %ld\n", bad, i);
                return 1;
            }
            /* check_object() asks "does [p, p+n) lie inside one chunk?", so it
               must be given the object BASE. keep[k].p is deliberately an
               interior pointer (offset keep[k].off), and passing it together
               with the full size n would overshoot the object's end by off
               bytes and report a bogus "spans past chunk" for any object that
               happens to sit at the tail of a chunk. */
            for (size_t k = 0; k < KEEP; k++)
                if (keep[k].p)
                    check_object((unsigned char *)keep[k].p - keep[k].off, keep[k].n);
        }
    }

    /* Lookup timing: 10^7 lookups over live objects. */
    double t0_lookup = now_s();
    volatile size_t sink = 0;
    for (long i = 0; i < LOOKUP_ROUNDS; i++) {
        const GC_gpu_chunk *c = GC_gpu_lookup((uintptr_t)keep[i % KEEP].p);
        sink += (size_t)(c != NULL);
    }
    double lookup_ns = (now_s() - t0_lookup) * 1e9 / (double)LOOKUP_ROUNDS;

    /* Both totals come from independent timers over identical sizes. Whether
       the ratio is meaningful depends on which allocator was built in. */
    double gpu_mean = gpu_ns / (double)gpu_ops;
    double atomic_mean = atomic_ns / (double)atomic_ops;
    double ratio = gpu_mean / atomic_mean;
    printf("chunks=%zu samples=%zu gpu_ns=%.1f atomic_ns=%.1f ratio=%.2f lookup_ns=%.1f sink=%zu\n",
           GC_gpu_chunk_count(), gpu_ops, gpu_mean, atomic_mean, ratio, lookup_ns, sink);
    /* Lowest set bit of the OR of all base pointers. */
    printf("min_align=%zu\n", (size_t)(align_or & (~align_or + 1)));
#ifdef TINYGO_GPU_POOL_BDWGC
    /* The real STOP GATE: GC_malloc_gpu is the dual-pool allocator here. */
    if (ratio > 1.5) {
        fprintf(stderr, "FAIL: GC_malloc_gpu mean %.1f ns is more than 1.5x GC_malloc_atomic mean %.1f ns (ratio %.2f)\n",
                gpu_mean, atomic_mean, ratio);
        return 1;
    }
#else
    printf("NOTE: bump-allocator ratio, not stop-gate evidence "
           "(build with -DTINYGO_GPU_POOL_BDWGC to enforce the <=1.5x gate)\n");
    printf("NOTE: interior-pointer retention checks are vacuous without "
           "-DTINYGO_GPU_POOL_BDWGC (chunks are not GC-managed yet)\n");
#endif
    if (lookup_ns >= 20.0) { fprintf(stderr, "FAIL: lookup %.1f ns >= 20 ns\n", lookup_ns); return 1; }
    printf("PASS\n");
    return 0;
}
