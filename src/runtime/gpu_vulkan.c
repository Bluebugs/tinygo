//go:build none

// Ignore the //go:build line above: it only stops the go tool from treating
// this as a cgo C file. compileopts/target.go adds it to ExtraFiles when
// -gpu=webgpu -gpu-host=vulkan is passed on a native linux/amd64 target.

// SPMD GPU offload host backend: direct Vulkan compute (no wgpu-native).
//
// Backs the three externs declared in gpu_vulkan.go with the same contract
// as gpu_native.c: synchronous launches, fail closed (return 0) on any error
// after the GPU path was chosen, and a CPU fallback only through
// spmd_vk_available() returning 0.
//
// Differences from gpu_native.c, all driven by the Vulkan PoC in
// docs/data/gpu-vulkan-poc/ (see phase2-pagefault-vulkan-poc-report.md):
//   - Shaders arrive as SPIR-V produced at build time by naga from the same
//     WGSL, so there is no runtime shader compiler.
//   - Buffers are GPU-owned, host-visible and PERSISTENTLY mapped: upload
//     and readback are plain memcpy into/out of the mapping, with no staging
//     buffers and no map/unmap per launch.
//   - Vulkan has no `layout: 'auto'`, so each kernel gets an explicit
//     descriptor set layout: binding 0 uniform params, bindings 1..bufCount
//     storage buffers, all in descriptor set 0. This matches the
//     @group(0) @binding(n) declarations the transpiler emits, which naga
//     preserves as DescriptorSet 0 / Binding n.
//
// Core Vulkan 1.1 only; no extensions.

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <vulkan/vulkan.h>

// The bdwgc GPU block pool. gc_gpu.h is SPMD-owned (tinygo/lib/bdwgc-gpu);
// compileopts/target.go puts that directory and bdwgc's public include dir on
// this file's include path for -gpu-host=vulkan builds.
#include "gc_gpu.h"

// Fixed-size kernel table (also sizes the descriptor pool); a 65th distinct
// kernel fails register. Tracked in PLAN.md (Vulkan register failures).
#define SPMD_VK_MAX_KERNELS 64
// Must match gpuVulkanMaxBuffers in compiler/gpu_offload.go, which keeps
// larger kernels on the CPU at compile time.
#define SPMD_VK_MAX_BUFFERS 8
#define SPMD_VK_SLOTS (1 + SPMD_VK_MAX_BUFFERS)
#define SPMD_VK_PAGE 4096u
// Fence wait bound. A dispatch normally completes in milliseconds; reaching
// this means the device is hung or lost, and returning 0 makes Go panic
// instead of spinning forever.
#define SPMD_VK_FENCE_TIMEOUT_NS 30000000000ull

// All state below is process-global and guarded by this one lock, for the
// same reason as gpu_native.c: TinyGo's linux target uses threads, and the
// buffer slots are keyed by argument index, so two concurrent launches would
// otherwise silently exchange data.
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// Must match gpuBufferDesc in gpu_vulkan.go and gpuBuildBuffers in
// compiler/gpu_offload.go.
typedef struct {
    uint64_t dataPtr;
    uint32_t byteLen;
    uint32_t mode;
} spmd_vk_buffer_desc;

_Static_assert(sizeof(spmd_vk_buffer_desc) == 16, "gpuBufferDesc must be 16 bytes on a 64-bit target");
_Static_assert(offsetof(spmd_vk_buffer_desc, dataPtr) == 0, "gpuBufferDesc.dataPtr offset");
_Static_assert(offsetof(spmd_vk_buffer_desc, byteLen) == 8, "gpuBufferDesc.byteLen offset");
_Static_assert(offsetof(spmd_vk_buffer_desc, mode) == 12, "gpuBufferDesc.mode offset");

// A persistently mapped buffer. gen increments every time the VkBuffer is
// recreated, so kernels can tell their descriptor set points at a dead
// buffer and must be rewritten.
typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    void *mapped;
    VkDeviceSize size;
    uint64_t gen;
} spmd_vk_slot;

// One binding's descriptor contents, cached so vkUpdateDescriptorSets is
// called only when something actually changes.
typedef struct {
    VkBuffer buf;
    VkDeviceSize off, range;
    uint64_t gen; // slot generation for copied bindings; 0 when bound zero-copy
} spmd_vk_binding;

typedef struct {
    int32_t id;
    VkShaderModule module;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkDescriptorSet set;
    int bufCount;
    // The descriptor contents this kernel's set was last written with, one
    // entry per binding. Replaces the old boundGen[] scheme, which could only
    // express "the slot's VkBuffer was recreated": a zero-copy binding also
    // varies by offset and range, so the full triple must be compared.
    //
    // gen is still part of the identity for COPIED bindings. Destroying and
    // recreating a slot buffer can hand back the SAME VkBuffer handle value,
    // which would make (buf, off, range) compare equal while the descriptor
    // actually referenced a destroyed object. Chunk buffers are never
    // destroyed, so bound bindings carry gen 0.
    spmd_vk_binding bound[SPMD_VK_SLOTS];
    // Per-phase minimum wall time over all launches of this kernel, in
    // nanoseconds. Only maintained when g_phases is set (SPMD_GPU_PHASES).
    uint64_t ph_upload_min, ph_dispatch_min, ph_readback_min, launches;
    int valid;
} spmd_vk_kernel;

static VkInstance g_instance;
static VkPhysicalDevice g_phys;
static VkDevice g_device;
static VkQueue g_queue;
static uint32_t g_qfam;
static VkPhysicalDeviceMemoryProperties g_memprops;
static VkCommandPool g_cmdpool;
static VkCommandBuffer g_cmd;
static VkFence g_fence;
static VkDescriptorPool g_descpool;
// Device limits read at init (VkPhysicalDeviceLimits).
static uint32_t g_max_storage_buffers;
static uint32_t g_max_storage_range;
static uint32_t g_max_uniform_range;

static int g_init_done;
static int g_available;
static int g_broken; // a fence timed out: the command buffer may still be pending
static int g_verbose;
static int g_memtype_printed;
// SPMD_GPU_PHASES: maintain and print per-kernel per-phase minimum times.
// Off by default; when off the only cost on the launch path is this flag test.
static int g_phases;
static void spmd_vk_phases_atexit(void);

static spmd_vk_kernel g_kernels[SPMD_VK_MAX_KERNELS];
static spmd_vk_slot g_slots[SPMD_VK_SLOTS];

static spmd_vk_kernel *spmd_vk_find(int32_t id) {
    spmd_vk_kernel *free_slot = NULL;
    for (int i = 0; i < SPMD_VK_MAX_KERNELS; i++) {
        if (g_kernels[i].valid && g_kernels[i].id == id) {
            return &g_kernels[i];
        }
        if (!g_kernels[i].valid && free_slot == NULL) {
            free_slot = &g_kernels[i];
        }
    }
    return free_slot;
}

// Device type ranking: discrete > integrated > virtual. CPU (llvmpipe,
// lavapipe) and OTHER are never used: a software rasterizer is strictly
// slower than the SIMD CPU path the loop would otherwise take.
static int spmd_vk_device_rank(VkPhysicalDeviceType t) {
    switch (t) {
    case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU: return 3;
    case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return 2;
    case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU: return 1;
    default: return 0;
    }
}

static int spmd_vk_compute_family(VkPhysicalDevice pd, uint32_t *fam) {
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, NULL);
    if (nq == 0) {
        return 0;
    }
    VkQueueFamilyProperties *qf = calloc(nq, sizeof(*qf));
    if (qf == NULL) {
        return 0;
    }
    vkGetPhysicalDeviceQueueFamilyProperties(pd, &nq, qf);
    int found = 0;
    for (uint32_t i = 0; i < nq; i++) {
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
            *fam = i;
            found = 1;
            break;
        }
    }
    free(qf);
    return found;
}

static int32_t spmd_vk_available_locked(void) {
    if (g_init_done) {
        return g_available && !g_broken;
    }
    g_init_done = 1;

    // Objects created before a failing init step are intentionally not
    // destroyed: this runs once per process and the loop falls back to CPU.
    const char *v = getenv("SPMD_GPU_VERBOSE");
    g_verbose = (v != NULL && v[0] != '\0' && v[0] != '0');
    const char *ph = getenv("SPMD_GPU_PHASES");
    g_phases = (ph != NULL && ph[0] != '\0' && ph[0] != '0');
    if (g_phases) {
        // This is the host C file, not bdwgc, so atexit is available (bdwgc's
        // DONT_USE_ATEXIT does not apply here).
        atexit(spmd_vk_phases_atexit);
    }

    VkApplicationInfo ai = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO,
                            .pApplicationName = "tinygo-spmd",
                            .apiVersion = VK_API_VERSION_1_1};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &ai};
    VkResult r = vkCreateInstance(&ici, NULL, &g_instance);
    if (r != VK_SUCCESS) {
        if (g_verbose) fprintf(stderr, "spmd_gpu(vulkan): vkCreateInstance failed (%d)\n", (int)r);
        return 0;
    }

    uint32_t n = 0;
    if (vkEnumeratePhysicalDevices(g_instance, &n, NULL) != VK_SUCCESS || n == 0) {
        if (g_verbose) fprintf(stderr, "spmd_gpu(vulkan): no physical devices\n");
        return 0;
    }
    VkPhysicalDevice *pds = calloc(n, sizeof(*pds));
    if (pds == NULL) {
        return 0;
    }
    r = vkEnumeratePhysicalDevices(g_instance, &n, pds);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) {
        free(pds);
        return 0;
    }
    int bestRank = 0;
    VkPhysicalDeviceProperties bestProps;
    for (uint32_t i = 0; i < n; i++) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(pds[i], &p);
        uint32_t fam;
        int rank = spmd_vk_device_rank(p.deviceType);
        if (g_verbose) {
            fprintf(stderr, "spmd_gpu(vulkan): candidate %s type %d rank %d\n", p.deviceName, (int)p.deviceType, rank);
        }
        if (rank > bestRank && spmd_vk_compute_family(pds[i], &fam)) {
            bestRank = rank;
            g_phys = pds[i];
            g_qfam = fam;
            bestProps = p;
        }
    }
    free(pds);
    if (bestRank == 0) {
        if (g_verbose) fprintf(stderr, "spmd_gpu(vulkan): no non-CPU device with a compute queue\n");
        return 0;
    }
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_memprops);
    g_max_storage_buffers = bestProps.limits.maxPerStageDescriptorStorageBuffers;
    g_max_storage_range = bestProps.limits.maxStorageBufferRange;
    g_max_uniform_range = bestProps.limits.maxUniformBufferRange;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO,
                                   .queueFamilyIndex = g_qfam, .queueCount = 1, .pQueuePriorities = &prio};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci};
    r = vkCreateDevice(g_phys, &dci, NULL, &g_device);
    if (r != VK_SUCCESS) {
        if (g_verbose) fprintf(stderr, "spmd_gpu(vulkan): vkCreateDevice failed (%d)\n", (int)r);
        return 0;
    }
    vkGetDeviceQueue(g_device, g_qfam, 0, &g_queue);

    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                    .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT,
                                    .queueFamilyIndex = g_qfam};
    if (vkCreateCommandPool(g_device, &cpci, NULL, &g_cmdpool) != VK_SUCCESS) {
        return 0;
    }
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                        .commandPool = g_cmdpool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                        .commandBufferCount = 1};
    if (vkAllocateCommandBuffers(g_device, &cbai, &g_cmd) != VK_SUCCESS) {
        return 0;
    }
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g_device, &fci, NULL, &g_fence) != VK_SUCCESS) {
        return 0;
    }
    VkDescriptorPoolSize ps[2] = {
        {VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, SPMD_VK_MAX_KERNELS},
        {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, SPMD_VK_MAX_KERNELS * SPMD_VK_MAX_BUFFERS},
    };
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .maxSets = SPMD_VK_MAX_KERNELS, .poolSizeCount = 2, .pPoolSizes = ps};
    if (vkCreateDescriptorPool(g_device, &dpi, NULL, &g_descpool) != VK_SUCCESS) {
        return 0;
    }

    g_available = 1;
    if (g_verbose) {
        fprintf(stderr, "spmd_gpu(vulkan): device %s type %d\n", bestProps.deviceName, (int)bestProps.deviceType);
    }
    return 1;
}

// spmd_vk_pick_memtype chooses a host-visible memory type for a mapped
// buffer. The PoC measured mapped reads from host-visible types WITHOUT
// HOST_CACHED at ~0.2 GB/s on RADV (write-combined mappings), so a cached
// type is always preferred when one exists; among equals, DEVICE_LOCAL wins.
static uint32_t spmd_vk_pick_memtype(uint32_t allowed) {
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    // HOST_COHERENT is mandatory: this host never calls
    // vkFlushMappedMemoryRanges/vkInvalidateMappedMemoryRanges, so a
    // non-coherent mapping could silently drop uploads or return stale
    // readbacks. Without a coherent type the slot fails (fail closed).
    const VkMemoryPropertyFlags CO = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags want[4] = {DL | HV | HC | CO, HV | HC | CO, DL | HV | CO, HV | CO};
    for (int w = 0; w < 4; w++) {
        for (uint32_t t = 0; t < g_memprops.memoryTypeCount; t++) {
            VkMemoryPropertyFlags f = g_memprops.memoryTypes[t].propertyFlags;
            if ((allowed & (1u << t)) && (f & want[w]) == want[w]) {
                return t;
            }
        }
    }
    return UINT32_MAX;
}

// ---------------------------------------------------------------------------
// GPU allocation pool (zero-copy): bdwgc chunk provider
// ---------------------------------------------------------------------------

// bdwgc requires a chunk base aligned to HBLKSIZE. HBLKSIZE lives in bdwgc's
// private gc_priv.h, which this file deliberately does not include (it needs a
// large set of build-time defines), so the value is spelled out here. It is
// only ever used to reject a misaligned mapping early with a clear message:
// GC_gpu_expand independently re-validates `base % HBLKSIZE` and disables the
// pool if this constant were ever to disagree with bdwgc's.
#define SPMD_VK_HBLKSIZE 4096u

int g_pool_enabled;
uint32_t g_min_ssbo_align = 4;
static int g_pool_reason_printed;

// Test hook (SPMD_GPU_POOL_FAIL_AFTER=N): make the N+1'th chunk request fail,
// to exercise the mid-run provider-failure path, which is otherwise
// unreachable without exhausting device memory. -1 (the default) disables it.
// Read once, in spmd_vk_pool_register, while still single-threaded.
static long g_pool_fail_after = -1;
static long g_pool_chunks_made;

static void spmd_vk_pool_off(const char *why) {
    g_pool_enabled = 0;
    GC_gpu_disable();
    if (g_verbose && !g_pool_reason_printed) {
        g_pool_reason_printed = 1;
        fprintf(stderr, "spmd_gpu(vulkan): zerocopy pool disabled: %s\n", why);
    }
}

// Like spmd_vk_pick_memtype, but both HOST_CACHED and HOST_COHERENT are
// mandatory, so this fails instead of falling through to a weaker type.
//
// HOST_CACHED, because CPU code reads and writes pool memory directly and the
// PoC measured uncached (write-combined) host-visible mappings reading at
// ~0.2 GB/s -- silently turning every CPU access to a marked slice into a
// disaster.
//
// HOST_COHERENT, deliberately, even though design section 1 asks only for
// HOST_VISIBLE|HOST_CACHED. See spmd_vk_has_cached_noncoherent below.
static uint32_t spmd_vk_pick_memtype_cached(uint32_t allowed) {
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    const VkMemoryPropertyFlags DL = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    const VkMemoryPropertyFlags CO = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const VkMemoryPropertyFlags want[2] = {DL | HV | HC | CO, HV | HC | CO};
    for (int w = 0; w < 2; w++) {
        for (uint32_t t = 0; t < g_memprops.memoryTypeCount; t++) {
            VkMemoryPropertyFlags f = g_memprops.memoryTypes[t].propertyFlags;
            if ((allowed & (1u << t)) && (f & want[w]) == want[w]) {
                return t;
            }
        }
    }
    return UINT32_MAX;
}

// Would design section 1's wider HOST_VISIBLE|HOST_CACHED predicate have found
// a type that spmd_vk_pick_memtype_cached rejects for not being coherent?
// Used only to print an accurate disablement reason.
//
// Such a device is supportable, but not by this task alone: a non-coherent
// mapping needs vkFlushMappedMemoryRanges before a dispatch reads pool memory
// and vkInvalidateMappedMemoryRanges after one writes it, both rounded to
// nonCoherentAtomSize. Those calls belong at the launch boundary, which is
// where buffers are bound -- code this task does not own. Rather than ship
// coherence handling that no available device can exercise, the pool fails
// closed here: allocation falls back to the normal heap and every buffer to
// the existing copy path, which is correct, just not zero-copy.
static int spmd_vk_has_cached_noncoherent(uint32_t allowed) {
    const VkMemoryPropertyFlags HV = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    const VkMemoryPropertyFlags HC = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    for (uint32_t t = 0; t < g_memprops.memoryTypeCount; t++) {
        VkMemoryPropertyFlags f = g_memprops.memoryTypes[t].propertyFlags;
        if ((allowed & (1u << t)) && (f & (HV | HC)) == (HV | HC) &&
            (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) == 0) {
            return 1;
        }
    }
    return 0;
}

// Chunk provider for the bdwgc GPU pool: one VkDeviceMemory of a
// HOST_VISIBLE|HOST_CACHED|HOST_COHERENT type, persistently mapped, with one
// storage VkBuffer spanning it.
//
// LOCKING: called with bdwgc's allocation lock held, and possibly with the
// world stopped for a collection. It therefore takes NO lock of its own -- in
// particular not g_lock, which spmd_vk_pool_register holds while registering
// this callback -- and reads only state that is immutable after registration
// (g_device, g_memprops). Do not add a g_lock acquisition here: it would
// invert the registration path's lock order and deadlock.
//
// OWNERSHIP: on success the VkBuffer, the VkDeviceMemory and the mapping are
// kept for the lifetime of the process and are never destroyed. bdwgc does not
// return pool memory to the provider, so live Go objects point into this
// mapping until the program exits. Every FAILURE path, by contrast, destroys
// everything it created before returning.
static int spmd_vk_chunk_alloc(size_t bytes, GC_gpu_chunk *out) {
    if (!g_pool_enabled) {
        return 0;
    }
    if (bytes == 0) {
        spmd_vk_pool_off("zero-sized chunk request");
        return 0;
    }
    if (g_pool_fail_after >= 0 && g_pool_chunks_made >= g_pool_fail_after) {
        // Test hook, not a real failure: exercise the mid-run fallback.
        spmd_vk_pool_off("injected chunk failure (SPMD_GPU_POOL_FAIL_AFTER)");
        return 0;
    }
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
                              .size = bytes,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    void *map = NULL;
    if (vkCreateBuffer(g_device, &bci, NULL, &buf) != VK_SUCCESS) {
        spmd_vk_pool_off("vkCreateBuffer failed");
        return 0;
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_device, buf, &mr);
    uint32_t t = spmd_vk_pick_memtype_cached(mr.memoryTypeBits);
    if (t == UINT32_MAX) {
        vkDestroyBuffer(g_device, buf, NULL);
        spmd_vk_pool_off(spmd_vk_has_cached_noncoherent(mr.memoryTypeBits)
                             ? "only cached-but-non-coherent host-visible memory "
                               "(flush/invalidate not implemented)"
                             : "no HOST_VISIBLE|HOST_CACHED|HOST_COHERENT memory type");
        return 0;
    }
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = mr.size,
                                .memoryTypeIndex = t};
    if (vkAllocateMemory(g_device, &mai, NULL, &mem) != VK_SUCCESS) {
        vkDestroyBuffer(g_device, buf, NULL);
        spmd_vk_pool_off("vkAllocateMemory failed");
        return 0;
    }
    if (vkBindBufferMemory(g_device, buf, mem, 0) != VK_SUCCESS ||
        vkMapMemory(g_device, mem, 0, VK_WHOLE_SIZE, 0, &map) != VK_SUCCESS) {
        vkFreeMemory(g_device, mem, NULL);
        vkDestroyBuffer(g_device, buf, NULL);
        spmd_vk_pool_off("bind/map failed");
        return 0;
    }
    // bdwgc needs an HBLKSIZE-aligned base; a mapped VkDeviceMemory is page
    // aligned, which is >= HBLKSIZE (4096) on this target. Assert, don't
    // assume -- and release everything if the assumption ever fails.
    if ((uintptr_t)map % (uintptr_t)SPMD_VK_HBLKSIZE != 0) {
        vkUnmapMemory(g_device, mem);
        vkFreeMemory(g_device, mem, NULL);
        vkDestroyBuffer(g_device, buf, NULL);
        spmd_vk_pool_off("mapping is not HBLKSIZE aligned");
        return 0;
    }
    // From here the chunk is handed to bdwgc and owned by the pool. Note that
    // GC_gpu_expand re-validates it (`base % HBLKSIZE`, `size < want`) and, if
    // it were ever to reject it, would disable the pool WITHOUT calling back
    // here -- orphaning this buffer and mapping for the life of the process.
    // Unreachable in practice: the alignment check above is the same test, and
    // mr.size >= the requested (HBLKSIZE-multiple) size by Vulkan's own
    // contract. Stated rather than guarded, because there is no correct way to
    // reclaim it from this side once ownership has transferred.
    out->base = (uintptr_t)map;
    out->size = (size_t)mr.size;
    out->tag = (uint64_t)buf;
    g_pool_chunks_made++;
    if (g_verbose) {
        fprintf(stderr, "spmd_gpu(vulkan): zerocopy chunk %zu bytes at %p (memtype %u)\n",
                (size_t)mr.size, map, t);
    }
    return 1;
}

// ---------------------------------------------------------------------------
// Zero-copy launch binding
// ---------------------------------------------------------------------------

static uint64_t spmd_vk_now(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static void spmd_vk_phase_min(uint64_t *slot, uint64_t v, uint64_t launches) {
    if (launches == 0 || v < *slot) {
        *slot = v;
    }
}

static void spmd_vk_phases_atexit(void) {
    for (int i = 0; i < SPMD_VK_MAX_KERNELS; i++) {
        spmd_vk_kernel *k = &g_kernels[i];
        if (!k->valid || k->launches == 0) {
            continue;
        }
        fprintf(stderr,
                "spmd_gpu(vulkan): phases kernel=%d launches=%llu upload_ns=%llu "
                "dispatch_ns=%llu readback_ns=%llu\n",
                (int)k->id, (unsigned long long)k->launches,
                (unsigned long long)k->ph_upload_min,
                (unsigned long long)k->ph_dispatch_min,
                (unsigned long long)k->ph_readback_min);
    }
}

// Decide the binding for one buffer. Returns 1 when the caller must bind
// (*buf/*off/*range filled) and 0 when it must use the slot + memcpy path;
// *why names the reason in the latter case.
//
// LOCKING: GC_gpu_lookup is a lock-free read of a sorted, grow-only chunk
// table, so it is safe here. GC_gpu_pool_enabled() is deliberately NOT called
// -- it can create the GPU allocation kind and take bdwgc's allocation lock.
// g_pool_enabled is the plain flag, re-read every launch because the pool can
// be disabled mid-run by a failing chunk request.
static int spmd_vk_zerocopy(const spmd_vk_buffer_desc *desc, uint32_t i, uint32_t bufCount,
                            VkBuffer *buf, VkDeviceSize *off, VkDeviceSize *range,
                            const char **why) {
    *why = "pool-disabled";
    if (!g_pool_enabled) return 0;
    uintptr_t p = (uintptr_t)desc[i].dataPtr;
    uint32_t len = desc[i].byteLen;
    VkDeviceSize bound = ((VkDeviceSize)len + 3u) & ~(VkDeviceSize)3u;
    const GC_gpu_chunk *c = GC_gpu_lookup(p);
    if (c == NULL) { *why = "outside-pool"; return 0; }
    if (p + (uintptr_t)bound > c->base + c->size) { *why = "spans-chunks"; return 0; }
    VkDeviceSize o = (VkDeviceSize)(p - c->base);
    if (g_min_ssbo_align != 0 && (o % (VkDeviceSize)g_min_ssbo_align) != 0) { *why = "misaligned"; return 0; }
    // Byte-tail rule: a whole-word write could change bytes past the slice's
    // end, which belong to the caller. Read-only bindings are always safe.
    if (desc[i].mode != 0 && (len % 4u) != 0) { *why = "byte-tail-write"; return 0; }
    if (bound > (VkDeviceSize)g_max_storage_range) { *why = "outside-pool"; return 0; }
    // Aliasing: a written binding that overlaps any other buffer in this
    // launch would let the kernel observe partially written data, which the
    // copy path never does. Measured, not theoretical: deliberately aliasing a
    // bound RW over a bound RO corrupted 100% of output bytes (spike 2).
    for (uint32_t j = 0; j < bufCount; j++) {
        if (j == i) continue;
        uintptr_t q = (uintptr_t)desc[j].dataPtr;
        uintptr_t qe = q + (((uintptr_t)desc[j].byteLen + 3u) & ~(uintptr_t)3u);
        if (desc[i].mode != 0 && p < qe && q < p + (uintptr_t)bound) { *why = "aliased-write"; return 0; }
    }
    *buf = (VkBuffer)c->tag; *off = o; *range = bound;
    return 1;
}

static void spmd_vk_slot_free(spmd_vk_slot *s) {
    if (s->mapped != NULL) {
        vkUnmapMemory(g_device, s->mem);
        s->mapped = NULL;
    }
    if (s->buf != VK_NULL_HANDLE) {
        vkDestroyBuffer(g_device, s->buf, NULL);
        s->buf = VK_NULL_HANDLE;
    }
    if (s->mem != VK_NULL_HANDLE) {
        vkFreeMemory(g_device, s->mem, NULL);
        s->mem = VK_NULL_HANDLE;
    }
    s->size = 0;
}

// spmd_vk_slot_ensure grows a slot to at least `need` bytes (page rounded).
// Slots only grow: a relaunched kernel with the same sizes reuses the
// buffer and its mapping with zero Vulkan calls.
static int spmd_vk_slot_ensure(spmd_vk_slot *s, VkDeviceSize need, VkBufferUsageFlags usage) {
    if (need < 4) {
        need = 4;
    }
    if (s->buf != VK_NULL_HANDLE && s->size >= need) {
        return 1;
    }
    spmd_vk_slot_free(s);
    VkDeviceSize size = (need + SPMD_VK_PAGE - 1) & ~(VkDeviceSize)(SPMD_VK_PAGE - 1);
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkResult r = vkCreateBuffer(g_device, &bci, NULL, &s->buf);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): vkCreateBuffer(%llu) failed (%d)\n", (unsigned long long)size, (int)r);
        s->buf = VK_NULL_HANDLE;
        return 0;
    }
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(g_device, s->buf, &mr);
    uint32_t t = spmd_vk_pick_memtype(mr.memoryTypeBits);
    if (t == UINT32_MAX) {
        fprintf(stderr, "spmd_gpu(vulkan): no host-visible HOST_COHERENT memory type (allowed 0x%x)\n", mr.memoryTypeBits);
        spmd_vk_slot_free(s);
        return 0;
    }
    if (g_verbose && !g_memtype_printed) {
        g_memtype_printed = 1;
        fprintf(stderr, "spmd_gpu(vulkan): memory type %u flags 0x%x heap %u\n", t,
                g_memprops.memoryTypes[t].propertyFlags, g_memprops.memoryTypes[t].heapIndex);
    }
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO,
                                .allocationSize = mr.size, .memoryTypeIndex = t};
    r = vkAllocateMemory(g_device, &mai, NULL, &s->mem);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): vkAllocateMemory(%llu) failed (%d)\n", (unsigned long long)mr.size, (int)r);
        s->mem = VK_NULL_HANDLE;
        spmd_vk_slot_free(s);
        return 0;
    }
    if (vkBindBufferMemory(g_device, s->buf, s->mem, 0) != VK_SUCCESS ||
        vkMapMemory(g_device, s->mem, 0, VK_WHOLE_SIZE, 0, &s->mapped) != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): bind/map failed\n");
        s->mapped = NULL;
        spmd_vk_slot_free(s);
        return 0;
    }
    s->size = size;
    s->gen++;
    return 1;
}

static void spmd_vk_kernel_destroy(spmd_vk_kernel *k) {
    // The descriptor set is not freed individually: the pool lacks
    // FREE_DESCRIPTOR_SET_BIT, and sets are released with the pool.
    if (k->pipeline != VK_NULL_HANDLE) vkDestroyPipeline(g_device, k->pipeline, NULL);
    if (k->layout != VK_NULL_HANDLE) vkDestroyPipelineLayout(g_device, k->layout, NULL);
    if (k->dsl != VK_NULL_HANDLE) vkDestroyDescriptorSetLayout(g_device, k->dsl, NULL);
    if (k->module != VK_NULL_HANDLE) vkDestroyShaderModule(g_device, k->module, NULL);
    memset(k, 0, sizeof(*k));
}

static int32_t spmd_vk_register_locked(int32_t kernelID, const char *spirv, uint32_t spirvLen,
                                       const char *entry, uint32_t entryLen, int32_t bufCount) {
    if (!spmd_vk_available_locked()) {
        return 0;
    }
    spmd_vk_kernel *k = spmd_vk_find(kernelID);
    if (k == NULL) {
        fprintf(stderr, "spmd_gpu(vulkan): kernel table full (max %d)\n", SPMD_VK_MAX_KERNELS);
        return 0;
    }
    if (k->valid) {
        return 1;
    }
    if (bufCount < 0 || bufCount > SPMD_VK_MAX_BUFFERS) {
        fprintf(stderr, "spmd_gpu(vulkan): kernel %d wants %d buffers, max is %d\n",
                (int)kernelID, (int)bufCount, SPMD_VK_MAX_BUFFERS);
        return 0;
    }
    if ((uint32_t)bufCount > g_max_storage_buffers) {
        // Pipeline layout creation would fail on this device.
        fprintf(stderr, "spmd_gpu(vulkan): kernel %d wants %d storage buffers, device maxPerStageDescriptorStorageBuffers is %u\n",
                (int)kernelID, (int)bufCount, g_max_storage_buffers);
        return 0;
    }
    if (spirv == NULL || spirvLen == 0 || spirvLen % 4 != 0) {
        fprintf(stderr, "spmd_gpu(vulkan): kernel %d has invalid SPIR-V length %u\n", (int)kernelID, spirvLen);
        return 0;
    }
    memset(k, 0, sizeof(*k));
    k->id = kernelID;
    k->bufCount = bufCount;

    // The SPIR-V arrives as a Go string constant with no alignment
    // guarantee, but pCode is a uint32_t array: copy it into malloc'd
    // (suitably aligned) memory.
    uint32_t *code = malloc(spirvLen);
    // The entry name is a Go string, not NUL-terminated.
    char *name = malloc((size_t)entryLen + 1);
    if (code == NULL || name == NULL) {
        free(code);
        free(name);
        return 0;
    }
    memcpy(code, spirv, spirvLen);
    memcpy(name, entry, entryLen);
    name[entryLen] = '\0';

    int ok = 0;
    VkShaderModuleCreateInfo smi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
                                    .codeSize = spirvLen, .pCode = code};
    VkResult r = vkCreateShaderModule(g_device, &smi, NULL, &k->module);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): vkCreateShaderModule failed for kernel %d (%d)\n", (int)kernelID, (int)r);
        goto out;
    }

    VkDescriptorSetLayoutBinding lb[SPMD_VK_SLOTS];
    for (int i = 0; i <= bufCount; i++) {
        lb[i] = (VkDescriptorSetLayoutBinding){
            .binding = (uint32_t)i,
            .descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
            .descriptorCount = 1,
            .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
    }
    VkDescriptorSetLayoutCreateInfo dli = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                           .bindingCount = (uint32_t)bufCount + 1, .pBindings = lb};
    if ((r = vkCreateDescriptorSetLayout(g_device, &dli, NULL, &k->dsl)) != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): vkCreateDescriptorSetLayout failed (%d)\n", (int)r);
        goto out;
    }
    VkPipelineLayoutCreateInfo pli = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                      .setLayoutCount = 1, .pSetLayouts = &k->dsl};
    if ((r = vkCreatePipelineLayout(g_device, &pli, NULL, &k->layout)) != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): vkCreatePipelineLayout failed (%d)\n", (int)r);
        goto out;
    }
    VkComputePipelineCreateInfo cpi = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .layout = k->layout,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                  .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = k->module, .pName = name}};
    if ((r = vkCreateComputePipelines(g_device, VK_NULL_HANDLE, 1, &cpi, NULL, &k->pipeline)) != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): vkCreateComputePipelines failed for kernel %d entry %s (%d)\n",
                (int)kernelID, name, (int)r);
        goto out;
    }
    VkDescriptorSetAllocateInfo dai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                       .descriptorPool = g_descpool, .descriptorSetCount = 1,
                                       .pSetLayouts = &k->dsl};
    if ((r = vkAllocateDescriptorSets(g_device, &dai, &k->set)) != VK_SUCCESS) {
        fprintf(stderr, "spmd_gpu(vulkan): vkAllocateDescriptorSets failed (%d)\n", (int)r);
        k->set = VK_NULL_HANDLE;
        goto out;
    }
    k->valid = 1;
    ok = 1;
    if (g_verbose) fprintf(stderr, "spmd_gpu(vulkan): registered kernel %d (%d buffers)\n", (int)kernelID, (int)bufCount);
out:
    free(code);
    free(name);
    if (!ok) {
        spmd_vk_kernel_destroy(k);
    }
    return ok;
}

// spmd_vk_check_lost disables the host for the rest of the process when the
// device is lost, so later loops take the CPU path instead of each panicking.
static void spmd_vk_check_lost(VkResult r) {
    if (r == VK_ERROR_DEVICE_LOST) {
        g_broken = 1;
    }
}

static int32_t spmd_vk_launch_locked(int32_t kernelID, uint32_t n,
                                     const void *params, uint32_t paramsLen,
                                     const void *bufs, uint32_t bufCount) {
    if (!g_available || g_broken) {
        return 0;
    }
    spmd_vk_kernel *k = spmd_vk_find(kernelID);
    if (k == NULL || !k->valid || k->id != kernelID) {
        fprintf(stderr, "spmd_gpu(vulkan): launch of unregistered kernel %d\n", (int)kernelID);
        return 0;
    }
    if (bufCount != (uint32_t)k->bufCount) {
        fprintf(stderr, "spmd_gpu(vulkan): kernel %d launched with %u buffers, registered with %d\n",
                (int)kernelID, bufCount, k->bufCount);
        return 0;
    }
    const spmd_vk_buffer_desc *desc = (const spmd_vk_buffer_desc *)bufs;

    // --- zero-copy decisions ---------------------------------------------
    // Decided for every buffer ONCE, before sizing, because the rest of the
    // launch keys off it: a bound buffer needs no slot, no upload and no
    // readback. A single launch may freely mix bound and copied buffers.
    int zbound[SPMD_VK_MAX_BUFFERS];
    VkBuffer zbuf[SPMD_VK_MAX_BUFFERS];
    VkDeviceSize zoff[SPMD_VK_MAX_BUFFERS], zrange[SPMD_VK_MAX_BUFFERS];
    uint64_t t0 = 0, t1 = 0, t2 = 0, t3 = 0;
    for (uint32_t i = 0; i < bufCount; i++) {
        const char *why = NULL;
        zbound[i] = spmd_vk_zerocopy(desc, i, bufCount, &zbuf[i], &zoff[i], &zrange[i], &why);
        if (g_verbose) {
            if (zbound[i]) {
                fprintf(stderr, "spmd_gpu(vulkan): launch kernel=%d buf=%u mode=%u len=%u zerocopy=bound\n",
                        (int)kernelID, i, desc[i].mode, desc[i].byteLen);
            } else {
                fprintf(stderr, "spmd_gpu(vulkan): launch kernel=%d buf=%u mode=%u len=%u zerocopy=copied(%s)\n",
                        (int)kernelID, i, desc[i].mode, desc[i].byteLen, why);
            }
        }
    }

    // --- sizes ---------------------------------------------------------------
    // Uniform: round to 16 so the bound range always covers the std140 size
    // of the Params struct naga declares (struct sizes round to a multiple
    // of their 4-byte alignment and are never smaller than the Go blob).
    VkDeviceSize paramsSize = ((VkDeviceSize)paramsLen + 15u) & ~(VkDeviceSize)15u;
    if (paramsSize < 16) paramsSize = 16;
    // Descriptors are bound with VK_WHOLE_SIZE, so the range is the slot's
    // page-rounded size; that must fit the device's per-descriptor range limit.
    if (((paramsSize + SPMD_VK_PAGE - 1) & ~(VkDeviceSize)(SPMD_VK_PAGE - 1)) > g_max_uniform_range) {
        fprintf(stderr, "spmd_gpu(vulkan): kernel %d params (%llu bytes) exceed device maxUniformBufferRange %u\n",
                (int)kernelID, (unsigned long long)paramsSize, g_max_uniform_range);
        return 0;
    }
    if (!spmd_vk_slot_ensure(&g_slots[0], paramsSize, VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT)) {
        return 0;
    }
    for (uint32_t i = 0; i < bufCount; i++) {
        if (zbound[i]) {
            continue; // bound straight into its chunk buffer: no slot needed
        }
        VkDeviceSize sz = ((VkDeviceSize)desc[i].byteLen + 3u) & ~(VkDeviceSize)3u;
        VkDeviceSize szPage = ((sz < 4 ? 4 : sz) + SPMD_VK_PAGE - 1) & ~(VkDeviceSize)(SPMD_VK_PAGE - 1);
        if (szPage > g_max_storage_range) {
            fprintf(stderr, "spmd_gpu(vulkan): kernel %d buffer %u (%llu bytes) exceeds device maxStorageBufferRange %u\n",
                    (int)kernelID, i, (unsigned long long)sz, g_max_storage_range);
            return 0;
        }
        if (!spmd_vk_slot_ensure(&g_slots[i + 1], sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
            return 0;
        }
    }

    if (g_phases) t0 = spmd_vk_now();
    // --- upload --------------------------------------------------------------
    if (paramsLen > 0) {
        memcpy(g_slots[0].mapped, params, paramsLen);
    }
    memset((uint8_t *)g_slots[0].mapped + paramsLen, 0, (size_t)(paramsSize - paramsLen));
    for (uint32_t i = 0; i < bufCount; i++) {
        // A bound buffer IS the program's memory: there is nothing to upload,
        // for any mode (RO, RW and WO alike).
        if (zbound[i]) {
            continue;
        }
        // mode 2 (write-only, every element provably written) skips upload.
        if (desc[i].mode == 2) {
            continue;
        }
        uint32_t len = desc[i].byteLen;
        VkDeviceSize rounded = ((VkDeviceSize)len + 3u) & ~(VkDeviceSize)3u;
        if (len > 0) {
            memcpy(g_slots[i + 1].mapped, (const void *)(uintptr_t)desc[i].dataPtr, len);
        }
        // Byte slices are packed into u32 words: the last word's unused
        // bytes must be zero, not stale data from an earlier launch.
        if (rounded != len) {
            memset((uint8_t *)g_slots[i + 1].mapped + len, 0, (size_t)(rounded - len));
        }
    }

    // --- descriptor set --------------------------------------------------------
    // Bound with VK_WHOLE_SIZE (the slot's page-rounded size, possibly larger
    // than this launch needs). Safe for the same reason as gpu_native.c: the
    // transpiled shader bounds every index by params.n and never calls
    // arrayLength().
    // Binding 0 (the uniform Params slot) is always a copied slot and is
    // untouched by zero-copy. Bindings 1..bufCount are per-buffer: a bound
    // buffer points into its chunk VkBuffer at an offset, a copied one at its
    // slot with VK_WHOLE_SIZE, exactly as before.
    spmd_vk_binding want[SPMD_VK_SLOTS];
    want[0] = (spmd_vk_binding){.buf = g_slots[0].buf, .off = 0, .range = VK_WHOLE_SIZE, .gen = g_slots[0].gen};
    for (uint32_t i = 0; i < bufCount; i++) {
        if (zbound[i]) {
            want[i + 1] = (spmd_vk_binding){.buf = zbuf[i], .off = zoff[i], .range = zrange[i], .gen = 0};
        } else {
            want[i + 1] = (spmd_vk_binding){
                .buf = g_slots[i + 1].buf, .off = 0, .range = VK_WHOLE_SIZE, .gen = g_slots[i + 1].gen};
        }
    }
    int dirty = 0;
    for (uint32_t i = 0; i <= bufCount; i++) {
        if (k->bound[i].buf != want[i].buf || k->bound[i].off != want[i].off ||
            k->bound[i].range != want[i].range || k->bound[i].gen != want[i].gen) {
            dirty = 1;
        }
    }
    if (dirty) {
        VkDescriptorBufferInfo bi[SPMD_VK_SLOTS];
        VkWriteDescriptorSet wr[SPMD_VK_SLOTS];
        for (uint32_t i = 0; i <= bufCount; i++) {
            bi[i] = (VkDescriptorBufferInfo){.buffer = want[i].buf, .offset = want[i].off, .range = want[i].range};
            wr[i] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = k->set, .dstBinding = i,
                .descriptorCount = 1,
                .descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[i]};
        }
        vkUpdateDescriptorSets(g_device, bufCount + 1, wr, 0, NULL);
        for (uint32_t i = 0; i <= bufCount; i++) {
            k->bound[i] = want[i];
        }
    }
    if (g_phases) t1 = spmd_vk_now();

    // --- record + submit --------------------------------------------------------
    VkResult r = vkResetCommandBuffer(g_cmd, 0);
    VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    if (r != VK_SUCCESS || (r = vkBeginCommandBuffer(g_cmd, &cbi)) != VK_SUCCESS) {
        spmd_vk_check_lost(r);
        fprintf(stderr, "spmd_gpu(vulkan): command buffer begin failed (%d)\n", (int)r);
        return 0;
    }
    vkCmdBindPipeline(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->pipeline);
    vkCmdBindDescriptorSets(g_cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k->layout, 0, 1, &k->set, 0, NULL);
    // Workgroup size 64 is fixed by the transpiler's @workgroup_size(64);
    // the compiler's gpuMaxSafeTrip guard bounds the group count.
    vkCmdDispatch(g_cmd, (uint32_t)(((uint64_t)n + 63u) / 64u), 1, 1);
    if ((r = vkEndCommandBuffer(g_cmd)) != VK_SUCCESS) {
        spmd_vk_check_lost(r);
        fprintf(stderr, "spmd_gpu(vulkan): vkEndCommandBuffer failed (%d)\n", (int)r);
        return 0;
    }
    if ((r = vkResetFences(g_device, 1, &g_fence)) != VK_SUCCESS) {
        spmd_vk_check_lost(r);
        fprintf(stderr, "spmd_gpu(vulkan): vkResetFences failed (%d)\n", (int)r);
        return 0;
    }
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &g_cmd};
    if ((r = vkQueueSubmit(g_queue, 1, &si, g_fence)) != VK_SUCCESS) {
        spmd_vk_check_lost(r);
        fprintf(stderr, "spmd_gpu(vulkan): vkQueueSubmit failed (%d)\n", (int)r);
        return 0;
    }
    r = vkWaitForFences(g_device, 1, &g_fence, VK_TRUE, SPMD_VK_FENCE_TIMEOUT_NS);
    if (r != VK_SUCCESS) {
        // The command buffer may still be executing, so it must not be
        // reset or resubmitted: disable the host for the rest of the process.
        g_broken = 1;
        fprintf(stderr, "spmd_gpu(vulkan): vkWaitForFences failed or timed out (%d)\n", (int)r);
        return 0;
    }
    if (g_phases) t2 = spmd_vk_now();

    // --- readback ---------------------------------------------------------------
    for (uint32_t i = 0; i < bufCount; i++) {
        // A bound buffer was written in place: there is nothing to read back,
        // for RW and WO alike.
        if (zbound[i]) {
            continue;
        }
        // Exactly byteLen, never the rounded length: bytes past a byte
        // slice's end belong to the caller (e.g. a sub-slice).
        if (desc[i].mode != 0 && desc[i].byteLen > 0) {
            memcpy((void *)(uintptr_t)desc[i].dataPtr, g_slots[i + 1].mapped, desc[i].byteLen);
        }
    }
    if (g_phases) {
        t3 = spmd_vk_now();
        spmd_vk_phase_min(&k->ph_upload_min, t1 - t0, k->launches);
        spmd_vk_phase_min(&k->ph_dispatch_min, t2 - t1, k->launches);
        spmd_vk_phase_min(&k->ph_readback_min, t3 - t2, k->launches);
        k->launches++;
    }
    vkResetFences(g_device, 1, &g_fence);
    vkResetCommandBuffer(g_cmd, 0);
    return 1;
}

// Public entry points: one lock/unlock site each, so early returns inside
// the _locked functions cannot leak the (non-recursive) mutex.

// spmd_vk_pool_register registers a bdwgc GPU chunk provider backed by
// host-visible, persistently mapped Vulkan memory. Returns 1 if the pool is
// enabled, 0 if it stays disabled (every allocation then falls back to the
// normal heap and every buffer to the copy path).
//
// spmd_vk_pool_register runs once, from runtime.initHeap, while the program
// is still single-threaded. The Vulkan device is created eagerly here (not
// lazily inside the provider) because the provider can be entered while the
// world is stopped for a GC, and blocking there on a lock held by a stopped
// thread would deadlock.
int32_t spmd_vk_pool_register(void) {
    // Single-threaded entry is a correctness precondition, not a style note:
    // the provider can later run while the world is stopped, so the device
    // must already exist. initHeap runs before any other thread is created.
    static int called;
    if (called) {
        fprintf(stderr, "spmd_gpu(vulkan): spmd_vk_pool_register called twice\n");
        abort();
    }
    called = 1;
    pthread_mutex_lock(&g_lock);
    int32_t ok = spmd_vk_available_locked();
    if (ok) {
        VkPhysicalDeviceProperties p;
        vkGetPhysicalDeviceProperties(g_phys, &p);
        g_min_ssbo_align = (uint32_t)p.limits.minStorageBufferOffsetAlignment;
        if (g_min_ssbo_align == 0) {
            g_min_ssbo_align = 4;
        }
        const char *fa = getenv("SPMD_GPU_POOL_FAIL_AFTER");
        if (fa != NULL && fa[0] != '\0') {
            g_pool_fail_after = strtol(fa, NULL, 10);
        }
        g_pool_enabled = 1;
        // Registration may create the GPU allocation kind, which takes
        // bdwgc's allocation lock. That lock is therefore acquired while
        // g_lock is held; the provider itself never takes g_lock, so the
        // reverse order cannot arise and this cannot deadlock.
        GC_set_gpu_chunk_provider(spmd_vk_chunk_alloc);
    } else {
        spmd_vk_pool_off("no Vulkan device");
    }
    pthread_mutex_unlock(&g_lock);
    return ok;
}

int32_t spmd_vk_available(void) {
    pthread_mutex_lock(&g_lock);
    int32_t r = spmd_vk_available_locked();
    pthread_mutex_unlock(&g_lock);
    return r;
}

int32_t spmd_vk_register(int32_t kernelID, const char *spirv, uint32_t spirvLen,
                         const char *entry, uint32_t entryLen, int32_t bufCount) {
    pthread_mutex_lock(&g_lock);
    int32_t r = spmd_vk_register_locked(kernelID, spirv, spirvLen, entry, entryLen, bufCount);
    pthread_mutex_unlock(&g_lock);
    return r;
}

int32_t spmd_vk_launch(int32_t kernelID, uint32_t n,
                       const void *params, uint32_t paramsLen,
                       const void *bufs, uint32_t bufCount) {
    pthread_mutex_lock(&g_lock);
    int32_t r = spmd_vk_launch_locked(kernelID, n, params, paramsLen, bufs, bufCount);
    pthread_mutex_unlock(&g_lock);
    return r;
}
