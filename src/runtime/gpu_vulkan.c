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

#include <vulkan/vulkan.h>

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

typedef struct {
    int32_t id;
    VkShaderModule module;
    VkDescriptorSetLayout dsl;
    VkPipelineLayout layout;
    VkPipeline pipeline;
    VkDescriptorSet set;
    int bufCount;
    // Slot generations this kernel's descriptor set was last written with;
    // 0 means never written (slot generations start at 1).
    uint64_t boundGen[SPMD_VK_SLOTS];
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

    // --- upload --------------------------------------------------------------
    if (paramsLen > 0) {
        memcpy(g_slots[0].mapped, params, paramsLen);
    }
    memset((uint8_t *)g_slots[0].mapped + paramsLen, 0, (size_t)(paramsSize - paramsLen));
    for (uint32_t i = 0; i < bufCount; i++) {
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
    int dirty = 0;
    for (uint32_t i = 0; i <= bufCount; i++) {
        if (k->boundGen[i] != g_slots[i].gen) dirty = 1;
    }
    if (dirty) {
        VkDescriptorBufferInfo bi[SPMD_VK_SLOTS];
        VkWriteDescriptorSet wr[SPMD_VK_SLOTS];
        for (uint32_t i = 0; i <= bufCount; i++) {
            bi[i] = (VkDescriptorBufferInfo){.buffer = g_slots[i].buf, .offset = 0, .range = VK_WHOLE_SIZE};
            wr[i] = (VkWriteDescriptorSet){
                .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = k->set, .dstBinding = i,
                .descriptorCount = 1,
                .descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
                .pBufferInfo = &bi[i]};
        }
        vkUpdateDescriptorSets(g_device, bufCount + 1, wr, 0, NULL);
        for (uint32_t i = 0; i <= bufCount; i++) {
            k->boundGen[i] = g_slots[i].gen;
        }
    }

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

    // --- readback ---------------------------------------------------------------
    for (uint32_t i = 0; i < bufCount; i++) {
        // Exactly byteLen, never the rounded length: bytes past a byte
        // slice's end belong to the caller (e.g. a sub-slice).
        if (desc[i].mode != 0 && desc[i].byteLen > 0) {
            memcpy((void *)(uintptr_t)desc[i].dataPtr, g_slots[i + 1].mapped, desc[i].byteLen);
        }
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
// TEMPORARY: the real provider lands in Task 5 of the zero-copy plan. Until
// then this returns 0, so runtime.gpuPoolInit links and the pool is inert.
int32_t spmd_vk_pool_register(void) {
    return 0;
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
