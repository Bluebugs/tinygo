//go:build none

// Ignore the //go:build line above: it only stops the go tool from treating
// this as a cgo C file. compileopts/target.go adds it to ExtraFiles (and so
// to the build) when -gpu=webgpu is passed on a native target.

// SPMD GPU offload host backend for native targets: wgpu-native + Vulkan.
//
// This is the native counterpart of the browser/JS glue (Task 7). It backs
// the three externs declared in gpu_native.go and implements exactly the
// same contract, with one large simplification: a native host can BLOCK on
// the GPU. wgpuDevicePoll(device, /*wait=*/true, NULL) parks the calling
// thread until the queue makes progress, so spmd_gpu_launch is synchronous
// and there is no seq counter, no pending map, no completion callback into
// Go and no scheduler resume -- all of which the wasm path needs only
// because a browser cannot block the JS event loop.
//
// Why this lives in C rather than in Go: the WebGPU C API is a large pile
// of nested descriptor structs, and mirroring them into the TinyGo runtime
// package (which cannot use cgo) would mean hand-transcribing dozens of
// layouts that must stay bit-identical to webgpu.h. Keeping the WebGPU
// surface in C reduces the Go/C contract to three flat functions plus one
// descriptor struct, which IS hand-checked (see gpu_native.go).
//
// Compiled into the binary only when -gpu=webgpu is passed on a native
// target; see compileopts/target.go, which appends this file to ExtraFiles
// and adds -lwgpu_native to the link.

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "webgpu/webgpu.h"
#include "webgpu/wgpu.h"

#define SPMD_GPU_MAX_KERNELS 64
#define SPMD_GPU_MAX_BUFFERS 8

// Bound on how many times a wait loop will poll before giving up. Every
// wait in this file is for an event that normally arrives in microseconds,
// so any of these limits being reached means the device is lost or hung.
// Without a bound, that state is an infinite silent spin; with one, it
// falls through to a `return 0`, which the Go side turns into the same
// clear fail-closed panic every other error path produces.
#define SPMD_GPU_POLL_LIMIT 10000000

// EVERYTHING below -- the device, the kernel table, the buffer slots, and
// the map-completion flags -- is process-global mutable state protected by
// this single lock.
//
// This is NOT belt-and-braces. TinyGo's linux target uses
// `Scheduler: threads`, so two goroutines can be inside spmd_gpu_launch at
// the same time, and g_buffers[] is keyed by ARGUMENT INDEX, not by kernel
// or by call: it holds the LIVE device buffer for the in-flight dispatch,
// not a cache that merely gets recomputed. Unsynchronised, two concurrent
// launches would both wgpuQueueWriteBuffer into g_buffers[0].dev, the
// second clobbering the first's input, and both would then read back --
// so one goroutine silently receives the other's data, with no error, no
// panic, and nothing any correctness gate can observe. Silent wrong
// answers are exactly the failure class this project refuses to ship.
//
// The same lock also closes three narrower holes: g_map_done/g_map_status
// are single globals, so one thread's map callback could release another
// thread's wait loop and leave it calling
// wgpuBufferGetConstMappedRange on an unmapped buffer; ensure_slot can
// Release a buffer another thread has bound or mapped (use-after-free
// inside wgpu); and unsynchronised g_init_done double-entry would create
// two devices.
//
// Documenting "single-threaded use only" was considered and rejected:
// nothing enforces it and the violation is silent.
//
// Cost: none worth measuring. Launches already serialise on the GPU queue,
// and each critical section is one synchronous dispatch, so the lock is
// almost always uncontended and, when it is contended, it is serialising
// work that the hardware was going to serialise anyway.
static pthread_mutex_t g_lock = PTHREAD_MUTEX_INITIALIZER;

// Must match gpuBufferDesc in src/runtime/gpu_native.go AND the LLVM struct
// built by gpuBuildBuffers in compiler/gpu_offload.go for 8-byte pointers:
// {u64 dataPtr, u32 byteLen, u32 mode}, 16 bytes, no padding.
typedef struct {
    uint64_t dataPtr;
    uint32_t byteLen;
    uint32_t mode;
} spmd_gpu_buffer_desc;

_Static_assert(sizeof(spmd_gpu_buffer_desc) == 16, "gpuBufferDesc must be 16 bytes on a 64-bit target");

// Kernel IDs are content hashes assigned by the compiler, not small
// indices, so kernels live in a tiny linear-scan table keyed by that id
// rather than in an array indexed by it. A program has a handful of
// offloaded loops at most, so linear scan is the right structure.
typedef struct {
    int32_t id;
    WGPUComputePipeline pipeline;
    WGPUBindGroupLayout layout;
    int valid;
} spmd_gpu_kernel;

static spmd_gpu_kernel *spmd_gpu_find(int32_t id);

// A cached device-side buffer plus its host-visible staging twin. Buffers
// are cached across launches and only recreated when the required size
// grows: a mandelbrot benchmark relaunches the same kernel hundreds of
// times with identical sizes, and buffer creation is expensive enough to
// dominate the measurement otherwise.
typedef struct {
    WGPUBuffer dev;
    WGPUBuffer staging;
    uint64_t size;
} spmd_gpu_slot;

static WGPUInstance g_instance;
static WGPUAdapter g_adapter;
static WGPUDevice g_device;
static WGPUQueue g_queue;
static int g_init_done;   // 0 = not attempted, 1 = attempted
static int g_available;   // 1 = usable device obtained

static spmd_gpu_kernel g_kernels[SPMD_GPU_MAX_KERNELS];
static spmd_gpu_slot g_uniform;
static spmd_gpu_slot g_buffers[SPMD_GPU_MAX_BUFFERS];

static int g_verbose; // SPMD_GPU_VERBOSE=1 in the environment

// spmd_gpu_find returns the slot holding kernel `id`, or a free slot if it
// is not registered yet, or NULL if the table is full.
static spmd_gpu_kernel *spmd_gpu_find(int32_t id) {
    spmd_gpu_kernel *free_slot = NULL;
    for (int i = 0; i < SPMD_GPU_MAX_KERNELS; i++) {
        if (g_kernels[i].valid && g_kernels[i].id == id) {
            return &g_kernels[i];
        }
        if (!g_kernels[i].valid && free_slot == NULL) {
            free_slot = &g_kernels[i];
        }
    }
    return free_slot;
}

static WGPUStringView sv(const char *p, size_t n) {
    WGPUStringView v;
    v.data = p;
    v.length = n;
    return v;
}

static void spmd_gpu_error_cb(WGPUDevice const *device, WGPUErrorType type,
                              WGPUStringView message, void *u1, void *u2) {
    (void)device; (void)u1; (void)u2;
    fprintf(stderr, "spmd_gpu: WebGPU error %d: %.*s\n", (int)type,
            (int)message.length, message.data ? message.data : "");
}

static WGPUAdapter g_cb_adapter;
static int g_cb_adapter_done;
static void spmd_gpu_adapter_cb(WGPURequestAdapterStatus status, WGPUAdapter adapter,
                                WGPUStringView message, void *u1, void *u2) {
    (void)u1; (void)u2;
    if (status == WGPURequestAdapterStatus_Success) {
        g_cb_adapter = adapter;
    } else if (g_verbose) {
        fprintf(stderr, "spmd_gpu: requestAdapter failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
    }
    g_cb_adapter_done = 1;
}

static WGPUDevice g_cb_device;
static int g_cb_device_done;
static void spmd_gpu_device_cb(WGPURequestDeviceStatus status, WGPUDevice device,
                               WGPUStringView message, void *u1, void *u2) {
    (void)u1; (void)u2;
    if (status == WGPURequestDeviceStatus_Success) {
        g_cb_device = device;
    } else if (g_verbose) {
        fprintf(stderr, "spmd_gpu: requestDevice failed: %.*s\n",
                (int)message.length, message.data ? message.data : "");
    }
    g_cb_device_done = 1;
}

static volatile int g_map_done;
static WGPUMapAsyncStatus g_map_status;
static void spmd_gpu_map_cb(WGPUMapAsyncStatus status, WGPUStringView message,
                            void *u1, void *u2) {
    (void)message; (void)u1; (void)u2;
    g_map_status = status;
    g_map_done = 1;
}

static int32_t spmd_gpu_available_locked(void) {
    if (g_init_done) {
        return g_available;
    }
    g_init_done = 1;

    const char *v = getenv("SPMD_GPU_VERBOSE");
    g_verbose = (v != NULL && v[0] != '\0' && v[0] != '0');

    g_instance = wgpuCreateInstance(NULL);
    if (g_instance == NULL) {
        if (g_verbose) fprintf(stderr, "spmd_gpu: wgpuCreateInstance returned NULL\n");
        return 0;
    }

    WGPURequestAdapterOptions ropts;
    memset(&ropts, 0, sizeof(ropts));
    ropts.backendType = WGPUBackendType_Vulkan;
    ropts.powerPreference = WGPUPowerPreference_HighPerformance;
    WGPURequestAdapterCallbackInfo rci;
    memset(&rci, 0, sizeof(rci));
    rci.mode = WGPUCallbackMode_AllowProcessEvents;
    rci.callback = spmd_gpu_adapter_cb;
    wgpuInstanceRequestAdapter(g_instance, &ropts, rci);
    for (long spins = 0; !g_cb_adapter_done; spins++) {
        if (spins >= SPMD_GPU_POLL_LIMIT) {
            fprintf(stderr, "spmd_gpu: timed out waiting for requestAdapter\n");
            return 0;
        }
        wgpuInstanceProcessEvents(g_instance);
    }
    g_adapter = g_cb_adapter;
    if (g_adapter == NULL) {
        return 0;
    }

    WGPUDeviceDescriptor dd;
    memset(&dd, 0, sizeof(dd));
    dd.uncapturedErrorCallbackInfo.callback = spmd_gpu_error_cb;
    WGPURequestDeviceCallbackInfo dci;
    memset(&dci, 0, sizeof(dci));
    dci.mode = WGPUCallbackMode_AllowProcessEvents;
    dci.callback = spmd_gpu_device_cb;
    wgpuAdapterRequestDevice(g_adapter, &dd, dci);
    for (long spins = 0; !g_cb_device_done; spins++) {
        if (spins >= SPMD_GPU_POLL_LIMIT) {
            fprintf(stderr, "spmd_gpu: timed out waiting for requestDevice\n");
            return 0;
        }
        wgpuInstanceProcessEvents(g_instance);
    }
    g_device = g_cb_device;
    if (g_device == NULL) {
        return 0;
    }
    g_queue = wgpuDeviceGetQueue(g_device);
    if (g_queue == NULL) {
        return 0;
    }
    g_available = 1;
    if (g_verbose) fprintf(stderr, "spmd_gpu: native wgpu device ready (Vulkan)\n");
    return 1;
}

static int32_t spmd_gpu_register_locked(int32_t kernelID, const char *wgsl, uint32_t wgslLen,
                                        const char *entry, uint32_t entryLen) {
    if (!spmd_gpu_available_locked()) {
        return 0;
    }
    spmd_gpu_kernel *k = spmd_gpu_find(kernelID);
    if (k == NULL) {
        fprintf(stderr, "spmd_gpu: kernel table full (max %d)\n", SPMD_GPU_MAX_KERNELS);
        return 0;
    }
    if (k->valid) {
        return 1;
    }

    WGPUShaderSourceWGSL src;
    memset(&src, 0, sizeof(src));
    src.chain.sType = WGPUSType_ShaderSourceWGSL;
    src.code = sv(wgsl, wgslLen);
    WGPUShaderModuleDescriptor smd;
    memset(&smd, 0, sizeof(smd));
    smd.nextInChain = (WGPUChainedStruct *)&src;
    WGPUShaderModule module = wgpuDeviceCreateShaderModule(g_device, &smd);
    if (module == NULL) {
        fprintf(stderr, "spmd_gpu: createShaderModule failed for kernel %d\n", (int)kernelID);
        return 0;
    }

    // The entry point name arrives as a Go string, which is NOT
    // NUL-terminated; WGPUStringView carries an explicit length, so it is
    // passed through unmodified rather than copied into a C string.
    WGPUComputePipelineDescriptor cpd;
    memset(&cpd, 0, sizeof(cpd));
    cpd.compute.module = module;
    cpd.compute.entryPoint = sv(entry, entryLen);
    WGPUComputePipeline pipeline = wgpuDeviceCreateComputePipeline(g_device, &cpd);
    wgpuShaderModuleRelease(module);
    if (pipeline == NULL) {
        fprintf(stderr, "spmd_gpu: createComputePipeline failed for kernel %d\n", (int)kernelID);
        return 0;
    }
    k->id = kernelID;
    k->pipeline = pipeline;
    k->layout = wgpuComputePipelineGetBindGroupLayout(pipeline, 0);
    k->valid = 1;
    if (g_verbose) fprintf(stderr, "spmd_gpu: registered kernel %d\n", (int)kernelID);
    return 1;
}

// ensure_slot grows a cached buffer pair to at least `size` bytes. staging
// is only created when needReadback is set, so read-only inputs do not pay
// for a MapRead twin.
static int ensure_slot(spmd_gpu_slot *slot, uint64_t size, WGPUBufferUsage usage, int needReadback) {
    if (size == 0) {
        size = 4;
    }
    size = (size + 3u) & ~(uint64_t)3u; // WebGPU requires 4-byte-aligned copies
    if (slot->dev != NULL && slot->size >= size && (!needReadback || slot->staging != NULL)) {
        return 1;
    }
    if (slot->dev != NULL) {
        wgpuBufferDestroy(slot->dev);
        wgpuBufferRelease(slot->dev);
        slot->dev = NULL;
    }
    if (slot->staging != NULL) {
        wgpuBufferDestroy(slot->staging);
        wgpuBufferRelease(slot->staging);
        slot->staging = NULL;
    }
    WGPUBufferDescriptor bd;
    memset(&bd, 0, sizeof(bd));
    bd.usage = usage;
    bd.size = size;
    slot->dev = wgpuDeviceCreateBuffer(g_device, &bd);
    if (slot->dev == NULL) {
        return 0;
    }
    if (needReadback) {
        memset(&bd, 0, sizeof(bd));
        bd.usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
        bd.size = size;
        slot->staging = wgpuDeviceCreateBuffer(g_device, &bd);
        if (slot->staging == NULL) {
            return 0;
        }
    }
    slot->size = size;
    return 1;
}

// spmd_gpu_launch runs one kernel dispatch to completion and returns 1 only
// if every output buffer was actually read back. Returning 0 makes the Go
// side panic rather than continue over unwritten memory -- the same
// fail-closed rule the wasm path uses (I2).
static int32_t spmd_gpu_launch_locked(int32_t kernelID, uint32_t n,
                                      const void *params, uint32_t paramsLen,
                                      const void *bufs, uint32_t bufCount) {
    if (!g_available) {
        return 0;
    }
    spmd_gpu_kernel *k = spmd_gpu_find(kernelID);
    if (k == NULL || !k->valid || k->id != kernelID) {
        fprintf(stderr, "spmd_gpu: launch of unregistered kernel %d\n", (int)kernelID);
        return 0;
    }
    if (bufCount > SPMD_GPU_MAX_BUFFERS) {
        fprintf(stderr, "spmd_gpu: kernel %d wants %u buffers, max is %d\n",
                (int)kernelID, bufCount, SPMD_GPU_MAX_BUFFERS);
        return 0;
    }
    const spmd_gpu_buffer_desc *desc = (const spmd_gpu_buffer_desc *)bufs;

    // --- uniform Params ----------------------------------------------------
    if (!ensure_slot(&g_uniform, paramsLen,
                     WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst, 0)) {
        return 0;
    }
    uint32_t paddedParams = (paramsLen + 3u) & ~3u;
    wgpuQueueWriteBuffer(g_queue, g_uniform.dev, 0, params, paddedParams);

    // --- storage buffers ---------------------------------------------------
    int anyReadback = 0;
    for (uint32_t i = 0; i < bufCount; i++) {
        int readback = desc[i].mode != 0;
        anyReadback |= readback;
        if (!ensure_slot(&g_buffers[i], desc[i].byteLen,
                         WGPUBufferUsage_Storage | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst,
                         readback)) {
            return 0;
        }
        // mode 2 (write-only, every element provably written by the kernel)
        // skips the upload; modes 0 and 1 upload. See gpuBuildBuffers.
        if (desc[i].mode != 2) {
            // WebGPU writes must be 4-byte aligned, but byteLen is the Go
            // byte length (a packed byte slice need not be a multiple of 4)
            // and reading past it could run off the end of a mapping. Write
            // the aligned prefix straight from Go memory and the trailing
            // 1-3 bytes from a zero-padded stack word.
            const uint8_t *data = (const uint8_t *)(uintptr_t)desc[i].dataPtr;
            uint32_t aligned = desc[i].byteLen & ~3u;
            if (aligned > 0) {
                wgpuQueueWriteBuffer(g_queue, g_buffers[i].dev, 0, data, aligned);
            }
            if (aligned != desc[i].byteLen) {
                uint8_t tail[4] = {0, 0, 0, 0};
                memcpy(tail, data + aligned, desc[i].byteLen - aligned);
                wgpuQueueWriteBuffer(g_queue, g_buffers[i].dev, aligned, tail, 4);
            }
        }
    }

    // Each entry is bound with the CACHED slot size, which can exceed the
    // caller's byteLen after a slot was grown by an earlier, larger launch.
    // That is harmless as long as the shader derives all of its bounds from
    // Params.n (which the transpiler's `if (idx >= params.n) { return; }`
    // guard does), and it is what lets a slot be reused instead of
    // recreated. It WOULD matter if a generated kernel ever called WGSL's
    // arrayLength(), which reports the bound size rather than n; the
    // transpiler does not emit arrayLength today, and this binding would
    // have to switch to the exact per-launch length if it ever did.
    WGPUBindGroupEntry entries[SPMD_GPU_MAX_BUFFERS + 1];
    memset(entries, 0, sizeof(entries));
    entries[0].binding = 0;
    entries[0].buffer = g_uniform.dev;
    entries[0].size = g_uniform.size;
    for (uint32_t i = 0; i < bufCount; i++) {
        entries[i + 1].binding = i + 1;
        entries[i + 1].buffer = g_buffers[i].dev;
        entries[i + 1].size = g_buffers[i].size;
    }
    WGPUBindGroupDescriptor bgd;
    memset(&bgd, 0, sizeof(bgd));
    bgd.layout = k->layout;
    bgd.entryCount = bufCount + 1;
    bgd.entries = entries;
    WGPUBindGroup bg = wgpuDeviceCreateBindGroup(g_device, &bgd);
    if (bg == NULL) {
        return 0;
    }

    WGPUCommandEncoder enc = wgpuDeviceCreateCommandEncoder(g_device, NULL);
    WGPUComputePassEncoder pass = wgpuCommandEncoderBeginComputePass(enc, NULL);
    wgpuComputePassEncoderSetPipeline(pass, k->pipeline);
    wgpuComputePassEncoderSetBindGroup(pass, 0, bg, 0, NULL);
    // Workgroup size 64 is fixed by the WGSL the transpiler emits
    // (@compute @workgroup_size(64)); the compiler's gpuMaxSafeTrip guard
    // already refused any n that would exceed 65535 workgroups.
    wgpuComputePassEncoderDispatchWorkgroups(pass, (n + 63u) / 64u, 1, 1);
    wgpuComputePassEncoderEnd(pass);
    wgpuComputePassEncoderRelease(pass);

    for (uint32_t i = 0; i < bufCount; i++) {
        if (desc[i].mode != 0) {
            uint32_t len = (desc[i].byteLen + 3u) & ~3u;
            if (len > 0) {
                wgpuCommandEncoderCopyBufferToBuffer(enc, g_buffers[i].dev, 0,
                                                     g_buffers[i].staging, 0, len);
            }
        }
    }
    WGPUCommandBuffer cmd = wgpuCommandEncoderFinish(enc, NULL);
    wgpuQueueSubmit(g_queue, 1, &cmd);
    wgpuCommandBufferRelease(cmd);
    wgpuCommandEncoderRelease(enc);
    wgpuBindGroupRelease(bg);

    if (!anyReadback) {
        // Nothing to map, but the caller still expects the kernel to have
        // finished before it observes anything, so drain the queue.
        wgpuDevicePoll(g_device, 1, NULL);
        return 1;
    }

    int32_t ok = 1;
    for (uint32_t i = 0; i < bufCount; i++) {
        if (desc[i].mode == 0) {
            continue;
        }
        uint32_t len = (desc[i].byteLen + 3u) & ~3u;
        if (len == 0) {
            continue;
        }
        g_map_done = 0;
        WGPUBufferMapCallbackInfo mci;
        memset(&mci, 0, sizeof(mci));
        mci.mode = WGPUCallbackMode_AllowProcessEvents;
        mci.callback = spmd_gpu_map_cb;
        wgpuBufferMapAsync(g_buffers[i].staging, WGPUMapMode_Read, 0, len, mci);
        // This is the whole reason the native backend exists: a blocking
        // poll instead of an event-loop round trip. On the Deno/browser
        // path the equivalent wait costs a fixed ~11.3 ms per launch
        // regardless of payload size; here it costs tens of microseconds.
        int timedOut = 0;
        for (long spins = 0; !g_map_done; spins++) {
            if (spins >= SPMD_GPU_POLL_LIMIT) {
                fprintf(stderr, "spmd_gpu: timed out waiting for buffer %u readback\n", i);
                timedOut = 1;
                break;
            }
            wgpuDevicePoll(g_device, 1, NULL);
        }
        if (timedOut) {
            // Fail closed: the output buffer was NOT written, so do not
            // touch the mapped range and do not report success.
            ok = 0;
            continue;
        }
        if (g_map_status != WGPUMapAsyncStatus_Success) {
            fprintf(stderr, "spmd_gpu: mapAsync failed for buffer %u (status %d)\n",
                    i, (int)g_map_status);
            ok = 0;
            continue;
        }
        const void *src = wgpuBufferGetConstMappedRange(g_buffers[i].staging, 0, len);
        if (src == NULL) {
            ok = 0;
        } else {
            // Exactly byteLen, never the 4-aligned staging length: this is
            // what makes packed byte buffers safe, since the bytes past a
            // byte slice's end belong to the caller (e.g. a sub-slice).
            memcpy((void *)(uintptr_t)desc[i].dataPtr, src, desc[i].byteLen);
        }
        wgpuBufferUnmap(g_buffers[i].staging);
    }
    return ok;
}

// ---------------------------------------------------------------------------
// Public entry points. These are the only symbols gpu_native.go binds to.
//
// Each is a thin lock/call/unlock wrapper around the corresponding _locked
// function, so that the many early returns inside those functions cannot
// leak the lock: there is exactly ONE unlock site per entry point, on the
// single return path of the wrapper. Adding an early return inside a
// _locked function therefore cannot introduce a lock leak.
//
// The mutex is not recursive, so the _locked functions must never call the
// public wrappers -- spmd_gpu_register_locked calls
// spmd_gpu_available_locked directly for that reason.
// ---------------------------------------------------------------------------

int32_t spmd_gpu_available(void) {
    pthread_mutex_lock(&g_lock);
    int32_t r = spmd_gpu_available_locked();
    pthread_mutex_unlock(&g_lock);
    return r;
}

int32_t spmd_gpu_register(int32_t kernelID, const char *wgsl, uint32_t wgslLen,
                          const char *entry, uint32_t entryLen) {
    pthread_mutex_lock(&g_lock);
    int32_t r = spmd_gpu_register_locked(kernelID, wgsl, wgslLen, entry, entryLen);
    pthread_mutex_unlock(&g_lock);
    return r;
}

int32_t spmd_gpu_launch(int32_t kernelID, uint32_t n,
                        const void *params, uint32_t paramsLen,
                        const void *bufs, uint32_t bufCount) {
    pthread_mutex_lock(&g_lock);
    int32_t r = spmd_gpu_launch_locked(kernelID, n, params, paramsLen, bufs, bufCount);
    pthread_mutex_unlock(&g_lock);
    return r;
}
