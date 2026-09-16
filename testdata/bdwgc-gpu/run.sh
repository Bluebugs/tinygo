#!/usr/bin/env bash
# Builds the vendored bdwgc (TinyGo flags + GC_ASSERTIONS + TINYGO_GPU_POOL)
# and runs the GPU-pool stress harness. Exit 0 only on "PASS".
set -euo pipefail
cd "$(dirname "$0")"
# Sources come from two places: the patched bdwgc tree that `make build-tinygo`
# materialises (pristine submodule + lib/bdwgc-spmd-gpu-pool.patch + the
# SPMD-owned sources) and lib/bdwgc-gpu/ itself. Building from the same tree as
# the product means the harness and TinyGo compile identical sources.
# BDWGC_DIR is overridable, e.g. to point at the raw submodule.
LIB="${BDWGC_DIR:-../../build/bdwgc-patched}"
GPUDIR=../../lib/bdwgc-gpu
OUT=$(mktemp -d)
trap 'rm -rf "$OUT"' EXIT
FLAGS="-DUSE_MMAP -DUSE_MUNMAP -DGC_BUILTIN_ATOMIC -DNO_EXECUTE_PERMISSION \
 -DALL_INTERIOR_POINTERS -DIGNORE_DYNAMIC_LOADING -DNO_GETCONTEXT \
 -DGC_DISABLE_INCREMENTAL -DNO_MSGBOX_ON_ERROR -DDONT_USE_ATEXIT -DNO_GETENV \
 -DNO_CLOCK -DNO_DEBUGGING -DGC_NO_FINALIZATION -DGC_DONT_REGISTER_MAIN_STATIC_DATA \
 -DNO_PROC_STAT -DGC_ASSERTIONS -DTINYGO_GPU_POOL ${EXTRA_FLAGS:-}"
# EXTRA_FLAGS lets Task 1b select the bdwgc-backed allocator
# (-DTINYGO_GPU_POOL_BDWGC) without editing this script.
# TinyGo's own list plus mallocx.c/ptr_chck.c: TinyGo adds those two only on
# Windows, whose linker rejects undefined symbols, but linking this harness as
# a plain ELF executable hits the same undefined references from dbg_mlc.c
# (GC_realloc, GC_malloc_ignore_off_page, GC_is_visible, ...).
SRC="allchblk.c alloc.c blacklst.c dbg_mlc.c dyn_load.c headers.c mach_dep.c malloc.c \
 mark.c mark_rts.c misc.c new_hblk.c os_dep.c reclaim.c mallocx.c ptr_chck.c"
INC="-I$LIB/include -I$LIB/include/gc -I$GPUDIR"
for f in $SRC; do
    cc -O2 -c $FLAGS $INC -o "$OUT/$(basename "$f" .c).o" "$LIB/$f"
done
# The SPMD-owned pool lives outside the submodule.
cc -O2 -c $FLAGS $INC -o "$OUT/gpu_pool.o" "$GPUDIR/gpu_pool.c"
cc -O2 $FLAGS $INC -o "$OUT/stress" stress.c "$OUT"/*.o -lpthread
"$OUT/stress"
