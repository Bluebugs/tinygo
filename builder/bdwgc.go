package builder

// The well-known conservative Boehm-Demers-Weiser GC.
// This file provides a way to compile this GC for use with TinyGo.

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/tinygo-org/tinygo/goenv"
)

// The SPMD GPU block pool (see docs/data/gpu-zerocopy/spike1.md) needs edits to
// bdwgc's own sources. Those edits are kept as a checked-in patch rather than as
// commits on the lib/bdwgc submodule, so the submodule stays at its upstream
// commit and no unfetchable gitlink is ever shipped. The patch is applied to a
// throwaway copy of the tree at build time.
const (
	bdwgcPatchName  = "lib/bdwgc-spmd-gpu-pool.patch"
	bdwgcGPUSrcName = "lib/bdwgc-gpu"
	bdwgcPatchedRel = "build/bdwgc-patched"
	bdwgcStampName  = ".spmd-stamp"
)

var (
	bdwgcPatchedOnce sync.Once
	bdwgcPatchedPath string
	bdwgcPatchedErr  error
)

// bdwgcPatchedDir materialises the SPMD GPU-pool build tree: a copy of the
// pristine lib/bdwgc submodule with lib/bdwgc-spmd-gpu-pool.patch applied and
// the SPMD-owned sources from lib/bdwgc-gpu/ copied in. The submodule itself
// stays at its upstream commit, so no unfetchable gitlink is ever shipped.
// It is regenerated only when the patch, the SPMD sources or the submodule
// contents change (a stamp file records a hash of all three).
func bdwgcPatchedDir() (string, error) {
	bdwgcPatchedOnce.Do(func() {
		bdwgcPatchedPath, bdwgcPatchedErr = buildBdwgcPatchedDir()
	})
	return bdwgcPatchedPath, bdwgcPatchedErr
}

func buildBdwgcPatchedDir() (string, error) {
	root := goenv.Get("TINYGOROOT")
	srcDir := filepath.Join(root, "lib/bdwgc")
	gpuDir := filepath.Join(root, bdwgcGPUSrcName)
	patchFile := filepath.Join(root, bdwgcPatchName)
	outDir := filepath.Join(root, bdwgcPatchedRel)

	want, err := bdwgcStamp(srcDir, gpuDir, patchFile)
	if err != nil {
		return "", err
	}

	// Fast path: the tree on disk already matches all three inputs, so
	// repeated builds do no work.
	stampPath := filepath.Join(outDir, bdwgcStampName)
	if got, err := os.ReadFile(stampPath); err == nil && string(got) == want {
		return outDir, nil
	}

	// Rebuild from scratch. Materialise into a sibling temporary directory and
	// rename, so a killed build can never leave a half-patched tree behind that
	// a later build would mistake for a good one.
	if err := os.MkdirAll(filepath.Dir(outDir), 0o777); err != nil {
		return "", err
	}
	tmpDir, err := os.MkdirTemp(filepath.Dir(outDir), "bdwgc-patched.tmp*")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(tmpDir)

	if err := copyTree(srcDir, tmpDir); err != nil {
		return "", fmt.Errorf("bdwgc: copying %s: %w", srcDir, err)
	}
	if err := applyPatch(tmpDir, patchFile); err != nil {
		return "", err
	}
	// The SPMD-owned sources live outside the submodule: .c files next to
	// bdwgc's own, headers where gc_priv.h's #include "gc_gpu.h" can find them.
	if err := copyGPUSources(gpuDir, tmpDir); err != nil {
		return "", err
	}
	if err := os.WriteFile(filepath.Join(tmpDir, bdwgcStampName), []byte(want), 0o644); err != nil {
		return "", err
	}

	if err := os.RemoveAll(outDir); err != nil {
		return "", err
	}
	if err := os.Rename(tmpDir, outDir); err != nil {
		return "", err
	}
	return outDir, nil
}

// bdwgcStamp hashes the three inputs of the patched tree: the pristine
// submodule contents, the SPMD-owned sources and the patch itself. Hashing the
// submodule's files rather than its git commit keeps this working in source
// trees that have no .git (release tarballs).
func bdwgcStamp(srcDir, gpuDir, patchFile string) (string, error) {
	h := sha256.New()
	for _, dir := range []string{srcDir, gpuDir} {
		if err := hashTree(h, dir); err != nil {
			return "", err
		}
	}
	patch, err := os.ReadFile(patchFile)
	if err != nil {
		return "", fmt.Errorf("bdwgc: reading %s: %w", patchFile, err)
	}
	fmt.Fprintf(h, "patch %d\n", len(patch))
	h.Write(patch)
	return hex.EncodeToString(h.Sum(nil)), nil
}

// hashTree feeds every regular file under dir (path and contents) into h, in a
// deterministic order.
func hashTree(h io.Writer, dir string) error {
	var paths []string
	err := filepath.WalkDir(dir, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			// Skip the submodule's git metadata: it changes without the
			// sources changing.
			if d.Name() == ".git" {
				return fs.SkipDir
			}
			return nil
		}
		if !d.Type().IsRegular() {
			return nil
		}
		paths = append(paths, path)
		return nil
	})
	if err != nil {
		return err
	}
	sort.Strings(paths)
	for _, path := range paths {
		rel, err := filepath.Rel(dir, path)
		if err != nil {
			return err
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		fmt.Fprintf(h, "file %s %d\n", filepath.ToSlash(rel), len(data))
		h.Write(data)
	}
	return nil
}

func copyTree(src, dst string) error {
	return filepath.WalkDir(src, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, err := filepath.Rel(src, path)
		if err != nil {
			return err
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return fs.SkipDir
			}
			return os.MkdirAll(filepath.Join(dst, rel), 0o777)
		}
		if !d.Type().IsRegular() {
			return nil
		}
		return copyFile(path, filepath.Join(dst, rel))
	})
}

func copyFile(src, dst string) error {
	if err := os.MkdirAll(filepath.Dir(dst), 0o777); err != nil {
		return err
	}
	in, err := os.Open(src)
	if err != nil {
		return err
	}
	defer in.Close()
	out, err := os.Create(dst)
	if err != nil {
		return err
	}
	if _, err := io.Copy(out, in); err != nil {
		out.Close()
		return err
	}
	return out.Close()
}

// applyPatch applies the GPU-pool patch to the freshly copied tree. The patch
// only ever touches files tracked by the submodule; the SPMD-owned sources are
// copied in separately.
func applyPatch(dir, patchFile string) error {
	f, err := os.Open(patchFile)
	if err != nil {
		return fmt.Errorf("bdwgc: reading %s: %w", patchFile, err)
	}
	defer f.Close()
	// --batch: never prompt. Without it, `patch` asks on stdin when a hunk
	// looks already applied (which happens if lib/bdwgc is left dirty), and a
	// build tool must fail with a message rather than block.
	cmd := exec.Command("patch", "-p1", "--batch", "--no-backup-if-mismatch", "-s")
	cmd.Dir = dir
	cmd.Stdin = f
	if out, err := cmd.CombinedOutput(); err != nil {
		return fmt.Errorf("bdwgc: applying %s: %w\n%s", patchFile, err, out)
	}
	return nil
}

// copyGPUSources copies the SPMD-owned pool sources into the patched tree: .c
// files into the root (so librarySources can name them) and headers into
// include/gc (so gc_priv.h's #include "gc_gpu.h" resolves).
func copyGPUSources(gpuDir, dst string) error {
	entries, err := os.ReadDir(gpuDir)
	if err != nil {
		return fmt.Errorf("bdwgc: reading %s: %w", gpuDir, err)
	}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		var target string
		switch filepath.Ext(name) {
		case ".c":
			target = filepath.Join(dst, name)
		case ".h":
			target = filepath.Join(dst, "include", "gc", name)
		default:
			continue
		}
		if err := copyFile(filepath.Join(gpuDir, name), target); err != nil {
			return err
		}
	}
	return nil
}

// mustBdwgcPatchedDir is for the two Library callbacks that cannot report an
// error (sourceDir and cflags both return bare values). A failure here means
// the source tree is broken (missing patch, missing `patch` tool), which is not
// something a build can continue past.
func mustBdwgcPatchedDir() string {
	dir, err := bdwgcPatchedDir()
	if err != nil {
		panic("could not prepare the bdwgc GPU-pool build tree: " + err.Error())
	}
	return dir
}

var BoehmGC = Library{
	name: "bdwgc",
	cflags: func(target, headerPath string) []string {
		libdir := mustBdwgcPatchedDir()
		flags := []string{
			// use a modern environment
			"-DUSE_MMAP",              // mmap is available
			"-DUSE_MUNMAP",            // return memory to the OS using munmap
			"-DGC_BUILTIN_ATOMIC",     // use compiler intrinsics for atomic operations
			"-DNO_EXECUTE_PERMISSION", // don't make the heap executable

			// specific flags for TinyGo
			"-DALL_INTERIOR_POINTERS",  // scan interior pointers (needed for Go)
			"-DIGNORE_DYNAMIC_LOADING", // we don't support dynamic loading at the moment
			"-DNO_GETCONTEXT",          // musl doesn't support getcontext()
			"-DGC_DISABLE_INCREMENTAL", // don't mess with SIGSEGV and such

			// Use a minimal environment.
			"-DNO_MSGBOX_ON_ERROR", // don't call MessageBoxA on Windows
			"-DDONT_USE_ATEXIT",
			"-DNO_GETENV",          // smaller binary, more predictable configuration
			"-DNO_CLOCK",           // don't use system clock
			"-DNO_DEBUGGING",       // reduce code size
			"-DGC_NO_FINALIZATION", // finalization is not used at the moment

			// Special flag to work around the lack of __data_start in ld.lld.
			// TODO: try to fix this in LLVM/lld directly so we don't have to
			// work around it anymore.
			"-DGC_DONT_REGISTER_MAIN_STATIC_DATA",

			// Do not scan the stack. We have our own mechanism to do this.
			"-DSTACK_NOT_SCANNED",
			"-DNO_PROC_STAT",  // we scan the stack manually (don't read /proc/self/stat on Linux)
			"-DSTACKBOTTOM=0", // dummy value, we scan the stack manually

			// Assertions can be enabled while debugging GC issues.
			//"-DGC_ASSERTIONS",

			// We use our own way of dealing with threads (that is a bit hacky).
			// See src/runtime/gc_boehm.go.
			//"-DGC_THREADS",
			//"-DTHREAD_LOCAL_ALLOC",

			// SPMD: build the second (GPU) block pool. It is inert until a
			// chunk provider is registered, which only the Vulkan GPU host
			// does, so this is safe for every target.
			"-DTINYGO_GPU_POOL",

			"-I" + libdir + "/include",
			// gc_gpu.h is installed here by copyGPUSources.
			"-I" + libdir + "/include/gc",
		}
		return flags
	},
	needsLibc: true,
	sourceDir: func() string {
		return mustBdwgcPatchedDir()
	},
	librarySources: func(target string, _ bool) ([]string, error) {
		sources := []string{
			"allchblk.c",
			"alloc.c",
			"blacklst.c",
			"dbg_mlc.c",
			"dyn_load.c",
			"headers.c",
			"mach_dep.c",
			"malloc.c",
			"mark.c",
			"mark_rts.c",
			"misc.c",
			"new_hblk.c",
			"os_dep.c",
			"reclaim.c",

			// SPMD GPU block pool bookkeeping (lib/bdwgc-gpu/gpu_pool.c,
			// copied into the patched tree).
			"gpu_pool.c",
		}
		if strings.Split(target, "-")[2] == "windows" {
			// Due to how the linker on Windows works (that doesn't allow
			// undefined functions), we need to include these extra files.
			sources = append(sources,
				"mallocx.c",
				"ptr_chck.c",
			)
		}
		return sources, nil
	},
}
