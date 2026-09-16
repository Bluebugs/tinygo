package compiler

import (
	"go/token"
	"go/types"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"

	"github.com/tinygo-org/tinygo/compileopts"
	"github.com/tinygo-org/tinygo/loader"
	"golang.org/x/tools/go/ssa"
)

// testBuildConfig returns a *compileopts.Config for the given options,
// resolving the target the same way testCompilePackage (compiler_test.go:217)
// already does.
func testBuildConfig(t *testing.T, opts *compileopts.Options) *compileopts.Config {
	t.Helper()
	target, err := compileopts.LoadTarget(opts)
	if err != nil {
		t.Fatal("failed to load target:", err)
	}
	return &compileopts.Config{Options: opts, Target: target}
}

// markSites writes src to a temp package, loads it through TinyGo's own
// loader (the ONLY path that resolves `import "lanes"`, which needs the SPMD
// fork's GOROOT and GOEXPERIMENT=spmd -- go/importer.Default() cannot), builds
// the whole-program SSA the way builder/build.go does, runs the marking pass,
// and returns that package's marked sites.
//
// NOTE on GOEXPERIMENT: the forked go/parser gates `go for` on
// buildcfg.Experiment.SPMD, which internal/buildcfg initialises from the
// environment once, at package-init time -- so a t.Setenv here would run far
// too late to affect it. No setenv is needed regardless: the fork's
// zbootstrap.go sets defaultGOEXPERIMENT = `spmd`, so SPMD parsing is on by
// default (verified empirically with and without GOEXPERIMENT set).
// Options.GOExperiment below is still required, because compileopts reads that
// field rather than the environment.
func markSites(t *testing.T, src string) map[string]GPUSiteKind {
	t.Helper()

	// The temp package must be its own main module: `go list` (which
	// loader.Load shells out to) rejects a bare directory with "outside main
	// module or its selected dependencies". A go.mod here, plus
	// Options.Directory so `go list` runs inside it, makes the directory
	// self-contained. `lanes` still resolves, because it lives in the cached
	// SPMD GOROOT rather than in the module graph.
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(src), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module spmdmarktest\n\ngo 1.25\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	// Same target/options as the other GPU compiler tests, plus the Vulkan
	// host so Config.GPUZeroCopy() is true.
	config := testBuildConfig(t, &compileopts.Options{
		GOOS: "linux", GOARCH: "amd64", Opt: "2",
		GOExperiment:    "spmd",
		GPU:             "webgpu",
		GPUHost:         "vulkan",
		GPUThresholdOps: 1,
		Directory:       dir,
	})
	lprogram, err := loader.Load(config, ".", types.Config{
		Sizes: types.SizesFor("gc", "amd64"),
	})
	if err != nil {
		t.Fatalf("loader.Load: %v", err)
	}
	if err := lprogram.Parse(); err != nil {
		t.Fatalf("parse: %v", err)
	}
	prog := lprogram.LoadSSA()
	prog.Build()

	loops := map[string]map[token.Pos]*SPMDLoopInfo{}
	for _, pkg := range lprogram.Sorted() {
		loops[pkg.ImportPath] = ExtractSPMDLoopsForPackage(pkg)
	}
	marks := MarkGPUZeroCopySites(prog, loops, 1)

	// The package under test is the main package of the temp dir. Look it up
	// by its TYPES package path, which is what MarkGPUZeroCopySites keys its
	// result by, and which is also what the per-package compile sees
	// (CompilePackage receives program.Package(pkg.Pkg)). For a main package
	// these two spellings differ: loader's ImportPath is the module-relative
	// path ("spmdmarktest" here) while the types path is "main".
	out := map[string]GPUSiteKind{}
	for key, kind := range marks[lprogram.MainPkg().Pkg.Path()] {
		out[key] = kind
	}
	return out
}

// compileTestIR compiles src (a temp main package) to LLVM IR, with every
// MakeSlice and append site in the package either marked or not marked for
// GPU-pool allocation, and returns the module's IR text. The site keys are
// computed from the SSA rather than hardcoded, so they cannot drift from what
// the lowering looks up.
func compileTestIR(t *testing.T, src string, marked bool) string {
	return compileTestIRPred(t, src, marked, false)
}

// compileTestIRMarkAll marks EVERY site, including pointer-containing element
// types that the real pass would never mark. It exists to test the lowering's
// own belt-and-braces pointer-free re-check in isolation.
func compileTestIRMarkAll(t *testing.T, src string) string {
	return compileTestIRPred(t, src, true, true)
}

func compileTestIRPred(t *testing.T, src string, marked, markAll bool) string {
	t.Helper()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(src), 0o644); err != nil {
		t.Fatalf("write: %v", err)
	}
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module spmdmarktest\n\ngo 1.25\n"), 0o644); err != nil {
		t.Fatalf("write go.mod: %v", err)
	}
	opts := &compileopts.Options{
		GOOS: "linux", GOARCH: "amd64", Opt: "2",
		GOExperiment: "spmd",
		Directory:    dir,
	}
	config := testBuildConfig(t, opts)

	compilerConfig := &Config{
		Triple:          config.Triple(),
		Features:        config.Features(),
		ABI:             config.ABI(),
		GOOS:            config.GOOS(),
		GOARCH:          config.GOARCH(),
		CodeModel:       config.CodeModel(),
		RelocationModel: config.RelocationModel(),
		Scheduler:       config.Scheduler(),
		MaxStackAlloc:   config.MaxStackAlloc(),
	}
	machine, err := NewTargetMachine(compilerConfig)
	if err != nil {
		t.Fatal("failed to create target machine:", err)
	}
	defer machine.Dispose()

	lprogram, err := loader.Load(config, ".", types.Config{
		Sizes: Sizes(machine),
	})
	if err != nil {
		t.Fatalf("loader.Load: %v", err)
	}
	if err := lprogram.Parse(); err != nil {
		t.Fatalf("parse: %v", err)
	}
	prog := lprogram.LoadSSA()
	prog.Build()

	pkg := lprogram.MainPkg()
	ssaPkg := prog.Package(pkg.Pkg)

	if marked {
		sites := map[string]GPUSiteKind{}
		// Walk every reachable function, not just ssaPkg.Members: generic
		// INSTANCES (mk[int32], mk[*Foo]) are created by the builder, do not
		// appear as package members, and do not necessarily carry the
		// package in fn.Pkg -- yet they are exactly what the generic-instance
		// tests are about. No package filter is applied: site keys embed an
		// absolute filename, so keys belonging to other packages simply never
		// match anything the package under test emits.
		var seenSites []string
		for fn := range gpuZCAllFunctions(prog) {
			for _, b := range fn.Blocks {
				for _, instr := range b.Instrs {
					switch it := instr.(type) {
					case *ssa.MakeSlice:
						if slice, ok := it.Type().Underlying().(*types.Slice); ok {
							if strings.HasPrefix(prog.Fset.Position(it.Pos()).Filename, dir) {
								seenSites = append(seenSites, fn.String()+" elem="+types.TypeString(slice.Elem(), nil))
							}
							if markAll || gpuZCElemPointerFree(slice.Elem()) {
								sites[GPUZeroCopySiteKey(prog.Fset, it.Pos(), GPUSiteMakeSlice, slice.Elem())] = GPUSiteMakeSlice
							}
						}
					case *ssa.Call:
						if bi, ok := it.Common().Value.(*ssa.Builtin); ok && bi.Name() == "append" {
							if slice, ok := it.Type().Underlying().(*types.Slice); ok {
								if markAll || gpuZCElemPointerFree(slice.Elem()) {
									sites[GPUZeroCopySiteKey(prog.Fset, it.Pos(), GPUSiteAppend, slice.Elem())] = GPUSiteAppend
								}
							}
						}
					}
				}
			}
		}
		if len(sites) == 0 {
			t.Fatalf("no markable MakeSlice or append site found in the test package; "+
				"the source must contain a variable-length make or an append with a "+
				"pointer-free element type. MakeSlice sites seen in this package: %v", seenSites)
		}
		compilerConfig.GPUZeroCopySites = sites
	}

	mod, errs := CompilePackage("main.go", pkg, ssaPkg, machine, compilerConfig, false)
	if errs != nil {
		for _, err := range errs {
			t.Error(err)
		}
		t.FailNow()
	}
	defer mod.Context().Dispose()
	defer mod.Dispose()
	return mod.String()
}

// siteLine parses the line number out of a "<file>:<line>:<col>#<kind>" key.
// The filename may itself contain colons, so it splits from the right.
func siteLine(t *testing.T, key string) int {
	t.Helper()
	k := key
	if i := strings.LastIndex(k, "#"); i >= 0 {
		k = k[:i]
	}
	col := strings.LastIndex(k, ":")
	if col < 0 {
		t.Fatalf("malformed site key %q", key)
	}
	k = k[:col]
	ln := strings.LastIndex(k, ":")
	if ln < 0 {
		t.Fatalf("malformed site key %q", key)
	}
	n, err := strconv.Atoi(k[ln+1:])
	if err != nil {
		t.Fatalf("malformed line in site key %q: %v", key, err)
	}
	return n
}

func hasMark(t *testing.T, marks map[string]GPUSiteKind, line int, kind GPUSiteKind) bool {
	t.Helper()
	for key, k := range marks {
		if k == kind && siteLine(t, key) == line {
			return true
		}
	}
	return false
}

func TestMarkSameFunctionMake(t *testing.T) {
	marks := markSites(t, `
package main

func f(n int) []int32 {
	dst := make([]int32, n)
	src := make([]int32, n)
	go for i := range src {
		dst[i] = src[i] * 3
	}
	return dst
}

func main() { _ = f(1024) }
`)
	if !hasMark(t, marks, 5, GPUSiteMakeSlice) || !hasMark(t, marks, 6, GPUSiteMakeSlice) {
		t.Fatalf("both makes should be marked, got %v", marks)
	}
}

func TestMarkOneCallAway(t *testing.T) {
	marks := markSites(t, `
package main

func kernel(dst, src []int32) {
	go for i := range src {
		dst[i] = src[i] * 3
	}
}

func run(n int) []int32 {
	dst := make([]int32, n)
	src := make([]int32, n)
	kernel(dst, src)
	return dst
}

func main() { _ = run(1024) }
`)
	if !hasMark(t, marks, 11, GPUSiteMakeSlice) || !hasMark(t, marks, 12, GPUSiteMakeSlice) {
		t.Fatalf("caller's makes should be marked, got %v", marks)
	}
}

func TestMarkMethodTableClosure(t *testing.T) {
	marks := markSites(t, `
package main

type method struct {
	run func(dst, src []int32)
}

func kernel(dst, src []int32) {
	go for i := range src {
		dst[i] = src[i] * 3
	}
}

func run(n int) []int32 {
	methods := []method{{run: kernel}}
	dst := make([]int32, n)
	src := make([]int32, n)
	for _, m := range methods {
		m.run(dst, src)
	}
	return dst
}

func main() { _ = run(1024) }
`)
	if !hasMark(t, marks, 16, GPUSiteMakeSlice) || !hasMark(t, marks, 17, GPUSiteMakeSlice) {
		t.Fatalf("closure-dispatched buffers should be marked, got %v", marks)
	}
}

func TestMarkGlobalGrownByAppend(t *testing.T) {
	marks := markSites(t, `
package main

var summary []int32

func kernel(sum, src []int32) {
	go for i := range src {
		sum[i] = src[i] * 3
	}
}

func run(n int, src []int32) {
	for len(summary) < n {
		summary = append(summary, 0)
	}
	kernel(summary[:n], src)
}

func main() { run(4, make([]int32, 4)) }
`)
	// Line 14, not the brief's 13: counting from the leading newline of the
	// raw string, `summary = append(...)` is the 14th line. The brief's
	// inline comment is off by one; the marking itself is what it expects.
	if !hasMark(t, marks, 14, GPUSiteAppend) {
		t.Fatalf("append growing a global GPU buffer should be marked, got %v", marks)
	}
}

func TestMarkInlinedGlobal(t *testing.T) {
	// NOTE: the makes here use a variable length on purpose. A CONSTANT
	// length (the brief's `make([]int32, 3)`) does not produce an
	// ssa.MakeSlice at all -- go/ssa emits `Alloc [3]int32` + `Slice`, a
	// shape this pass deliberately does not mark. See the report's "missed
	// shapes" and the PLAN.md deferred item. Using a variable length keeps
	// this test on its actual subject: a global reached through an inlined
	// callee.
	marks := markSites(t, `
package main

import (
	"lanes"
	"os"
)

var table []int32

func addG(x lanes.Varying[int32]) lanes.Varying[int32] {
	return x + lanes.Varying[int32](table[0])
}

func run(dst, src []int32) {
	go for i, v := range src {
		dst[i] = addG(v)
	}
}

func setup(n int) { table = make([]int32, n) }

func main() {
	n := len(os.Args)
	setup(n)
	run(make([]int32, n), make([]int32, n))
}
`)
	if !hasMark(t, marks, 21, GPUSiteMakeSlice) {
		t.Fatalf("global read by an inlined callee should be marked, got %v", marks)
	}
}

func TestLoweringUsesGPUAllocators(t *testing.T) {
	// A marked make lowers to runtime.allocGPU; an unmarked one keeps
	// runtime.alloc.
	//
	// Only the PRESENCE of allocGPU is asserted for the marked case, not the
	// absence of plain alloc: the module contains other allocations besides
	// the make under test (package initialisation, os.Args), so requiring
	// `@runtime.alloc(` to disappear entirely would be wrong. The unmarked
	// case is what proves the lowering is actually driven by the mark rather
	// than emitting allocGPU unconditionally.
	const src = `
package main

import "os"

func main() {
	dst := make([]int32, len(os.Args))
	_ = dst
}
`
	for _, marked := range []bool{true, false} {
		ir := compileTestIR(t, src, marked)
		gotGPU := strings.Contains(ir, "@runtime.allocGPU")
		gotPlain := strings.Contains(ir, "@runtime.alloc(")
		if gotGPU != marked {
			t.Errorf("marked=%v: allocGPU present = %v, want %v", marked, gotGPU, marked)
			for _, line := range strings.Split(ir, "\n") {
				if strings.Contains(line, "runtime.alloc") {
					t.Logf("    %s", strings.TrimSpace(line))
				}
			}
		}
		if !marked && !gotPlain {
			t.Errorf("marked=false: expected a plain @runtime.alloc( call, found none")
		}
	}
}

// TestLoweringGenericInstancesDoNotCollide is the regression test for the
// position-only site key. `mk[int32]` and `mk[*Foo]` are two DIFFERENT SSA
// functions whose MakeSlice instructions report the SAME source position, so a
// key built from position+kind alone cannot tell them apart. Marking the
// pointer-free instance would then rewrite the pointer-containing instance to
// allocGPU as well, putting pointer-bearing memory into an atomic, unscanned
// pool -- a premature free.
//
// Only the int32 instance is marked here (compileTestIR marks pointer-free
// element types only, exactly as the real pass does), so exactly ONE
// allocGPU call must appear. With a position-only key there are two.
func TestLoweringGenericInstancesDoNotCollide(t *testing.T) {
	const src = `
package main

import "os"

type Foo struct{ p *int }

func mk[T any](n int) []T { return make([]T, n) }

func main() {
	n := len(os.Args)
	a := mk[int32](n)
	b := mk[*Foo](n)
	_ = a
	_ = b
}
`
	ir := compileTestIR(t, src, true)

	// Count allocGPU CALL SITES, skipping the module's `declare` line for the
	// symbol -- a naive occurrence count includes that declaration and so can
	// never reach the expected value.
	calls := 0
	for _, line := range strings.Split(ir, "\n") {
		if strings.Contains(line, "@runtime.allocGPU") &&
			!strings.HasPrefix(strings.TrimSpace(line), "declare") {
			calls++
		}
	}

	// Corroborate per function: of the two generic instances TinyGo emits
	// (as linkonce_odr functions), exactly one may allocate from the pool.
	// Instance NAMES are deliberately not matched -- the emitted headers do
	// not spell the type arguments in a form worth depending on -- so the
	// evidence is "how many function bodies contain an allocGPU call".
	gpuFuncs := 0
	var headers []string
	for _, fn := range strings.Split(ir, "\ndefine ")[1:] {
		body := fn
		if i := strings.Index(fn, "\n}"); i >= 0 {
			body = fn[:i]
		}
		header := strings.TrimSpace(fn)
		if i := strings.Index(header, "\n"); i >= 0 {
			header = header[:i]
		}
		has := strings.Contains(body, "@runtime.allocGPU")
		if has {
			gpuFuncs++
		}
		headers = append(headers, header+"  allocGPU="+strconv.FormatBool(has))
	}

	if calls != 1 || gpuFuncs != 1 {
		t.Errorf("allocGPU call sites = %d (want 1), function bodies allocating from the pool = %d "+
			"(want 1). The mk[*Foo] instance must keep runtime.alloc: pointer-bearing memory in the "+
			"atomic, unscanned GPU pool is a premature free.\ndefine headers:\n%s",
			calls, gpuFuncs, strings.Join(headers, "\n"))
	}
}

func TestLoweringUsesGPUAppend(t *testing.T) {
	// The append half of the lowering: a marked append must call
	// runtime.sliceAppendGPU instead of runtime.sliceAppend.
	const src = `
package main

import "os"

var summary []int32

func main() {
	for i := 0; i < len(os.Args); i++ {
		summary = append(summary, int32(i))
	}
}
`
	for _, marked := range []bool{true, false} {
		ir := compileTestIR(t, src, marked)
		gotGPU := strings.Contains(ir, "@runtime.sliceAppendGPU")
		gotPlain := strings.Contains(ir, "@runtime.sliceAppend(")
		if gotGPU != marked {
			t.Errorf("marked=%v: sliceAppendGPU present = %v, want %v", marked, gotGPU, marked)
		}
		if gotPlain == marked {
			t.Errorf("marked=%v: plain sliceAppend present = %v, want %v", marked, gotPlain, !marked)
		}
	}
}

func TestMarkTwoCallPaths(t *testing.T) {
	// The kernel is reached through a package-level FUNC VALUE, so
	// CallCommon.StaticCallee() is nil at both call sites and the walk can
	// only get from kernel's parameters back to these makes via a dynamic
	// call edge. Calling kernel directly would make this test pass on static
	// edges alone and prove nothing about the call graph.
	marks := markSites(t, `
package main

import "os"

func kernel(dst, src []int32) {
	go for i := range src {
		dst[i] = src[i] * 3
	}
}

var dispatch = kernel

func mainPath(n int)  { dispatch(make([]int32, n), make([]int32, n)) }
func benchPath(n int) { dispatch(make([]int32, n), make([]int32, n)) }

func main() { n := len(os.Args); mainPath(n); benchPath(n) }
`)
	if !hasMark(t, marks, 14, GPUSiteMakeSlice) || !hasMark(t, marks, 15, GPUSiteMakeSlice) {
		t.Fatalf("both call paths should be marked through the func value, got %v", marks)
	}
}

// siteElem returns the element-type component of a site key
// ("<file>:<line>:<col>#<kind>#<elem>").
func siteElem(t *testing.T, key string) string {
	t.Helper()
	i := strings.LastIndex(key, "#")
	if i < 0 {
		t.Fatalf("malformed site key %q", key)
	}
	return key[i+1:]
}

// TestMarkNeverMarksPointerElemsAtPassLevel is the pass-level counterpart to
// TestMarkPointerElemNeverMarked, which only exercises the helper directly and
// so could not have caught a key that conflates two element types. Here a
// pointer-containing slice is live right next to the GPU buffers; no marked
// site may carry a pointer-bearing element type.
func TestMarkNeverMarksPointerElemsAtPassLevel(t *testing.T) {
	marks := markSites(t, `
package main

import "os"

type Foo struct{ p *int }

func kernel(dst, src []int32) {
	go for i := range src {
		dst[i] = src[i] * 3
	}
}

func main() {
	n := len(os.Args)
	boxes := make([]*Foo, n)
	names := make([]string, n)
	nested := make([][]int32, n)
	_, _, _ = boxes, names, nested
	kernel(make([]int32, n), make([]int32, n))
}
`)
	if len(marks) == 0 {
		t.Fatal("expected at least the kernel's own buffers to be marked")
	}
	for key := range marks {
		elem := siteElem(t, key)
		switch {
		case strings.HasPrefix(elem, "*"),
			strings.HasPrefix(elem, "[]"),
			strings.HasPrefix(elem, "map["),
			strings.HasPrefix(elem, "func"),
			strings.Contains(elem, "interface"),
			elem == "string":
			t.Errorf("marked site %q has pointer-bearing element type %q; such memory "+
				"must never reach the atomic, unscanned GPU pool", key, elem)
		}
	}
}

// TestLoweringRefusesPointerElems proves the lowering's own guard: even when
// the site map explicitly marks a pointer-containing element type (which the
// real pass would never do), the lowering must still emit plain runtime.alloc.
func TestLoweringRefusesPointerElems(t *testing.T) {
	ir := compileTestIRMarkAll(t, `
package main

import "os"

type Foo struct{ p *int }

func main() {
	boxes := make([]*Foo, len(os.Args))
	_ = boxes
}
`)
	if strings.Contains(ir, "@runtime.allocGPU") {
		t.Error("a pointer-containing element type was lowered to allocGPU despite the " +
			"lowering's pointer-free re-check")
	}
}

func TestMarkPointerElemNeverMarked(t *testing.T) {
	if gpuZCElemPointerFree(types.NewPointer(types.Typ[types.Int32])) {
		t.Error("*int32 elements must not be pointer-free")
	}
	if gpuZCElemPointerFree(types.NewSlice(types.Typ[types.Int32])) {
		t.Error("[]int32 elements (slice headers) must not be pointer-free")
	}
	if !gpuZCElemPointerFree(types.Typ[types.Int32]) {
		t.Error("int32 elements must be pointer-free")
	}
	if !gpuZCElemPointerFree(types.Typ[types.Byte]) {
		t.Error("byte elements must be pointer-free")
	}
}
