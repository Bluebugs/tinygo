package compiler

// Whole-program marking of the allocation sites whose slices end up bound as
// GPU buffers by an offloaded `go for`. Precision is a performance concern
// only: over-marking puts extra memory in the GPU pool, under-marking copies
// at launch (see the design's "Imprecision is performance-only").
//
// Seeds: every buffer of every `go for` whose plan passes the CHEAP part of
// the offload gate (analyzeGPULoop + MinTrip bounds + transpileWGSL). The
// full gate (gpuCFGReject, gpuResolveArgs, naga) needs an LLVM builder and
// runs per package later; skipping it here only over-approximates.
//
// Flow: a backward walk over go/ssa values, using a call graph for callers,
// callees and closures, plus store-site indexes for globals, struct fields
// and slice/array elements. The design specifies a VTA call graph; VTA panics
// on this repo's SPMD instructions, so gpuZCCallGraph below builds a
// CHA-style graph instead -- see its doc comment for the full reasoning.

import (
	"go/ast"
	"go/token"
	"go/types"
	"strconv"

	"github.com/tinygo-org/tinygo/loader"
	"golang.org/x/tools/go/callgraph"
	"golang.org/x/tools/go/ssa"
)

// gpuZCCallGraph builds a CHA-style (Class Hierarchy Analysis) call graph over
// all: every call site gets an edge to its static callee, or, for dynamic and
// interface calls, to every function whose signature is identical. That is an
// over-approximation, which is the safe direction for this pass -- a missing
// edge would lose a mark and cost a copy at launch, while a spurious edge only
// marks extra memory.
//
// WHY NOT vta.CallGraph, which the design specifies: VTA cannot run on this
// repo's SSA at all. golang.org/x/tools/go/callgraph/vta's builder.instr has an
// exhaustive instruction type switch whose default is
// panic("unsupported instruction %v"), and x-tools-spmd adds SPMD instructions
// (SPMDSelect, SPMDLoad, SPMDStore, SPMDIndex, SPMDExtractMask,
// SPMDVectorFromMemory, SPMDVectorFromPtr) that it does not handle. Any `go for`
// body reaches that panic -- observed as
// `panic: unsupported instruction spmd_load<4> ...`. Teaching VTA the SPMD
// instructions would mean editing x-tools-spmd, which is out of scope here.
//
// go/callgraph/cha is not a drop-in replacement either: it calls
// ssautil.AllFunctions internally, which would pull golang.org/x/tools/go/packages
// (and golang.org/x/mod, golang.org/x/sync) into TinyGo's module graph, and its
// helper golang.org/x/tools/go/callgraph/internal/chautil is internal to x/tools
// and therefore not importable from here.
func gpuZCCallGraph(all map[*ssa.Function]bool) *callgraph.Graph {
	cg := callgraph.New(nil)

	// Index candidate callees for dynamic dispatch. Signatures are compared
	// by canonical string, which is equivalent to types.Identical for this
	// purpose and keeps the lookup O(1) instead of O(functions) per site.
	bySig := map[string][]*ssa.Function{}
	byMethod := map[string][]*ssa.Function{}
	for fn := range all {
		cg.CreateNode(fn) // so every function has a node, even leaf ones
		sig := fn.Signature
		if sig == nil {
			continue
		}
		key := gpuZCSigKey(sig)
		bySig[key] = append(bySig[key], fn)
		if sig.Recv() != nil {
			byMethod[fn.Name()+"|"+key] = append(byMethod[fn.Name()+"|"+key], fn)
		}
	}

	for fn := range all {
		fnode := cg.CreateNode(fn)
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				site, ok := instr.(ssa.CallInstruction)
				if !ok {
					continue
				}
				common := site.Common()
				if _, isBuiltin := common.Value.(*ssa.Builtin); isBuiltin {
					continue
				}
				if g := common.StaticCallee(); g != nil {
					callgraph.AddEdge(fnode, site, cg.CreateNode(g))
					continue
				}
				if common.IsInvoke() {
					msig, ok := common.Method.Type().(*types.Signature)
					if !ok {
						continue
					}
					for _, g := range byMethod[common.Method.Name()+"|"+gpuZCSigKey(msig)] {
						callgraph.AddEdge(fnode, site, cg.CreateNode(g))
					}
					continue
				}
				sig, ok := common.Value.Type().Underlying().(*types.Signature)
				if !ok {
					continue
				}
				for _, g := range bySig[gpuZCSigKey(sig)] {
					callgraph.AddEdge(fnode, site, cg.CreateNode(g))
				}
			}
		}
	}
	return cg
}

// gpuZCSigKey is a canonical key for a signature, ignoring any receiver so
// that a method and a matching func value compare equal.
func gpuZCSigKey(sig *types.Signature) string {
	if sig.Recv() != nil {
		sig = types.NewSignatureType(nil, nil, nil, sig.Params(), sig.Results(), sig.Variadic())
	}
	return types.TypeString(sig, nil)
}

// gpuZCAllFunctions returns the set of functions potentially needed by prog,
// by the same linker-style reachability walk as ssautil.AllFunctions.
//
// It is inlined here rather than imported because the ssautil PACKAGE also
// contains load.go, which imports golang.org/x/tools/go/packages; that would
// pull go/packages (and with it golang.org/x/mod and golang.org/x/sync) into
// TinyGo's module graph, forcing a go.mod change for a dependency this
// compiler otherwise has no use for. vta itself imports nothing outside the
// already-replaced x/tools module.
//
// One deliberate difference: ssautil guards its legacy "exported type hack"
// with isSyntactic, which it reaches through a //go:linkname into an
// unexported go/ssa symbol. That hack is not reproducible from another
// module, so this copy applies the branch unconditionally. That is an
// OVER-approximation (it can only add functions to the set), which is the
// safe direction here: extra functions may cost a little analysis time, but a
// missing function could lose a call edge and hence a mark.
func gpuZCAllFunctions(prog *ssa.Program) map[*ssa.Function]bool {
	seen := make(map[*ssa.Function]bool)

	var function func(fn *ssa.Function)
	function = func(fn *ssa.Function) {
		if fn == nil || seen[fn] {
			return
		}
		seen[fn] = true
		var buf [10]*ssa.Value // avoid alloc in common case
		for _, b := range fn.Blocks {
			for _, instr := range b.Instrs {
				for _, op := range instr.Operands(buf[:0]) {
					if fn, ok := (*op).(*ssa.Function); ok {
						function(fn)
					}
				}
			}
		}
	}

	methodsOf := func(T types.Type) {
		if !types.IsInterface(T) {
			mset := prog.MethodSets.MethodSet(T)
			for method := range mset.Methods() {
				function(prog.MethodValue(method))
			}
		}
	}

	for _, pkg := range prog.AllPackages() {
		for _, mem := range pkg.Members {
			switch mem := mem.(type) {
			case *ssa.Function:
				function(mem)
			case *ssa.Type:
				// See the isSyntactic note above: applied unconditionally.
				if ast.IsExported(mem.Name()) && !types.IsInterface(mem.Type()) {
					if named, ok := mem.Type().(*types.Named); ok && named.TypeParams() == nil {
						methodsOf(named)
						methodsOf(types.NewPointer(named))
					}
				}
			}
		}
	}

	// Methods of types with materialized runtime types are reachable
	// through reflection.
	for _, T := range prog.RuntimeTypes() {
		methodsOf(T)
	}

	return seen
}

// GPUSiteKind says which allocator a marked site lowers to.
type GPUSiteKind string

const (
	GPUSiteMakeSlice GPUSiteKind = "make"   // -> runtime.allocGPU
	GPUSiteAppend    GPUSiteKind = "append" // -> runtime.sliceAppendGPU
)

// GPUZeroCopySiteKey is the stable, fileset-independent identity of a marked
// site: "<filename>:<line>:<col>#<kind>#<elem type>". Both the marking pass and
// the lowering compute it, so they must never diverge.
//
// The ELEMENT TYPE is part of the key, and must stay part of it. A position is
// NOT unique: loader/ssa.go builds with ssa.InstantiateGenerics, so a generic
// `func mk[T any](n int) []T { return make([]T, n) }` produces one MakeSlice
// instruction PER INSTANTIATION, all reporting the same source position. With a
// position-only key, marking the mk[int32] instance would also match the
// mk[*Foo] instance and rewrite it to allocGPU -- putting pointer-bearing
// memory into the pool's atomic, unscanned kind, where the collector never
// traces it: a premature free. gpuZCElemPointerFree cannot prevent that on its
// own, because the guard runs at mark time on one instance while the lowering
// looks up a key shared by both.
func GPUZeroCopySiteKey(fset *token.FileSet, pos token.Pos, kind GPUSiteKind, elem types.Type) string {
	p := fset.Position(pos)
	elemStr := ""
	if elem != nil {
		elemStr = types.TypeString(elem, nil)
	}
	return p.Filename + ":" + strconv.Itoa(p.Line) + ":" + strconv.Itoa(p.Column) + "#" + string(kind) + "#" + elemStr
}

// ExtractSPMDLoopsForPackage exports the existing, verified, unexported
// extractSPMDLoops (compiler/spmd.go:76) so the builder can collect loop
// metadata before the per-package compile loop.
func ExtractSPMDLoopsForPackage(pkg *loader.Package) map[token.Pos]*SPMDLoopInfo {
	return extractSPMDLoops(pkg)
}

// gpuZCElemPointerFree reports whether a slice element type can live in the
// GPU pool: no pointers anywhere inside it (the pool kind is atomic, so the
// collector never scans its objects).
func gpuZCElemPointerFree(t types.Type) bool {
	switch u := t.Underlying().(type) {
	case *types.Basic:
		switch u.Kind() {
		case types.String, types.UnsafePointer, types.UntypedNil, types.Invalid:
			return false
		}
		return true
	case *types.Array:
		return gpuZCElemPointerFree(u.Elem())
	case *types.Struct:
		for i := 0; i < u.NumFields(); i++ {
			if !gpuZCElemPointerFree(u.Field(i).Type()) {
				return false
			}
		}
		return true
	default:
		// Pointer, Slice, Map, Chan, Signature, Interface, TypeParam: all
		// either are or contain pointers.
		return false
	}
}

// gpuZCFieldKey identifies a struct field for the store index.
type gpuZCFieldKey struct {
	Struct types.Type // the *struct type pointed to by FieldAddr.X
	Field  int
}

type gpuZCMarker struct {
	prog *ssa.Program
	cg   *callgraph.Graph
	all  map[*ssa.Function]bool
	out  map[string]map[string]GPUSiteKind

	globalStores map[*ssa.Global][]ssa.Value
	allocStores  map[*ssa.Alloc][]ssa.Value
	fieldStores  map[gpuZCFieldKey][]ssa.Value
	elemStores   map[types.Type][]ssa.Value
	closures     map[*ssa.Function][]*ssa.MakeClosure

	seen map[ssa.Value]bool
}

// MarkGPUZeroCopySites runs the whole-program analysis and returns, per
// package, the set of marked sites in that package.
//
// KEY: the returned outer map is keyed by the TYPES package path
// (ssa.Package.Pkg.Path()), which is what the per-package compile sees, since
// builder/build.go hands CompilePackage program.Package(pkg.Pkg). Note this is
// NOT always loader.Package.ImportPath: for a main package the import path is
// module-relative (e.g. "spmdmarktest") while the types path is "main". A
// consumer selecting this package's sites must therefore index with
// pkg.Pkg.Path(), not pkg.ImportPath.
//
// loops is keyed by import path purely because that is the shape the builder
// can cheaply produce; its keys are never used for output, only iterated.
func MarkGPUZeroCopySites(prog *ssa.Program, loops map[string]map[token.Pos]*SPMDLoopInfo, thresholdOps uint64) map[string]map[string]GPUSiteKind {
	out := map[string]map[string]GPUSiteKind{}
	if prog == nil {
		return out
	}
	all := gpuZCAllFunctions(prog)
	cg := gpuZCCallGraph(all)
	m := &gpuZCMarker{
		prog:         prog,
		cg:           cg,
		all:          all,
		out:          out,
		globalStores: map[*ssa.Global][]ssa.Value{},
		allocStores:  map[*ssa.Alloc][]ssa.Value{},
		fieldStores:  map[gpuZCFieldKey][]ssa.Value{},
		elemStores:   map[types.Type][]ssa.Value{},
		closures:     map[*ssa.Function][]*ssa.MakeClosure{},
		seen:         map[ssa.Value]bool{},
	}
	m.indexStores()
	for _, ls := range loops {
		for _, li := range ls {
			for _, v := range m.seeds(li, thresholdOps) {
				m.trace(v)
			}
		}
	}
	return out
}

// indexStores builds, in one pass over every function, the reverse indexes the
// backward walk needs to get from a load back to the values that were stored.
func (m *gpuZCMarker) indexStores() {
	for fn := range m.all {
		for _, bb := range fn.Blocks {
			for _, instr := range bb.Instrs {
				switch it := instr.(type) {
				case *ssa.Store:
					m.indexStore(it.Addr, it.Val)
				case *ssa.MakeClosure:
					if callee, ok := it.Fn.(*ssa.Function); ok {
						m.closures[callee] = append(m.closures[callee], it)
					}
				}
			}
		}
	}
	// Package-level variable initialisers are Stores in the package's init
	// function, which AllFunctions includes, so globals are already covered.
}

func (m *gpuZCMarker) indexStore(addr, val ssa.Value) {
	switch a := addr.(type) {
	case *ssa.Global:
		m.globalStores[a] = append(m.globalStores[a], val)
	case *ssa.Alloc:
		m.allocStores[a] = append(m.allocStores[a], val)
	case *ssa.FieldAddr:
		m.fieldStores[gpuZCFieldKey{Struct: gpuZCDeref(a.X.Type()), Field: a.Field}] =
			append(m.fieldStores[gpuZCFieldKey{Struct: gpuZCDeref(a.X.Type()), Field: a.Field}], val)
	case *ssa.IndexAddr:
		// Keyed by element type: an over-approximation, which is exactly
		// what this pass is allowed to be. The key must be computed the
		// same way loadSources computes it, hence gpuZCElemOf.
		k := gpuZCElemOf(a)
		m.elemStores[k] = append(m.elemStores[k], val)
	}
}

// gpuZCDeref returns the element type of a pointer type, or t unchanged.
func gpuZCDeref(t types.Type) types.Type {
	if p, ok := t.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return t
}

// seeds returns the SSA values bound as GPU buffers by the given loop, if that
// loop passes the cheap part of the offload gate.
func (m *gpuZCMarker) seeds(li *SPMDLoopInfo, thresholdOps uint64) []ssa.Value {
	plan := analyzeGPULoop(li, thresholdOps)
	if plan.Reject != "" {
		return nil
	}
	if plan.MinTrip > gpuMaxSafeTrip {
		return nil
	}
	kernel, err := transpileWGSL(plan, 0)
	if err != nil {
		return nil
	}

	// The function that syntactically contains the loop; its parameters and
	// DebugRefs resolve the loop's non-global free variables.
	fn := m.innermostFuncAt(li.ForPos)

	var out []ssa.Value
	for _, bufv := range kernel.Buffers {
		obj := bufv.Obj
		if obj == nil {
			continue
		}
		// A package-level variable: trace everything ever stored into it.
		if v, ok := obj.(*types.Var); ok && v.Pkg() != nil && v.Parent() == v.Pkg().Scope() {
			if sp := m.prog.Package(v.Pkg()); sp != nil {
				if g := sp.Var(v.Name()); g != nil {
					out = append(out, m.globalStores[g]...)
					continue
				}
			}
		}
		if fn == nil {
			continue
		}
		for _, p := range fn.Params {
			if p.Object() == obj {
				out = append(out, p)
			}
		}
		for _, bb := range fn.Blocks {
			for _, instr := range bb.Instrs {
				if ref, ok := instr.(*ssa.DebugRef); ok && !ref.IsAddr && ref.Object() == obj {
					out = append(out, ref.X)
				}
			}
		}
	}
	return out
}

// innermostFuncAt returns the function with the smallest syntactic span
// containing pos.
func (m *gpuZCMarker) innermostFuncAt(pos token.Pos) *ssa.Function {
	var best *ssa.Function
	for fn := range m.all {
		syn := fn.Syntax()
		if syn == nil {
			continue
		}
		if syn.Pos() > pos || syn.End() < pos {
			continue
		}
		if best == nil || (syn.Pos() >= best.Syntax().Pos() && syn.End() <= best.Syntax().End()) {
			best = fn
		}
	}
	return best
}

// trace walks backward from v to the allocation sites that can produce it,
// marking each one it reaches.
func (m *gpuZCMarker) trace(root ssa.Value) {
	work := []ssa.Value{root}
	for len(work) > 0 {
		v := work[len(work)-1]
		work = work[:len(work)-1]
		if v == nil || m.seen[v] {
			continue
		}
		m.seen[v] = true

		switch it := v.(type) {
		case *ssa.MakeSlice:
			if slice, ok := it.Type().Underlying().(*types.Slice); ok && gpuZCElemPointerFree(slice.Elem()) {
				m.mark(it.Parent(), it.Pos(), GPUSiteMakeSlice, slice.Elem())
			}
		case *ssa.Slice:
			work = append(work, it.X)
		case *ssa.Phi:
			work = append(work, it.Edges...)
		case *ssa.ChangeType:
			work = append(work, it.X)
		case *ssa.ChangeInterface:
			work = append(work, it.X)
		case *ssa.Parameter:
			work = append(work, m.callerArgs(it)...)
		case *ssa.FreeVar:
			work = append(work, m.closureBindings(it)...)
		case *ssa.Extract:
			if call, ok := it.Tuple.(*ssa.Call); ok {
				work = append(work, m.calleeResults(call, it.Index)...)
			} else {
				work = append(work, it.Tuple)
			}
		case *ssa.Call:
			if b, ok := it.Common().Value.(*ssa.Builtin); ok {
				if b.Name() == "append" {
					if slice, ok := it.Type().Underlying().(*types.Slice); ok && gpuZCElemPointerFree(slice.Elem()) {
						m.mark(it.Parent(), it.Pos(), GPUSiteAppend, slice.Elem())
					}
					if args := it.Common().Args; len(args) > 0 {
						work = append(work, args[0])
					}
				}
				break
			}
			work = append(work, m.calleeResults(it, 0)...)
		case *ssa.UnOp:
			if it.Op == token.MUL {
				work = append(work, m.loadSources(it.X)...)
			}
		}
		// Everything else (Convert from string, TypeAssert, Const, ...)
		// terminates the walk; those shapes are out of scope by design.
	}
}

// loadSources returns the values that may have been stored into the address
// being loaded.
func (m *gpuZCMarker) loadSources(addr ssa.Value) []ssa.Value {
	switch a := addr.(type) {
	case *ssa.Global:
		return m.globalStores[a]
	case *ssa.Alloc:
		return m.allocStores[a]
	case *ssa.FieldAddr:
		return m.fieldStores[gpuZCFieldKey{Struct: gpuZCDeref(a.X.Type()), Field: a.Field}]
	case *ssa.IndexAddr:
		return m.elemStores[gpuZCElemOf(a)]
	}
	return nil
}

// gpuZCElemOf returns the element type an IndexAddr yields.
func gpuZCElemOf(a *ssa.IndexAddr) types.Type {
	return gpuZCDeref(a.Type())
}

// callerArgs returns the actual arguments passed at every call site that the
// CHA-style call graph (gpuZCCallGraph) records as reaching p's function.
func (m *gpuZCMarker) callerArgs(p *ssa.Parameter) []ssa.Value {
	fn := p.Parent()
	if fn == nil {
		return nil
	}
	idx := -1
	for i, q := range fn.Params {
		if q == p {
			idx = i
			break
		}
	}
	if idx < 0 {
		return nil
	}
	node := m.cg.Nodes[fn]
	if node == nil {
		return nil
	}
	var out []ssa.Value
	for _, e := range node.In {
		if e.Site == nil {
			continue
		}
		args := e.Site.Common().Args
		if len(args) > idx {
			out = append(out, args[idx])
		}
	}
	return out
}

// closureBindings returns the values bound to fv by every MakeClosure that
// creates fv's function.
func (m *gpuZCMarker) closureBindings(fv *ssa.FreeVar) []ssa.Value {
	fn := fv.Parent()
	if fn == nil {
		return nil
	}
	idx := -1
	for i, q := range fn.FreeVars {
		if q == fv {
			idx = i
			break
		}
	}
	if idx < 0 {
		return nil
	}
	var out []ssa.Value
	for _, mc := range m.closures[fn] {
		if len(mc.Bindings) > idx {
			out = append(out, mc.Bindings[idx])
		}
	}
	return out
}

// calleeResults returns the idx'th returned value of every callee the
// CHA-style call graph (gpuZCCallGraph) records for the call.
func (m *gpuZCMarker) calleeResults(call ssa.CallInstruction, idx int) []ssa.Value {
	fn := call.Parent()
	if fn == nil {
		return nil
	}
	node := m.cg.Nodes[fn]
	if node == nil {
		return nil
	}
	var out []ssa.Value
	for _, e := range node.Out {
		if e.Site != call || e.Callee == nil || e.Callee.Func == nil {
			continue
		}
		for _, bb := range e.Callee.Func.Blocks {
			for _, instr := range bb.Instrs {
				ret, ok := instr.(*ssa.Return)
				if !ok || len(ret.Results) <= idx {
					continue
				}
				out = append(out, ret.Results[idx])
			}
		}
	}
	return out
}

// mark records a site in the package that syntactically contains it. elem is
// the slice element type, which is part of the key -- see GPUZeroCopySiteKey.
func (m *gpuZCMarker) mark(fn *ssa.Function, pos token.Pos, kind GPUSiteKind, elem types.Type) {
	if fn == nil || fn.Pkg == nil || fn.Pkg.Pkg == nil || !pos.IsValid() {
		return
	}
	path := fn.Pkg.Pkg.Path()
	if m.out[path] == nil {
		m.out[path] = map[string]GPUSiteKind{}
	}
	m.out[path][GPUZeroCopySiteKey(m.prog.Fset, pos, kind, elem)] = kind
}
