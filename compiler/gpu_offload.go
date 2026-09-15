// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

// This file implements dual-path (CPU + GPU) emission for `go for` loops that
// gpu_eligible.go declared offload-eligible and gpu_wgsl.go transpiled to
// WGSL.
//
// The shape emitted for one offloaded loop is:
//
//	<loop EntryBlock, all its original instructions>
//	  %avail = call i1 @runtime.spmdGPUAvailable()
//	  %big   = icmp sge i32 %bound, <MinTrip>
//	  %go    = and i1 %avail, %big
//	  br i1 %go, label %spmd.gpu, label %spmd.cpu
//	spmd.gpu:                       ; register the kernel once, then launch
//	  ...
//	  call void @runtime.spmdGPULaunch(...)
//	  br label %<loop DoneBlock>
//	spmd.cpu:                       ; the untouched original terminator
//	  <original br/switch of EntryBlock>
//
// The CPU side is byte-for-byte the existing lowering: the guard is appended
// to the *end* of EntryBlock (after every one of its instructions, before its
// terminator) and the terminator is then emitted into spmd.cpu.  Because
// TinyGo resolves phi incoming blocks through blockInfo[pred].exit -- which is
// updated to the current insert block after each SSA block finishes -- the
// successors' phis automatically pick up spmd.cpu instead of EntryBlock.  No
// phi operand is rewritten by hand.
//
// The only genuinely new CFG edge is spmd.gpu -> DoneBlock.  gpuCFGReject
// proves that edge cannot break SSA dominance before anything is emitted (see
// its comment), and gpuRepairDonePhis is a belt-and-braces post-pass that
// catches any phi some other part of the compiler put in DoneBlock anyway.

import (
	"fmt"
	"go/token"
	"go/types"
	"hash/fnv"
	"math"
	"os"
	"os/exec"
	"sort"
	"sync"

	"golang.org/x/tools/go/ssa"
	"tinygo.org/x/go-llvm"
)

// gpuLoopOffload is the per-loop GPU state hung off spmdActiveLoop.
type gpuLoopOffload struct {
	plan   *gpuLoopPlan
	kernel *gpuKernel

	// gpuExit is the LLVM block that branches to the loop's DoneBlock on the
	// GPU path.  Recorded during emission so the post-pass phi check knows
	// which predecessor to look for.
	gpuExit llvm.BasicBlock
}

// gpuLaunchArgs holds the SSA values, resolved at analysis time, that feed the
// launch.  Every one of them is proven to be defined in a block that dominates
// the guard point (the end of the loop's EntryBlock), so b.getValue on them is
// safe there.
type gpuLaunchArgs struct {
	bound   ssa.Value   // trip count; Params[0] ("n"), and the launch's n argument after dividing by LanesPerInvocation
	scalars []ssa.Value // one per kernel.Params[1:], same order
	buffers []ssa.Value // one per kernel.Buffers, same order
}

// spmdGPUAnalyze runs the GPU-offload eligibility analysis and WGSL
// transpilation for every SPMD loop in the current function, attaching the
// result to the corresponding spmdActiveLoop.  It is a no-op unless
// -gpu=webgpu was passed.
func (b *builder) spmdGPUAnalyze() {
	if b.gpu != "webgpu" || b.spmdLoopState == nil {
		return
	}

	// Deduplicate: bodyBlocks/loopBlocks map many block indices onto the same
	// *spmdActiveLoop.
	seen := map[*spmdActiveLoop]bool{}
	var loops []*spmdActiveLoop
	for _, m := range []map[int]*spmdActiveLoop{b.spmdLoopState.bodyBlocks, b.spmdLoopState.loopBlocks} {
		for _, loop := range m {
			if !seen[loop] {
				seen[loop] = true
				loops = append(loops, loop)
			}
		}
	}
	// Deterministic order (map iteration is random, and kernel ids must be
	// stable across builds).
	sort.Slice(loops, func(i, j int) bool {
		return loops[i].info.ForPos < loops[j].info.ForPos
	})

	for _, loop := range loops {
		b.gpuAnalyzeLoop(loop)
	}
}

// gpuMaxSafeTrip is the largest `go for` trip count that is safe to
// dispatch to the GPU in a single launch. The JS glue (test/webgpu/spmd_gpu.js)
// dispatches ceil(n / wgslWorkgroupSize) workgroups in a single dispatch
// dimension; the WebGPU spec guarantees maxComputeWorkgroupsPerDimension is
// AT LEAST 65535 on every conformant device (some report more, none
// report less), so 65535*wgslWorkgroupSize is the largest trip count that
// is portably safe -- not just safe on this machine. Above it, the launch
// can silently truncate or fail into an uncaptured WebGPU error scope,
// producing wrong output with no signal to the caller (see PLAN.md "GPU
// dispatch workgroup-count ceiling"); gpuEmitGuard's upper bound and
// gpuAnalyzeLoop's compile-time reject below both exist so that path is
// simply never taken -- a loop whose trip count could exceed this always
// falls back to the ordinary CPU path instead.
const gpuMaxSafeTrip = 65535 * wgslWorkgroupSize

// gpuMaxSafeTripConst returns gpuMaxSafeTrip as an LLVM constant of the
// same integer width as the trip-count value being compared against.
func gpuMaxSafeTripConst(ty llvm.Type) llvm.Value {
	return llvm.ConstInt(ty, uint64(gpuMaxSafeTrip), false)
}

// gpuReport prints a -gpu-verbose line for one loop.
func (b *builder) gpuReport(loop *spmdActiveLoop, format string, args ...interface{}) {
	if !b.GPUVerbose {
		return
	}
	position := b.program.Fset.Position(loop.info.ForPos)
	fmt.Fprintf(os.Stderr, "gpu: %s: %s\n", position, fmt.Sprintf(format, args...))
}

func (b *builder) gpuAnalyzeLoop(loop *spmdActiveLoop) {
	plan := analyzeGPULoop(loop.info, b.GPUThresholdOps)
	if plan.Reject != "" {
		b.gpuReport(loop, "skipped: %s", plan.Reject)
		return
	}
	if plan.MinTrip > math.MaxInt32 {
		b.gpuReport(loop, "skipped: minTrip %d does not fit in an int32 trip-count comparison", plan.MinTrip)
		return
	}
	if plan.MinTrip > gpuMaxSafeTrip {
		// The runtime guard requires minTrip <= n <= gpuMaxSafeTrip; if
		// minTrip alone already exceeds the safe ceiling, that range is
		// empty and the GPU path could never fire at any trip count --
		// skip emitting it at all rather than emit permanently-dead IR.
		b.gpuReport(loop, "skipped: minTrip %d exceeds gpuMaxSafeTrip %d (dispatch-workgroup-count ceiling; loop's per-element cost is too low to ever safely reach the GPU-worthwhile threshold within the dispatchable range)", plan.MinTrip, gpuMaxSafeTrip)
		return
	}
	if reason := b.gpuCFGReject(loop); reason != "" {
		b.gpuReport(loop, "skipped: %s", reason)
		return
	}
	// I3: the id must be unique across the WHOLE BUILD (the JS host keys its
	// kernel map globally), so it is derived from the kernel's own WGSL
	// rather than from a per-package counter. Transpile once with a
	// placeholder id to obtain the body, hash it, then transpile again with
	// the final id so the entry-point name embedded in the module matches.
	probe, err := transpileWGSL(plan, 0)
	if err != nil {
		b.gpuReport(loop, "skipped: WGSL transpilation failed: %v", err)
		return
	}
	id := gpuAllocKernelID(probe.WGSL)
	kernel, err := transpileWGSL(plan, id)
	if err != nil {
		b.gpuReport(loop, "skipped: WGSL transpilation failed: %v", err)
		return
	}
	if reason := gpuHostBufferLimitReject(b.GPUHost, len(kernel.Buffers)); reason != "" {
		b.gpuReport(loop, "skipped: %s", reason)
		return
	}
	if b.GPUHost == "vulkan" {
		if _, err := exec.LookPath(nagaPath()); err != nil {
			// Hard error, not a CPU fallback: silently skipping every loop
			// would make all GPU correctness gates vacuous.
			if !b.gpuNagaMissing {
				b.gpuNagaMissing = true
				b.addError(loop.info.ForPos, "-gpu-host=vulkan requires the naga WGSL compiler (set NAGA or install naga-cli)")
			}
			return
		}
		spv, err := wgslToSPIRV(kernel.WGSL)
		if err != nil {
			b.gpuReport(loop, "skipped: SPIR-V conversion failed: %v", err)
			return
		}
		kernel.SPIRV = spv
	}
	// Resolve every launch argument to an SSA value available at the guard
	// point *before* committing to the GPU path, so a failure here is a clean
	// CPU-only fallback rather than half-emitted IR.
	args, reason := b.gpuResolveArgs(loop, kernel)
	if reason != "" {
		b.gpuReport(loop, "skipped: %s", reason)
		return
	}
	loop.gpu = &gpuLoopOffload{plan: plan, kernel: kernel}
	loop.gpuArgs = args
	b.gpuReport(loop, "offload (kernel=%s, cost=%d, minTrip=%d, maxSafeTrip=%d, params=%d, buffers=%d)",
		kernel.Entry, plan.BodyCost, plan.MinTrip, gpuMaxSafeTrip, len(kernel.Params), len(kernel.Buffers))
}

// gpuVulkanMaxBuffers is the storage-buffer cap of the direct Vulkan host.
// It must match SPMD_VK_MAX_BUFFERS in src/runtime/gpu_vulkan.c: register
// rejects a larger bufCount, which would panic after the guard picked the GPU.
const gpuVulkanMaxBuffers = 8

// gpuHostBufferLimitReject returns a skip reason when a kernel binding n
// storage buffers cannot be registered on host, or "" when it fits.
func gpuHostBufferLimitReject(host string, n int) string {
	if host == "vulkan" && n > gpuVulkanMaxBuffers {
		return fmt.Sprintf("%d buffers exceeds the Vulkan host limit of %d", n, gpuVulkanMaxBuffers)
	}
	return ""
}

// gpuLoopShapeReject holds the parts of the eligibility proof that depend only
// on the loop's SSA shape, split out so they can be unit-tested directly.
// These are the design's accumulator/reduction exclusions: they cannot be
// reached from legal Go source today because gpu_eligible.go rejects every
// construct that would create a loop-carried accumulator (a written free
// scalar, reduce.*, break, return) before the loop ever gets here, so this is
// defense in depth against a future widening of the eligibility rules.
func gpuLoopShapeReject(ssaLoop *ssa.SPMDLoopInfo) string {
	if ssaLoop == nil {
		return "loop has no SSA-level SPMDLoopInfo (not peeled)"
	}
	entry, done := ssaLoop.EntryBlock, ssaLoop.DoneBlock
	if entry == nil || done == nil {
		return "loop has no SSA entry/done block"
	}
	if entry == done {
		return "loop entry and done block are the same block"
	}
	if len(ssaLoop.Accumulators) != 0 {
		return fmt.Sprintf("loop has %d loop-carried accumulator(s)", len(ssaLoop.Accumulators))
	}
	if ssaLoop.TrampolineBlock != nil {
		return "loop has an accumulator trampoline block"
	}
	// The guard is appended after every instruction of entry, so entry must
	// have at least a terminator for that to mean anything.
	if len(entry.Instrs) == 0 {
		return "loop entry block is empty"
	}
	return ""
}

// gpuCFGReject proves that adding the edge guard->DoneBlock cannot invalidate
// SSA dominance for any existing use, and rejects the loop otherwise.
//
// Reasoning.  The guard is appended to the end of EntryBlock, so EntryBlock
// keeps dominating everything it dominated before.  The single new edge is
// EntryBlock -> DoneBlock.  In the resulting CFG the dominator set of
// DoneBlock shrinks to exactly those blocks that also dominate EntryBlock:
//
//	Lost = { B : B != DoneBlock, B dominates DoneBlock,
//	             B does not dominate EntryBlock }
//
// EntryBlock itself is not in Lost (it dominates itself), and neither is
// DoneBlock (it keeps dominating its own instructions).  Lost is precisely the
// set of loop-internal blocks -- e.g. spmd.tail.check -- whose definitions
// could previously be used after the loop without a phi.
//
// A block L in Lost stops dominating exactly those blocks the new edge can
// reach from DoneBlock without going through L again, so for each L we walk
// DoneBlock's successors with L removed and reject if any block in that set
// uses a value defined in L.  Phis in DoneBlock itself are rejected outright,
// because a phi needs one incoming per predecessor and the GPU edge has no
// meaningful value to contribute (this is the accumulator/reduction case the
// design excludes).
func (b *builder) gpuCFGReject(loop *spmdActiveLoop) string {
	if reason := gpuLoopShapeReject(loop.ssaLoopInfo); reason != "" {
		return reason
	}
	ssaLoop := loop.ssaLoopInfo
	entry, done := ssaLoop.EntryBlock, ssaLoop.DoneBlock
	if entry.Parent() != b.fn || done.Parent() != b.fn {
		return "loop entry/done block belongs to another function"
	}
	// gpuBufferDesc has exactly two spellings (see gpuBuildBuffers): the
	// wasm32 {u32,u32,u32} form the JS glue reads as a stride-3
	// Uint32Array, and the native {u64,u32,u32} form gpu_native.go/.c
	// reads.  Which one gpuBuildBuffers emits is decided by POINTER WIDTH,
	// but which one is actually read at runtime is decided by the BUILD
	// TAGS that select gpu_wasm.go vs gpu_native.go -- and those are
	// GOOS/GOARCH predicates, not width predicates.  Those two selectors
	// must be checked together: a hypothetical 64-bit target that still
	// picked up gpu_wasm.go's 12-byte descriptor would be handed a 16-byte
	// one, and the TypeAllocSize assertion in gpuBuildBuffers could not
	// catch it, because that assertion only checks the emitted struct
	// against the pointer width it was derived from.  So the supported
	// (GOOS, GOARCH, pointer width) triples are enumerated explicitly here,
	// matching builder/config.go's front-door allowlist and the build tags
	// on the three src/runtime/gpu_*.go files.
	ps := b.targetData.PointerSize()
	switch {
	case b.GOARCH == "wasm" && ps == 4:
		// gpu_wasm.go (tinygo.wasm && js), 12-byte descriptor.
	case b.GOOS == "linux" && b.GOARCH == "amd64" && ps == 8:
		// gpu_native.go (!tinygo.wasm && linux && amd64), 16-byte descriptor.
	default:
		return fmt.Sprintf("GPU offload has no descriptor layout for %s/%s with %d-byte pointers (supported: wasm/wasm32, linux/amd64)",
			b.GOOS, b.GOARCH, ps)
	}

	for _, instr := range done.Instrs {
		if _, ok := instr.(*ssa.Phi); ok {
			return "loop done block has phi nodes (loop-carried results escape the loop)"
		}
	}

	// Lost = dominators of done that are not dominators of entry.
	var lost []*ssa.BasicBlock
	for _, bb := range b.fn.Blocks {
		if bb != done && bb.Dominates(done) && !bb.Dominates(entry) {
			lost = append(lost, bb)
		}
	}

	// A Lost block L stops dominating exactly those blocks that the new
	// entry->done edge can reach without passing through L again.  Computing
	// that reachability directly is mechanical and exact; it is deliberately
	// NOT approximated by "blocks that done dominates", which is unsafe: for
	// L with edges L->B and L->done->...->B where done does not dominate B,
	// the new path bypasses L and breaks L's values at B, yet B would be
	// filtered out.
	for _, l := range lost {
		reach := map[*ssa.BasicBlock]bool{done: true}
		queue := []*ssa.BasicBlock{done}
		for len(queue) > 0 {
			bb := queue[0]
			queue = queue[1:]
			for _, succ := range bb.Succs {
				if succ == l || reach[succ] {
					continue // L itself still dominates everything beyond it
				}
				reach[succ] = true
				queue = append(queue, succ)
			}
		}
		for bb := range reach {
			for _, instr := range bb.Instrs {
				for _, opnd := range instr.Operands(nil) {
					if opnd == nil || *opnd == nil {
						continue
					}
					def, ok := (*opnd).(ssa.Instruction)
					if !ok || def.Block() != l {
						continue
					}
					return fmt.Sprintf("value %s defined inside the loop (block %d) is reachable-used after the GPU guard bypasses it (block %d)",
						(*opnd).Name(), l.Index, bb.Index)
				}
			}
		}
	}
	return ""
}

// gpuResolveArgs maps every kernel parameter/buffer, which gpu_wgsl.go
// identifies by types.Object, onto an SSA value that is live at the guard
// point.  Returns a non-empty reason string when any of them cannot be
// resolved safely.
func (b *builder) gpuResolveArgs(loop *spmdActiveLoop, k *gpuKernel) (*gpuLaunchArgs, string) {
	entry := loop.ssaLoopInfo.EntryBlock
	bound := loop.ssaLoopInfo.BoundValue
	if bound == nil {
		return nil, "loop has no SSA bound value"
	}
	if !gpuDominatesGuard(bound, entry) {
		return nil, "loop bound is not available at the guard point"
	}

	args := &gpuLaunchArgs{bound: bound}

	if len(k.Params) == 0 {
		return nil, "kernel has no Params (expected the mandatory trip-count field)"
	}
	// Params[0] is the synthetic trip-count field "n" (RULING B in
	// gpu_wgsl.go); it is fed by the bound, not by a Go object.
	for _, p := range k.Params[1:] {
		v, reason := b.gpuValueForObject(p.Obj, loop)
		if reason != "" {
			return nil, reason
		}
		args.scalars = append(args.scalars, v)
	}
	for _, bufv := range k.Buffers {
		v, reason := b.gpuValueForObject(bufv.Obj, loop)
		if reason != "" {
			return nil, reason
		}
		slice, ok := gpuArgType(v).Underlying().(*types.Slice)
		if !ok {
			return nil, fmt.Sprintf("buffer %s is not a slice", bufv.Obj.Name())
		}
		if size := b.targetData.TypeAllocSize(b.getLLVMType(slice.Elem())); size != 1 && size != 4 {
			return nil, fmt.Sprintf("buffer %s has %d-byte elements, GPU buffers support 1 (packed) or 4", bufv.Obj.Name(), size)
		}
		args.buffers = append(args.buffers, v)
	}
	return args, ""
}

// gpuValueForObject finds the SSA value denoted by a source-level object.
//
// The loader builds SSA with ssa.GlobalDebug, so every source reference to a
// local variable is accompanied by an *ssa.DebugRef naming the object.  All
// DebugRefs whose position falls inside the loop body must agree on a single
// value -- if the variable is reassigned inside the loop it is not a uniform
// free variable and the loop is rejected.  Parameters and package-level
// objects that never get a DebugRef are looked up directly.
//
// Resolution is exact or nothing: there is deliberately no "closest
// definition" or "last reference" heuristic here.  An *ssa.DebugRef marks a
// READ of an object, not a definition, so any rule that treats the textually
// last DebugRef as the object's current value is a last-use heuristic in
// disguise and silently mis-resolves e.g. `x := 5; use(x); x = 7`.  A loop
// whose free variables cannot be pinned down unambiguously is rejected and
// falls back to CPU-only emission.
func (b *builder) gpuValueForObject(obj types.Object, loop *spmdActiveLoop) (ssa.Value, string) {
	entry := loop.ssaLoopInfo.EntryBlock
	bodyStart, bodyEnd := loop.info.BodyStart, loop.info.BodyEnd

	var inBody, live []ssa.Value
	add := func(list []ssa.Value, v ssa.Value) []ssa.Value {
		for _, e := range list {
			if e == v {
				return list
			}
		}
		return append(list, v)
	}
	for _, bb := range b.fn.Blocks {
		for _, instr := range bb.Instrs {
			ref, ok := instr.(*ssa.DebugRef)
			if !ok || ref.IsAddr || ref.Object() != obj {
				continue
			}
			if pos := ref.Pos(); pos >= bodyStart && pos <= bodyEnd {
				inBody = add(inBody, ref.X)
			}
			if gpuDominatesGuard(ref.X, entry) {
				live = add(live, ref.X)
			}
		}
	}

	var v ssa.Value
	switch {
	case len(inBody) == 1:
		// The value the loop body actually reads.
		v = inBody[0]
	case len(inBody) > 1:
		return nil, fmt.Sprintf("free variable %s is not uniform across the loop body", obj.Name())
	case len(live) == 1:
		// Not referenced by name inside the body (e.g. it only appears in the
		// range expression): exactly one definition of this object is live at
		// the guard, so that one is unambiguous.
		v = live[0]
	case len(live) > 1:
		return nil, fmt.Sprintf("free variable %s has %d candidate definitions at the guard point", obj.Name(), len(live))
	default:
		v = gpuParamForObject(b.fn, obj)
	}
	if len(inBody) == 0 {
		// A package-level variable referenced only inside an inlined callee
		// has no DebugRef in this function. Load it from its global at the
		// guard: the eligibility gate rejects every write to it inside the
		// kernel and the body calls nothing else, so the guard-time value is
		// the one every iteration reads. This also takes precedence over a
		// pre-loop read in the caller, which a call in between could stale.
		if g := b.gpuGlobalForObject(obj); g != nil {
			v = g
		}
	}
	if v == nil {
		return nil, fmt.Sprintf("cannot resolve free variable %s to an SSA value", obj.Name())
	}
	if !gpuDominatesGuard(v, entry) {
		return nil, fmt.Sprintf("free variable %s is not available at the guard point", obj.Name())
	}
	return v, ""
}

// gpuGlobalForObject returns the *ssa.Global for a package-level variable,
// or nil.
func (b *builder) gpuGlobalForObject(obj types.Object) *ssa.Global {
	v, ok := obj.(*types.Var)
	if !ok || v.Pkg() == nil || v.Parent() != v.Pkg().Scope() {
		return nil
	}
	pkg := b.fn.Prog.Package(v.Pkg())
	if pkg == nil {
		return nil
	}
	return pkg.Var(v.Name())
}

// gpuArgType is the Go type of the launch argument v: for a *ssa.Global
// (see gpuValueForObject) that is the variable's type, not the pointer.
func gpuArgType(v ssa.Value) types.Type {
	if g, ok := v.(*ssa.Global); ok {
		return g.Type().(*types.Pointer).Elem()
	}
	return v.Type()
}

// gpuArgValue lowers a launch argument, loading a *ssa.Global at the current
// insert point (the guard).
func (b *builder) gpuArgValue(v ssa.Value, pos token.Pos) llvm.Value {
	if g, ok := v.(*ssa.Global); ok {
		return b.CreateLoad(b.getLLVMType(gpuArgType(g)), b.getValue(g, pos), "")
	}
	return b.getValue(v, pos)
}

func gpuParamForObject(fn *ssa.Function, obj types.Object) ssa.Value {
	for _, p := range fn.Params {
		if p.Object() == obj {
			return p
		}
	}
	return nil
}

// gpuDominatesGuard reports whether v is defined at the guard point, which
// sits at the very end of entry (after all of entry's instructions).
func gpuDominatesGuard(v ssa.Value, entry *ssa.BasicBlock) bool {
	instr, ok := v.(ssa.Instruction)
	if !ok {
		// Parameter, FreeVar, Const, Global, Function: always available.
		return true
	}
	blk := instr.Block()
	if blk == nil {
		return true
	}
	return blk.Dominates(entry)
}

// spmdGPUMaybeEmitGuard is called from the block-instruction loop just before
// each instruction is lowered.  When instr is the terminator of an offloaded
// loop's EntryBlock, it appends the runtime guard and the GPU path and leaves
// the insert point in the fresh "spmd.cpu" block, into which the caller then
// lowers the original terminator unchanged.
func (b *builder) spmdGPUMaybeEmitGuard(block *ssa.BasicBlock, instr ssa.Instruction) {
	if b.gpu != "webgpu" || b.spmdLoopState == nil {
		return
	}
	// Only fire on the block terminator.
	if len(block.Instrs) == 0 || block.Instrs[len(block.Instrs)-1] != instr {
		return
	}
	seen := map[*spmdActiveLoop]bool{}
	for _, m := range []map[int]*spmdActiveLoop{b.spmdLoopState.bodyBlocks, b.spmdLoopState.loopBlocks} {
		for _, loop := range m {
			if seen[loop] || loop.gpu == nil || loop.ssaLoopInfo.EntryBlock != block {
				continue
			}
			seen[loop] = true
			b.gpuEmitGuard(loop)
		}
	}
}

// gpuRegisterArgs describes the host-specific part of the spmdGPURegister
// call. It is kept free of LLVM so the per-host call shape can be unit-tested.
type gpuRegisterArgs struct {
	shader       string // WGSL source, or SPIR-V bytes for the vulkan host
	shaderName   string // LLVM global name for the shader constant
	withBufCount bool   // vulkan: trailing i32 buffer count argument
	bufCount     int32
}

// gpuRegisterPayload selects the spmdGPURegister arguments for host:
// runtime.spmdGPURegister(id, spirv, entry, bufCount) for "vulkan", and the
// original runtime.spmdGPURegister(id, wgsl, entry) for every other host.
func gpuRegisterPayload(host string, k *gpuKernel) gpuRegisterArgs {
	if host == "vulkan" {
		return gpuRegisterArgs{
			shader:       string(k.SPIRV),
			shaderName:   "spmd$gpu$spirv",
			withBufCount: true,
			bufCount:     int32(len(k.Buffers)),
		}
	}
	return gpuRegisterArgs{shader: k.WGSL, shaderName: "spmd$gpu$wgsl"}
}

func (b *builder) gpuEmitGuard(loop *spmdActiveLoop) {
	off := loop.gpu
	k := off.kernel
	pos := loop.info.ForPos

	gpuBlock := b.ctx.AddBasicBlock(b.llvmFn, "spmd.gpu")
	regBlock := b.ctx.AddBasicBlock(b.llvmFn, "spmd.gpu.register")
	launchBlock := b.ctx.AddBasicBlock(b.llvmFn, "spmd.gpu.launch")
	cpuBlock := b.ctx.AddBasicBlock(b.llvmFn, "spmd.cpu")

	// --- guard -------------------------------------------------------------
	avail := b.createRuntimeCall("spmdGPUAvailable", nil, "gpu.avail")
	if avail.Type() != b.ctx.Int1Type() {
		avail = b.CreateTrunc(avail, b.ctx.Int1Type(), "gpu.avail.i1")
	}
	bound := b.getValue(loop.gpuArgs.bound, pos)
	minTrip := llvm.ConstInt(bound.Type(), off.plan.MinTrip, false)
	big := b.CreateICmp(llvm.IntSGE, bound, minTrip, "gpu.big")
	// gpuMaxSafeTrip caps the trip count from ABOVE, not just below: the JS
	// glue dispatches ceil(n / wgslWorkgroupSize) workgroups in a single
	// dimension, and the WebGPU spec only GUARANTEES
	// maxComputeWorkgroupsPerDimension >= 65535 (some devices report more,
	// none report less) -- so 65535*wgslWorkgroupSize is the largest trip
	// count that is safe to dispatch on every conformant WebGPU device, not
	// just this one. Above it, dispatchWorkgroups can silently truncate or
	// error into an uncaptured error scope, producing wrong output with no
	// signal (see PLAN.md "GPU dispatch workgroup-count ceiling"). A loop
	// whose trip count exceeds this always falls back to the CPU path,
	// which is unconditionally safe.
	small := b.CreateICmp(llvm.IntSLE, bound, gpuMaxSafeTripConst(bound.Type()), "gpu.small")
	cond := b.CreateAnd(avail, big, "gpu.offload.lo")
	cond = b.CreateAnd(cond, small, "gpu.offload")
	b.CreateCondBr(cond, gpuBlock, cpuBlock)

	// --- register-once -----------------------------------------------------
	b.SetInsertPointAtEnd(gpuBlock)
	flagName := fmt.Sprintf("spmd$gpu$registered$%d", k.ID)
	flag := llvm.AddGlobal(b.mod, b.ctx.Int1Type(), flagName)
	flag.SetInitializer(llvm.ConstInt(b.ctx.Int1Type(), 0, false))
	flag.SetLinkage(llvm.InternalLinkage)
	registered := b.CreateLoad(b.ctx.Int1Type(), flag, "gpu.registered")
	b.CreateCondBr(registered, launchBlock, regBlock)

	b.SetInsertPointAtEnd(regBlock)
	payload := gpuRegisterPayload(b.GPUHost, k)
	regArgs := []llvm.Value{
		llvm.ConstInt(b.ctx.Int32Type(), uint64(k.ID), true),
		b.gpuStringConstant(payload.shader, payload.shaderName),
		b.gpuStringConstant(k.Entry, "spmd$gpu$entry"),
	}
	if payload.withBufCount {
		regArgs = append(regArgs, llvm.ConstInt(b.ctx.Int32Type(), uint64(payload.bufCount), true))
	}
	b.createRuntimeCall("spmdGPURegister", regArgs, "")
	b.CreateStore(llvm.ConstInt(b.ctx.Int1Type(), 1, false), flag)
	b.CreateBr(launchBlock)

	// --- launch ------------------------------------------------------------
	b.SetInsertPointAtEnd(launchBlock)
	paramsPtr, paramsSize := b.gpuBuildParams(loop, bound, pos)
	bufsPtr, bufCount := b.gpuBuildBuffers(loop, bound, pos)
	i32 := b.ctx.Int32Type()
	// launchN counts compute INVOCATIONS, not loop iterations: the host
	// dispatches ceil(launchN / wgslWorkgroupSize) workgroups, and each
	// invocation runs k.LanesPerInvocation iterations (Params.n, set from
	// bound above, still bounds the iteration index in the shader).
	launchN := b.gpuToI32(bound, false)
	if k.LanesPerInvocation > 1 {
		if k.LanesPerInvocation != 4 {
			panic(fmt.Sprintf("gpu: kernel %s has unsupported LanesPerInvocation %d", k.Entry, k.LanesPerInvocation))
		}
		launchN = b.CreateLShr(b.CreateAdd(launchN, llvm.ConstInt(i32, 3, false), ""),
			llvm.ConstInt(i32, 2, false), "gpu.launch.n")
	}
	b.createRuntimeCall("spmdGPULaunch", []llvm.Value{
		llvm.ConstInt(i32, uint64(k.ID), true),
		launchN,
		paramsPtr,
		llvm.ConstInt(i32, uint64(paramsSize), false),
		bufsPtr,
		llvm.ConstInt(i32, uint64(bufCount), false),
	}, "")
	b.CreateBr(b.blockInfo[loop.ssaLoopInfo.DoneBlock.Index].entry)
	off.gpuExit = b.GetInsertBlock()

	// --- continue on the CPU path ------------------------------------------
	// The caller lowers the original terminator here.  createFunction's
	// "b.currentBlockInfo.exit = b.GetInsertBlock()" at the end of the block
	// then makes cpuBlock the phi predecessor for EntryBlock's successors.
	b.SetInsertPointAtEnd(cpuBlock)
}

// gpuBuildParams materialises the uniform Params struct on the stack.  The
// field list is taken verbatim from gpuKernel.Params (which IS the layout);
// trailing padding is whatever gpuKernel.ParamsSize says is left over.  The
// resulting LLVM struct size is asserted against ParamsSize.
func (b *builder) gpuBuildParams(loop *spmdActiveLoop, bound llvm.Value, pos token.Pos) (llvm.Value, uint32) {
	k := loop.gpu.kernel
	i32 := b.ctx.Int32Type()
	f32 := b.ctx.FloatType()

	// Every WGSL scalar field is 4 bytes, so ParamsSize/4 is the total field
	// count including padding.  Never re-derive the padding rule here.
	if k.ParamsSize%4 != 0 || uint32(len(k.Params))*4 > k.ParamsSize {
		panic(fmt.Sprintf("gpu: kernel %s has inconsistent ParamsSize %d for %d params",
			k.Entry, k.ParamsSize, len(k.Params)))
	}
	padFields := int(k.ParamsSize/4) - len(k.Params)

	fieldTypes := make([]llvm.Type, 0, len(k.Params)+padFields)
	for _, p := range k.Params {
		if p.WGSLTy == "f32" {
			fieldTypes = append(fieldTypes, f32)
		} else {
			fieldTypes = append(fieldTypes, i32)
		}
	}
	for i := 0; i < padFields; i++ {
		fieldTypes = append(fieldTypes, i32)
	}
	structTy := b.ctx.StructType(fieldTypes, false)
	if got := uint32(b.targetData.TypeAllocSize(structTy)); got != k.ParamsSize {
		panic(fmt.Sprintf("gpu: kernel %s Params struct is %d bytes but WGSL says %d",
			k.Entry, got, k.ParamsSize))
	}

	alloca := b.gpuEntryAlloca(structTy, "gpu.params")
	for i, p := range k.Params {
		var val llvm.Value
		if i == 0 {
			// The mandatory trip-count field.
			val = b.gpuToI32(bound, false)
		} else {
			val = b.gpuCoerce(b.gpuArgValue(loop.gpuArgs.scalars[i-1], pos), p.WGSLTy, p.Obj.Name())
		}
		ptr := b.CreateStructGEP(structTy, alloca, i, "")
		b.CreateStore(val, ptr)
	}
	for i := 0; i < padFields; i++ {
		ptr := b.CreateStructGEP(structTy, alloca, len(k.Params)+i, "")
		b.CreateStore(llvm.ConstInt(i32, 0, false), ptr)
	}
	return alloca, k.ParamsSize
}

// gpuBuildBuffers materialises the [k x gpuBufferDesc] array on the stack.
// gpuBufferDesc is {uptr dataPtr, u32 byteLen, u32 mode} -- see
// src/runtime/gpu_wasm.go (wasm32, where the JS glue reads it as a flat
// stride-3 Uint32Array) and src/runtime/gpu_native.go (native, 8-byte
// dataPtr).  byteLen is the Go byte length of the slice (len * elemSize);
// the host rounds the device buffer up to a multiple of 4 and reads back
// exactly byteLen bytes.  Byte slices travel unmodified, packed four per
// WGSL u32 word (little-endian), so hosts never need the element size.
// The layout is load-bearing on both sides; the
// write-only optimization below therefore encodes itself as a new *value*
// of the existing mode field (2), never as an extra field.
//
// bound is the loop's dynamic trip count (n), needed for the runtime
// `n == len(slice)` test that selects mode 2.
func (b *builder) gpuBuildBuffers(loop *spmdActiveLoop, bound llvm.Value, pos token.Pos) (llvm.Value, int) {
	k := loop.gpu.kernel
	i32 := b.ctx.Int32Type()
	// gpuBufferDesc is pointer-width dependent, and BOTH spellings are a
	// hard ABI contract with a host-side reader:
	//
	//   wasm32 (PointerSize 4): {u32 dataPtr, u32 byteLen, u32 mode}, 12
	//     bytes.  The JS glue (Task 7) reads the descriptor array as a flat
	//     stride-3 Uint32Array.  This layout MUST NOT change -- do not "fix"
	//     dataPtr by widening it to b.uintptrType.
	//
	//   native (PointerSize 8): {u64 dataPtr, u32 byteLen, u32 mode}, 16
	//     bytes, no padding.  Read by src/runtime/gpu_native.go's
	//     gpuBufferDesc and by spmd_gpu_launch in gpu_native.c.
	//
	// The size is asserted below against the layout this compiler believes
	// it emitted, the same way gpuBuildParams asserts ParamsSize: a silent
	// mismatch here is memory corruption, not a wrong answer.
	ptrInt := b.uintptrType
	wantDescSize := uint64(12)
	if b.targetData.PointerSize() == 8 {
		wantDescSize = 16
	}
	descTy := b.ctx.StructType([]llvm.Type{ptrInt, i32, i32}, false)
	if got := b.targetData.TypeAllocSize(descTy); got != wantDescSize {
		panic(fmt.Sprintf("gpu: gpuBufferDesc is %d bytes, expected %d for %d-byte pointers",
			got, wantDescSize, b.targetData.PointerSize()))
	}
	n := len(k.Buffers)
	if n == 0 {
		return llvm.ConstNull(b.dataPtrType), 0
	}
	arrTy := llvm.ArrayType(descTy, n)
	alloca := b.gpuEntryAlloca(arrTy, "gpu.buffers")

	for i, bufv := range k.Buffers {
		slice := b.gpuArgValue(loop.gpuArgs.buffers[i], pos)
		// Convert at pointer width (the convention everywhere else in this
		// compiler), then narrow to the descriptor's 32-bit field explicitly.
		// gpuCFGReject guarantees uintptrType is already i32 here, so the
		// trunc is a no-op; it is written out so the narrowing is visible.
		// dataPtr occupies a full pointer-width field in the descriptor, so
		// no narrowing happens on either target: i32 on wasm32, i64 on
		// native, both exactly uintptrType.
		dataPtr := b.CreatePtrToInt(b.CreateExtractValue(slice, 0, ""), ptrInt, "")
		// byteLen is the Go byte length of the slice. gpuResolveArgs admits
		// 1-byte elements (byte slices, uploaded unmodified and packed four
		// per u32 word by the shader; hosts round the device buffer up to a
		// multiple of 4 and read back exactly byteLen) and 4-byte elements.
		// No i32 overflow: gpuEmitGuard bounds n by gpuMaxSafeTrip, and
		// gpuMaxSafeTrip*4 < 2^32.
		byteLen := b.gpuToI32(b.CreateExtractValue(slice, 1, ""), false)
		if b.targetData.TypeAllocSize(b.getLLVMType(bufv.Obj.Type().Underlying().(*types.Slice).Elem())) == 4 {
			byteLen = b.CreateShl(byteLen, llvm.ConstInt(i32, 2, false), "")
		}
		// mode 0 = read-only (upload, no readback)
		// mode 1 = read-write (upload AND read back)
		// mode 2 = write-only (do NOT upload, but read back)
		//
		// Mode 2 is only sound when EVERY element of the buffer is written
		// by the kernel.  The eligibility analysis
		// (gpu_eligible.go, writeOnlySliceObjs) proves the kernel writes
		// exactly element `idx` for every idx in [0, n) and never reads the
		// slice; what it cannot prove statically is that the slice's length
		// is n rather than something longer.  If len(slice) > n the tail
		// elements are never touched by the kernel, and skipping the upload
		// would read device garbage back over the user's live data.
		//
		// So the mode is chosen at RUNTIME here: n == len(slice) picks 2,
		// anything else falls back to the always-correct 1.  No attempt is
		// made to prove the equality statically.
		var mode llvm.Value
		switch {
		case bufv.Kind != gpuSliceRW:
			mode = llvm.ConstInt(i32, 0, false)
		case !bufv.WriteOnly:
			mode = llvm.ConstInt(i32, 1, false)
		default:
			sliceLen := b.gpuToI32(b.CreateExtractValue(slice, 1, ""), false)
			full := b.CreateICmp(llvm.IntEQ, b.gpuToI32(bound, false), sliceLen, "gpu.buf.full")
			mode = b.CreateSelect(full,
				llvm.ConstInt(i32, 2, false),
				llvm.ConstInt(i32, 1, false), "gpu.buf.mode")
		}
		base := b.CreateInBoundsGEP(arrTy, alloca, []llvm.Value{
			llvm.ConstInt(i32, 0, false),
			llvm.ConstInt(i32, uint64(i), false),
		}, "")
		b.CreateStore(dataPtr, b.CreateStructGEP(descTy, base, 0, ""))
		b.CreateStore(byteLen, b.CreateStructGEP(descTy, base, 1, ""))
		b.CreateStore(mode, b.CreateStructGEP(descTy, base, 2, ""))
	}
	return alloca, n
}

// gpuCoerce converts a Go-typed LLVM scalar to the 4-byte representation the
// WGSL Params struct expects.
func (b *builder) gpuCoerce(v llvm.Value, wgslTy, name string) llvm.Value {
	switch wgslTy {
	case "f32":
		switch v.Type().TypeKind() {
		case llvm.FloatTypeKind:
			return v
		case llvm.DoubleTypeKind:
			return b.CreateFPTrunc(v, b.ctx.FloatType(), "")
		}
		panic("gpu: free variable " + name + " is not a float but WGSL says f32")
	case "i32", "u32":
		if v.Type().TypeKind() != llvm.IntegerTypeKind {
			panic("gpu: free variable " + name + " is not an integer but WGSL says " + wgslTy)
		}
		return b.gpuToI32(v, wgslTy == "u32")
	}
	panic("gpu: unsupported WGSL scalar type " + wgslTy)
}

// gpuToI32 narrows/widens an integer to i32.  unsigned selects zero-extension
// for the widening case.
func (b *builder) gpuToI32(v llvm.Value, unsigned bool) llvm.Value {
	i32 := b.ctx.Int32Type()
	switch width := v.Type().IntTypeWidth(); {
	case width == 32:
		return v
	case width > 32:
		return b.CreateTrunc(v, i32, "")
	case unsigned || width == 1:
		return b.CreateZExt(v, i32, "")
	default:
		return b.CreateSExt(v, i32, "")
	}
}

// gpuEntryAlloca allocates in the function's LLVM entry block, so that an
// offloaded loop nested inside an outer loop does not grow the stack on every
// outer iteration.
func (b *builder) gpuEntryAlloca(t llvm.Type, name string) llvm.Value {
	saved := b.GetInsertBlock()
	entry := b.llvmFn.EntryBasicBlock()
	if first := entry.FirstInstruction(); !first.IsNil() {
		b.SetInsertPointBefore(first)
	} else {
		b.SetInsertPointAtEnd(entry)
	}
	alloca := b.CreateAlloca(t, name)
	b.SetInsertPointAtEnd(saved)
	return alloca
}

// gpuStringConstant builds a Go string value (a {ptr, len} struct) for a
// compile-time constant, mirroring createConst's string handling.
func (b *builder) gpuStringConstant(s, name string) llvm.Value {
	strTy := b.getLLVMType(types.Typ[types.String])
	globalType := llvm.ArrayType(b.ctx.Int8Type(), len(s))
	global := llvm.AddGlobal(b.mod, globalType, name)
	global.SetInitializer(b.ctx.ConstString(s, false))
	global.SetLinkage(llvm.InternalLinkage)
	global.SetGlobalConstant(true)
	global.SetUnnamedAddr(true)
	global.SetAlignment(1)
	v := llvm.Undef(strTy)
	v = b.CreateInsertValue(v, global, 0, "")
	v = b.CreateInsertValue(v, llvm.ConstInt(b.uintptrType, uint64(len(s)), false), 1, "")
	return v
}

// gpuRepairDonePhis is a post-pass safety net run after phi resolution.  The
// GPU edge adds a predecessor to each offloaded loop's DoneBlock; gpuCFGReject
// already proved the SSA has no phi there, but other parts of the compiler
// (e.g. the vector-shadow bookkeeping) can synthesise LLVM phis directly.  Any
// such phi that is missing an incoming for the GPU block is repaired only when
// every existing incoming carries the identical value; otherwise the
// compilation fails loudly rather than emitting invalid IR.
func (b *builder) gpuRepairDonePhis() {
	if b.gpu != "webgpu" || b.spmdLoopState == nil {
		return
	}
	seen := map[*spmdActiveLoop]bool{}
	for _, m := range []map[int]*spmdActiveLoop{b.spmdLoopState.bodyBlocks, b.spmdLoopState.loopBlocks} {
		for _, loop := range m {
			if seen[loop] || loop.gpu == nil || loop.gpu.gpuExit.IsNil() {
				continue
			}
			seen[loop] = true
			done := b.blockInfo[loop.ssaLoopInfo.DoneBlock.Index].entry
			for instr := done.FirstInstruction(); !instr.IsNil(); instr = llvm.NextInstruction(instr) {
				if instr.IsAPHINode().IsNil() {
					break // phis are always at the top of a block
				}
				b.gpuRepairPhi(loop, instr)
			}
		}
	}
}

func (b *builder) gpuRepairPhi(loop *spmdActiveLoop, phi llvm.Value) {
	gpuExit := loop.gpu.gpuExit
	n := phi.IncomingCount()
	// Pass 1: a complete scan for an existing incoming from the GPU edge.  It
	// may sit at any index, so this must finish before value divergence is
	// considered -- otherwise a phi that is already correct could be reported
	// as broken.
	for i := 0; i < n; i++ {
		if phi.IncomingBlock(i) == gpuExit {
			return
		}
	}
	// Pass 2: the GPU edge is missing; it can only be added when every
	// existing incoming carries the identical value.
	var common llvm.Value
	for i := 0; i < n; i++ {
		val := phi.IncomingValue(i)
		if i == 0 {
			common = val
		} else if common != val {
			common = llvm.Value{}
			break
		}
	}
	if common.IsNil() {
		b.addError(loop.info.ForPos, "GPU offload would break a phi node in the loop's merge block")
		return
	}
	phi.AddIncoming([]llvm.Value{common}, []llvm.BasicBlock{gpuExit})
}

// gpuKernelIDs records, for the whole build, which kernel id has been handed
// out for which WGSL body. Compilation of different packages can run
// concurrently, so it is mutex-guarded.
var (
	gpuKernelIDMu sync.Mutex
	gpuKernelIDs  = map[int32]string{}
)

// gpuAllocKernelID returns a build-unique, rebuild-stable id for a kernel
// whose (placeholder-id) WGSL body is wgsl.
//
// I3: a per-compilerContext counter numbered kernels per PACKAGE while the
// JS host (test/webgpu/spmd_gpu.js) keys its kernel map GLOBALLY, so two
// offloadable loops in two packages both got id 0 and the second
// `spmd_gpu.register` silently overwrote the first -- reachable today in a
// single build. Hashing the WGSL instead makes the id a property of the
// kernel itself: unique across packages, identical across rebuilds (unlike
// a global counter, whose value would depend on package compilation order).
// Identical bodies deliberately share an id: they are the same shader.
func gpuAllocKernelID(wgsl string) int32 {
	h := fnv.New32a()
	h.Write([]byte(wgsl))
	// Keep it non-negative: the id is passed through an LLVM i32 and printed
	// into symbol names.
	id := int32(h.Sum32() & 0x7fffffff)

	gpuKernelIDMu.Lock()
	defer gpuKernelIDMu.Unlock()
	for {
		prev, taken := gpuKernelIDs[id]
		if !taken {
			gpuKernelIDs[id] = wgsl
			return id
		}
		if prev == wgsl {
			return id // same shader, same id
		}
		// Genuine hash collision between two different kernels: probe.
		id = (id + 1) & 0x7fffffff
	}
}
