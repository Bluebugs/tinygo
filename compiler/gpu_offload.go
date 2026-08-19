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
	"math"
	"os"
	"sort"

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
	bound   ssa.Value   // trip count; Params[0] ("n") and the launch's n argument
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
	if reason := b.gpuCFGReject(loop); reason != "" {
		b.gpuReport(loop, "skipped: %s", reason)
		return
	}
	id := b.gpuKernelCounter
	kernel, err := transpileWGSL(plan, id)
	if err != nil {
		b.gpuReport(loop, "skipped: WGSL transpilation failed: %v", err)
		return
	}
	// Resolve every launch argument to an SSA value available at the guard
	// point *before* committing to the GPU path, so a failure here is a clean
	// CPU-only fallback rather than half-emitted IR.
	args, reason := b.gpuResolveArgs(loop, kernel)
	if reason != "" {
		b.gpuReport(loop, "skipped: %s", reason)
		return
	}
	b.gpuKernelCounter++
	loop.gpu = &gpuLoopOffload{plan: plan, kernel: kernel}
	loop.gpuArgs = args
	b.gpuReport(loop, "offload (kernel=%s, cost=%d, minTrip=%d, params=%d, buffers=%d)",
		kernel.Entry, plan.BodyCost, plan.MinTrip, len(kernel.Params), len(kernel.Buffers))
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
	if b.targetData.PointerSize() != 4 {
		// gpuBufferDesc is a 32-bit ABI shared with the JS glue; see
		// gpuBuildBuffers.  This is also why -gpu=webgpu is meaningless on a
		// 64-bit (i.e. non-wasm32) target.
		return fmt.Sprintf("GPU offload requires a 32-bit pointer target, this one has %d-byte pointers",
			b.targetData.PointerSize())
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
		slice, ok := v.Type().Underlying().(*types.Slice)
		if !ok {
			return nil, fmt.Sprintf("buffer %s is not a slice", bufv.Obj.Name())
		}
		if size := b.targetData.TypeAllocSize(b.getLLVMType(slice.Elem())); size != 4 {
			return nil, fmt.Sprintf("buffer %s has %d-byte elements, WGSL storage buffers require 4", bufv.Obj.Name(), size)
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
	if v == nil {
		return nil, fmt.Sprintf("cannot resolve free variable %s to an SSA value", obj.Name())
	}
	if !gpuDominatesGuard(v, entry) {
		return nil, fmt.Sprintf("free variable %s is not available at the guard point", obj.Name())
	}
	return v, ""
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
	cond := b.CreateAnd(avail, big, "gpu.offload")
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
	b.createRuntimeCall("spmdGPURegister", []llvm.Value{
		llvm.ConstInt(b.ctx.Int32Type(), uint64(k.ID), true),
		b.gpuStringConstant(k.WGSL, "spmd$gpu$wgsl"),
		b.gpuStringConstant(k.Entry, "spmd$gpu$entry"),
	}, "")
	b.CreateStore(llvm.ConstInt(b.ctx.Int1Type(), 1, false), flag)
	b.CreateBr(launchBlock)

	// --- launch ------------------------------------------------------------
	b.SetInsertPointAtEnd(launchBlock)
	paramsPtr, paramsSize := b.gpuBuildParams(loop, bound, pos)
	bufsPtr, bufCount := b.gpuBuildBuffers(loop, pos)
	i32 := b.ctx.Int32Type()
	b.createRuntimeCall("spmdGPULaunch", []llvm.Value{
		llvm.ConstInt(i32, uint64(k.ID), true),
		b.gpuToI32(bound, false),
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
			val = b.gpuCoerce(b.getValue(loop.gpuArgs.scalars[i-1], pos), p.WGSLTy, p.Obj.Name())
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
// gpuBufferDesc is {u32 dataPtr, u32 byteLen, u32 mode} -- see
// src/runtime/gpu_wasm.go, where the JS glue reads it as a flat Uint32Array
// with stride 3.
func (b *builder) gpuBuildBuffers(loop *spmdActiveLoop, pos token.Pos) (llvm.Value, int) {
	k := loop.gpu.kernel
	i32 := b.ctx.Int32Type()
	// gpuBufferDesc is {u32 dataPtr, u32 byteLen, u32 mode}. The 32-bit field
	// width is a wasm ABI requirement shared with the JS glue (Task 7), which
	// reads the descriptor array as a flat stride-3 Uint32Array -- do NOT
	// "fix" dataPtr by widening it to b.uintptrType, that would desynchronise
	// the reader on any 64-bit target. gpuCFGReject instead refuses GPU
	// offload unless pointers are 4 bytes wide.
	descTy := b.ctx.StructType([]llvm.Type{i32, i32, i32}, false)
	n := len(k.Buffers)
	if n == 0 {
		return llvm.ConstNull(b.dataPtrType), 0
	}
	arrTy := llvm.ArrayType(descTy, n)
	alloca := b.gpuEntryAlloca(arrTy, "gpu.buffers")

	for i, bufv := range k.Buffers {
		slice := b.getValue(loop.gpuArgs.buffers[i], pos)
		// Convert at pointer width (the convention everywhere else in this
		// compiler), then narrow to the descriptor's 32-bit field explicitly.
		// gpuCFGReject guarantees uintptrType is already i32 here, so the
		// trunc is a no-op; it is written out so the narrowing is visible.
		dataPtr := b.CreatePtrToInt(b.CreateExtractValue(slice, 0, ""), b.uintptrType, "")
		dataPtr = b.gpuToI32(dataPtr, true)
		// Element size is asserted to be 4 in gpuResolveArgs.
		byteLen := b.CreateShl(b.gpuToI32(b.CreateExtractValue(slice, 1, ""), false),
			llvm.ConstInt(i32, 2, false), "")
		mode := uint64(0)
		if bufv.Kind == gpuSliceRW {
			mode = 1
		}
		base := b.CreateInBoundsGEP(arrTy, alloca, []llvm.Value{
			llvm.ConstInt(i32, 0, false),
			llvm.ConstInt(i32, uint64(i), false),
		}, "")
		b.CreateStore(dataPtr, b.CreateStructGEP(descTy, base, 0, ""))
		b.CreateStore(byteLen, b.CreateStructGEP(descTy, base, 1, ""))
		b.CreateStore(llvm.ConstInt(i32, mode, false), b.CreateStructGEP(descTy, base, 2, ""))
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
