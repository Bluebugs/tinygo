package compiler

import (
	"testing"

	"golang.org/x/tools/go/ssa"
)

// TestGPUOffloadParamsLayoutInvariants locks down the two assumptions
// gpuBuildParams makes when it materialises the uniform buffer: every WGSL
// scalar field is exactly 4 bytes, and gpuKernel.ParamsSize therefore is a
// multiple of 4 that is at least 4*len(Params). gpuBuildParams derives the
// trailing padding purely from ParamsSize, so it never re-implements
// wgslprint.PadTo16.
func TestGPUOffloadParamsLayoutInvariants(t *testing.T) {
	for _, tc := range []struct {
		name string
		src  string
	}{
		{"mandelbrotFlat", mandelbrotFlatGPUSrc},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := parseAndAnalyzeGPULoop(t, tc.src, 1)
			if plan.Reject != "" {
				t.Fatalf("plan rejected: %s", plan.Reject)
			}
			k, err := transpileWGSL(plan, 0)
			if err != nil {
				t.Fatalf("transpileWGSL: %v", err)
			}
			if k.ParamsSize%4 != 0 {
				t.Errorf("ParamsSize %d is not a multiple of 4", k.ParamsSize)
			}
			if min := uint32(len(k.Params)) * 4; k.ParamsSize < min {
				t.Errorf("ParamsSize %d < %d bytes needed for %d params", k.ParamsSize, min, len(k.Params))
			}
			if len(k.Params) == 0 || k.Params[0].Obj.Name() != "n" {
				t.Errorf("Params[0] must be the mandatory trip-count field, got %v", k.Params)
			}
			for _, p := range k.Params {
				switch p.WGSLTy {
				case "i32", "u32", "f32":
				default:
					t.Errorf("param %s has non-4-byte WGSL type %q", p.Obj.Name(), p.WGSLTy)
				}
			}
		})
	}
}

// TestGPUOffloadLoopShapeReject covers the accumulator/reduction fallback in
// gpuCFGReject. No legal Go source reaches it today -- gpu_eligible.go rejects
// written free scalars, reduce.*, break and return before a loop ever gets
// here, and an SSA accumulator is exactly the shape those rules exclude -- so
// this exercises the guard directly instead of through a source fixture.
// Every non-empty result makes gpuAnalyzeLoop fall back to CPU-only emission
// and report "skipped: <reason>" under -gpu-verbose.
func TestGPUOffloadLoopShapeReject(t *testing.T) {
	entry := &ssa.BasicBlock{Index: 0}
	entry.Instrs = []ssa.Instruction{&ssa.Jump{}}
	done := &ssa.BasicBlock{Index: 1}
	trampoline := &ssa.BasicBlock{Index: 2}

	ok := func() *ssa.SPMDLoopInfo {
		return &ssa.SPMDLoopInfo{EntryBlock: entry, DoneBlock: done}
	}

	tests := []struct {
		name string
		info *ssa.SPMDLoopInfo
		want string
	}{
		{"nil (not peeled)", nil, "loop has no SSA-level SPMDLoopInfo (not peeled)"},
		{"no done block", &ssa.SPMDLoopInfo{EntryBlock: entry}, "loop has no SSA entry/done block"},
		{"no entry block", &ssa.SPMDLoopInfo{DoneBlock: done}, "loop has no SSA entry/done block"},
		{"entry == done", &ssa.SPMDLoopInfo{EntryBlock: entry, DoneBlock: entry}, "loop entry and done block are the same block"},
		{"one accumulator", &ssa.SPMDLoopInfo{
			EntryBlock:   entry,
			DoneBlock:    done,
			Accumulators: []ssa.SPMDAccumulator{{}},
		}, "loop has 1 loop-carried accumulator(s)"},
		{"two accumulators", &ssa.SPMDLoopInfo{
			EntryBlock:   entry,
			DoneBlock:    done,
			Accumulators: []ssa.SPMDAccumulator{{}, {}},
		}, "loop has 2 loop-carried accumulator(s)"},
		{"accumulator trampoline", &ssa.SPMDLoopInfo{
			EntryBlock:      entry,
			DoneBlock:       done,
			TrampolineBlock: trampoline,
		}, "loop has an accumulator trampoline block"},
		{"empty entry block", &ssa.SPMDLoopInfo{
			EntryBlock: &ssa.BasicBlock{Index: 3},
			DoneBlock:  done,
		}, "loop entry block is empty"},
		{"eligible shape", ok(), ""},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := gpuLoopShapeReject(tc.info); got != tc.want {
				t.Errorf("gpuLoopShapeReject = %q, want %q", got, tc.want)
			}
		})
	}
}
