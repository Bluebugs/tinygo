// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"
	"testing"
)

// Identifiers referenced only inside an inlined same-package callee must be
// discovered as free variables; they used to be skipped, so the WGSL referred
// to an undeclared name and shader creation failed at runtime.
func TestGPUInlinedCalleeGlobalSliceRead(t *testing.T) {
	src := `
package p

import "lanes"

var g []int32

func addG(x lanes.Varying[int32]) lanes.Varying[int32] {
	return x + lanes.Varying[int32](g[0])
}

func f(dst []int32, src []int32) {
	go for i, v := range src {
		dst[i] = addG(v)
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	fv, ok := freeVarKind(t, plan, "g")
	if !ok || fv.Kind != gpuSliceRead {
		t.Fatalf("g missing or not read-only in Free: %v (kinds %+v)", paramNames(plan.Free), plan.Free)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}
	if !strings.Contains(k.WGSL, "var<storage, read> g: array<i32>;") {
		t.Errorf("g not declared as a read-only buffer:\n%s", k.WGSL)
	}
	if !hasBuffer(k, "g") {
		t.Errorf("g not in kernel buffers: %v", paramNames(k.Buffers))
	}
	runNaga(t, k.WGSL)
}

func TestGPUInlinedCalleeGlobalScalarRead(t *testing.T) {
	src := `
package p

import "lanes"

var bias int32

func addBias(x lanes.Varying[int32]) lanes.Varying[int32] {
	return x + lanes.Varying[int32](bias)
}

func f(dst []int32, src []int32) {
	go for i, v := range src {
		dst[i] = addBias(v)
	}
}
`
	k := transpileKernelOK(t, src)
	found := false
	for _, p := range k.Params {
		if p.Obj.Name() == "bias" && p.Kind == gpuScalar {
			found = true
		}
	}
	if !found {
		t.Fatalf("bias not a Params field: %v", paramNames(k.Params))
	}
	if !strings.Contains(k.WGSL, "params.bias") {
		t.Errorf("WGSL does not read params.bias:\n%s", k.WGSL)
	}
	runNaga(t, k.WGSL)
}

// Writes from an inlined callee are not visible to the caller-side index
// proof, so they must fail closed.
func TestGPUInlinedCalleeWriteRejected(t *testing.T) {
	tests := []struct{ name, src, want string }{
		{
			name: "global-slice-write",
			src: `
package p

import "lanes"

var g []int32

func put(x lanes.Varying[int32]) lanes.Varying[int32] {
	g[0] = 7
	return x
}

func f(dst []int32, src []int32) {
	go for i, v := range src {
		dst[i] = put(v)
	}
}
`,
			want: "write",
		},
		{
			name: "global-slice-incdec",
			src: `
package p

import "lanes"

var g []int32

func put(x lanes.Varying[int32]) lanes.Varying[int32] {
	g[1]++
	return x
}

func f(dst []int32, src []int32) {
	go for i, v := range src {
		dst[i] = put(v)
	}
}
`,
			want: "write",
		},
		{
			name: "global-scalar-write",
			src: `
package p

import "lanes"

var total int32

func put(x lanes.Varying[int32]) lanes.Varying[int32] {
	total = 3
	return x
}

func f(dst []int32, src []int32) {
	go for i, v := range src {
		dst[i] = put(v)
	}
}
`,
			want: "written",
		},
		{
			// dst is written by the body at dst[i]; a callee reading any
			// element of it could observe another invocation's write.
			name: "callee-reads-body-written-slice",
			src: `
package p

import "lanes"

var dst []int32

func peek(x lanes.Varying[int32]) lanes.Varying[int32] {
	return x + lanes.Varying[int32](dst[0])
}

func f(src []int32) {
	go for i, v := range src {
		dst[i] = peek(v)
	}
}
`,
			want: "race",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			plan := parseAndAnalyzeGPULoop(t, tt.src, 1)
			if plan.Reject == "" {
				t.Fatalf("callee write/race accepted; Free=%v", paramNames(plan.Free))
			}
			if !strings.Contains(plan.Reject, tt.want) {
				t.Errorf("reject %q does not mention %q", plan.Reject, tt.want)
			}
		})
	}
}

func TestGPUInlinedCalleeSharedBufferBoundOnce(t *testing.T) {
	src := `
package p

import "lanes"

var g []int32

func addG(x lanes.Varying[int32]) lanes.Varying[int32] {
	return x + lanes.Varying[int32](g[1])
}

func f(dst []int32, src []int32) {
	go for i, v := range src {
		t := addG(v)
		dst[i] = t + lanes.Varying[int32](g[0])
	}
}
`
	k := transpileKernelOK(t, src)
	if n := strings.Count(k.WGSL, "> g: array<i32>;"); n != 1 {
		t.Errorf("g declared %d times, want 1:\n%s", n, k.WGSL)
	}
	count := 0
	for _, b := range k.Buffers {
		if b.Obj.Name() == "g" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("g bound %d times, want 1: %v", count, paramNames(k.Buffers))
	}
	runNaga(t, k.WGSL)
}

// The emitter must never spell an unresolved variable by its raw Go name.
func TestGPUEmitUnresolvedVarFailsClosed(t *testing.T) {
	info := &types.Info{Uses: map[*ast.Ident]types.Object{}, Defs: map[*ast.Ident]types.Object{}}
	id := ast.NewIdent("g")
	info.Uses[id] = types.NewVar(token.NoPos, nil, "g", types.NewSlice(types.Typ[types.Int32]))
	e := &wgslEmitter{
		info:      info,
		freeSubst: map[types.Object]string{},
		locals:    map[types.Object]string{},
		usedBufs:  map[types.Object]bool{},
	}
	if s, err := e.emitExpr(id); err == nil {
		t.Fatalf("unresolved var emitted as %q, want error", s)
	}
}

func hasBuffer(k *gpuKernel, name string) bool {
	for _, b := range k.Buffers {
		if b.Obj.Name() == name {
			return true
		}
	}
	return false
}
