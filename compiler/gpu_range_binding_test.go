// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"fmt"
	"strings"
	"testing"
)

func transpileKernelOK(t *testing.T, src string) *gpuKernel {
	t.Helper()
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}
	return k
}

// The range value of `go for i, v := range s` must be declared in the entry
// point with the same read the explicit `s[i]` form lowers to; it used to be
// referenced without any declaration, failing shader creation at runtime.
func TestGPURangeValueDeclared(t *testing.T) {
	tests := []struct {
		name, src, decl string
	}{
		{
			name: "byte",
			src: `
package p

func f(text, head, cand []byte) {
	go for i, a := range head {
		cand[i] = a&15 ^ text[i+1]
	}
}
`,
			decl: "var a: u32 = " + packedByteRead("head", "i") + ";",
		},
		{
			name: "int32",
			src: `
package p

func f(dst, src []int32) {
	go for i, v := range src {
		v += 3
		dst[i] = v * 2
	}
}
`,
			decl: "var v: i32 = src[i];",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			k := transpileKernelOK(t, tt.src)
			if !strings.Contains(k.WGSL, tt.decl) {
				t.Errorf("WGSL missing range value declaration %q:\n%s", tt.decl, k.WGSL)
			}
			runNaga(t, k.WGSL)
		})
	}
}

// A range value over something other than a free slice has no WGSL read to
// lower to; it must stay on the CPU.
func TestGPURangeValueNonSliceRejected(t *testing.T) {
	src := `
package p

func f(dst []int32, src [8]int32) {
	go for i, v := range src {
		dst[i] = v
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject == "" {
		t.Fatalf("range value over an array must be rejected")
	}
}

// A slice used only as the range bound is never read by the shader. With a
// `layout: 'auto'` pipeline an unreferenced binding is dropped from the
// layout while the hosts still bind every kernel buffer, so the kernel must
// neither declare nor list it.
func TestGPURangeBoundOnlyBufferNotBound(t *testing.T) {
	src := `
package p

func f(text, head, cand []byte) {
	go for i := range head {
		cand[i] = text[i] ^ text[i+1]
	}
}
`
	k := transpileKernelOK(t, src)
	if strings.Contains(k.WGSL, "head") {
		t.Errorf("range-bound-only slice declared in WGSL:\n%s", k.WGSL)
	}
	if len(k.Buffers) != 2 {
		t.Fatalf("got %d buffers %s, want 2 (cand, text)", len(k.Buffers), paramNames(k.Buffers))
	}
	for i, b := range k.Buffers {
		decl := fmt.Sprintf("@binding(%d) var<storage, ", i+1)
		if !strings.Contains(k.WGSL, decl) || !strings.Contains(k.WGSL, "> "+b.Obj.Name()+": ") {
			t.Errorf("buffer %s not declared at binding %d:\n%s", b.Obj.Name(), i+1, k.WGSL)
		}
	}
	runNaga(t, k.WGSL)
}
