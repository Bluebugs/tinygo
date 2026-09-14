// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"strings"
	"testing"
)

func transpileOK(t *testing.T, src string) string {
	t.Helper()
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if plan.Reject != "" {
		t.Fatalf("plan rejected: %s", plan.Reject)
	}
	k, err := transpileWGSL(plan, 0)
	if err != nil {
		t.Fatalf("transpileWGSL: %v", err)
	}
	return k.WGSL
}

func TestGPUByteSliceEligible(t *testing.T) {
	src := `
package p

func f(dst, src []byte) {
	go for i := range len(src) {
		dst[i] = src[i]*3 + 7
	}
}
`
	wgsl := transpileOK(t, src)
	for _, want := range []string{
		"var<storage, read> src: array<u32>;",
		"var<storage, read_write> dst: array<u32>;",
		"& 0xffu)",
	} {
		if !strings.Contains(wgsl, want) {
			t.Errorf("WGSL missing %q:\n%s", want, wgsl)
		}
	}
}

func TestGPUByteWrapOnlyOnOverflowingOps(t *testing.T) {
	src := `
package p

func f(dst []uint32, src []uint32, b byte) {
	go for i := range len(src) {
		dst[i] = src[i] + uint32((b>>4)&0x0f)
	}
}
`
	wgsl := transpileOK(t, src)
	if strings.Contains(wgsl, "0xffu") {
		t.Errorf("range-preserving ops must not be wrapped:\n%s", wgsl)
	}
}

func TestGPUByteLocalAndScalar(t *testing.T) {
	src := `
package p

import "lanes"

func f(dst, src []byte, bias byte) {
	go for i := range len(src) {
		var v lanes.Varying[byte] = src[i]
		dst[i] = v + bias
	}
}
`
	wgsl := transpileOK(t, src)
	for _, want := range []string{"bias: u32,", ": u32", "& 0xffu)"} {
		if !strings.Contains(wgsl, want) {
			t.Errorf("WGSL missing %q:\n%s", want, wgsl)
		}
	}
}

func TestGPUByteConversions(t *testing.T) {
	src := `
package p

func f(dst []uint32, src []int32) {
	go for i := range len(src) {
		dst[i] = uint32(byte(src[i]))
	}
}
`
	wgsl := transpileOK(t, src)
	if !strings.Contains(wgsl, "(u32(src[i]) & 0xffu)") {
		t.Errorf("byte(int32) must emit (u32(x) & 0xffu):\n%s", wgsl)
	}
}

func TestGPUByteUnaryMinus(t *testing.T) {
	src := `
package p

func f(dst, src []uint32, b byte) {
	go for i := range len(src) {
		dst[i] = src[i] + uint32(-b)
	}
}
`
	wgsl := transpileOK(t, src)
	if !strings.Contains(wgsl, "((0u - params.b) & 0xffu)") {
		t.Errorf("unary minus on byte must emit ((0u - x) & 0xffu):\n%s", wgsl)
	}
}

func TestGPUByteFromFloatRejected(t *testing.T) {
	src := `
package p

func f(dst []byte, src []float32) {
	go for i := range len(src) {
		dst[i] = byte(src[i])
	}
}
`
	plan := parseAndAnalyzeGPULoop(t, src, 1)
	if !strings.Contains(plan.Reject, "float") {
		t.Fatalf("Reject = %q, want float->byte conversion rejection", plan.Reject)
	}
}

func TestGPUByteCompoundAssignWraps(t *testing.T) {
	for _, op := range []string{"+=", "-=", "*="} {
		t.Run(op, func(t *testing.T) {
			src := `
package p

func f(dst []uint32, src []uint32, n byte) {
	go for i := range len(src) {
		var v byte = 1
		v ` + op + ` n
		dst[i] = src[i] + uint32(v)
	}
}
`
			wgsl := transpileOK(t, src)
			if !strings.Contains(wgsl, "& 0xffu)") {
				t.Errorf("compound assign %s on byte must be masked with & 0xffu:\n%s", op, wgsl)
			}
		})
	}
}

func TestGPUByteIncDecWraps(t *testing.T) {
	for _, stmt := range []string{"v++", "v--"} {
		t.Run(stmt, func(t *testing.T) {
			src := `
package p

func f(dst []uint32, src []uint32) {
	go for i := range len(src) {
		var v byte = 1
		` + stmt + `
		dst[i] = src[i] + uint32(v)
	}
}
`
			wgsl := transpileOK(t, src)
			if !strings.Contains(wgsl, "& 0xffu)") {
				t.Errorf("%s on byte must be masked with & 0xffu:\n%s", stmt, wgsl)
			}
		})
	}
}

func TestGPUNonByteCompoundAssignNotWrapped(t *testing.T) {
	src := `
package p

func f(dst []uint32, src []uint32, n int32) {
	go for i := range len(src) {
		var x int32 = 1
		x += n
		dst[i] = src[i] + uint32(x)
	}
}
`
	wgsl := transpileOK(t, src)
	if strings.Contains(wgsl, "0xffu") {
		t.Errorf("non-byte compound assign must not be masked:\n%s", wgsl)
	}
}

func TestGPUNarrowSignedStillRejected(t *testing.T) {
	for _, ty := range []string{"int8", "int16", "uint16"} {
		t.Run(ty, func(t *testing.T) {
			src := "package p\n\nfunc f(dst, src []" + ty + ") {\n\tgo for i := range len(src) {\n\t\tdst[i] = src[i]\n\t}\n}\n"
			plan := parseAndAnalyzeGPULoop(t, src, 1)
			if !strings.Contains(plan.Reject, "unsupported basic type") {
				t.Fatalf("Reject = %q, want unsupported basic type", plan.Reject)
			}
		})
	}
}
