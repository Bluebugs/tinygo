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

func TestGPUByteForPostCounterWraps(t *testing.T) {
	src := `
package p

func f(dst, src []uint32) {
	go for i := range len(src) {
		var sum uint32 = 0
		var b byte = 250
		for ; b != 4; b++ {
			sum++
		}
		dst[i] = src[i] + sum
	}
}
`
	wgsl := transpileOK(t, src)
	// Isolate the for-post masking from the (already-tested) byte-var
	// declaration and DeclStmt/IncDecStmt-outside-a-for-loop masking: the
	// post clause itself, "b = (...)", must be the thing carrying the mask.
	if !strings.Contains(wgsl, "b = ((b + 1) & 0xffu)") {
		t.Errorf("byte counter in for-loop post clause must be masked with & 0xffu:\n%s", wgsl)
	}
}

func TestGPUIntForPostCounterNotWrapped(t *testing.T) {
	src := `
package p

func f(dst, src []uint32) {
	go for i := range len(src) {
		var sum uint32 = 0
		for c := 0; c != 4; c++ {
			sum++
		}
		dst[i] = src[i] + sum
	}
}
`
	wgsl := transpileOK(t, src)
	if strings.Contains(wgsl, "0xffu") {
		t.Errorf("int counter in for-loop post clause must not be masked:\n%s", wgsl)
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

func TestGPUPackedByteReadWrite(t *testing.T) {
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
		"for (var lane: u32 = 0u; lane < 4u; lane++)",
		">> 2u] >> ((u32(i) & 3u) * 8u)) & 0xffu)", // packed byte read of src
		"& ~(0xffu << sh)) | (",                    // read-modify-write store into dst
	} {
		if !strings.Contains(wgsl, want) {
			t.Errorf("WGSL missing %q:\n%s", want, wgsl)
		}
	}
}

func TestGPUNoByteBufferNoLaneLoop(t *testing.T) {
	wgsl := transpileOK(t, "package p\n\nfunc f(dst, src []int32, b byte) {\n\tgo for i := range len(src) {\n\t\tdst[i] = src[i] + int32(b)\n\t}\n}\n")
	if strings.Contains(wgsl, "lane") {
		t.Errorf("kernel without byte buffers must keep the one-lane entry point:\n%s", wgsl)
	}
}

func TestGPUPackedLanesPerInvocation(t *testing.T) {
	for _, tc := range []struct {
		name, src string
		want      int
	}{
		{"byte", "package p\n\nfunc f(dst, src []byte) {\n\tgo for i := range len(src) {\n\t\tdst[i] = src[i]\n\t}\n}\n", 4},
		{"int32", "package p\n\nfunc f(dst, src []int32) {\n\tgo for i := range len(src) {\n\t\tdst[i] = src[i]\n\t}\n}\n", 1},
		{"byte read only", "package p\n\nfunc f(dst []int32, src []byte) {\n\tgo for i := range len(src) {\n\t\tdst[i] = int32(src[i])\n\t}\n}\n", 4},
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
			if k.LanesPerInvocation != tc.want {
				t.Errorf("LanesPerInvocation = %d, want %d", k.LanesPerInvocation, tc.want)
			}
		})
	}
}

func TestGPUPackedContinueStaysInLaneLoop(t *testing.T) {
	src := `
package p

func f(dst, src []byte) {
	go for i := range len(src) {
		if src[i] == 0 {
			continue
		}
		dst[i] = src[i]
	}
}
`
	wgsl := transpileOK(t, src)
	body := wgsl[strings.Index(wgsl, "for (var lane"):]
	if strings.Contains(body, "return;") && !strings.Contains(body, "continue;") {
		t.Errorf("a Go continue must continue the lane loop, not return from the invocation:\n%s", wgsl)
	}
}

func TestGPUPackedByteWriteOwnershipRejected(t *testing.T) {
	for _, tc := range []struct{ name, body, want string }{
		{"offset not word aligned", "dst[i+1] = src[i]", "slice write index"},
		{"read while written", "dst[i] = dst[i] + src[i]", "both read and written"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := "package p\n\nfunc f(dst, src []byte) {\n\tgo for i := range len(src) {\n\t\t" + tc.body + "\n\t}\n}\n"
			plan := parseAndAnalyzeGPULoop(t, src, 1)
			if !strings.Contains(plan.Reject, tc.want) {
				t.Fatalf("Reject = %q, want %q", plan.Reject, tc.want)
			}
		})
	}
}

func TestGPUPackedByteWriteOwnershipAccepted(t *testing.T) {
	for _, body := range []string{
		"dst[i] = src[i]",
		"dst[i*2] = src[i]\n\t\tdst[i*2+1] = src[i]",
		"dst[i*4+4] = src[i]",
		"dst[i]++",
	} {
		src := "package p\n\nfunc f(dst, src []byte) {\n\tgo for i := range len(src) {\n\t\t" + body + "\n\t}\n}\n"
		transpileOK(t, src)
	}
}

const hexGPUSrc = `
package p

const hextable = "0123456789abcdef"

func Encode(dst, src []byte) {
	go for i := range dst {
		v := src[i>>1]
		if i%2 == 0 {
			dst[i] = hextable[v>>4]
		} else {
			dst[i] = hextable[v&0x0f]
		}
	}
}
`

const hexSrcGPUSrc = `
package p

const hextable = "0123456789abcdef"

func EncodeSrc(dst, src []byte) {
	go for i := range src {
		dst[i*2] = hextable[src[i]>>4]
		dst[i*2+1] = hextable[src[i]&0x0f]
	}
}
`

func TestGPUTableHexEncodeEligible(t *testing.T) {
	for name, src := range map[string]string{"Encode": hexGPUSrc, "EncodeSrc": hexSrcGPUSrc} {
		t.Run(name, func(t *testing.T) {
			wgsl := transpileOK(t, src)
			decl := "var<private> hextable_tbl: array<u32, 16> = array<u32, 16>(48u, 49u, 50u, 51u, 52u, 53u, 54u, 55u, 56u, 57u, 97u, 98u, 99u, 100u, 101u, 102u);"
			if strings.Count(wgsl, decl) != 1 {
				t.Errorf("want exactly one table declaration %q:\n%s", decl, wgsl)
			}
			if strings.Index(wgsl, decl) > strings.Index(wgsl, "@compute") {
				t.Errorf("table must be declared at module scope before the entry point:\n%s", wgsl)
			}
			if !strings.Contains(wgsl, "hextable_tbl[") {
				t.Errorf("table not referenced:\n%s", wgsl)
			}
		})
	}
}

func TestGPUTableIndexUnprovenRejected(t *testing.T) {
	for _, tc := range []struct{ name, idx string }{
		{"raw int", "src[i]"},
		{"mask too wide", "src[i] & 0x1f"},
		{"shift too small", "byte(src[i]) >> 3"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := "package p\n\nconst tbl = \"0123456789abcdef\"\n\nfunc f(dst []byte, src []int32) {\n\tgo for i := range len(src) {\n\t\tdst[i] = tbl[" + tc.idx + "]\n\t}\n}\n"
			plan := parseAndAnalyzeGPULoop(t, src, 1)
			if !strings.Contains(plan.Reject, "table index") {
				t.Fatalf("Reject = %q, want a table index rejection", plan.Reject)
			}
		})
	}
}

func TestGPUTableIndexProvenAccepted(t *testing.T) {
	for _, tc := range []struct{ name, idx string }{
		{"mask", "src[i] & 0x0f"},
		{"mask rev", "0x0f & src[i]"},
		{"byte shift", "byte(src[i]) >> 4"},
		{"literal", "3"},
		{"unsigned mod", "uint32(src[i]) % 16"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			src := "package p\n\nconst tbl = \"0123456789abcdef\"\n\nfunc f(dst []byte, src []int32) {\n\tgo for i := range len(src) {\n\t\tdst[i] = tbl[" + tc.idx + "]\n\t}\n}\n"
			transpileOK(t, src)
		})
	}
}

func TestGPUTableNoTableUnchangedOutput(t *testing.T) {
	// A kernel without tables must not gain any var<private> line.
	wgsl := transpileOK(t, "package p\n\nfunc f(dst, src []int32) {\n\tgo for i := range len(src) {\n\t\tdst[i] = src[i] + 1\n\t}\n}\n")
	if strings.Contains(wgsl, "var<private>") {
		t.Errorf("unexpected table declaration:\n%s", wgsl)
	}
}
