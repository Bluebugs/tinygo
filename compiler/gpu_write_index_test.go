// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"strings"
	"testing"
)

func gpuWriteIndexSrc(body string) string {
	return `
package p

func f(dst, src []int32, n int) {
	go for i := range n {
		` + body + `
	}
}
`
}

func TestGPUWriteIndexAccepted(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"identity", "dst[i] = src[i]"},
		{"plus const", "dst[i+1] = src[i]"},
		{"const plus", "dst[3+i] = src[i]"},
		{"minus const", "dst[i-1] = src[i]"},
		{"scaled", "dst[i*2] = src[i]"},
		{"scaled rev", "dst[2*i] = src[i]"},
		{"stride pair", "dst[i*2] = src[i]\n\t\tdst[i*2+1] = src[i]"},
		{"paren", "dst[(i*2)+1] = src[i]"},
		{"same index both arms", "if src[i] > 0 {\n\t\t\tdst[i] = 1\n\t\t} else {\n\t\t\tdst[i] = 2\n\t\t}"},
		{"incdec", "dst[i]++"},
		{"named const", "const k = 2\n\t\tdst[i*k] = src[i]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := parseAndAnalyzeGPULoop(t, gpuWriteIndexSrc(tc.body), 1)
			if plan.Reject != "" {
				t.Fatalf("unexpected reject: %s", plan.Reject)
			}
		})
	}
}

func TestGPUWriteIndexRejected(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"data dependent", "dst[src[i]] = 1"},
		{"constant", "_ = i\n\t\tdst[0] = 1"},
		{"shift", "dst[i>>1] = src[i]"},
		{"modulo", "dst[i%4] = src[i]"},
		{"uniform var", "_ = i\n\t\tdst[n] = 1"},
		{"uniform scale", "dst[i*n] = src[i]"},
		{"overlapping offsets", "dst[i] = 1\n\t\tdst[i+1] = 2"},
		{"stride offsets too far", "dst[i*2] = 1\n\t\tdst[i*2+2] = 2"},
		{"mixed scales", "dst[i] = 1\n\t\tdst[i*2] = 2"},
		{"negative scale", "dst[-1*i] = 1"},
		{"local index", "j := i\n\t\tdst[j] = 1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := parseAndAnalyzeGPULoop(t, gpuWriteIndexSrc(tc.body), 1)
			if !strings.Contains(plan.Reject, "slice write index") {
				t.Fatalf("Reject = %q, want a slice write index rejection", plan.Reject)
			}
		})
	}
}

func TestGPUWrittenSliceReadAccepted(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"own element plain", "dst[i] = dst[i] + 1"},
		{"own element scaled", "dst[i*2+1] = dst[i*2+1] ^ src[i]"},
		{"compound", "dst[i] += src[i]"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := parseAndAnalyzeGPULoop(t, gpuWriteIndexSrc(tc.body), 1)
			if plan.Reject != "" {
				t.Fatalf("unexpected reject: %s", plan.Reject)
			}
		})
	}
}

func TestGPUWrittenSliceReadRejected(t *testing.T) {
	for _, tc := range []struct{ name, body string }{
		{"neighbor", "dst[i] = dst[i+1]"},
		{"constant", "dst[i] = dst[0]"},
		{"other statement", "x := dst[i+1]\n\t\tdst[i] = x"},
		{"same index other statement", "x := dst[i]\n\t\tdst[i] = x + 1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			plan := parseAndAnalyzeGPULoop(t, gpuWriteIndexSrc(tc.body), 1)
			if !strings.Contains(plan.Reject, "both read and written") {
				t.Fatalf("Reject = %q, want a read-while-written rejection", plan.Reject)
			}
		})
	}
}
