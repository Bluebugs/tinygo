// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"fmt"
	"strings"
	"testing"
)

// gpuNBufferSrc builds an eligible kernel reading n-1 distinct slices and
// writing one, i.e. n storage buffers.
func gpuNBufferSrc(n int) string {
	var params, sum []string
	for i := 0; i < n-1; i++ {
		params = append(params, fmt.Sprintf("s%d", i))
		sum = append(sum, fmt.Sprintf("s%d[i]", i))
	}
	return fmt.Sprintf(`
package p

func f(dst, %s []int32) {
	go for i := range dst {
		dst[i] = %s
	}
}
`, strings.Join(params, ", "), strings.Join(sum, " + "))
}

// The Vulkan host binds at most 8 storage buffers (SPMD_VK_MAX_BUFFERS in
// src/runtime/gpu_vulkan.c); a larger kernel must stay on the CPU instead of
// passing the guard and panicking in register.
func TestGPUVulkanBufferLimit(t *testing.T) {
	for _, tc := range []struct {
		n    int
		skip bool
	}{{8, false}, {9, true}} {
		t.Run(fmt.Sprint(tc.n), func(t *testing.T) {
			k := transpileKernelOK(t, gpuNBufferSrc(tc.n))
			if len(k.Buffers) != tc.n {
				t.Fatalf("kernel has %d buffers, want %d", len(k.Buffers), tc.n)
			}
			reason := gpuHostBufferLimitReject("vulkan", len(k.Buffers))
			if got := reason != ""; got != tc.skip {
				t.Fatalf("vulkan skip = %v (%q), want %v", got, reason, tc.skip)
			}
			if tc.skip && !strings.Contains(reason, "exceeds the Vulkan host limit of 8") {
				t.Errorf("unexpected reason %q", reason)
			}
			for _, host := range []string{"", "browser"} {
				if r := gpuHostBufferLimitReject(host, len(k.Buffers)); r != "" {
					t.Errorf("host %q rejected: %s", host, r)
				}
			}
		})
	}
}
