// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// This file adapts two small, high-value pieces of
// cogentcore.org/lab/gosl (BSD-3-Clause, v0.1.18):
//
//   - The Go->WGSL builtin-function name table, in the spirit of
//     gosl/gotosl/sledits.go's `Replaces` table (which maps Go/gosl source
//     text substrings to WGSL equivalents via byte-slice replacement). We
//     do not vendor that table verbatim (it targets gosl's own sltype/slvec
//     vector-math shim packages, which SPMD's GPU offload subset does not
//     use); instead MathFuncs below captures the same idea — a Go stdlib
//     name to WGSL builtin name mapping — restricted to the math.* subset
//     gpu_eligible.go's gpuApprovedMathFuncs allowlist accepts.
//
//   - The 16-byte uniform-buffer struct alignment/padding rule enforced by
//     gosl/alignsl/alignsl.go's CheckStructImpl (which verifies, rather than
//     computes, that a struct's total size is a multiple of 16 bytes given
//     4-byte scalar fields). PadTo16 below computes the padding
//     alignsl would otherwise merely flag as missing.
//
// Original source: cogentcore.org/lab@v0.1.18/gosl/{gotosl,alignsl}.
package wgslprint

// MathFuncs maps an approved math.* function name (see
// compiler.gpuApprovedMathFuncs) to its WGSL builtin spelling.
var MathFuncs = map[string]string{
	"Sqrt": "sqrt",
}

// PadTo16 returns the number of additional 4-byte (i32/u32/f32) padding
// fields needed so that a WGSL uniform-buffer struct made up of fieldCount
// 4-byte scalar fields has a total size that is a multiple of 16 bytes, per
// the rule alignsl.CheckStructImpl enforces (total size % 16 == 0).
func PadTo16(fieldCount int) int {
	size := fieldCount * 4
	rem := size % 16
	if rem == 0 {
		return 0
	}
	return (16 - rem) / 4
}
