// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

// Write-index distinctness proof for GPU-offloaded `go for` loops.
//
// On the CPU every lane's store is sequenced; on the GPU every invocation
// runs concurrently against one shared storage buffer, so two invocations
// that write the same element race and the result is unspecified. The
// eligibility gate therefore accepts a slice write only when its index is
// C*iter + K (C >= 1 and K compile-time constants) and, across all writes
// to that slice, every write shares C and max(K)-min(K) < C: then
// C*i + K == C*i' + K' with i != i' would need |K-K'| >= C.
//
// Byte slices travel to the GPU packed four per u32 word, and each
// invocation runs four lanes (gpu_wgsl.go), so distinct bytes are not
// enough: two invocations must never read-modify-write the same WORD.
// With shared C and max(K)-min(K) < C, invocation w writes bytes
// [4Cw+minK, 4Cw+minK+4C); requiring minK%4 == 0 makes that exactly the
// words [Cw+minK/4, Cw+minK/4+C), disjoint across invocations. A written
// byte slice must also not be read elsewhere in the kernel, or one
// invocation could observe a word another is rewriting.

import (
	"fmt"
	"go/ast"
	"go/constant"
	"go/token"
	"go/types"
)

type gpuAffineWrite struct {
	c, k int64
}

func (a *gpuAnalyzer) writeIndexReject(body *ast.BlockStmt, iterIdent *ast.Ident) string {
	if iterIdent == nil {
		return "slice write index cannot be proven distinct: loop has no plain iteration variable"
	}
	iterObj := a.info.Defs[iterIdent]
	if iterObj == nil {
		return "slice write index cannot be proven distinct: loop iteration variable has no object"
	}

	writes := map[*types.Var][]gpuAffineWrite{}
	targets := map[*ast.IndexExpr]bool{}
	var reject string
	record := func(target ast.Expr) {
		if reject != "" {
			return
		}
		idx, ok := target.(*ast.IndexExpr)
		if !ok {
			return
		}
		ident, ok := idx.X.(*ast.Ident)
		if !ok {
			return
		}
		v, ok := a.info.Uses[ident].(*types.Var)
		if !ok {
			return
		}
		if _, isSlice := v.Type().Underlying().(*types.Slice); !isSlice {
			return
		}
		c, k, ok := a.affineInIter(idx.Index, iterObj)
		if !ok || c < 1 {
			reject = fmt.Sprintf("slice write index to %s is not of the form C*%s+K with constant C>=1 (GPU invocations could write the same element)", ident.Name, iterIdent.Name)
			return
		}
		writes[v] = append(writes[v], gpuAffineWrite{c, k})
		targets[idx] = true
	}
	ast.Inspect(body, func(n ast.Node) bool {
		switch s := n.(type) {
		case *ast.AssignStmt:
			for _, lhs := range s.Lhs {
				record(lhs)
			}
		case *ast.IncDecStmt:
			record(s.X)
		}
		return reject == ""
	})
	if reject != "" {
		return reject
	}

	for v, ws := range writes {
		c := ws[0].c
		minK, maxK := ws[0].k, ws[0].k
		for _, w := range ws[1:] {
			if w.c != c {
				return fmt.Sprintf("slice write index to %s uses different scales (%d and %d); GPU invocations could overlap", v.Name(), c, w.c)
			}
			minK = min(minK, w.k)
			maxK = max(maxK, w.k)
		}
		if maxK-minK >= c {
			return fmt.Sprintf("slice write index to %s has offsets spanning %d >= scale %d; GPU invocations could overlap", v.Name(), maxK-minK, c)
		}
		if gpuIsByteSlice(v.Type()) && minK%4 != 0 {
			return fmt.Sprintf("slice write index to %s has offset %d not aligned to a 4-byte word; GPU invocations could share a word", v.Name(), minK)
		}
	}

	ast.Inspect(body, func(n ast.Node) bool {
		if reject != "" {
			return false
		}
		idx, ok := n.(*ast.IndexExpr)
		if !ok || targets[idx] {
			return true
		}
		ident, ok := idx.X.(*ast.Ident)
		if !ok {
			return true
		}
		if v, ok := a.info.Uses[ident].(*types.Var); ok && writes[v] != nil && gpuIsByteSlice(v.Type()) {
			reject = fmt.Sprintf("byte slice %s is both read and written in GPU-offloaded loop (invocations could race on a shared word)", ident.Name)
		}
		return true
	})
	return reject
}

// gpuIsByteSlice reports whether t is a slice whose element type has
// underlying kind uint8; such slices are packed into u32 words on the GPU.
func gpuIsByteSlice(t types.Type) bool {
	s, ok := t.Underlying().(*types.Slice)
	if !ok {
		return false
	}
	b, ok := s.Elem().Underlying().(*types.Basic)
	return ok && b.Kind() == types.Uint8
}

// gpuAffineLimit bounds every constant and coefficient affineInIter
// accepts, so the int64 products and sums it (and writeIndexReject's
// maxK-minK) computes cannot overflow; larger values are simply unproven.
const gpuAffineLimit = 1 << 31

func gpuAffineInRange(vs ...int64) bool {
	for _, v := range vs {
		if v > gpuAffineLimit || v < -gpuAffineLimit {
			return false
		}
	}
	return true
}

// affineInIter matches e against C*iter + K (with the commutative and
// subtractive spellings, and parentheses), where C and K are integer
// constants known to the type checker. A bare constant yields c == 0.
func (a *gpuAnalyzer) affineInIter(e ast.Expr, iterObj types.Object) (c, k int64, ok bool) {
	if v, isConst := a.constInt(e); isConst {
		return 0, v, gpuAffineInRange(v)
	}
	switch x := e.(type) {
	case *ast.ParenExpr:
		return a.affineInIter(x.X, iterObj)
	case *ast.Ident:
		if a.info.Uses[x] == iterObj {
			return 1, 0, true
		}
		return 0, 0, false
	case *ast.BinaryExpr:
		switch x.Op {
		case token.ADD, token.SUB:
			lc, lk, lok := a.affineInIter(x.X, iterObj)
			rc, rk, rok := a.affineInIter(x.Y, iterObj)
			if !lok || !rok {
				return 0, 0, false
			}
			if x.Op == token.SUB {
				rc, rk = -rc, -rk
			}
			c, k = lc+rc, lk+rk
			return c, k, gpuAffineInRange(c, k)
		case token.MUL:
			if m, isConst := a.constInt(x.Y); isConst && gpuAffineInRange(m) {
				ic, ik, iok := a.affineInIter(x.X, iterObj)
				c, k = ic*m, ik*m
				return c, k, iok && gpuAffineInRange(c, k)
			}
			if m, isConst := a.constInt(x.X); isConst && gpuAffineInRange(m) {
				ic, ik, iok := a.affineInIter(x.Y, iterObj)
				c, k = ic*m, ik*m
				return c, k, iok && gpuAffineInRange(c, k)
			}
		}
	}
	return 0, 0, false
}

func (a *gpuAnalyzer) constInt(e ast.Expr) (int64, bool) {
	tv, ok := a.info.Types[e]
	if !ok || tv.Value == nil {
		return 0, false
	}
	v := constant.ToInt(tv.Value)
	if v.Kind() != constant.Int {
		return 0, false
	}
	return constant.Int64Val(v)
}
