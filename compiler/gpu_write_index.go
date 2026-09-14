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
	}
	return ""
}

// affineInIter matches e against C*iter + K (with the commutative and
// subtractive spellings, and parentheses), where C and K are integer
// constants known to the type checker. A bare constant yields c == 0.
func (a *gpuAnalyzer) affineInIter(e ast.Expr, iterObj types.Object) (c, k int64, ok bool) {
	if v, isConst := a.constInt(e); isConst {
		return 0, v, true
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
			return lc + rc, lk + rk, true
		case token.MUL:
			if m, isConst := a.constInt(x.Y); isConst {
				ic, ik, iok := a.affineInIter(x.X, iterObj)
				return ic * m, ik * m, iok
			}
			if m, isConst := a.constInt(x.X); isConst {
				ic, ik, iok := a.affineInIter(x.Y, iterObj)
				return ic * m, ik * m, iok
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
