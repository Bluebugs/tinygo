// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/ast"
	"go/token"
	"go/types"
	"testing"

	"github.com/tinygo-org/tinygo/loader"
)

// TestSPMDExtractNoSPMDCode verifies that packages without SPMD code return empty results.
func TestSPMDExtractNoSPMDCode(t *testing.T) {
	pkg := createTestPackage(t, []*ast.File{
		{
			Name: ast.NewIdent("test"),
			Decls: []ast.Decl{
				// Regular function with regular for loop
				&ast.FuncDecl{
					Name: ast.NewIdent("regularFunc"),
					Type: &ast.FuncType{
						Params:  &ast.FieldList{},
						Results: &ast.FieldList{},
					},
					Body: &ast.BlockStmt{
						List: []ast.Stmt{
							&ast.RangeStmt{
								For:    token.Pos(10),
								Range:  token.Pos(14),
								Key:    ast.NewIdent("i"),
								TokPos: token.Pos(16),
								Tok:    token.DEFINE,
								X:      &ast.BasicLit{Kind: token.INT, Value: "10"},
								Body: &ast.BlockStmt{
									Lbrace: token.Pos(20),
									Rbrace: token.Pos(30),
								},
								IsSpmd: false, // regular for loop
							},
						},
					},
				},
			},
		},
	}, nil)

	loops := extractSPMDLoops(pkg)
	if len(loops) != 0 {
		t.Errorf("expected 0 SPMD loops, got %d", len(loops))
	}

	funcs := extractSPMDFuncs(pkg)
	if len(funcs) != 0 {
		t.Errorf("expected 0 SPMD functions, got %d", len(funcs))
	}
}

// TestSPMDExtractLoop verifies basic SPMD loop extraction.
func TestSPMDExtractLoop(t *testing.T) {
	pkg := createTestPackage(t, []*ast.File{
		{
			Name: ast.NewIdent("test"),
			Decls: []ast.Decl{
				&ast.FuncDecl{
					Name: ast.NewIdent("spmdFunc"),
					Type: &ast.FuncType{
						Params:  &ast.FieldList{},
						Results: &ast.FieldList{},
					},
					Body: &ast.BlockStmt{
						List: []ast.Stmt{
							&ast.RangeStmt{
								For:    token.Pos(100),
								Range:  token.Pos(104),
								Key:    ast.NewIdent("i"),
								TokPos: token.Pos(106),
								Tok:    token.DEFINE,
								X:      &ast.BasicLit{Kind: token.INT, Value: "16"},
								Body: &ast.BlockStmt{
									Lbrace: token.Pos(110),
									Rbrace: token.Pos(200),
								},
								IsSpmd:    true,
								LaneCount: 4,
							},
						},
					},
				},
			},
		},
	}, nil)

	loops := extractSPMDLoops(pkg)
	if len(loops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(loops))
	}

	info := loops[token.Pos(100)]
	if info == nil {
		t.Fatal("expected loop info at position 100")
	}

	if info.ForPos != token.Pos(100) {
		t.Errorf("expected ForPos=100, got %d", info.ForPos)
	}
	if info.BodyStart != token.Pos(110) {
		t.Errorf("expected BodyStart=110, got %d", info.BodyStart)
	}
	if info.BodyEnd != token.Pos(200) {
		t.Errorf("expected BodyEnd=200, got %d", info.BodyEnd)
	}
	if info.LaneCount != 4 {
		t.Errorf("expected LaneCount=4, got %d", info.LaneCount)
	}
	if info.Constraint != -1 {
		t.Errorf("expected Constraint=-1 (unconstrained), got %d", info.Constraint)
	}
}

// TestSPMDExtractConstrainedLoop verifies extraction of constrained SPMD loops.
func TestSPMDExtractConstrainedLoop(t *testing.T) {
	pkg := createTestPackage(t, []*ast.File{
		{
			Name: ast.NewIdent("test"),
			Decls: []ast.Decl{
				&ast.FuncDecl{
					Name: ast.NewIdent("constrainedFunc"),
					Type: &ast.FuncType{
						Params:  &ast.FieldList{},
						Results: &ast.FieldList{},
					},
					Body: &ast.BlockStmt{
						List: []ast.Stmt{
							&ast.RangeStmt{
								For:    token.Pos(150),
								Range:  token.Pos(154),
								Key:    ast.NewIdent("i"),
								TokPos: token.Pos(156),
								Tok:    token.DEFINE,
								X:      &ast.BasicLit{Kind: token.INT, Value: "16"},
								Body: &ast.BlockStmt{
									Lbrace: token.Pos(160),
									Rbrace: token.Pos(250),
								},
								IsSpmd:     true,
								LaneCount:  4,
								Constraint: &ast.BasicLit{Kind: token.INT, Value: "4"},
							},
						},
					},
				},
			},
		},
	}, nil)

	loops := extractSPMDLoops(pkg)
	if len(loops) != 1 {
		t.Fatalf("expected 1 SPMD loop, got %d", len(loops))
	}

	info := loops[token.Pos(150)]
	if info == nil {
		t.Fatal("expected loop info at position 150")
	}

	if info.Constraint != 4 {
		t.Errorf("expected Constraint=4, got %d", info.Constraint)
	}
}

// TestSPMDExtractMultipleLoops verifies extraction of multiple SPMD loops.
func TestSPMDExtractMultipleLoops(t *testing.T) {
	pkg := createTestPackage(t, []*ast.File{
		{
			Name: ast.NewIdent("test"),
			Decls: []ast.Decl{
				&ast.FuncDecl{
					Name: ast.NewIdent("multiLoopFunc"),
					Type: &ast.FuncType{
						Params:  &ast.FieldList{},
						Results: &ast.FieldList{},
					},
					Body: &ast.BlockStmt{
						List: []ast.Stmt{
							// SPMD loop 1
							&ast.RangeStmt{
								For:    token.Pos(200),
								Range:  token.Pos(204),
								Key:    ast.NewIdent("i"),
								TokPos: token.Pos(206),
								Tok:    token.DEFINE,
								X:      &ast.BasicLit{Kind: token.INT, Value: "8"},
								Body: &ast.BlockStmt{
									Lbrace: token.Pos(210),
									Rbrace: token.Pos(250),
								},
								IsSpmd:    true,
								LaneCount: 4,
							},
							// Regular loop (should be ignored)
							&ast.RangeStmt{
								For:    token.Pos(300),
								Range:  token.Pos(304),
								Key:    ast.NewIdent("j"),
								TokPos: token.Pos(306),
								Tok:    token.DEFINE,
								X:      &ast.BasicLit{Kind: token.INT, Value: "5"},
								Body: &ast.BlockStmt{
									Lbrace: token.Pos(310),
									Rbrace: token.Pos(350),
								},
								IsSpmd: false,
							},
							// SPMD loop 2
							&ast.RangeStmt{
								For:    token.Pos(400),
								Range:  token.Pos(404),
								Key:    ast.NewIdent("k"),
								TokPos: token.Pos(406),
								Tok:    token.DEFINE,
								X:      &ast.BasicLit{Kind: token.INT, Value: "16"},
								Body: &ast.BlockStmt{
									Lbrace: token.Pos(410),
									Rbrace: token.Pos(500),
								},
								IsSpmd:    true,
								LaneCount: 4,
							},
						},
					},
				},
			},
		},
	}, nil)

	loops := extractSPMDLoops(pkg)
	if len(loops) != 2 {
		t.Fatalf("expected 2 SPMD loops, got %d", len(loops))
	}

	if loops[token.Pos(200)] == nil {
		t.Error("expected loop at position 200")
	}
	if loops[token.Pos(400)] == nil {
		t.Error("expected loop at position 400")
	}
	if loops[token.Pos(300)] != nil {
		t.Error("regular loop should not be extracted")
	}
}

// TestSPMDAnalyzeSignature verifies signature analysis for varying parameters.
func TestSPMDAnalyzeSignature(t *testing.T) {
	// Signature with varying parameter
	varyingType := types.NewVarying(types.Typ[types.Int32])
	sig := types.NewSignature(nil,
		types.NewTuple(types.NewParam(token.NoPos, nil, "x", varyingType)),
		nil,
		false,
	)

	info := analyzeSPMDSignature(sig)
	if info == nil {
		t.Fatal("expected non-nil SPMDFuncInfo")
	}

	if !info.HasVaryingParams {
		t.Error("expected HasVaryingParams=true")
	}
	if info.HasVaryingResults {
		t.Error("expected HasVaryingResults=false")
	}
	if len(info.VaryingParams) != 1 {
		t.Fatalf("expected 1 varying param, got %d", len(info.VaryingParams))
	}
	if info.VaryingParams[0].Index != 0 {
		t.Errorf("expected param index 0, got %d", info.VaryingParams[0].Index)
	}
	if info.VaryingParams[0].ElemType != types.Typ[types.Int32] {
		t.Errorf("expected int32 elem type, got %v", info.VaryingParams[0].ElemType)
	}
	if info.VaryingParams[0].Constraint != -1 {
		t.Errorf("expected constraint -1, got %d", info.VaryingParams[0].Constraint)
	}

	// Signature without varying parameters
	regularSig := types.NewSignature(nil,
		types.NewTuple(types.NewParam(token.NoPos, nil, "x", types.Typ[types.Int32])),
		nil,
		false,
	)

	regularInfo := analyzeSPMDSignature(regularSig)
	if regularInfo != nil {
		t.Error("expected nil SPMDFuncInfo for regular signature")
	}
}

// TestSPMDAnalyzeSignatureWithResults verifies signature analysis for varying results.
func TestSPMDAnalyzeSignatureWithResults(t *testing.T) {
	varyingType := types.NewVarying(types.Typ[types.Float32])
	sig := types.NewSignature(nil,
		types.NewTuple(types.NewParam(token.NoPos, nil, "x", types.Typ[types.Int])),
		types.NewTuple(types.NewParam(token.NoPos, nil, "", varyingType)),
		false,
	)

	info := analyzeSPMDSignature(sig)
	if info == nil {
		t.Fatal("expected non-nil SPMDFuncInfo")
	}

	if info.HasVaryingParams {
		t.Error("expected HasVaryingParams=false")
	}
	if !info.HasVaryingResults {
		t.Error("expected HasVaryingResults=true")
	}
}

// TestSPMDAnalyzeSignatureMultipleParams verifies signature with multiple varying params.
func TestSPMDAnalyzeSignatureMultipleParams(t *testing.T) {
	varying1 := types.NewVarying(types.Typ[types.Int32])
	varying2 := types.NewVaryingConstrained(types.Typ[types.Float32], 4)
	sig := types.NewSignature(nil,
		types.NewTuple(
			types.NewParam(token.NoPos, nil, "a", types.Typ[types.Int]),
			types.NewParam(token.NoPos, nil, "b", varying1),
			types.NewParam(token.NoPos, nil, "c", varying2),
		),
		nil,
		false,
	)

	info := analyzeSPMDSignature(sig)
	if info == nil {
		t.Fatal("expected non-nil SPMDFuncInfo")
	}

	if !info.HasVaryingParams {
		t.Error("expected HasVaryingParams=true")
	}
	if len(info.VaryingParams) != 2 {
		t.Fatalf("expected 2 varying params, got %d", len(info.VaryingParams))
	}

	// Check first varying param (index 1 in signature)
	if info.VaryingParams[0].Index != 1 {
		t.Errorf("expected param 0 index=1, got %d", info.VaryingParams[0].Index)
	}
	if info.VaryingParams[0].Constraint != -1 {
		t.Errorf("expected param 0 constraint=-1, got %d", info.VaryingParams[0].Constraint)
	}

	// Check second varying param (index 2 in signature)
	if info.VaryingParams[1].Index != 2 {
		t.Errorf("expected param 1 index=2, got %d", info.VaryingParams[1].Index)
	}
	if info.VaryingParams[1].Constraint != 4 {
		t.Errorf("expected param 1 constraint=4, got %d", info.VaryingParams[1].Constraint)
	}
}

// TestSPMDExtractFuncs verifies function extraction from package scope.
func TestSPMDExtractFuncs(t *testing.T) {
	pkg := types.NewPackage("test/pkg", "pkg")
	varyingType := types.NewVarying(types.Typ[types.Int32])
	sig := types.NewSignature(nil,
		types.NewTuple(types.NewParam(token.NoPos, nil, "x", varyingType)),
		nil,
		false,
	)
	fn := types.NewFunc(token.NoPos, pkg, "spmdFunc", sig)
	pkg.Scope().Insert(fn)

	loaderPkg := &loader.Package{
		Pkg: pkg,
		Files: []*ast.File{
			{Name: ast.NewIdent("pkg")},
		},
	}

	funcs := extractSPMDFuncs(loaderPkg)
	if len(funcs) != 1 {
		t.Fatalf("expected 1 SPMD function, got %d", len(funcs))
	}

	info := funcs[fn]
	if info == nil {
		t.Fatal("expected function info")
	}
	if !info.HasVaryingParams {
		t.Error("expected HasVaryingParams=true")
	}
}

// TestSPMDLoadInfo verifies loadSPMDInfo populates compiler context.
func TestSPMDLoadInfo(t *testing.T) {
	pkg := createTestPackage(t, []*ast.File{
		{
			Name: ast.NewIdent("test"),
			Decls: []ast.Decl{
				&ast.FuncDecl{
					Name: ast.NewIdent("testFunc"),
					Type: &ast.FuncType{
						Params:  &ast.FieldList{},
						Results: &ast.FieldList{},
					},
					Body: &ast.BlockStmt{
						List: []ast.Stmt{
							&ast.RangeStmt{
								For:    token.Pos(50),
								Range:  token.Pos(54),
								Key:    ast.NewIdent("i"),
								TokPos: token.Pos(56),
								Tok:    token.DEFINE,
								X:      &ast.BasicLit{Kind: token.INT, Value: "10"},
								Body: &ast.BlockStmt{
									Lbrace: token.Pos(60),
									Rbrace: token.Pos(100),
								},
								IsSpmd:    true,
								LaneCount: 4,
							},
						},
					},
				},
			},
		},
	}, map[string]*types.Func{
		"spmdFunc": types.NewFunc(token.NoPos, nil, "spmdFunc",
			types.NewSignature(nil,
				types.NewTuple(types.NewParam(token.NoPos, nil, "x", types.NewVarying(types.Typ[types.Int32]))),
				nil,
				false,
			),
		),
	})

	ctx := &compilerContext{}
	ctx.loadSPMDInfo(pkg)

	if !ctx.hasSPMDCode() {
		t.Fatal("expected hasSPMDCode=true")
	}

	if ctx.spmdInfo == nil {
		t.Fatal("expected non-nil spmdInfo")
	}

	if len(ctx.spmdInfo.Loops) != 1 {
		t.Errorf("expected 1 loop, got %d", len(ctx.spmdInfo.Loops))
	}

	if len(ctx.spmdInfo.Funcs) != 1 {
		t.Errorf("expected 1 function, got %d", len(ctx.spmdInfo.Funcs))
	}

	// Verify loopRanges is sorted
	if len(ctx.spmdInfo.loopRanges) != 1 {
		t.Fatalf("expected 1 loop range, got %d", len(ctx.spmdInfo.loopRanges))
	}
	if ctx.spmdInfo.loopRanges[0].start != token.Pos(60) {
		t.Errorf("expected start=60, got %d", ctx.spmdInfo.loopRanges[0].start)
	}
	if ctx.spmdInfo.loopRanges[0].end != token.Pos(100) {
		t.Errorf("expected end=100, got %d", ctx.spmdInfo.loopRanges[0].end)
	}
}

// TestSPMDLoadInfoEmpty verifies loadSPMDInfo with no SPMD code.
func TestSPMDLoadInfoEmpty(t *testing.T) {
	pkg := createTestPackage(t, []*ast.File{
		{
			Name: ast.NewIdent("test"),
			Decls: []ast.Decl{
				&ast.FuncDecl{
					Name: ast.NewIdent("regularFunc"),
					Type: &ast.FuncType{
						Params:  &ast.FieldList{},
						Results: &ast.FieldList{},
					},
					Body: &ast.BlockStmt{},
				},
			},
		},
	}, nil)

	ctx := &compilerContext{}
	ctx.loadSPMDInfo(pkg)

	if ctx.hasSPMDCode() {
		t.Error("expected hasSPMDCode=false")
	}

	if ctx.spmdInfo != nil {
		t.Error("expected nil spmdInfo")
	}
}

// TestSPMDIsInSPMDLoop verifies position-based loop detection.
func TestSPMDIsInSPMDLoop(t *testing.T) {
	ctx := &compilerContext{
		spmdInfo: &SPMDInfo{
			Loops: map[token.Pos]*SPMDLoopInfo{
				token.Pos(10): {
					ForPos:    token.Pos(10),
					BodyStart: token.Pos(20),
					BodyEnd:   token.Pos(50),
					LaneCount: 4,
				},
			},
			loopRanges: []spmdPosRange{
				{
					start: token.Pos(20),
					end:   token.Pos(50),
					info: &SPMDLoopInfo{
						ForPos:    token.Pos(10),
						BodyStart: token.Pos(20),
						BodyEnd:   token.Pos(50),
						LaneCount: 4,
					},
				},
			},
		},
	}

	tests := []struct {
		pos      token.Pos
		expected bool
	}{
		{token.Pos(5), false},   // before loop
		{token.Pos(15), false},  // between for and body start
		{token.Pos(20), true},   // at body start
		{token.Pos(30), true},   // inside loop
		{token.Pos(50), true},   // at body end
		{token.Pos(51), false},  // after loop
		{token.Pos(100), false}, // far after loop
	}

	for _, tc := range tests {
		result := ctx.isInSPMDLoop(tc.pos)
		got := result != nil
		if got != tc.expected {
			t.Errorf("isInSPMDLoop(%d): expected %v, got %v", tc.pos, tc.expected, got)
		}
		if got && result.ForPos != token.Pos(10) {
			t.Errorf("isInSPMDLoop(%d): expected ForPos=10, got %d", tc.pos, result.ForPos)
		}
	}

	// Test with nil spmdInfo
	emptyCtx := &compilerContext{}
	if emptyCtx.isInSPMDLoop(token.Pos(30)) != nil {
		t.Error("expected nil for context without SPMD info")
	}
}

// TestSPMDIsInSPMDLoopMultiple verifies binary search with multiple loops.
func TestSPMDIsInSPMDLoopMultiple(t *testing.T) {
	loop1 := &SPMDLoopInfo{ForPos: token.Pos(10), BodyStart: token.Pos(20), BodyEnd: token.Pos(50)}
	loop2 := &SPMDLoopInfo{ForPos: token.Pos(100), BodyStart: token.Pos(110), BodyEnd: token.Pos(150)}
	loop3 := &SPMDLoopInfo{ForPos: token.Pos(200), BodyStart: token.Pos(210), BodyEnd: token.Pos(250)}

	ctx := &compilerContext{
		spmdInfo: &SPMDInfo{
			loopRanges: []spmdPosRange{
				{start: token.Pos(20), end: token.Pos(50), info: loop1},
				{start: token.Pos(110), end: token.Pos(150), info: loop2},
				{start: token.Pos(210), end: token.Pos(250), info: loop3},
			},
		},
	}

	tests := []struct {
		pos         token.Pos
		expectedFor token.Pos // 0 if should return nil
	}{
		{token.Pos(15), 0},    // before first loop
		{token.Pos(30), 10},   // in first loop
		{token.Pos(60), 0},    // between first and second
		{token.Pos(120), 100}, // in second loop
		{token.Pos(180), 0},   // between second and third
		{token.Pos(230), 200}, // in third loop
		{token.Pos(260), 0},   // after third loop
	}

	for _, tc := range tests {
		result := ctx.isInSPMDLoop(tc.pos)
		if tc.expectedFor == 0 {
			if result != nil {
				t.Errorf("isInSPMDLoop(%d): expected nil, got loop at %d", tc.pos, result.ForPos)
			}
		} else {
			if result == nil {
				t.Errorf("isInSPMDLoop(%d): expected loop at %d, got nil", tc.pos, tc.expectedFor)
			} else if result.ForPos != tc.expectedFor {
				t.Errorf("isInSPMDLoop(%d): expected ForPos=%d, got %d", tc.pos, tc.expectedFor, result.ForPos)
			}
		}
	}
}

// TestSPMDGetSPMDLoopAt verifies direct loop lookup by for position.
func TestSPMDGetSPMDLoopAt(t *testing.T) {
	ctx := &compilerContext{
		spmdInfo: &SPMDInfo{
			Loops: map[token.Pos]*SPMDLoopInfo{
				token.Pos(100): {
					ForPos:    token.Pos(100),
					BodyStart: token.Pos(110),
					BodyEnd:   token.Pos(200),
					LaneCount: 4,
				},
			},
		},
	}

	// Lookup existing loop
	info := ctx.getSPMDLoopAt(token.Pos(100))
	if info == nil {
		t.Fatal("expected non-nil loop info")
	}
	if info.ForPos != token.Pos(100) {
		t.Errorf("expected ForPos=100, got %d", info.ForPos)
	}

	// Lookup non-existent loop
	if ctx.getSPMDLoopAt(token.Pos(50)) != nil {
		t.Error("expected nil for non-existent loop")
	}

	// Test with nil spmdInfo
	emptyCtx := &compilerContext{}
	if emptyCtx.getSPMDLoopAt(token.Pos(100)) != nil {
		t.Error("expected nil for context without SPMD info")
	}
}

// createTestPackage creates a loader.Package for testing with the given AST files and functions.
func createTestPackage(t *testing.T, files []*ast.File, funcs map[string]*types.Func) *loader.Package {
	pkg := types.NewPackage("test/pkg", "pkg")

	// Add functions to package scope if provided
	if funcs != nil {
		for _, fn := range funcs {
			pkg.Scope().Insert(fn)
		}
	}

	return &loader.Package{
		Files: files,
		Pkg:   pkg,
	}
}
