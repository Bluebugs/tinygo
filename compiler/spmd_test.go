// Copyright 2026 The TinyGo Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package compiler

import (
	"go/ast"
	"go/token"
	"go/types"
	"os"
	"reflect"
	"strings"
	"testing"
	"unsafe"

	"github.com/tinygo-org/tinygo/compileopts"
	"github.com/tinygo-org/tinygo/loader"
	"golang.org/x/tools/go/ssa"
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
	varying2 := types.NewVarying(types.Typ[types.Float32])
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

	// Check second varying param (index 2 in signature)
	if info.VaryingParams[1].Index != 2 {
		t.Errorf("expected param 1 index=2, got %d", info.VaryingParams[1].Index)
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

// ssaAllocWithType creates an *ssa.Alloc with the given type set via reflection,
// because ssa.Alloc.setType is unexported.
func ssaAllocWithType(t types.Type, heap bool) *ssa.Alloc {
	alloc := &ssa.Alloc{Heap: heap}
	// Access the unexported typ field through the embedded register struct.
	// reflect.ValueOf(alloc).Elem() gives the Alloc struct value.
	// The first field of Alloc is the embedded register struct (anInstruction + num + typ ...).
	// We traverse the struct fields to find "typ" in the embedded register.
	v := reflect.ValueOf(alloc).Elem()
	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)
		if field.Kind() == reflect.Struct {
			for j := 0; j < field.NumField(); j++ {
				inner := field.Type().Field(j)
				if inner.Name == "typ" {
					// Use unsafe to write through the unexported field.
					ptr := (*types.Type)(unsafe.Pointer(field.Field(j).UnsafeAddr()))
					*ptr = t
					return alloc
				}
			}
		}
	}
	panic("could not find typ field in ssa.Alloc — struct layout may have changed")
}

// TestSPMDIsAllocaOrigin verifies that spmdIsAllocaOriginStatic correctly
// identifies stack-allocated arrays large enough for full vector operations.
func TestSPMDIsAllocaOrigin(t *testing.T) {
	i32Type := types.Typ[types.Int32]
	arrType16 := types.NewArray(i32Type, 16)
	ptrToArr16 := types.NewPointer(arrType16)

	// Stack alloc (Heap=false) with array length 16.
	stackAlloc := ssaAllocWithType(ptrToArr16, false)

	if !spmdIsAllocaOriginStatic(stackAlloc, 4) {
		t.Error("stack alloc with array[16] should be detected for laneCount=4")
	}
	if !spmdIsAllocaOriginStatic(stackAlloc, 16) {
		t.Error("stack alloc with array[16] should be detected for laneCount=16")
	}

	// Heap alloc.
	heapAlloc := ssaAllocWithType(ptrToArr16, true)
	if spmdIsAllocaOriginStatic(heapAlloc, 4) {
		t.Error("heap alloc should NOT be detected as alloca")
	}

	// Small array (length 2, less than laneCount=4).
	arrType2 := types.NewArray(i32Type, 2)
	ptrToArr2 := types.NewPointer(arrType2)
	smallAlloc := ssaAllocWithType(ptrToArr2, false)
	if spmdIsAllocaOriginStatic(smallAlloc, 4) {
		t.Error("small array[2] should NOT be detected for laneCount=4")
	}

	// Array length exactly one less than laneCount — should NOT match.
	arrType3 := types.NewArray(i32Type, 3)
	ptrToArr3 := types.NewPointer(arrType3)
	oneShortAlloc := ssaAllocWithType(ptrToArr3, false)
	if spmdIsAllocaOriginStatic(oneShortAlloc, 4) {
		t.Error("array[3] should NOT be detected for laneCount=4")
	}

	// Nil value should return false.
	if spmdIsAllocaOriginStatic(nil, 4) {
		t.Error("nil should NOT be detected as alloca")
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

// compileSPMDSource compiles Go source (with GOEXPERIMENT=spmd) through
// testCompilePackage and returns the LLVM IR as a string.
// The source must have package main and a main function.
// Uses the wasm target so that SIMD128 is available for the SPMD backend.
func compileSPMDSource(t *testing.T, src string) string {
	t.Helper()
	// The SPMD parser gating in go/parser reads buildcfg.Experiment.SPMD
	// which is set at process init from GOEXPERIMENT. Ensure the env var is
	// present for any sub-processes and for the in-process parser call.
	t.Setenv("GOEXPERIMENT", "spmd")

	// Write source to a temp file under the testdata directory so that
	// testCompilePackage (which expects ./testdata/<file>) can load it.
	dir := "./testdata"
	f, err := os.CreateTemp(dir, "spmd_test_*.go")
	if err != nil {
		t.Fatalf("failed to create temp source file: %v", err)
	}
	name := f.Name()
	defer os.Remove(name)
	if _, err := f.WriteString(src); err != nil {
		f.Close()
		t.Fatalf("failed to write source: %v", err)
	}
	f.Close()

	// Use just the filename relative to ./testdata/ as testCompilePackage expects.
	relName := strings.TrimPrefix(name, dir+"/")

	options := &compileopts.Options{
		Target:       "wasm",
		GOExperiment: "spmd",
	}
	mod, errs := testCompilePackage(t, options, relName)
	if len(errs) > 0 {
		msgs := make([]string, len(errs))
		for i, e := range errs {
			msgs[i] = e.Error()
		}
		t.Fatalf("compile errors:\n%s", strings.Join(msgs, "\n"))
	}
	return mod.String()
}

// mustContain fails the test if ir does not contain substr.
func mustContain(t *testing.T, ir, substr string) {
	t.Helper()
	if !strings.Contains(ir, substr) {
		t.Errorf("IR missing expected pattern %q", substr)
	}
}

// mustNotContain fails the test if ir contains substr.
func mustNotContain(t *testing.T, ir, substr string) {
	t.Helper()
	if strings.Contains(ir, substr) {
		t.Errorf("IR contains unexpected pattern %q", substr)
	}
}

// mustContainAny fails the test if ir contains none of the given substrings.
func mustContainAny(t *testing.T, ir string, substrs ...string) {
	t.Helper()
	for _, s := range substrs {
		if strings.Contains(ir, s) {
			return
		}
	}
	t.Errorf("IR missing all of: %v", substrs)
}

// TestSPMDVaryingPointerFieldAddr_Gather verifies that a field read through a
// Varying[*Struct] value compiles to per-lane GEPs (via spmdFieldAddrPerLane)
// followed by a masked gather or masked vector load.
//
// Case D in the *ssa.FieldAddr handler does not exist yet, so this test
// is expected to FAIL until Task 8 adds it.
func TestSPMDVaryingPointerFieldAddr_Gather(t *testing.T) {
	src := `package main

import "lanes"

type Point struct{ X, Y int }

var pts [16]Point

func main() {
	var acc lanes.Varying[int]
	go for i := range 16 {
		p := &pts[i]  // Varying[*Point]
		acc += p.X    // gather read of field X
	}
	_ = acc
}
`
	ir := compileSPMDSource(t, src)
	// spmdFieldAddrPerLane emits extract/GEP/insert for each lane of the pointer vector.
	mustContain(t, ir, "extractelement")
	mustContain(t, ir, "getelementptr")
	mustContain(t, ir, "insertelement")
	// The field address vector feeds a gather load or masked vector load.
	mustContainAny(t, ir, "masked.gather", "masked.load")
}

// TestSPMDVaryingPointerFieldAddr_Scatter verifies that a field write through a
// Varying[*Struct] value compiles to a masked scatter or masked vector store.
//
// Case D in the *ssa.FieldAddr handler does not exist yet, so this test
// is expected to FAIL until Task 8 adds it.
func TestSPMDVaryingPointerFieldAddr_Scatter(t *testing.T) {
	src := `package main

type Point struct{ X, Y int }

var pts [16]Point

func main() {
	go for i := range 16 {
		p := &pts[i]  // Varying[*Point]
		p.Y = i       // scatter write to field Y
	}
}
`
	ir := compileSPMDSource(t, src)
	mustContain(t, ir, "extractelement")
	mustContain(t, ir, "getelementptr")
	mustContain(t, ir, "insertelement")
	mustContainAny(t, ir, "masked.scatter", "masked.store")
}

// TestSPMDVaryingPointerFieldAddr_Contiguous verifies that when the base
// access is contiguous (uniform base + laneIndex), the FieldAddr propagates
// contiguous-ness and the backend emits a masked vector store rather than scatter.
//
// Case D in the *ssa.FieldAddr handler does not exist yet, so this test
// is expected to FAIL until Task 8 adds it.
func TestSPMDVaryingPointerFieldAddr_Contiguous(t *testing.T) {
	src := `package main

type Point struct{ X, Y int }

var pts [32]Point

func kernel(base int) {
	go for i := range 16 {
		p := &pts[base+i]  // contiguous: uniform base + varying lane index
		p.X = i
	}
}

func main() {
	kernel(0)
}
`
	ir := compileSPMDSource(t, src)
	// D2 (contiguous Varying[*S]) propagates contiguous-ness from IndexAddr through FieldAddr.
	// The main body (all-ones mask) emits a plain vector store via the all-ones fast path;
	// "masked.store" is not present because the mask is omitted for all-lanes-active paths.
	// The scalar field GEP ("fieldaddr.scalar") plus a plain store confirm the contiguous path.
	mustContain(t, ir, "fieldaddr.scalar")
	// Contiguous main body must not scatter (scatter only appears in the dead tail).
	// Verify the main body's store is a plain vector store, not scatter-per-field.
	mustContain(t, ir, "store <4 x i32>")
}

// TestSPMDVaryingLocalMaskedInTail verifies that a varying-local
// compound assignment in a partial-mask go-for gets masked in the
// tail-body block, preventing inactive-lane NaN leaks.
//
// 5 iterations on 4-wide SIMD => main=4 iters, tail=1 iter (lanes 1-3
// inactive). The tail-body store of the accumulator must be masked,
// either via @llvm.masked.store or a load-select-store blend with a
// <4 x i1> select.
func TestSPMDVaryingLocalMaskedInTail(t *testing.T) {
	src := `package main

import (
	"lanes"
	"reduce"
)

var data = []float64{1, 2, 3, 4, 5}

func main() {
	var acc lanes.Varying[float64]
	go for i, x := range data {
		_ = i
		acc += x
	}
	_ = reduce.Add(acc)
}
`
	ir := compileSPMDSource(t, src)

	// Tail-body store must use the mask — either an explicit
	// masked.store intrinsic, or a load-select-store blend with
	// <4 x i1> select. Without the fix, the alloca is lifted away
	// and neither pattern appears.
	mustContainAny(t, ir, "masked.store", "select <4 x i1>")
}
