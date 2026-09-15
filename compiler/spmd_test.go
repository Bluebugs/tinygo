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
	"tinygo.org/x/go-llvm"
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

// TestSPMDV5VaryingAllocaLLVMType verifies that a Varying[int] alloca
// inside a go-for over []float64 is materialized at the loop's
// iteration width (2 on WASM SIMD128), and ALL load/reduce ops on the
// alloca are also at that width. v5 expects this without any
// TinyGo-side audit because the type itself carries the width.
func TestSPMDV5VaryingAllocaLLVMType(t *testing.T) {
	src := `package main

import (
	"lanes"
	"reduce"
)

var data = []float64{1, 2, 3, 4, 5, 6, 7, 8}

func main() {
	var acc lanes.Varying[int]
	acc = 0
	go for i, _ := range data {
		acc += i
	}
	_ = reduce.Add(acc)
}
`
	ir := compileSPMDSource(t, src)
	mustContain(t, ir, "alloca <2 x i32>")
	mustContain(t, ir, "load <2 x i32>, ptr %acc")
	mustContain(t, ir, "reduce.add.v2i32")
	mustNotContain(t, ir, "load <4 x i32>, ptr %acc")
	mustNotContain(t, ir, "reduce.add.v4i32")
}

// TestSPMDV5VaryingByteIndexAddr verifies that b[i] = c in a []byte
// loop produces matching <N x ptr> address and <N x i8> value vectors,
// where N is the loop's lane count (16 on WASM128). This is the
// to-upper bug -- v3/v4 broke this because the IndexAddr address vector
// was sized at int's natural width, not the loop width.
func TestSPMDV5VaryingByteIndexAddr(t *testing.T) {
	src := `package main

var src = []byte("hello world")

func main() {
	b := make([]byte, len(src))
	go for i, c := range src {
		if 'a' <= c && c <= 'z' {
			b[i] = c - 32
		} else {
			b[i] = c
		}
	}
	_ = b
}
`
	ir := compileSPMDSource(t, src)
	// On WASM128, byte iter = 16 lanes. Address and value vectors must
	// agree at <16 x ptr> / <16 x i8>. Check absence of int-natural
	// width leak.
	mustNotContain(t, ir, "<4 x ptr>")
	mustNotContain(t, ir, "scatter.v16i8.v4p0")
}

// TestSPMDV5LoContainsReduceAny verifies that reduce.Any on a varying
// comparison result produces the right-width reduce intrinsic.
// lo-contains failed in v3/v4 because reduce.Any read element-natural
// width.
func TestSPMDV5LoContainsReduceAny(t *testing.T) {
	src := `package main

import (
	"lanes"
	"reduce"
)

var data = []int32{1, 2, 3, 4, 5, 6, 7, 8}

func find(target int32) bool {
	var found lanes.Varying[bool]
	go for _, x := range data {
		if x == target {
			found = true
		}
	}
	return reduce.Any(found)
}

func main() {
	_ = find(5)
}
`
	ir := compileSPMDSource(t, src)
	// On WASM128, int32 iter = 4 lanes. reduce.Any over a <4 x i1>-based
	// mask is lowered via sext to <4 x i32> then llvm.wasm.anytrue.v4i32.
	// The v4 width must appear: a width mismatch (e.g. v4i1 on 8-wide
	// accumulator) would produce anytrue.v8i32 instead.
	mustContainAny(t, ir, "anytrue.v4i32", "reduce.or.v4i1")
}

// TestSPMDVaryingSliceAllocaSize asserts that a Varying[[]int] alloca
// inside a SPMD loop is sized [N x slice_struct] where N is the loop's
// lane count, not [1 x slice_struct]. The latter caused the v6.1
// array-counting UB (write 4 lanes into 1-lane alloca).
func TestSPMDVaryingSliceAllocaSize(t *testing.T) {
	src := `package main

var arrays = [][]int{
	{1, 2, 3},
	{4, 5},
	{6},
	{7, 8, 9, 10},
}

func countArrays(arrays [][]int) []int {
	result := make([]int, len(arrays))
	go for i, secondLevel := range arrays {
		t := 0
		for _, v := range secondLevel {
			t += v
		}
		result[i] = t
	}
	return result
}

func main() {
	_ = countArrays(arrays)
}
`
	ir := compileSPMDSource(t, src)
	// On WASM128, [][]int outer range is 4-wide (int = i32 = 4 lanes).
	// The secondLevel Varying[[]int] alloca must be [4 x { ptr, i32, i32 }],
	// not [1 x { ptr, i32, i32 }] (the pre-fix size from spmdLaneCount=1).
	if !strings.Contains(ir, "alloca [4 x { ptr, i32, i32 }]") {
		t.Errorf("expected secondLevel alloca [4 x { ptr, i32, i32 }] in IR; full IR snippet:\n%s",
			extractAllocaLines(ir))
	}
	if strings.Contains(ir, "alloca [1 x { ptr, i32, i32 }]") {
		t.Errorf("found stale [1 x { ptr, i32, i32 }] alloca (pre-fix size); full IR snippet:\n%s",
			extractAllocaLines(ir))
	}
}

// extractAllocaLines returns lines from ir that contain "alloca [" for
// diagnostic output in test failures.
func extractAllocaLines(ir string) string {
	var lines []string
	for _, line := range strings.Split(ir, "\n") {
		if strings.Contains(line, "alloca [") {
			lines = append(lines, line)
		}
	}
	if len(lines) == 0 {
		return "(no alloca [ lines found)"
	}
	return strings.Join(lines, "\n")
}

// spmdInnerLoopTargets are the target configurations exercised by the
// inner-loop go for lowering regression tests.
var spmdInnerLoopTargets = []struct {
	name string
	opts compileopts.Options
}{
	{"sse", compileopts.Options{GOOS: "linux", GOARCH: "amd64", LLVMFeatures: "+ssse3,+sse4.2", GOExperiment: "spmd"}},
	{"avx2", compileopts.Options{GOOS: "linux", GOARCH: "amd64", LLVMFeatures: "+ssse3,+sse4.2,+avx2", GOExperiment: "spmd"}},
	{"wasm-simd", compileopts.Options{Target: "wasi", GOExperiment: "spmd"}},
	{"wasm-scalar", compileopts.Options{Target: "wasi", GOExperiment: "spmd", SIMD: "false"}},
}

// compileSPMDSourceVerified compiles src for opts and fails the test if the
// resulting module does not pass LLVM verification.
func compileSPMDSourceVerified(t *testing.T, src string, opts compileopts.Options) string {
	t.Helper()
	ir, errs := compileSPMDSourceErrors(t, src, opts)
	if len(errs) > 0 {
		t.Fatalf("compile errors:\n%s", strings.Join(errs, "\n"))
	}
	return ir
}

// compileSPMDSourceErrors compiles src for opts and returns the IR and the
// compile errors. It fails the test if a module without compile errors does
// not pass LLVM verification.
func compileSPMDSourceErrors(t *testing.T, src string, opts compileopts.Options) (string, []string) {
	t.Helper()
	t.Setenv("GOEXPERIMENT", "spmd")
	f, err := os.CreateTemp("./testdata", "spmd_test_*.go")
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

	mod, errs := testCompilePackage(t, &opts, strings.TrimPrefix(name, "./testdata/"))
	if len(errs) > 0 {
		msgs := make([]string, len(errs))
		for i, e := range errs {
			msgs[i] = e.Error()
		}
		return "", msgs
	}
	ir := mod.String()
	if err := llvm.VerifyModule(mod, llvm.ReturnStatusAction); err != nil {
		t.Fatalf("LLVM verification failed: %v", err)
	}
	return ir, nil
}

// spmdFuncIR returns the body of the LLVM function whose name contains name.
func spmdFuncIR(t *testing.T, ir, name string) string {
	t.Helper()
	start := strings.Index(ir, "@"+name+"(")
	if start < 0 {
		t.Fatalf("function %s not found in IR", name)
	}
	end := strings.Index(ir[start:], "\n}\n")
	if end < 0 {
		t.Fatalf("function %s has no end in IR", name)
	}
	return ir[start : start+end]
}

// TestSPMDInnerLoopVaryingAccumulatorVerifies covers a go for body whose
// constant-bound inner loop ORs table lookups into a varying accumulator.
func TestSPMDInnerLoopVaryingAccumulatorVerifies(t *testing.T) {
	src := `package main

import "lanes"

const tbl = "\x00\x01\x00\x02\x00\x04\x00\x08\x00\x10\x00\x20\x00\x40\x00\x80"

var text [32 * 64]byte
var sum [64]uint32

func kernel() {
	go for w := range 64 {
		var acc lanes.Varying[uint32]
		for j := range 32 {
			c := tbl[text[32*w+j]&15]
			acc = acc | lanes.Varying[uint32](c)
		}
		sum[w] = acc
	}
}
`
	for _, tc := range spmdInnerLoopTargets {
		t.Run(tc.name, func(t *testing.T) {
			ir := compileSPMDSourceVerified(t, src, tc.opts)
			if tc.opts.SIMD == "false" {
				return
			}
			// The outer w loop is the vectorized one: sum[w] is written through
			// the contiguous pointer of the loop index, not a per-lane scatter.
			fn := spmdFuncIR(t, ir, "main.kernel")
			mustContain(t, fn, "spmd.contiguous.ptr")
			mustNotContain(t, fn, "masked.scatter")
		})
	}
}

// TestSPMDInnerLoopOwnElementAccumulationRejected covers an inner loop that
// accumulates straight into sum[w]. The go for cannot be peeled, and the
// unpeeled lowering does not mask the inner loop's memory accesses, so SIMD
// builds must fail closed; the scalar build is still correct.
func TestSPMDInnerLoopOwnElementAccumulationRejected(t *testing.T) {
	src := `package main

import "lanes"

const tbl = "\x00\x01\x00\x02\x00\x04\x00\x08\x00\x10\x00\x20\x00\x40\x00\x80"

var text [32 * 64]byte
var sum [64]uint32

func kernel() {
	go for w := range 64 {
		for j := range 32 {
			c := tbl[text[32*w+j]&15]
			sum[w] = sum[w] | lanes.Varying[uint32](c)
		}
	}
}
`
	for _, tc := range spmdInnerLoopTargets {
		t.Run(tc.name, func(t *testing.T) {
			_, errs := compileSPMDSourceErrors(t, src, tc.opts)
			if tc.opts.SIMD == "false" {
				if len(errs) > 0 {
					t.Fatalf("scalar build rejected: %v", errs)
				}
				return
			}
			if len(errs) != 1 || !strings.Contains(errs[0], "nested loop that indexes memory by the go for index") {
				t.Fatalf("want nested-loop rejection, got %v", errs)
			}
		})
	}
}

// TestSPMDGoForBodyLeadingInnerLoop covers a go for body whose first
// statement is a counted loop independent of the go for index. The go for
// body block then has no positioned instruction, which used to let the inner
// loop claim the SPMD loop and be vectorized instead of the go for.
func TestSPMDGoForBodyLeadingInnerLoop(t *testing.T) {
	src := `package main

import "lanes"

var sum [67]uint32
var scratch [4]byte

func kernel(n int) {
	go for w := range n {
		for j := range 4 {
			scratch[j] = byte(j)
		}
		sum[w] = lanes.Varying[uint32](w) * 3
	}
}
`
	for _, tc := range spmdInnerLoopTargets {
		t.Run(tc.name, func(t *testing.T) {
			ir := compileSPMDSourceVerified(t, src, tc.opts)
			if tc.opts.SIMD == "false" {
				return
			}
			fn := spmdFuncIR(t, ir, "main.kernel")
			mustContain(t, fn, "store i8")
			mustContain(t, fn, "spmd.contiguous.ptr")
			mustNotContain(t, fn, "masked.scatter")
		})
	}
}

// TestSPMDScatterValueWidth checks that the plain *ssa.Store scatter path
// reconciles the value width with the pointer vector width, so the masked
// scatter intrinsic always verifies.
func TestSPMDScatterValueWidth(t *testing.T) {
	tests := []struct {
		name     string
		valLanes int // 0 means a scalar value
		ptrLanes int
	}{
		{"scalar", 0, 4},
		{"same", 4, 4},
		{"narrow", 2, 4},
		{"wide", 8, 4},
		{"narrow-avx2", 4, 8},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := newTestCompilerContext(t)
			defer c.dispose()
			b := newTestBuilder(t, c)
			defer b.Dispose()

			i32 := c.ctx.Int32Type()
			var val llvm.Value
			if tc.valLanes == 0 {
				val = llvm.ConstInt(i32, 7, false)
			} else {
				elems := make([]llvm.Value, tc.valLanes)
				for i := range elems {
					elems[i] = llvm.ConstInt(i32, uint64(i), false)
				}
				val = llvm.ConstVector(elems, false)
			}
			ptrType := llvm.VectorType(c.dataPtrType, tc.ptrLanes)
			ptrs := b.splatScalar(b.CreateAlloca(llvm.ArrayType(i32, 8), "dst"), ptrType)
			if ptrs.Type() != ptrType {
				t.Fatalf("ptrs type = %s, want %s", ptrs.Type(), ptrType)
			}

			got := b.spmdScatterValue(val, tc.ptrLanes)
			if want := llvm.VectorType(i32, tc.ptrLanes); got.Type() != want {
				t.Fatalf("value type = %s, want %s", got.Type(), want)
			}
			mask := llvm.ConstAllOnes(llvm.VectorType(b.spmdMaskElemType(tc.ptrLanes), tc.ptrLanes))
			b.spmdMaskedScatter(got, ptrs, mask)
			b.CreateRetVoid()
			if err := llvm.VerifyModule(c.mod, llvm.ReturnStatusAction); err != nil {
				t.Errorf("invalid IR: %v\n%s", err, c.mod.String())
			}
		})
	}
}
