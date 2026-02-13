package compiler

// This file extracts SPMD metadata from the typed AST for use during LLVM IR
// generation. TinyGo uses golang.org/x/tools/go/ssa which has no SPMD support,
// so we build a side-table from go/ast and go/types (which our Go fork extends).

import (
	"go/ast"
	"go/token"
	"go/types"
	"sort"
	"strconv"

	"github.com/tinygo-org/tinygo/loader"
	"golang.org/x/tools/go/ssa"
)

// SPMDLoopInfo holds metadata about a go for loop extracted from the AST.
type SPMDLoopInfo struct {
	ForPos     token.Pos // position of "for" keyword
	BodyStart  token.Pos // start of loop body (opening brace)
	BodyEnd    token.Pos // end of loop body (closing brace)
	LaneCount  int64     // SIMD lane count (from type checker)
	Constraint int64     // range[N] constraint; -1 if unconstrained
}

// SPMDParamInfo holds info about a single varying parameter.
type SPMDParamInfo struct {
	Index      int
	ElemType   types.Type
	Constraint int64 // -1 if unconstrained
}

// SPMDFuncInfo holds metadata about a function with varying parameters.
type SPMDFuncInfo struct {
	HasVaryingParams  bool
	HasVaryingResults bool
	VaryingParams     []SPMDParamInfo
}

// SPMDInfo holds aggregated SPMD metadata for a package.
type SPMDInfo struct {
	Loops      map[token.Pos]*SPMDLoopInfo // keyed by ForPos
	Funcs      map[*types.Func]*SPMDFuncInfo
	loopRanges []spmdPosRange // sorted by Start for binary search
}

// spmdPosRange is a position range for "inside SPMD loop?" queries.
type spmdPosRange struct {
	start, end token.Pos
	info       *SPMDLoopInfo
}

// extractSPMDLoops walks the AST and extracts all SPMD loop metadata.
// Closures inside SPMD loops are handled by position: their instructions
// have positions inside the loop body, so isInSPMDLoop works for them.
func extractSPMDLoops(pkg *loader.Package) map[token.Pos]*SPMDLoopInfo {
	loops := make(map[token.Pos]*SPMDLoopInfo)

	for _, file := range pkg.Files {
		ast.Inspect(file, func(n ast.Node) bool {
			rangeStmt, ok := n.(*ast.RangeStmt)
			if !ok {
				return true
			}

			// Check if this is an SPMD loop (go for).
			// The IsSpmd field is set by our Go fork's parser.
			if !rangeStmt.IsSpmd {
				return true
			}

			// Extract constraint if present.
			constraint := int64(-1) // -1 means unconstrained
			if rangeStmt.Constraint != nil {
				if basicLit, ok := rangeStmt.Constraint.(*ast.BasicLit); ok {
					if basicLit.Kind == token.INT {
						if val, err := strconv.ParseInt(basicLit.Value, 10, 64); err == nil {
							constraint = val
						}
					}
				}
			}

			// Create loop info.
			info := &SPMDLoopInfo{
				ForPos:     rangeStmt.For,
				BodyStart:  rangeStmt.Body.Lbrace,
				BodyEnd:    rangeStmt.Body.Rbrace,
				LaneCount:  rangeStmt.LaneCount, // set by type checker
				Constraint: constraint,
			}

			loops[rangeStmt.For] = info
			return true
		})
	}

	return loops
}

// extractSPMDFuncs walks the package scope and extracts functions with varying
// parameters or results.
func extractSPMDFuncs(pkg *loader.Package) map[*types.Func]*SPMDFuncInfo {
	funcs := make(map[*types.Func]*SPMDFuncInfo)

	// Process top-level functions.
	scope := pkg.Pkg.Scope()
	for _, name := range scope.Names() {
		obj := scope.Lookup(name)
		if fn, ok := obj.(*types.Func); ok {
			if info := analyzeSPMDSignature(fn.Type().(*types.Signature)); info != nil {
				funcs[fn] = info
			}
		}

		// Also check methods on named types.
		if typeName, ok := obj.(*types.TypeName); ok {
			if namedType, ok := typeName.Type().(*types.Named); ok {
				for i := 0; i < namedType.NumMethods(); i++ {
					method := namedType.Method(i)
					if info := analyzeSPMDSignature(method.Type().(*types.Signature)); info != nil {
						funcs[method] = info
					}
				}
			}
		}
	}

	return funcs
}

// analyzeSPMDSignature checks if a function signature has varying parameters
// or results. Returns nil if the function is not an SPMD function.
func analyzeSPMDSignature(sig *types.Signature) *SPMDFuncInfo {
	info := &SPMDFuncInfo{}

	// Check parameters.
	params := sig.Params()
	for i := 0; i < params.Len(); i++ {
		param := params.At(i)
		if spmdType, ok := param.Type().(*types.SPMDType); ok {
			if spmdType.IsVarying() {
				info.HasVaryingParams = true
				info.VaryingParams = append(info.VaryingParams, SPMDParamInfo{
					Index:      i,
					ElemType:   spmdType.Elem(),
					Constraint: spmdType.Constraint(),
				})
			}
		}
	}

	// Check results.
	results := sig.Results()
	for i := 0; i < results.Len(); i++ {
		result := results.At(i)
		if spmdType, ok := result.Type().(*types.SPMDType); ok {
			if spmdType.IsVarying() {
				info.HasVaryingResults = true
			}
		}
	}

	// Return nil if no SPMD characteristics found.
	if !info.HasVaryingParams && !info.HasVaryingResults {
		return nil
	}

	return info
}

// loadSPMDInfo extracts SPMD metadata from a package and stores it in the
// compiler context. If the package contains no SPMD code, spmdInfo remains nil.
func (c *compilerContext) loadSPMDInfo(pkg *loader.Package) {
	loops := extractSPMDLoops(pkg)
	funcs := extractSPMDFuncs(pkg)

	// If no SPMD code, leave spmdInfo as nil.
	if len(loops) == 0 && len(funcs) == 0 {
		return
	}

	// Build the SPMDInfo structure.
	info := &SPMDInfo{
		Loops: loops,
		Funcs: funcs,
	}

	// Build sorted position ranges for binary search.
	info.loopRanges = make([]spmdPosRange, 0, len(loops))
	for _, loopInfo := range loops {
		info.loopRanges = append(info.loopRanges, spmdPosRange{
			start: loopInfo.BodyStart,
			end:   loopInfo.BodyEnd,
			info:  loopInfo,
		})
	}

	// Sort by start position for binary search.
	sort.Slice(info.loopRanges, func(i, j int) bool {
		return info.loopRanges[i].start < info.loopRanges[j].start
	})

	c.spmdInfo = info
}

// isInSPMDLoop checks if a given position is inside an SPMD loop body.
// Returns the loop info if found, nil otherwise.
func (c *compilerContext) isInSPMDLoop(pos token.Pos) *SPMDLoopInfo {
	if c.spmdInfo == nil {
		return nil
	}

	// Binary search through sorted ranges.
	idx := sort.Search(len(c.spmdInfo.loopRanges), func(i int) bool {
		return c.spmdInfo.loopRanges[i].start > pos
	})

	// Check the range before the found index (if it exists).
	if idx > 0 {
		r := &c.spmdInfo.loopRanges[idx-1]
		if r.start <= pos && pos <= r.end {
			return r.info
		}
	}

	return nil
}

// getSPMDLoopAt returns the SPMD loop info for a loop starting at the given
// position (the "for" keyword position). Returns nil if not found.
func (c *compilerContext) getSPMDLoopAt(forPos token.Pos) *SPMDLoopInfo {
	if c.spmdInfo == nil {
		return nil
	}
	return c.spmdInfo.Loops[forPos]
}

// getSPMDFuncInfo returns the SPMD function info for a given function.
// Returns nil if the function is not an SPMD function.
func (c *compilerContext) getSPMDFuncInfo(fn *types.Func) *SPMDFuncInfo {
	if c.spmdInfo == nil {
		return nil
	}
	return c.spmdInfo.Funcs[fn]
}

// isSPMDFunction checks if an SSA function is an SPMD function (has varying
// parameters or results).
func (c *compilerContext) isSPMDFunction(fn *ssa.Function) bool {
	if c.spmdInfo == nil {
		return false
	}

	// Try to get the function object.
	if obj := fn.Object(); obj != nil {
		if typesFn, ok := obj.(*types.Func); ok {
			return c.spmdInfo.Funcs[typesFn] != nil
		}
	}

	// Fallback: analyze the signature directly for cross-package calls.
	return analyzeSPMDSignature(fn.Signature) != nil
}

// hasSPMDCode returns true if the current package contains any SPMD code
// (loops or functions).
func (c *compilerContext) hasSPMDCode() bool {
	return c.spmdInfo != nil
}
