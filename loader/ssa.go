package loader

import (
	"golang.org/x/tools/go/ssa"
)

// LoadSSA constructs the SSA form of the loaded packages.
//
// The program must already be parsed and type-checked with the .Parse() method.
func (p *Program) LoadSSA() *ssa.Program {
	// TODO: re-enable SanityCheckFunctions when x/tools is upgraded to
	// a version with a fix for https://golang.org/issues/73594.
	prog := ssa.NewProgram(p.fset /*ssa.SanityCheckFunctions|*/, ssa.BareInits|ssa.GlobalDebug|ssa.InstantiateGenerics)

	// Propagate the SIMD register width so SSA-level helpers (lane count
	// computation, gather stride, promote size check) use the same value as
	// the type checker instead of the hardcoded 128-bit default.
	if p.typeChecker.SIMDRegisterSize > 0 {
		prog.SIMDRegisterBits = int(p.typeChecker.SIMDRegisterSize) * 8
	}

	for _, pkg := range p.sorted {
		prog.CreatePackage(pkg.Pkg, pkg.Files, &pkg.info, true)
	}

	return prog
}

// LoadSSA constructs the SSA form of this package.
//
// The program must already be parsed and type-checked with the .Parse() method.
func (p *Package) LoadSSA() *ssa.Package {
	prog := ssa.NewProgram(p.program.fset, ssa.SanityCheckFunctions|ssa.BareInits|ssa.GlobalDebug)
	return prog.CreatePackage(p.Pkg, p.Files, &p.info, true)
}
