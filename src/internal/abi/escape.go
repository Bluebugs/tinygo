package abi

import "unsafe"

// Tell the compiler the given pointer doesn't escape.
// The compiler knows about this function and will give the nocapture parameter
// attribute.
func NoEscape(p unsafe.Pointer) unsafe.Pointer {
	return p
}

func Escape[T any](x T) T {
	// This function is used from internal/synctest which doesn't seem to be
	// used much by other packages.
	// This probably needs support from the compiler to implement correctly.
	panic("internal/abi.Escape: unimplemented")
}
