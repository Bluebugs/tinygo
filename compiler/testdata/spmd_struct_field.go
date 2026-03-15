package main

import "lanes"

// Point is a simple struct for testing field access through *Varying[Point].
type Point struct {
	X int
	Y int
}

// processPoints iterates over a slice of Points and adds the lane index to
// each point's Y field.  This exercises FieldAddr through a *Varying[Point]
// pointer (the SSA FieldAddr result from &points[i]).
func processPoints(points []Point) {
	go for i := range len(points) {
		points[i].Y += lanes.Index()
	}
}
