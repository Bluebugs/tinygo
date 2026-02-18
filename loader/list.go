package loader

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"github.com/tinygo-org/tinygo/compileopts"
	"github.com/tinygo-org/tinygo/goenv"
)

// List returns a ready-to-run *exec.Cmd for running the `go list` command with
// the configuration used for TinyGo.
func List(config *compileopts.Config, extraArgs, pkgs []string) (*exec.Cmd, error) {
	goroot, err := GetCachedGoroot(config)
	if err != nil {
		return nil, err
	}
	args := append([]string{"list"}, extraArgs...)
	if len(config.BuildTags()) != 0 {
		args = append(args, "-tags", strings.Join(config.BuildTags(), " "))
	}
	args = append(args, pkgs...)
	cmd := exec.Command(filepath.Join(goenv.Get("GOROOT"), "bin", "go"), args...)
	cmd.Env = append(os.Environ(), "GOROOT="+goroot, "GOOS="+config.GOOS(), "GOARCH="+config.GOARCH(), "CGO_ENABLED=1")
	// Always set GOEXPERIMENT explicitly: strip "spmd" which the system Go
	// binary used for "go list" doesn't recognize, and override any inherited
	// GOEXPERIMENT from the parent process environment (via os.Environ above).
	if exp := config.GOExperiment(); exp != "" {
		var filtered []string
		for _, e := range strings.Split(exp, ",") {
			if e != "spmd" && e != "nospmd" {
				filtered = append(filtered, e)
			}
		}
		cmd.Env = append(cmd.Env, "GOEXPERIMENT="+strings.Join(filtered, ","))
	} else {
		cmd.Env = append(cmd.Env, "GOEXPERIMENT=")
	}
	if config.Options.Directory != "" {
		cmd.Dir = config.Options.Directory
	}
	return cmd, nil
}
