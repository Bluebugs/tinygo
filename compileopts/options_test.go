package compileopts_test

import (
	"errors"
	"testing"

	"github.com/tinygo-org/tinygo/compileopts"
)

func TestVerifyOptions(t *testing.T) {

	expectedGCError := errors.New(`invalid gc option 'incorrect': valid values are none, leaking, conservative, custom, precise, boehm`)
	expectedSchedulerError := errors.New(`invalid scheduler option 'incorrect': valid values are none, tasks, asyncify, threads, cores`)
	expectedPrintSizeError := errors.New(`invalid size option 'incorrect': valid values are none, short, full, html`)
	expectedPanicStrategyError := errors.New(`invalid panic option 'incorrect': valid values are print, trap`)
	expectedGPUError := errors.New(`invalid -gpu=metal: valid values are none, webgpu`)

	testCases := []struct {
		name          string
		opts          compileopts.Options
		expectedError error
	}{
		{
			name: "OptionsEmpty",
			opts: compileopts.Options{},
		},
		{
			name: "InvalidGCOption",
			opts: compileopts.Options{
				GC: "incorrect",
			},
			expectedError: expectedGCError,
		},
		{
			name: "GCOptionNone",
			opts: compileopts.Options{
				GC: "none",
			},
		},
		{
			name: "GCOptionLeaking",
			opts: compileopts.Options{
				GC: "leaking",
			},
		},
		{
			name: "GCOptionConservative",
			opts: compileopts.Options{
				GC: "conservative",
			},
		},
		{
			name: "GCOptionCustom",
			opts: compileopts.Options{
				GC: "custom",
			},
		},
		{
			name: "InvalidSchedulerOption",
			opts: compileopts.Options{
				Scheduler: "incorrect",
			},
			expectedError: expectedSchedulerError,
		},
		{
			name: "SchedulerOptionNone",
			opts: compileopts.Options{
				Scheduler: "none",
			},
		},
		{
			name: "SchedulerOptionTasks",
			opts: compileopts.Options{
				Scheduler: "tasks",
			},
		},
		{
			name: "InvalidPrintSizeOption",
			opts: compileopts.Options{
				PrintSizes: "incorrect",
			},
			expectedError: expectedPrintSizeError,
		},
		{
			name: "PrintSizeOptionNone",
			opts: compileopts.Options{
				PrintSizes: "none",
			},
		},
		{
			name: "PrintSizeOptionShort",
			opts: compileopts.Options{
				PrintSizes: "short",
			},
		},
		{
			name: "PrintSizeOptionFull",
			opts: compileopts.Options{
				PrintSizes: "full",
			},
		},
		{
			name: "InvalidPanicOption",
			opts: compileopts.Options{
				PanicStrategy: "incorrect",
			},
			expectedError: expectedPanicStrategyError,
		},
		{
			name: "PanicOptionPrint",
			opts: compileopts.Options{
				PanicStrategy: "print",
			},
		},
		{
			name: "PanicOptionTrap",
			opts: compileopts.Options{
				PanicStrategy: "trap",
			},
		},
		{
			name: "InvalidGPUOption",
			opts: compileopts.Options{
				GPU: "metal",
			},
			expectedError: expectedGPUError,
		},
		{
			name: "GPUOptionNone",
			opts: compileopts.Options{
				GPU: "none",
			},
		},
		{
			name: "GPUOptionWebgpu",
			opts: compileopts.Options{
				GPU: "webgpu",
			},
		},
		{
			name:          "InvalidGPUHostOption",
			opts:          compileopts.Options{GPUHost: "metal"},
			expectedError: errors.New("invalid -gpu-host=metal: valid values are browser, vulkan"),
		},
		{
			name: "GPUHostVulkan",
			opts: compileopts.Options{GPU: "webgpu", GPUHost: "vulkan"},
		},
		{
			name: "GPUHostBrowser",
			opts: compileopts.Options{GPUHost: "browser"},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.opts.Verify()
			if tc.expectedError != err {
				if tc.expectedError.Error() != err.Error() {
					t.Errorf("expected %v, got %v", tc.expectedError, err)
				}
			}
		})
	}
}

// TestGPUZeroCopyConfig covers Config.GPUZeroCopy()'s truth table: zero-copy
// is on by default ONLY for the Vulkan host with the Boehm GC (no other host
// can bind Go memory, and the pool needs Boehm), and -gpu-zerocopy=off always
// wins. This is what keeps every existing build unaffected by default.
func TestGPUZeroCopyConfig(t *testing.T) {
	tests := []struct {
		name                    string
		gpu, host, gc, zerocopy string
		want                    bool
	}{
		{"vulkan+boehm default on", "webgpu", "vulkan", "boehm", "", true},
		{"vulkan+boehm explicit on", "webgpu", "vulkan", "boehm", "on", true},
		{"explicit off wins", "webgpu", "vulkan", "boehm", "off", false},
		{"browser host not eligible", "webgpu", "browser", "boehm", "", false},
		{"no host not eligible", "webgpu", "", "boehm", "", false},
		{"no gpu not eligible", "", "vulkan", "boehm", "", false},
		{"non-boehm gc not eligible", "webgpu", "vulkan", "conservative", "", false},
		{"plain build not eligible", "", "", "", "", false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := &compileopts.Config{Options: &compileopts.Options{
				GPU: tc.gpu, GPUHost: tc.host, GC: tc.gc, GPUZeroCopy: tc.zerocopy,
			}}
			if got := c.GPUZeroCopy(); got != tc.want {
				t.Errorf("GPUZeroCopy() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestGPUZeroCopyOption(t *testing.T) {
	if err := (&compileopts.Options{GPUZeroCopy: "off"}).Verify(); err != nil {
		t.Errorf("-gpu-zerocopy=off rejected: %v", err)
	}
	if err := (&compileopts.Options{GPUZeroCopy: "maybe"}).Verify(); err == nil {
		t.Error("-gpu-zerocopy=maybe accepted, want an error")
	}
}
