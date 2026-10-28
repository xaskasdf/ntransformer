foreach name {NTRPU_HLS_PART NTRPU_HLS_CLOCK_NS NTRPU_HLS_STAGE} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {error "Missing $name"}
}
if {$::env(NTRPU_HLS_STAGE) ni {csim synth cosim}} {error "Invalid RMSNorm stage"}
if {$::env(NTRPU_HLS_PART) ne "xcvu5p-flva2104-2-i" || $::env(NTRPU_HLS_CLOCK_NS) != 5} {error "Unqualified part/clock"}
set source_dir [file dirname [file normalize [info script]]]
if {[file exists project]} {error "Never overwrite an existing project"}
open_project project
set_top ntrpu_rmsnorm_fp32_r1
set flags "-std=c++14 -ffp-contract=off -DNTRPU_USE_HLS_MATH"
add_files [file join $source_dir ntrpu_rmsnorm_fp32_r1.cpp] -cflags $flags
add_files -tb [file join $source_dir test_ntrpu_hls_rmsnorm_r1.cpp] -cflags $flags
add_files -tb [file join $source_dir llama32_norm_r1.bin]
open_solution -flow_target vivado solution1
set_part $::env(NTRPU_HLS_PART)
create_clock -period 5 -name default
config_compile -unsafe_math_optimizations false
csim_design -argv llama32_norm_r1.bin
if {$::env(NTRPU_HLS_STAGE) in {synth cosim}} {csynth_design}
if {$::env(NTRPU_HLS_STAGE) eq "cosim"} {cosim_design -rtl verilog -argv llama32_norm_r1.bin}
close_project
puts NTRPU_RMSNORM_R1_EXPORTED
exit
