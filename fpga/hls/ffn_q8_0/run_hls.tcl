# Use the runner to preserve inputs and avoid overwriting a previous project.
foreach name {NTRPU_HLS_PART NTRPU_HLS_CLOCK_NS NTRPU_HLS_STAGE} {
    if {![info exists ::env($name)] || $::env($name) eq ""} { error "Missing $name" }
}
set stage $::env(NTRPU_HLS_STAGE)
if {$stage ni {csim synth cosim export}} { error "Invalid stage" }
set clock_ns $::env(NTRPU_HLS_CLOCK_NS)
if {![string is double -strict $clock_ns] || !($clock_ns > 0 && $clock_ns < 1000)} {
    error "Invalid clock period"
}
set source_dir [file dirname [file normalize [info script]]]
if {[file exists project]} { error "Refusing to overwrite an existing HLS project" }
open_project project
set_top ntrpu_ffn_q8_0
set flags "-std=c++14 -DNTRPU_USE_HLS_MATH"
foreach name {ntrpu_ffn_q8_0.cpp ntrpu_gemv_q8_0.cpp ntrpu_quantize_q8_0.cpp ntrpu_swiglu.cpp} {
    add_files [file join $source_dir $name] -cflags $flags
}
add_files -tb [file join $source_dir test_ntrpu_hls_ffn.cpp] -cflags $flags
set test_args ""
set test_mode full
if {[info exists ::env(NTRPU_HLS_TEST_MODE)]} { set test_mode $::env(NTRPU_HLS_TEST_MODE) }
if {$test_mode ni {full smoke}} { error "Invalid FFN test mode" }
if {$test_mode eq "smoke"} { set test_args "smoke" }
if {[info exists ::env(NTRPU_HLS_FIXTURE)] && $::env(NTRPU_HLS_FIXTURE) eq "1"} {
    if {$test_mode ne "full"} { error "FFN smoke excludes real-weight fixtures; use full" }
    set fixture [file join $source_dir brandon_ffn.bin]
    add_files -tb $fixture
    set test_args [list --all-fixture $fixture]
}
open_solution -flow_target vivado solution1
set_part $::env(NTRPU_HLS_PART)
create_clock -period $clock_ns -name default
config_interface -m_axi_addr64=true
csim_design -argv $test_args
if {$stage in {synth cosim export}} { csynth_design }
if {$stage in {cosim export}} { cosim_design -rtl verilog -argv $test_args }
if {$stage eq "export"} { export_design -format ip_catalog -rtl verilog }
close_project
exit
