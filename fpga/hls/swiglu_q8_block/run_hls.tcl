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
set_top ntrpu_swiglu_q8_block
# Same AMD math implementation in C simulation and RTL; host CMake uses std::exp.
set flags "-std=c++14 -DNTRPU_USE_HLS_MATH"
foreach name {ntrpu_swiglu_q8_block ntrpu_swiglu ntrpu_quantize_q8_0} {
    add_files [file join $source_dir ${name}.cpp] -cflags $flags
}
add_files -tb [file join $source_dir test_ntrpu_hls_swiglu_block.cpp] -cflags $flags
open_solution -flow_target vivado solution1
set_part $::env(NTRPU_HLS_PART)
create_clock -period $clock_ns -name default
csim_design
if {$stage in {synth cosim export}} { csynth_design }
if {$stage in {cosim export}} { cosim_design -rtl verilog }
if {$stage eq "export"} { export_design -format ip_catalog -rtl verilog }
close_project
exit
