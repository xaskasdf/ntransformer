foreach name {NTRPU_HLS_PART NTRPU_HLS_CLOCK_NS NTRPU_HLS_STAGE} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {error "Missing $name"}
}
if {$::env(NTRPU_HLS_STAGE) ne "cosim"} {error "Attention quantization requires all stages"}
if {$::env(NTRPU_HLS_PART) ne "xcvu5p-flva2104-2-i" || $::env(NTRPU_HLS_CLOCK_NS) != 5} {error "Unqualified part/clock"}
set source_dir [file dirname [file normalize [info script]]]
if {[file exists project]} {error "Never overwrite an existing project"}
open_project project
set_top ntrpu_quantize_abi1_r1
set flags "-std=c++14 -ffp-contract=off -fno-fast-math"
foreach name {ntrpu_quantize_abi1_r1.cpp ntrpu_quantize_q8_0.cpp} {
    add_files [file join $source_dir $name] -cflags $flags
}
add_files -tb [file join $source_dir test_ntrpu_hls_attention_quantize_r1.cpp] -cflags $flags
add_files -tb [file join $source_dir attention_quantize_r1.bin]
open_solution -flow_target vivado solution1
set_part $::env(NTRPU_HLS_PART)
create_clock -period 5 -name default
config_compile -unsafe_math_optimizations=false
csim_design -argv attention_quantize_r1.bin
csynth_design
cosim_design -rtl verilog -argv attention_quantize_r1.bin
close_project
puts NTRPU_ATTENTION_QUANTIZE_R1_EXPORTED
exit
