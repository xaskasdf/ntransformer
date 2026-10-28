# Launched inside a fresh, recorded run directory by scripts/run_ntrpu_hls.py.
foreach name {NTRPU_HLS_PART NTRPU_HLS_CLOCK_NS NTRPU_HLS_STAGE} {
    if {![info exists ::env($name)] || $::env($name) eq ""} {
        error "Missing $name; use scripts/run_ntrpu_hls.py"
    }
}
set stage $::env(NTRPU_HLS_STAGE)
if {$stage ni {csim synth cosim export}} { error "Invalid stage: $stage" }
set clock_ns $::env(NTRPU_HLS_CLOCK_NS)
if {![string is double -strict $clock_ns] || !($clock_ns > 0 && $clock_ns < 1000)} {
    error "Invalid clock period"
}
set source_dir [file dirname [file normalize [info script]]]
if {[file exists project]} { error "Refusing to overwrite an existing HLS project" }

open_project project
set_top ntrpu_gemv_int8
add_files [file join $source_dir ntrpu_gemv_int8.cpp] -cflags "-std=c++14"
add_files -tb [file join $source_dir test_ntrpu_hls.cpp] -cflags "-std=c++14"
open_solution -flow_target vivado solution1
set_part $::env(NTRPU_HLS_PART)
create_clock -period $clock_ns -name default
config_interface -m_axi_addr64=true

# Every stronger stage first runs the preceding checks on these exact sources.
csim_design
if {$stage in {synth cosim export}} { csynth_design }
if {$stage in {cosim export}} { cosim_design -rtl verilog }
if {$stage eq "export"} { export_design -format ip_catalog -rtl verilog }
close_project
exit
