# Full Q8 layer0 FFN; separate bench, unchanged device RTL and HLS cores.
set script_dir [file dirname [file normalize [info script]]]
proc llama32_ffn_residual_real {arguments} {
    global script_dir
    if {[llength $arguments]!=5 || [llength [get_projects -quiet]]} {error "Expected MAC_HLS SWIGLU_HLS OUTPUT FIXTURE EXTRA"}
    lassign $arguments mac swiglu output fixture extra
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project llama32_ffn_residual_real [file join $output project] -part xcvu5p-flva2104-2-i
    foreach input [list $mac $swiglu] kernel {q8_block_mac swiglu_q8_block} {
        set input [file normalize $input]
        set f [open [file join $input manifest.json] r];set text [read $f];close $f
        foreach pattern [list {*"result": "tool_succeeded"*} "*\"kernel\": \"$kernel\"*" {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}] {
            if {![string match $pattern $text]} {error "Pinned HLS manifest mismatch"}
        }
        set rtl [file join $output ${kernel}_rtl]
        file copy [file join $input project solution1 syn verilog] $rtl
        file copy [file join $input manifest.json] [file join $output ${kernel}_manifest.json]
        cd $rtl
        add_files [glob *.v]
        set data [glob -nocomplain *.dat]
        if {[llength $data]} {add_files $data}
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    foreach name {ntrpu_fp64_to_fp32 ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream ntrpu_q8_gemv_parallel ntrpu_activation_loader ntrpu_q8_gemv_memory ntrpu_result_writer ntrpu_q8_gemv_store ntrpu_matrix_sequencer ntrpu_q8_matrix_memory ntrpu_activation_writer ntrpu_swiglu_memory ntrpu_ffn_sequencer ntrpu_ffn_memory ntrpu_llama32_ffn_owner_r3 ntrpu_llama32_owned_ffn_r1 ntrpu_residual_loader_r1 ntrpu_residual_owner_r1 ntrpu_residual_store_r1 ntrpu_residual_path_r1 ntrpu_residual_real_r1 ntrpu_residual_hls_ports_r1 ntrpu_ffn_residual_handoff_r1 ntrpu_llama32_ffn_residual_r3} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    set residual_generated [glob [file normalize [file join $script_dir .. generated *.v]]]
    if {[llength $residual_generated]!=9} {error "Expected nine original residual HLS files"}
    add_files $residual_generated
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    synth_design -top ntrpu_llama32_ffn_residual_r3 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output ffn_residual_real_synth.dcp]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    set f [open [file join $output primitive_counts.tsv] {WRONLY CREAT EXCL}]
    puts $f "top\t[get_property NAME [current_design]]"
    puts $f "part\t[get_property PART [current_design]]"
    puts $f "clock_period\t[get_property PERIOD [get_clocks clk]]"
    puts $f "blackboxes\t[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    foreach ref {LUT1 LUT2 LUT3 LUT4 LUT5 LUT6 FDRE FDSE FDCE FDPE LDCE LDPE CARRY8 RAMB36E2 RAMB18E2 DSP48E2} {
        puts $f "$ref\t[llength [get_cells -quiet -hierarchical -filter "REF_NAME == $ref"]]"
    }
    close $f
    set f [open [file join $output check_timing.rpt] r];set checks [read $f];close $f
    if {![regexp {checking loops \(([0-9]+)\)} $checks all loops] || $loops!=0} {error "Combinational loop preflight failed"}
    puts NTRPU_FFN_RESIDUAL_REAL_R1_SYNTH_PREFLIGHT_EXPORTED
    close_design
    file copy [file normalize $fixture] [file join $output fixture]
    file copy [file normalize $extra] [file join $output extra]
    add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_llama32_ffn_residual_real_r1.sv]]
    set_property top ntrpu_llama32_ffn_residual_r3 [get_filesets sources_1]
    set_property top tb_ntrpu_llama32_ffn_residual_real_r1 [get_filesets sim_1]
    set_property verilog_define {NTRPU_FFN_FP32 NTRPU_FFN_FIXTURE NTRPU_FFN_LLAMA_REAL} [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=1 -testplusarg FIXTURE=[file join $output fixture] -testplusarg EXTRA=[file join $output extra]"] [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    report_ip_status -file [file join $output ip_status.rpt]
    puts NTRPU_FFN_RESIDUAL_REAL_R1_START
    launch_simulation
    run all
    close_sim
    set logfile [file join $output project llama32_ffn_residual_real.sim sim_1 behav xsim simulate.log]
    file copy $logfile [file join $output seed-1.log]
    set f [open $logfile r];set text [read $f];close $f
    if {![regexp {(?m)^FFRREAL_PASS jobs=1 seed=1 .* bytes=163840 } $text] ||
        [string first "FFRREAL_CONFIG dim=2048 hidden=8192 tile_k=32 weight_bytes=53477376 input_abi_requantized=0" $text]<0 ||
        [string first "FFRREAL_REUSABLE jobs=1" $text]<0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Full actual FFN/residual composition did not pass"}
    close_project
    puts NTRPU_FFN_RESIDUAL_REAL_R1_EXPORTED
}
if {[catch {llama32_ffn_residual_real $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
