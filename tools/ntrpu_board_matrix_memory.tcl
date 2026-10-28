# Full-matrix sequencer plus the preserved memory GEMV and real HLS MAC.
set script_dir [file dirname [file normalize [info script]]]
proc matrix_memory {arguments} {
    global script_dir
    if {[llength $arguments] ni {3 4 5}} { error "Expected HLS_MAC_RUN sim|synth NEW_DIRECTORY ?FIXTURE? ?LARGE_JOB_ID?" }
    lassign $arguments input mode output fixture selected
    if {$mode ni {sim synth}} { error "Invalid mode" }
    if {$fixture ne "" && $mode ne "sim"} { error "Fixture only supported in simulation" }
    if {$selected ne "" && ($selected ni {1 2} || $fixture eq "")} { error "Large fixture job must be 1 or 2 with a fixture" }
    set input [file normalize $input];set output [file normalize $output]
    if {[file exists $output]} { error "Output exists" }
    set f [open [file join $input manifest.json] r];set text [read $f];close $f
    foreach pattern {{*"result": "tool_succeeded"*} {*"kernel": "q8_block_mac"*} {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}} {
        if {![string match $pattern $text]} { error "MAC manifest mismatch" }
    }
    file mkdir $output
    file copy [file join $input project solution1 syn verilog] [file join $output mac_rtl]
    file copy [file join $input manifest.json] [file join $output input_hls_manifest.json]
    file copy [file join $input project solution1 sim report ntrpu_q8_block_mac_cosim.rpt] [file join $output input_cosim.rpt]
    create_project matrix_memory [file join $output project] -part xcvu5p-flva2104-2-i
    cd [file join $output mac_rtl]
    add_files [glob *.v]
    foreach script [lsort [glob *_ip.tcl]] { source $script }
    foreach name {ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream ntrpu_q8_gemv_parallel ntrpu_activation_loader ntrpu_q8_gemv_memory ntrpu_result_writer ntrpu_q8_gemv_store ntrpu_matrix_sequencer ntrpu_q8_matrix_memory} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    set_property top ntrpu_q8_matrix_memory [get_filesets sources_1]
    if {$mode eq "sim"} {
        set cases 10;set extra {};set seeds {1 7 42}
        if {$fixture ne ""} {
            if {![file isfile [file join $fixture manifest.json]]} { error "Fixture manifest missing" }
            file copy [file normalize $fixture] [file join $output fixture]
            set extra " -testplusarg FIXTURE=[file join $output fixture]";set cases 3
            set_property verilog_define NTRPU_MATRIX_FIXTURE [get_filesets sim_1]
            if {$selected ne ""} {
                set cases 1;set seeds {1}
                append extra " -testplusarg FIXTURE_JOB=$selected"
                set_property verilog_define {NTRPU_MATRIX_FIXTURE NTRPU_MATRIX_LARGE} [get_filesets sim_1]
            }
        }
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_q8_matrix_memory.sv]]
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb ntrpu_result_memory_model.sv]]
        set_property top tb_ntrpu_q8_matrix_memory [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach seed $seeds {
            set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed$extra"] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project matrix_memory.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output seed-${seed}.log]
            set f [open $log r];set text [read $f];close $f
            if {![string match "*NTRPU_MATRIX_MEMORY_PASS cases=$cases *" $text] ||
                ![string match "*NTRPU_MATRIX_MEMORY_SEED seed=$seed\n*" $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} { error "Memory GEMV failed seed=$seed" }
        }
        puts NTRPU_MATRIX_MEMORY_SUITE_RETURNED
    } else {
        set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
        read_xdc [file join $output clock.xdc]
        synth_design -top ntrpu_q8_matrix_memory -mode out_of_context -part xcvu5p-flva2104-2-i
        write_checkpoint [file join $output matrix_memory_synth.dcp]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} { error "Unresolved black boxes" }
        puts NTRPU_MATRIX_MEMORY_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {matrix_memory $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
