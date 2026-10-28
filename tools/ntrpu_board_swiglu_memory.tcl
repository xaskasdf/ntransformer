set script_dir [file dirname [file normalize [info script]]]
proc swiglu_memory {arguments} {
    global script_dir
    if {[llength $arguments]!=3} { error "Expected HLS_RUN sim|large|synth NEW_DIRECTORY" }
    lassign $arguments input mode output
    if {$mode ni {sim large synth}} { error "Invalid mode" }
    set input [file normalize $input];set output [file normalize $output]
    if {[file exists $output]} { error "Output exists" }
    set f [open [file join $input manifest.json] r];set text [read $f];close $f
    foreach pattern {{*"result": "tool_succeeded"*} {*"kernel": "swiglu_q8_block"*} {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}} {
        if {![string match $pattern $text]} { error "HLS manifest mismatch" }
    }
    file mkdir $output
    file copy [file join $input project solution1 syn verilog] [file join $output hls_rtl]
    file copy [file join $input manifest.json] [file join $output input_hls_manifest.json]
    create_project swiglu_memory [file join $output project] -part xcvu5p-flva2104-2-i
    cd [file join $output hls_rtl]
    add_files [glob *.v]
    set data [glob -nocomplain *.dat]
    if {[llength $data]} { add_files $data }
    foreach script [lsort [glob *_ip.tcl]] { source $script }
    foreach name {ntrpu_axi_read_master ntrpu_activation_writer ntrpu_swiglu_memory} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    set_property top ntrpu_swiglu_memory [get_filesets sources_1]
    if {$mode ne "synth"} {
        set cases 35;set seeds {1 7 42}
        if {$mode eq "large"} {
            set cases 1;set seeds {1}
            set_property verilog_define NTRPU_SWIGLU_LARGE [get_filesets sim_1]
        }
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_swiglu_memory.sv]]
        set_property top tb_ntrpu_swiglu_memory [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach seed $seeds {
            set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed"] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project swiglu_memory.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output seed-${seed}.log]
            set f [open $log r];set text [read $f];close $f
            if {![string match "*NTRPU_SWIGLU_MEMORY_PASS cases=$cases *" $text] ||
                ![string match "*NTRPU_SWIGLU_MEMORY_SEED seed=$seed\n*" $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} { error "Memory SwiGLU failed seed=$seed" }
        }
        puts NTRPU_SWIGLU_MEMORY_SUITE_RETURNED
    } else {
        set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
        read_xdc [file join $output clock.xdc]
        synth_design -top ntrpu_swiglu_memory -mode out_of_context -part xcvu5p-flva2104-2-i
        write_checkpoint [file join $output swiglu_memory_synth.dcp]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} { error "Unresolved black boxes" }
        puts NTRPU_SWIGLU_MEMORY_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {swiglu_memory $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
