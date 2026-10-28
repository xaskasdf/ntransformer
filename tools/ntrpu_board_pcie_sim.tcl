# Work on a NEW copy of an already generated AMD example; never mutate evidence.
# Usage: -tclargs INPUT_EXAMPLE_DIRECTORY NEW_DIRECTORY
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_pcie_sim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] ni {2 3} || [llength [get_projects -quiet]]} {
        error "Use a fresh process: -tclargs INPUT_EXAMPLE_DIRECTORY NEW_DIRECTORY ?multi|multi-stall|host64-stall|host64-all-stall?"
    }
    lassign $arguments input output mode
    if {$mode ni {{} multi multi-stall host64-stall host64-all-stall}} { error "Unknown mode: $mode" }
    set multi [expr {$mode in {multi multi-stall host64-stall host64-all-stall}}]
    set stalled [expr {$mode in {multi-stall host64-stall host64-all-stall}}]
    set host64 [expr {$mode in {host64-stall host64-all-stall}}]
    set all_stalled [expr {$mode eq "host64-all-stall"}]
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists: $output" }
    if {![file exists [file join $input ntrpu_xdma_ex.xpr]]} { error "Missing generated XDMA project" }
    file mkdir $output
    file copy $input [file join $output example]
    if {$multi} {
        exec python3 [file join $ntrpu_script_dir patch_ntrpu_pcie_multi.py] [file join $output example] > [file join $output patch.log] 2>@1
    }
    if {$host64} {
        exec python3 [file join $ntrpu_script_dir patch_ntrpu_pcie_host64.py] [file join $output example] > [file join $output host64_patch.log] 2>@1
    }
    if {$stalled} {
        set stall_options {}
        if {$all_stalled} { lappend stall_options --all-channels }
        exec python3 [file join $ntrpu_script_dir patch_ntrpu_pcie_stalls.py] [file join $output example] {*}$stall_options > [file join $output stalls_patch.log] 2>@1
    }
    set project [file join $output example ntrpu_xdma_ex.xpr]
    open_project $project
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {
        error "Unexpected example FPGA part"
    }
    foreach file [get_files -all] {
        if {[string first $input [file normalize $file]] == 0} {
            error "Copied project still references immutable input: $file"
        }
    }
    # Instantiate the independent monitor in the copied testbench only.
    set board_path [file join $output example imports board.v]
    set stream [open $board_path r]; set board [read $stream]; close $stream
    if {[regexp -all {module board;} $board] != 1} { error "Unexpected vendor board module" }
    if {[string match {*ntrpu_monitor*} $board]} { error "Input is already instrumented" }
    set monitor [expr {$multi ? "ntrpu_pcie_multi_monitor" : "ntrpu_pcie_monitor"}]
    set instances "$monitor ntrpu_monitor();"
    if {$host64} { set instances "$monitor #(.HOST64(1)) ntrpu_monitor(); ntrpu_host_memory ntrpu_host();" }
    set board [string map [list {module board;} "module board; $instances"] $board]
    set stream [open $board_path w]; puts -nonewline $stream $board; close $stream
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ${monitor}.sv]
    if {$multi} {
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_cq_check.sv]
    }
    set defines {}
    if {$stalled} {
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_sim_ready_gate.sv]
        lappend defines NTRPU_PCIE_STALLS
        if {$all_stalled} { lappend defines NTRPU_PCIE_ALL_STALLS }
    }
    if {$host64} {
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_host_memory.sv]
        lappend defines NTRPU_PCIE_HOST64
    }
    set_property verilog_define $defines [get_filesets sim_1]
    set_property top board [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    # Vivado already emits --mt from its elaboration settings; do not duplicate it.
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options {-testplusarg TESTNAME=dma_test0}] [get_filesets sim_1]
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set previous_time [current_time -s]
    for {set step 0} {$step < 2000} {incr step} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_PCIE_PROGRESS time=$now ep_link=[get_value -radix bin /board/EP/user_lnk_up] rp_link=[get_value -radix bin /board/RP/user_lnk_up] rp_ltssm=[get_value -radix hex /board/RP/cfg_ltssm_state]"
        flush stdout
        if {$now eq $previous_time} { break }
        set previous_time $now
    }
    puts NTRPU_PCIE_SIMULATOR_RETURNED
    close_sim
    close_project
    puts "Review numerical/status markers and error logs; this marker alone is not a PASS."
}
if {[catch {ntrpu_pcie_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
