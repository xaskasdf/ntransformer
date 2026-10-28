# Guard-only safety tests and common OOC physical comparison.
set script_dir [file dirname [file normalize [info script]]]
proc guard_study {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set part xcvu5p-flva2104-2-i
    set rtl [list [file join $script_dir .. fpga rtl ntrpu_axi_capacity_guard.sv] \
        [file join $script_dir .. fpga rtl ntrpu_axi_write_capacity_queue.sv] \
        [file join $script_dir .. fpga rtl ntrpu_rv_elastic.sv]]
    create_project guard_unit [file join $output unit_project] -part $part
    add_files $rtl
    add_files -fileset sim_1 [file join $script_dir .. fpga tb tb_ntrpu_axi_write_capacity_queue.sv]
    set_property top tb_ntrpu_axi_write_capacity_queue [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    foreach depth {1 2 4} {
        set_property generic DEPTH=$depth [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output unit_project guard_unit.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output unit-${depth}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "WRITE_GUARD_SUITE_PASS depth=$depth" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Guard unit failed depth=$depth"}
    }
    foreach negative {1 2 3} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg NEGATIVE=$negative"] [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output unit_project guard_unit.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output unit-negative-${negative}.log]
        set f [open $log r];set text [read $f];close $f
        if {![regexp {Fatal: WRITE_GUARD_(AW_MISMATCH|DATA_MISMATCH|RESPONSE)} $text] || [string first WRITE_GUARD_SUITE_PASS $text]>=0} {error "Unit negative escaped: $negative"}
    }
    close_project
    set xdc [file join $output fixture.xdc]
    set f [open $xdc w];puts $f {create_clock -period 5.000 -name core_clk [get_ports clk]};close $f
    foreach slots {0 1 2 4} {
        set dir [file join $output physical-$slots];file mkdir $dir
        create_project guard_phys_$slots -in_memory -part $part
        read_verilog -sv $rtl
        read_verilog -sv [file join $script_dir .. fpga tb ntrpu_write_guard_select.sv]
        read_verilog -sv [file join $script_dir .. fpga tb ntrpu_write_guard_fixture.sv]
        read_xdc $xdc
        synth_design -top ntrpu_write_guard_fixture -part $part -mode out_of_context -flatten_hierarchy rebuilt -generic GUARD_SLOTS=$slots
        if {[llength [get_cells -hier -filter {IS_BLACKBOX==1}]]} {error "Black box in guard"}
        report_utilization -hierarchical -file [file join $dir synth_utilization.rpt]
        opt_design
        place_design -directive Default
        route_design -directive RuntimeOptimized
        report_utilization -hierarchical -file [file join $dir route_utilization.rpt]
        report_utilization -cells [get_cells guard] -file [file join $dir guard_utilization.rpt]
        report_timing_summary -delay_type min_max -report_unconstrained -file [file join $dir timing.rpt]
        report_timing -delay_type max -max_paths 10 -file [file join $dir setup_paths.rpt]
        report_timing -delay_type min -max_paths 10 -file [file join $dir hold_paths.rpt]
        report_route_status -file [file join $dir route_status.rpt]
        report_drc -file [file join $dir drc.rpt]
        write_checkpoint [file join $dir routed.dcp]
        set setup [get_timing_paths -delay_type max -max_paths 1]
        set hold [get_timing_paths -delay_type min -max_paths 1]
        if {![llength $setup] || ![llength $hold]} {error "Missing timed paths"}
        set f [open [file join $dir metrics.txt] w]
        puts $f "SLOTS=$slots"
        puts $f "SETUP_NS=[get_property SLACK $setup]"
        puts $f "HOLD_NS=[get_property SLACK $hold]"
        puts $f "DRC_ERRORS=[llength [get_drc_violations -quiet -filter {SEVERITY == Error}]]"
        close $f
        puts "CREDIT_GUARD_PHYSICAL_EXPORTED SLOTS=$slots"
        close_project
    }
    puts NTRPU_CREDIT_GUARD_EXPORTED
}
if {[catch {guard_study $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
