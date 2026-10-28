# Registered two-initiator AXI arbiter; protocol and contention bank.
set script_dir [file dirname [file normalize [info script]]]
proc axi_arbiter {arguments} {
    global script_dir
    if {[llength $arguments] ni {2 3}} {error "Expected MODE NEW_DIRECTORY ?REFERENCE_RTL?"}
    lassign $arguments mode output reference
    if {$mode ni {sim sim_pipeline sim_pipeline_no_wlast_guard sim_rate sim_rate_reference synth}} {error "Invalid mode"}
    set rate [expr {$mode in {sim_rate sim_rate_reference}}]
    set pipeline [expr {$mode in {sim_pipeline sim_pipeline_no_wlast_guard}}]
    set interval [expr {$mode eq "sim_rate_reference" ? 2:1}]
    if {($mode eq "sim_rate_reference") != ($reference ne "")} {error "Reference RTL is required only for sim_rate_reference"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    create_project axi_arbiter [file join $output project] -part xcvu5p-flva2104-2-i
    set rtl [file normalize [file join $script_dir .. fpga rtl ntrpu_axi_arbiter.sv]]
    if {$reference ne ""} {
        file copy [file normalize $reference] [file join $output reference_arbiter.sv]
        set rtl [file join $output reference_arbiter.sv]
    } elseif {$mode eq "sim_pipeline_no_wlast_guard"} {
        set f [open $rtl r];set text [read $f];close $f
        set needle {!write_last_accepted && }
        if {[string first $needle $text]<0 || [string first $needle $text]!=[string last $needle $text]} {error "Mutation guard identity"}
        set rtl [file join $output mutated_arbiter.sv]
        set f [open $rtl {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $needle {}] $text];close $f
    }
    add_files $rtl
    set_property top ntrpu_axi_arbiter [get_filesets sources_1]
    if {$mode ne "synth"} {
        set bench [expr {$rate ? "tb_ntrpu_axi_arbiter_rate":"tb_ntrpu_axi_arbiter"}]
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb ${bench}.sv]]
        set_property top $bench [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set seeds [expr {$rate ? {1}:{1 7 42}}]
        set extra {}
        if {$pipeline} {append extra " -testplusarg PIPELINED=1"}
        if {$rate} {append extra " -testplusarg EXPECTED_INTERVAL=$interval"}
        foreach seed $seeds {
            set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed$extra"] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project axi_arbiter.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output seed-${seed}.log]
            set f [open $log r];set text [read $f];close $f
            if {[regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "AXI arbiter failed mode=$mode seed=$seed"}
            if {$rate} {
                if {![string match "*NTRPU_RATE_PASS cases=4 interval=$interval beats=1024*" $text]} {error "Rate bank incomplete"}
            } elseif {![string match "*NTRPU_ARB_PASS banks=9 resets=4 *" $text] ||
                ![string match "*NTRPU_ARB_SEED seed=$seed\n*" $text]} {error "AXI arbiter unit failed seed=$seed"}
        }
        if {$rate} {puts "NTRPU_RATE_SUITE_RETURNED cases=4 interval=$interval"} else {puts "NTRPU_ARB_SUITE_RETURNED banks=27 mode=$mode"}
    } else {
        set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
        read_xdc [file join $output clock.xdc]
        synth_design -top ntrpu_axi_arbiter -mode out_of_context -part xcvu5p-flva2104-2-i
        write_checkpoint [file join $output axi_arbiter_synth.dcp]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
        if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Unexpected latches"}
        puts NTRPU_ARB_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {axi_arbiter $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
