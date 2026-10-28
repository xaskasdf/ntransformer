set script_dir [file dirname [file normalize [info script]]]
proc simulate {directory sources tb netlist variant} {
    file mkdir $directory;cd $directory
    exec xvlog --sv {*}$sources $tb > compile.log 2>@1
    set libs [expr {$netlist ? {-L unisims_ver -L secureip} : {}}]
    set tops [expr {$netlist ? {tb_ntrpu_pcie_fragment_pool glbl} : {tb_ntrpu_pcie_fragment_pool}}]
    exec xelab {*}$tops --debug typical --mt 4 {*}$libs -s ownership > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    set seeds [expr {$variant eq "normal" ? {1 7 42} : {1}}]
    foreach seed $seeds {
        set code [catch {exec xsim ownership -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
        set f [open seed-${seed}.console.log r];set text [read $f];close $f
        if {$variant eq "normal"} {
            if {$code || ![regexp -line {^OWN_PASS } $text] || [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Ownership positive failed seed=$seed: $message"}
        } else {
            set expected [dict get {joint SPLIT_ARM_POOL sticky JOINT_ARM_TIMEOUT cancel RETIRE_READY atomic LEASE_UNTIL_ACK barrier ACK_BARRIER arrival ACK_ARRIVAL error RESULT_STATUS channel COMMAND_CONTEXT} $variant]
            if {![regexp -line [format {^Fatal: .*%s.*$} $expected] $text] || [regexp -line {^OWN_PASS } $text]} {error "Ownership mutant not detected: $variant"}
        }
        puts "NTRPU_OWNERSHIP_REVIEW netlist=$netlist variant=$variant seed=$seed PASS"
    }
}
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=1} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set root [file normalize [file join $script_dir ..]]
    set rtl [file join $root fpga rtl ntrpu_pcie_fragment_pool.sv]
    set tb [file join $root fpga tb tb_ntrpu_pcie_fragment_pool.sv]
    set dependencies {}
    foreach n {ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_fragment_pool_bridge} {lappend dependencies [file join $root fpga rtl $n.sv]}
    set f [open $rtl r];set original [read $f];close $f
    foreach variant {normal joint sticky cancel atomic barrier arrival error channel} {
        set source $rtl
        if {$variant ne "normal"} {
            switch $variant {
                joint {set before {.stop(collector_stop), .arm_valid(joint_arm)};set after {.stop(collector_stop), .arm_valid(arm_valid)}}
                sticky {set before {(failed && bridge_lease_held)};set after {failed}}
                cancel {set before {(failed && bridge_lease_held)};set after {1'b0}}
                atomic {set before {.retire_valid(result_ready)};set after {.retire_valid(result_valid)}}
                barrier {set before {wire ack_barrier = source_stopped && ingress_isolated && quiesce_epoch == epoch};set after {wire ack_barrier = 1'b1}}
                arrival {set before {transport_idle && !s_axi_awvalid && !s_axi_wvalid};set after {transport_idle}}
                error {set before {assign done_error = done_status != 0 && done_status != 1};set after {assign done_error = 1'b0}}
                channel {set before {.epoch(bridge_epoch)};set after {.arm_channel(~arm_channel), .epoch(bridge_epoch)}}
            }
            if {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]!=2} {error "Mutation match count: $variant"}
            set source [file join $output ${variant}.sv]
            set f [open $source {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
        }
        simulate [file join $output $variant] [concat $dependencies [list $source]] $tb 0 $variant
    }
    set synth [file join $output synth];file mkdir $synth;cd $synth
    create_project ownership [file join $synth project] -part xcvu5p-flva2104-2-i
    add_files [concat $dependencies [list $rtl]]
    set_property top ntrpu_pcie_fragment_pool [current_fileset]
    synth_design -top ntrpu_pcie_fragment_pool -mode out_of_context -part xcvu5p-flva2104-2-i -generic {FIRST_COOKIE=64'hffffffff}
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE || IS_BLACKBOX == 1}]]} {error "Latch or unresolved cell"}
    create_clock -name clk -period 5.000 [get_ports clk]
    report_utilization -hierarchical -file utilization.rpt
    report_timing_summary -report_unconstrained -file timing.rpt
    check_timing -verbose -file check_timing.rpt
    report_drc -file drc.rpt
    write_checkpoint [file join $synth ownership_synth.dcp]
    write_xdc -constraints all [file join $synth applied.xdc]
    write_verilog -mode funcsim [file join $synth ownership_funcsim.v]
    close_project
    # The synthesized top has its parameter resolved, so remove only the TB override.
    set f [open $tb r];set net_tb [string map [list { #(.FIRST_COOKIE(64'hffffffff))} {}] [read $f]];close $f
    set n [file join $output netlist_tb.sv];set f [open $n {WRONLY CREAT EXCL}];puts -nonewline $f $net_tb;close $f
    simulate [file join $output netlist] [list [file join $synth ownership_funcsim.v]] $n 1 normal
    puts NTRPU_FRAGMENT_OWNERSHIP_SUITE_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
