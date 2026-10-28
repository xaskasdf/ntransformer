set script_dir [file dirname [file normalize [info script]]]
proc simulate_managed {directory sources tb variant netlist} {
    file mkdir $directory;cd $directory
    exec xvlog --sv {*}$sources $tb > compile.log 2>@1
    set libraries [expr {$netlist ? {-L unisims_ver -L secureip} : {}}]
    set tops [expr {$netlist ? {tb_ntrpu_pcie_ingress_managed glbl} : {tb_ntrpu_pcie_ingress_managed}}]
    exec xelab {*}$tops --debug typical --mt 4 {*}$libraries -s peer > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    set seeds [expr {$variant eq "normal" ? {1 7 42} : {1}}]
    foreach seed $seeds {
        set code [catch {exec xsim peer -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
        set f [open seed-${seed}.console.log r];set text [read $f];close $f
        if {$variant eq "normal"} {
            if {$code || ![regexp -line {^NTRPU_GUARDED_PASS jobs=21 results=21 writes=14 } $text] ||
                [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Peer positive failed seed=$seed: $message"}
        } else {
            set expected [dict get {barrier ARM_WITHOUT_BARRIER ack MONITOR_RESULT_STABLE data DESTINATION_DATA} $variant]
            if {![regexp -line [format {^Fatal: .*%s.*$} $expected] $text] || [regexp -line {^NTRPU_GUARDED_PASS } $text]} {
                error "Peer mutant not detected: $variant"
            }
        }
        puts "NTRPU_MANAGED_REVIEW variant=$variant netlist=$netlist seed=$seed PASS"
    }
}
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=1} {error "Expected NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set dependencies {}
    foreach name {ntrpu_pcie_dword_decode ntrpu_pcie_line_ingress ntrpu_peer_rx_guard ntrpu_pcie_ingress_peer ntrpu_axi_capacity_guard ntrpu_pcie_ingress_guarded ntrpu_pcie_ingress_control} {
        lappend dependencies [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    set rtl [file normalize [file join $script_dir .. fpga rtl ntrpu_pcie_ingress_managed.sv]]
    set tb [file normalize [file join $script_dir .. fpga tb tb_ntrpu_pcie_ingress_managed.sv]]
    set f [open $rtl r];set original [read $f];close $f
    foreach variant {normal barrier ack data} {
        set source $rtl
        if {$variant ne "normal"} {
            switch $variant {
                barrier {set before {.source_stopped(source_stopped), .ingress_isolated(ingress_isolated),
        .quiesce_epoch(quiesce_epoch)};set after {.source_stopped(1'b1), .ingress_isolated(1'b1),
        .quiesce_epoch(epoch)}}
                ack {set before {.result_ready(result_ready)};set after {.result_ready(1'b1)}}
                data {set before {.s_axi_wdata(s_axi_wdata)};set after {.s_axi_wdata(s_axi_wdata ^ 512'b1)}}
            }
            if {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]!=2} {error "Mutation match count: $variant"}
            set source [file join $output ${variant}.sv]
            set f [open $source {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
        }
        simulate_managed [file join $output $variant] [concat $dependencies [list $source]] $tb $variant 0
    }
    cd $output
    create_project peer [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [concat $dependencies [list $rtl]]
    set_property top ntrpu_pcie_ingress_managed [current_fileset]
    synth_design -top ntrpu_pcie_ingress_managed -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE || IS_BLACKBOX == 1}]]} {error "Latch or unresolved cell"}
    create_clock -name clk -period 5.000 [get_ports clk]
    report_utilization -hierarchical -file utilization.rpt
    report_timing_summary -report_unconstrained -file timing.rpt
    check_timing -verbose -file check_timing.rpt
    report_drc -file drc.rpt
    write_checkpoint [file join $output peer_synth.dcp]
    write_xdc -constraints all [file join $output applied.xdc]
    write_verilog -mode funcsim [file join $output peer_funcsim.v]
    close_project
    simulate_managed [file join $output netlist] [list [file join $output peer_funcsim.v]] $tb normal 1
    puts NTRPU_MANAGED_SUITE_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
