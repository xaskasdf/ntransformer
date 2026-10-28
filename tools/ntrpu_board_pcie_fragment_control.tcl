set script_dir [file dirname [file normalize [info script]]]
proc simulate_control {directory sources tb variant netlist} {
    file mkdir $directory;cd $directory
    exec xvlog --sv {*}$sources $tb > compile.log 2>@1
    set libraries [expr {$netlist ? {-L unisims_ver -L secureip} : {}}]
    set tops [expr {$netlist ? {tb_ntrpu_pcie_fragment_control glbl} : {tb_ntrpu_pcie_fragment_control}}]
    exec xelab {*}$tops --debug typical --mt 4 {*}$libraries -s control > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    set seeds [expr {$variant eq "normal" ? {1 7 42} : {1}}]
    foreach seed $seeds {
        set code [catch {exec xsim control -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
        set f [open seed-${seed}.console.log r];set text [read $f];close $f
        if {$variant eq "normal"} {
            if {$code || ![regexp -line {^NTRPU_CONTROL_PASS cases=12 } $text] ||
                [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Control positive failed seed=$seed: $message"}
        } else {
            set expected [dict get {ack WRITE_RESPONSE high SNAPSHOT_64 readonly WRITE_RESPONSE epoch WRITE_RESPONSE fault UNEXPECTED_RESULT_NOT_FAULTED profile READ_RESPONSE telemetry READ_RESPONSE capture READ_RESPONSE reason READ_RESPONSE} $variant]
            if {![regexp -line [format {^Fatal: .*%s.*$} $expected] $text] || [regexp -line {^NTRPU_CONTROL_PASS } $text]} {error "Control mutant not detected: $variant"}
        }
        puts "NTRPU_CONTROL_REVIEW variant=$variant netlist=$netlist seed=$seed PASS"
    }
}
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=1} {error "Expected NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set rtl [file normalize [file join $script_dir .. fpga rtl ntrpu_pcie_fragment_control.sv]]
    set tb [file normalize [file join $script_dir .. fpga tb tb_ntrpu_pcie_fragment_control.sv]]
    set f [open $rtl r];set original [read $f];close $f
    foreach variant {normal ack high readonly epoch fault profile telemetry capture reason} {
        set source $rtl
        if {$variant ne "normal"} {
            switch $variant {
                profile {set before {32'h4652544e};set after {32'h4952544e}}
                telemetry {set before {32'h90:read_register=snapshot_frames};set after {32'h90:read_register=rejected_frames}}
                capture {set before {snapshot_identity<=rejected_identity};set after {snapshot_identity<=rejected_frames}}
                reason {set before {{last_reject_faults,last_reject_code}};set after {{7'd0,last_reject_code}}}
                ack {set before {!identity_ok || !ack_ok};set after {!identity_ok || 1'b0}}
                high {set before {arm_epoch<={descriptor[1],descriptor[0]}};set after {arm_epoch<={32'd0,descriptor[0]}}}
                readonly {set before {response=SLVERR;
    end};set after {response=OKAY;
    end}}
                epoch {set before {{descriptor[1],descriptor[0]}>epoch};set after {{descriptor[1],descriptor[0]}>=epoch}}
                fault {set before {if(unexpected_result) protocol_fault<=1};set after {if(1'b0) protocol_fault<=1}}
            }
            if {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]!=2} {error "Mutation match count: $variant"}
            set source [file join $output ${variant}.sv]
            set f [open $source {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
        }
        simulate_control [file join $output $variant] [list $source] $tb $variant 0
    }
    cd $output
    create_project control [file join $output project] -part xcvu5p-flva2104-2-i
    add_files $rtl
    set_property top ntrpu_pcie_fragment_control [current_fileset]
    synth_design -top ntrpu_pcie_fragment_control -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE || IS_BLACKBOX == 1}]]} {error "Latch or unresolved cell"}
    create_clock -name clk -period 5.000 [get_ports clk]
    report_utilization -hierarchical -file utilization.rpt
    report_timing_summary -report_unconstrained -file timing.rpt
    check_timing -verbose -file check_timing.rpt
    report_drc -file drc.rpt
    write_checkpoint [file join $output control_synth.dcp]
    write_xdc -constraints all [file join $output applied.xdc]
    write_verilog -mode funcsim [file join $output control_funcsim.v]
    close_project
    simulate_control [file join $output netlist] [list [file join $output control_funcsim.v]] $tb normal 1
    puts NTRPU_CONTROL_SUITE_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
