set script_dir [file dirname [file normalize [info script]]]
proc simulate_ingress {directory sources tb variant netlist} {
    file mkdir $directory;cd $directory
    exec xvlog --sv {*}$sources $tb > compile.log 2>@1
    set libraries [expr {$netlist ? {-L unisims_ver -L secureip} : {}}]
    set tops [expr {$netlist ? {tb_ntrpu_pcie_fragment_ingress glbl} : {tb_ntrpu_pcie_fragment_ingress}}]
    exec xelab {*}$tops --debug typical --mt 4 {*}$libraries -s ingress > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    set seeds [expr {$variant eq "normal" ? {1 7 42} : {1}}]
    foreach seed $seeds {
        set code [catch {exec xsim ingress -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
        set f [open seed-${seed}.console.log r];set text [read $f];close $f
        if {$variant eq "normal"} {
            if {$code || ![regexp -line {^NTRPU_FRAGMENT_PASS jobs=20 results=20 lines=16 } $text] ||
                [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Ingress positive failed seed=$seed: $message"}
        } else {
            set expected [dict get {conflict B_ID_OR_RESPONSE holes (PARTIAL_PUBLICATION|LINE_CONTENT) retract MONITOR_LINE_STABLE barrier RESULT_WITHOUT_BARRIER foreign FOREIGN_RESPONSE_COMPLETED drain STOP_DROPPED_AW poison REJECT_PRESERVE burst BURST_PREMATURE_B retired RETIRED_OUTCOME_CHANGED counter B_REJECT_COUNT} $variant]
            if {![regexp -line [format {^Fatal: FRAGMENT_CHECK .* %s$} $expected] $text] || [regexp -line {^NTRPU_FRAGMENT_PASS } $text]} {
                error "Ingress mutant not detected: $variant"
            }
        }
        puts "NTRPU_FRAGMENT_REVIEW variant=$variant netlist=$netlist seed=$seed PASS"
    }
}
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=1} {error "Expected NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set decoder [file normalize [file join $script_dir .. fpga rtl ntrpu_pcie_fragment_decode.sv]]
    set rtl [file normalize [file join $script_dir .. fpga rtl ntrpu_pcie_fragment_ingress.sv]]
    set tb [file normalize [file join $script_dir .. fpga tb tb_ntrpu_pcie_fragment_ingress.sv]]
    set f [open $rtl r];set original [read $f];close $f
    foreach variant {normal conflict holes retract barrier foreign drain poison burst retired counter} {
        set source $rtl
        if {$variant ne "normal"} {
            switch $variant {
                conflict {set before {&& valid_bytes[lane]};set after {&& 1'b0}}
                holes {set before {valid_bytes == 64'hffffffffffffffff};set after {valid_bytes != 0}}
                retract {set before {if (line_valid && line_ready)};set after {if (line_valid && (line_ready || stop))}}
                barrier {set before {transport_idle && no_arrivals && quiescent &&};set after {transport_idle && no_arrivals &&}}
                foreign {set before {if (done_epoch==epoch && done_tag==tag)};set after {if (1'b1)}}
                drain {set before {if (stop && lease_held && !reported && !failed)};set after {if (stop) aw_held<=0;
            if (stop && lease_held && !reported && !failed)}}
                poison {set before {if (local_violation)};set after {if (local_violation || (lease_held && !decode_accept))}}
                burst {set before {if (wlast)};set after {if (1'b1)}}
                retired {set before {lease_held && !reported && !failed && !stop &&};set after {lease_held && !failed && !stop &&}}
                counter {set before {rejected_frames+1};set after {rejected_frames+2}}
            }
            if {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]!=2} {error "Mutation match count: $variant"}
            set source [file join $output ${variant}.sv]
            set f [open $source {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
        }
        simulate_ingress [file join $output $variant] [list $decoder $source] $tb $variant 0
    }
    cd $output
    create_project ingress [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [list $decoder $rtl]
    set_property top ntrpu_pcie_fragment_ingress [current_fileset]
    synth_design -top ntrpu_pcie_fragment_ingress -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE || IS_BLACKBOX == 1}]]} {error "Latch or unresolved cell"}
    create_clock -name clk -period 5.000 [get_ports clk]
    report_utilization -hierarchical -file utilization.rpt
    report_timing_summary -report_unconstrained -file timing.rpt
    check_timing -verbose -file check_timing.rpt
    report_drc -file drc.rpt
    write_checkpoint [file join $output ingress_synth.dcp]
    write_xdc -constraints all [file join $output applied.xdc]
    write_verilog -mode funcsim [file join $output ingress_funcsim.v]
    close_project
    simulate_ingress [file join $output netlist] [list [file join $output ingress_funcsim.v]] $tb normal 1
    puts NTRPU_FRAGMENT_SUITE_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
