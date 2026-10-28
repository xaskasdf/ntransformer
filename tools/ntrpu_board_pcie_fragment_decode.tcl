set script_dir [file dirname [file normalize [info script]]]
proc run_decode_sim {directory rtl tb mutant netlist} {
    file mkdir $directory
    cd $directory
    exec xvlog --sv $rtl $tb > compile.log 2>@1
    if {$netlist} {
        exec xelab tb_ntrpu_pcie_fragment_decode --debug typical --mt 2 -L unisims_ver -L secureip -s decoder > elaborate.log 2>@1
    } else {
        exec xelab tb_ntrpu_pcie_fragment_decode --debug typical --mt 2 -s decoder > elaborate.log 2>@1
    }
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    set code [catch {exec xsim decoder -tclbatch run.tcl -log simulation.log > simulation.console.log 2>@1} message]
    set f [open simulation.console.log r];set text [read $f];close $f
    if {$mutant eq "normal" || $netlist} {
        if {$code || ![regexp -line {^NTRPU_FRAGMENT_PASS cases=1176 allowed=516 rejected=660$} $text] ||
            [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Decoder positive failed $directory: $message"}
    } else {
        set expected [dict get {range FRAGMENT_DECISION epoch FRAGMENT_DECISION gate FRAGMENT_BITMAP data FRAGMENT_PAYLOAD tag FRAGMENT_DECISION format FRAGMENT_DECISION} $mutant]
        if {![regexp -line [format {^Fatal: %s } $expected] $text] || [regexp -line {^NTRPU_FRAGMENT_PASS } $text]} {
            error "Decoder mutant not detected: $mutant"
        }
    }
    puts "NTRPU_FRAGMENT_REVIEW variant=$mutant netlist=$netlist PASS"
}
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=1} {error "Expected NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set rtl [file normalize [file join $script_dir .. fpga rtl ntrpu_pcie_fragment_decode.sv]]
    set tb [file normalize [file join $script_dir .. fpga tb tb_ntrpu_pcie_fragment_decode.sv]]
    set f [open $rtl r];set original [read $f];close $f
    foreach variant {normal range epoch gate data tag format} {
        set source $rtl
        if {$variant ne "normal"} {
            switch $variant {
                range {set before {awaddr != lease_window};set after {awaddr[31:0] != lease_window[31:0]}}
                epoch {set before {wdata[127:64] != lease_epoch};set after {wdata[95:64] != lease_epoch[31:0]}}
                gate {set before {accept ? selected_bytes : 64'b0};set after {selected_bytes}}
                data {set before {wdata[511:256]};set after {wdata[503:248]}}
                tag {set before {wdata[191:128] != lease_tag};set after {1'b0}}
                format {set before { || wstrb != ~64'b0};set after {}}
            }
            set matches [expr {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]-1}]
            if {$matches != ($variant eq "data" ? 2 : 1)} {error "Mutation match count: $variant"}
            set source [file join $output ${variant}.sv]
            set f [open $source {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
        }
        run_decode_sim [file join $output $variant] $source $tb $variant 0
    }
    cd $output
    create_project decoder [file join $output project] -part xcvu5p-flva2104-2-i
    add_files $rtl
    set_property top ntrpu_pcie_fragment_decode [current_fileset]
    synth_design -top ntrpu_pcie_fragment_decode -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE || IS_BLACKBOX == 1}]]} {error "Latch or unresolved cell"}
    # Combinational budget only; no placed/routed or board timing claim.
    set_max_delay 5.000 -from [all_inputs] -to [all_outputs]
    report_utilization -file utilization.rpt
    report_timing -from [all_inputs] -to [all_outputs] -max_paths 10 -file timing.rpt
    report_drc -file drc.rpt
    write_checkpoint [file join $output decoder_synth.dcp]
    write_xdc -constraints all [file join $output applied.xdc]
    write_verilog -mode funcsim [file join $output decoder_funcsim.v]
    close_project
    run_decode_sim [file join $output netlist] [file join $output decoder_funcsim.v] $tb normal 1
    puts NTRPU_FRAGMENT_SUITE_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
