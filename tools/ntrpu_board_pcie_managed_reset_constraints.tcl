# Separate DCP revision: audit first, then constrain only the45 proven async CLR pins.
proc audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected SYNTH_DCP NEW_OUTPUT in a fresh process"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_dual_host_cold" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 5} {
        error "Unexpected input checkpoint"
    }
    set root core/receiver/startup
    set chains {source_count_cdc source_reset_cdc source_up_cdc source_held_cdc admission_cdc}
    foreach g {0 1} {
        foreach name {count_cdc reset_cdc up_cdc held_cdc ready_cdc} {
            lappend chains [format {memory[%d].%s} $g $name]
        }
    }
    set releases [get_cells -quiet -hierarchical -filter {NAME =~ *release_ff_reg*}]
    set f [open [file join $output release_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg\tclock\tdata_driver\tclear_drivers\tclear_ports\tclear_startpoints"
    set total 0
    foreach suffix $chains {
        set chain $root/$suffix
        set cells {}
        foreach cell $releases {
            if {[file dirname $cell] eq $chain} {lappend cells $cell}
        }
        if {[llength $cells] != 3} {error "Expected three release stages: $chain"}
        set previous ""
        foreach cell [lsort $cells] {
            set ref [get_property REF_NAME $cell]
            set init [get_property INIT $cell]
            set async [get_property ASYNC_REG $cell]
            set clk [get_clocks -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]]
            set dn [get_nets -segments -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == D}]]
            set dd [get_pins -quiet -leaf -of_objects $dn -filter {DIRECTION == OUT}]
            set clr [get_pins -of_objects $cell -filter {REF_PIN_NAME == CLR}]
            set cn [get_nets -segments -of_objects $clr]
            set cd [get_pins -quiet -leaf -of_objects $cn -filter {DIRECTION == OUT}]
            set cp [get_ports -quiet -of_objects $cn]
            set starts [all_fanin -flat -startpoints_only -to $clr]
            puts $f "$cell\t$ref\t$init\t$async\t$clk\t$dd\t$cd\t$cp\t$starts"
            if {$ref ne "FDCE" || $init ne "1'b0" || $async ni {1 TRUE true} || [llength $clk] != 1} {
                error "Unrecognized reset release primitive: $cell"
            }
            if {$previous eq ""} {
                if {[llength $dd] != 1 || [get_property REF_NAME [get_cells -of_objects $dd]] ne "VCC"} {
                    error "First release stage must shift in one: $cell"
                }
            } elseif {$dd ne "$previous/Q"} {error "Broken D/Q release chain: $cell"}
            set previous $cell
            incr total
        }
    }
    close $f
    if {$total != 45 || [llength $releases] != 45} {error "Unexpected release register inventory"}
    set f [open [file join $output external_reset.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg\tclock\tpreset_ports\tpreset_drivers\tdata_driver"
    for {set i 0} {$i < 3} {incr i} {
        set name [format {%s/management_reset/arststages_ff_reg[%d]} $root $i]
        set cell [get_cells -quiet $name]
        if {[llength $cell] != 1} {error "Missing management reset stage"}
        set ref [get_property REF_NAME $cell]
        set init [get_property INIT $cell]
        set async [get_property ASYNC_REG $cell]
        set clk [get_clocks -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]]
        set net [get_nets -segments -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == PRE}]]
        set ports [get_ports -quiet -of_objects $net]
        set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
        set dn [get_nets -segments -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == D}]]
        set dd [get_pins -quiet -leaf -of_objects $dn -filter {DIRECTION == OUT}]
        puts $f "$cell\t$ref\t$init\t$async\t$clk\t$ports\t$drivers\t$dd"
        if {$ref ne "FDPE" || $init ne "1'b0" || $async ni {1 TRUE true} ||
            $ports ne "external_reset_async" || [llength $drivers] || $clk ne "mgmt_clk"} {
            error "External reset must directly preset three marked management FDPEs"
        }
        if {$i == 0} {
            if {[llength $dd] != 1 || [get_property REF_NAME [get_cells -of_objects $dd]] ne "GND"} {
                error "First management reset stage must shift in zero"
            }
        } elseif {$dd ne [format {%s/management_reset/arststages_ff_reg[%d]/Q} $root [expr {$i-1}]]} {
            error "Broken management reset D/Q chain"
        }
    }
    close $f
    set f [open [file join $output interclock_paths.tsv] {WRONLY CREAT EXCL}]
    puts $f "from_clock\tto_clock\tdelay_type\tstartpoint\tendpoint\tslack\tdatapath_delay"
    foreach {from to} {mgmt_clk source_clk source_clk mgmt_clk mgmt_clk ui0_clk ui0_clk mgmt_clk mgmt_clk ui1_clk ui1_clk mgmt_clk} {
        foreach delay {max min} {
            set paths [get_timing_paths -quiet -from [get_clocks $from] -to [get_clocks $to] -delay_type $delay -max_paths 1000 -nworst 1]
            if {[llength $paths] == 1000} {error "Timing path inventory truncated"}
            foreach p $paths {
                puts $f "$from\t$to\t$delay\t[get_property STARTPOINT_PIN $p]\t[get_property ENDPOINT_PIN $p]\t[get_property SLACK $p]\t[get_property DATAPATH_DELAY $p]"
            }
            report_timing -from [get_clocks $from] -to [get_clocks $to] -delay_type $delay -max_paths 1000 -nworst 1 -file [file join $output ${from}_${to}_${delay}.rpt]
        }
    }
    close $f
    report_cdc -details -all_checks_per_endpoint -file [file join $output cdc_all.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    puts "NTRPU_MANAGED_RESET_AUDIT_EXPORTED release_registers=$total management_registers=3"
    # The original DCP, reports and constraints above stay available as evidence.
    write_verilog -mode funcsim [file join $output before_funcsim.v]
    set clr_pins [get_pins -of_objects $releases -filter {REF_PIN_NAME == CLR}]
    if {[llength $clr_pins] != 45} {error "Expected only45 audited CLR pins"}
    set f [open [file join $output constrained_endpoints.txt] {WRONLY CREAT EXCL}]
    foreach pin [lsort $clr_pins] {puts $f $pin}
    close $f
    set_false_path -to $clr_pins
    foreach {from to} {mgmt_clk source_clk source_clk mgmt_clk mgmt_clk ui0_clk ui0_clk mgmt_clk mgmt_clk ui1_clk ui1_clk mgmt_clk} {
        foreach delay {max min} {
            if {[llength [get_timing_paths -quiet -from [get_clocks $from] -to [get_clocks $to] -delay_type $delay -max_paths 1000 -nworst 1]]} {
                error "Unresolved path outside the audited async CLR endpoints: $from -> $to"
            }
        }
    }
    write_verilog -mode funcsim [file join $output after_funcsim.v]
    write_checkpoint [file join $output managed_host_reset_constrained.dcp]
    write_xdc -constraints all [file join $output constrained.xdc]
    report_utilization -file [file join $output constrained_utilization.rpt]
    report_timing_summary -report_unconstrained -file [file join $output constrained_timing.rpt]
    check_timing -verbose -file [file join $output constrained_check_timing.rpt]
    report_clocks -file [file join $output constrained_clocks.rpt]
    report_clock_interaction -file [file join $output constrained_clock_interaction.rpt]
    report_cdc -details -file [file join $output constrained_cdc.rpt]
    report_bus_skew -file [file join $output constrained_bus_skew.rpt]
    report_exceptions -coverage -file [file join $output constrained_exceptions.rpt]
    report_exceptions -ignored -file [file join $output constrained_exceptions_ignored.rpt]
    report_drc -file [file join $output constrained_drc.rpt]
    report_methodology -file [file join $output constrained_methodology.rpt]
    puts "NTRPU_MANAGED_RESET_CONSTRAINTS_EXPORTED endpoints=45"
    close_project
}
if {[catch {audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
