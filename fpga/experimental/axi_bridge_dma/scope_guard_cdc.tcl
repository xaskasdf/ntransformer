# Experimental constraints on an immutable OOC checkpoint, never production XDC.
set ::guard_cdc_library_only 1
source [file join [file dirname [info script]] audit_guard_cdc.tcl]
proc one_cell {name} {
    set c [get_cells -quiet -filter [format {NAME == "%s"} $name]]
    if {[llength $c]!=1} {error "Expected one cell $name"}
    return $c
}
proc one_pin {name} {
    set p [get_pins -quiet -filter [format {NAME == "%s"} $name]]
    if {[llength $p]!=1} {error "Expected one pin $name"}
    return $p
}
proc pin_driver {name} {
    return [get_pins -quiet -of_objects [get_nets -of_objects [one_pin $name]] -filter {DIRECTION==OUT}]
}
if {[catch {
    if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {error "CHECKPOINT NEW_OUTPUT required"}
    lassign $argv checkpoint out
    set out [file normalize $out]
    if {![file isfile $checkpoint] || [file exists $out]} {error "Immutable checkpoint and new output required"}
    file mkdir $out
    open_checkpoint $checkpoint
    # Check all seven chains before restricting any paths.
    set chains {
        cold_m FDCE mgmt_clk VCC/P
        cold_a FDCE axi_clk VCC/P
        healthy_m FDCE mgmt_clk VCC/P
        admit_a FDCE axi_clk VCC/P
        stop_a FDPE axi_clk GND/G
        echo_a FDRE axi_clk request_reg/Q
        echo_m FDRE mgmt_clk {echo_a_reg[2]/Q}
    }
    set f [open $out/stage_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "endpoint\tdelay\tbefore_slack\tafter_slack\tstartpoint"
    set baseline [dict create]
    set expected_cells {}
    foreach {chain ref clock first} $chains {
        set control_net {}
        for {set i 0} {$i<3} {incr i} {
            set name [format {%s_reg[%d]} $chain $i]
            set c [one_cell $name];lappend expected_cells $c
            if {[get_property REF_NAME $c] ne $ref || ![get_property ASYNC_REG $c] || [get_property INIT $c] ne "1'b0"} {error "Chain properties differ: $name"}
            if {[get_clocks -of_objects [one_pin $name/C]] ne $clock || [pin_driver $name/CE] ne "VCC/P"} {error "Clock/enable differs: $name"}
            set expected $first
            if {$i>0} {set expected [format {%s_reg[%d]/Q} $chain [expr {$i-1}]]}
            if {[pin_driver $name/D] ne $expected} {error "D/Q chain differs: $name"}
            if {$ref eq "FDRE"} {
                if {[pin_driver $name/R] ne "GND/G"} {error "Unexpected data reset $name"}
            } else {
                set ctrl [expr {$ref eq "FDCE" ? "CLR" : "PRE"}]
                set net [get_nets -of_objects [one_pin $name/$ctrl]]
                if {$i==0} {set control_net $net} elseif {$net ne $control_net} {error "Split asynchronous controls: $name"}
            }
            if {$i>0} {
                foreach delay {max min} {
                    set p [get_timing_paths -quiet -to [one_pin $name/D] -delay_type $delay -max_paths 1]
                    if {[llength $p]!=1} {error "Missing interstage path $name $delay"}
                    dict set baseline [list $name/D $delay] [list [get_property SLACK $p] [get_property STARTPOINT_PIN $p]]
                }
            }
        }
    }
    if {[lsort [get_cells -filter {ASYNC_REG == 1}]] ne [lsort $expected_cells]} {error "Unexpected synchronizers"}
    # The current driver cones remain recorded in pins.tsv and are independently reviewed.
    # Bound only the two first-stage data nets, using each destination period.
    set_max_delay -datapath_only 4.000 -from [one_pin request_reg/C] -to [one_pin {echo_a_reg[0]/D}]
    set_max_delay -datapath_only 10.000 -from [one_pin {echo_a_reg[2]/C}] -to [one_pin {echo_m_reg[0]/D}]
    # Cut only twelve source/control-pin pairs. External controls remain unconstrained.
    foreach source {armed_reg/C fault_reg/C {cold_m_reg[2]/C}} {
        for {set i 0} {$i<3} {incr i} {
            set_false_path -from [one_pin $source] -to [one_pin [format {admit_a_reg[%d]/CLR} $i]]
        }
    }
    for {set i 0} {$i<3} {incr i} {
        set_false_path -from [one_pin fault_reg/C] -to [one_pin [format {stop_a_reg[%d]/PRE} $i]]
    }
    dict for {key values} $baseline {
        lassign $key endpoint delay
        lassign $values slack start
        set p [get_timing_paths -quiet -to [one_pin $endpoint] -delay_type $delay -max_paths 1]
        if {[llength $p]!=1 || [get_property SLACK $p] ne $slack || [get_property STARTPOINT_PIN $p] ne $start} {error "Interstage timing changed $key"}
        puts $f [join [list $endpoint $delay $slack [get_property SLACK $p] $start] "\t"]
    }
    close $f
    guard_cdc_reports $out
    guard_cdc_inventory $out
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    # No DCP is rewritten and no waivers are applied.
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f "GUARD_CDC_SCOPED_TRIAL_EXPORTED_NOT_TIMING_CLOSED"
    close $f
    close_design
} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
puts GUARD_CDC_SCOPED_TRIAL_EXPORTED_NOT_TIMING_CLOSED
exit 0
