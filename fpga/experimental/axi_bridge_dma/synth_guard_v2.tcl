# Isolated V2 OOC comparison; these are diagnostic constraints, not board XDC.
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
proc v2_reports {out} {
    file mkdir $out
    guard_cdc_reports $out
    guard_cdc_inventory $out
    set f [open $out/all_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "source_clock\tdestination_clock\tdelay\tstartpoint\tendpoint\tslack"
    foreach source {mgmt_clk axi_clk} {
        foreach dest {mgmt_clk axi_clk} {
            foreach delay {max min} {
                set paths [get_timing_paths -quiet -from [get_clocks $source] -to [get_clocks $dest] -delay_type $delay -max_paths 10000 -nworst 10000]
                if {[llength $paths]>=10000} {error "Timing path export reached its cap"}
                foreach p $paths {
                    puts $f [join [list $source $dest $delay [get_property STARTPOINT_PIN $p] [get_property ENDPOINT_PIN $p] [get_property SLACK $p]] "\t"]
                }
            }
        }
    }
    close $f
}
if {[catch {
    if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {error "RTL NEW_OUTPUT required"}
    lassign $argv rtl out
    set out [file normalize $out]
    if {![file isfile $rtl] || [file exists $out]} {error "Existing RTL and new output required"}
    file mkdir $out
    read_verilog -sv $rtl
    synth_design -top ntrpu_axi_bridge_guard_v2 -part xcvu47p-fsvh2892-2-e -mode out_of_context
    create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]
    create_clock -name axi_clk -period 4.000 [get_ports axi_clk]
    v2_reports $out/baseline
    write_checkpoint $out/guard_v2_synth.dcp
    write_verilog -mode funcsim $out/guard_v2_synth.v
    set chains {
        cold_m FDCE mgmt_clk VCC/P
        cold_a FDCE axi_clk VCC/P
        armed_a FDCE axi_clk armed_reg/Q
        stop_a FDPE axi_clk GND/G
        echo_a FDRE axi_clk request_reg/Q
        echo_m FDRE mgmt_clk {echo_a_reg[2]/Q}
    }
    for {set n 0} {$n<4} {incr n} {
        lappend chains [format {availability[%d].release_m} $n] FDCE mgmt_clk VCC/P
        lappend chains [format {availability[%d].release_a} $n] FDCE axi_clk VCC/P
    }
    set baseline [dict create];set expected_cells {}
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
    set_max_delay -datapath_only 4.000 -from [one_pin request_reg/C] -to [one_pin {echo_a_reg[0]/D}]
    set_max_delay -datapath_only 10.000 -from [one_pin {echo_a_reg[2]/C}] -to [one_pin {echo_m_reg[0]/D}]
    set_max_delay -datapath_only 4.000 -from [one_pin armed_reg/C] -to [one_pin {armed_a_reg[0]/D}]
    for {set n 0} {$n<4} {incr n} {
        for {set i 0} {$i<3} {incr i} {
            set_false_path -from [one_pin [format {availability[%d].release_m_reg[2]/C} $n]] -to [one_pin [format {availability[%d].release_a_reg[%d]/CLR} $n $i]]
        }
    }
    for {set i 0} {$i<3} {incr i} {
        set_false_path -from [one_pin fault_reg/C] -to [one_pin [format {stop_a_reg[%d]/PRE} $i]]
    }
    # The same-domain cold_a -> armed_a reset remains timed.
    v2_reports $out/scoped
    set f [open $out/scoped/stage_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "endpoint\tdelay\tbefore_slack\tafter_slack\tstartpoint"
    dict for {key values} $baseline {
        lassign $key endpoint delay;lassign $values slack start
        set p [get_timing_paths -quiet -to [one_pin $endpoint] -delay_type $delay -max_paths 1]
        if {[llength $p]!=1 || [get_property SLACK $p] ne $slack || [get_property STARTPOINT_PIN $p] ne $start} {error "Interstage timing changed $key"}
        puts $f [join [list $endpoint $delay $slack [get_property SLACK $p] $start] "\t"]
    }
    close $f
    report_exceptions -ignored -file $out/scoped/exceptions_ignored.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f "GUARD_V2_SYNTHESIS_EXPORTED_NOT_PHYSICAL_TIMING_CLOSED"
    close $f
    close_design
} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
puts GUARD_V2_SYNTHESIS_EXPORTED_NOT_PHYSICAL_TIMING_CLOSED
exit 0
