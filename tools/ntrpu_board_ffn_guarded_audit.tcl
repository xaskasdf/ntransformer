# Read-only structural audit of the supervised FFN synthesis checkpoint.
proc supervisor_audit {arguments} {
    if {[llength $arguments] != 2} {error "Expected SYNTH_DCP NEW_DIRECTORY"}
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_ffn_guarded_fabric" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 3} {
        error "Unexpected design context"
    }
    set f [open [file join $output release_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg"
    set total 0
    foreach name {host_reset_cdc device_reset_cdc host_admit_cdc device_admit_cdc host_up_cdc device_up_cdc host_count_cdc device_count_cdc} {
        set cells [get_cells -quiet -hierarchical -filter "NAME =~ compute/supervisor/$name/release_ff_reg*"]
        if {[llength $cells] != 3} {error "Expected three release stages: $name"}
        foreach cell [lsort $cells] {
            set ref [get_property REF_NAME $cell]
            set init [get_property INIT $cell]
            set attribute [get_property ASYNC_REG $cell]
            puts $f "$cell\t$ref\t$init\t$attribute"
            if {$ref ne "FDCE" || $init ne "1'b0" || $attribute ni {1 TRUE true}} {
                error "Reset release must be an INIT0 asynchronous-clear synchronizer: $cell"
            }
            incr total
        }
    }
    close $f
    set latches [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]
    set boxes [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
    if {[llength $latches] || [llength $boxes]} {error "Transparent storage or blackbox"}
    # Audit the external reset net directly; do not waive its CDC report.
    set f [open [file join $output management_reset_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg\tpreset_ports\tpreset_drivers\tdata_driver\tclock"
    for {set index 0} {$index < 3} {incr index} {
        set name [format {compute/supervisor/management_reset/arststages_ff_reg[%d]} $index]
        set cell [get_cells -quiet $name]
        if {[llength $cell] != 1} {error "Missing external reset stage"}
        set ref [get_property REF_NAME $cell]
        set init [get_property INIT $cell]
        set attribute [get_property ASYNC_REG $cell]
        set net [get_nets -segments -of_objects [get_pins $name/PRE]]
        set ports [get_ports -quiet -of_objects $net]
        set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
        set dn [get_nets -segments -of_objects [get_pins $name/D]]
        set dd [get_pins -quiet -leaf -of_objects $dn -filter {DIRECTION == OUT}]
        set clocks [get_clocks -of_objects [get_pins $name/C]]
        puts $f "$name\t$ref\t$init\t$attribute\t$ports\t$drivers\t$dd\t$clocks"
        if {$ref ne "FDPE" || $init ne "1'b0" || $attribute ni {1 TRUE true} ||
            $ports ne "external_reset_async" || [llength $drivers] || $clocks ne "mgmt_clk"} {
            error "External reset must directly drive three marked FDPEs without a LUT"
        }
        if {$index > 0} {
            set expected [format {compute/supervisor/management_reset/arststages_ff_reg[%d]/Q} [expr {$index-1}]]
            if {$dd ne $expected} {error "Broken reset shift chain"}
        } else {
            if {[llength $dd] != 1 || [get_property REF_NAME [get_cells -of_objects $dd]] ne "GND"} {
                error "First reset stage must shift in zero"
            }
        }
    }
    close $f
    set f [open [file join $output primitive_counts.tsv] {WRONLY CREAT EXCL}]
    puts $f "ref_name\tcount"
    set counts [dict create]
    foreach cell [get_cells -hierarchical -filter {IS_PRIMITIVE == 1}] {
        set ref [get_property REF_NAME $cell]
        dict incr counts $ref
    }
    dict for {ref count} $counts {puts $f "$ref\t$count"}
    close $f
    set f [open [file join $output transport_initialization.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit"
    set cells [get_cells -quiet -hierarchical -filter {NAME =~ compute/fabric/fabric/device/with_irq.control/transport/event_payload_reg* || NAME =~ compute/fabric/fabric/device/with_irq.control/transport/service_payload_reg* || NAME == compute/fabric/fabric/device/with_irq.control/transport/event_send_reg || NAME == compute/fabric/fabric/device/with_irq.control/transport/service_send_reg}]
    if {[llength $cells]!=131} {error "Expected131 persistent transport registers"}
    foreach cell [lsort $cells] {
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]"
        if {[get_property REF_NAME $cell] ne "FDRE" || [get_property INIT $cell] ne "1'b0"} {error "Persistent transport INIT"}
    }
    close $f
    # Export all seventeen three-stage chains and their direct D/C/CLR
    # connectivity. Export one logic level behind CLR to inspect polarity
    # inversion separately; do not infer source registration from names alone.
    set chains {}
    foreach name {host_reset_cdc device_reset_cdc host_admit_cdc device_admit_cdc host_up_cdc device_up_cdc host_count_cdc device_count_cdc} {
        lappend chains [list compute/supervisor/$name release_ff_reg]
    }
    lappend chains [list compute/fabric/fabric/device/with_irq.control/transport/armed_cdc level_ff_reg]
    for {set i 0} {$i<8} {incr i} {lappend chains [list [format {compute/status_cdc[%d].sync} $i] level_ff_reg]}
    set f [open [file join $output level_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg\tclock\td_drivers\td_ports\tclr_drivers\tclr_ports"
    set g [open [file join $output level_clear_drivers.tsv] {WRONLY CREAT EXCL}]
    puts $g "endpoint\tdriver\tref_name\tinit\tinput\tinput_drivers\tinput_ports\tinput_refs\tinput_inits"
    set total 0
    foreach chain $chains {
        lassign $chain path register
        for {set i 0} {$i<3} {incr i} {
            set name [format {%s/%s[%d]} $path $register $i]
            set cell [get_cells -quiet $name]
            if {[llength $cell]!=1} {error "Missing stage: $name"}
            set ref [get_property REF_NAME $cell]
            set init [get_property INIT $cell]
            set attr [get_property ASYNC_REG $cell]
            if {$ref ne "FDCE" || $init ne "1'b0" || $attr ni {1 TRUE true}} {error "Level qualification must initialize low"}
            set clock [get_clocks -of_objects [get_pins $name/C]]
            set dn [get_nets -segments -of_objects [get_pins $name/D]]
            set dd [get_pins -quiet -leaf -of_objects $dn -filter {DIRECTION == OUT}]
            set dp [get_ports -quiet -of_objects $dn]
            set cn [get_nets -segments -of_objects [get_pins $name/CLR]]
            set cd [get_pins -quiet -leaf -of_objects $cn -filter {DIRECTION == OUT}]
            set cp [get_ports -quiet -of_objects $cn]
            puts $f "$name\t$ref\t$init\t$attr\t$clock\t$dd\t$dp\t$cd\t$cp"
            foreach driver $cd {
                set dc [get_cells -of_objects $driver]
                set dr [get_property REF_NAME $dc]
                set di [get_property -quiet INIT $dc]
                if {[string match LUT* $dr]} {
                    foreach pin [get_pins -of_objects $dc -filter {DIRECTION == IN}] {
                        set net [get_nets -segments -of_objects $pin]
                        set inputs [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
                        set ports [get_ports -quiet -of_objects $net]
                        set refs {};set inits {}
                        foreach input $inputs {
                            set ic [get_cells -of_objects $input]
                            lappend refs [get_property REF_NAME $ic]
                            lappend inits [get_property -quiet INIT $ic]
                        }
                        puts $g "$name/CLR\t$driver\t$dr\t$di\t$pin\t$inputs\t$ports\t$refs\t$inits"
                    }
                } else {
                    puts $g "$name/CLR\t$driver\t$dr\t$di\t\t\t\t\t"
                }
            }
            incr total
        }
    }
    close $f;close $g
    if {$total!=51} {error "Expected51 asynchronous-clear stages"}
    set f [open [file join $output armed_source.tsv] {WRONLY CREAT EXCL}]
    set name compute/fabric/fabric/device/with_irq.control/transport/host_armed_registered_reg
    set cell [get_cells -quiet $name]
    puts $f "cell\tref_name\tinit\tclock"
    puts $f "$name\t[get_property REF_NAME $cell]\t[get_property INIT $cell]\t[get_clocks -of_objects [get_pins $name/C]]"
    close $f
    # The guard must remain an observable block after synthesis.
    set guard [get_cells -quiet capacity_guard]
    if {[llength $guard]!=1 || [get_property REF_NAME $guard] ne "ntrpu_axi_capacity_guard"} {error "Missing capacity guard hierarchy"}
    set f [open [file join $output guard_ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "group\texpected\tobserved"
    # Rebuilt synthesis moves combinational logic across the RTL hierarchy.
    # Qualify the public buses and original intermediate nets, not stale RTL
    # hierarchy pin names. The diagnostic inventory retains those actual names.
    foreach {group count} {s_host_axi_araddr 64 s_host_axi_awaddr 64 m_axi_araddr 37 m_axi_awaddr 37} {
        set pins [get_ports -quiet $group*]
        puts $f "$group\t$count\t[llength $pins]"
        if {[llength $pins]!=$count} {error "Guard pin width: $group"}
    }
    close $f
    set f [open [file join $output address_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "signal\tnets\tdrivers\tdriver_refs\tclocks\tloads\tendpoints"
    set address_pins {}
    foreach channel {ar aw} {
        for {set i 0} {$i<64} {incr i} {
            set name [format {pre_axi_%saddr[%d]} $channel $i]
            # Upper bits can remain entirely inside the rebuilt arbiter.
            # Follow the actual registered shared-AXI output for every bit.
            set origin [format {compute/fabric/fabric/arbiter/m_axi_%saddr_reg[%d]/Q} $channel $i]
            set source [get_pins -quiet $origin]
            if {[llength $source]!=1} {error "Missing registered address bit: $origin"}
            set segments [get_nets -segments -of_objects $source]
            set drivers [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]
            set loads [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]
            set refs {};set clocks {}
            foreach driver $drivers {
                set cell [get_cells -of_objects $driver]
                lappend refs [get_property REF_NAME $cell]
                set clocks [concat $clocks [get_clocks -quiet -of_objects [get_pins -of_objects $cell -filter {IS_CLOCK == 1}]]]
            }
            set endpoints [all_fanout -flat -endpoints_only -trace_arcs all $source]
            puts $f "$name\t$segments\t$drivers\t$refs\t$clocks\t$loads\t$endpoints"
            if {[llength $drivers]!=1 || ![llength $loads] || ![llength $endpoints]} {error "Unconnected address bit: $name"}
            lappend address_pins $drivers
        }
    }
    close $f
    set f [open [file join $output guard_connectivity.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tdirection\tnets\tports\tdrivers\tloads\tclocks"
    foreach pin [lsort [get_pins -of_objects $guard]] {
        set nets [get_nets -segments -of_objects $pin]
        set ports [get_ports -quiet -of_objects $nets]
        set drivers [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]
        set loads [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == IN}]
        puts $f "$pin\t[get_property DIRECTION $pin]\t$nets\t$ports\t$drivers\t$loads\t[get_clocks -quiet -of_objects $pin]"
    }
    close $f
    foreach {name count} {s_host_axi_araddr 64 s_host_axi_awaddr 64 m_axi_araddr 37 m_axi_awaddr 37} {
        set ports [get_ports -quiet $name*]
        if {[llength $ports]!=$count} {error "Public port width: $name"}
    }
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing -from $address_pins -max_paths 20 -path_type full_clock_expanded -file [file join $output address_timing.rpt]
    puts NTRPU_GUARDED_TOP_STRUCTURE_PASS
    report_bus_skew -warn_on_violation -file [file join $output bus_skew.rpt]
    report_cdc -details -all_checks_per_endpoint -file [file join $output cdc_all.rpt]
    write_xdc -constraints all [file join $output audit_constraints.xdc]
    puts NTRPU_SUPERVISED_FFN_LEVEL_STRUCTURE_PASS
    report_cdc -details -file [file join $output cdc.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_clocks -file [file join $output clocks.rpt]
    puts "NTRPU_SUPERVISED_FFN_INIT_AUDIT_PASS registers=$total"
    close_project
}
if {[catch {supervisor_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
