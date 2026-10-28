# Combined read-only audit: one checkpoint open, four independent exports.
# No placement, routing, constraints, waiver or property mutation is performed.
proc guarded_clock {output} {
    set latches [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE}]
    set blackboxes [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]
    set stream [open [file join $output storage_audit.txt] {WRONLY CREAT EXCL}]
    puts $stream "TRANSPARENT_STORAGE_CELLS [llength $latches]\nBLACKBOX_CELLS [llength $blackboxes]"
    close $stream
    if {[llength $latches] || [llength $blackboxes]} { error "Unexpected storage/blackbox" }
    # These payload/send registers intentionally survive application reset.
    # Inspect physical configuration INIT independently after implementation.
    set prefix compute/fabric/fabric/device/with_irq.control/transport
    set registers [get_cells -quiet -hierarchical -filter {NAME =~ *with_irq.control/transport/event_payload_reg* || NAME =~ *with_irq.control/transport/service_payload_reg* || NAME =~ *with_irq.control/transport/event_send_reg || NAME =~ *with_irq.control/transport/service_send_reg}]
    if {[llength $registers] != 131} { error "Transport storage count: [llength $registers]" }
    set stream [open [file join $output initialization.tsv] {WRONLY CREAT EXCL}]
    puts $stream "cell\tref_name\tinit"
    foreach cell [lsort $registers] {
        puts $stream "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]"
        if {[get_property REF_NAME $cell] ne "FDRE" || [get_property INIT $cell] ne "1'b0"} { error "Transport configuration initialization: $cell" }
    }
    close $stream
    foreach suffix {event_send_reg service_send_reg} {
        if {[llength [get_cells -quiet $prefix/$suffix]] != 1} { error "Missing send register: $suffix" }
    }
    set routes [dict create]
    foreach clock {clk host_clk mgmt_clk} {
        set route [get_property ROUTE [get_nets $clock]]
        if {![string match -nocase *gap* $route]} { error "Expected OOC parent gap: $clock" }
        dict set routes $clock $route
        set stream [open [file join $output ${clock}_route.txt] {WRONLY CREAT EXCL}]
        puts $stream $route
        close $stream
        report_property -all [get_nets $clock] -file [file join $output ${clock}_net.rpt]
        report_route_status -of_objects [get_nets $clock] -file [file join $output ${clock}_route.rpt]
    }
    set stream [open [file join $output partial_clock_aliases.tsv] {WRONLY CREAT EXCL}]
    puts $stream "net\tstatus\tmatching_parent_clock"
    set counts [dict create]
    set partial_counts [dict create clk 0 host_clk 0 mgmt_clk 0]
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] {
        dict incr counts $status
        if {$status ne "PARTIAL"} { continue }
        set matches {}
        set route [get_property ROUTE $net]
        dict for {clock canonical} $routes {
            if {[string equal $route $canonical]} { lappend matches $clock }
        }
        if {[llength $matches] != 1} { error "Unexplained PARTIAL net: $net" }
        set clock [lindex $matches 0]
        dict incr partial_counts $clock
        puts $stream "$net\t$status\t$clock"
    }
    close $stream
    set stream [open [file join $output audit_totals.txt] {WRONLY CREAT EXCL}]
    puts $stream "PARTIAL_COUNTS $partial_counts\nROUTE_STATUS_COUNTS $counts"
    close $stream
    foreach clock {clk host_clk mgmt_clk} {
        if {[dict get $partial_counts $clock] == 0} { error "No partial aliases: $clock" }
    }
    set stream [open [file join $output port_context.tsv] {WRONLY CREAT EXCL}]
    puts $stream "port\tdirection\tHD.CLK_SRC\tHD.PARTPIN_LOCS"
    foreach port [get_ports] {
        puts $stream "$port\t[get_property DIRECTION $port]\t[get_property HD.CLK_SRC $port]\t[get_property HD.PARTPIN_LOCS $port]"
    }
    close $stream
    # TIMING-37: record the real endpoints of every AXI-Lite R bundle source.
    # A fanout diagnostic is retained, never converted into a new exception.
    set cells [get_cells -quiet -hierarchical -filter {NAME =~ *clock_conv_lite_resp_r/handshake/dest_hsdata_ff_reg*}]
    if {![llength $cells]} { error "Read-response capture registers missing" }
    set stream [open [file join $output axil_read_fanout.tsv] {WRONLY CREAT EXCL}]
    puts $stream "capture_pin\tdriver\tendpoint_count\tendpoints"
    foreach cell $cells {
        set dst [get_pins $cell/D]
        set nets [get_nets -segments -of_objects $dst]
        set drivers [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]
        if {[llength $drivers] != 1} { error "Unexpected capture driver: $dst" }
        set endpoints [lsort [all_fanout -flat -endpoints_only -from $drivers]]
        puts $stream "$dst\t$drivers\t[llength $endpoints]\t[join $endpoints {;}]"
    }
    close $stream
    report_route_status -file [file join $output route.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_bus_skew -warn_on_violation -file [file join $output bus_skew.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    puts "NTRPU_FFN_SUPERVISED_CLOCK_AUDIT_EXPORTED; read-only, three parent gaps, read-response fanout and131 INIT registers recorded"
}

proc guarded_structure {output} {
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
}

proc guarded_polarity {output} {
    set cells [get_cells -quiet -hierarchical -filter {NAME =~ compute/supervisor/*/release_ff_reg* || NAME =~ compute/status_cdc*.sync/level_ff_reg* || NAME =~ *transport/armed_cdc/level_ff_reg*}]
    if {[llength $cells] != 51} {error "Expected all51 level stages"}
    set f [open [file join $output cell_properties.txt] {WRONLY CREAT EXCL}]
    foreach cell [lsort $cells] {
        puts $f "CELL $cell"
        puts $f [report_property -all $cell -return_string]
    }
    close $f
    set f [open [file join $output physical_level_pins.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tinit\tasync_reg\tclr_inverted\tclock\tpin\tdrivers\tports\tdriver_refs\tdriver_inits"
    foreach cell [lsort $cells] {
        set properties [list_property $cell]
        if {[lsearch -exact $properties IS_CLR_INVERTED] < 0} {error "CLR inversion property unavailable; inspect cell_properties"}
        foreach pin {D C CE CLR} {
            set net [get_nets -segments -of_objects [get_pins $cell/$pin]]
            set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
            set ports [get_ports -quiet -of_objects $net]
            set refs {};set inits {}
            foreach driver $drivers {
                set owner [get_cells -of_objects $driver]
                lappend refs [get_property REF_NAME $owner]
                lappend inits [get_property INIT $owner]
            }
            puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]\t[get_property ASYNC_REG $cell]\t[get_property IS_CLR_INVERTED $cell]\t[get_clocks -of_objects [get_pins $cell/C]]\t$pin\t[join $drivers {;}]\t[join $ports {;}]\t[join $refs {;}]\t[join $inits {;}]"
        }
    }
    close $f
    set buffers [get_cells -quiet -hierarchical -filter {REF_NAME == BUFGCE}]
    set f [open [file join $output buffer_properties.txt] {WRONLY CREAT EXCL}]
    set pins [open [file join $output buffer_pins.tsv] {WRONLY CREAT EXCL}]
    puts $pins "cell\tpin\tdrivers\tports\tdriver_refs\tdriver_inits"
    foreach cell [lsort $buffers] {
        puts $f "CELL $cell"
        puts $f [report_property -all $cell -return_string]
        foreach pin {I CE} {
            set net [get_nets -segments -of_objects [get_pins $cell/$pin]]
            set drivers [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION == OUT}]
            set refs {};set inits {}
            foreach driver $drivers {set owner [get_cells -of_objects $driver];lappend refs [get_property REF_NAME $owner];lappend inits [get_property INIT $owner]}
            puts $pins "$cell\t$pin\t[join $drivers {;}]\t[join [get_ports -quiet -of_objects $net] {;}]\t[join $refs {;}]\t[join $inits {;}]"
        }
    }
    close $f;close $pins
    write_xdc -constraints all [file join $output unchanged_constraints.xdc]
    puts "NTRPU_PHYSICAL_LEVEL_POLARITY_EXPORTED stages=51; independent review pending"
}

proc guarded_boundary {output} {
    help get_drc_violations
    help get_nets
    report_drc -checks [get_drc_checks RTSTAT-10] -file [file join $output boundary_drc.rpt]
    set violations [get_drc_violations -quiet *RTSTAT-10*]
    if {[llength $violations]!=1} {error "Expected one RTSTAT-10 finding"}
    report_property -all $violations -file [file join $output violation_properties.rpt]
    set nets [get_nets -quiet -of_objects $violations]
    if {![llength $nets]} {error "DRC finding did not expose its nets"}
    set f [open [file join $output boundary_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\troute_status\tdrivers\tloads\tports\tdriver_sites\tload_sites"
    set with_loads 0
    set with_ports 0
    foreach net [lsort $nets] {
        set segments [get_nets -segments $net]
        set drivers [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]
        set loads [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]
        set ports [get_ports -quiet -of_objects $segments]
        set driver_sites {}
        set load_sites {}
        foreach cell [get_cells -quiet -of_objects $drivers] {lappend driver_sites [get_property LOC $cell]}
        foreach cell [get_cells -quiet -of_objects $loads] {lappend load_sites [get_property LOC $cell]}
        if {[llength $loads]} {incr with_loads}
        if {[llength $ports]} {incr with_ports}
        puts $f "$net\t[get_property ROUTE_STATUS $net]\t[join [lsort $drivers] {;}]\t[join [lsort $loads] {;}]\t[join [lsort $ports] {;}]\t[join [lsort -unique $driver_sites] {;}]\t[join [lsort -unique $load_sites] {;}]"
    }
    close $f
    set f [open [file join $output boundary_totals.txt] {WRONLY CREAT EXCL}]
    puts $f "nets=[llength $nets]\nwith_leaf_loads=$with_loads\nwith_ports=$with_ports"
    close $f
    write_xdc -constraints all [file join $output unchanged_constraints.xdc]
    puts "NTRPU_SUPERVISED_BOUNDARY_EXPORTED nets=[llength $nets]; classification pending"
}

proc guarded_route_audit {arguments} {
    if {[llength $arguments]!=2 || [llength [get_projects -quiet]]} {error "Expected ROUTED_DCP NEW_DIRECTORY"}
    lassign $arguments checkpoint root_output
    set root_output [file normalize $root_output]
    if {[file exists $root_output]} {error "Output exists"}
    file mkdir $root_output
    file copy $checkpoint [file join $root_output input.dcp]
    open_checkpoint [file join $root_output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_ffn_guarded_fabric" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]]!=3} {error "Unexpected guarded design"}
    foreach {clock period site} {clk 5.0 BUFGCE_X0Y60 host_clk 4.0 BUFGCE_X0Y61 mgmt_clk 10.0 BUFGCE_X0Y62} {
        if {[get_property PERIOD [get_clocks $clock]]!=$period ||
            [get_property HD.CLK_SRC [get_ports $clock]] ne $site} {error "Changed provisional clock context"}
    }
    write_xdc -constraints all [file join $root_output before.xdc]
    foreach section {clock structure polarity boundary} {
        set output [file join $root_output $section]
        file mkdir $output
        puts "NTRPU_GUARDED_ROUTE_AUDIT_BEGIN $section"
        guarded_$section $output
        puts "NTRPU_GUARDED_ROUTE_AUDIT_END $section"
    }
    write_xdc -constraints all [file join $root_output after.xdc]
    puts NTRPU_GUARDED_ROUTE_AUDIT_EXPORTED
    close_project
}
if {[catch {guarded_route_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
