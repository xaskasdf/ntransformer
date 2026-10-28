# Narrow reset exceptions, and structural evidence independent of CDC labels.
proc dual_release_chains {output constrain} {
    set cells [get_cells -hier -quiet -filter {NAME =~ core/startup/*/release_ff_reg*}]
    if {[llength $cells]!=45} {error "Expected fifteen three-stage custom chains, got [llength $cells]"}
    set groups [dict create]
    foreach cell $cells {dict lappend groups [file dirname $cell] $cell}
    if {[dict size $groups]!=15} {error "Expected fifteen release chains"}
    set f [open [file join $output release_chains.tsv] w]
    puts $f "cell\tref\tinit\tasync_reg\tclock\tdata_driver\tclear_driver\tloc"
    dict for {name group} $groups {
        if {[llength $group]!=3} {error "Three stages required: $name"}
        foreach cell [lsort $group] {
            set ref [get_property REF_NAME $cell];set init [get_property INIT $cell]
            set attribute [get_property ASYNC_REG $cell]
            if {$ref ne "FDCE" || $init ne "1'b0" || $attribute ni {1 TRUE true}} {error "Unexpected release primitive: $cell"}
            set clocks [get_clocks -of_objects [get_pins $cell/C]]
            if {[llength $clocks]!=1} {error "Release clock ambiguous"}
            set drivers [dict create]
            foreach pin {D CLR} {
                set nets [get_nets -segments -of_objects [get_pins $cell/$pin]]
                set dd [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION==OUT}]
                if {[llength $dd]!=1} {error "Expected one $pin driver on $cell"}
                dict set drivers $pin $dd
            }
            puts $f "$cell\t$ref\t$init\t$attribute\t$clocks\t[dict get $drivers D]\t[dict get $drivers CLR]\t[get_property LOC $cell]"
        }
        if {$constrain} {
            set pins [get_pins -of_objects $group -filter {REF_PIN_NAME==CLR}]
            if {[llength $pins]!=3} {error "Only asynchronous clear pins may be excepted"}
            set_false_path -to $pins
        }
    }
    close $f
}

proc dual_physical_reports {output} {
    dual_release_chains $output 0
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_utilization -cells [get_cells core] -file [file join $output core_utilization.rpt]
    report_utilization -cells [get_cells core/startup] -file [file join $output startup_utilization.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_timing -delay_type max -max_paths 20 -file [file join $output setup_paths.rpt]
    report_timing -delay_type min -max_paths 20 -file [file join $output hold_paths.rpt]
    report_cdc -details -all_checks_per_endpoint -file [file join $output cdc.rpt]
    report_cdc -details -all_checks_per_endpoint -no_waiver -file [file join $output cdc_unwaived.rpt]
    report_bus_skew -delay_type min_max -max_paths 1 -warn_on_violation -file [file join $output bus_skew.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    report_route_status -file [file join $output route_status.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_waivers -file [file join $output waivers.rpt]
    write_xdc -constraints all [file join $output implemented.xdc]
    set f [open [file join $output metrics.txt] w]
    foreach {label clock} {SOURCE source_clk MANAGEMENT mgmt_clk MEMORY0 memory0_clk MEMORY1 memory1_clk} {
        puts $f "${label}_PERIOD_NS=[get_property PERIOD [get_clocks $clock]]"
        foreach {kind delay} {SETUP max HOLD min} {
            set paths [get_timing_paths -from [get_clocks $clock] -to [get_clocks $clock] -delay_type $delay -max_paths 1]
            if {![llength $paths]} {error "No $kind path on $clock"}
            puts $f "${label}_${kind}_NS=[get_property SLACK $paths]"
        }
    }
    puts $f "CLOCKS=[llength [get_clocks]]"
    puts $f "BUFGS=[llength [get_cells -hier -filter {REF_NAME==BUFGCE}]]"
    puts $f "DRC_ERRORS=[llength [get_drc_violations -quiet -filter {SEVERITY==Error}]]"
    puts $f "BLACKBOXES=[llength [get_cells -hier -quiet -filter {IS_BLACKBOX==1}]]"
    puts $f "LATCHES=[llength [get_cells -hier -quiet -filter {REF_NAME==LDCE || REF_NAME==LDPE}]]"
    close $f
    set f [open [file join $output cross_clock_paths.tsv] w]
    puts $f "source\tdestination\tremaining_timed_paths\tendpoint"
    foreach a [get_clocks] {
        foreach b [get_clocks] {
            if {$a eq $b} {continue}
            set p [get_timing_paths -quiet -from $a -to $b -max_paths 1]
            set endpoint {};if {[llength $p]} {set endpoint [get_property ENDPOINT_PIN $p]}
            puts $f "$a\t$b\t[llength $p]\t$endpoint"
        }
    }
    close $f
    set f [open [file join $output clock_binding.txt] w]
    set converters [get_cells -hier -filter {REF_NAME =~ ntrpu_peer_memory_cdc*}]
    puts $f "CONVERTERS=[llength $converters]"
    if {[llength $converters]!=2} {error "Expected two converters"}
    foreach cell $converters {
        foreach port {s_axi_aclk m_axi_aclk} {
            set pin [get_pins -of_objects $cell -filter "REF_PIN_NAME == $port"]
            set clocks [get_clocks -of_objects $pin]
            if {[llength $pin]!=1 || [llength $clocks]!=1} {error "Ambiguous CDC clock"}
            puts $f "$cell/$port=$clocks"
        }
    }
    foreach cell [get_cells -hier -filter {REF_NAME==BUFGCE}] {
        set net [get_nets -of_objects [get_pins $cell/O]]
        puts $f "BUFG=$cell CLOCK=[get_clocks -of_objects $net] ROUTE=[get_property ROUTE_STATUS $net]"
    }
    close $f
    set f [open [file join $output startup_registers.tsv] w]
    puts $f "cell\tref\tinit\tasync_reg\tclock\tdata_driver\tasync_pin\tasync_drivers\tasync_ports\tloc"
    foreach cell [lsort [get_cells -hier -filter {NAME =~ core/startup/* && IS_PRIMITIVE==1 && REF_NAME =~ FD*}]] {
        set dd [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects [get_pins $cell/D]] -filter {DIRECTION==OUT}]
        set ap [get_pins -quiet -of_objects $cell -filter {REF_PIN_NAME==CLR || REF_PIN_NAME==PRE}]
        set ad {};set ports {}
        if {[llength $ap]} {
            set net [get_nets -segments -of_objects $ap]
            set ad [get_pins -quiet -leaf -of_objects $net -filter {DIRECTION==OUT}]
            set ports [get_ports -quiet -of_objects $net]
        }
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]\t[get_property ASYNC_REG $cell]\t[get_clocks -of_objects [get_pins $cell/C]]\t$dd\t$ap\t$ad\t$ports\t[get_property LOC $cell]"
    }
    close $f
}
