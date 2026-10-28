set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc section_constraints {output checkpoint} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    # Read-only inspection of the sealed synthesized design. No reset/rewrite
    # of timing constraints, no new false paths and no checkpoint replacement.
    foreach command {report_exceptions report_cdc report_bus_skew report_waivers get_timing_paths reset_path reset_false_path} {
        puts "NTRPU_HELP_BEGIN $command"
        if {[llength [info commands $command]]} {
            if {[catch {help $command} diagnostic]} {puts "HELP_DIAGNOSTIC $diagnostic"}
        } else {puts "COMMAND_UNAVAILABLE $command"}
        puts "NTRPU_HELP_END $command"
    }
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_drc -file [file join $output drc.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_exceptions -coverage -file [file join $output exceptions_coverage.rpt]
    report_waivers -file [file join $output waivers.rpt]
    set f [open [file join $output ram_scope.tsv] {WRONLY CREAT EXCL}]
    puts $f "scope\tclock\tcell\tref\tclock_pin\toutput_pin"
    # Evaluate the same all_fanout/filter expressions shown in the effective
    # exception report. Preserve every matched cell, including non-IP cells.
    foreach clock {host_clk device_clk} {
        set cells [filter [all_fanout -from [get_ports $clock] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
        foreach c $cells {
            set scope [expr {[string match {cdc/*} $c]?"vendor_cdc":"custom_rtl"}]
            set cp [get_pins -of $c -filter {REF_PIN_NAME==CLK}]
            set op [get_pins -of $c -filter {REF_PIN_NAME==O}]
            puts $f "$scope\t$clock\t$c\t[get_property REF_NAME $c]\t$cp\t$op"
        }
    }
    close $f
    set f [open [file join $output fault_cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tref_name\tinit\tasync_reg"
    foreach c [get_cells -hierarchical -filter {NAME =~ *admission_sources* || NAME =~ *fault_sources_reg* || NAME =~ admission/*fault_reg* || NAME =~ admission/guard/*quarantine*}] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property -quiet INIT $c]\t[get_property -quiet ASYNC_REG $c]"
    }
    close $f
    set f [open [file join $output fault_outputs.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tnet\tdriver_pin\tdriver_ref"
    for {set i 0} {$i<7} {incr i} {
        set port [get_ports [format {admission_fault_sources[%d]} $i]]
        set nets [get_nets -segments -of $port]
        foreach pin [get_pins -of $nets -filter {DIRECTION==OUT}] {
            puts $f "$port\t[get_nets -of $pin]\t$pin\t[get_property REF_NAME [get_cells -of $pin]]"
        }
    }
    close $f
    set f [open [file join $output reset_port_fanout.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tcell\tref_name"
    foreach reset {cold_rst_n host_rst_n device_rst_n} {
        foreach c [all_fanout -from [get_ports $reset] -flat -endpoints_only -only_cells] {
            puts $f "$reset\t$c\t[get_property REF_NAME $c]"
        }
    }
    close $f
    close_design
    puts NTRPU_AXI4_SECTION_CONSTRAINTS_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {section_constraints [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
