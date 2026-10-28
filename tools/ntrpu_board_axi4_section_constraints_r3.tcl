set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc section_constraints {output checkpoint} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    # Inspect the sealed synthesized design, then apply one private negative
    # timing control in memory. Never save or replace the input checkpoint.
    foreach command {report_exceptions report_cdc report_bus_skew report_waivers get_timing_paths get_waivers reset_path reset_false_path} {
        puts "NTRPU_HELP_BEGIN $command"
        if {[llength [info commands $command]]} {
            if {[catch {help $command} diagnostic]} {puts "HELP_DIAGNOSTIC $diagnostic"} else {puts $diagnostic}
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
    # Prove that the effective native exceptions leave custom RAM paths
    # timed. The printed selectors can lose scoped context when displayed.
    set f [open [file join $output custom_ram_paths.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tclock_pin\toutput_pin\tloads\tpath_count\tslack\tstartpoint\tendpoint"
    set first_clock {};set first_output {};set first_slack {}
    foreach c [get_cells -hierarchical -filter {(PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM) && NAME !~ cdc/*}] {
        set cp [get_pins -of $c -filter {REF_PIN_NAME==CLK}]
        set op [get_pins -of $c -filter {REF_PIN_NAME==O}]
        set loads [all_fanout -from $op -flat -endpoints_only]
        set paths [get_timing_paths -quiet -from $cp -through $op -max_paths 1 -delay_type max]
        set slack {};set start {};set end {}
        if {[llength $paths]} {
            set path [lindex $paths 0]
            set slack [get_property SLACK $path]
            set start [get_property STARTPOINT_PIN $path]
            set end [get_property ENDPOINT_PIN $path]
            if {$first_clock eq {} && [string is double -strict $slack] && $slack ne "inf"} {
                set first_clock $cp;set first_output $op;set first_slack $slack
                report_property -all -file [file join $output ram_path_properties.rpt] $path
                report_timing -from $cp -through $op -max_paths 1 -file [file join $output custom_ram_positive_path.rpt]
            }
        }
        puts $f "$c\t$cp\t$op\t[llength $loads]\t[llength $paths]\t$slack\t$start\t$end"
    }
    close $f
    # Intentional negative control on the in-memory inspection copy only.
    # Never write a checkpoint or reuse this modified timing state elsewhere.
    if {$first_clock ne {}} {
        set_false_path -from $first_clock -through $first_output
        set paths [get_timing_paths -quiet -from $first_clock -through $first_output -max_paths 1 -delay_type max]
        set f [open [file join $output custom_ram_negative_control.tsv] {WRONLY CREAT EXCL}]
        puts $f "clock_pin\toutput_pin\tbaseline_slack\tmasked_path_count"
        puts $f "$first_clock\t$first_output\t$first_slack\t[llength $paths]"
        close $f
    }
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
