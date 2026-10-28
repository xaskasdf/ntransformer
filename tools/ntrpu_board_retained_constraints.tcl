proc retained_constraints {output checkpoint mode} {
    if {[file exists $output]} {error "Output exists"}
    if {$mode ni {host start}} {error "Unknown retained design"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint $checkpoint
    # Read the immutable synthesized DCP. No timing exceptions are changed,
    # no waivers are added, and no new checkpoint is written by this audit.
    report_methodology -file [file join $output methodology.rpt]
    report_drc -file [file join $output drc.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_cdc -details -show_waiver -file [file join $output cdc_with_waivers.rpt]
    report_exceptions -coverage -file [file join $output exceptions_coverage.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_waivers -file [file join $output waivers.rpt]
    write_xdc -constraints all [file join $output effective.xdc]
    if {$mode eq "host"} {
        report_bus_skew -file [file join $output bus_skew.rpt]
        set f [open [file join $output ram_scope.tsv] {WRONLY CREAT EXCL}]
        puts $f "scope\tclock\tcell\tref\tclock_pin\toutput_pin"
        foreach clock {host_clk source_clk mgmt_clk} {
            set cells [filter [all_fanout -from [get_ports $clock] -flat -endpoints_only -only_cells] {PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM}]
            foreach c $cells {
                set scope custom_rtl
                if {[string match {frame_path/cdc/*} $c] || [string match {control_path/shutdown_manager/*} $c]} {set scope vendor_ip}
                puts $f "$scope\t$clock\t$c\t[get_property REF_NAME $c]\t[get_pins -of $c -filter {REF_PIN_NAME==CLK}]\t[get_pins -of $c -filter {REF_PIN_NAME==O}]"
            }
        }
        close $f
        set f [open [file join $output custom_ram_paths.tsv] {WRONLY CREAT EXCL}]
        puts $f "cell\tclock_pin\toutput_pin\tloads\tconstrained_count\tslack\tstartpoint\tendpoint"
        foreach c [get_cells -hierarchical -filter {(PRIMITIVE_SUBGROUP==dram || PRIMITIVE_SUBGROUP==LUTRAM) && NAME !~ frame_path/cdc/* && NAME !~ control_path/shutdown_manager/*}] {
            set cp [get_pins -of $c -filter {REF_PIN_NAME==CLK}]
            set op [get_pins -of $c -filter {REF_PIN_NAME==O}]
            set loads [all_fanout -from $op -flat -endpoints_only]
            set paths [get_timing_paths -quiet -no_report_unconstrained -from $cp -through $op -max_paths 1 -delay_type max]
            set slack {};set start {};set end {}
            if {[llength $paths]} {
                set p [lindex $paths 0]
                set slack [get_property SLACK $p]
                set start [get_property STARTPOINT_PIN $p]
                set end [get_property ENDPOINT_PIN $p]
            }
            puts $f "$c\t$cp\t$op\t[llength $loads]\t[llength $paths]\t$slack\t$start\t$end"
        }
        close $f
    } else {
        set f [open [file join $output retained_state_pins.tsv] {WRONLY CREAT EXCL}]
        puts $f "cell\tref\tinit\tasync_reg\tpin\tnet\tdriver"
        foreach c [get_cells -hierarchical -filter {NAME =~ opened_reg* || NAME =~ hold_reset_reg* || NAME =~ clear_counters_reg* || NAME =~ block_admission_reg* || NAME =~ state_reg* || NAME =~ management_reset/*}] {
            foreach pin [get_pins -of $c -filter {DIRECTION==IN}] {
                set nets [get_nets -segments -of $pin]
                set drivers [get_pins -of $nets -filter {DIRECTION==OUT}]
                puts $f "$c\t[get_property REF_NAME $c]\t[get_property -quiet INIT $c]\t[get_property -quiet ASYNC_REG $c]\t$pin\t[get_nets -of $pin]\t$drivers"
            }
        }
        close $f
        set f [open [file join $output external_reset_fanout.tsv] {WRONLY CREAT EXCL}]
        puts $f "cell\tref\tinit"
        foreach c [all_fanout -from [get_ports external_reset_async] -flat -endpoints_only -only_cells] {
            puts $f "$c\t[get_property REF_NAME $c]\t[get_property -quiet INIT $c]"
        }
        close $f
    }
    close_design
    puts "NTRPU_RETAINED_CONSTRAINTS_EXPORTED $mode"
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {retained_constraints [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]] [lindex $argv 2]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
