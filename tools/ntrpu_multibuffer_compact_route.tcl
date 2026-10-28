# Isolated implementation from an immutable, functionally validated checkpoint.
set script_dir [file dirname [file normalize [info script]]]
proc multibuffer_route {arguments} {
    global script_dir
    if {[llength $arguments] != 4} {error "Expected RESERVE SYNTH_DCP ASSUMED_BUFGCE_SITE NEW_DIRECTORY"}
    lassign $arguments reserve checkpoint clock_site output
    if {$reserve ni {4 8}} {error "Unvalidated reserve"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    report_property -all [current_design] -file [file join $output input_design.rpt]
    report_property -all [current_fileset] -file [file join $output input_fileset.rpt]
    report_clocks -file [file join $output input_clocks.rpt]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property TOP [current_design]] ne "ntrpu_peer_multibuffer_compact" ||
        [llength [get_clocks]] != 1 ||
        [get_property PERIOD [get_clocks multibuffer_clk]] != 5.0} {error "Unexpected checkpoint"}
    set site [get_sites -quiet $clock_site]
    if {[llength $site] != 1 || [get_property SITE_TYPE $site] ne "BUFGCE"} {error "Invalid clock site"}
    report_property -all $site -file [file join $output assumed_clock_site.rpt]
    set_property HD.CLK_SRC $clock_site [get_ports clk]
    read_xdc [file join $script_dir .. fpga constraints ntrpu_multibuffer_physical_ooc.xdc]
    write_xdc -constraints all [file join $output input_context.xdc]
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Provisional compact multibuffer OOC context\n"
    puts $f "5 ns clock; assumed parent BUFGCE at $clock_site, using HD.CLK_SRC."
    puts $f "Input delay max/min 1.000/0.200 ns; output delay max/min 1.000/-0.100 ns."
    puts $f "Additional setup/hold uncertainty 0.100/0.050 ns. All data/control ports included."
    puts $f "Budgets are proposed synchronous parent contracts, not measured external device delays."
    puts $f "No physical parent BUFG instantiated, boundary partpins, board pins, pblock, PCIe/HBM IP or CDC."
    puts $f "Port and clock routing must be requalified in the integrated parent. No timing waivers."
    close $f
    set failed [catch {
        opt_design
        place_design
        write_checkpoint [file join $output placed.dcp]
        report_timing_summary -report_unconstrained -file [file join $output placed_timing.rpt]
        phys_opt_design
        route_design
    } message options]
    set f [open [file join $output route_status.txt] {WRONLY CREAT EXCL}]
    puts $f "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} {puts $f [dict get $options -errorinfo]}
    close $f
    write_checkpoint [file join $output route_attempt.dcp]
    report_route_status -file [file join $output route.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_property -all [get_nets clk] -file [file join $output clock_net.rpt]
    report_route_status -of_objects [get_nets clk] -file [file join $output clock_route.rpt]
    write_xdc -constraints all [file join $output implemented.xdc]
    set f [open [file join $output paths.tsv] {WRONLY CREAT EXCL}]
    puts $f "scope\ttype\tslack_ns\tsource\tdestination"
    foreach scope {all internal input output} {
        set filters {}
        if {$scope eq "internal"} {set filters [list -from [all_registers] -to [all_registers]]}
        if {$scope eq "input"} {set filters [list -from [get_ports -filter {DIRECTION == IN && NAME != clk}]]}
        if {$scope eq "output"} {set filters [list -to [get_ports -filter {DIRECTION == OUT}]]}
        foreach delay {max min} {
            set paths [get_timing_paths -quiet -delay_type $delay -max_paths 1 {*}$filters]
            if {[llength $paths] == 1} {
                puts $f "$scope\t$delay\t[get_property SLACK $paths]\t[get_property STARTPOINT_PIN $paths]\t[get_property ENDPOINT_PIN $paths]"
                report_timing -delay_type $delay -max_paths 10 -path_type full_clock_expanded \
                    {*}$filters -file [file join $output ${scope}_${delay}_paths.rpt]
            } else {error "Missing timing path for $scope $delay"}
        }
    }
    close $f
    set counts [dict create]
    set gaps 0
    set f [open [file join $output route_gaps.txt] {WRONLY CREAT EXCL}]
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] route [get_property ROUTE $nets] {
        dict incr counts $status
        if {[string match -nocase *gap* $route]} {
            incr gaps
            puts $f "NET $net STATUS $status\n$route"
        }
    }
    puts $f "ROUTE_STRING_GAP_NETS $gaps\nROUTE_STATUS_COUNTS $counts"
    close $f
    set f [open [file join $output route_counts.tsv] {WRONLY CREAT EXCL}]
    dict for {key value} $counts {puts $f "$key\t$value"}
    close $f
    set f [open [file join $output port_context.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection\tHD.CLK_SRC\tHD.PARTPIN_LOCS"
    foreach p [get_ports] {
        puts $f "$p\t[get_property DIRECTION $p]\t[get_property HD.CLK_SRC $p]\t[get_property HD.PARTPIN_LOCS $p]"
    }
    close $f
    set f [open [file join $output summary.tsv] {WRONLY CREAT EXCL}]
    foreach {key value} [list part [get_property PART [current_design]] tool_version [version -short] \
        reserve_banks $reserve period_ns [get_property PERIOD [get_clocks multibuffer_clk]] \
        clock_count [llength [get_clocks]] assumed_clock_site $clock_site \
        route_command_failed $failed route_string_gap_nets $gaps \
        max_threads [get_param general.maxThreads] \
        latch_count [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]] \
        blackbox_count [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]] {puts $f "$key\t$value"}
    close $f
    if {$failed} {error "Implementation command failed; reports preserved"}
    write_verilog -mode funcsim [file join $output multibuffer_funcsim.v]
    puts "MULTIBUFFER_COMPACT_ROUTE_EXPORTED RESERVE=$reserve; OOC context, integration pending"
    close_project
}
if {[catch {multibuffer_route $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
