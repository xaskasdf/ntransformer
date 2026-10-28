# Read-only inspection of the implemented inventory; never optimize/place/route here.
proc inventory_boundary_audit {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected SYNTH_DCP IMPLEMENTED_DCP NEW_OUTPUT"
    }
    lassign $arguments original implemented output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    foreach command {open_checkpoint opt_design place_design route_design report_route_status report_timing_summary report_timing all_registers report_high_fanout_nets write_edif help} {
        set text [help $command]
        if {[string trim $text] eq ""} {error "Empty help return: $command"}
        set f [open [file join $output help_${command}.txt] {WRONLY CREAT EXCL}]
        puts $f $text
        close $f
        puts "NTRPU_HELP_CAPTURED $command characters=[string length $text]"
    }
    set f [open [file join $output VERSION.txt] {WRONLY CREAT EXCL}]
    puts $f [version]
    close $f
    set ports_before {}
    foreach variant {synthesis implemented} checkpoint [list $original $implemented] {
        file copy $checkpoint [file join $output ${variant}_input.dcp]
        open_checkpoint [file join $output ${variant}_input.dcp]
        if {[get_property TOP [current_design]] ne "ntrpu_pcie_inventory_join" ||
            [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected checkpoint"}
        if {[llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {
            error "Unexpected clock inventory/period"
        }
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {
            error "Unexpected blackbox or transparent storage"
        }
        set ports [lsort [get_ports]]
        set inventory {}
        set f [open [file join $output ${variant}_ports.tsv] {WRONLY CREAT EXCL}]
        puts $f "port\tdirection\tpartition_pin_locations"
        foreach port $ports {
            set direction [get_property DIRECTION $port]
            lappend inventory [list $port $direction]
            puts $f "$port\t$direction\t[get_property HD.PARTPIN_LOCS $port]"
        }
        close $f
        report_property -all [get_clocks pcie_user_clk] -file [file join $output ${variant}_clock.rpt]
        report_property -all [get_ports clk] -file [file join $output ${variant}_clock_port.rpt]
        write_xdc -constraints all [file join $output ${variant}.xdc]
        if {$variant eq "synthesis"} {
            if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected synthesis clock context"}
            set ports_before $inventory
            close_project
        } else {
            if {$inventory ne $ports_before} {error "Public interface changed"}
            if {[get_property HD.CLK_SRC [get_ports clk]] ne "BUFGCE_X0Y60"} {error "Unexpected implemented clock context"}
        }
    }
    # Export through Vivado after reopening the exact archived DCP: the embedded
    # implemented EDIF uses an encoded Xilinx header, not a bare Zstd stream.
    write_edif [file join $output reloaded.edf]
    set f [open [file join $output reloaded_primitives.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive"
    foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]"
    }
    close $f
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output timing.rpt]
    # Additional path subsets diagnose internal state feedback without replacing
    # the complete timing summary or introducing any timing exception.
    set registers [all_registers]
    if {![llength $registers]} {error "Missing register timing endpoints"}
    set f [open [file join $output register_timing_endpoints.txt] {WRONLY CREAT EXCL}]
    foreach object [lsort $registers] {puts $f $object}
    close $f
    foreach {kind delay} {setup max hold min} {
        report_timing -from $registers -to $registers -delay_type $delay -max_paths 20 -input_pins -file [file join $output register_to_register_${kind}.rpt]
    }
    report_high_fanout_nets -timing -load_types -max_nets 30 -file [file join $output high_fanout_timing.rpt]
    report_high_fanout_nets -slr -max_nets 30 -file [file join $output high_fanout_slr.rpt]
    report_route_status -ignore_cache -file [file join $output route.rpt]
    foreach kind {PARTIAL GAPS UNPLACED UNROUTED CONFLICTS ANTENNAS NODRIVER MULTI_DRIVER} {
        report_route_status -route_type $kind -list_all_nets -show_all -file [file join $output route_${kind}.rpt]
    }
    set f [open [file join $output route_flags.tsv] {WRONLY CREAT EXCL}]
    puts $f "flag\tvalue"
    foreach flag {PLACED_FULLY PARTIALLY_ROUTED ROUTED_FULLY ERRORS_IN_ROUTES} {
        puts $f "$flag\t[report_route_status -boolean_check $flag]"
    }
    close $f
    set clock_net [get_nets -of_objects [get_ports clk]]
    if {[llength $clock_net] != 1} {error "Nonunique parent clock net"}
    set clock_route [get_property ROUTE $clock_net]
    report_property -all $clock_net -file [file join $output parent_clock_net.rpt]
    set f [open [file join $output route_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus"
    set details [open [file join $output incomplete_nets.tsv] {WRONLY CREAT EXCL}]
    puts $details "net\tstatus\tmatching_parent_route\tports\tdrivers\tloads\troute"
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] {
        puts $f "$net\t$status"
        if {$status ni {PARTIAL UNROUTED UNPLACED CONFLICTS ANTENNAS NODRIVER MULTI_DRIVER}} {continue}
        set segments [get_nets -segments $net]
        set net_ports [get_ports -quiet -of_objects $segments]
        set drivers [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]
        set loads [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]
        set net_route [get_property ROUTE $net]
        set matches [expr {$clock_route ne "" && $net_route eq $clock_route}]
        puts $details "$net\t$status\t$matches\t[join [lsort $net_ports] {;}]\t[join [lsort $drivers] {;}]\t[join [lsort $loads] {;}]\t$net_route"
    }
    close $details
    close $f
    report_drc -file [file join $output drc.rpt]
    set f [open [file join $output drc_objects.tsv] {WRONLY CREAT EXCL}]
    puts $f "index\tviolation\tnets\tcells\tports"
    set index 0
    foreach violation [lsort [get_drc_violations -quiet]] {
        report_property -all $violation -file [file join $output drc_violation_${index}.rpt]
        puts $f "$index\t$violation\t[join [lsort [get_nets -quiet -of_objects $violation]] {;}]\t[join [lsort [get_cells -quiet -of_objects $violation]] {;}]\t[join [lsort [get_ports -quiet -of_objects $violation]] {;}]"
        incr index
    }
    close $f
    report_exceptions -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    write_xdc -constraints all [file join $output after_audit.xdc]
    puts "NTRPU_INVENTORY_BOUNDARY_AUDIT_EXPORTED violations=$index"
    close_project
}
if {[catch {inventory_boundary_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
