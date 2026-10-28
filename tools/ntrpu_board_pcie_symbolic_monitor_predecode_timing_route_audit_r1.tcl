# Record actual cell placement before and after routing, including router changes.
proc monitor_timing_route_geometry {output name} {
    set cells [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE && REF_NAME != VCC && REF_NAME != GND}]]
    set f [open [file join $output ${name}_geometry.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive\tloc\tbel"
    foreach cell $cells primitive [get_property REF_NAME $cells] loc [get_property LOC $cells] bel [get_property BEL $cells] {
        puts $f "$cell\t$primitive\t$loc\t$bel"
    }
    close $f
}

proc monitor_predecode_ed_timing {output} {
    set e_registers [get_cells -hierarchical -filter {REF_NAME == FDRE && NAME =~ e_stage/*}]
    set d_registers [get_cells -hierarchical -filter {REF_NAME == FDRE && NAME =~ d_stage/symbolic_rows_reg*}]
    if {![llength $e_registers] || ![llength $d_registers]} {error "Missing E/D register subset"}
    set f [open [file join $output ed_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "stage\tregister"
    foreach name [lsort [get_property NAME $e_registers]] {puts $f "E\t$name"}
    foreach name [lsort [get_property NAME $d_registers]] {puts $f "D\t$name"}
    close $f
    foreach {kind delay} {setup max hold min} {
        report_timing -from $e_registers -to $d_registers -delay_type $delay -max_paths 20 -input_pins -file [file join $output ed_${kind}.rpt]
    }
}

# Read-only inspection of the implemented inventory; never optimize/place/route here.
proc monitor_clock_route_audit {arguments} {
    if {[llength $arguments] != 4 || [llength [get_projects -quiet]]} {
        error "Expected PLACED_DCP IMPLEMENTED_DCP EXPECTED_TOP NEW_OUTPUT"
    }
    lassign $arguments original implemented expected_top output
    if {$expected_top ni {ntrpu_pcie_symbolic_monitor_r1 ntrpu_pcie_symbolic_monitor_predecode_r1}} {error "Unsupported comparison top"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
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
    file copy [file join $::env(XILINX_VIVADO) data verilog src unisims BUFGCE.v] [file join $output BUFGCE_library.v]
    set ports_before {}
    foreach variant {reference implemented} checkpoint [list $original $implemented] {
        file copy $checkpoint [file join $output ${variant}_input.dcp]
        open_checkpoint [file join $output ${variant}_input.dcp]
        if {[get_property TOP [current_design]] ne $expected_top ||
            [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected checkpoint"}
        if {[llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {
            error "Unexpected clock inventory/period"
        }
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {
            error "Unexpected blackbox or transparent storage"
        }
        set ports [lsort [get_ports]]
        if {[llength $ports] != 1831 || [llength [get_ports -filter {DIRECTION == IN}]] != 1384 ||
            [llength [get_ports -filter {DIRECTION == OUT}]] != 447} {error "Unexpected monitor public interface"}
        monitor_timing_route_geometry $output $variant
        set inventory {}
        set f [open [file join $output ${variant}_ports.tsv] {WRONLY CREAT EXCL}]
        puts $f "port\tdirection\tpartition_pin_locations"
        foreach port $ports {
            set direction [get_property DIRECTION $port]
            lappend inventory [list [get_property NAME $port] $direction]
            puts $f "$port\t$direction\t[get_property HD.PARTPIN_LOCS $port]"
        }
        close $f
        report_property -all [get_clocks pcie_user_clk] -file [file join $output ${variant}_clock.rpt]
        report_property -all [get_ports clk] -file [file join $output ${variant}_clock_port.rpt]
        set buffers [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]
        if {[llength $buffers] != 1 || [get_property REF_NAME $buffers] ne "BUFGCE"} {error "Unexpected clock buffer"}
        report_property -all $buffers -file [file join $output ${variant}_buffer.rpt]
        set f [open [file join $output ${variant}_primitives.tsv] {WRONLY CREAT EXCL}]
        puts $f "cell\tprimitive"
        foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {puts $f "[get_property NAME $cell]\t[get_property REF_NAME $cell]"}
        close $f
        set f [open [file join $output ${variant}_register_names.txt] {WRONLY CREAT EXCL}]
        foreach name [lsort [get_property NAME [get_cells -hierarchical -filter {REF_NAME == FDRE}]]] {puts $f $name}
        close $f
        write_xdc -constraints all [file join $output ${variant}.xdc]
        if {$variant eq "reference"} {
            if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected synthesis clock context"}
            write_edif [file join $output reference.edf]
            set ports_before $inventory
            close_project
        } else {
            if {$inventory ne $ports_before} {error "Public interface changed"}
            if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected implemented external clock context"}
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
    set buffer [get_property NAME $buffers]
    set f [open [file join $output buffer_connectivity.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tnets\tstatus\tports\tdrivers\tloads\troute"
    foreach pin [lsort [get_pins -of_objects $buffers]] {
        set segments [lsort [get_nets -segments -of_objects $pin]]
        puts $f "[get_property NAME $pin]\t[join $segments {;}]\t[join [get_property ROUTE_STATUS $segments] {;}]\t[join [lsort [get_ports -quiet -of_objects $segments]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]] {;}]\t[join [get_property ROUTE $segments] {;}]"
    }
    close $f
    set input_nets [get_nets -segments -of_objects [get_pins ${buffer}/I]]
    if {[get_property NAME [get_ports -of_objects $input_nets]] ne "clk" || [llength [get_pins -quiet -leaf -of_objects $input_nets -filter {DIRECTION == OUT}]]} {error "Unexpected input source"}
    if {[lsort [get_property NAME [get_pins -leaf -of_objects $input_nets -filter {DIRECTION == IN}]]] ne [list ${buffer}/I]} {error "Clock bypass"}
    set ce_nets [get_nets -segments -of_objects [get_pins ${buffer}/CE]]
    set ce_drivers [get_pins -quiet -leaf -of_objects $ce_nets -filter {DIRECTION == OUT}]
    if {[llength $ce_drivers] != 1 || [get_property REF_NAME [get_cells -of_objects $ce_drivers]] ne "VCC" || [llength [get_ports -quiet -of_objects $ce_nets]]} {error "Unexpected enable"}
    set output_nets [get_nets -segments -of_objects [get_pins ${buffer}/O]]
    set fdre [get_cells -hierarchical -filter {REF_NAME == FDRE}]
    set loads [lsort [get_property NAME [get_pins -leaf -of_objects $output_nets -filter {DIRECTION == IN}]]]
    set expected [lsort [get_property NAME [get_pins -of_objects $fdre -filter {REF_PIN_NAME == C}]]]
    if {$loads ne $expected || [llength $loads] != [llength $fdre] || [llength [get_ports -quiet -of_objects $output_nets]]} {error "Unexpected clock output loads"}
    report_methodology -file [file join $output methodology.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output timing.rpt]
    # Additional path subsets diagnose internal state feedback without replacing
    # the complete timing summary or introducing any timing exception.
    set registers [get_cells -hierarchical -filter {REF_NAME == FDRE}]
    if {![llength $registers]} {error "Missing register timing endpoints"}
    set f [open [file join $output register_timing_endpoints.txt] {WRONLY CREAT EXCL}]
    foreach name [lsort [get_property NAME $registers]] {puts $f $name}
    close $f
    foreach {kind delay} {setup max hold min} {
        report_timing -from $registers -to $registers -delay_type $delay -max_paths 20 -input_pins -file [file join $output register_to_register_${kind}.rpt]
        report_timing -from [get_ports -filter {DIRECTION == IN && NAME != clk}] -to $registers -delay_type $delay -max_paths 20 -input_pins -file [file join $output input_to_register_${kind}.rpt]
        report_timing -from $registers -to [all_outputs] -delay_type $delay -max_paths 20 -input_pins -file [file join $output register_to_output_${kind}.rpt]
        report_timing -from [get_ports -filter {DIRECTION == IN && NAME != clk}] -to [all_outputs] -delay_type $delay -max_paths 20 -input_pins -file [file join $output input_to_output_${kind}.rpt]
    }
    monitor_predecode_ed_timing $output
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
    report_exceptions -summary -file [file join $output exception_summary.rpt]
    write_xdc -constraints all [file join $output after_audit.xdc]
    puts "NTRPU_MONITOR_PREDECODE_TIMING_ROUTE_AUDIT_EXPORTED violations=$index"
    close_project
}
if {[catch {monitor_clock_route_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
