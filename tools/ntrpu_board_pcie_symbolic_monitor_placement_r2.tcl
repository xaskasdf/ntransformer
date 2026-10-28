# Diagnostic place/route of the reviewed optimized DCP with its real internal BUFGCE.
# No board pinout, endpoint, DDR, SDF or physical qualification is implied.
proc monitor_placement_reports {output prefix} {
    write_checkpoint [file join $output ${prefix}.dcp]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output ${prefix}_timing.rpt]
    check_timing -verbose -file [file join $output ${prefix}_check_timing.rpt]
    report_utilization -file [file join $output ${prefix}_utilization.rpt]
}

proc monitor_placement {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected OPTIMIZED_CLOCK_DCP NEW_OUTPUT"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach command {open_checkpoint place_design route_design report_route_status report_timing_summary report_timing all_registers report_high_fanout_nets write_edif} {
        set text [help $command]
        if {[string trim $text] eq ""} {error "Empty help: $command"}
        set f [open [file join $output help_${command}.txt] {WRONLY CREAT EXCL}]
        puts $f $text; close $f
        puts "NTRPU_HELP_CAPTURED $command characters=[string length $text]"
    }
    set f [open [file join $output VERSION.txt] {WRONLY CREAT EXCL}]
    puts $f [version]; close $f
    file copy $checkpoint [file join $output input.dcp]
    file copy [file join $::env(XILINX_VIVADO) data verilog src unisims BUFGCE.v] [file join $output BUFGCE_library.v]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_symbolic_monitor_r1" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected checkpoint"}
    if {[llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {
        error "Unexpected clock inventory/period"
    }
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {
        error "Unexpected blackbox or transparent storage"
    }
    set ports [lsort [get_property NAME [get_ports]]]
    if {[llength $ports] != 1831 || [llength [get_ports -filter {DIRECTION == IN}]] != 1384 ||
        [llength [get_ports -filter {DIRECTION == OUT}]] != 447} {error "Unexpected monitor public interface"}
    set f [open [file join $output ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection"
    foreach port [lsort [get_ports]] {puts $f "$port\t[get_property DIRECTION $port]"}
    close $f
    write_xdc -constraints all [file join $output before_context.xdc]
    report_property -all [current_design] -file [file join $output input_design.rpt]
    report_property -all [get_clocks pcie_user_clk] -file [file join $output before_clock.rpt]
    report_property -all [get_ports clk] -file [file join $output before_clock_port.rpt]
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output before_context_timing.rpt]
    report_exceptions -file [file join $output before_exceptions.rpt]
    set buffers [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]
    if {[llength $buffers] != 1 || [get_property REF_NAME $buffers] ne "BUFGCE"} {error "Expected one reviewed BUFGCE"}
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected external clock source assumption"}
    report_property -all $buffers -file [file join $output input_buffer.rpt]
    monitor_placement_reports $output input_optimized
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Internal clock placement diagnostic\n\nStart from the exact reviewed optimized DCP. One internal BUFGCE. No resynthesis or repeated opt_design."
    puts $f "Inherited 4ns clock, zero input/output delays and uncertainty retained. No HD.CLK_SRC assumption, pin assignments, timing exceptions or DRC severity changes."
    puts $f "Default placement then Quick non-timing-driven route, maximum 2 threads. This first measurement does not target or qualify timing closure. Explicit register/register, input/register, register/output and input/output setup/hold reports. Setup/hold violations remain failures; OOC boundaries and gaps require independent review."
    puts $f "Exported DCP and netlist require exact readback and their own sixteen-bank replay plus six actual input controls. No board/native/DDR/SDF qualification."
    close $f
    set phase placement
    set failed [catch {
        place_design
        monitor_placement_reports $output placed
        set phase routing
        route_design -directive Quick
        set phase completed
    } message options]
    set f [open [file join $output implementation_status.txt] {WRONLY CREAT EXCL}]
    puts $f "return_code=$failed\nphase=$phase\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} {puts $f [dict get $options -errorinfo]}
    close $f
    # Preserve terminal or partial state even if implementation did not finish.
    write_checkpoint [file join $output route_attempt.dcp]
    write_verilog -mode funcsim [file join $output routed_funcsim.v]
    write_edif [file join $output routed.edf]
    write_xdc -constraints all [file join $output implemented.xdc]
    set f [open [file join $output primitives.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive"
    foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]"
    }
    close $f
    set reports_failed {}
    foreach {name command} {
        timing {report_timing_summary -report_unconstrained -max_paths 20}
        check_timing {check_timing -verbose}
        utilization {report_utilization}
        hierarchy {report_utilization -hierarchical}
        route {report_route_status}
        drc {report_drc}
        methodology {report_methodology}
        clocks {report_clocks}
        exceptions {report_exceptions}
        ignored_exceptions {report_exceptions -ignored}
        exception_summary {report_exceptions -summary}
    } {
        if {[catch {{*}$command -file [file join $output ${name}.rpt]} report_error]} {
            lappend reports_failed [list $name $report_error]
        }
    }
    set f [open [file join $output report_errors.txt] {WRONLY CREAT EXCL}]
    puts $f $reports_failed
    close $f
    set f [open [file join $output route_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus"
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] {puts $f "$net\t$status"}
    close $f
    set clock_net [get_nets -of_objects [get_ports clk]]
    if {[llength $clock_net] != 1} {error "Nonunique parent clock net"}
    set clock_route [get_property ROUTE $clock_net]
    set f [open [file join $output parent_clock.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tclock\tperiod\texternal_clock_source\tnet\tstatus\troute"
    puts $f "clk\tpcie_user_clk\t[get_property PERIOD [get_clocks pcie_user_clk]]\t[get_property HD.CLK_SRC [get_ports clk]]\t$clock_net\t[get_property ROUTE_STATUS $clock_net]\t$clock_route"
    close $f
    set f [open [file join $output partial_routes.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus\tmatching_parent_route\troute"
    foreach net $nets status [get_property ROUTE_STATUS $nets] {
        if {$status ne "PARTIAL"} {continue}
        set net_route [get_property ROUTE $net]
        set matches [expr {$clock_route ne "" && $net_route eq $clock_route}]
        puts $f "$net\t$status\t$matches\t$net_route"
    }
    close $f
    # Supplement full timing with explicit FDRE-to-FDRE subsets and route boundaries.
    foreach {name command} {
        register_to_register_setup {report_timing -from [get_cells -hierarchical -filter {REF_NAME == FDRE}] -to [get_cells -hierarchical -filter {REF_NAME == FDRE}] -delay_type max -max_paths 20 -input_pins}
        register_to_register_hold {report_timing -from [get_cells -hierarchical -filter {REF_NAME == FDRE}] -to [get_cells -hierarchical -filter {REF_NAME == FDRE}] -delay_type min -max_paths 20 -input_pins}
        input_to_register_setup {report_timing -from [get_ports -filter {DIRECTION == IN && NAME != clk}] -to [get_cells -hierarchical -filter {REF_NAME == FDRE}] -delay_type max -max_paths 20 -input_pins}
        input_to_register_hold {report_timing -from [get_ports -filter {DIRECTION == IN && NAME != clk}] -to [get_cells -hierarchical -filter {REF_NAME == FDRE}] -delay_type min -max_paths 20 -input_pins}
        register_to_output_setup {report_timing -from [get_cells -hierarchical -filter {REF_NAME == FDRE}] -to [all_outputs] -delay_type max -max_paths 20 -input_pins}
        register_to_output_hold {report_timing -from [get_cells -hierarchical -filter {REF_NAME == FDRE}] -to [all_outputs] -delay_type min -max_paths 20 -input_pins}
        input_to_output_setup {report_timing -from [get_ports -filter {DIRECTION == IN && NAME != clk}] -to [all_outputs] -delay_type max -max_paths 20 -input_pins}
        input_to_output_hold {report_timing -from [get_ports -filter {DIRECTION == IN && NAME != clk}] -to [all_outputs] -delay_type min -max_paths 20 -input_pins}
        high_fanout_timing {report_high_fanout_nets -timing -load_types -max_nets 30}
        high_fanout_slr {report_high_fanout_nets -slr -max_nets 30}
    } {
        # The timing commands require object substitution before list expansion.
        if {[catch {eval $command [list -file [file join $output ${name}.rpt]]} report_error]} {
            lappend reports_failed [list $name $report_error]
        }
    }
    foreach kind {PARTIAL GAPS UNPLACED UNROUTED CONFLICTS ANTENNAS NODRIVER MULTI_DRIVER} {
        if {[catch {report_route_status -route_type $kind -list_all_nets -show_all -file [file join $output route_${kind}.rpt]} report_error]} {
            lappend reports_failed [list route_$kind $report_error]
        }
    }
    set f [open [file join $output final_report_errors.txt] {WRONLY CREAT EXCL}]
    puts $f $reports_failed; close $f
    set f [open [file join $output route_flags.tsv] {WRONLY CREAT EXCL}]
    puts $f "flag\tvalue"
    foreach flag {PLACED_FULLY PARTIALLY_ROUTED ROUTED_FULLY ERRORS_IN_ROUTES} {
        puts $f "$flag\t[report_route_status -boolean_check $flag]"
    }
    close $f
    set f [open [file join $output final_ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection\tpartition_pin_locations"
    foreach port [lsort [get_ports]] {puts $f "$port\t[get_property DIRECTION $port]\t[get_property HD.PARTPIN_LOCS $port]"}
    close $f
    set f [open [file join $output register_timing_endpoints.txt] {WRONLY CREAT EXCL}]
    set registers [get_cells -hierarchical -filter {REF_NAME == FDRE}]
    foreach name [lsort [get_property NAME $registers]] {puts $f $name}
    close $f
    set buffers [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]
    if {[llength $buffers] != 1 || [get_property REF_NAME $buffers] ne "BUFGCE"} {error "Clock buffer changed during implementation"}
    set buffer [get_property NAME $buffers]
    report_property -all $buffers -file [file join $output final_buffer.rpt]
    report_property -all [get_clocks pcie_user_clk] -file [file join $output final_clock.rpt]
    report_property -all [get_ports clk] -file [file join $output final_clock_port.rpt]
    set f [open [file join $output buffer_connectivity.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tnets\tstatus\tports\tdrivers\tloads\troute"
    foreach pin [lsort [get_pins -of_objects $buffers]] {
        set segments [lsort [get_nets -segments -of_objects $pin]]
        puts $f "[get_property NAME $pin]\t[join $segments {;}]\t[join [get_property ROUTE_STATUS $segments] {;}]\t[join [lsort [get_ports -quiet -of_objects $segments]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]] {;}]\t[join [get_property ROUTE $segments] {;}]"
    }
    close $f
    set input_nets [get_nets -segments -of_objects [get_pins ${buffer}/I]]
    if {[get_property NAME [get_ports -of_objects $input_nets]] ne "clk" || [llength [get_pins -quiet -leaf -of_objects $input_nets -filter {DIRECTION == OUT}]]} {error "Unexpected buffer input source"}
    if {[lsort [get_property NAME [get_pins -leaf -of_objects $input_nets -filter {DIRECTION == IN}]]] ne [list ${buffer}/I]} {error "Clock buffer input bypass"}
    set ce_nets [get_nets -segments -of_objects [get_pins ${buffer}/CE]]
    set drivers [get_pins -quiet -leaf -of_objects $ce_nets -filter {DIRECTION == OUT}]
    if {[llength $drivers] != 1 || [get_property REF_NAME [get_cells -of_objects $drivers]] ne "VCC" || [llength [get_ports -quiet -of_objects $ce_nets]]} {error "Buffer CE changed"}
    set output_nets [get_nets -segments -of_objects [get_pins ${buffer}/O]]
    set loads [lsort [get_property NAME [get_pins -leaf -of_objects $output_nets -filter {DIRECTION == IN}]]]
    set expected [lsort [get_property NAME [get_pins -of_objects $registers -filter {REF_PIN_NAME == C}]]]
    if {$loads ne $expected || [llength $loads] != [llength $registers]} {error "Clock loads changed"}
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "External clock assumption appeared"}
    write_xdc -constraints all [file join $output after_reports.xdc]
    if {[lsort [get_property NAME [get_ports]]] ne $ports || [llength [get_clocks]] != 1 ||
        [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {error "Interface/clock changed"}
    if {$failed || [llength $reports_failed]} {error "Implementation/report failure; partial evidence retained"}
    puts NTRPU_MONITOR_CLOCK_PLACEMENT_DIAGNOSTIC_EXPORTED
    close_project
}
if {[catch {monitor_placement $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
