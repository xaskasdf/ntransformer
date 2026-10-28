# Diagnostic only: request an internal BUFGCE and stop after optimization.
# No placement, routing, bitstream, RTL change, or timing exception.
proc clock_preflight_inventory {output prefix} {
    set f [open [file join $output ${prefix}_primitives.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive"
    foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]"
    }
    close $f
    set f [open [file join $output ${prefix}_ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection"
    foreach port [lsort [get_ports]] {puts $f "$port\t[get_property DIRECTION $port]"}
    close $f
    write_xdc -constraints all [file join $output ${prefix}.xdc]
}

proc clock_preflight_check_design {} {
    if {[get_property TOP [current_design]] ne "ntrpu_pcie_symbolic_monitor_predecode_row_enable_r1" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected checkpoint"}
    if {[llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {
        error "Unexpected clock inventory/period"
    }
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {
        error "Unexpected blackbox or latch"
    }
}

proc clock_preflight {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected REVIEWED_MONITOR_SYNTH_DCP NEW_OUTPUT"
    }
    lassign $arguments synthesis output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach command {open_checkpoint opt_design set_property get_nets get_pins write_edif} {
        set text [help $command]
        if {[string trim $text] eq ""} {error "Empty help return"}
        set f [open [file join $output help_${command}.txt] {WRONLY CREAT EXCL}]
        puts $f $text
        close $f
    }
    set f [open [file join $output VERSION.txt] {WRONLY CREAT EXCL}]
    puts $f [version]
    close $f
    file copy [file join $::env(XILINX_VIVADO) data verilog src unisims BUFGCE.v] [file join $output BUFGCE_library.v]
    file copy $synthesis [file join $output synthesis_input.dcp]
    open_checkpoint [file join $output synthesis_input.dcp]
    clock_preflight_check_design
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected synthesis clock source"}
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]]} {error "Unexpected synthesis clock buffer"}
    opt_design
    write_checkpoint [file join $output reference_optimized.dcp]
    write_edif [file join $output reference_optimized.edf]
    write_verilog -mode funcsim [file join $output reference_optimized_funcsim.v]
    clock_preflight_inventory $output reference
    set reference_registers [lsort [get_property NAME [get_cells -hierarchical -filter {REF_NAME == FDRE}]]]
    set reference_ports {}
    foreach port [lsort [get_ports]] {lappend reference_ports [list [get_property NAME $port] [get_property DIRECTION $port]]}
    close_project
    open_checkpoint [file join $output synthesis_input.dcp]
    clock_preflight_check_design
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Expected synthesis without parent-clock assumption"}
    if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]]} {error "Unexpected pre-existing global buffer"}
    clock_preflight_inventory $output before
    set net [get_nets -of_objects [get_ports clk]]
    if {[llength $net] != 1} {error "Expected one clock-port net"}
    report_property -all $net -file [file join $output before_clock_net.rpt]
    set_property CLOCK_BUFFER_TYPE BUFGCE $net
    opt_design
    # Save the actual outcome before the structural gate, including failures.
    write_checkpoint [file join $output optimized_clock.dcp]
    write_verilog -mode funcsim [file join $output optimized_clock_funcsim.v]
    write_edif [file join $output optimized_clock.edf]
    clock_preflight_inventory $output after
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output timing.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_drc -file [file join $output drc.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    clock_preflight_check_design
    set buffers [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]
    if {[llength $buffers] != 1 || [get_property REF_NAME $buffers] ne "BUFGCE"} {
        error "Expected exactly one BUFGCE; no implementation allowed"
    }
    set buffer [lindex $buffers 0]
    report_property -all $buffer -file [file join $output buffer.rpt]
    set f [open [file join $output buffer_connectivity.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tnets\tports\tdrivers\tloads"
    foreach pin [lsort [get_pins -of_objects $buffer]] {
        set segments [get_nets -segments -of_objects $pin]
        puts $f "$pin\t[join [lsort $segments] {;}]\t[join [lsort [get_ports -quiet -of_objects $segments]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]] {;}]"
    }
    close $f
    set ports {}
    foreach port [lsort [get_ports]] {lappend ports [list [get_property NAME $port] [get_property DIRECTION $port]]}
    if {$ports ne $reference_ports} {error "Public interface differs from reference"}
    set registers [lsort [get_cells -hierarchical -filter {REF_NAME == FDRE}]]
    if {[lsort [get_property NAME $registers]] ne $reference_registers} {error "Register inventory differs from same-seed optimized reference; inspect saved evidence"}
    set expected_clocks [lsort [get_pins -of_objects $registers -filter {REF_PIN_NAME == C}]]
    set output_nets [get_nets -segments -of_objects [get_pins ${buffer}/O]]
    set loads [lsort [get_pins -leaf -of_objects $output_nets -filter {DIRECTION == IN}]]
    if {$loads ne $expected_clocks || [llength $loads] != [llength $registers]} {
        error "BUFGCE output does not exclusively drive every register clock"
    }
    set input_nets [get_nets -segments -of_objects [get_pins ${buffer}/I]]
    if {[lsort [get_ports -of_objects $input_nets]] ne {clk} ||
        [llength [get_pins -quiet -leaf -of_objects $input_nets -filter {DIRECTION == OUT}]]} {
        error "BUFGCE input is not driven solely by external clock port"
    }
    set ce_nets [get_nets -segments -of_objects [get_pins ${buffer}/CE]]
    set ce_drivers [get_pins -quiet -leaf -of_objects $ce_nets -filter {DIRECTION == OUT}]
    if {[llength $ce_drivers] != 1 ||
        [get_property REF_NAME [get_cells -of_objects $ce_drivers]] ne "VCC" ||
        [llength [get_ports -quiet -of_objects $ce_nets]]} {error "BUFGCE CE must have constant VCC driver"}
    puts "NTRPU_MONITOR_PREDECODE_ROW_ENABLE_CLOCK_PREFLIGHT_EXPORTED registers=[llength $registers] buffer=$buffer"
    close_project
}
if {[catch {clock_preflight $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
