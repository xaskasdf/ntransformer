# Read-only completion of the internal-clock preflight evidence.
proc audit_clock_design {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {error "Expected REFERENCE_DCP CANDIDATE_DCP NEW_OUTPUT"}
    lassign $arguments reference candidate output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    foreach command {open_checkpoint get_property get_pins get_nets write_edif} {
        set text [help $command]
        if {[string trim $text] eq ""} {error "Empty command help"}
        set f [open [file join $output help_${command}.txt] {WRONLY CREAT EXCL}]
        puts $f $text; close $f
    }
    set f [open [file join $output VERSION.txt] {WRONLY CREAT EXCL}]
    puts $f [version]; close $f
    file copy [file join $::env(XILINX_VIVADO) data verilog src unisims BUFGCE.v] [file join $output BUFGCE_library.v]
    foreach variant {reference candidate} checkpoint [list $reference $candidate] {
        file copy $checkpoint [file join $output ${variant}_input.dcp]
        open_checkpoint [file join $output ${variant}_input.dcp]
        if {[get_property TOP [current_design]] ne "ntrpu_pcie_inventory_join" || [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Unexpected DCP"}
        if {[llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks pcie_user_clk]] != 4.0} {error "Unexpected clocks"}
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Blackbox or latch"}
        set raw_registers [lsort [get_cells -hierarchical -filter {REF_NAME == FDRE}]]
        # Materialize stable NAME values before closing a design; do not retain
        # only database object handles across open_checkpoint/close_project.
        set register_names [lsort [get_property NAME $raw_registers]]
        set names_text [join $register_names "\n"]
        set f [open [file join $output ${variant}_register_names.txt] {WRONLY CREAT EXCL}]
        puts $f $names_text; close $f
        set f [open [file join $output ${variant}_primitives.tsv] {WRONLY CREAT EXCL}]
        puts $f "cell\tprimitive"
        foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {puts $f "[get_property NAME $cell]\t[get_property REF_NAME $cell]"}
        close $f
        set f [open [file join $output ${variant}_ports.tsv] {WRONLY CREAT EXCL}]
        puts $f "port\tdirection"
        foreach port [lsort [get_ports]] {puts $f "[get_property NAME $port]\t[get_property DIRECTION $port]"}
        close $f
        write_edif [file join $output ${variant}.edf]
        write_xdc -constraints all [file join $output ${variant}.xdc]
        if {$variant eq "reference"} {
            set reference_names $names_text
            set raw_reference $raw_registers
            close_project
        }
    }
    set f [open [file join $output register_comparison.txt] {WRONLY CREAT EXCL}]
    puts $f "raw_object_lists_equal=[expr {$raw_registers eq $raw_reference}]"
    puts $f "materialized_names_equal=[expr {$names_text eq $reference_names}]"
    puts $f "candidate_registers=[llength $register_names]"
    puts $f "reference_raw_after_reload=[llength $raw_reference]"
    close $f
    if {$names_text ne $reference_names} {error "Materialized register names differ"}
    set buffers [get_cells -quiet -hierarchical -filter {REF_NAME =~ BUFG*}]
    if {[llength $buffers] != 1 || [get_property REF_NAME $buffers] ne "BUFGCE"} {error "Expected one BUFGCE"}
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Candidate has external clock assumption"}
    set buffer [lindex $buffers 0]
    report_property -all $buffer -file [file join $output buffer.rpt]
    set f [open [file join $output buffer_connectivity.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tnets\tports\tdrivers\tloads"
    foreach pin [lsort [get_pins -of_objects $buffer]] {
        set segments [get_nets -segments -of_objects $pin]
        puts $f "[get_property NAME $pin]\t[join [lsort $segments] {;}]\t[join [lsort [get_ports -quiet -of_objects $segments]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]] {;}]\t[join [lsort [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]] {;}]"
    }
    close $f
    set expected_clocks [lsort [get_property NAME [get_pins -of_objects $raw_registers -filter {REF_PIN_NAME == C}]]]
    set output_nets [get_nets -segments -of_objects [get_pins ${buffer}/O]]
    set loads [lsort [get_property NAME [get_pins -leaf -of_objects $output_nets -filter {DIRECTION == IN}]]]
    if {$loads ne $expected_clocks || [llength $loads] != [llength $register_names]} {error "Incomplete or extra buffered clock loads"}
    set input_nets [get_nets -segments -of_objects [get_pins ${buffer}/I]]
    if {[get_property NAME [get_ports -of_objects $input_nets]] ne "clk" || [llength [get_pins -quiet -leaf -of_objects $input_nets -filter {DIRECTION == OUT}]]} {error "Unexpected clock source"}
    set ce_nets [get_nets -segments -of_objects [get_pins ${buffer}/CE]]
    set drivers [get_pins -quiet -leaf -of_objects $ce_nets -filter {DIRECTION == OUT}]
    if {[llength $drivers] != 1 || [get_property REF_NAME [get_cells -of_objects $drivers]] ne "VCC" || [llength [get_ports -quiet -of_objects $ce_nets]]} {error "Clock enable is not constant VCC"}
    report_timing_summary -report_unconstrained -max_paths 20 -file [file join $output timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_exceptions -file [file join $output exceptions.rpt]
    write_xdc -constraints all [file join $output after_audit.xdc]
    puts "NTRPU_INVENTORY_CLOCK_AUDIT_EXPORTED registers=[llength $register_names] buffer=$buffer"
    close_project
}
if {[catch {audit_clock_design $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
