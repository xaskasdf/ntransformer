# Generic channel synthesis and pin-planning exploration; not exact DIMM qualification.
# Usage: -tclargs COMPLETED_EXAMPLE_DIRECTORY NEW_DIRECTORY
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ddr_pin_synth {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected a fresh process and COMPLETED_EXAMPLE_DIRECTORY NEW_DIRECTORY"
    }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    if {![file exists [file join $input ntrpu_ddr_geometry_ex.xpr]]} { error "Missing example project" }
    file mkdir $output
    file copy $input [file join $output example]
    file copy [file join $ntrpu_script_dir .. docs fpga board ddr4_geometry_exploratory.csv] [file join $output geometry.csv]
    open_project [file join $output example ntrpu_ddr_geometry_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected FPGA" }
    set ip [get_ips ntrpu_ddr_geometry]
    set_property CONFIG.C0.DDR4_CustomParts [file join $output geometry.csv] $ip
    foreach {property expected} {
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY
        CONFIG.C0.DDR4_AxiAddressWidth 37 CONFIG.C0.DDR4_Ecc true
        CONFIG.C0.DDR4_AxiDataWidth 512 CONFIG.C0.DDR4_Slot Dual
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_CasLatency 12
    } {
        if {[get_property $property $ip] ne $expected} { error "Unexpected $property" }
    }
    report_property -all -file [file join $output controller_properties.rpt] $ip
    generate_target all $ip
    # Synthesized netlists are required by AMD UG899 for memory I/O planning.
    launch_runs synth_1 -jobs 2
    wait_on_run synth_1
    report_property -all -file [file join $output synthesis_run.rpt] [get_runs synth_1]
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%" ||
        [string match -nocase *error* [get_property STATUS [get_runs synth_1]]]} {
        error "Synthesis incomplete: [get_property STATUS [get_runs synth_1]]"
    }
    open_run synth_1
    write_checkpoint [file join $output synthesized.dcp]
    report_utilization -file [file join $output utilization.rpt]
    report_io -file [file join $output generated_io.rpt]
    report_drc -file [file join $output generated_drc.rpt]
    set stream [open [file join $output memory_drc_checks.txt] {WRONLY CREAT EXCL}]
    foreach check [get_drc_checks] {
        set desc [get_property DESCRIPTION $check]
        if {[regexp -nocase {memory|ddr|mig|xiphy} "$check $desc"]} { puts $stream "$check: $desc" }
    }
    close $stream
    # The automatic example currently spills into bank 47. Capture that baseline,
    # then ask the tool to place the same complete interface in the reserved trio.
    set ports [get_ports]
    reset_property PACKAGE_PIN $ports
    set stream [open [file join $output three_bank_attempt.txt] {WRONLY CREAT EXCL}]
    set failed [catch {place_ports -iobank [get_iobanks {44 45 46}] $ports} message options]
    puts $stream "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
    close $stream
    report_io -file [file join $output three_bank_io.rpt]
    report_drc -file [file join $output three_bank_drc.rpt]
    write_xdc [file join $output three_bank_candidate.xdc]
    puts "NTRPU_DDR_PIN_SYNTH_EXPORTED three_bank_command_error=$failed"
    puts "Review placement, coverage and memory DRCs. Command success is not pinout approval."
    close_project
}
if {[catch {ntrpu_ddr_pin_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
