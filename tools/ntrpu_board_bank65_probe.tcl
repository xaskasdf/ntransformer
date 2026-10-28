# V4 voltage/pin diagnostic only: one synthetic SSTL12 output, not a MIG.
# Usage: -tclargs EXACT_FPGA_PART NEW_DIRECTORY
proc ntrpu_bank65_probe {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Use a fresh process: -tclargs EXACT_FPGA_PART NEW_DIRECTORY"
    }
    lassign $arguments part output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists: $output" }
    file mkdir $output
    create_project ntrpu_bank65 [file join $output project] -part $part
    set stream [open [file join $output bank65_probe.v] {WRONLY CREAT EXCL}]
    puts $stream {module bank65_probe(input sys_rst_n, output ddr_probe); assign ddr_probe = sys_rst_n; endmodule}
    close $stream
    read_verilog [file join $output bank65_probe.v]
    synth_design -top bank65_probe -part $part
    set reset_pin [get_package_pins -filter {PIN_FUNC =~ *_PERSTN0_65}]
    if {[llength $reset_pin] != 1 || [get_property NAME $reset_pin] ne "AM17"} {
        error "Unexpected PERSTN0 ball for this diagnostic"
    }
    set candidate [get_package_pins -filter {BANK == 65 && PIN_FUNC =~ IO_L1P_*}]
    if {[llength $candidate] != 1} { error "Expected one synthetic bank65 output candidate" }
    set_property PACKAGE_PIN AM17 [get_ports sys_rst_n]
    set_property PACKAGE_PIN [get_property NAME $candidate] [get_ports ddr_probe]
    set_property IOSTANDARD SSTL12_DCI [get_ports ddr_probe]
    set checks [get_drc_checks -quiet BIVC-1]
    if {[llength $checks] != 1} { error "Expected BIVC-1 voltage check" }
    set_property IOSTANDARD LVCMOS18 [get_ports sys_rst_n]
    report_drc -checks $checks -file [file join $output mixed_voltage.rpt]
    set mixed [llength [get_drc_violations -quiet]]
    if {$mixed == 0} { error "Expected mixed bank voltage conflict" }
    set_property IOSTANDARD LVCMOS12 [get_ports sys_rst_n]
    report_drc -checks $checks -file [file join $output shared_1v2.rpt]
    set shared [llength [get_drc_violations -quiet]]
    if {$shared != 0} { error "1.2V diagnostic still has bank-voltage conflict" }
    report_io -file [file join $output synthetic_io.rpt]
    set stream [open [file join $output site_mapping.txt] {WRONLY CREAT EXCL}]
    puts $stream "PERSTN0 AM17; synthetic SSTL12 output [get_property NAME $candidate]"
    for {set i 0} {$i < 16} {incr i} {
        set site [get_sites GTYE4_CHANNEL_X1Y$i]
        puts $stream "GTYE4_CHANNEL_X1Y$i: banks [get_iobanks -of_objects $site]"
    }
    set common [get_sites -filter {NAME =~ *COMMON*} -of_objects [get_iobanks -of_objects [get_sites GTYE4_CHANNEL_X1Y7]]]
    foreach suffix {REFCLK0P REFCLK0N} {
        puts $stream "$suffix: [get_package_pins -of_objects [get_bels $common/$suffix]]"
    }
    close $stream
    write_checkpoint [file join $output bank65_probe.dcp]
    puts "NTRPU_BANK65_VOLTAGE_PROBE_PASS mixed=$mixed shared=$shared"
    puts "Only BIVC-1 checked; no MIG/PCIe logic, placement, reset timing or level shifter validation."
    close_project
}
if {[catch {ntrpu_bank65_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
