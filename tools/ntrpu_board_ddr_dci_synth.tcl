# Integrate real DCI startup into a fresh generic example, then inspect bank 65.
# Usage: -tclargs COMPLETED_EXAMPLE_DIRECTORY MAPPING_TSV NEW_DIRECTORY
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ddr_dci_synth {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected fresh process and EXAMPLE_DIRECTORY MAPPING_TSV NEW_DIRECTORY"
    }
    lassign $arguments input mapping output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $input [file join $output example]
    file copy $mapping [file join $output input_mapping.tsv]
    file copy [file join $ntrpu_script_dir .. docs fpga board ddr4_geometry_exploratory.csv] [file join $output geometry.csv]
    open_project [file join $output example ntrpu_ddr_geometry_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected part" }
    set ip [get_ips ntrpu_ddr_geometry]
    set_property CONFIG.C0.DDR4_CustomParts [file join $output geometry.csv] $ip
    set_property CONFIG.System_Clock No_Buffer $ip
    foreach {property expected} {
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY
        CONFIG.C0.DDR4_AxiAddressWidth 37 CONFIG.C0.DDR4_Ecc true
        CONFIG.C0.DDR4_AxiDataWidth 512 CONFIG.C0.DDR4_Slot Dual
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_CasLatency 12
        CONFIG.C0.DDR4_InputClockPeriod 4000 CONFIG.System_Clock No_Buffer
    } {
        if {[get_property $property $ip] ne $expected} { error "Unexpected $property" }
    }
    report_property -all -file [file join $output controller_properties.rpt] $ip
    generate_target all $ip
    set top [file join $output example imports example_top.sv]
    set stream [open $top r]; set text [read $stream]; close $stream
    if {[regsub {\.sys_rst\s+\(sys_rst\)} $text {.sys_rst (ntrpu_mig_reset)} text] != 1 ||
        [regsub {\.c0_sys_clk_p\s+\(c0_sys_clk_p\),\s*\.c0_sys_clk_n\s+\(c0_sys_clk_n\),} $text {.c0_sys_clk_i (ntrpu_ref_raw),} text] != 1} {
        error "Unexpected example clock/reset connections"
    }
    set added {
wire ntrpu_ref_raw, ntrpu_ref_global, ntrpu_mig_reset;
IBUFDS ntrpu_ref_input (.I(c0_sys_clk_p), .IB(c0_sys_clk_n), .O(ntrpu_ref_raw));
BUFGCE ntrpu_ref_buffer (.I(ntrpu_ref_raw), .CE(1'b1), .O(ntrpu_ref_global));
ntrpu_board_startup ntrpu_startup (
    .clk(ntrpu_ref_global), .board_reset_async(sys_rst),
    .mig_sys_rst(ntrpu_mig_reset), .ready()
);
}
    if {[regsub {ntrpu_ddr_geometry u_ntrpu_ddr_geometry} $text "$added\nntrpu_ddr_geometry u_ntrpu_ddr_geometry" text] != 1} {
        error "Missing unique controller instance"
    }
    set stream [open $top w]; puts $stream $text; close $stream
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_dci_startup.sv]
    set xdc [file join $output startup_clock.xdc]
    set stream [open $xdc {WRONLY CREAT EXCL}]
    puts $stream {create_clock -name ntrpu_board_ref -period 4.000 [get_ports c0_sys_clk_p]}
    close $stream
    add_files -fileset constrs_1 $xdc
    update_compile_order -fileset sources_1
    launch_runs synth_1 -jobs 2
    wait_on_run synth_1
    if {[get_property PROGRESS [get_runs synth_1]] ne "100%" ||
        [string match -nocase *error* [get_property STATUS [get_runs synth_1]]]} {
        error "Synthesis incomplete"
    }
    open_run synth_1
    if {[llength [get_cells -hier -filter {REF_NAME == DCIRESET}]] != 1 ||
        [llength [get_cells -hier -filter {REF_NAME == STARTUPE3}]] != 1} {
        error "Startup primitives missing/duplicated"
    }
    set stream [open [file join $output input_mapping.tsv] r]
    set rows [split [string trim [read $stream]] \n]; close $stream
    if {[llength $rows] != 155 || [llength [get_ports]] != 154} { error "Unexpected mapping/port count" }
    reset_property PACKAGE_PIN [get_ports]
    foreach row [lrange $rows 1 end] {
        lassign [split $row \t] port oldball oldbank position newball newbank
        if {[llength [get_ports $port]] != 1} { error "Missing mapped port $port" }
        set_property PACKAGE_PIN $newball [get_ports $port]
    }
    if {[llength [get_ports -filter {PACKAGE_PIN == AM17}]]} { error "AM17 no longer free" }
    write_checkpoint [file join $output synthesized.dcp]
    write_xdc [file join $output candidate.xdc]
    report_utilization -file [file join $output utilization.rpt]
    report_io -file [file join $output io.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_drc -file [file join $output drc.rpt]
    report_cdc -file [file join $output cdc.rpt]
    report_drc -checks [get_drc_checks DCIRST-1] -file [file join $output dci_only.rpt]
    set violations [get_drc_violations -quiet]
    set stream [open [file join $output startup_checks.txt] {WRONLY CREAT EXCL}]
    puts $stream "DCIRESET_count=1 STARTUPE3_count=1 DCIRST_violations=[llength $violations] AM17_free=1"
    puts $stream "reference_clocks=[get_clocks -of_objects [get_pins ntrpu_ref_buffer/O]]"
    close $stream
    if {[llength $violations]} { error "DCI requirement remains unresolved" }
    puts "NTRPU_DDR_DCI_SYNTH_EXPORTED; not placement, routing or exact DIMM approval"
    close_project
}
if {[catch {ntrpu_ddr_dci_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
