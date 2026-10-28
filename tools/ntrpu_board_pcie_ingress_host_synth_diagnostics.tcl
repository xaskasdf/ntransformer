# Characterize the functionally qualified ingress with the actual AMD CDC IP.
# OOC internal timing only: parent I/O budgets and board placement are pending.
set script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ingress_host_synth {arguments} {
    global script_dir
    if {[llength $arguments] != 1 || [llength [get_projects -quiet]]} {
        error "Expected NEW_DIRECTORY in a fresh Vivado process"
    }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output already exists"}
    file mkdir $output
    create_project ingress [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_dword_decode ntrpu_pcie_line_ingress ntrpu_peer_rx_guard ntrpu_pcie_ingress_peer ntrpu_axi_capacity_guard ntrpu_pcie_ingress_guarded ntrpu_pcie_ingress_async ntrpu_pcie_ingress_control ntrpu_axil_cdc_recode ntrpu_pcie_ingress_host} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    foreach {name protocol width address} {ntrpu_ingress_axi_cdc AXI4 512 64 ntrpu_host_axil_cdc AXI4LITE 32 32} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set config [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3]
        if {$protocol eq "AXI4"} {lappend config CONFIG.ID_WIDTH 4}
        set_property -dict $config $ip
        foreach {property value} $config {
            if {[get_property $property $ip] ne $value} {error "IP configuration differs: $property"}
        }
        generate_target all $ip
        report_property -all -file [file join $output ${name}_properties.rpt] $ip
    }
    report_ip_status -file [file join $output ip_status.rpt]
    # Disable only AMD's standalone placeholder clocks. Preserve scoped CDC
    # constraints and vendor waivers; add no false path or multicycle exception.
    set placeholders [get_files -all -quiet *_ooc.xdc]
    if {[llength $placeholders] != 2} {error "Expected two IP placeholder clock XDC"}
    set_property IS_ENABLED false $placeholders
    set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name host_clk -period 4.000 [get_ports host_clk]}
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]}
    close $f
    read_xdc [file join $output clocks.xdc]
    set_property top ntrpu_pcie_ingress_host [get_filesets sources_1]
    update_compile_order -fileset sources_1
    set f [open [file join $output synthesis_inputs.txt] {WRONLY CREAT EXCL}]
    foreach file [get_files -compile_order sources -used_in synthesis] {puts $f $file}
    close $f
    set f [open [file join $output constraint_inputs.tsv] {WRONLY CREAT EXCL}]
    puts $f "path\tenabled\tused_in\tscoped_to_ref\tprocessing_order"
    foreach file [get_files -all -filter {FILE_TYPE == XDC}] {
        puts $f "$file\t[get_property IS_ENABLED $file]\t[get_property USED_IN $file]\t[get_property SCOPED_TO_REF $file]\t[get_property PROCESSING_ORDER $file]"
    }
    close $f
    # Expand diagnostics only; no severity, RTL or timing change.
    help set_msg_config
    set_msg_config -id {Synth 8-7129} -limit 10000
    synth_design -mode out_of_context -top ntrpu_pcie_ingress_host -part xcvu5p-flva2104-2-i
    if {[llength [get_clocks]] != 2 || [get_property PERIOD [get_clocks host_clk]] != 4.0 || [get_property PERIOD [get_clocks clk]] != 5.0} {error "Unexpected clocks"}
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Black box or latch"}
    write_checkpoint [file join $output ingress_host_synth.dcp]
    write_verilog -mode funcsim [file join $output ingress_host_funcsim.v]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_clock_interaction -file [file join $output clock_interaction.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_bus_skew -file [file join $output bus_skew.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Ingress synthesis context\n\nHost clock 4 ns; memory UI clock 5 ns. Actual AMD AXI clock converter 2.1, 512-bit data, 64-bit address, 4-bit ID, three synchronization stages."
    puts $f "Ten production RTL inputs are unchanged from the functional host/control/data CDC MIG bank. Only cold coordinated reset is qualified."
    puts $f "No input/output delays, parent clock locations, board pins or physical timing qualification. AXI-Lite control crosses a second AMD CDC with reversible RRESP coding; transport quiescence remains externally trusted. Diagnostic outputs remain in UI domain."
    puts $f "No new false paths, multicycle exceptions, severity changes or user waivers. Generated vendor CDC constraints remain active. Reports require independent review."
    close $f
    puts NTRPU_INGRESS_HOST_SYNTH_EXPORTED
    close_project
}
if {[catch {ntrpu_ingress_host_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
