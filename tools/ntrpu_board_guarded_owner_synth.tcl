# NTRD retained owner, automatic guard, eight-channel boundary and two AMD CDCs.
# Endpoint PCIe and MIG PHY/controllers remain outside this OOC boundary.
# OOC internal timing only: parent I/O budgets and board placement are pending.
set script_dir [file dirname [file normalize [info script]]]
proc ntrpu_guarded_owner_synth {arguments} {
    global script_dir
    if {[llength $arguments] != 1 || [llength [get_projects -quiet]]} {
        error "Expected NEW_DIRECTORY in a fresh Vivado process"
    }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output already exists"}
    file mkdir $output
    set_param general.maxThreads 4
    create_project ingress [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_retained_channel ntrpu_host_boundary_cut ntrpu_reset_supervisor ntrpu_host_loss_guard ntrpu_axil_cdc_recode ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_fragment_pool_bridge ntrpu_pcie_fragment_pool ntrpu_pcie_fragment_dual_control ntrpu_pcie_fragment_dual_managed ntrpu_pcie_fragment_guarded_host} {
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
    puts $f {create_clock -name host_clk -period 7.300 [get_ports host_clk]}
    puts $f {create_clock -name source_clk -period 5.000 [get_ports source_clk]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
    close $f
    read_xdc [file join $output clocks.xdc]
    set_property top ntrpu_pcie_fragment_guarded_host [get_filesets sources_1]
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
    # Expand optional-port diagnostics; do not alter severity or constraints.
    set_msg_config -id {Synth 8-7129} -limit 10000
    synth_design -mode out_of_context -top ntrpu_pcie_fragment_guarded_host -part xcvu5p-flva2104-2-i -generic {FIRST_COOKIE=64'hffffffff}
    if {[llength [get_clocks]] != 3} {error "Expected three clocks"}
    foreach {name period} {host_clk 7.3 source_clk 5.0 mgmt_clk 10.0} {
        if {[get_property PERIOD [get_clocks $name]] != $period} {error "Unexpected clock $name"}
    }
    if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Black box or latch"}
    write_checkpoint [file join $output guarded_owner_synth.dcp]
    write_verilog -mode funcsim [file join $output guarded_owner_funcsim.v]
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
    set f [open [file join $output cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tinit\tasync_reg"
    foreach c [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property -quiet INIT $c]\t[get_property -quiet ASYNC_REG $c]"
    }
    close $f
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Guarded NTRD owner OOC\n\nHost7.3ns/source5ns/management10ns assumed parent clocks. Real NTRD control/collector/bridge, eight retained boundary channels, automatic guard and two actual AMD AXI Clock Converters. FIRST_COOKIE=0xffffffff matches the functional bank."
    puts $f "Pool CMD/DATA/CPL and adjacent peers remain external; no memory pool, MIG, native PCIe endpoint, compute, four-channel board or physical pinout."
    puts $f "No physical clock sites or external delays. No new false paths, multicycle constraints or waivers. AMD scoped constraints preserved; only two standalone placeholder clock files disabled. Reset/CDC paths need independent review before routing."
    close $f
    puts NTRPU_GUARDED_OWNER_SYNTH_EXPORTED
    close_project
}
if {[catch {ntrpu_guarded_owner_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
