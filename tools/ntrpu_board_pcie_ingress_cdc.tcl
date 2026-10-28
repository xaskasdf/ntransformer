# Cold ingress/controller/receiver/capacity-guard -> real MIG readback.
# Generic geometry only; no exact Micron DIMM or warm-reset qualification.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ingress_cdc_sim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=2 || [llength [get_projects -quiet]]} {
        error "Expected fresh process and COMPLETED_EXAMPLE_DIRECTORY NEW_DIRECTORY"
    }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    if {![file exists [file join $input ntrpu_ddr_geometry_ex.xpr]]} { error "Missing generic example" }
    file mkdir $output
    file copy $input [file join $output example]
    file copy [file join $ntrpu_script_dir .. docs fpga board ddr4_geometry_exploratory.csv] [file join $output geometry.csv]
    open_project [file join $output example ntrpu_ddr_geometry_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected FPGA part" }
    foreach file [get_files -all] {
        if {[string first $input [file normalize $file]] == 0} { error "References immutable input: $file" }
    }
    set ip [get_ips ntrpu_ddr_geometry]
    set_property CONFIG.C0.DDR4_CustomParts [file join $output geometry.csv] $ip
    foreach {property expected} {
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY
        CONFIG.C0.DDR4_AxiAddressWidth 37 CONFIG.C0.DDR4_Ecc true
        CONFIG.C0.DDR4_AxiDataWidth 512 CONFIG.C0.DDR4_Slot Dual
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_InputClockPeriod 4000
        CONFIG.C0.DDR4_CasLatency 12 CONFIG.C0.DDR4_CasWriteLatency 9
    } {
        if {[get_property $property $ip] ne $expected} {
            error "Unexpected $property: [get_property $property $ip]; expected $expected"
        }
    }
    report_property -all -file [file join $output controller_properties.rpt] $ip
    generate_target all $ip

    # These definitions follow AMD ddr4_v2_2/hdl/model_wrapper.ttcl, including
    # its DDR4_1250_Timing -> FIXED_1600 branch. The model itself stays intact.
    set model_path [file join $output example imports ddr4_sdram_model_wrapper.sv]
    set stream [open $model_path r]; set model [read $stream]; close $stream
    foreach {before after} {
        DDR4_4G_X8 DDR4_16G_X4
        DDR4_750_Timing DDR4_1250_Timing
        FIXED_2666 FIXED_1600
    } {
        if {[regexp -all $before $model] != 1} { error "Unexpected model template: $before" }
        set model [string map [list $before $after] $model]
    }
    set stream [open $model_path w]; puts -nonewline $stream $model; close $stream
    # Replace the generated early-exit testbench completely, retaining it as evidence.
    set_property USED_IN_SIMULATION false [get_files [file join $output example imports sim_tb_top.sv]]
    foreach name {ntrpu_pcie_dword_decode ntrpu_pcie_line_ingress ntrpu_peer_rx_guard ntrpu_pcie_ingress_peer ntrpu_axi_capacity_guard ntrpu_pcie_ingress_guarded ntrpu_pcie_ingress_async} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_ingress_cdc.sv]
    set_property top tb_ntrpu_pcie_ingress_cdc [get_filesets sim_1]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_ingress_axi_cdc
    set cdc [get_ips ntrpu_ingress_axi_cdc]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $cdc]]
    set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] $cdc
    generate_target all $cdc
    report_property -all -file [file join $output cdc_properties.rpt] $cdc
    report_ip_status -file [file join $output ip_status.rpt]
    set_property generic {HOST_PERIOD_PS=4000 HOST_PHASE_PS=700} [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set previous [current_time -s]
    for {set i 0} {$i<510} {incr i} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_MIG_LINE_PROGRESS time=$now";flush stdout
        if {$now eq $previous} {break}
        set previous $now
    }
    puts NTRPU_MIG_LINE_SIMULATOR_RETURNED
    close_sim
    close_project
    puts "Review calibration, comparisons, ranks and model diagnostics; exit zero is not PASS."
}
if {[catch {ntrpu_ingress_cdc_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
