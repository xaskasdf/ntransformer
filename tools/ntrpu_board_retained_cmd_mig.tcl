# Cold ingress/controller/receiver/capacity-guard -> real MIG readback.
# Generic geometry only; no exact Micron DIMM or warm-reset qualification.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_retained_cmd_mig_sim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=3 || [llength [get_projects -quiet]]} {
        error "Expected fresh process and COMPLETED_EXAMPLE_DIRECTORY NEW_DIRECTORY SOURCE_PERIOD_PS"
    }
    lassign $arguments input output source_period
    if {$source_period ni {5000 37000}} {error "Unapproved source period"}
    set_param general.maxThreads 2
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    if {![file exists [file join $input ntrpu_ddr_geometry_ex.xpr]]} { error "Missing generic example" }
    file mkdir $output
    create_project cold_start_unit [file join $output startup_project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach name {ntrpu_reset_supervisor ntrpu_dual_mig_retained_start} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_dual_mig_retained_start.sv]
    set_property top tb_ntrpu_dual_mig_retained_start [get_filesets sim_1]
    set_property generic [list SOURCE_PERIOD_PS=$source_period] [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set log [file join $output startup_project cold_start_unit.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output startup.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first DUAL_RETAINED_START_SUITE_PASS $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|ERROR:|FAIL )} $text]} {error "Cold startup unit failed"}
    close_project
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
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_peer_memory_cdc
    set cdc [get_ips ntrpu_peer_memory_cdc]
    set_property -dict {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $cdc
    generate_target all $cdc
    report_property -all $cdc -file [file join $output cdc_configured.rpt]
    foreach {name protocol width address id} {ntrpu_axi4_cdc_probe AXI4 512 64 4 ntrpu_host_axil_cdc AXI4LITE 32 32 0} {
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        set_property -dict [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3] $ip
        if {$protocol eq "AXI4"} {set_property CONFIG.ID_WIDTH $id $ip}
        generate_target all $ip
        report_property -all $ip -file [file join $output ${name}_configured.rpt]
    }
    create_ip -vlnv xilinx.com:ip:dfx_axi_shutdown_manager:1.0 -module_name ntrpu_host_axil_shutdown
    set ip [get_ips ntrpu_host_axil_shutdown]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    set_property -dict {CONFIG.RESET_ACTIVE_LEVEL 0 CONFIG.RP_IS_MASTER false CONFIG.CTRL_INTERFACE_TYPE 0 CONFIG.DP_PROTOCOL AXI4LITE CONFIG.DP_AXI_ADDR_WIDTH 32 CONFIG.DP_AXI_DATA_WIDTH 32 CONFIG.DP_AXI_RESP 0} $ip
    generate_target all $ip
    report_property -all -file [file join $output shutdown_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    foreach name {ntrpu_axi4_drain_monitor_r2 ntrpu_axi4_pair_drain_r2 ntrpu_axi4_response_guard_r2 ntrpu_axi4_retained_admission_r2 ntrpu_axi4_section_certificate_r2 ntrpu_axi_capacity_guard ntrpu_axi_write_capacity_queue ntrpu_axil_cdc_recode ntrpu_axil_drain_monitor ntrpu_axil_fence_certificate_r5 ntrpu_axil_fence_section_r3 ntrpu_axil_pair_drain ntrpu_axil_response_guard ntrpu_axil_retained_admission_r2 ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded ntrpu_drain_certificate_join ntrpu_dual_mig_retained_start ntrpu_dual_multibuffer_mig_retained ntrpu_fragment_pool_bridge ntrpu_local_reply_certificate_r2 ntrpu_owned_drain_coordinator ntrpu_ownership_certificate ntrpu_pcie_fragment_certified_owner ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_dual_control ntrpu_pcie_fragment_dual_coordinated ntrpu_pcie_fragment_ingress ntrpu_pcie_fragment_pool ntrpu_pcie_fragment_retained_host ntrpu_pcie_fragment_retained_mig_r2 ntrpu_peer_memory_credit_port ntrpu_peer_multibuffer_credits ntrpu_peer_multibuffer_elastic ntrpu_peer_multibuffer_fillstage ntrpu_reset_supervisor ntrpu_retained_axi4_path ntrpu_retained_axil_path ntrpu_rv_elastic ntrpu_tagged_reset_mailbox} {
        add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_retained_cmd_mig.sv]
    set_property top tb_ntrpu_retained_cmd_mig [get_filesets sim_1]
    set_property generic [list SOURCE_PERIOD_PS=$source_period] [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    # Static clock waits in the bank avoid the variable-event compilation issue.
    # Use default optimization; timing checks and periodic traffic stay enabled.
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_dual_mig_periodic.vh]
    set_property file_type {Verilog Header} [get_files *ntrpu_dual_mig_periodic.vh]
    set_property include_dirs [concat [get_property include_dirs [get_filesets sim_1]] [list [file normalize [file join $ntrpu_script_dir .. fpga tb]]]] [get_filesets sim_1]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_dual_mig_fixture.sv]
    update_compile_order -fileset sim_1
    launch_simulation
    set previous [current_time -s]
    for {set i 0} {$i<405} {incr i} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_RETAINED_CMD_MIG_PROGRESS time=$now";flush stdout
        if {$now eq $previous} {break}
        set previous $now
    }
    close_sim
    set log [file join $output example ntrpu_ddr_geometry_ex.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output memory.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "MANAGED_MIG_PASS arms=18 lines=16 commands=16" $text]<0 || [string first "RETAINED_CMD_MIG_PASS " $text]<0 || [regexp -nocase {(^|
)[ 	]*(Fatal:|ERROR:|FAIL )} $text]} {error "Cold multibuffer MIG failed"}
    close_project
    puts NTRPU_RETAINED_CMD_MIG_EXPORTED
}
if {[catch {ntrpu_retained_cmd_mig_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
