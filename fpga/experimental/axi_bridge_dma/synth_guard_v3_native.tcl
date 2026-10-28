set_param general.maxThreads 2
source [file join [file dirname [info script]] audit_guard_v3.tcl]
if {[catch {
    if {[llength $argv]!=4} {error "SOURCE EXAMPLE NEW_OUTPUT required"}
    lassign $argv source example out reference
    if {[file exists $out]} {error "New output required"};file mkdir $out
    v3_reference_alias $reference $out/reference_alias.tsv
    create_project guard_v3_native $out/project -part xcvu47p-fsvh2892-2-e
    foreach name {ntrpu_bridge axi_bram_ctrl_0} {
        import_ip -files $example/ntrpu_bridge_ex.srcs/sources_1/ip/$name/$name.xci
        set owned [file normalize [get_property IP_FILE [get_ips $name]]]
        if {[string first "[file normalize $out]/" $owned]!=0} {error "IP import escaped private project"}
        set_property generate_synth_checkpoint false [get_files $owned]
        generate_target synthesis [get_ips $name]
    }
    foreach {key value} {CONFIG.functional_mode AXI_Bridge CONFIG.axisten_freq 250 CONFIG.axi_addr_width 64 CONFIG.axi_data_width 512_bit} {
        if {[get_property $key [get_ips ntrpu_bridge]] ne $value} {error "Native profile changed: $key"}
    }
    foreach name {ntrpu_axi_bridge_guard_v3 ntrpu_axi_bridge_guarded_client_v3 ntrpu_axi_bridge_endpoint_client ntrpu_axi_bridge_writer ntrpu_axi_bridge_mapper ntrpu_xdma_local_master ntrpu_axi_bridge_guard_native} {read_verilog -sv $source/$name.sv}
    read_verilog $source/xilinx_axi_pcie_ep_app.v
    set_property top ntrpu_axi_bridge_guard_native [current_fileset]
    set f [open $out/clocks.xdc {WRONLY CREAT EXCL}]
    puts $f {create_clock -name pcie_refclk -period 10.000 [get_ports sys_clk_p]}
    puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk_in]};close $f
    read_xdc $out/clocks.xdc
    update_compile_order -fileset sources_1
    synth_design -top ntrpu_axi_bridge_guard_native -part xcvu47p-fsvh2892-2-e -mode out_of_context -flatten_hierarchy rebuilt
    write_checkpoint $out/synth.dcp
    if {[llength [get_cells -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    set g guarded/guard
    foreach {from to bound} [list request_reg/C {echo_a_reg[0]/D} 4.000 {echo_a_reg[2]/C} {echo_m_reg[0]/D} 10.000 armed_reg/C {armed_a_reg[0]/D} 4.000] {
        set_max_delay -datapath_only $bound -from [v3_pin $g/$from] -to [v3_pin $g/$to]
    }
    for {set n 0} {$n<3} {incr n} {
        for {set i 0} {$i<3} {incr i} {
            set_false_path -from [v3_pin [format {%s/availability[%d].release_m_reg[2]/C} $g $n]] -to [v3_pin [format {%s/availability[%d].release_a_reg[%d]/CLR} $g $n $i]]
        }
    }
    foreach {n reg} {0 user_lnk_up_reg 1 user_reset_reg} {
        for {set i 0} {$i<3} {incr i} {
            set_false_path -from [v3_pin ntrpu_bridge_i/inst/pcie4c_ip_i/inst/$reg/C] -to [v3_pin [format {%s/availability[%d].release_m_reg[%d]/CLR} $g $n $i]]
        }
    }
    for {set i 0} {$i<3} {incr i} {set_false_path -from [v3_pin $g/fault_reg/C] -to [v3_pin [format {%s/stop_a_reg[%d]/PRE} $g $i]]}
    opt_design
    write_checkpoint $out/optimized.dcp
    v3_evidence $out/optimized
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f V3_SHARED_NATIVE_OPTIMIZED;close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
