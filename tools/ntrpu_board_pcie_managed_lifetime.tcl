# Bounded, separate AXI memory-boundary experiment; never the board queue.
set script_dir [file dirname [file normalize [info script]]]
proc lifetime {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project peer_memory [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    set defs [get_ipdefs -all -quiet xilinx.com:ip:axi_clock_converter:2.1]
    if {[llength $defs]!=1} {error "Installed AXI Clock Converter 2.1 required"}
    report_property -all $defs -file [file join $output ip_definition.rpt]
    create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_peer_memory_cdc
    set ip [get_ips ntrpu_peer_memory_cdc]
    report_property -all $ip -file [file join $output cdc_defaults.rpt]
    set config {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3}
    foreach {key value} $config {
        if {$key ni [list_property $ip]} {error "Unavailable CDC setting $key"}
    }
    set_property -dict $config $ip
    foreach {key value} $config {
        if {[get_property $key $ip] ne $value} {error "CDC setting mismatch $key"}
    }
    report_property -all $ip -file [file join $output cdc_configured.rpt]
    generate_target all $ip
    report_ip_status -file [file join $output ip_status.rpt]
    foreach name {ntrpu_buffer_pool_allocator ntrpu_buffer_pool_allocator_decoded
        ntrpu_peer_multibuffer_fillstage ntrpu_rv_elastic ntrpu_peer_multibuffer_elastic
        ntrpu_axi_capacity_guard ntrpu_axi_write_capacity_queue ntrpu_peer_memory_credit_port ntrpu_peer_multibuffer_credits ntrpu_fragment_pool_bridge ntrpu_reset_supervisor ntrpu_dual_mig_cold_start ntrpu_dual_multibuffer_mig ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_pcie_fragment_pool ntrpu_pcie_fragment_dual_control ntrpu_pcie_fragment_dual_managed ntrpu_pcie_fragment_dual_mig} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_pcie_managed_lifetime.sv]]
    set_property top ntrpu_pcie_fragment_dual_mig [get_filesets sources_1]
    set_property top tb_ntrpu_pcie_managed_lifetime [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    update_compile_order -fileset sim_1
    set wrapper [file normalize [file join $script_dir .. fpga rtl ntrpu_pcie_fragment_dual_mig.sv]]
    set f [open $wrapper r];set original [read $f];close $f
    set active_source $wrapper
    foreach variant {normal reset channel payload} {
        if {$variant ne "normal"} {
            switch $variant {
                reset {set before {assign lifetime_reset_n = lifetime_started;};set after {assign lifetime_reset_n = source_reset_n;}}
                channel {set before {.stop(pool_stop), .fault(pool_fault)};set after {.cmd_channel(~cmd_channel), .stop(pool_stop), .fault(pool_fault)}}
                payload {set before {.stop(pool_stop), .fault(pool_fault)};set after {.data(data ^ 512'd1), .stop(pool_stop), .fault(pool_fault)}}
            }
            if {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]!=2} {error "Mutation match count"}
            set replacement [file join $output ${variant}.sv]
            set f [open $replacement {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
            remove_files [get_files $active_source];add_files $replacement;set active_source $replacement
        }
        update_compile_order -fileset sim_1
        launch_simulation
        run all
        close_sim
        set log [file join $output project peer_memory.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${variant}.log]
        set f [open $log r];set text [read $f];close $f
        if {$variant eq "normal"} {
            if {![regexp -line {^NTRD_LIFETIME_SUITE_PASS cases=8$} $text] || [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Positive lifetime failed"}
        } else {
            set expected [dict get {reset LOST_LEASE channel MEMORY_ADDRESS payload MEMORY_PAYLOAD} $variant]
            if {![regexp -line [format {^Fatal: .*%s$} $expected] $text] || [regexp -line {^NTRD_LIFETIME_SUITE_PASS } $text]} {error "Lifetime mutant escaped: $variant"}
        }
        puts "NTRPU_MANAGED_LIFETIME_REVIEW variant=$variant PASS"
    }
    close_project
    puts NTRPU_MANAGED_LIFETIME_SUITE_PASS
}
if {[catch {lifetime $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
