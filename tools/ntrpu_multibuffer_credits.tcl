# Bounded, separate AXI memory-boundary experiment; never the board queue.
set script_dir [file dirname [file normalize [info script]]]
proc memory_side {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project peer_memory [file join $output project] -part xcvu5p-flva2104-2-i
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
        ntrpu_axi_capacity_guard ntrpu_axi_write_capacity_queue ntrpu_peer_memory_credit_port ntrpu_peer_multibuffer_credits} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_peer_multibuffer_credits.sv]]
    set_property top ntrpu_peer_multibuffer_credits [get_filesets sources_1]
    set_property top tb_ntrpu_peer_multibuffer_credits [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach slots {0 1 2 4} {
        set_property generic GUARD_SLOTS=$slots [get_filesets sim_1]
        foreach profile {0 1 2} {
            set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg PROFILE=$profile"] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project peer_memory.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output slots-${slots}-profile-${profile}.log]
            set f [open $log r]; set text [read $f]; close $f
            if {[string first "MEMORY_SUITE_PASS profile=$profile" $text]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Positive credit suite failed: slots=$slots profile=$profile"}
            puts "CREDIT_PROFILE_REVIEWED SLOTS=$slots PROFILE=$profile"
        }
    }
    foreach {negative marker} {1 MEMORY_PAYLOAD_MISMATCH 2 MEMORY_COMPLETION_STATUS_ORDER 3 MEMORY_UNAUTHORIZED_AW} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg PROFILE=0 -testplusarg NEGATIVE=$negative"] [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project peer_memory.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output negative-${negative}.log]
        set f [open $log r]; set text [read $f]; close $f
        if {[string first $marker $text]<0 || ![regexp -nocase {Fatal:} $text] || [string first MEMORY_SUITE_PASS $text]>=0} {error "Negative escaped: $negative"}
    }
    close_project
    puts NTRPU_MEMORY_CREDITS_EXPORTED
}
if {[catch {memory_side $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
