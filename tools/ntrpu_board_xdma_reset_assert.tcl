# Native observation of optional reset assertion. No reset release or MIG.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_reset_assert {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=2 || [llength [get_projects -quiet]]} {
        error "Use fresh process: RESET_ENABLED_EXAMPLE NEW_OUTPUT"
    }
    lassign $arguments input output
    set input [file normalize $input]; set output [file normalize $output]
    if {[file exists $output] || ![file exists [file join $input ntrpu_xdma_ex.xpr]]} {error "Invalid paths"}
    file mkdir $output
    file copy $input [file join $output example]
    exec python3 [file join $ntrpu_script_dir patch_ntrpu_xdma_reset_assert.py] [file join $output example] > [file join $output patch.log] 2>@1
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    foreach f [get_files -all] {
        if {[string first $input [file normalize $f]]==0} {error "Reference to immutable seed: $f"}
    }
    foreach {key value} {CONFIG.functional_mode DMA CONFIG.soft_reset_en true CONFIG.pipe_sim true CONFIG.axist_bypass_en true CONFIG.axilite_master_en true} {
        if {[get_property $key [get_ips ntrpu_xdma]] ne $value} {error "Wrong $key"}
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb ntrpu_xdma_reset_assert.sv]
    set_property top board [get_filesets sim_1]
    set_property xsim.elaborate.debug_level typical [get_filesets sim_1]
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.xsim.more_options {-testplusarg TESTNAME=ntrpu_reset_assert} [get_filesets sim_1]
    report_property -all -file [file join $output simulation_properties.rpt] [get_filesets sim_1]
    report_ip_status -file [file join $output ip_status.rpt]
    update_compile_order -fileset sim_1
    launch_simulation
    set names {/board/sys_rst_n /board/ep_sys_clk_p /board/RP/user_clk /board/RP/user_lnk_up /board/RP/cfg_ltssm_state /board/EP/user_clk /board/EP/user_resetn /board/EP/user_lnk_up /board/ntrpu_reset_probe/reset_n /board/ntrpu_reset_probe/phase}
    foreach prefix {s_axis_rq m_axis_rc} {
        foreach suffix {tvalid tready} {lappend names /board/RP/${prefix}_${suffix}}
    }
    foreach prefix {m_axi m_axib m_axil} {
        foreach suffix {awvalid wvalid bvalid arvalid rvalid} {lappend names /board/EP/${prefix}_${suffix}}
    }
    set objects {}
    set f [open [file join $output trace_objects.txt] {WRONLY CREAT EXCL}]
    foreach name $names {
        set found [get_objects $name]
        if {[llength $found]!=1} {error "Missing native signal: $name"}
        puts $f $name; lappend objects {*}$found
    }
    close $f
    open_vcd [file join $output native.vcd]
    log_vcd $objects
    set previous [current_time -s]
    for {set i 0} {$i<201} {incr i} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_RESET_ASSERT_PROGRESS time=$now"
        flush stdout
        if {$now eq $previous} {break}
        set previous $now
    }
    close_vcd
    close_sim
    close_project
    puts NTRPU_RESET_ASSERT_SIMULATOR_RETURNED
}
if {[catch {ntrpu_reset_assert $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
