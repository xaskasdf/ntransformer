set root [file normalize [file join [file dirname [info script]] ..]]
set_param general.maxThreads 2
proc local_master {root output} {
    if {[file exists $output]} {error "Output exists"}
    create_project local_master [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $root fpga rtl ntrpu_xdma_local_master.sv]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_xdma_local_master.sv]
    set_property top tb_ntrpu_xdma_local_master [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    set signals {clk cold_rst_n admit req_valid req_ready req_write req_address req_data req_strobe req_identity rsp_valid rsp_ready rsp_data rsp_code rsp_identity rsp_timeout timeout_latched busy phase}
    foreach channel {aw w b ar r} {
        foreach suffix {valid ready} {lappend signals m_axil_${channel}${suffix}}
    }
    foreach name {awaddr awprot wdata wstrb bresp araddr arprot rdata rresp} {lappend signals m_axil_$name}
    set objects {}
    foreach name $signals {
        set found [get_objects /tb_ntrpu_xdma_local_master/$name]
        if {[llength $found]!=1} {error "Missing port $name"}
        lappend objects {*}$found
    }
    open_vcd [file join $output local_master.vcd]
    log_vcd $objects
    run all
    close_vcd
    close_sim
    set path [file join $output project local_master.sim sim_1 behav xsim simulate.log]
    file copy $path [file join $output simulate.log]
    set f [open $path r];set log [read $f];close $f
    if {[lsearch -exact [split $log \n] {LOCAL_MASTER_PASS cases=10 native_xdma=0 reset_permission=0}]<0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:)} $log]} {error "Local master tests failed"}
    close_project
    puts NTRPU_LOCAL_MASTER_SIMULATED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {local_master $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
