set local_master_script_dir [file dirname [file normalize [info script]]]
set_param general.maxThreads 2
proc replay_local_master {seed output} {
    global local_master_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    foreach limit {32 65536} {
        foreach mode {rtl netlist} {
            set dir [file join $output wait${limit}-$mode]
            file mkdir $dir
            create_project local_master [file join $dir project] -part xcvu5p-flva2104-2-i
            set defines [list NTRPU_REPLAY_LIMIT=$limit]
            if {$mode eq "netlist"} {
                set input [file join $seed wait$limit local_master_funcsim.v]
                if {![file isfile $input]} {error "Missing synthesized netlist"}
                file copy $input [file join $dir local_master_funcsim.v]
                add_files [file join $dir local_master_funcsim.v]
                lappend defines NTRPU_GATE_NETLIST
            } else {
                add_files [file join $local_master_script_dir .. fpga rtl ntrpu_xdma_local_master.sv]
            }
            add_files -fileset sim_1 [file join $local_master_script_dir .. fpga tb tb_ntrpu_xdma_local_master_replay_r2.sv]
            set_property top tb_ntrpu_xdma_local_master_replay [get_filesets sim_1]
            set_property verilog_define $defines [get_filesets sim_1]
            puts "NTRPU_REPLAY_TOP_SCOPES [get_property top [get_filesets sim_1]]"
            # Use Vivado default elaboration libraries, including unisims_ver.
            set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
            report_property -all -file [file join $dir simulation_properties.rpt] [get_filesets sim_1]
            update_compile_order -fileset sim_1
            launch_simulation
            set signals {clk cold_rst_n admit req_valid req_ready req_write req_address req_data req_strobe req_identity rsp_valid rsp_ready rsp_data rsp_code rsp_identity rsp_timeout timeout_latched busy phase}
            foreach channel {aw w b ar r} {
                foreach suffix {valid ready} {lappend signals m_axil_${channel}${suffix}}
            }
            foreach name {awaddr awprot wdata wstrb bresp araddr arprot rdata rresp} {lappend signals m_axil_$name}
            set objects {}
            foreach name $signals {
                set found [get_objects /tb_ntrpu_xdma_local_master_replay/$name]
                if {[llength $found]!=1} {error "Missing public port $name"}
                lappend objects {*}$found
            }
            open_vcd [file join $dir local_master.vcd]
            log_vcd $objects
            run all
            close_vcd
            close_sim
            set logpath [file join $dir project local_master.sim sim_1 behav xsim simulate.log]
            file copy $logpath [file join $dir simulate.log]
            set f [open $logpath r];set log [read $f];close $f
            if {[lsearch -exact [split $log \n] {LOCAL_MASTER_PASS cases=10 native_xdma=0 reset_permission=0}]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:)} $log]} {error "Replay failed: $limit $mode"}
            close_project
            puts "NTRPU_LOCAL_MASTER_REPLAY_RETURNED limit=$limit mode=$mode"
        }
    }
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {replay_local_master [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
