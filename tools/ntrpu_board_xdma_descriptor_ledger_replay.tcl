set descriptor_ledger_script_dir [file dirname [file normalize [info script]]]
set_param general.maxThreads 2
proc replay_descriptor_ledger {seed output} {
    global descriptor_ledger_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    foreach bits {32 4} {
        foreach mode {rtl netlist} {
            set dir [file join $output bits${bits}-$mode]
            file mkdir $dir
            create_project descriptor_ledger [file join $dir project] -part xcvu5p-flva2104-2-i
            set defines [list NTRPU_LEDGER_BITS=$bits]
            if {$mode eq "netlist"} {
                set input [file join $seed bits$bits descriptor_ledger_funcsim.v]
                if {![file isfile $input]} {error "Missing synthesized netlist"}
                file copy $input [file join $dir descriptor_ledger_funcsim.v]
                add_files [file join $dir descriptor_ledger_funcsim.v]
                lappend defines NTRPU_GATE_NETLIST
            } else {
                add_files [file join $descriptor_ledger_script_dir .. fpga rtl ntrpu_xdma_descriptor_ledger.sv]
            }
            add_files -fileset sim_1 [file join $descriptor_ledger_script_dir .. fpga tb tb_ntrpu_xdma_descriptor_ledger_replay.sv]
            set_property top tb_ntrpu_xdma_descriptor_ledger [get_filesets sim_1]
            set_property verilog_define $defines [get_filesets sim_1]
            puts "NTRPU_REPLAY_TOP_SCOPES [get_property top [get_filesets sim_1]]"
            # Use Vivado default elaboration libraries, including unisims_ver.
            set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
            report_property -all -file [file join $dir simulation_properties.rpt] [get_filesets sim_1]
            update_compile_order -fileset sim_1
            launch_simulation
            set signals {clk cold_rst_n core_ready arm_valid arm_ready arm_generation arm_tag
                publish_valid publish_channel publish_generation publish_tag publish_index
                h2c_status c2h_status close_request host_closed host_generation host_tag release_valid
                active closed generation tag started faults published completed outstanding accounted phase}
            set objects {}
            foreach name $signals {
                set found [get_objects /tb_ntrpu_xdma_descriptor_ledger/$name]
                if {[llength $found]!=1} {error "Missing public port $name"}
                lappend objects {*}$found
            }
            open_vcd [file join $dir ledger.vcd]
            log_vcd $objects
            run all
            close_vcd
            close_sim
            set logpath [file join $dir project descriptor_ledger.sim sim_1 behav xsim simulate.log]
            file copy $logpath [file join $dir simulate.log]
            set f [open $logpath r];set log [read $f];close $f
            set cases [expr {$bits==4 ? 16 : 15}]
            if {[lsearch -exact [split $log \n] "LEDGER_PASS cases=$cases bits=$bits native_xdma=0 global_drain=0 reset_permission=0"]<0 ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:)} $log]} {error "Replay failed: $bits $mode"}
            close_project
            puts "NTRPU_DESCRIPTOR_LEDGER_REPLAY_RETURNED bits=$bits mode=$mode"
        }
    }
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {replay_descriptor_ledger [file normalize [lindex $argv 0]] [file normalize [lindex $argv 1]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
