# Isolated HBM-compatible write conversion. No HBM hard controller or CDC.
set root [file normalize [file join [file dirname [info script]] ..]]
proc save_ip {output name} {
    set ip [get_ips $name]
    report_property -file [file join $output ${name}_properties.rpt] $ip
    set f [open [file join $output ${name}_config.tsv] w]
    foreach p [lsort [list_property $ip]] {
        if {[string match CONFIG.* $p] || $p in {IPDEF IP_FILE}} {puts $f "$p\t[get_property $p $ip]"}
    }
    close $f
}
proc hbm_adapter {root output stage} {
    set_param general.maxThreads 2
    file mkdir $output
    create_project hbm_adapter [file join $output project] -part xcvu5p-flva2104-2-i
    create_ip -name axi_dwidth_converter -vendor xilinx.com -library ip -version 2.1 -module_name hbm_width
    create_ip -name axi_protocol_converter -vendor xilinx.com -library ip -version 2.1 -module_name hbm_protocol
    if {$stage ne "probe"} {
        set_property -dict [list CONFIG.PROTOCOL {AXI4} CONFIG.READ_WRITE_MODE {WRITE_ONLY} \
            CONFIG.ADDR_WIDTH {34} CONFIG.SI_DATA_WIDTH {512} CONFIG.MI_DATA_WIDTH {256} \
            CONFIG.SI_ID_WIDTH {4} CONFIG.MAX_SPLIT_BEATS {16} CONFIG.ACLK_ASYNC {0} CONFIG.FIFO_MODE {0}] [get_ips hbm_width]
        set_property -dict [list CONFIG.SI_PROTOCOL {AXI4} CONFIG.MI_PROTOCOL {AXI3} \
            CONFIG.READ_WRITE_MODE {WRITE_ONLY} CONFIG.ADDR_WIDTH {34} CONFIG.DATA_WIDTH {256} \
            CONFIG.ID_WIDTH {4} CONFIG.TRANSLATION_MODE {2}] [get_ips hbm_protocol]
        foreach name {hbm_width hbm_protocol} {
            generate_target all [get_ips $name]
            set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips $name]]]
            file copy [file join $output project hbm_adapter.gen sources_1 ip $name sim ${name}.v] [file join $output ${name}_wrapper.v]
        }
    }
    foreach name {hbm_width hbm_protocol} {save_ip $output $name}
    if {$stage eq "sim"} {
        foreach name {ntrpu_rv_elastic ntrpu_axi_write_capacity_queue ntrpu_hbm_write_adapter} {
            add_files [file join $root fpga rtl ${name}.sv]
        }
        add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_hbm_write_adapter.sv]
        set_property top tb_ntrpu_hbm_write_adapter [get_filesets sim_1]
        set_property xsim.elaborate.debug_level off [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        launch_simulation;run all;close_sim
        set log [file join $output project hbm_adapter.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output simulation.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first {HA ALL_PASS cases=11 } $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "HBM adapter simulation failed"}
    }
    close_project
    puts "NTRPU_HBM_ADAPTER_COMPLETE $stage"
}
if {[llength $argv]!=2 || [lindex $argv 1] ni {probe ip sim}} {error "Expected output and stage"}
if {[catch {hbm_adapter $root [file normalize [lindex $argv 0]] [lindex $argv 1]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
