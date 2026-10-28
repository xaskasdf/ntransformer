# Original RoPE HLS with actual AXI loader and owner; 5 ns OOC.
set kv_script_dir [file dirname [file normalize [info script]]]
proc rope_qkv_real_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project rope_qkv_real [file join $output project] -part xcvu5p-flva2104-2-i
    set arithmetic_rtl [file join $output rope_input_rtl]
    file copy [file join $kv_script_dir .. generated rope_input] $arithmetic_rtl
    cd $arithmetic_rtl
    add_files [lsort [glob *.v]]
    foreach script [lsort [glob *_ip.tcl]] {source $script}
    foreach name {fpga/rtl/ntrpu_rope_qkv_dispatch_r1.sv fpga/rtl/ntrpu_rope_input_real_r1.sv fpga/rtl/ntrpu_rope_input_path_r2.sv fpga/rtl/ntrpu_rope_owner_r2.sv fpga/rtl/ntrpu_rope_loader_r1.sv fpga/rtl/ntrpu_rope_hls_ports_r1.sv fpga/rtl/ntrpu_qkv_capture_dispatch_r1.sv fpga/rtl/ntrpu_qkv_snapshot_r1.sv fpga/rtl/ntrpu_qkv_dispatch_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv fpga/rtl/ntrpu_attention_root_guard_r1.sv fpga/rtl/ntrpu_attention_head_sequencer_r1.sv} {
        add_files [file normalize [file join $kv_script_dir .. $name]]
    }
    set_property top ntrpu_rope_qkv_dispatch_r1 [get_filesets sources_1]
    puts NTRPU_ROPE_QKV_REAL_R1_REUSES_PREDECESSOR_SYNTHESIS
    add_files -fileset sim_1 [file normalize [file join $kv_script_dir .. fpga tb tb_ntrpu_rope_qkv_real_r1.sv]]
    set_property top ntrpu_rope_qkv_dispatch_r1 [get_filesets sources_1]
    set_property top tb_ntrpu_rope_qkv_real_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $kv_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    foreach {seed name} {1 seed-1} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed -testplusarg FIXTURE=[file normalize [file join $kv_script_dir .. fixture]]"] [get_filesets sim_1]
        update_compile_order -fileset sources_1
        update_compile_order -fileset sim_1
        report_ip_status -file [file join $output ip_status.rpt]
        launch_simulation
        run all
        close_sim
        set log [file join $output project rope_qkv_real.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "ROPEQKV_REAL_PASS seed=$seed jobs=82 " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |DISPATCH_FAIL)} $text]} {error "RoPE input arithmetic failed: $name"}
    }
    close_project
    puts NTRPU_ROPE_QKV_REAL_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {rope_qkv_real_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
