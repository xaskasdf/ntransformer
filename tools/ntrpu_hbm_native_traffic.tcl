# Compile a fresh native-controller probe without spending its startup window.
set_param general.maxThreads 2
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file normalize [lindex $argv 0]]
if {[catch {
    file mkdir $out
    create_project hbm_native_traffic [file join $out project] -part xcvu47p-fsvh2892-2-e
    create_ip -name hbm -vendor xilinx.com -library ip -version 1.0 -module_name native_hbm
    set_property -dict [list CONFIG.USER_HBM_DENSITY {16GB} CONFIG.USER_SWITCH_ENABLE_00 {TRUE} \
        CONFIG.USER_SWITCH_ENABLE_01 {TRUE} CONFIG.USER_AXI_CLK_FREQ {450} CONFIG.USER_AXI_CLK1_FREQ {450}] [get_ips native_hbm]
    set f [open [file join $out native_hbm_config.tsv] w]
    foreach key [lsort [list_property [get_ips native_hbm]]] {
        if {[string match CONFIG.* $key] || $key eq "IPDEF"} {puts $f "$key\t[get_property $key [get_ips native_hbm]]"}
    };close $f
    generate_target all [get_ips native_hbm]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips native_hbm]]]
    set ipdir [file join $out project hbm_native_traffic.gen sources_1 ip native_hbm]
    file copy [file join $ipdir native_hbm.veo] [file join $out native_hbm.veo]
    file copy [get_property IP_FILE [get_ips native_hbm]] [file join $out native_hbm.xci]
    file mkdir [file join $out model_inputs]
    foreach p [glob -nocomplain [file join $ipdir hdl rtl *.mem]] {file copy $p [file join $out model_inputs [file tail $p]]}
    exec python3 -B [file join $root tools prepare_ntrpu_hbm_native_traffic.py] $out
    add_files [file join $out ntrpu_hbm_native_rw.sv]
    add_files -fileset sim_1 [file join $out tb_ntrpu_hbm_native_traffic.sv]
    set_property top tb_ntrpu_hbm_native_traffic [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation -step compile
    launch_simulation -step elaborate
    close_project
} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
puts NTRPU_NATIVE_TRAFFIC_COMPILED
exit 0
