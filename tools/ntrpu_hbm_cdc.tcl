# Bounded isolated HBM startup and CDC studies. Frozen sources are preserved.
set root [file normalize [file join [file dirname [info script]] ..]]
proc save_config {out name} {
    set f [open [file join $out ${name}_config.tsv] w]
    foreach p [lsort [list_property [get_ips $name]]] {
        if {[string match CONFIG.* $p] || $p in {IPDEF IP_FILE}} {puts $f "$p\t[get_property $p [get_ips $name]]"}
    };close $f
}
proc run_stage {root out stage} {
    if {$stage eq "async"} {
        source [file join $root tools ntrpu_hbm_cdc_async.tcl]
        run_async $root $out
        return
    }
    set_param general.maxThreads 2
    file mkdir $out
    create_project hbm_cdc [file join $out project] -part xcvu47p-fsvh2892-2-e
    create_ip -name hbm -vendor xilinx.com -library ip -version 1.0 -module_name native_hbm
    set_property -dict [list CONFIG.USER_HBM_DENSITY {16GB} CONFIG.USER_SWITCH_ENABLE_00 {TRUE} \
        CONFIG.USER_SWITCH_ENABLE_01 {TRUE} CONFIG.USER_AXI_CLK_FREQ {450} CONFIG.USER_AXI_CLK1_FREQ {450}] [get_ips native_hbm]
    save_config $out native_hbm
    generate_target all [get_ips native_hbm]
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE [get_ips native_hbm]]]
    set ipdir [file join $out project hbm_cdc.gen sources_1 ip native_hbm]
    file copy [file join $ipdir native_hbm.veo] [file join $out native_hbm.veo]
    file copy [get_property IP_FILE [get_ips native_hbm]] [file join $out native_hbm.xci]
    exec python3 -B [file join $root tools prepare_ntrpu_hbm_cdc.py] $out --stage $stage
    add_files [file join $out binding ntrpu_hbm_native_link.sv]
    add_files -fileset sim_1 [file join $out tb_ntrpu_hbm_boot.sv]
    set_property top tb_ntrpu_hbm_boot [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation;run all;close_sim
    set log [file join $out project hbm_cdc.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $out simulation.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first {BOOT PASS } $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "HBM boot failed"}
    set copies [file join $out model_inputs];file mkdir $copies
    foreach p [glob -nocomplain [file join $ipdir hdl rtl *.mem]] {file copy $p [file join $copies [file tail $p]]}
    close_project
    puts "NTRPU_HBM_CDC_COMPLETE $stage"
}
if {[llength $argv]!=2 || [lindex $argv 1] ni {boot async}} {error "Expected output and stage"}
if {[catch {run_stage $root [file normalize [lindex $argv 0]] [lindex $argv 1]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
