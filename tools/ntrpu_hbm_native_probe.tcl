# Read configuration and exported native ports on an exploratory HBM part.
# This does not change the board target or claim a runnable HBM subsystem.
set_param general.maxThreads 2
if {[llength $argv]!=2 || [lindex $argv 1] ni {probe link}} {error "Expected output directory and stage"}
set out [file normalize [lindex $argv 0]]
set stage [lindex $argv 1]
file mkdir $out
create_project native_probe [file join $out project] -part xcvu47p-fsvh2892-2-e
create_ip -name hbm -vendor xilinx.com -library ip -version 1.0 -module_name native_hbm
set ip [get_ips native_hbm]
if {$stage eq "link"} {
    set_property -dict [list CONFIG.USER_HBM_DENSITY {16GB} CONFIG.USER_SWITCH_ENABLE_00 {TRUE} \
        CONFIG.USER_SWITCH_ENABLE_01 {TRUE} CONFIG.USER_AXI_CLK_FREQ {450} CONFIG.USER_AXI_CLK1_FREQ {450}] $ip
    if {[get_property CONFIG.USER_HBM_STACK $ip]!=2 || [get_property CONFIG.USER_STACK_8HI $ip] ne "true"} {error "Expected two 8-high stacks"}
}
report_property -file [file join $out properties.rpt] $ip
set f [open [file join $out config.tsv] w]
foreach p [lsort [list_property $ip]] {
    if {[string match CONFIG.* $p] || $p in {IPDEF IP_FILE}} {puts $f "$p\t[get_property $p $ip]"}
}
close $f
generate_target instantiation_template $ip
foreach extension {veo vho} {
    set paths [glob -nocomplain [file join $out project native_probe.gen sources_1 ip native_hbm *.${extension}]]
    foreach p $paths {file copy $p [file join $out [file tail $p]]}
}
set f [open [file join $out part.txt] w]
puts $f [get_property PART [current_project]]
close $f
if {$stage eq "link"} {
    set root [file normalize [file join [file dirname [info script]] ..]]
    exec python3 -B [file join $root tools prepare_ntrpu_hbm_native_link.py] [file join $out native_hbm.veo] [file join $out binding]
    generate_target synthesis $ip
    set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
    add_files [file join $out binding ntrpu_hbm_native_link.sv]
    synth_design -rtl -top ntrpu_hbm_native_link -part xcvu47p-fsvh2892-2-e
    set f [open [file join $out elaboration.txt] w]
    puts $f "TOP [get_property TOP [current_fileset]]"
    puts $f "PART [get_property PART [current_project]]"
    puts $f "HBM_INSTANCE [get_cells -hier -filter {REF_NAME == native_hbm}]"
    close $f
    close_design
}
close_project
puts "NTRPU_NATIVE_HBM_PROBE_COMPLETE $stage"
exit 0
