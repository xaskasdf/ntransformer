set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc axi4_drain_sim {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project drain $output -part xcvu5p-flva2104-2-i
    add_files [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_drain_monitor.sv]
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_drain_monitor.sv]
    set_property top tb_ntrpu_axi4_drain_monitor [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation
    run all
    close_sim
    set log [file join $output drain.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output drain.log]
    set f [open $log r];set text [read $f];close $f
    if {[string first "AXI4_DRAIN_SUITE_PASS cases=22 native_endpoint=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "AXI4 drain bank failed"}
    close_project
    puts NTRPU_AXI4_DRAIN_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_drain_sim [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
