set script_dir [file dirname [file normalize [info script]]]
proc multibuffer_netlist {arguments} {
    global script_dir
    if {[llength $arguments] != 3} {error "Expected RESERVE SYNTH_PROFILE NEW_DIRECTORY"}
    lassign $arguments reserve input output
    if {$reserve ni {4 8}} {error "Unvalidated profile"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    file copy [file join $input multibuffer_funcsim.v] [file join $output input_netlist.v]
    exec python3 -B [file join $script_dir prepare_ntrpu_multibuffer_netlist_tb.py] \
        [file join $script_dir .. fpga tb tb_ntrpu_peer_multibuffer.sv] [file join $output netlist_tb.sv]
    create_project multibuffer_netlist [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $output input_netlist.v]
    set_property top ntrpu_peer_multibuffer [get_filesets sources_1]
    add_files -fileset sim_1 [file join $output netlist_tb.sv]
    set_property top tb_ntrpu_peer_multibuffer [get_filesets sim_1]
    set_property generic "CHANNELS=2 RESERVE_BANKS=$reserve BANK_BEATS=64 DESCRIPTORS=16 MAX_OUTSTANDING=4" [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set logfile [file join $output project multibuffer_netlist.sim sim_1 behav xsim simulate.log]
    file copy $logfile [file join $output simulation.log]
    set f [open $logfile r]; set log [read $f]; close $f
    if {![regexp {PASS seed=1 cycles=[0-9]+ peak_pending=4} $log] ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $log]} {
        error "Functional netlist simulation failed; diagnostics preserved"
    }
    puts "MULTIBUFFER_NETLIST_EXPORTED RESERVE=$reserve"
    close_project
}
if {[catch {multibuffer_netlist $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
