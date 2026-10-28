# Replay the existing public-pin bank against an immutable synthesized netlist.
set script_dir [file dirname [file normalize [info script]]]
proc peer_netlist {arguments} {
    global script_dir
    if {[llength $arguments] != 3} {error "Expected MAX_BEATS SYNTH_PROFILE NEW_DIRECTORY"}
    lassign $arguments beats input output
    if {$beats ni {4 64}} {error "Unvalidated profile"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output already exists"}
    file mkdir $output
    set_param general.maxThreads 2
    file copy [file join $input peer_rx_funcsim.v] [file join $output input_netlist.v]
    set f [open [file join $script_dir .. fpga tb tb_ntrpu_peer_rx_guard.sv] r]
    set bench [read $f]; close $f
    set parameterized {ntrpu_peer_rx_guard #(.MAX_BEATS(MAX_BEATS)) dut (.*);}
    set initial_start "initial begin\n        repeat (3) step();"
    foreach pattern [list $parameterized $initial_start] {
        if {[string first $pattern $bench] < 0 || [string first $pattern $bench] != [string last $pattern $bench]} {
            error "Testbench transformation must match exactly once"
        }
    }
    # The synthesized module is specialized already; only the bench retains
    # its MAX_BEATS generic. Wait for vendor GSR before applying cold reset.
    set bench [string map [list $parameterized {ntrpu_peer_rx_guard dut (.*);} \
        $initial_start "initial begin\n        #200;\n        repeat (3) step();"] $bench]
    set f [open [file join $output netlist_tb.sv] w]; puts $f $bench; close $f
    create_project peer_netlist [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file join $output input_netlist.v]
    set_property top ntrpu_peer_rx_guard [get_filesets sources_1]
    add_files -fileset sim_1 [file join $output netlist_tb.sv]
    set_property top tb_ntrpu_peer_rx_guard [get_filesets sim_1]
    set_property generic MAX_BEATS=$beats [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    set logfile [file join $output project peer_netlist.sim sim_1 behav xsim simulate.log]
    file copy $logfile [file join $output simulation.log]
    set f [open $logfile r]; set log [read $f]; close $f
    if {![string match "*PASS cases=37 max_beats=$beats *" $log] ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $log]} {
        error "Netlist replay failed; original logs preserved"
    }
    puts "NTRPU_PEER_RX_NETLIST_EXPORTED MAX_BEATS=$beats"
    close_project
}
if {[catch {peer_netlist $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
