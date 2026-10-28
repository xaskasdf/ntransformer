# Exercise orchestration with mocked XSim calls, not native DDR simulation.
if {[llength $argv] != 2} {error "Expected CAPTURE_SCRIPT NEW_TEST_DIRECTORY"}
lassign $argv capture_script output
set capture_script [file normalize $capture_script]
set output [file normalize $output]
if {[file exists $output]} {error "Test output exists"}
file mkdir $output
proc require {condition message} {if {![uplevel 1 [list expr $condition]]} {error $message}}
proc current_scope {args} {
    if {[llength $args]} {set ::scope [lindex $args 0]}
    return $::scope
}
set clocks {}
foreach slot {0 1} {
    foreach rank {0 1} {
        lappend clocks [format {/ntrpu_ddr_example_activity_tb/dimms[%d].memory/rcd/rank_instances[%d].rank/Micron_model.instance_of_sdram_devices[0].iDDR4/CK} $slot $rank]
    }
}
proc get_objects {args} {
    if {$args eq {-r CK}} {
        if {$::mode eq "missing_rank"} {return [lrange $::clocks 0 2]}
        return $::clocks
    }
    if {$args eq {-r *}} {return {DUT_OBJECTS_ONLY}}
    set name [lindex $args 0]
    if {[string index $name 0] eq "/"} {return [list $name]}
    return [list $::scope/$name]
}
proc get_value {args} {
    set name [file tail [lindex $args end]]
    set cycles [expr {$::time_ns / 5}]
    switch -- $name {
        capture_ready {return [expr {$::time_ns >= 36000}]}
        healthy {return [expr {$::mode ne "unhealthy"}]}
        prbs_running {return 1}
        cycles {return $cycles}
        sample_ps {return [expr {$cycles * 5000}]}
        aw_count - b_count - ar_count - rlast_count {return [expr {$cycles / 80}]}
        pending_w - pending_r - w_stalls - r_stalls {return 0}
        w_beats - r_beats {return [expr {$::mode eq "no_traffic" ? 0 : $cycles / 2}]}
        w_bytes {return [expr {32 * $cycles}]}
        default {error "Unexpected monitor $name"}
    }
}
proc current_time {args} {return "$::time_ns ns"}
proc run {amount unit} {
    require {$unit eq "ns"} "Wrong time unit"
    if {$::mode eq "short_window" && $amount == 20480} {incr amount -5}
    incr ::time_ns $amount
}
proc open_vcd {path} {
    lappend ::events [list vcd_open $::time_ns]
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "MOCK ONLY"
    close $f
}
proc log_vcd {objects} {require {[llength $objects] == 48} "Wrong pin selection"}
proc flush_vcd {} {}
proc close_vcd {} {
    if {$::mode eq "close_failure"} {error "Mock close failure"}
    lappend ::events [list vcd_close $::time_ns]
}
proc open_saif {path} {
    require {$::scope eq "/ntrpu_ddr_example_activity_tb/dut"} "Changed SAIF scope"
    lappend ::events [list saif_open $::time_ns]
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "MOCK ONLY"
    close $f
}
proc log_saif {objects} {require {$objects eq {DUT_OBJECTS_ONLY}} "SAIF objects changed"}
proc close_saif {} {lappend ::events [list saif_close $::time_ns]}
proc quit {} {error MOCK_QUIT}
set checks {}
foreach mode {normal missing_rank no_traffic unhealthy short_window close_failure} {
    file mkdir [file join $output $mode]
    cd [file join $output $mode]
    set scope /original
    set time_ns 0
    set events {}
    set channels [chan names]
    require {[catch {source $capture_script} result] && $result eq "MOCK_QUIT"} "Unexpected capture exit"
    foreach channel [chan names] {if {$channel ni $channels} {close $channel}}
    set completed [file exists activity_capture/CAPTURE_COMPLETE.txt]
    if {$mode eq "normal"} {
        require {$completed} "No completion receipt"
        require {$events eq {{vcd_open 0} {saif_open 36000} {saif_close 56480} {saif_open 56480} {saif_close 76960} {vcd_close 76980}}} "Trace/SAIF ordering or time changed"
        set f [open activity_capture/windows.tsv]
        set windows [read $f]
        close $f
        require {[string first "prbs_0\tstart\tsimulator_time\t36000 ns" $windows] >= 0} "Missing actual time"
    } else {
        require {!$completed} "Failed capture acquired completion receipt: $mode"
    }
    lappend checks $mode
}
puts "PASS_DDR_CAPTURE_R2_TCL cases=[llength $checks] native_simulation=0"
