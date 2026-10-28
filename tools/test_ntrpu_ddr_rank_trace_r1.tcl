# Mock XSim object queries only: this does not elaborate or simulate DDR.
if {[llength $argv] != 2} {error "Expected HELPER NEW_TEST_DIRECTORY"}
lassign $argv helper output
source $helper
file mkdir $output
set scope /original
set mode normal
set logged {}
set opened 0
set closed 0
set clocks {}
foreach slot {0 1} {
    foreach rank {0 1} {
        lappend clocks [format {/ntrpu_ddr_example_activity_tb/dimms[%d].memory/rcd/rank_instances[%d].rank/Micron_model.instance_of_sdram_devices[0].iDDR4/CK} $slot $rank]
    }
}
proc current_scope {args} {
    if {[llength $args]} {set ::scope [lindex $args 0]}
    return $::scope
}
proc get_objects {args} {
    if {[llength $args]==2} {
        if {$::mode eq "missing_rank"} {return [lrange $::clocks 0 2]}
        if {$::mode eq "duplicate_rank"} {return [concat $::clocks [lindex $::clocks 0]]}
        return $::clocks
    }
    set pin [lindex $args 0]
    if {$::mode eq "missing_pin" && $pin eq "TEN"} {return {}}
    if {$::mode eq "wrong_scope" && $pin eq "TEN"} {return /alien/TEN}
    return [list $::scope/$pin]
}
proc open_vcd {path} {incr ::opened;set f [open $path {WRONLY CREAT EXCL}];close $f}
proc log_vcd {objects} {set ::logged $objects}
proc flush_vcd {} {}
proc close_vcd {} {incr ::closed}
proc require {condition message} {if {![uplevel 1 [list expr $condition]]} {error $message}}
ntrpu_rank_trace_open [file join $output valid.vcd] [file join $output valid.tsv]
require {[llength $logged]==48 && $scope eq "/original" && $opened==1} "Valid trace selection"
ntrpu_rank_trace_close
require {$closed==1} "Trace close"
set f [open [file join $output valid.tsv]];set rows [split [string trim [read $f]] \n];close $f
require {[llength $rows]==49} "Inventory size"
set checks {selection_48 scope_restored close inventory}
foreach mode {missing_rank duplicate_rank missing_pin wrong_scope} {
    require {[catch {ntrpu_rank_trace_open [file join $output $mode.vcd] [file join $output $mode.tsv]}]} "Accepted $mode"
    require {$opened==1 && $scope eq "/original"} "Partial selection acquired trace or changed scope"
    lappend checks $mode
}
set mode normal
require {[catch {ntrpu_rank_trace_open [file join $output valid.vcd] [file join $output retry.tsv]}]} "Overwrote VCD"
require {[catch {ntrpu_rank_trace_open [file join $output one] [file join $output one]}]} "Identical outputs"
lappend checks existing_output identical_outputs
puts "PASS_RANK_TRACE_MOCK checks=[llength $checks] cases=[join $checks ,] native_simulation=0"
