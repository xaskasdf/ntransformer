set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ready_gate_unit {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 1} { error "Expected NEW_DIRECTORY" }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    cd $output
    set tb [file normalize [file join $ntrpu_script_dir .. fpga tb]]
    set stream [open [file join $tb ntrpu_sim_ready_gate.sv] r]; set gate [read $stream]; close $stream
    if {[regexp -all {held_out \|\| \(phase >= BLOCKED\)} $gate] != 1} { error "Unexpected gate expression" }
    set broken [string map [list {held_out || (phase >= BLOCKED)} {(phase >= BLOCKED)}] $gate]
    set stream [open broken_gate.sv {WRONLY CREAT EXCL}]; puts -nonewline $stream $broken; close $stream
    set stream [open run.tcl {WRONLY CREAT EXCL}]; puts $stream {run all; quit}; close $stream
    set expected {PASS GATE_SOURCE_CHANGED GATE_OUTPUT_CHANGED PASS PASS GATE_SOURCE_CHANGED}
    for {set mode 0} {$mode < 6} {incr mode} {
        set source [expr {$mode == 2 ? "broken_gate.sv" : [file join $tb ntrpu_sim_ready_gate.sv]}]
        exec xvlog --sv $source [file join $tb ntrpu_sim_ready_gate_tb.sv] > compile${mode}.log 2>@1
        set top ntrpu_sim_ready_gate_tb
        if {$mode in {3 5}} { set top ntrpu_sim_ready_gate_delay4_tb }
        if {$mode == 4} { set top ntrpu_sim_ready_gate_delay9_tb }
        exec xelab $top --debug typical --mt 2 -s gate${mode} > elaborate${mode}.log 2>@1
        set stimulus [expr {$mode in {1 5} ? 1 : 0}]
        set code [catch {exec xsim gate${mode} -tclbatch run.tcl -testplusarg MODE=$stimulus -log mode${mode}.log > mode${mode}.console.log 2>@1} message]
        set stream [open mode${mode}.console.log r]; set text [read $stream]; close $stream
        if {[lindex $expected $mode] eq "PASS"} {
            if {$code || ![string match {*NTRPU_READY_GATE_UNIT_PASS sent=64 received=64*} $text] ||
                [regexp -nocase {\m(error|fatal)\M} $text]} { error "Ready gate positive test rejected" }
        } else {
            if {![string match "*Fatal: [lindex $expected $mode]*" $text] ||
                [string match {*NTRPU_READY_GATE_UNIT_PASS*} $text]} { error "Negative mode $mode did not reject as expected" }
        }
        puts "NTRPU_READY_GATE_REVIEW mode=$mode expected=[lindex $expected $mode] PASS"
    }
    puts "NTRPU_READY_GATE_SUITE_PASS positive=3 negative=3"
}
if {[catch {ntrpu_ready_gate_unit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
