set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_logger_unit {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2} { error "Expected EXAMPLE NEW_DIRECTORY" }
    lassign $arguments example output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output; cd $output
    exec python3 [file join $ntrpu_script_dir prepare_ntrpu_pcie_logger_unit.py] $example extracted > extraction.log 2>@1
    exec xvlog --sv -i extracted [file join $ntrpu_script_dir .. fpga tb ntrpu_pcie_logger_tb.sv] > compile.log 2>@1
    exec xelab board --debug typical --mt 2 -s logger > elaborate.log 2>@1
    set stream [open run.tcl {WRONLY CREAT EXCL}]; puts $stream {run all; quit}; close $stream
    set expected {PASS NTRPU_LOG_RQ_MULTIBEAT_UNSUPPORTED NTRPU_LOG_RQ_LAYOUT NTRPU_LOG_RQ_SIZE_UNSUPPORTED NTRPU_LOG_UNKNOWN_REMAINDER NTRPU_LOG_RQ_LAYOUT}
    for {set mode 0} {$mode < 6} {incr mode} {
        set code [catch {exec xsim logger -tclbatch run.tcl -testplusarg MODE=$mode -log mode${mode}.log > mode${mode}.console.log 2>@1} message]
        set stream [open mode${mode}.console.log r]; set text [read $stream]; close $stream
        if {$mode == 0} {
            if {$code || ![string match {*NTRPU_LOGGER_UNIT_PASS checks=42 remainder_cases=32 original_defect=1*} $text] ||
                ![string match {*NTRPU_LOGGER_STATIC_RACE_REPRODUCED*} $text] ||
                ![string match {*NTRPU_LOGGER_CONCURRENT_SYNC_PASS parses=1 bytes=76*} $text] ||
                [regexp -nocase {\m(error|fatal)\M} $text]} { error "Logger positive test rejected" }
        } else {
            if {![string match "*Fatal: [lindex $expected $mode]*" $text] ||
                [string match {*NTRPU_LOGGER_UNIT_PASS*} $text]} { error "Logger negative mode $mode not rejected" }
        }
        puts "NTRPU_LOGGER_REVIEW mode=$mode expected=[lindex $expected $mode] PASS"
    }
    puts "NTRPU_LOGGER_SUITE_PASS positive=1 negative=5"
}
if {[catch {ntrpu_logger_unit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
