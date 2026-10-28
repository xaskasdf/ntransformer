set script_dir [file dirname [file normalize [info script]]]
proc concurrent_log {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected EXAMPLE NEW_OUTPUT"}
    lassign $arguments example output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output;cd $output
    exec python3 [file join $script_dir prepare_ntrpu_pcie_concurrent_log.py] $example extracted > extraction.log 2>@1
    exec xvlog --sv -i extracted [file join $script_dir .. fpga tb ntrpu_pcie_concurrent_log_tb.sv] > compile.log 2>@1
    exec xelab board --debug typical --mt 2 -s concurrent_log > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all;quit};close $f
    set expected {PASS NTRPU_LOG_CHANNEL_SEQUENCE NTRPU_LOG_CHANNEL_SEQUENCE NTRPU_LOG_CHANNEL_UNKNOWN NTRPU_LOG_CHANNEL_OVERFLOW NTRPU_LOG_UNKNOWN_REMAINDER NTRPU_LOG_RQ_LAYOUT UNIT_CC_DATA NTRPU_LOG_RQ_LAYOUT UNIT_CALLBACK_FALSE_WAKE UNIT_CALLBACK_RX_VALUE}
    for {set mode 0} {$mode<11} {incr mode} {
        set code [catch {exec xsim concurrent_log -tclbatch run.tcl -testplusarg MODE=$mode -log mode${mode}.log > mode${mode}.console.log 2>@1} message]
        set f [open mode${mode}.console.log r];set text [read $f];close $f
        if {$mode==0} {
            if {$code || ![string match {*NTRPU_CONCURRENT_LOG_PASS rq=8 cc=8 bytes=2256 phases=8*} $text] ||
                ![string match {*NTRPU_CONCURRENT_CALLBACK_PASS cases=4 legacy_false_wakeup=1 received_data=12345678*} $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:)} $text]} {error "Concurrent logger positive rejected"}
        } elseif {![string match "*Fatal: [lindex $expected $mode]*" $text] || [string match {*NTRPU_CONCURRENT_LOG_PASS*} $text]} {
            error "Expected diagnostic absent in mode $mode"
        }
        puts "NTRPU_CONCURRENT_LOG_REVIEW mode=$mode expected=[lindex $expected $mode] PASS"
    }
    puts NTRPU_CONCURRENT_LOG_SUITE_PASS
}
if {[catch {concurrent_log $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
