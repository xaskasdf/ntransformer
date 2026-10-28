set script_dir [file dirname [file normalize [info script]]]
proc rx_log {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected EXAMPLE NEW_OUTPUT"}
    lassign $arguments example output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output;cd $output
    exec python3 [file join $script_dir prepare_ntrpu_pcie_rx_log.py] $example extracted > extraction.log 2>@1
    exec xvlog --sv -i extracted [file join $script_dir .. fpga tb ntrpu_pcie_rx_log_tb.sv] > compile.log 2>@1
    exec xelab board --debug typical --mt 2 -s rx_log > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all;quit};close $f
    set expected {PASS NTRPU_RX_LOG_ARGUMENT NTRPU_RX_LOG_OVERFLOW NTRPU_RX_LOG_SIZE NTRPU_RX_LOG_ARGUMENT UNIT_RX_BYTE UNIT_RX_PACKET_SIZE NTRPU_RX_LOG_PARSER_DRAIN NTRPU_RX_LOG_ARGUMENT}
    for {set mode 0} {$mode<9} {incr mode} {
        set code [catch {exec xsim rx_log -tclbatch run.tcl -testplusarg MODE=$mode -log mode${mode}.log > mode${mode}.console.log 2>@1} message]
        set f [open mode${mode}.console.log r];set text [read $f];close $f
        if {$mode==0} {
            if {$code || ![string match {*NTRPU_RX_LOG_PASS cq=8 rc=8 bytes=2304 callbacks=8 legacy_mixture_reproduced=1*} $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:)} $text]} {error "RX logger positive rejected"}
        } elseif {![string match "*Fatal: [lindex $expected $mode]*" $text] || [string match {*NTRPU_RX_LOG_PASS*} $text]} {
            error "Expected diagnostic absent in mode $mode"
        }
        puts "NTRPU_RX_LOG_REVIEW mode=$mode expected=[lindex $expected $mode] PASS"
    }
    puts NTRPU_RX_LOG_SUITE_PASS
}
if {[catch {rx_log $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
