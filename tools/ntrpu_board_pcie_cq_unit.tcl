set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_cq_unit {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] ni {1 2}} { error "Expected NEW_DIRECTORY ?host64?" }
    set wide [expr {[llength $arguments] == 2}]
    if {$wide && [lindex $arguments 1] ne "host64"} { error "Unknown CQ unit profile" }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    cd $output
    set tb [file normalize [file join $ntrpu_script_dir .. fpga tb]]
    exec xvlog --sv [file join $tb ntrpu_pcie_cq_check.sv] [file join $tb ntrpu_pcie_cq_check_tb.sv] > compile.log 2>@1
    set top [expr {$wide ? "ntrpu_pcie_cq_host64_tb" : "ntrpu_pcie_cq_check_tb"}]
    exec xelab $top --debug typical --mt 2 -s cq_tb > elaborate.log 2>@1
    set stream [open run.tcl {WRONLY CREAT EXCL}]; puts $stream {run all; quit}; close $stream
    set expected {PASS CQ_DATA_MISMATCH CQ_C2H_RANGE CQ_BAD_LAST CQ_BAD_BYTE_ENABLE CQ_UNKNOWN_CONTROL CQ_STALLED_CHANGED CQ_INCOMPLETE}
    if {$wide} { lappend expected CQ_PAGE_OR_SIZE CQ_C2H_RANGE }
    for {set mode 0} {$mode < [llength $expected]} {incr mode} {
        set code [catch {exec xsim cq_tb -tclbatch run.tcl -testplusarg MODE=$mode -log mode${mode}.log > mode${mode}.console.log 2>@1} message]
        set stream [open mode${mode}.console.log r]; set text [read $stream]; close $stream
        if {$mode == 0} {
            if {$code || ![string match {*NTRPU_CQ_UNIT_PASS mode=0*} $text] ||
                [regexp -nocase {\m(error|fatal)\M} $text]} { error "CQ positive test rejected" }
        } else {
            if {![string match "*Fatal: [lindex $expected $mode]*" $text] ||
                [string match {*NTRPU_CQ_UNIT_PASS*} $text]} { error "Negative mode $mode did not reject as expected" }
        }
        puts "NTRPU_CQ_UNIT_REVIEW mode=$mode expected=[lindex $expected $mode] PASS"
    }
    puts "NTRPU_CQ_UNIT_SUITE_PASS positive=1 negative=[expr {[llength $expected]-1}] host64=$wide"
}
if {[catch {ntrpu_cq_unit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
