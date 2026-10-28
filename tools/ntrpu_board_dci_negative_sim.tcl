# Verify the reset stress test rejects two deliberate regressions in copied RTL.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_dci_negative {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 1} { error "Expected NEW_DIRECTORY" }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    set root [file normalize [file join $ntrpu_script_dir ..]]
    set stream [open [file join $root fpga rtl ntrpu_dci_startup.sv] r]
    set original [read $stream]; close $stream
    foreach mode {synchronous_assert one_release_stage} {
        set changed $original
        if {$mode eq "synchronous_assert"} {
            set needle {always_ff @(posedge clk or posedge hold_reset)}
            set at [string first $needle $changed]
            if {$at < 0 || [string first $needle $changed [expr {$at + 1}]] >= 0} {
                error "Unexpected assertion source"
            }
            set changed [string replace $changed $at [expr {$at + [string length $needle] - 1}] {always_ff @(posedge clk)}]
            # Both independent monitors sample 1 ps after assertion. Whichever
            # runs first must report the same missing assertion, not a timeout.
            set expected {Fatal: NTRPU_DCI_STRESS (missed asynchronous fault|unexpected release)}
        } else {
            set count [regsub -all {reset_release\[1\]} $changed {reset_release[0]} changed]
            set expected {Fatal: NTRPU_DCI_STRESS early/asynchronous release}
            if {$count != 2} { error "Unexpected release source" }
        }
        set directory [file join $output $mode]; file mkdir $directory; cd $directory
        set stream [open mutant.sv {WRONLY CREAT EXCL}]; puts $stream $changed; close $stream
        exec xvlog --sv mutant.sv [file join $root fpga tb ntrpu_dci_reset_stress_tb.sv] > compile.log 2>@1
        exec xelab ntrpu_dci_reset_stress_tb -L unisims_ver --debug typical --mt 2 -s negative_tb > elaborate.log 2>@1
        set stream [open run.tcl {WRONLY CREAT EXCL}]; puts $stream {run all; quit}; close $stream
        set failed [catch {exec xsim negative_tb -tclbatch run.tcl -log simulate.log > simulate.console.log 2>@1} reason]
        set stream [open simulate.console.log r]; set text [read $stream]; close $stream
        if {![regexp $expected $text] || [string first NTRPU_DCI_STRESS_PASS $text] >= 0} {
            error "Regression was not rejected for the expected reason: $mode"
        }
        set stream [open expected_rejection.txt {WRONLY CREAT EXCL}]
        puts $stream "mutation=$mode\nexpected=$expected\nprocess_catch=$failed\nreason=$reason"
        close $stream
        puts "NTRPU_DCI_NEGATIVE_REJECTED mutation=$mode"
    }
    puts "NTRPU_DCI_NEGATIVE_PASS rejected=2; production RTL unchanged"
}
if {[catch {ntrpu_dci_negative $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
