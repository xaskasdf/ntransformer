# Unit simulation uses the installed AMD DCIRESET primitive, not a substitute.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_dci_sim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] ni {1 2}} { error "Expected NEW_DIRECTORY ?stress?" }
    set stress [expr {[llength $arguments] == 2}]
    if {$stress && [lindex $arguments 1] ne "stress"} { error "Unknown test mode" }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    set root [file normalize [file join $ntrpu_script_dir ..]]
    set rtl [file join $root fpga rtl ntrpu_dci_startup.sv]
    set top [expr {$stress ? "ntrpu_dci_reset_stress_tb" : "ntrpu_dci_startup_tb"}]
    set tb [file join $root fpga tb ${top}.sv]
    cd $output
    exec xvlog --sv $rtl $tb > compile.log 2>@1
    exec xelab $top -L unisims_ver --debug typical --mt 2 -s dci_tb > elaborate.log 2>@1
    set stream [open run.tcl {WRONLY CREAT EXCL}]
    puts $stream {run all; quit}
    close $stream
    exec xsim dci_tb -tclbatch run.tcl -log simulate.log > simulate.console.log 2>@1
    set stream [open simulate.console.log r]
    set text [read $stream]
    close $stream
    set pass {NTRPU_DCI_PASS pulses=1 releases=4 time_ps=[0-9]+}
    if {$stress} { set pass {NTRPU_DCI_STRESS_PASS cases=588 clock_stops=3 pulses=1 releases=591 time_ps=[0-9]+} }
    if {![regexp $pass $text] ||
        [regexp -nocase {\m(error|fatal|violation|incomplete)\M} $text]} {
        error "DCI simulation rejected; inspect vendor diagnostics and comparisons"
    }
    puts "NTRPU_DCI_SIM_REVIEW_PASS"
}
if {[catch {ntrpu_dci_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
