# Replay a completed XSIM elaboration in a new directory, without recompiling
# or changing the source run. Tool version must match the original snapshot.
# Usage: -tclargs INPUT_XSIM_DIRECTORY NEW_DIRECTORY
proc ntrpu_pcie_replay {arguments} {
    if {[llength $arguments] != 2} { error "Expected INPUT_XSIM_DIRECTORY NEW_DIRECTORY" }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    if {![file exists [file join $input xsim.dir board_behav xsimk]]} {
        error "Missing completed board_behav simulation snapshot"
    }
    file mkdir $output
    file copy [file join $input xsim.dir] [file join $output xsim.dir]
    set stream [open [file join $output origin.txt] {WRONLY CREAT EXCL}]
    puts $stream $input
    puts $stream "Reuses compiled vendor model and independent monitor; no HDL changes."
    close $stream
    set stream [open [file join $output replay.tcl] {WRONLY CREAT EXCL}]
    puts $stream {set previous_time [current_time -s]}
    puts $stream {for {set step 0} {$step < 2000} {incr step} {
        run 1 us
        set now [current_time -s]
        puts "NTRPU_PCIE_PROGRESS time=$now ep_link=[get_value -radix bin /board/EP/user_lnk_up] rp_link=[get_value -radix bin /board/RP/user_lnk_up] rp_ltssm=[get_value -radix hex /board/RP/cfg_ltssm_state]"
        flush stdout
        if {$now eq $previous_time} { break }
        set previous_time $now
    }}
    puts $stream {quit}
    close $stream
    cd $output
    exec xsim board_behav -tclbatch replay.tcl -log simulate.log -testplusarg TESTNAME=dma_test0 > replay.console.log 2>@1
    puts NTRPU_PCIE_REPLAY_RETURNED
    puts "Inspect simulate.log and numerical/status checks; process success alone is not PASS."
}
if {[catch {ntrpu_pcie_replay $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
