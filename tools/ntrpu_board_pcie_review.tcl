# Review a terminal simulation's immutable console without rerunning the IP.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_pcie_review {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2} { error "Expected INPUT_CONSOLE NEW_DIRECTORY" }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $input [file join $output input.console.log]
    set f [open [file join $output origin.txt] {WRONLY CREAT EXCL}]
    puts $f $input
    puts $f "Read-only post-simulation review; no change to HDL, logs or archived results."
    close $f
    exec python3 [file join $ntrpu_script_dir check_ntrpu_pcie_multi.py] \
        [file join $output input.console.log] --json [file join $output review.json] \
        > [file join $output review.console.log] 2>@1
    puts "NTRPU_PCIE_POST_REVIEW_PASS"
}
if {[catch {ntrpu_pcie_review $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
