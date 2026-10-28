# Pipeline scheduling experiment in copied sources; baseline HLS is untouched.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_swiglu_pipeline {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=3 || [llength [get_projects -quiet]]} {
        error "Expected VITIS_RUN activation|quantize|both NEW_DIRECTORY"
    }
    lassign $arguments vitis_run variant output
    if {![file executable $vitis_run] || $variant ni {activation quantize both}} {error "Invalid runner/variant"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set cpus [exec python3 -c {import os; c=sorted(os.sched_getaffinity(0)); print(','.join(map(str,c[8:12] if len(c)>=12 else c[:4])))}]
    set f [open [file join $output cpu_affinity.txt] {WRONLY CREAT EXCL}]
    puts $f $cpus;close $f
    exec taskset -c $cpus python3 [file join $ntrpu_script_dir run_ntrpu_swiglu_pipeline.py] \
        --variant $variant --tool $vitis_run --output [file join $output experiment] \
        > [file join $output runner.log] 2>@1
    puts "NTRPU_SWIGLU_PIPELINE_RETURNED; numerical equivalence and schedule still require review"
}
if {[catch {ntrpu_swiglu_pipeline $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
