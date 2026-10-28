# V7 target-specific FFN C/RTL regression. Isolated synthetic memory only.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ffn_cosim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected VITIS_RUN NEW_DIRECTORY smoke|full"
    }
    lassign $arguments vitis_run output test_mode
    if {$test_mode ni {smoke full}} { error "Unsupported test mode" }
    if {![file executable $vitis_run]} { error "Missing executable Vitis runner" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    set cpus [exec python3 -c {import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:4])))}]
    set stream [open [file join $output cpu_affinity.txt] {WRONLY CREAT EXCL}]
    puts $stream $cpus
    close $stream
    exec taskset -c $cpus python3 [file join $ntrpu_script_dir .. scripts run_ntrpu_hls.py] \
        --kernel ffn_q8_0 --part xcvu5p-flva2104-2-i --clock-ns 5 --stage cosim \
        --test-mode $test_mode --tool $vitis_run --output-root [file join $output hls] \
        > [file join $output runner.log] 2>@1
    puts "NTRPU_FFN_VU5P_COSIM_RETURNED mode=$test_mode; review numerical results and RTL transaction report"
}
if {[catch {ntrpu_ffn_cosim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
