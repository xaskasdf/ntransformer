# V7 independent compute budgeting for the provisional VU5P.
# No DDR/PCIe connection or new cosimulation claim.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_ffn_synth {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected VITIS_RUN NEW_DIRECTORY"
    }
    lassign $arguments vitis_run output
    if {![file executable $vitis_run]} { error "Missing executable Vitis runner" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    set cpus [exec python3 -c {import os; print(','.join(map(str, sorted(os.sched_getaffinity(0))[:4])))}]
    set stream [open [file join $output cpu_affinity.txt] {WRONLY CREAT EXCL}]
    puts $stream $cpus
    close $stream
    # taskset is inherited by Vitis/HLS and bounds runnable CPU use to four.
    exec taskset -c $cpus python3 [file join $ntrpu_script_dir .. scripts run_ntrpu_hls.py] \
        --kernel ffn_q8_0 --part xcvu5p-flva2104-2-i --clock-ns 5 --stage synth \
        --tool $vitis_run --output-root [file join $output hls] > [file join $output runner.log] 2>@1
    puts "NTRPU_FFN_VU5P_SYNTH_RETURNED; review C simulation, synthesis report and interface widths; not routed or integrated"
}
if {[catch {ntrpu_ffn_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
