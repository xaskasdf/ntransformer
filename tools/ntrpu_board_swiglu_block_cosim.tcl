# V7 SwiGLU/Q8 activation block validation on the provisional VU5P. New isolated HLS project.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_swiglu_block_cosim {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected VITIS_RUN NEW_DIRECTORY"
    }
    lassign $arguments vitis_run output
    if {![file executable $vitis_run]} { error "Missing executable Vitis runner" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    # Prefer a separate four-CPU set when another HLS regression is running.
    set cpus [exec python3 -c {import os; c=sorted(os.sched_getaffinity(0)); print(','.join(map(str,c[4:8] if len(c)>=8 else c[:4])))}]
    set stream [open [file join $output cpu_affinity.txt] {WRONLY CREAT EXCL}]
    puts $stream $cpus
    close $stream
    exec taskset -c $cpus python3 [file join $ntrpu_script_dir .. scripts run_ntrpu_hls.py] \
        --kernel swiglu_q8_block --part xcvu5p-flva2104-2-i --clock-ns 5 --stage cosim \
        --tool $vitis_run --output-root [file join $output hls] > [file join $output runner.log] 2>@1
    puts "NTRPU_SWIGLU_BLOCK_VU5P_COSIM_RETURNED; review C/RTL transaction counts and numerical results"
}
if {[catch {ntrpu_swiglu_block_cosim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
