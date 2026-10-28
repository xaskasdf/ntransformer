# Connect the immutable, verified HLS MAC RTL to the AXI/Q8 transport and job FSM.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_gemv_stream_sim {arguments} {
    global ntrpu_script_dir
    set_param general.maxThreads 2
    if {[llength $arguments] ni {2 3 4 5} || [llength [get_projects -quiet]]} { error "Expected HLS_MAC_RUN NEW_DIRECTORY ?FIXTURE_DIRECTORY|-? ?scalar|dual? ?default|max|packed?" }
    lassign $arguments input output fixture variant workload
    if {$fixture eq "-"} { set fixture {} }
    if {$variant eq ""} { set variant scalar }
    if {$variant ni {scalar dual}} { error "Unknown GEMV variant" }
    if {$workload eq ""} { set workload default }
    if {$workload ni {default max packed} || ($workload eq "max" && $fixture ne "") ||
        ($workload eq "packed" && ($fixture eq "" || $variant ne "dual"))} { error "Invalid workload" }
    if {$workload ne "packed" || $variant ne "dual" || $fixture eq ""} {error "Llama requires packed dual fixture"}
    set top [expr {$variant eq "dual" ? "ntrpu_q8_gemv_parallel" : "ntrpu_q8_gemv_stream"}]
    set input [file normalize $input]; set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    set stream [open [file join $input manifest.json] r]; set manifest [read $stream]; close $stream
    foreach pattern {{*"result": "tool_succeeded"*} {*"kernel": "q8_block_mac"*} {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}} {
        if {![string match $pattern $manifest]} { error "MAC manifest mismatch: $pattern" }
    }
    file mkdir $output
    set stream [open [file join $output variant.txt] {WRONLY CREAT EXCL}]
    puts $stream $variant
    close $stream
    set cases [expr {$variant eq "dual" ? 29 : 24}]
    set marker NTRPU_GEMV_STREAM_PASS
    set extra {}
    if {$workload eq "max"} {
        set cases 4
        set marker NTRPU_GEMV_MAX_PASS
        set extra " -testplusarg MAX_SHAPES"
    }
    if {$fixture ne ""} {
        if {![file isfile [file join $fixture manifest.json]]} { error "Missing fixture manifest" }
        file copy [file normalize $fixture] [file join $output fixture]
        set cases 54
        set marker NTRPU_GEMV_FIXTURE_PASS
        set extra " -testplusarg FIXTURE=[file join $output fixture]"
        if {$workload eq "packed"} { set cases 28 }
    }
    file copy [file join $input project solution1 syn verilog] [file join $output mac_rtl]
    file copy [file join $input manifest.json] [file join $output input_hls_manifest.json]
    file copy [file join $input project solution1 sim report ntrpu_q8_block_mac_cosim.rpt] [file join $output input_cosim.rpt]
    create_project ntrpu_gemv_stream [file join $output project] -part xcvu5p-flva2104-2-i
    cd [file join $output mac_rtl]
    add_files [glob *.v]
    foreach script [lsort [glob *_ip.tcl]] { source $script }
    foreach name [list ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream $top] {
        add_files [file normalize [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]]
    }
    add_files -fileset sim_1 [file normalize [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_llama32_q8_gemv_r1.sv]]
    set_property top $top [get_filesets sources_1]
    set_property top tb_ntrpu_q8_gemv_stream [get_filesets sim_1]
    set defines {}
    if {$variant eq "dual"} { lappend defines NTRPU_DUAL }
    if {$workload eq "packed"} { lappend defines NTRPU_PACKED }
    if {[llength $defines]} { set_property verilog_define $defines [get_filesets sim_1] }
    set_property xsim.elaborate.xelab.more_options {} [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $ntrpu_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    report_ip_status -file [file join $output ip_status.rpt]
    foreach seed {1 7 42} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed$extra"] [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project ntrpu_gemv_stream.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output seed-${seed}.log]
        set stream [open $log r]; set text [read $stream]; close $stream
        if {![string match "*$marker cases=$cases *" $text] ||
            ![string match "*NTRPU_GEMV_SEED seed=$seed\n*" $text] ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {
            error "GEMV numerical/protocol validation failed for seed $seed"
        }
        puts "NTRPU_GEMV_STREAM_SEED_PASS seed=$seed cases=$cases"
    }
    puts "NTRPU_GEMV_STREAM_SUITE_RETURNED; review [expr {3*$cases}] cases and diagnostics"
    close_project
    puts NTRPU_LLAMA32_Q8_GEMV_R1_EXPORTED
}
if {[catch {ntrpu_gemv_stream_sim $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
