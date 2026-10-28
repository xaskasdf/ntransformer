# Full Q8 layer0 FFN; separate bench, unchanged device RTL and HLS cores.
set script_dir [file dirname [file normalize [info script]]]
proc llama32_full_ffn {arguments} {
    global script_dir
    if {[llength $arguments]!=4 || [llength [get_projects -quiet]]} {error "Expected MAC_HLS SWIGLU_HLS OUTPUT FIXTURE"}
    lassign $arguments mac swiglu output fixture
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project llama32_full_ffn [file join $output project] -part xcvu5p-flva2104-2-i
    foreach input [list $mac $swiglu] kernel {q8_block_mac swiglu_q8_block} {
        set input [file normalize $input]
        set f [open [file join $input manifest.json] r];set text [read $f];close $f
        foreach pattern [list {*"result": "tool_succeeded"*} "*\"kernel\": \"$kernel\"*" {*"stage": "cosim"*} {*"part": "xcvu5p-flva2104-2-i"*}] {
            if {![string match $pattern $text]} {error "Pinned HLS manifest mismatch"}
        }
        set rtl [file join $output ${kernel}_rtl]
        file copy [file join $input project solution1 syn verilog] $rtl
        file copy [file join $input manifest.json] [file join $output ${kernel}_manifest.json]
        cd $rtl
        add_files [glob *.v]
        set data [glob -nocomplain *.dat]
        if {[llength $data]} {add_files $data}
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    foreach name {ntrpu_fp64_to_fp32 ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream ntrpu_q8_gemv_parallel ntrpu_activation_loader ntrpu_q8_gemv_memory ntrpu_result_writer ntrpu_q8_gemv_store ntrpu_matrix_sequencer ntrpu_q8_matrix_memory ntrpu_activation_writer ntrpu_swiglu_memory ntrpu_ffn_sequencer ntrpu_ffn_memory} {
        add_files [file normalize [file join $script_dir .. fpga rtl ${name}.sv]]
    }
    file copy [file normalize $fixture] [file join $output fixture]
    add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_llama32_full_ffn_r2.sv]]
    set_property top ntrpu_ffn_memory [get_filesets sources_1]
    set_property top tb_ntrpu_ffn_memory [get_filesets sim_1]
    set_property verilog_define {NTRPU_FFN_FP32 NTRPU_FFN_FIXTURE NTRPU_FFN_LLAMA_REAL} [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=1 -testplusarg FIXTURE=[file join $output fixture]"] [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    report_ip_status -file [file join $output ip_status.rpt]
    puts NTRPU_LLAMA32_FULL_FFN_R1_START
    launch_simulation
    run all
    close_sim
    set logfile [file join $output project llama32_full_ffn.sim sim_1 behav xsim simulate.log]
    file copy $logfile [file join $output seed-1.log]
    set f [open $logfile r];set text [read $f];close $f
    if {![regexp {(?m)^NTRPU_FFN_MEMORY_PASS cases=1 .* bytes=155648 resets=0 cancels=0 } $text] ||
        ![regexp {(?m)^NTRPU_LLAMA32_REAL_WEIGHTS layer=0 source_sample=2 bytes=53477376$} $text] ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Full real-weight FFN did not pass"}
    close_project
    puts NTRPU_LLAMA32_FULL_FFN_R1_EXPORTED
}
if {[catch {llama32_full_ffn $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
