# Full vocabulary arithmetic using the original generated dual-MAC datapath.
# Synthesis belongs to the already queued parent; this batch is behavioral RTL.
set vocab_script_dir [file dirname [file normalize [info script]]]
proc final_vocab_arithmetic_bank {output fixture norm_fixture} {
    global vocab_script_dir
    set output [file normalize $output]
    set fixture [file normalize $fixture]
    set norm_fixture [file normalize $norm_fixture]
    if {[file exists $output] || ![file isfile [file join $fixture manifest.json]]} {error "Output exists or fixture missing"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project final_vocab_arithmetic [file join $output project] -part xcvu5p-flva2104-2-i
    foreach family {input_norm mac} {
        set rtl [file join $output ${family}_rtl]
        file copy [file join $vocab_script_dir .. generated $family] $rtl
        cd $rtl
        add_files [lsort [glob *.v]]
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    foreach name {fpga/rtl/ntrpu_axi_read_master.sv fpga/rtl/ntrpu_residual_loader_r1.sv fpga/rtl/ntrpu_input_norm_owner_r1.sv fpga/rtl/ntrpu_activation_writer.sv fpga/rtl/ntrpu_attention_activation_store_r1.sv fpga/rtl/ntrpu_input_norm_path_r1.sv fpga/rtl/ntrpu_input_norm_hls_ports_r1.sv fpga/rtl/ntrpu_input_norm_real_r1.sv fpga/rtl/ntrpu_fp64_to_fp32.sv fpga/rtl/ntrpu_weight_tile_buffer.sv fpga/rtl/ntrpu_q8_unpacker.sv fpga/rtl/ntrpu_q8_weight_stream.sv fpga/rtl/ntrpu_q8_gemv_parallel.sv fpga/rtl/ntrpu_activation_loader.sv fpga/rtl/ntrpu_q8_gemv_memory.sv fpga/rtl/ntrpu_result_writer.sv fpga/rtl/ntrpu_q8_gemv_store.sv fpga/rtl/ntrpu_matrix_sequencer.sv fpga/rtl/ntrpu_q8_matrix_memory.sv fpga/rtl/ntrpu_llama32_vocab_control_r1.sv fpga/rtl/ntrpu_llama32_vocab_memory_r1.sv fpga/rtl/ntrpu_final_vocab_handoff_r1.sv fpga/rtl/ntrpu_final_vocab_real_r1.sv} {
        add_files [file normalize [file join $vocab_script_dir .. $name]]
    }
    add_files -fileset sim_1 [file normalize [file join $vocab_script_dir .. fpga tb tb_ntrpu_final_vocab_arithmetic_r1.sv]]
    set_property top ntrpu_final_vocab_real_r1 [get_filesets sources_1]
    set_property top tb_ntrpu_final_vocab_arithmetic_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $vocab_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    set_property xsim.simulate.xsim.more_options "-testplusarg SEED=1 -testplusarg FIXTURE=$fixture -testplusarg NORM_FIXTURE=$norm_fixture" [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    report_ip_status -file [file join $output ip_status.rpt]
    launch_simulation
    run all
    close_sim
    set sim [file join $output project final_vocab_arithmetic.sim sim_1 behav xsim]
    file copy [file join $sim simulate.log] [file join $output arithmetic.log]
    foreach name [concat {vocab-result.hex} [lmap phase {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15} {format "vocab-phase-%02d.hex" $phase}]] {
        if {![file isfile [file join $sim $name]] || [file size [file join $sim $name]] == 0} {error "Missing output $name"}
        file copy [file join $sim $name] [file join $output $name]
    }
    set f [open [file join $output arithmetic.log] r]
    set text [read $f]
    close $f
    if {[string first "FVREAL_PASS seed=1 jobs=3 publics=3 source_acks=2 cancels=1 matrices=17 groups=4009 results=128256 " $text] < 0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |FVREAL_FAIL)} $text]} {error "Full vocabulary arithmetic failed"}
    close_project
    puts NTRPU_FINAL_VOCAB_ARITHMETIC_R1_EXPORTED
}
if {[llength $argv] != 3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {final_vocab_arithmetic_bank {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
