# Complete layer arithmetic with eight original HLS cores; reuse exact predecessor synthesis.
set kv_script_dir [file dirname [file normalize [info script]]]
proc llama32_layer_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project llama32_layer_real [file join $output project] -part xcvu5p-flva2104-2-i
    foreach family {input_norm mac rope_input attention quantize post_attention swiglu residual} {
        set arithmetic_rtl [file join $output ${family}_rtl]
        file copy [file join $kv_script_dir .. generated $family] $arithmetic_rtl
        cd $arithmetic_rtl
        add_files [lsort [glob *.v]]
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    foreach name {fpga/rtl/ntrpu_llama32_layer_r1.sv fpga/rtl/ntrpu_llama32_ffn_residual_r3.sv fpga/rtl/ntrpu_ffn_residual_handoff_r1.sv fpga/rtl/ntrpu_residual_real_r1.sv fpga/rtl/ntrpu_residual_hls_ports_r1.sv fpga/rtl/ntrpu_residual_path_r1.sv fpga/rtl/ntrpu_residual_store_r1.sv fpga/rtl/ntrpu_residual_owner_r1.sv fpga/rtl/ntrpu_residual_loader_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv fpga/rtl/ntrpu_llama32_owned_ffn_r1.sv fpga/rtl/ntrpu_ffn_memory.sv fpga/rtl/ntrpu_swiglu_memory.sv fpga/rtl/ntrpu_activation_writer.sv fpga/rtl/ntrpu_q8_matrix_memory.sv fpga/rtl/ntrpu_q8_gemv_store.sv fpga/rtl/ntrpu_q8_gemv_memory.sv fpga/rtl/ntrpu_q8_gemv_parallel.sv fpga/rtl/ntrpu_q8_weight_stream.sv fpga/rtl/ntrpu_q8_unpacker.sv fpga/rtl/ntrpu_weight_tile_buffer.sv fpga/rtl/ntrpu_activation_loader.sv fpga/rtl/ntrpu_result_writer.sv fpga/rtl/ntrpu_fp64_to_fp32.sv fpga/rtl/ntrpu_matrix_sequencer.sv fpga/rtl/ntrpu_ffn_sequencer.sv fpga/rtl/ntrpu_llama32_ffn_owner_r3.sv fpga/rtl/ntrpu_post_ffn_handoff_r2.sv fpga/rtl/ntrpu_post_attention_real_r2.sv fpga/rtl/ntrpu_post_attention_hls_ports_r1.sv fpga/rtl/ntrpu_post_attention_path_r1.sv fpga/rtl/ntrpu_post_attention_store_r1.sv fpga/rtl/ntrpu_post_attention_owner_r1.sv fpga/rtl/ntrpu_post_attention_loader_r1.sv fpga/rtl/vivado_explicit_nets_r2/ntrpu_attention_output_path_r1.sv fpga/rtl/vivado_explicit_nets_r2/ntrpu_attention_projection_memory_r1.sv fpga/rtl/registered_watchdog_r3/ntrpu_attention_projection_control_r1.sv fpga/rtl/ntrpu_attention_quantize_store_r1.sv fpga/rtl/ntrpu_attention_activation_store_r1.sv fpga/rtl/ntrpu_attention_quantize_owner_r2.sv fpga/rtl/ntrpu_input_attention_r1.sv fpga/rtl/ntrpu_rope_attention_query_r1.sv fpga/rtl/ntrpu_attention_root_query_r1.sv fpga/rtl/ntrpu_attention_root_guard_query_r1.sv fpga/rtl/ntrpu_attention_head_sequencer_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_engine_r1.sv fpga/rtl/ntrpu_attention_hls_ports_prefetch_r3.sv fpga/rtl/ntrpu_attention_state_read_guard_r2.sv fpga/rtl/ntrpu_attention_kv_read_guard_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_memory_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_context_r2.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_path_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_ram_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_cache_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_tile_loader_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv fpga/rtl/ntrpu_rope_qkv_dispatch_r1.sv fpga/rtl/ntrpu_qkv_capture_dispatch_r1.sv fpga/rtl/ntrpu_qkv_dispatch_r1.sv fpga/rtl/ntrpu_qkv_snapshot_r1.sv fpga/rtl/ntrpu_rope_input_real_r1.sv fpga/rtl/ntrpu_rope_hls_ports_r1.sv fpga/rtl/ntrpu_rope_input_path_r2.sv fpga/rtl/ntrpu_rope_owner_r2.sv fpga/rtl/ntrpu_rope_loader_r1.sv fpga/rtl/ntrpu_qkv_rope_handoff_r1.sv fpga/rtl/ntrpu_input_qkv_real_r1.sv fpga/rtl/ntrpu_llama32_qkv_memory_r1.sv fpga/rtl/ntrpu_llama32_qkv_control_r2.sv fpga/rtl/ntrpu_input_norm_real_r1.sv fpga/rtl/ntrpu_input_norm_hls_ports_r1.sv fpga/rtl/ntrpu_input_norm_path_r1.sv fpga/rtl/ntrpu_input_norm_owner_r1.sv fpga/rtl/ntrpu_input_qkv_handoff_r2.sv fpga/rtl/ntrpu_llama32_layer_preflight_r1.sv} {
        add_files [file normalize [file join $kv_script_dir .. $name]]
    }
    set_property top ntrpu_llama32_layer_r1 [get_filesets sources_1]
    puts NTRPU_LLAMA32_LAYER_REAL_R1_REUSES_PREDECESSOR_SYNTHESIS
    add_files -fileset sim_1 [file normalize [file join $kv_script_dir .. fpga tb tb_ntrpu_llama32_layer_real_r2.sv]]
    set_property top ntrpu_llama32_layer_r1 [get_filesets sources_1]
    set_property top tb_ntrpu_llama32_layer_real_r2 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $kv_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    foreach {seed name} {1 seed-1} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed -testplusarg FIXTURE=[file normalize [file join $kv_script_dir .. fixture]]"] [get_filesets sim_1]
        update_compile_order -fileset sources_1
        update_compile_order -fileset sim_1
        report_ip_status -file [file join $output ip_status.rpt]
        launch_simulation
        run all
        close_sim
        set log [file join $output project llama32_layer_real.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "LREAL_PASS seed=$seed jobs=1 " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |LREAL_FAIL)} $text]} {error "Whole-layer arithmetic failed: $name"}
    }
    close_project
    puts NTRPU_LLAMA32_LAYER_REAL_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {llama32_layer_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
