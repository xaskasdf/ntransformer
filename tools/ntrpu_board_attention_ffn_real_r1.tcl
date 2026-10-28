# Actual attention root -> quantize/store -> full O projection. Synthesis gate first.
set kv_script_dir [file dirname [file normalize [info script]]]
proc attention_ffn_real_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project attention_ffn_real [file join $output project] -part xcvu5p-flva2104-2-i
    foreach kind {attention quantize mac post_attention swiglu} {
        set rtl [file join $output ${kind}_rtl]
        if {$kind eq "swiglu"} {
            set original [file join $kv_script_dir .. inputs swiglu project solution1 syn verilog]
        } else {set original [file join $kv_script_dir .. generated $kind]}
        file copy $original $rtl
        cd $rtl
        add_files [lsort [glob *.v]]
        set data [glob -nocomplain *.dat]
        if {[llength $data]} {add_files $data}
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    set residual [glob [file normalize [file join $kv_script_dir .. generated *.v]]]
    if {[llength $residual]!=9} {error "Expected nine original residual HLS modules"}
    add_files $residual
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_tile_loader_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_cache_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_ram_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_path_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_context_r2.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_memory_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_engine_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_kv_read_guard_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_state_read_guard_r2.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_hls_ports_prefetch_r3.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_root_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_root_guard_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_head_sequencer_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_fp64_to_fp32.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_weight_tile_buffer.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_q8_unpacker.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_q8_weight_stream.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_q8_gemv_parallel.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_activation_loader.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_q8_gemv_memory.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_result_writer.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_q8_gemv_store.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_matrix_sequencer.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_q8_matrix_memory.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/registered_watchdog_r3/ntrpu_attention_projection_control_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r2/ntrpu_attention_projection_memory_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_activation_store_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_quantize_owner_r2.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_activation_writer.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_quantize_store_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/vivado_explicit_nets_r2/ntrpu_attention_output_path_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_projected_root_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_post_attention_loader_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_post_attention_owner_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_post_attention_store_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_post_attention_path_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_post_attention_hls_ports_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_post_attention_real_r2.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_attention_post_path_r2.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_swiglu_memory.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_ffn_sequencer.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_ffn_memory.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_llama32_ffn_owner_r3.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_llama32_owned_ffn_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_residual_loader_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_residual_owner_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_residual_store_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_residual_path_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_residual_real_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_residual_hls_ports_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_ffn_residual_handoff_r1.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_llama32_ffn_residual_r3.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_post_ffn_handoff_r2.sv]]
    add_files [file normalize [file join $kv_script_dir .. fpga/rtl/ntrpu_llama32_attention_ffn_r4.sv]]
    set_property verilog_define {NTRPU_FFN_FP32 NTRPU_FFN_FIXTURE NTRPU_FFN_LLAMA_REAL} [get_filesets sources_1]
    set_property top ntrpu_llama32_attention_ffn_r4 [get_filesets sources_1]
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    synth_design -top ntrpu_llama32_attention_ffn_r4 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output attention_ffn_real_synth.dcp]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    set f [open [file join $output primitive_counts.tsv] {WRONLY CREAT EXCL}]
    puts $f "top\t[get_property NAME [current_design]]"
    puts $f "part\t[get_property PART [current_design]]"
    puts $f "clock_period\t[get_property PERIOD [get_clocks clk]]"
    puts $f "blackboxes\t[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]"
    foreach ref {LUT1 LUT2 LUT3 LUT4 LUT5 LUT6 FDRE FDSE FDCE FDPE LDCE LDPE CARRY8 RAMB36E2 RAMB18E2 DSP48E2} {
        puts $f "$ref\t[llength [get_cells -quiet -hierarchical -filter "REF_NAME == $ref"]]"
    }
    close $f
    set f [open [file join $output check_timing.rpt] r];set checks [read $f];close $f
    if {![regexp {checking loops \(([0-9]+)\)} $checks all loops] || $loops!=0} {error "Combinational loop preflight failed"}
    puts NTRPU_ATTENTION_FFN_REAL_R1_SYNTH_PREFLIGHT_EXPORTED
    close_design
    foreach kind {fixture root-fixture post-fixture ffn-fixture extra} {
        file copy [file normalize [file join $kv_script_dir .. $kind]] [file join $output $kind]
    }
    set_property verilog_define {NTRPU_FFN_FP32 NTRPU_FFN_FIXTURE NTRPU_FFN_LLAMA_REAL} [get_filesets sim_1]
    add_files -fileset sim_1 [file normalize [file join $kv_script_dir .. fpga tb tb_ntrpu_llama32_attention_ffn_real_r5.sv]]
    set_property top ntrpu_llama32_attention_ffn_r4 [get_filesets sources_1]
    set_property top tb_ntrpu_llama32_attention_ffn_real_r5 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $kv_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    foreach {seed name} {1 seed-1} {
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed -testplusarg FIXTURE=[file join $output fixture] -testplusarg ROOTFIXTURE=[file join $output root-fixture] -testplusarg POSTFIXTURE=[file join $output post-fixture] -testplusarg FFNFIXTURE=[file join $output ffn-fixture] -testplusarg EXTRA=[file join $output extra]"] [get_filesets sim_1]
        update_compile_order -fileset sources_1
        update_compile_order -fileset sim_1
        report_ip_status -file [file join $output ip_status.rpt]
        launch_simulation
        run all
        close_sim
        set log [file join $output project attention_ffn_real.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {![regexp {(?m)^AFREAL_PASS jobs=1 seed=1 .* bytes=163840 } $text] ||
            [string first "AFREAL_CONFIG dim=2048 hidden=8192 sample=1 token=0 attention_position=0 full_weight_bytes=57933824 input_abi_requantized=0" $text]<0 ||
            [string first "AFREAL_REUSABLE jobs=1" $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |AFREAL_FAIL|PRREAL_FAIL|APREAL_FAIL)} $text]} {error "Actual attention/FFN/residual arithmetic did not pass"}
    }
    close_project
    puts NTRPU_ATTENTION_FFN_REAL_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {attention_ffn_real_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
