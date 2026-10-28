# Original norm/Q8 and dual-MAC QKV connected through actual AXI ABI1 memory; 5 ns OOC.
set kv_script_dir [file dirname [file normalize [info script]]]
proc input_qkv_real_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project input_qkv_real [file join $output project] -part xcvu5p-flva2104-2-i
    foreach {generated_dir output_dir} {input_norm input_norm_rtl mac mac_rtl} {
        set arithmetic_rtl [file join $output $output_dir]
        file copy [file join $kv_script_dir .. generated $generated_dir] $arithmetic_rtl
        cd $arithmetic_rtl
        add_files [lsort [glob *.v]]
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    foreach name {fpga/rtl/ntrpu_axi_read_master.sv fpga/rtl/ntrpu_residual_loader_r1.sv fpga/rtl/ntrpu_input_norm_owner_r1.sv fpga/rtl/ntrpu_activation_writer.sv fpga/rtl/ntrpu_attention_activation_store_r1.sv fpga/rtl/ntrpu_input_norm_path_r1.sv fpga/rtl/ntrpu_input_norm_hls_ports_r1.sv fpga/rtl/ntrpu_input_norm_real_r1.sv fpga/rtl/ntrpu_fp64_to_fp32.sv fpga/rtl/ntrpu_weight_tile_buffer.sv fpga/rtl/ntrpu_q8_unpacker.sv fpga/rtl/ntrpu_q8_weight_stream.sv fpga/rtl/ntrpu_q8_gemv_parallel.sv fpga/rtl/ntrpu_activation_loader.sv fpga/rtl/ntrpu_q8_gemv_memory.sv fpga/rtl/ntrpu_result_writer.sv fpga/rtl/ntrpu_q8_gemv_store.sv fpga/rtl/ntrpu_matrix_sequencer.sv fpga/rtl/ntrpu_q8_matrix_memory.sv fpga/rtl/ntrpu_llama32_qkv_control_r2.sv fpga/rtl/ntrpu_llama32_qkv_memory_r1.sv fpga/rtl/ntrpu_input_qkv_handoff_r2.sv fpga/rtl/ntrpu_input_qkv_real_r1.sv} {
        add_files [file normalize [file join $kv_script_dir .. $name]]
    }
    set_property top ntrpu_input_qkv_real_r1 [get_filesets sources_1]
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    synth_design -top ntrpu_input_qkv_real_r1 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output input_qkv_real_synth.dcp]
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
    puts NTRPU_INPUT_QKV_REAL_R1_SYNTH_PREFLIGHT_EXPORTED
    close_design
    add_files -fileset sim_1 [file normalize [file join $kv_script_dir .. fpga tb tb_ntrpu_input_qkv_real_r2.sv]]
    set_property top ntrpu_input_qkv_real_r1 [get_filesets sources_1]
    set_property top tb_ntrpu_input_qkv_real_r2 [get_filesets sim_1]
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
        set log [file join $output project input_qkv_real.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "IQREAL_PASS seed=$seed jobs=12 publics=12 source_acks=11 cancels=3 matrices=21 groups=627 results=19968 " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |IQREAL_FAIL)} $text]} {error "Live norm/QKV arithmetic failed: $name"}
    }
    close_project
    puts NTRPU_INPUT_QKV_REAL_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {input_qkv_real_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
