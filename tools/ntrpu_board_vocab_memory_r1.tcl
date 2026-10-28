# Full vocabulary owner and original dual-MAC FP32 matrix engine; separate protocol bank, 5 ns OOC.
set kv_script_dir [file dirname [file normalize [info script]]]
proc vocab_memory_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project vocab_memory [file join $output project] -part xcvu5p-flva2104-2-i
    foreach family {mac} {
        set arithmetic_rtl [file join $output ${family}_rtl]
        file copy [file join $kv_script_dir .. generated $family] $arithmetic_rtl
        cd $arithmetic_rtl
        add_files [lsort [glob *.v]]
        foreach script [lsort [glob -nocomplain *_ip.tcl]] {source $script}
    }
    foreach name {fpga/rtl/ntrpu_fp64_to_fp32.sv fpga/rtl/ntrpu_axi_read_master.sv fpga/rtl/ntrpu_weight_tile_buffer.sv fpga/rtl/ntrpu_q8_unpacker.sv fpga/rtl/ntrpu_q8_weight_stream.sv fpga/rtl/ntrpu_q8_gemv_parallel.sv fpga/rtl/ntrpu_activation_loader.sv fpga/rtl/ntrpu_q8_gemv_memory.sv fpga/rtl/ntrpu_result_writer.sv fpga/rtl/ntrpu_q8_gemv_store.sv fpga/rtl/ntrpu_matrix_sequencer.sv fpga/rtl/ntrpu_q8_matrix_memory.sv fpga/rtl/ntrpu_llama32_vocab_control_r1.sv fpga/rtl/ntrpu_llama32_vocab_memory_r1.sv} {
        add_files [file normalize [file join $kv_script_dir .. $name]]
    }
    set_property top ntrpu_llama32_vocab_memory_r1 [get_filesets sources_1]
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    update_compile_order -fileset sources_1
    set f [open [file join $output synthesis_inputs.txt] {WRONLY CREAT EXCL}]
    foreach item [get_files -compile_order sources -used_in synthesis] {puts $f [file normalize $item]}
    close $f
    synth_design -top ntrpu_llama32_vocab_memory_r1 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output vocab_memory_synth.dcp]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_timing -max_paths 50 -file [file join $output critical_paths.rpt]
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
    puts NTRPU_VOCAB_MEMORY_R1_SYNTH_PREFLIGHT_EXPORTED
    set f [open [file join $output arithmetic_cells.tsv] {WRONLY CREAT EXCL}]
    foreach cell [get_cells -hierarchical -filter {IS_PRIMITIVE == 0}] {
        puts $f "$cell\t[get_property REF_NAME $cell]"
    }
    close $f
    close_design
    close_project
    create_project vocab_memory_protocol [file join $output protocol_project] -part xcvu5p-flva2104-2-i
    foreach name {fpga/rtl/ntrpu_llama32_vocab_control_r1.sv fpga/tb/tb_ntrpu_llama32_vocab_control_r2.sv} {add_files [file normalize [file join $kv_script_dir .. $name]]}
    set_property top tb_ntrpu_llama32_vocab_control_r2 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $kv_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    foreach {seed mode tag name} {
        1 0 64'h00000000ffffff00 seed-1
        7 0 64'h00000000ffffff00 seed-7
        42 0 64'h00000000ffffff00 seed-42
        1 1 64'hfffffffffffff000 tag-last
        1 2 64'h0000000000000001 tag-unaligned
        1 2 64'hfffffffffffff100 tag-insufficient
    } {
        set_property generic [list INITIAL_TAG=$tag TAG_TEST=$mode] [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed"] [get_filesets sim_1]
        update_compile_order -fileset sources_1
        update_compile_order -fileset sim_1
        launch_simulation
        run all
        close_sim
        set log [file join $output protocol_project vocab_memory_protocol.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "VOCAB_PASS seed=$seed tag_test=$mode " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |VOCAB_FAIL)} $text]} {error "Whole owner protocol failed: $name"}
    }
    close_project
    puts NTRPU_VOCAB_MEMORY_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {vocab_memory_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
