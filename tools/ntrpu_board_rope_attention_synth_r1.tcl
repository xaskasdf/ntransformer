# Original RoPE HLS with actual AXI loader and owner; 5 ns OOC.
set kv_script_dir [file dirname [file normalize [info script]]]
proc rope_attention_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project rope_attention [file join $output project] -part xcvu5p-flva2104-2-i
    foreach family {rope_input attention} {
        set arithmetic_rtl [file join $output ${family}_rtl]
        file copy [file join $kv_script_dir .. generated $family] $arithmetic_rtl
        cd $arithmetic_rtl
        add_files [lsort [glob *.v]]
        foreach script [lsort [glob *_ip.tcl]] {source $script}
    }
    foreach name {fpga/rtl/ntrpu_rope_attention_r1.sv fpga/rtl/ntrpu_rope_qkv_dispatch_r1.sv fpga/rtl/ntrpu_rope_input_real_r1.sv fpga/rtl/ntrpu_rope_input_path_r2.sv fpga/rtl/ntrpu_rope_owner_r2.sv fpga/rtl/ntrpu_rope_loader_r1.sv fpga/rtl/ntrpu_rope_hls_ports_r1.sv fpga/rtl/ntrpu_qkv_capture_dispatch_r1.sv fpga/rtl/ntrpu_qkv_snapshot_r1.sv fpga/rtl/ntrpu_qkv_dispatch_r1.sv fpga/rtl/ntrpu_attention_root_r1.sv fpga/rtl/ntrpu_attention_root_guard_r1.sv fpga/rtl/ntrpu_attention_head_sequencer_r1.sv fpga/rtl/ntrpu_attention_hls_ports_prefetch_r3.sv fpga/rtl/ntrpu_attention_state_read_guard_r2.sv fpga/rtl/ntrpu_attention_kv_read_guard_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_axi_read_master.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_tile_loader_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_append_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_kv_cache_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_ram_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_kv_path_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_context_r2.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_memory_r1.sv fpga/rtl/vivado_explicit_nets_r1/ntrpu_attention_engine_r1.sv} {
        add_files [file normalize [file join $kv_script_dir .. $name]]
    }
    set_property top ntrpu_rope_attention_r1 [get_filesets sources_1]
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    update_compile_order -fileset sources_1
    set f [open [file join $output synthesis_inputs.txt] {WRONLY CREAT EXCL}]
    foreach item [get_files -compile_order sources -used_in synthesis] {puts $f [file normalize $item]}
    close $f
    synth_design -top ntrpu_rope_attention_r1 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output rope_attention_synth.dcp]
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
    puts NTRPU_ROPE_ATTENTION_SYNTH_R1_SYNTH_PREFLIGHT_EXPORTED
    close_design
    close_project
    puts NTRPU_ROPE_ATTENTION_SYNTH_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {rope_attention_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
