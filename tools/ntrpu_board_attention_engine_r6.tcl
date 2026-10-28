# Actual generated attention, physical AXI fixture, then OOC synthesis.
set engine_script_dir [file dirname [file normalize [info script]]]
proc attention_engine_bank {output} {
    global engine_script_dir
    set root [file normalize [file join $engine_script_dir ..]]
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project attention_engine [file join $output project] -part xcvu5p-flva2104-2-i
    set rtl [file join $output attention_rtl]
    file copy [file join $root generated attention] $rtl
    cd $rtl
    add_files [lsort [glob *.v]]
    set data [glob -nocomplain *.dat]
    if {[llength $data]} {add_files $data}
    foreach script [lsort [glob *_ip.tcl]] {source $script}
    add_files [file join $root fpga rtl ntrpu_attention_hls_ports_prefetch_r3.sv]
    add_files [file join $root fpga rtl ntrpu_attention_state_read_guard_r2.sv]
    add_files [file join $root fpga rtl ntrpu_attention_kv_read_guard_r1.sv]
    foreach name {ntrpu_axi_read_master ntrpu_kv_tile_loader_r1 ntrpu_kv_append_r1 ntrpu_kv_cache_r1 ntrpu_attention_kv_ram_r1 ntrpu_attention_kv_path_r1 ntrpu_attention_context_r2 ntrpu_attention_memory_r1 ntrpu_attention_engine_r1} {
        add_files [file join $root fpga rtl vivado_explicit_nets_r1 ${name}.sv]
    }
    file copy [file join $root fixture] [file join $output fixture]
    add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_attention_engine_r4.sv]
    set_property top ntrpu_attention_engine_r1 [get_filesets sources_1]
    set_property top tb_ntrpu_attention_engine_r1 [get_filesets sim_1]
    set_property generic {WATCHDOG_CYCLES=65536} [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $engine_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=1 -testplusarg FIXTURE=[file join $output fixture]"] [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    report_ip_status -file [file join $output ip_status.rpt]
    puts NTRPU_ATTENTION_ENGINE_R1_SIM_START
    launch_simulation
    run all
    close_sim
    set log [file join $output project attention_engine.sim sim_1 behav xsim simulate.log]
    file copy $log [file join $output seed-1.log]
    set f [open $log r];set text [read $f];close $f
    if {![regexp {(?m)^AHE_PASS seed=1 watchdog=65536 cases=143 queries=155 completions=152 results=9600 resets=4 binds=19 appends=756 } $text] ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL:|AHE_FAIL)} $text]} {error "Generated attention integration did not pass"}
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    synth_design -top ntrpu_attention_engine_r1 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output attention_engine_synth.dcp]
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
    close_project
    puts NTRPU_ATTENTION_ENGINE_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {attention_engine_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
