# Four bounded AXI suites followed by OOC synthesis of the production watchdog.
set kv_script_dir [file dirname [file normalize [info script]]]
proc attention_kv_ram_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project attention_kv_ram [file join $output project] -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_attention_kv_ram_r1} {
        add_files [file normalize [file join $kv_script_dir .. fpga rtl ${name}.sv]]
    }
    add_files -fileset sim_1 [file normalize [file join $kv_script_dir .. fpga tb tb_ntrpu_attention_kv_ram_r1.sv]]
    set_property top ntrpu_attention_kv_ram_r1 [get_filesets sources_1]
    set_property top tb_ntrpu_attention_kv_ram_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    foreach {seed watchdog name} {1 32 seed-1 7 32 seed-7 42 32 seed-42 1 65536 seed-1-default} {
        set_property generic [list WATCHDOG_CYCLES=$watchdog] [get_filesets sim_1]
        set_property xsim.simulate.xsim.more_options "-testplusarg SEED=$seed" [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project attention_kv_ram.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "AKR_PASS seed=$seed watchdog=$watchdog jobs=81 resets=5 commands=81 tiles=1824 commits=69 computes=45 reads=12487 dones=77" $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "KV cache suite failed: $name"}
    }
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    synth_design -top ntrpu_attention_kv_ram_r1 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output attention_kv_ram_synth.dcp]
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
    puts NTRPU_ATTENTION_KV_RAM_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {attention_kv_ram_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
