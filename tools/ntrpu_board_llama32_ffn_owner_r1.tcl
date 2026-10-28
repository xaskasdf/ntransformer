# Public FFN identity / retained residual owner, controlled FFN peer; 5 ns OOC.
set kv_script_dir [file dirname [file normalize [info script]]]
proc llama32_ffn_owner_bank {output} {
    global kv_script_dir
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project llama32_ffn_owner [file join $output project] -part xcvu5p-flva2104-2-i
    add_files [file normalize [file join $kv_script_dir .. fpga rtl ntrpu_llama32_ffn_owner_r3.sv]]
    set_property top ntrpu_llama32_ffn_owner_r3 [get_filesets sources_1]
    set f [open [file join $output clock.xdc] {WRONLY CREAT EXCL}]
    puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
    read_xdc [file join $output clock.xdc]
    synth_design -top ntrpu_llama32_ffn_owner_r3 -mode out_of_context -part xcvu5p-flva2104-2-i
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
    write_checkpoint [file join $output llama32_ffn_owner_synth.dcp]
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
    puts NTRPU_LLAMA32_FFN_OWNER_R1_SYNTH_PREFLIGHT_EXPORTED
    close_design
    add_files -fileset sim_1 [file normalize [file join $kv_script_dir .. fpga tb tb_ntrpu_llama32_ffn_owner_r8.sv]]
    set_property top ntrpu_llama32_ffn_owner_r3 [get_filesets sources_1]
    set_property top tb_ntrpu_llama32_ffn_owner_r8 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $kv_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    foreach {config seed name jobs completed} {0 1 seed-1 52 48 0 7 seed-7 52 48 0 42 seed-42 52 48 1 1 exhaustion 3 3 2 1 misaligned 2 2} {
        set_property generic [list CONFIG=$config] [get_filesets sim_1]
        set_property -dict [list xsim.simulate.xsim.more_options "-testplusarg SEED=$seed"] [get_filesets sim_1]
        update_compile_order -fileset sources_1
        update_compile_order -fileset sim_1
        report_ip_status -file [file join $output ip_status.rpt]
        launch_simulation
        run all
        close_sim
        set log [file join $output project llama32_ffn_owner.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output ${name}.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "FOWNER_PASS config=$config seed=$seed jobs=$jobs completed=$completed " $text]<0 ||
            [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |FOWNER_FAIL)} $text]} {error "FFN owner failed: $name"}
    }
    close_project
    puts NTRPU_LLAMA32_FFN_OWNER_R1_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {llama32_ffn_owner_bank [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
