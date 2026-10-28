# Isolated XPM-based reset coordinator, not yet wired to XDMA or DDR.
set script_dir [file dirname [file normalize [info script]]]
proc supervisor {arguments} {
    global script_dir
    if {[llength $arguments] != 2} {error "Expected sim|synth NEW_DIRECTORY"}
    lassign $arguments mode output
    if {$mode ni {sim synth}} {error "Invalid mode"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    create_project supervisor [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    add_files [file join $script_dir .. fpga rtl ntrpu_reset_supervisor.sv]
    set_property top ntrpu_reset_supervisor [get_filesets sources_1]
    if {$mode eq "sim"} {
        add_files -fileset sim_1 [file join $script_dir .. fpga tb tb_ntrpu_reset_supervisor.sv]
        set_property top tb_ntrpu_reset_supervisor [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach profile {0 1 2} {
            set_property generic PROFILE=$profile [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project supervisor.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output profile-$profile.log]
            set f [open $log r];set text [read $f];close $f
            if {![string match *NTRPU_SUP_PASS* $text] || [regexp -nocase {\m(error|fatal|violation)\M} $text]} {
                error "Supervisor bank rejected; inspect complete simulation log"
            }
        }
        puts NTRPU_SUPERVISOR_SIM_EXPORTED
    } else {
        synth_design -top ntrpu_reset_supervisor -part xcvu5p-flva2104-2-i -mode out_of_context
        create_clock -period 10.0 -name mgmt_clk [get_ports mgmt_clk]
        create_clock -period 4.0 -name host_clk [get_ports host_clk]
        create_clock -period 5.0 -name device_clk [get_ports device_clk]
        # Only asynchronous clear pins in the four recognized 3-stage reset
        # release chains. No data paths or clock pairs are globally excluded.
        foreach chain {host_reset_cdc device_reset_cdc host_admit_cdc device_admit_cdc} {
            set cells [get_cells -quiet -hierarchical -filter "NAME =~ $chain/release_ff_reg*"]
            set pins [get_pins -of_objects $cells -filter {REF_PIN_NAME == CLR}]
            if {[llength $cells] != 3 || [llength $pins] != 3} {error "Unexpected release chain: $chain"}
            set_false_path -to $pins
        }
        write_checkpoint [file join $output supervisor_synth.dcp]
        write_xdc -constraints all [file join $output applied.xdc]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_cdc -details -file [file join $output cdc.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        report_exceptions -coverage -file [file join $output exceptions.rpt]
        puts NTRPU_SUPERVISOR_SYNTH_EXPORTED
    }
    close_project
}
if {[catch {supervisor $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
