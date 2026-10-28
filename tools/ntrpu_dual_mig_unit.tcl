# Positive cold-supervisor bank plus a deliberate late-latch negative control.
set root [file normalize [file join [file dirname [info script]] ..]]
proc dual_unit {root output} {
    if {[file exists $output]} {error "Expected new output directory"}
    file mkdir $output
    set_param general.maxThreads 2
    set f [open [file join $root fpga rtl ntrpu_dual_mig_cold_start.sv] r]
    set rtl [read $f];close $f
    set good {if (!mgmt_reset && clocks_alive && state == RELEASE && source_up_m &&
            (&memory_up_m) && (&memory_ready_m)) opened <= 1;}
    if {[string first $good $rtl]<0} {error "Missing exact atomic latch"}
    set bad [string map [list $good {if (running) opened <= 1;}] $rtl]
    set old [file join $output deliberate_late_latch.sv]
    set f [open $old w];puts -nonewline $f $bad;close $f
    foreach mode {positive negative} {
        create_project dual_unit_$mode [file join $output $mode] -part xcvu5p-flva2104-2-i
        set_property XPM_LIBRARIES {XPM_CDC} [current_project]
        add_files [file join $root fpga rtl ntrpu_reset_supervisor.sv]
        if {$mode eq "positive"} {add_files [file join $root fpga rtl ntrpu_dual_mig_cold_start.sv]} else {add_files $old}
        add_files -fileset sim_1 [file join $root fpga tb tb_ntrpu_dual_mig_cold_start.sv]
        set_property top tb_ntrpu_dual_mig_cold_start [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        launch_simulation
        catch {run all} simulation_message
        close_sim
        set log [file join $output $mode dual_unit_$mode.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output $mode.log]
        set f [open $log r];set text [read $f];close $f
        if {$mode eq "positive"} {
            if {[string first {DUAL_START_SUITE_PASS cases=13} $text]<0 || [regexp {Fatal:|ERROR:} $text]} {error "Positive supervisor bank failed"}
        } else {
            if {[string first {Fatal: DUAL_START_FAULT_NOT_CLOSED case=12} $text]<0 || [string first DUAL_START_SUITE_PASS $text]>=0} {error "Late-latch negative control not detected"}
        }
        close_project
    }
    puts NTRPU_DUAL_MIG_UNIT_EXPORTED
}
if {[llength $argv]!=1} {error "Expected new output directory"}
if {[catch {dual_unit $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
