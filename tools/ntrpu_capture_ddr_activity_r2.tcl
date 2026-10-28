# R2: passive four-rank VCD brackets the unchanged two SAIF windows.
source [file join [file dirname [file normalize [info script]]] ntrpu_ddr_rank_trace_r1.tcl]
# Source inside a fresh XSim session, loaded at time zero with the companion TB.
# Diagnostic capture only. Native execution and independent review are pending.
proc ntrpu_activity_value {name {radix unsigned}} {
    set objects [get_objects /ntrpu_ddr_example_activity_tb/$name]
    if {[llength $objects] != 1} { error "Missing/ambiguous monitor: $name" }
    set value [get_value -radix $radix $objects]
    if {$radix eq "bin"} {
        if {$value ni {0 1}} { error "Unknown monitor: $name=$value" }
    } elseif {![string is entier -strict $value]} {
        error "Non-integer monitor: $name=$value"
    }
    return $value
}
proc ntrpu_activity_snapshot {} {
    set values [dict create]
    foreach name {cycles sample_ps aw_count b_count ar_count rlast_count w_beats r_beats w_bytes w_stalls r_stalls} {
        dict set values $name [ntrpu_activity_value $name]
    }
    foreach name {pending_w pending_r} {
        dict set values $name [ntrpu_activity_value $name dec]
    }
    foreach name {capture_ready healthy prbs_running} {
        if {[ntrpu_activity_value $name bin] != 1} { error "Inactive $name" }
    }
    return $values
}
proc ntrpu_capture_activity {} {
    set out [file normalize [file join [pwd] activity_capture]]
    if {[file exists $out]} { error "Capture output already exists" }
    file mkdir $out
    ntrpu_rank_trace_open [file join $out rank_commands.vcd] [file join $out rank_pins.tsv]
    set started 0
    # Bounded calibration/boot wait; no resets, internal forces or TG overrides.
    for {set step 0} {$step < 1000} {incr step} {
        if {[ntrpu_activity_value capture_ready bin] == 1} { set started 1; break }
        run 1000 ns
    }
    if {!$started} { error "No stable calibrated PRBS within 1 ms simulated" }
    current_scope /ntrpu_ddr_example_activity_tb/dut
    set objects [get_objects -r *]
    if {![llength $objects]} { error "No objects under archived DUT" }
    set output [open [file join $out windows.tsv] {WRONLY CREAT EXCL}]
    puts $output "window\tedge\tmetric\tvalue"
    puts $output "all\tscope\tobjects_selected\t[llength $objects]"
    set first [ntrpu_activity_snapshot]
    set last $first
    for {set index 0} {$index < 2} {incr index} {
        set name prbs_$index
        set before [ntrpu_activity_snapshot]
        set sim_start [current_time -s]
        open_saif [file join $out $name.saif]
        log_saif $objects
        run 20480 ns
        close_saif
        set after [ntrpu_activity_snapshot]
        set sim_end [current_time -s]
        puts $output "$name\tstart\tsimulator_time\t$sim_start"
        puts $output "$name\tend\tsimulator_time\t$sim_end"
        dict for {key value} $before { puts $output "$name\tstart\t$key\t$value" }
        dict for {key value} $after { puts $output "$name\tend\t$key\t$value" }
        flush $output
        if {[dict get $after cycles] - [dict get $before cycles] != 4096 ||
            [dict get $after sample_ps] - [dict get $before sample_ps] != 20480000} {
            error "Incomplete window or unexpected UI clock in $name"
        }
        set beats [expr {[dict get $after w_beats] - [dict get $before w_beats] +
                         [dict get $after r_beats] - [dict get $before r_beats]}]
        if {$beats <= 0} { error "No accepted data in $name" }
        foreach key {pending_w pending_r} {
            if {[dict get $before $key] < 0 || [dict get $after $key] < 0} {
                error "Negative transaction accounting in $name"
            }
        }
        if {![file isfile [file join $out $name.saif]] ||
            [file size [file join $out $name.saif]] == 0} { error "Missing SAIF" }
        set last $after
    }
    close $output
    if {[dict get $last w_beats] <= [dict get $first w_beats] ||
        [dict get $last r_beats] <= [dict get $first r_beats]} {
        error "Two windows did not observe both read and write traffic"
    }
    # Keep 20 ns after the final half-open window for actual VCD extent.
    run 20 ns
    ntrpu_rank_trace_close
    if {![file isfile [file join $out rank_commands.vcd]] ||
        [file size [file join $out rank_commands.vcd]] == 0} {error "Missing rank VCD"}
    set done [open [file join $out CAPTURE_COMPLETE.txt] {WRONLY CREAT EXCL}]
    puts $done "NTRPU_ACTIVITY_CAPTURE_EXPORTED; NOT_POWER_QUALIFIED"
    puts $done "Two 4096-cycle UI windows; automatic vendor PRBS; no SAIF mapping review yet"
    close $done
    puts "NTRPU_ACTIVITY_CAPTURE_EXPORTED; NOT_POWER_QUALIFIED"
}
if {[catch {ntrpu_capture_activity} message options]} {
    puts stderr "NTRPU_ACTIVITY_CAPTURE_ERROR $message"
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    # Keep partial files; absence of the completion receipt is a failed capture.
    quit
} else {
    quit
}
