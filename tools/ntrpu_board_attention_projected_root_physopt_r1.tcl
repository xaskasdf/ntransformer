# Implement the frozen, qualified projected root checkpoint in an explicit OOC context.
set route_script_dir [file dirname [file normalize [info script]]]

proc root_clock_audit {output phase} {
    set inst ntrpu_projected_root_parent_bufg
    set cell [get_cells -quiet $inst]
    if {[llength $cell]!=1 || [get_property REF_NAME $cell] ne "BUFGCE" ||
        [get_property LOC $cell] ne "BUFGCE_X0Y60"} {error "Clock buffer identity/location"}
    if {[llength [get_clocks]]!=1 || [get_property PERIOD [get_clocks clk]]!=5.0 ||
        [get_clocks -of_objects [get_pins $inst/O]] ne "clk"} {error "Changed clock context"}
    set input_nets [get_nets -segments -of_objects [get_pins $inst/I]]
    set input_ports [get_ports -of_objects $input_nets]
    set input_loads [get_pins -leaf -of_objects $input_nets -filter {DIRECTION == IN}]
    if {$input_ports ne "clk" || $input_loads ne "$inst/I" ||
        [llength [get_pins -quiet -leaf -of_objects $input_nets -filter {DIRECTION == OUT}]]} {
        error "Clock input has bypass or foreign source"
    }
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected assumed clock source"}
    set ce_cells [get_cells -of_objects [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $inst/CE]] -filter {DIRECTION == OUT}]]
    if {[llength $ce_cells]!=1 || [get_property REF_NAME $ce_cells] ne "VCC"} {error "Clock enable not static high"}
    set output_net [get_nets -of_objects [get_pins $inst/O]]
    set loads [get_pins -leaf -of_objects [get_nets -segments $output_net] -filter {DIRECTION == IN}]
    if {[llength $loads]<1000} {error "Clock consumers missing"}
    set f [open [file join $output ${phase}_clock.tsv] {WRONLY CREAT EXCL}]
    puts $f "instance\tref\tloc\tclock\tperiod\tinput\tce\tloads\troute_status"
    puts $f "$inst\tBUFGCE\tBUFGCE_X0Y60\tclk\t5.000\t$input_ports\tVCC\t[llength $loads]\t[get_property ROUTE_STATUS $output_net]"
    close $f
}

proc root_route_reports {output} {
    report_route_status -file [file join $output route.rpt]
    report_timing_summary -delay_type min_max -report_unconstrained -max_paths 10 -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_drc -file [file join $output drc.rpt]
    set f [open [file join $output drc_violations.tsv] {WRONLY CREAT EXCL}]
    puts $f "name\tseverity"
    foreach v [get_drc_violations -quiet] {puts $f "[get_property NAME $v]\t[get_property SEVERITY $v]"}
    close $f
    report_methodology -file [file join $output methodology.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    write_xdc -constraints all [file join $output constraints_after.xdc]
    set f [open [file join $output metrics.tsv] {WRONLY CREAT EXCL}]
    puts $f "top\t[get_property TOP [current_design]]"
    puts $f "part\t[get_property PART [current_design]]"
    puts $f "clock_period\t[get_property PERIOD [get_clocks clk]]"
    foreach flag {PLACED_FULLY PARTIALLY_ROUTED ROUTED_FULLY ERRORS_IN_ROUTES} {puts $f "$flag\t[get_property $flag [current_design]]"}
    foreach {key filter} {blackboxes {IS_BLACKBOX == 1} latches {REF_NAME =~ LD*} buffers {REF_NAME =~ BUFG*}} {
        puts $f "$key\t[llength [get_cells -quiet -hier -filter $filter]]"
    }
    foreach {kind delay} {setup max hold min} {
        set regs [all_registers -clock clk]
        set path [get_timing_paths -from $regs -to $regs -delay_type $delay -max_paths 1]
        if {[llength $path]!=1} {error "Missing internal $kind path"}
        puts $f "internal_${kind}\t[get_property SLACK $path]"
        report_timing -from $regs -to $regs -delay_type $delay -max_paths 10 -input_pins -file [file join $output internal_${kind}.rpt]
    }
    close $f
    set f [open [file join $output route_inventory.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus\tgap"
    set nets [get_nets -hier]
    foreach net $nets status [get_property ROUTE_STATUS $nets] route [get_property ROUTE $nets] {
        puts $f "$net\t$status\t[expr {[string match -nocase *gap* $route] ? 1:0}]"
    }
    close $f
    root_clock_audit $output reopened
}

proc attention_projected_root_route {output} {
    global route_script_dir
    set root [file normalize [file join $route_script_dir ..]]
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint [file join $root evidence attention_projected_root_route_input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_attention_projected_root_r1" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [llength [get_clocks]]!=1 || [get_property PERIOD [get_clocks clk]]!=5.0 ||
        [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]] ||
        [get_property ROUTED_FULLY [current_design]] != 1} {error "Unexpected parent checkpoint"}
    write_xdc -constraints all [file join $output constraints_input.xdc]
    root_clock_audit $output prepared
    write_xdc -constraints all [file join $output constraints_before.xdc]
    write_checkpoint [file join $output prepared.dcp]
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Projected root OOC physical context\n\nFrozen routed projected root, post-route Explore; original static clock at BUFGCE_X0Y60 retained. Clock remains 5 ns."
    puts $f "The chosen buffer site is provisional; no parent floorplan, I/O budget or physical board clock is qualified."
    puts $f "No false paths, multicycle paths, severity changes or waivers are added. External ports remain unconstrained."
    puts $f "Original RTL arithmetic/protocol evidence belongs to the archived parent; post-route functional simulation remains separate."
    close $f
    set failed [catch {
        report_timing_summary -report_unconstrained -file [file join $output baseline_timing.rpt]
        # Post-route Explore reroutes its changed nets (same command as existing interface studies).
        phys_opt_design -directive Explore
    } message options]
    set f [open [file join $output route_command.txt] {WRONLY CREAT EXCL}]
    puts $f "RETURN_CODE $failed\nMESSAGE $message"
    if {$failed && [dict exists $options -errorinfo]} {puts $f [dict get $options -errorinfo]}
    close $f
    write_checkpoint [file join $output route_attempt.dcp]
    close_design
    open_checkpoint [file join $output route_attempt.dcp]
    root_route_reports $output
    write_verilog -mode funcsim [file join $output projected_root_route_funcsim.v]
    if {$failed} {error "Implementation failed; checkpoint and reports preserved"}
    puts NTRPU_ATTENTION_PROJECTED_ROOT_PHYSOPT_R1_EXPORTED
    close_design
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {attention_projected_root_route [lindex $argv 0]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
