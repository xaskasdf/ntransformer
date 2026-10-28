# Actual BUFG and registered measurement boundaries around the unchanged core DCP.
# This fixture does NOT implement an AXI register slice or a board interface.
set script_dir [file dirname [file normalize [info script]]]
proc registered_route {arguments} {
    global script_dir
    if {[llength $arguments] ni {4 6 7 8}} {error "Expected RESERVE SYNTH_DCP BUFGCE_SITE NEW_DIRECTORY ?DIRECTIVE RESUME_FOLDER ?THREADS ?PLACE_DIRECTIVE???"}
    lassign $arguments reserve checkpoint clock_site output directive resume_folder threads place_directive
    if {$directive eq {}} {set directive Default}
    if {$threads eq {}} {set threads 2}
    if {$threads ni {2 4 8}} {error "Unexpected thread budget"}
    if {$place_directive eq {}} {set place_directive Default}
    if {$place_directive ni {Default AltSpreadLogic_high}} {error "Unexpected placement directive"}
    if {$directive ni {Default RuntimeOptimized Quick}} {error "Unexpected route directive"}
    set resumed [expr {$resume_folder ne {} && [file exists [file join $resume_folder placed.dcp]]}]
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    exec python3 -B [file join $script_dir prepare_ntrpu_multibuffer_physical_fixture.py] \
        [file join $output input.dcp] [file join $output fixture] $reserve
    set f [open [file join $output fixture core_top.txt] r]
    set core_top [string trim [read $f]]
    close $f
    if {$core_top ni {ntrpu_peer_multibuffer_compact ntrpu_peer_multibuffer_control ntrpu_peer_multibuffer_decoded ntrpu_peer_multibuffer_fillstage ntrpu_peer_multibuffer_elastic}} {error "Unexpected fixture core"}
    set_param general.maxThreads $threads
    if {$resumed} {
        foreach name {core_before_export.rpt core_after_import.rpt} {
            file copy [file join $resume_folder $name] [file join $output $name]
        }
        file copy [file join $resume_folder placed.dcp] [file join $output input_placed.dcp]
        open_checkpoint [file join $output input_placed.dcp]
        if {$place_directive eq "Default"} {file copy [file join $output input_placed.dcp] [file join $output placed.dcp]}
        if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
            [get_property TOP [current_design]] ne "ntrpu_multibuffer_physical_fixture" ||
            [get_property LOC [get_cells clock_buffer]] ne $clock_site ||
            [llength [get_clocks]] != 1 || [get_property PERIOD [get_clocks fixture_clk]] != 5.0} {
            error "Unexpected resumed fixture"
        }
    } else {
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [get_property TOP [current_design]] ne $core_top} {error "Unexpected core checkpoint"}
    write_edif [file join $output core.edf]
    report_utilization -file [file join $output core_before_export.rpt]
    close_project
    create_project -in_memory -part xcvu5p-flva2104-2-i
    read_verilog [file join $output fixture core_stub.v]
    read_verilog -sv [file join $output fixture fixture.sv]
    synth_design -top ntrpu_multibuffer_physical_fixture -part xcvu5p-flva2104-2-i \
        -mode out_of_context -flatten_hierarchy none
    # Preserve the wrapper checkpoint and import Vivado's own plain EDIF export.
    # read_checkpoint -cell rejects this DCP in 2026.1 (Project 1-9), even
    # after reopening a structural wrapper. Never decompress internal DCP data.
    write_checkpoint [file join $output fixture_synth.dcp]
    close_project
    open_checkpoint [file join $output fixture_synth.dcp]
    update_design -cells core -from_file [file join $output core.edf]
    report_utilization -cells [get_cells core] -file [file join $output core_after_import.rpt]
    # Replace imported block clock definitions with the single real fixture clock.
    if {[llength [get_clocks -quiet]]} {delete_clocks [get_clocks]}
    create_clock -period 5.000 -name fixture_clk [get_ports clk]
    set_clock_uncertainty -setup 0.100 [get_clocks fixture_clk]
    set_clock_uncertainty -hold 0.050 [get_clocks fixture_clk]
    set_property LOC $clock_site [get_cells clock_buffer]
    }
    if {[get_property REF_NAME [get_cells core]] ne $core_top ||
        [get_property REF_NAME [get_cells clock_buffer]] ne "BUFGCE" ||
        [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]} {error "Incomplete fixture"}
    write_xdc -constraints all [file join $output input_context.xdc]
    set f [open [file join $output CONTEXT.md] {WRONLY CREAT EXCL}]
    puts $f "# Registered timing fixture\n"
    puts $f "Real BUFGCE at $clock_site; 5 ns clock, additional setup/hold uncertainty 0.100/0.050 ns."
    puts $f "Preserved registers bound every core data/control input/output; shared physical clock."
    puts $f "Raw probe pins intentionally have no external delays; no board pins, parent IP, CDC or full interface qualification."
    puts $f "The fixture is NOT an AXI adapter. Only core netlist is used for protocol regression."
    puts $f "No false paths, multicycle exceptions, case analysis, waivers or severity changes."
    puts $f "Route directive: $directive. Resumed placed checkpoint: $resumed."
    puts $f "Placement directive: $place_directive; max threads: $threads."
    close $f
    set failed [catch {
        if {!$resumed || $place_directive ne "Default"} {
            if {$resumed} {place_design -unplace} else {opt_design}
            place_design -directive $place_directive
            write_checkpoint [file join $output placed.dcp]
        }
        report_timing_summary -report_unconstrained -file [file join $output placed_timing.rpt]
        phys_opt_design
        route_design -directive $directive
    } message options]
    set f [open [file join $output route_status.txt] {WRONLY CREAT EXCL}]
    puts $f "return_code=$failed\nmessage=$message"
    if {$failed && [dict exists $options -errorinfo]} {puts $f [dict get $options -errorinfo]}
    close $f
    write_checkpoint [file join $output route_attempt.dcp]
    report_route_status -file [file join $output route.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    check_timing -verbose -file [file join $output check_timing.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_utilization -hierarchical -file [file join $output hierarchy.rpt]
    report_utilization -cells [get_cells core] -file [file join $output core_utilization.rpt]
    report_drc -file [file join $output drc.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    write_xdc -constraints all [file join $output implemented.xdc]
    set clock_net [get_nets -of_objects [get_pins clock_buffer/O]]
    report_property -all $clock_net -file [file join $output clock_net.rpt]
    report_route_status -of_objects $clock_net -file [file join $output clock_route.rpt]
    set regs [all_registers]
    set core_regs [filter $regs {NAME =~ core/*}]
    set probe_regs [filter $regs {NAME !~ core/*}]
    set f [open [file join $output paths.tsv] {WRONLY CREAT EXCL}]
    puts $f "scope\ttype\tslack_ns\tsource\tdestination"
    foreach scope {all core ingress egress} {
        if {$scope eq "all"} {set filters [list -from $regs -to $regs]}
        if {$scope eq "core"} {set filters [list -from $core_regs -to $core_regs]}
        if {$scope eq "ingress"} {set filters [list -from $probe_regs -to $core_regs]}
        if {$scope eq "egress"} {set filters [list -from $core_regs -to $probe_regs]}
        foreach delay {max min} {
            set path [get_timing_paths -quiet -delay_type $delay -max_paths 1 {*}$filters]
            if {[llength $path] != 1} {error "Missing $scope $delay path"}
            puts $f "$scope\t$delay\t[get_property SLACK $path]\t[get_property STARTPOINT_PIN $path]\t[get_property ENDPOINT_PIN $path]"
            report_timing -delay_type $delay -max_paths 10 -path_type full_clock_expanded \
                {*}$filters -file [file join $output ${scope}_${delay}_paths.rpt]
        }
    }
    close $f
    set counts [dict create]
    set gaps 0
    set f [open [file join $output route_gaps.txt] {WRONLY CREAT EXCL}]
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] route [get_property ROUTE $nets] {
        dict incr counts $status
        if {[string match -nocase *gap* $route]} {incr gaps; puts $f "NET $net STATUS $status\n$route"}
        if {$status ni {ROUTED INTRASITE}} {puts $f "NON_FULLY_ROUTED_NET $net STATUS $status"}
    }
    puts $f "ROUTE_STRING_GAP_NETS $gaps\nROUTE_STATUS_COUNTS $counts"
    close $f
    set f [open [file join $output route_counts.tsv] {WRONLY CREAT EXCL}]
    dict for {key value} $counts {puts $f "$key\t$value"}
    close $f
    set f [open [file join $output summary.tsv] {WRONLY CREAT EXCL}]
    foreach {key value} [list part [get_property PART [current_design]] tool_version [version -short] \
        reserve_banks $reserve period_ns [get_property PERIOD [get_clocks fixture_clk]] \
        clock_count [llength [get_clocks]] clock_site [get_property LOC [get_cells clock_buffer]] \
        clock_route_status [get_property ROUTE_STATUS $clock_net] \
        route_command_failed $failed route_string_gap_nets $gaps max_threads [get_param general.maxThreads] \
        route_directive $directive resumed_placed $resumed \
        place_directive $place_directive core_top $core_top \
        core_register_cells [llength $core_regs] probe_register_cells [llength $probe_regs] \
        latch_count [llength [get_cells -quiet -hier -filter {REF_NAME =~ LD*}]] \
        blackbox_count [llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]] {puts $f "$key\t$value"}
    close $f
    if {$failed} {error "Implementation failed; diagnostics preserved"}
    write_verilog -cell core -mode funcsim -rename_top ntrpu_peer_multibuffer_compact \
        [file join $output multibuffer_funcsim.v]
    puts "MULTIBUFFER_REGISTERED_ROUTE_EXPORTED RESERVE=$reserve; synchronous fixture only"
    close_project
}
if {[catch {registered_route $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
