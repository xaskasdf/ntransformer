# Private OOC diagnostic. No board timing closure and no timing exceptions.
set_param general.maxThreads 2
proc guard_cdc_reports {out} {
    check_timing -verbose -file $out/check_timing.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_clock_interaction -file $out/clock_interaction.rpt
    report_clocks -file $out/clocks.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    report_methodology -file $out/methodology.rpt
    report_drc -file $out/drc.rpt
    report_utilization -file $out/utilization.rpt
    report_exceptions -coverage -file $out/exceptions.rpt
    report_waivers -file $out/waivers.rpt
    write_xdc -constraints all $out/effective.xdc
}
proc guard_cdc_inventory {out} {
    set f [open $out/cells.tsv {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tasync_reg\tinit\tloc"
    foreach c [lsort [get_cells -hierarchical]] {
        puts $f [join [list $c [get_property REF_NAME $c] [get_property ASYNC_REG $c] [get_property INIT $c] [get_property LOC $c]] "\t"]
    }
    close $f
    set f [open $out/pins.tsv {WRONLY CREAT EXCL}]
    puts $f "pin\tdirection\tnet\tdrivers\tinput_ports\tclocks"
    foreach p [lsort [get_pins -hierarchical]] {
        set nets [get_nets -quiet -of_objects $p]
        set drivers [get_pins -quiet -of_objects $nets -filter {DIRECTION==OUT}]
        set ports [get_ports -quiet -of_objects $nets -filter {DIRECTION==IN}]
        set clocks [get_clocks -quiet -of_objects $p]
        puts $f [join [list $p [get_property DIRECTION $p] [join $nets ,] [join $drivers ,] [join $ports ,] [join $clocks ,]] "\t"]
    }
    close $f
    set f [open $out/paths.tsv {WRONLY CREAT EXCL}]
    puts $f "source_clock\tdestination_clock\tdelay\tstartpoint\tendpoint\tslack"
    foreach source {mgmt_clk axi_clk} {
        foreach dest {mgmt_clk axi_clk} {
            foreach delay {max min} {
                set paths [get_timing_paths -quiet -from [get_clocks $source] -to [get_clocks $dest] -delay_type $delay -max_paths 1000 -nworst 1000]
                foreach p $paths {
                    puts $f [join [list $source $dest $delay [get_property STARTPOINT_PIN $p] [get_property ENDPOINT_PIN $p] [get_property SLACK $p]] "\t"]
                }
            }
        }
    }
    close $f
}
if {[info exists ::guard_cdc_library_only]} {return}
if {[catch {
    if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {error "RTL NEW_OUTPUT required"}
    lassign $argv rtl out
    set rtl [file normalize $rtl];set out [file normalize $out]
    if {![file isfile $rtl] || [file exists $out]} {error "Expected existing RTL and new output"}
    file mkdir $out
    foreach command {report_cdc report_timing_summary report_synchronizer_mtbf get_timing_paths} {
        puts "GUARD_HELP $command"
        help $command
    }
    read_verilog -sv $rtl
    synth_design -top ntrpu_axi_bridge_guard -part xcvu47p-fsvh2892-2-e -mode out_of_context
    create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]
    create_clock -name axi_clk -period 4.000 [get_ports axi_clk]
    guard_cdc_reports $out
    guard_cdc_inventory $out
    write_checkpoint $out/guard_synth.dcp
    write_verilog -mode funcsim $out/guard_synth.v
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f "GUARD_CDC_DIAGNOSTIC_EXPORTED_NOT_TIMING_CLOSED"
    close $f
    close_design
} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
puts GUARD_CDC_DIAGNOSTIC_EXPORTED_NOT_TIMING_CLOSED
exit 0
