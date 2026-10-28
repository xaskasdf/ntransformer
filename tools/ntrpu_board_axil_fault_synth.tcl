set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_fence_synth {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected fresh process and NEW_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach label {certificate} {
        create_project fence_$label [file join $output $label] -part xcvu5p-flva2104-2-i
        set_property XPM_LIBRARIES {XPM_CDC} [current_project]
        if {$label eq "admission"} {
            set top ntrpu_axil_retained_admission
            set sources {ntrpu_axil_retained_admission ntrpu_axil_drain_monitor}
        } else {
            set top ntrpu_axil_fence_certificate_r2
            set sources {ntrpu_axil_fence_certificate_r2 ntrpu_axil_pair_drain ntrpu_axil_drain_monitor ntrpu_tagged_reset_mailbox}
        }
        foreach name $sources {add_files [file join $ntrpu_script_dir .. fpga rtl ${name}.sv]}
        set clocks [file join $output ${label}_clocks.xdc]
        set f [open $clocks {WRONLY CREAT EXCL}]
        if {$label eq "admission"} {
            puts $f {create_clock -name host_clk -period 7.000 [get_ports clk]}
        } else {
            puts $f {create_clock -name host_clk -period 7.000 [get_ports host_clk]}
            puts $f {create_clock -name mgmt_clk -period 10.000 [get_ports mgmt_clk]}
            puts $f {create_clock -name device_clk -period 37.000 [get_ports device_clk]}
        }
        close $f
        add_files -fileset constrs_1 $clocks
        set_property top $top [get_filesets sources_1]
        update_compile_order -fileset sources_1
        synth_design -mode out_of_context -top $top -part xcvu5p-flva2104-2-i
        report_utilization -hierarchical -file [file join $output ${label}_utilization.rpt]
        report_timing_summary -file [file join $output ${label}_timing.rpt]
        check_timing -verbose -file [file join $output ${label}_check_timing.rpt]
        report_cdc -details -file [file join $output ${label}_cdc.rpt]
        report_exceptions -file [file join $output ${label}_exceptions.rpt]
        report_clock_interaction -file [file join $output ${label}_clock_interaction.rpt]
        set f [open [file join $output ${label}_cells.tsv] {WRONLY CREAT EXCL}]
        puts $f "name\tref_name\tblackbox\tinit"
        foreach c [get_cells -hierarchical] {
            puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_BLACKBOX $c]\t[get_property -quiet INIT $c]"
        }
        close $f
        write_checkpoint [file join $output ${label}_synth.dcp]
        write_verilog -mode funcsim [file join $output ${label}_funcsim.v]
        close_project
    }
    puts NTRPU_AXIL_FAULT_SYNTH_EXPORTED
}
if {[catch {ntrpu_fence_synth $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
