set stop_registers_script_dir [file dirname [file normalize [info script]]]
proc synth_stop_registers {output} {
    global stop_registers_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach label {controller} {
        set dir [file join $output $label]
        file mkdir $dir
        create_project stop_registers [file join $dir project] -part xcvu5p-flva2104-2-i
        add_files [file join $stop_registers_script_dir .. fpga rtl ntrpu_xdma_stop_registers_r1.sv]
        set_property top ntrpu_xdma_stop_registers [get_filesets sources_1]
        set f [open [file join $dir clocks.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name axi_aclk -period 4.000 [get_ports clk]}
        close $f
        add_files -fileset constrs_1 [file join $dir clocks.xdc]
        update_compile_order -fileset sources_1
        synth_design -mode out_of_context -top ntrpu_xdma_stop_registers -part xcvu5p-flva2104-2-i
        report_utilization -hierarchical -file [file join $dir utilization.rpt]
        report_timing_summary -report_unconstrained -check_timing_verbose -file [file join $dir timing.rpt]
        check_timing -verbose -file [file join $dir check_timing.rpt]
        report_exceptions -file [file join $dir exceptions.rpt]
        report_cdc -details -file [file join $dir cdc.rpt]
        report_drc -file [file join $dir drc.rpt]
        set f [open [file join $dir cells.tsv] {WRONLY CREAT EXCL}]
        puts $f "name\tref_name\tblackbox\tinit"
        foreach cell [get_cells -hierarchical] {
            puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property IS_BLACKBOX $cell]\t[get_property -quiet INIT $cell]"
        }
        close $f
        set f [open [file join $dir ports.tsv] {WRONLY CREAT EXCL}]
        puts $f "name\tdirection"
        foreach port [get_ports] {puts $f "$port\t[get_property DIRECTION $port]"}
        close $f
        set f [open [file join $dir sources.tsv] {WRONLY CREAT EXCL}]
        puts $f "path\ttype"
        foreach source [get_files -all] {puts $f "$source\t[get_property FILE_TYPE $source]"}
        close $f
        write_checkpoint [file join $dir stop_registers_synth.dcp]
        write_verilog -mode funcsim [file join $dir stop_registers_funcsim.v]
        close_project
        puts "NTRPU_STOP_REGISTERS_SYNTH_EXPORTED"
    }
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {synth_stop_registers [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
