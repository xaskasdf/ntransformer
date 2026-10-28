# V4 structural coexistence probe, not V5/V6 or exact DIMM qualification.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_interface_coexist {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 4 || [llength [get_projects -quiet]]} {
        error "Expected INVENTORY DDR_DCP PCIE_DCP NEW_DIRECTORY"
    }
    lassign $arguments inventory ddr pcie output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    exec python3 [file join $ntrpu_script_dir prepare_ntrpu_interface_top.py] $inventory [file join $output wrapper] > [file join $output prepare.log] 2>@1
    file copy $ddr [file join $output ddr_input.dcp]
    file copy $pcie [file join $output pcie_input.dcp]
    create_project coexist [file join $output project] -part xcvu5p-flva2104-2-i
    read_verilog [glob [file join $output wrapper *.v]]
    synth_design -top ntrpu_ddr_pcie_coexist -part xcvu5p-flva2104-2-i -no_iobuf -flatten_hierarchy none
    read_checkpoint -cell u_ddr [file join $output ddr_input.dcp]
    read_checkpoint -cell u_pcie [file join $output pcie_input.dcp]
    set stream [open [file join $output wrapper ports.tsv] r]
    set lines [split [string trim [read $stream]] \n]; close $stream
    foreach row [lrange $lines 1 end] {
        lassign [split [string trimright $row \r] \t] port direction ball standard drive slew pulltype original interface
        set object [get_ports $port]
        if {[llength $object] != 1} { error "Missing combined port $port" }
        set_property PACKAGE_PIN $ball $object
        if {$standard ne ""} { set_property IOSTANDARD $standard $object }
        if {$drive ne ""} { set_property DRIVE $drive $object }
        if {$slew ne ""} { set_property SLEW $slew $object }
        if {$pulltype ne ""} { set_property PULLTYPE $pulltype $object }
    }
    write_checkpoint [file join $output combined_original_voltage.dcp]
    if {[llength [get_ports]] != 221} { error "Combined interface count differs" }
    foreach {type count} {DCIRESET 1 STARTUPE3 1 GTYE4_CHANNEL 16 PCIE40E4 1} {
        if {[llength [get_cells -hierarchical -filter "REF_NAME == $type"]] != $count} {
            error "Combined primitive count differs: $type"
        }
    }
    if {[llength [get_cells -hierarchical -filter {IS_BLACKBOX == 1}]]} { error "Unresolved black boxes" }
    if {[get_property PACKAGE_PIN [get_ports pcie_sys_rst_n]] ne "AM17"} { error "PERST ball changed" }
    if {[get_property CLOCK_DEDICATED_ROUTE [get_nets u_ddr/ntrpu_ref_raw_BUFGCE]] ne "BACKBONE"} {
        error "DDR dedicated backbone constraint lost"
    }
    report_drc -checks [get_drc_checks BIVC-1] -file [file join $output mixed_voltage.rpt]
    set mixed [llength [get_drc_violations -quiet]]
    if {$mixed == 0} { error "Expected original LVCMOS18/SSTL12 bank conflict" }
    set_property IOSTANDARD LVCMOS12 [get_ports pcie_sys_rst_n]
    report_drc -checks [get_drc_checks BIVC-1] -file [file join $output shared_1v2.rpt]
    set shared [llength [get_drc_violations -quiet]]
    report_drc -checks [get_drc_checks DCIRST-1] -file [file join $output dci.rpt]
    set dci [llength [get_drc_violations -quiet]]
    write_checkpoint [file join $output combined_1v2.dcp]
    write_xdc [file join $output combined_1v2.xdc]
    report_io -file [file join $output io.rpt]
    report_clocks -file [file join $output clocks.rpt]
    report_utilization -file [file join $output utilization.rpt]
    report_drc -file [file join $output drc.rpt]
    if {$shared || $dci} { error "Combined targeted checks failed: BIVC=$shared DCI=$dci" }
    puts "NTRPU_INTERFACE_COEXISTENCE_TARGETED_PASS ports=221 mixed=$mixed shared=$shared dci=$dci"
    puts "One generic DDR channel plus independent PCIe/BRAM; requires external PERST level translation. No placement/routing or board approval."
    close_project
}
if {[catch {ntrpu_interface_coexist $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
