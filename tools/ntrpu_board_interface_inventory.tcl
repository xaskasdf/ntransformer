# Read-only extraction from existing DDR and serial PCIe checkpoints.
# Creates stubs and constraints for a subsequent isolated coexistence probe.
proc ntrpu_interface_inventory {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected DDR_DCP PCIE_DCP NEW_DIRECTORY in a fresh process"
    }
    lassign $arguments ddr pcie output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    foreach {name checkpoint} [list ddr $ddr pcie $pcie] {
        open_checkpoint [file normalize $checkpoint]
        if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected part" }
        set stream [open [file join $output ${name}_ports.tsv] {WRONLY CREAT EXCL}]
        puts $stream "port\tdirection\tball\tstandard\tdrive\tslew\tpulltype"
        foreach port [lsort [get_ports]] {
            set values [list $port]
            foreach property {DIRECTION PACKAGE_PIN IOSTANDARD DRIVE SLEW PULLTYPE} {
                lappend values [get_property $property $port]
            }
            puts $stream [join $values \t]
        }
        close $stream
        set stream [open [file join $output ${name}_special_cells.tsv] {WRONLY CREAT EXCL}]
        puts $stream "cell\ttype\tloc"
        foreach cell [get_cells -hierarchical -filter {REF_NAME == STARTUPE3 || REF_NAME == DCIRESET || REF_NAME == PCIE40E4 || REF_NAME == GTYE4_CHANNEL || REF_NAME == MMCME4_ADV || REF_NAME == PLLE4_ADV}] {
            puts $stream "$cell\t[get_property REF_NAME $cell]\t[get_property LOC $cell]"
        }
        close $stream
        report_property -all -file [file join $output ${name}_design.rpt] [current_design]
        write_verilog -mode synth_stub [file join $output ${name}_stub.v]
        write_xdc [file join $output ${name}_constraints.xdc]
        report_clocks -file [file join $output ${name}_clocks.rpt]
        report_io -file [file join $output ${name}_io.rpt]
        close_project
    }
    puts "NTRPU_INTERFACE_INVENTORY_COMPLETE; metadata only; no merge or joint validation"
}
if {[catch {ntrpu_interface_inventory $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
