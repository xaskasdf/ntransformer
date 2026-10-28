# Trace an isolated SERIAL example through a real synthesized netlist.
# No placement, routing, bitstream, DDR coexistence or board approval.
proc ntrpu_pcie_lane_trace {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected SERIAL_EXAMPLE NEW_DIRECTORY in a fresh process"
    }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $input [file join $output example]
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected part" }
    set ip [get_ips ntrpu_xdma]
    if {[llength $ip] != 1 || [get_property CONFIG.pipe_sim $ip] ne "false"} {
        error "Only the unmodified serial example may be synthesized"
    }
    foreach source [get_files -all] {
        if {[string first $input [file normalize $source]] == 0} { error "Reference to immutable input: $source" }
    }
    # The application also instantiates a block-memory IP. Global synthesis
    # needs its RTL targets, not only the XDMA's generated targets.
    foreach current_ip [get_ips] {
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $current_ip]]
        generate_target synthesis $current_ip
    }
    update_compile_order -fileset sources_1
    synth_design -top xilinx_dma_pcie_ep -part xcvu5p-flva2104-2-i -flatten_hierarchy rebuilt
    write_checkpoint [file join $output pcie_synth.dcp]
    report_utilization -file [file join $output utilization.rpt]
    report_io -file [file join $output io.rpt]
    report_ip_status -file [file join $output ip_status.rpt]
    set gt [get_cells -hierarchical -filter {REF_NAME == GTYE4_CHANNEL}]
    if {[llength $gt] != 16} { error "Expected 16 real GT primitives, got [llength $gt]" }
    set stream [open [file join $output lane_trace.tsv] {WRONLY CREAT EXCL}]
    puts $stream "port\tprimitive_pin\tgt_site\tbank\tpackage_candidates"
    set seen [dict create]
    foreach {suffix primitive_pin} {rxp GTYRXP rxn GTYRXN txp GTYTXP txn GTYTXN} {
        for {set lane 0} {$lane < 16} {incr lane} {
            set name [format {pci_exp_%s[%d]} $suffix $lane]
            set port [get_ports $name]
            set nets [get_nets -segments -of_objects $port]
            set pins [get_pins -leaf -of_objects $nets -filter "REF_PIN_NAME == $primitive_pin"]
            if {[llength $pins] != 1} { error "Expected one GT pin for $name, got $pins" }
            set cell [get_cells -of_objects $pins]
            if {[get_property REF_NAME $cell] ne "GTYE4_CHANNEL"} { error "Unexpected driver for $name" }
            set site [get_property LOC $cell]
            if {$site eq ""} { error "No generated GT LOC for $name" }
            if {[dict exists $seen $suffix,$site]} { error "Duplicate GT site for $name" }
            dict set seen $suffix,$site 1
            dict set seen mapping,$suffix,$lane $site
            set site_object [get_sites $site]
            set candidates [get_package_pins -quiet -of_objects [get_bels -of_objects $site_object]]
            puts $stream "$name\t$pins\t$site\t[get_iobanks -of_objects $site_object]\t$candidates"
        }
    }
    close $stream
    for {set lane 0} {$lane < 16} {incr lane} {
        set reference [dict get $seen mapping,rxp,$lane]
        foreach suffix {rxn txp txn} {
            if {[dict get $seen mapping,$suffix,$lane] ne $reference} { error "P/N or TX/RX lane mismatch at $lane" }
        }
        if {$reference ne "GTYE4_CHANNEL_X1Y[expr {15-$lane}]"} {
            error "Netlist differs from source inference at lane $lane: $reference"
        }
    }
    puts "NTRPU_PCIE_LANE_TRACE_PASS lanes=16 scalar_ports=64 source_inference_matches=1"
    puts "Synthesis connectivity only. Package candidates require pin-function review; no placement or routing."
    close_project
}
if {[catch {ntrpu_pcie_lane_trace $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
