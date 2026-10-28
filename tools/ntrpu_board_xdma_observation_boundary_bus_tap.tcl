# Isolated ECO feasibility probe: export the eight real bus-number nets as sinks.
# No source edits, new logic, timing exceptions, resets or transport drivers.
proc dump_cells {path} {
    set f [open $path {WRONLY CREAT EXCL}]
    set rows {}
    puts $f "cell\treference\tprimitive\tblackbox"
    foreach c [lsort [get_cells -hierarchical]] {
        set row "$c\t[get_property REF_NAME $c]\t[get_property IS_PRIMITIVE $c]\t[get_property IS_BLACKBOX $c]"
        puts $f $row
        lappend rows $row
    }
    close $f
    return $rows
}
proc dump_transport {path} {
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "boundary_pin\tkind\tendpoint\tdirection\treference"
    set names {s_axis_rq_tdata s_axis_rq_tuser s_axis_rq_tvalid s_axis_rq_tready m_axis_rc_tdata m_axis_rc_tuser m_axis_rc_tvalid m_axis_rc_tready m_axis_rc_tkeep m_axis_rc_tlast cfg_bus_number user_clk user_reset user_lnk_up}
    set rows {}
    set count 0
    foreach pin [lsort [get_pins -of_objects [get_cells inst/pcie4_ip_i]]] {
        regsub {\[[0-9]+\]$} [file tail $pin] {} signal
        if {$signal ni $names} {continue}
        incr count
        set segments [get_nets -segments -boundary_type both -of_objects $pin]
        if {![llength $segments]} {error "Missing transport segments: $pin"}
        foreach endpoint [lsort -unique [get_pins -leaf -of_objects $segments]] {
            set row "$pin\tpin\t$endpoint\t[get_property DIRECTION $endpoint]\t[get_property REF_NAME [get_cells -of_objects $endpoint]]"
            puts $f $row
            lappend rows $row
        }
        foreach endpoint [lsort -unique [get_ports -of_objects $segments]] {
            set row "$pin\tport\t$endpoint\t[get_property DIRECTION $endpoint]\t"
            puts $f $row
            lappend rows $row
        }
    }
    close $f
    if {$count!=1357} {error "Incomplete transport boundary"}
    return $rows
}
proc bus_tap {input output} {
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    set seed [file join $input xdma_observation_boundary.dcp]
    if {![file isfile $seed]} {error "Missing reviewed XDMA DCP"}
    file mkdir $output
    file copy $seed [file join $output input.dcp]
    set_param general.maxThreads 2
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    set original_cells [dump_cells [file join $output before_cells.tsv]]
    set original_transport [dump_transport [file join $output before_transport.tsv]]
    set original_ports [lsort [get_ports]]
    set hard [get_cells -hierarchical -filter {REF_NAME == PCIE40E4}]
    if {[llength $hard]!=1} {error "Expected one retained PCIe primitive"}
    set bus_pins [dict create]
    foreach pin [get_pins -of_objects $hard] {
        if {[regexp {^CFGBUSNUMBER\[([0-7])\]$} [file tail $pin] whole bit]} {
            dict set bus_pins $bit $pin
        }
    }
    if {[dict size $bus_pins]!=8} {error "Missing bus-number bits"}
    set f [open [file join $output bus_taps.tsv] {WRONLY CREAT EXCL}]
    puts $f "bit\tdriver\tprimitive\toriginal_net\tnew_port\tleaves_before\tleaves_after\tports_after"
    set added_ports {}
    for {set bit 0} {$bit<8} {incr bit} {
        set pin [dict get $bus_pins $bit]
        if {[get_property DIRECTION $pin] ne "OUT"} {error "Bus pin is not an output"}
        set net [get_nets -of_objects $pin]
        if {[llength $net]!=1} {error "Expected one actual bus-number net"}
        set segments [get_nets -segments -of_objects $pin]
        set leaves [lsort -unique [get_pins -leaf -of_objects $segments]]
        if {$leaves ne [list $pin] || [llength [get_ports -of_objects $segments]]} {
            error "Unexpected existing bus-number endpoints"
        }
        set port [create_port -direction OUT ntrpu_observe_cfg_bus_number_$bit]
        if {[llength $port]!=1} {error "Expected one new observation sink"}
        lappend added_ports $port
        connect_net -hierarchical -basename ntrpu_bus_observe_$bit -net $net -objects $port
        set after_segments [get_nets -segments -of_objects $pin]
        set after_leaves [lsort -unique [get_pins -leaf -of_objects $after_segments]]
        set after_ports [lsort -unique [get_ports -of_objects $after_segments]]
        if {$after_leaves ne $leaves || $after_ports ne [list $port]} {
            error "Tap changed a primitive endpoint or added unexpected ports"
        }
        if {[get_property DIRECTION $port] ne "OUT"} {error "Observation port must only be a sink"}
        puts $f "$bit\t$pin\tPCIE40E4\t$net\t$port\t[join $leaves ,]\t[join $after_leaves ,]\t[join $after_ports ,]"
    }
    close $f
    if {[lsort [get_ports]] ne [lsort [concat $original_ports $added_ports]]} {error "Unexpected port mutation"}
    if {[dump_cells [file join $output after_cells.tsv]] ne $original_cells} {error "Original cells changed"}
    if {[dump_transport [file join $output after_transport.tsv]] ne $original_transport} {error "Transport connectivity changed"}
    report_clocks -file [file join $output clocks.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_drc -file [file join $output drc.rpt]
    write_checkpoint [file join $output xdma_bus_tap.dcp]
    write_verilog -mode funcsim [file join $output xdma_bus_tap_funcsim.v]
    close_design
    puts NTRPU_XDMA_OBSERVATION_BOUNDARY_BUS_TAP_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {bus_tap {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
