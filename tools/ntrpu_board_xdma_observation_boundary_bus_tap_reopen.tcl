# Read-only audit of the serialized DCP, no connection edits.
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

proc reopen_bus_tap {input output} {
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    set seed [file join $input xdma_bus_tap.dcp]
    if {![file isfile $seed]} {error "Missing reviewed bus-tap DCP"}
    file mkdir $output
    file copy $seed [file join $output input.dcp]
    set_param general.maxThreads 2
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    dump_cells [file join $output cells.tsv]
    dump_transport [file join $output transport.tsv]
    set f [open [file join $output ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection\tnets"
    foreach port [lsort [get_ports]] {
        puts $f "$port\t[get_property DIRECTION $port]\t[join [lsort -unique [get_nets -of_objects $port]] ,]"
    }
    close $f
    set hard [get_cells -hierarchical -filter {REF_NAME == PCIE40E4}]
    if {[llength $hard]!=1} {error "Expected one PCIe primitive"}
    set f [open [file join $output persisted_bus_taps.tsv] {WRONLY CREAT EXCL}]
    puts $f "bit\tdriver\tprimitive\tdriver_direction\tnew_port\tport_direction\tleaves\tports\tsegments"
    set count 0
    foreach pin [lsort [get_pins -of_objects $hard]] {
        if {![regexp {^CFGBUSNUMBER\[([0-7])\]$} [file tail $pin] whole bit]} {continue}
        incr count
        set port [get_ports ntrpu_observe_cfg_bus_number_$bit]
        set segments [lsort -unique [get_nets -segments -of_objects $pin]]
        set leaves [lsort -unique [get_pins -leaf -of_objects $segments]]
        set ports [lsort -unique [get_ports -of_objects $segments]]
        if {[llength $port]!=1 || [get_property DIRECTION $port] ne "OUT"} {error "Missing output sink"}
        if {[get_property DIRECTION $pin] ne "OUT" || $leaves ne [list $pin] || $ports ne [list $port]} {error "Persisted bus connectivity differs"}
        if {[get_nets -of_objects $port] ni $segments} {error "Port is not on the bus net"}
        puts $f "$bit\t$pin\tPCIE40E4\tOUT\t$port\tOUT\t[join $leaves ,]\t[join $ports ,]\t[join $segments ,]"
    }
    close $f
    if {$count!=8} {error "Incomplete bus source"}
    close_design
    puts NTRPU_XDMA_OBSERVATION_BOUNDARY_BUS_TAP_REOPEN_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {reopen_bus_tap {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
