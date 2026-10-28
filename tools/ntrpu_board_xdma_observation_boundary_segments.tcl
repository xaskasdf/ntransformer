# Read-only connectivity audit of the exact reviewed XDMA synthesis DCP.
# UG835 get_nets: upper is the default; query lower/both explicitly.
proc audit_boundary_segments {input output} {
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
    set cell [get_cells inst/pcie4_ip_i]
    if {[llength $cell]!=1} {error "Missing unique PCIe boundary"}
    set widths [dict create s_axis_rq_tdata 512 s_axis_rq_tuser 137 s_axis_rq_tvalid 1 s_axis_rq_tready 4 \
        m_axis_rc_tdata 512 m_axis_rc_tuser 161 m_axis_rc_tvalid 1 m_axis_rc_tready 1 \
        m_axis_rc_tkeep 16 m_axis_rc_tlast 1 cfg_bus_number 8 user_clk 1 user_reset 1 user_lnk_up 1]
    set counts [dict create]
    set f [open [file join $output boundary_segments.tsv] {WRONLY CREAT EXCL}]
    set e [open [file join $output boundary_endpoints.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tdirection\tupper\tlower\tsegments"
    puts $e "boundary_pin\tkind\tendpoint\tdirection\tcell\treference"
    foreach pin [lsort [get_pins -of_objects $cell]] {
        set name [file tail $pin]
        regsub {\[[0-9]+\]$} $name {} signal
        if {![dict exists $widths $signal]} {continue}
        dict incr counts $signal
        set upper [get_nets -boundary_type upper -of_objects $pin]
        set lower [get_nets -boundary_type lower -of_objects $pin]
        set segments [get_nets -segments -boundary_type both -of_objects $pin]
        puts $f "$pin\t[get_property DIRECTION $pin]\t[join [lsort $upper] ,]\t[join [lsort $lower] ,]\t[join [lsort $segments] ,]"
        if {![llength $segments]} {continue}
        foreach endpoint [lsort -unique [get_pins -leaf -of_objects $segments]] {
            set owner [get_cells -of_objects $endpoint]
            if {[llength $owner]!=1} {error "Leaf endpoint must have one owning cell"}
            puts $e "$pin\tpin\t$endpoint\t[get_property DIRECTION $endpoint]\t$owner\t[get_property REF_NAME $owner]"
        }
        foreach endpoint [lsort -unique [get_ports -of_objects $segments]] {
            puts $e "$pin\tport\t$endpoint\t[get_property DIRECTION $endpoint]\t\t"
        }
    }
    close $f
    close $e
    dict for {signal width} $widths {
        if {![dict exists $counts $signal] || [dict get $counts $signal]!=$width} {
            error "Missing or unexpected boundary width for $signal"
        }
    }
    set f [open [file join $output cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\treference\tprimitive\tblackbox"
    foreach c [lsort [get_cells -hierarchical]] {
        puts $f "$c\t[get_property REF_NAME $c]\t[get_property IS_PRIMITIVE $c]\t[get_property IS_BLACKBOX $c]"
    }
    close $f
    close_design
    puts NTRPU_XDMA_OBSERVATION_BOUNDARY_SEGMENTS_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {audit_boundary_segments {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
