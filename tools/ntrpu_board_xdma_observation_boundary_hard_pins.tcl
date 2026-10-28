# Read-only audit of bus/function outputs on the retained PCIE40E4 primitive.
proc audit_hard_pins {input output} {
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    set seed [file join $input xdma_observation_boundary.dcp]
    if {![file isfile $seed]} {error "Missing reviewed XDMA DCP"}
    file mkdir $output
    file copy $seed [file join $output input.dcp]
    set_param general.maxThreads 2
    open_checkpoint [file join $output input.dcp]
    set hard [get_cells -hierarchical -filter {REF_NAME == PCIE40E4}]
    if {[llength $hard]!=1} {error "Expected one retained PCIE40E4"}
    set f [open [file join $output hard_pins.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tpin\tdirection\tnets"
    foreach pin [lsort [get_pins -of_objects $hard]] {
        puts $f "$hard\t$pin\t[get_property DIRECTION $pin]\t[join [lsort [get_nets -of_objects $pin]] ,]"
    }
    close $f
    set f [open [file join $output hard_bdf_endpoints.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tsegment\tkind\tendpoint\tdirection\treference"
    set bus_count 0
    foreach pin [lsort [get_pins -of_objects $hard]] {
        set name [file tail $pin]
        if {![regexp {^CFG(BUSNUMBER|EXTFUNCTIONNUMBER)\[[0-9]+\]$} $name]} {continue}
        if {[string match {CFGBUSNUMBER*} $name]} {incr bus_count}
        if {[get_property DIRECTION $pin] ne "OUT"} {error "Unexpected BDF output direction"}
        set segments [get_nets -segments -of_objects $pin]
        if {![llength $segments]} {puts $f "$pin\t\tmissing\t\t\t"}
        foreach net [lsort $segments] {
            foreach endpoint [lsort -unique [get_pins -of_objects $net]] {
                set owner [get_cells -of_objects $endpoint]
                puts $f "$pin\t$net\tpin\t$endpoint\t[get_property DIRECTION $endpoint]\t[get_property REF_NAME $owner]"
            }
            foreach endpoint [lsort -unique [get_ports -of_objects $net]] {
                puts $f "$pin\t$net\tport\t$endpoint\t[get_property DIRECTION $endpoint]\t"
            }
        }
    }
    close $f
    if {$bus_count!=8} {error "Expected eight hard-block bus-number output bits"}
    close_design
    puts NTRPU_XDMA_OBSERVATION_BOUNDARY_HARD_PINS_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {audit_hard_pins {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
