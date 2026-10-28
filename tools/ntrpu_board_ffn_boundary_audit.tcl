# Read-only expansion of RTSTAT-10; no rule waiver or design modification.
proc boundary_audit {arguments} {
    if {[llength $arguments]!=2} {error "Expected ROUTED_DCP NEW_DIRECTORY"}
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_ffn_supervised_fabric" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" ||
        [llength [get_clocks]]!=3} {error "Unexpected design context"}
    help get_drc_violations
    help get_nets
    report_drc -checks [get_drc_checks RTSTAT-10] -file [file join $output boundary_drc.rpt]
    set violations [get_drc_violations -quiet *RTSTAT-10*]
    if {[llength $violations]!=1} {error "Expected one RTSTAT-10 finding"}
    report_property -all $violations -file [file join $output violation_properties.rpt]
    set nets [get_nets -quiet -of_objects $violations]
    if {![llength $nets]} {error "DRC finding did not expose its nets"}
    set f [open [file join $output boundary_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\troute_status\tdrivers\tloads\tports\tdriver_sites\tload_sites"
    set with_loads 0
    set with_ports 0
    foreach net [lsort $nets] {
        set segments [get_nets -segments $net]
        set drivers [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]
        set loads [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]
        set ports [get_ports -quiet -of_objects $segments]
        set driver_sites {}
        set load_sites {}
        foreach cell [get_cells -quiet -of_objects $drivers] {lappend driver_sites [get_property LOC $cell]}
        foreach cell [get_cells -quiet -of_objects $loads] {lappend load_sites [get_property LOC $cell]}
        if {[llength $loads]} {incr with_loads}
        if {[llength $ports]} {incr with_ports}
        puts $f "$net\t[get_property ROUTE_STATUS $net]\t[join [lsort $drivers] {;}]\t[join [lsort $loads] {;}]\t[join [lsort $ports] {;}]\t[join [lsort -unique $driver_sites] {;}]\t[join [lsort -unique $load_sites] {;}]"
    }
    close $f
    set f [open [file join $output boundary_totals.txt] {WRONLY CREAT EXCL}]
    puts $f "nets=[llength $nets]\nwith_leaf_loads=$with_loads\nwith_ports=$with_ports"
    close $f
    write_xdc -constraints all [file join $output unchanged_constraints.xdc]
    puts "NTRPU_SUPERVISED_BOUNDARY_EXPORTED nets=[llength $nets]; classification pending"
    close_project
}
if {[catch {boundary_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
