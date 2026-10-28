# Read-only inventory to diagnose synthesized hierarchy pin names.
proc pin_probe {arguments} {
    if {[llength $arguments]!=2} {error "Expected DCP NEW_DIRECTORY"}
    lassign $arguments checkpoint output
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_ffn_guarded_fabric"} {error "Unexpected top"}
    set guard [get_cells -quiet capacity_guard]
    if {[llength $guard]!=1} {error "Missing guard"}
    report_property -all $guard -file [file join $output guard_properties.rpt]
    set pins [get_pins -of_objects $guard]
    set f [open [file join $output guard_pin_inventory.tsv] {WRONLY CREAT EXCL}]
    puts $f "pin\tref_pin_name\tdirection\tnets\tports\tdrivers"
    foreach pin [lsort $pins] {
        set nets [get_nets -segments -of_objects $pin]
        puts $f "$pin\t[get_property -quiet REF_PIN_NAME $pin]\t[get_property DIRECTION $pin]\t$nets\t[get_ports -quiet -of_objects $nets]\t[get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]"
    }
    close $f
    if {[llength $pins]} {report_property -all [lindex $pins 0] -file [file join $output first_pin_properties.rpt]}
    set f [open [file join $output public_ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection"
    foreach port [lsort [get_ports]] {puts $f "$port\t[get_property DIRECTION $port]"}
    close $f
    puts "NTRPU_GUARDED_PIN_INVENTORY pins=[llength $pins]"
    close_project
}
if {[catch {pin_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
