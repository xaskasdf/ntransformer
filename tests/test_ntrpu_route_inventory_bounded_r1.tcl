# Native Tcl contract test with deterministic Vivado command doubles; no FPGA claim.
if {[llength $argv] != 2} {error "Usage: test helper_path output_dir"}
source [lindex $argv 0]
set output [lindex $argv 1]
file mkdir $output
set nets {}
set properties [dict create]
for {set i 0} {$i < 2051} {incr i} {
    set net "root/branch\[$i\]/net"
    lappend nets $net
    set status [expr {$i % 3 == 0 ? "INTRASITE" : "ROUTED"}]
    set route [list TILE_X0Y0 [list "WIRE_$i" [expr {$i % 47 == 0 ? "GaP" : "WIRE_OK"}]]]
    dict set properties $net [dict create ROUTE_STATUS $status ROUTE $route]
}
set mode normal
set queries {}
proc get_nets {option} {
    if {$option ne "-hier"} {error "Wrong query"}
    return $::nets
}
proc get_property {property batch} {
    lappend ::queries [list $property [llength $batch]]
    if {$::mode eq "throw" && [llength $::queries] == 4} {error "Injected native property failure"}
    set values {}
    foreach n $batch {lappend values [dict get $::properties $n $property]}
    if {$::mode eq "short_$property"} {return [lrange $values 0 end-1]}
    if {$::mode eq "empty_status" && $property eq "ROUTE_STATUS"} {lset values 0 ""}
    return $values
}
proc content {path} {
    set f [open $path r]
    try {return [read $f]} finally {close $f}
}
proc expect_error {script pattern} {
    if {![catch {uplevel 1 $script} reason]} {error "Expected rejection: $script"}
    if {![string match $pattern $reason]} {error "Unexpected rejection: $reason"}
    incr ::rejected
}
set rejected 0
# Independent scalar reference; exact row order, names, nested routes, mixed-case gaps.
set expected "net\tstatus\tgap\n"
foreach net $nets {
    set p [dict get $properties $net]
    append expected "$net\t[dict get $p ROUTE_STATUS]\t[expr {[string match -nocase *gap* [dict get $p ROUTE]] ? 1 : 0}]\n"
}
foreach size {1 7 64 4096} {
    set queries {}
    set path [file join $output "batch_$size.tsv"]
    set count [ntrpu_route_inventory_bounded_r1 $path $size]
    if {$count != 2051 || [content $path] ne $expected} {error "Output mismatch for batch $size"}
    foreach q $queries {
        if {[lindex $q 1] > $size} {error "Unbounded property call"}
    }
    if {[llength $queries] != 2 * int(ceil(2051.0 / $size))} {error "Missing or redundant property calls"}
}
set existing [file join $output batch_64.tsv]
expect_error {ntrpu_route_inventory_bounded_r1 $existing} {*file already exists*}
if {[content $existing] ne $expected} {error "Existing evidence changed"}
foreach size {0 -1 4097 2.5 text} {
    expect_error {ntrpu_route_inventory_bounded_r1 [file join $output invalid.tsv] $size} {Invalid route inventory batch size}
}
foreach mode {short_ROUTE_STATUS short_ROUTE empty_status throw} {
    set queries {}
    set path [file join $output "$mode.tsv"]
    set pattern [expr {$mode eq "throw" ? "Injected native property failure" : ($mode eq "empty_status" ? "Invalid or duplicate*" : "Incomplete route properties*")}]
    expect_error {ntrpu_route_inventory_bounded_r1 $path 64} $pattern
    # A failed export must close the file; retained partial evidence stays inspectable.
    if {![file exists $path]} {error "Partial diagnostic not retained"}
}
set mode normal
lappend nets [lindex $nets 0]
expect_error {ntrpu_route_inventory_bounded_r1 [file join $output duplicate.tsv] 64} {Invalid or duplicate*}
if {$rejected != 11} {error "Missing negative coverage"}
puts "ROUTE_INVENTORY_HELPER_PASS nets=2051 sizes=4 rejected=$rejected actual_vivado_execution=0"
