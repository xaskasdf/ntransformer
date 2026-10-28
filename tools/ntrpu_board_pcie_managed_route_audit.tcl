# Read-only physical boundary and synchronizer inventory. No constraint changes.
proc sync_inventory {path} {
    set f [open $path {WRONLY CREAT EXCL}]
    puts $f "cell\tref\tinit\tasync_reg\tclocks\td_nets\td_drivers\tq_nets\tq_loads"
    foreach cell [lsort [get_cells -hierarchical -filter {ASYNC_REG == 1}]] {
        set d [get_pins -quiet -of_objects $cell -filter {REF_PIN_NAME == D}]
        set q [get_pins -quiet -of_objects $cell -filter {REF_PIN_NAME == Q}]
        set c [get_pins -quiet -of_objects $cell -filter {REF_PIN_NAME == C}]
        set dn [get_nets -quiet -segments -of_objects $d]
        set qn [get_nets -quiet -segments -of_objects $q]
        set drivers [get_pins -quiet -leaf -of_objects $dn -filter {DIRECTION == OUT}]
        set loads [get_pins -quiet -leaf -of_objects $qn -filter {DIRECTION == IN}]
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]\t[get_property ASYNC_REG $cell]\t[join [get_clocks -quiet -of_objects $c] {;}]\t[join [lsort $dn] {;}]\t[join [lsort $drivers] {;}]\t[join [lsort $qn] {;}]\t[join [lsort $loads] {;}]"
    }
    close $f
}
proc managed_route_audit {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {error "Expected SYNTH_DCP ROUTED_DCP NEW_OUTPUT"}
    lassign $arguments original routed output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    foreach variant {synthesis routed} input [list $original $routed] {
        file copy $input [file join $output ${variant}_input.dcp]
        open_checkpoint [file join $output ${variant}_input.dcp]
        if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_dual_host_cold" ||
            [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 5} {error "Wrong context"}
        sync_inventory [file join $output ${variant}_sync.tsv]
        if {$variant eq "synthesis"} {close_project}
    }
    write_xdc -constraints all [file join $output before.xdc]
    set routes [dict create]
    set f [open [file join $output parent_clocks.tsv] {WRONLY CREAT EXCL}]
    puts $f "clock\tport\tnet\tassumed_site\troute_status\troute"
    foreach {clock port site} {
        source_clk source_clk BUFGCE_X0Y60 host_clk host_clk BUFGCE_X0Y61
        ui0_clk {ui_clk[0]} BUFGCE_X0Y62 ui1_clk {ui_clk[1]} BUFGCE_X0Y63
        mgmt_clk mgmt_clk BUFGCE_X0Y64
    } {
        set object [get_ports $port]
        if {[get_property HD.CLK_SRC $object] ne $site} {error "Changed parent context"}
        set net [get_nets -of_objects $object]
        if {[llength $net] != 1} {error "Nonunique clock net"}
        set route [get_property ROUTE $net]
        if {![string match -nocase *gap* $route]} {error "Missing expected parent gap: $clock"}
        dict set routes $clock $route
        puts $f "$clock\t$port\t$net\t$site\t[get_property ROUTE_STATUS $net]\t$route"
    }
    close $f
    set f [open [file join $output partial_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus\tmatching_parent"
    set counts [dict create]
    set unexplained {}
    set nets [get_nets -hierarchical]
    foreach net $nets status [get_property ROUTE_STATUS $nets] {
        dict incr counts $status
        if {$status ne "PARTIAL"} {continue}
        set matches {}
        set route [get_property ROUTE $net]
        dict for {clock canonical} $routes {if {$route eq $canonical} {lappend matches $clock}}
        puts $f "$net\t$status\t[join $matches {;}]"
        if {[llength $matches] != 1} {lappend unexplained $net}
    }
    close $f
    set f [open [file join $output route_counts.txt] {WRONLY CREAT EXCL}]
    puts $f "ROUTE_STATUS_COUNTS $counts\nUNEXPLAINED_PARTIAL $unexplained"
    close $f
    report_drc -checks [get_drc_checks RTSTAT-10] -file [file join $output boundary_drc.rpt]
    set violations [get_drc_violations -quiet *RTSTAT-10*]
    if {[llength $violations] != 1} {error "Expected one RTSTAT-10 finding"}
    set boundary [get_nets -quiet -of_objects $violations]
    if {![llength $boundary]} {error "No boundary net inventory"}
    set f [open [file join $output boundary_nets.tsv] {WRONLY CREAT EXCL}]
    puts $f "net\tstatus\tdrivers\tloads\tports"
    foreach net [lsort $boundary] {
        set segments [get_nets -segments $net]
        set drivers [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == OUT}]
        set loads [get_pins -quiet -leaf -of_objects $segments -filter {DIRECTION == IN}]
        set ports [get_ports -quiet -of_objects $segments]
        puts $f "$net\t[get_property ROUTE_STATUS $net]\t[join [lsort $drivers] {;}]\t[join [lsort $loads] {;}]\t[join [lsort $ports] {;}]"
    }
    close $f
    report_route_status -file [file join $output route.rpt]
    report_cdc -details -file [file join $output cdc.rpt]
    report_exceptions -ignored -file [file join $output ignored_exceptions.rpt]
    write_xdc -constraints all [file join $output after.xdc]
    puts "NTRPU_MANAGED_ROUTE_AUDIT_EXPORTED boundary=[llength $boundary] unexplained_partial=[llength $unexplained]"
    if {[llength $unexplained]} {error "Partial route not attributable to assumed parent clock"}
    close_project
}
if {[catch {managed_route_audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
