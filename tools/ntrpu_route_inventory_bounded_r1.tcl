# Export every net and its original ROUTE_STATUS/ROUTE gap check in bounded batches.
# This helper does not open a design, run implementation, or change a checkpoint.
proc ntrpu_route_inventory_bounded_r1 {path {batch_size 64}} {
    if {![string is integer -strict $batch_size] || $batch_size < 1 || $batch_size > 4096} {
        error "Invalid route inventory batch size"
    }
    set nets [get_nets -hier]
    set stream [open $path {WRONLY CREAT EXCL}]
    set seen [dict create]
    set count 0
    try {
        puts $stream "net\tstatus\tgap"
        for {set first 0} {$first < [llength $nets]} {incr first $batch_size} {
            set batch [lrange $nets $first [expr {$first + $batch_size - 1}]]
            set statuses [get_property ROUTE_STATUS $batch]
            set routes [get_property ROUTE $batch]
            if {[llength $statuses] != [llength $batch] || [llength $routes] != [llength $batch]} {
                error "Incomplete route properties at net $first"
            }
            foreach net $batch status $statuses route $routes {
                if {$net eq "" || [regexp {[\t\r\n]} $net] ||
                    $status eq "" || [regexp {[\t\r\n]} $status] || [dict exists $seen $net]} {
                    error "Invalid or duplicate route inventory row at net $count"
                }
                dict set seen $net 1
                puts $stream "$net\t$status\t[expr {[string match -nocase *gap* $route] ? 1 : 0}]"
                incr count
            }
            # Release expanded routing strings before querying the next batch.
            unset statuses routes route
            flush $stream
        }
    } finally {
        close $stream
    }
    return $count
}
