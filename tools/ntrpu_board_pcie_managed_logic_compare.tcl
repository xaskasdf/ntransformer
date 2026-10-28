# Deterministic, read-only exports from original and constraint-only checkpoints.
proc compare_exports {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected ORIGINAL_DCP CONSTRAINED_DCP NEW_OUTPUT"
    }
    lassign $arguments original constrained output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    set summary [open [file join $output inventory.tsv] {WRONLY CREAT EXCL}]
    puts $summary "variant\tprimitive_cells\tproperties\tnets\tports\tclocks"
    foreach variant {original constrained} checkpoint [list $original $constrained] {
        file copy $checkpoint [file join $output ${variant}_input.dcp]
        open_checkpoint [file join $output ${variant}_input.dcp]
        if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_dual_host_cold" ||
            [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 5} {
            error "Unexpected checkpoint context"
        }
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
            error "Unresolved blackbox prevents structural comparison"
        }
        set cells [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE == 1}]]
        set f [open [file join $output ${variant}_cells.txt] {WRONLY CREAT EXCL}]
        set n 0
        set props 0
        foreach cell $cells {
            foreach property [lsort [list_property $cell]] {
                puts $f [list $cell $property [get_property $property $cell]]
                incr props
            }
            incr n
            if {$n % 5000 == 0} {puts "NTRPU_LOGIC_EXPORT variant=$variant cells=$n properties=$props"}
        }
        close $f
        set nets [lsort [get_nets -hierarchical]]
        set f [open [file join $output ${variant}_nets.txt] {WRONLY CREAT EXCL}]
        set nn 0
        foreach net $nets {
            set pins [lsort [get_pins -quiet -of_objects $net]]
            set ports [lsort [get_ports -quiet -of_objects $net]]
            puts $f [list $net $pins $ports]
            incr nn
            if {$nn % 20000 == 0} {puts "NTRPU_LOGIC_EXPORT variant=$variant nets=$nn"}
        }
        close $f
        set ports [lsort [get_ports]]
        set f [open [file join $output ${variant}_ports.txt] {WRONLY CREAT EXCL}]
        foreach port $ports {
            foreach property [lsort [list_property $port]] {
                puts $f [list $port $property [get_property $property $port]]
            }
        }
        close $f
        write_edif [file join $output ${variant}.edf]
        write_xdc -constraints all [file join $output ${variant}.xdc]
        puts $summary "$variant\t$n\t$props\t$nn\t[llength $ports]\t[llength [get_clocks]]"
        puts "NTRPU_LOGIC_EXPORT_COMPLETE variant=$variant cells=$n properties=$props nets=$nn"
        close_project
    }
    close $summary
    puts NTRPU_MANAGED_LOGIC_COMPARISON_EXPORTED
}
if {[catch {compare_exports $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
