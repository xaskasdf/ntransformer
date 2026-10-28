# Bounded logical exports; EDIF carries primitive parameters and connections.
# No exhaustive queries for implementation-only HD properties.
proc export_logic {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Expected ORIGINAL_DCP CONSTRAINED_DCP NEW_OUTPUT"
    }
    lassign $arguments original constrained output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 4
    set summary [open [file join $output inventory.tsv] {WRONLY CREAT EXCL}]
    puts $summary "variant\tprimitive_cells\tnets\tports\tclocks"
    foreach variant {original constrained} checkpoint [list $original $constrained] {
        file copy $checkpoint [file join $output ${variant}_input.dcp]
        open_checkpoint [file join $output ${variant}_input.dcp]
        if {[get_property TOP [current_design]] ne "ntrpu_pcie_fragment_guarded_host" ||
            [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 3} {
            error "Unexpected checkpoint context"
        }
        if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1}]]} {
            error "Unresolved blackbox prevents comparison"
        }
        set cells [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE == 1}]]
        set f [open [file join $output ${variant}_primitive_identity.txt] {WRONLY CREAT EXCL}]
        foreach cell $cells {puts $f [list $cell [get_property REF_NAME $cell]]}
        close $f
        # EDIF writes protected .edn companions. Isolate each export so the
        # second variant cannot replace the first variant's protected cells.
        set variant_output [file join $output $variant]
        file mkdir $variant_output
        write_edif [file join $variant_output design.edf]
        write_xdc -constraints all [file join $output ${variant}.xdc]
        puts $summary "$variant\t[llength $cells]\t[llength [get_nets -hierarchical]]\t[llength [get_ports]]\t[llength [get_clocks]]"
        puts "NTRPU_LOGIC_EDIF_EXPORT variant=$variant primitive_cells=[llength $cells]"
        close_project
    }
    close $summary
    puts NTRPU_GUARDED_OWNER_LOGIC_EDIF_R2_EXPORTED
}
if {[catch {export_logic $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
