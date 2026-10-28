# Read a verified routed checkpoint and export leaf-pin connectivity only.
set_param general.maxThreads 2
if {[llength $argv]!=2} {error "Expected checkpoint and output directory"}
set checkpoint [file normalize [lindex $argv 0]]
set output [file normalize [lindex $argv 1]]
file mkdir $output
if {[catch {
    open_checkpoint $checkpoint
    set cells [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1}]]
    set f [open [file join $output cells.tsv] w];puts $f "cell\tref\tloc\tbel"
    foreach c $cells ref [get_property REF_NAME $cells] loc [get_property LOC $cells] bel [get_property BEL $cells] {
        puts $f "$c\t$ref\t$loc\t$bel"
    }
    close $f
    set combinational [lsort [get_cells -hier -filter {IS_PRIMITIVE == 1 && NAME =~ core/core/receiver/core/allocator/* && (REF_NAME =~ LUT* || REF_NAME == CARRY8)}]]
    set pins [lsort [get_pins -of_objects $combinational -filter {DIRECTION == IN}]]
    set f [open [file join $output edges.tsv] w];puts $f "sink\tdriver\tsegments"
    foreach pin $pins {
        set nets [get_nets -quiet -segments -of_objects $pin]
        set drivers [lsort -unique [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]]
        set ports [get_ports -quiet -of_objects $nets -filter {DIRECTION == IN}]
        if {![llength $nets] && ![llength $drivers] && ![llength $ports]} {
            set source "OPEN:$pin"
        } else {
            if {[llength $drivers]+[llength $ports]!=1} {error "Non-unique source for $pin: drivers=$drivers ports=$ports nets=$nets"}
            set source [expr {[llength $drivers] ? [lindex $drivers 0] : "PORT:[lindex $ports 0]"}]
        }
        puts $f "$pin\t$source\t[join [lsort $nets] {;}]"
    }
    close $f
    set f [open [file join $output inventory.txt] w]
    puts $f "PRIMITIVES [llength $cells]\nCOMBINATIONAL [llength $combinational]\nINPUT_PINS [llength $pins]"
    close $f
    close_design
    puts "NTRPU_CONNECTIVITY_COMPLETE"
} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
