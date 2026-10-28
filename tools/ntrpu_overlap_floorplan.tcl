# One placement hint, with the established compact RTL and physical flow.
if {[lindex $argv 1] eq "netlist"} {
    source [file join [file dirname [info script]] ntrpu_overlap_physical.tcl]
    exit 0
}
if {[lindex $argv 1] ni {place route}} {error "Expected place, route or netlist"}
proc save_members {path cells} {
    set cells [lsort $cells]
    set f [open $path w];puts $f "cell\tref\tloc\tbel"
    foreach c $cells ref [get_property REF_NAME $cells] loc [get_property LOC $cells] bel [get_property BEL $cells] {
        puts $f "$c\t$ref\t$loc\t$bel"
    }
    close $f
}
proc add_control_region {output} {
    if {[llength [get_pblocks -quiet]]} {error "Unexpected pre-existing Pblock"}
    set allocator [get_cells -hier -filter {IS_PRIMITIVE == 1 && NAME =~ core/core/receiver/core/allocator/* && REF_NAME != GND && REF_NAME != VCC}]
    set generation [get_cells -hier -filter {IS_PRIMITIVE == 1 && NAME =~ core/core/receiver/core/active_generation_reg* && REF_NAME == FDRE}]
    if {[llength $generation]!=128 || [llength $allocator]<1000 || [llength $allocator]>3000} {error "Control selection changed"}
    set cells [lsort -unique [concat $allocator $generation]]
    set pb [create_pblock ntrpu_compact_control]
    resize_pblock $pb -add {SLICE_X54Y260:SLICE_X71Y289}
    set_property IS_SOFT TRUE $pb
    set_property EXCLUDE_PLACEMENT FALSE $pb
    set_property CONTAIN_ROUTING FALSE $pb
    add_cells_to_pblock $pb $cells
    save_members [file join $output selected_before_place.tsv] $cells
    write_xdc -constraints all [file join $output floorplan_before.xdc]
    set f [open [file join $output selection.txt] w]
    puts $f "ALLOCATOR [llength $allocator]\nGENERATION [llength $generation]\nTOTAL [llength $cells]"
    puts $f "COMMAND opt_design; add_control_region; place_design"
    close $f
}
proc report_control_region {output} {
    set pb [get_pblocks ntrpu_compact_control]
    if {[llength $pb]!=1} {error "Control Pblock missing"}
    set f [open [file join $output floorplan.txt] w]
    foreach prop {NAME GRID_RANGES IS_SOFT EXCLUDE_PLACEMENT CONTAIN_ROUTING} {
        puts $f "$prop [get_property $prop $pb]"
    }
    close $f
    save_members [file join $output members_after.tsv] [get_cells -of_objects $pb]
    report_utilization -pblocks $pb -file [file join $output control_utilization.rpt]
}
set f [open [file join [file dirname [info script]] ntrpu_transport_study.tcl] r]
set script [read $f];close $f
set needle {opt_design;place_design}
if {[string first $needle $script]<0} {error "Placement flow changed"}
set script [string map [list $needle {opt_design;add_control_region $output;place_design}] $script]
set needle {        write_checkpoint [file join $output physical_${stage}.dcp]}
if {[string first $needle $script]<0} {error "Checkpoint flow changed"}
set script [string map [list $needle "        report_control_region \$output\n$needle"] $script]
eval $script
