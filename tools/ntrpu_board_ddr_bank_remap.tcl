# Relocate the complete generic DDR example by byte/nibble position.
# Usage: -tclargs SYNTHESIZED_DCP NEW_DIRECTORY
# This explores 64-67; it does not silently change the board's reserved 64-66 plan.
proc ntrpu_ddr_bank_remap {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected fresh process and SYNTHESIZED_DCP NEW_DIRECTORY"
    }
    lassign $arguments checkpoint output
    set checkpoint [file normalize $checkpoint]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} { error "Unexpected part" }
    set assignments {}
    foreach port [get_ports] {
        set oldball [get_property PACKAGE_PIN $port]
        if {$oldball eq ""} { error "Unassigned baseline port $port" }
        set oldpin [get_package_pins $oldball]
        set oldbank [get_property BANK $oldpin]
        if {$oldbank ni {44 45 46 47}} { error "Unexpected baseline bank $oldbank" }
        set function [get_property PIN_FUNC $oldpin]
        if {![regexp {_(T[0-3][LU]_N[0-9]+)(_|$)} $function -> position]} {
            error "No byte/nibble coordinate for $oldball"
        }
        set targetbank [expr {$oldbank+20}]
        set matches {}
        foreach pin [get_package_pins -of_objects [get_iobanks $targetbank]] {
            if {[regexp "_${position}(_|$)" [get_property PIN_FUNC $pin]]} { lappend matches $pin }
        }
        if {[llength $matches] != 1} { error "Nonunique target bank $targetbank position $position" }
        set target [lindex $matches 0]
        if {[string match *GC* $function] && ![string match *GC* [get_property PIN_FUNC $target]]} {
            error "GC capability changes for $port"
        }
        lappend assignments [list $port $oldball $oldbank $position [get_property NAME $target] $targetbank]
    }
    if {[llength $assignments] != 154} { error "Expected 154 example ports" }
    reset_property PACKAGE_PIN [get_ports]
    set stream [open [file join $output mapping.tsv] {WRONLY CREAT EXCL}]
    puts $stream "port\told_ball\told_bank\tposition\tnew_ball\tnew_bank"
    foreach row $assignments {
        lassign $row port oldball oldbank position newball newbank
        set_property PACKAGE_PIN $newball [get_ports $port]
        puts $stream [join $row "\t"]
    }
    close $stream
    set stream [open [file join $output perst_candidates.txt] {WRONLY CREAT EXCL}]
    foreach ball {AM17 AP17} {
        set users [get_ports -filter "PACKAGE_PIN == $ball"]
        puts $stream "$ball: users=$users function=[get_property PIN_FUNC [get_package_pins $ball]]"
    }
    close $stream
    if {[llength [get_ports -filter {PACKAGE_PIN == AM17}]] != 0} { error "PERSTN0 is occupied by DDR" }
    report_io -file [file join $output remapped_io.rpt]
    report_drc -file [file join $output remapped_drc.rpt]
    write_xdc [file join $output remapped_candidate.xdc]
    write_checkpoint [file join $output remapped.dcp]
    puts "NTRPU_DDR_REMAP_EXPORTED ports=154 target_banks=64,65,66,67 AM17_unoccupied=1"
    puts "Four-bank alternative, not accepted board pinout. Review all memory DRCs and implement PHY separately."
    close_project
}
if {[catch {ntrpu_ddr_bank_remap $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
