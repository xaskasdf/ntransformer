# Run in a fresh Vivado batch process. Creates only an isolated I/O study.
# Usage: -tclargs discover NEW_DIRECTORY
#        -tclargs EXACT_PART NEW_DIRECTORY
# This script does not generate MIG/PCIe, run synthesis, or validate a board.

proc ntrpu_dump_objects {path objects} {
    set stream [open $path {WRONLY CREAT EXCL}]
    fconfigure $stream -encoding utf-8 -translation lf
    try {
        foreach object $objects {
            puts $stream [report_property -all -return_string $object]
        }
    } finally {
        close $stream
    }
}

proc ntrpu_inventory {arguments} {
    if {[llength $arguments] != 2} {
        error "Usage: -tclargs discover|EXACT_PART NEW_DIRECTORY"
    }
    if {[llength [get_projects -quiet]] != 0} {
        error "Use a fresh batch process; an existing project is open"
    }
    lassign $arguments selection output
    set output [file normalize $output]
    if {[file exists $output]} {
        error "Output already exists; choose a new directory: $output"
    }

    set candidates [get_parts -quiet *xcvu5p*flva2104*]
    set other_candidates [get_parts -quiet *xcvu9p*flga2104*]
    # Copy names into an ordinary Tcl list rather than mixing Vivado containers.
    set names {}
    foreach part $candidates { lappend names [get_property NAME $part] }
    foreach part $other_candidates { lappend names [get_property NAME $part] }
    set names [lsort -unique $names]
    if {[llength $names] == 0} {
        error "No candidate parts installed; check Virtex UltraScale+ device support"
    }
    if {$selection ne "discover" && [lsearch -exact $names $selection] < 0} {
        error "Select an exact part from discover output; no automatic speed-grade choice"
    }

    file mkdir $output
    set stream [open [file join $output available_parts.txt] {WRONLY CREAT EXCL}]
    fconfigure $stream -encoding utf-8 -translation lf
    puts $stream [version]
    puts $stream [join $names \n]
    close $stream
    if {$selection eq "discover"} {
        puts "Candidate parts written to $output; no project created"
        return
    }

    create_project ntrpu_board_io [file join $output project] -part $selection
    set_property DESIGN_MODE PinPlanning [current_fileset]
    open_io_design -name ntrpu_io
    set pins [get_package_pins]
    if {[llength $pins] == 0} {
        error "No package pin objects returned; inventory incomplete"
    }
    set pin_count [llength $pins]
    ntrpu_dump_objects [file join $output part.rpt] [get_parts $selection]
    ntrpu_dump_objects [file join $output package_pins.rpt] $pins
    ntrpu_dump_objects [file join $output iobanks.rpt] [get_iobanks]
    ntrpu_dump_objects [file join $output slrs.rpt] [get_slrs]
    set stream [open [file join $output ip_catalog.txt] {WRONLY CREAT EXCL}]
    fconfigure $stream -encoding utf-8 -translation lf
    foreach pattern {xilinx.com:ip:ddr4:* xilinx.com:ip:pcie4_uscale_plus:* xilinx.com:ip:xdma:*} {
        puts $stream "$pattern: [get_ipdefs -all -quiet $pattern]"
    }
    close $stream
    close_project
    set stream [open [file join $output INVENTORY_COMPLETE.txt] {WRONLY CREAT EXCL}]
    puts $stream "Part: $selection"
    puts $stream "Package pin objects: $pin_count"
    puts $stream "Inventory only. MIG/PCIe coexistence, DRC and timing remain unvalidated."
    close $stream
}

if {[catch {ntrpu_inventory $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {
        puts stderr [dict get $options -errorinfo]
    }
    exit 1
}
exit 0
