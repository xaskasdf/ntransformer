# Audit actual routed reset synchronizer connectivity without changing constraints.
proc ntrpu_reset_structure {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected ROUTED_DCP NEW_DIRECTORY in a fresh process"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    set regs [get_cells -hier -filter {NAME =~ */sequencer/reset_release_reg* && IS_PRIMITIVE}]
    if {[llength $regs] != 2} { error "Expected two reset-release registers: $regs" }
    set pres [get_pins -of_objects $regs -filter {REF_PIN_NAME == PRE}]
    if {[llength $pres] != 2} { error "Expected two preset pins" }
    foreach reg $regs {
        if {[get_property REF_NAME $reg] ne "FDPE" || ![get_property ASYNC_REG $reg]} {
            error "Reset release must be a preset-based ASYNC_REG chain"
        }
        report_property -all -file [file join $output [string map {/ _ \[ _ \] _} $reg].rpt] $reg
    }
    set nets [get_nets -segments -of_objects $pres]
    set loads [get_pins -leaf -of_objects $nets -filter {DIRECTION == IN}]
    if {[lsort [get_property NAME $loads]] ne [lsort [get_property NAME $pres]]} {
        error "Combined asynchronous reset drives unexpected leaf loads: $loads"
    }
    set driverpins [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]
    set drivers [get_cells -of_objects $driverpins]
    if {[llength $drivers] != 1 || [get_property REF_NAME $drivers] ne "LUT4"} {
        error "Expected single LUT4 reset combiner: $drivers"
    }
    report_property -all -file [file join $output combiner.rpt] $drivers
    set stream [open [file join $output structure.txt] {WRONLY CREAT EXCL}]
    puts $stream "combiner=$drivers\nasync_loads=$loads\nreset_registers=$regs"
    foreach reg $regs {
        puts $stream "register=$reg site=[get_property LOC $reg] bel=[get_property BEL $reg]"
        foreach pin [get_pins -of_objects $reg] {
            puts $stream "pin=$pin net=[get_nets -of_objects $pin]"
        }
    }
    foreach pin [get_pins -of_objects $drivers -filter {DIRECTION == IN}] {
        puts $stream "combiner_input=$pin net=[get_nets -of_objects $pin]"
    }
    close $stream
    report_methodology -file [file join $output methodology.rpt]
    report_timing -delay_type min_max -to $pres -max_paths 10 -file [file join $output reset_timing.rpt]
    if {[llength [info commands report_synchronizer_mtbf]]} {
        help report_synchronizer_mtbf
        report_synchronizer_mtbf -file [file join $output synchronizer_mtbf.rpt]
    } else {
        puts "NTRPU_RESET_MTBF_COMMAND_UNAVAILABLE"
    }
    puts "NTRPU_RESET_STRUCTURE_EXPORTED; no analog glitch or MTBF approval inferred"
    close_project
}
if {[catch {ntrpu_reset_structure $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
