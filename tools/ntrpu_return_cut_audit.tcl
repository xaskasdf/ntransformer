# Read-only timing-cone audit: captured return metadata must stop at allocator registers.
set_param general.maxThreads 2
if {[llength $argv] != 2} {error "Expected checkpoint and output"}
lassign $argv checkpoint output
file mkdir $output
if {[catch {
    open_checkpoint $checkpoint
    set metadata [get_cells -hier -filter {IS_PRIMITIVE == 1 && REF_NAME =~ FD* && (NAME =~ */allocator/pending_generation_reg* || NAME =~ */allocator/pending_bank_reg*)}]
    if {[llength $metadata] == 0} {error "Captured return metadata registers missing"}
    set targets {}
    foreach c [all_registers] {
        if {![string match */allocator/* $c]} {lappend targets $c}
    }
    if {[llength $targets] < 1000} {error "Unexpectedly small outside-register inventory"}
    set paths [get_timing_paths -quiet -from $metadata -to $targets -max_paths 10]
    set f [open [file join $output audit.txt] w]
    puts $f "METADATA_REGISTERS [llength $metadata]"
    puts $f "OUTSIDE_REGISTERS [llength $targets]"
    puts $f "CROSSING_PATHS [llength $paths]"
    puts $f "CLOCK_PERIOD [get_property PERIOD [get_clocks transport_clk]]"
    close $f
    set f [open [file join $output metadata_cells.txt] w]; puts $f [join [lsort $metadata] \n]; close $f
    set f [open [file join $output outside_cells.txt] w]; puts $f [join [lsort $targets] \n]; close $f
    report_timing -from $metadata -to [all_registers] -max_paths 20 -file [file join $output metadata_paths.rpt]
    report_timing -from $metadata -to $targets -max_paths 10 -file [file join $output crossing_paths.rpt]
    report_exceptions -coverage -file [file join $output exceptions.rpt]
    write_xdc -constraints all [file join $output constraints.xdc]
    if {[llength $paths]} {error "Captured metadata still reaches registers outside allocator combinationally"}
    close_design
    puts "NTRPU_RETURN_CUT_AUDIT_COMPLETE"
} message options]} {
    puts stderr $message; puts stderr [dict get $options -errorinfo]; exit 1
}
exit 0
