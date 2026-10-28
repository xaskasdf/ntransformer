# Read-only discovery of installed memory-planning commands before pin assignment.
# Usage: -tclargs NEW_DIRECTORY
if {[llength $argv] != 1 || [llength [get_projects -quiet]]} {
    error "Expected a fresh process and NEW_DIRECTORY"
}
set output [file normalize [lindex $argv 0]]
if {[file exists $output]} { error "Output already exists" }
file mkdir $output
# Help output is retained in the stage's console.log and vivado.log.
puts [help help]
set stream [open [file join $output commands.txt] {WRONLY CREAT EXCL}]
set commands [lsort -unique [concat [info commands *mem*] [info commands *byte*] [info commands *io*]]]
foreach command $commands { puts $stream $command }
close $stream
foreach command {synth_design place_ports get_pkgpin_bytegroups get_iobanks} {
    set stream [open [file join $output ${command}_help.txt] {WRONLY CREAT EXCL}]
    puts $stream [help $command]
    close $stream
}
puts "NTRPU_MEMORY_PLANNER_COMMANDS_EXPORTED: discovery only; no placement or DRC approval"
exit 0
