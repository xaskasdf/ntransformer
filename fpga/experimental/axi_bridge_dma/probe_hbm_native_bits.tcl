# Diagnostic only: retain the actual constant/driving nets and functional netlist.
set_param general.maxThreads 2
if {[catch {
    lassign $argv checkpoint out
    if {[llength $argv]!=2 || [file exists $out]} {error "CHECKPOINT NEW_OUTPUT required"}
    file mkdir $out
    open_checkpoint $checkpoint
    set f [open $out/bits.tsv {WRONLY CREAT EXCL}]
    puts $f "pin\tdirection\tcell_reference\tnets\tdrivers\tdriver_references"
    foreach p [lsort [get_pins -hier -filter {REF_PIN_NAME =~ *awsize* || REF_PIN_NAME =~ *bresp*}]] {
        if {![string match core/* $p] && ![string match ntrpu_bridge_i/* $p]} {continue}
        set nets [get_nets -quiet -segments -of_objects [get_pins $p]]
        set ds {};set refs {}
        if {[llength $nets]} {
            set ds [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION == OUT}]
            foreach d $ds {lappend refs [get_property REF_NAME [get_cells -of_objects $d]]}
        }
        puts $f "$p\t[get_property DIRECTION [get_pins $p]]\t[get_property REF_NAME [get_cells -of_objects [get_pins $p]]]\t$nets\t$ds\t$refs"
    };close $f
    write_verilog -mode funcsim $out/netlist.v
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f NATIVE_BITS_DIAGNOSTIC_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
