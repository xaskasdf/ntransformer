# Read-only evidence for optimization of the registered readiness reset.
set_param general.maxThreads 2
proc bip_properties {f object} {
    foreach property [lsort [list_property $object]] {
        puts $f "$object\t$property\t[get_property $property $object]"
    }
}
if {[catch {
    lassign $argv checkpoint out
    if {[llength $argv]!=2 || [file exists $out]} {error "CHECKPOINT NEW_OUTPUT required"}
    file mkdir $out;open_checkpoint $checkpoint
    write_xdc -constraints all $out/before.xdc
    set f [open $out/reset_properties.tsv {WRONLY CREAT EXCL}]
    puts $f "object\tproperty\tvalue"
    bip_properties $f [get_cells loader_readiness/ready_reg]
    foreach name {loader_readiness/ready_reg/CLR core/management_cold_rst_n loader_readiness/cold_rst_n} {
        set pin [get_pins $name]
        puts $f "$pin\tFANIN_STARTS\t[all_fanin -flat -startpoints_only -to $pin]"
        set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $pin] -filter {DIRECTION == OUT}]
        puts $f "$pin\tLEAF_DRIVERS\t$drivers"
        foreach driver $drivers {
            bip_properties $f [get_pins $driver]
            set cell [get_cells -of_objects $driver];bip_properties $f $cell
            foreach input [get_pins -of_objects $cell -filter {DIRECTION == IN}] {
                puts $f "$input\tLEAF_DRIVERS\t[get_pins -leaf -of_objects [get_nets -segments -of_objects $input] -filter {DIRECTION == OUT}]"
                bip_properties $f $input
            }
        }
    }
    close $f
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f BOARD_PHYSICAL_RESET_PROBED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
