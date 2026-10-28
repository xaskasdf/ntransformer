# Read-only probe of the fresh synthesis reset representation.
set_param general.maxThreads 2
if {[catch {
    lassign $argv checkpoint out
    if {[file exists $out]} {error "Fresh reset probe required"}
    file mkdir $out
    open_checkpoint $checkpoint
    set f [open $out/reset_probe.tsv {WRONLY CREAT EXCL}]
    puts $f "object\tproperty\tvalue"
    set pending [list core/management_cold_rst_n loader_readiness/ready_reg/CLR]
    set seen {}
    for {set depth 0} {$depth<5} {incr depth} {
        set next {}
        foreach name $pending {
            if {$name in $seen} {continue}
            lappend seen $name
            set pin [get_pins $name]
            if {[llength $pin]!=1} {error "Nonunique reset probe pin $name"}
            foreach prop {DIRECTION REF_PIN_NAME IS_INVERTED} {
                puts $f "$name\t$prop\t[get_property $prop $pin]"
            }
            set nets [get_nets -of_objects $pin]
            set drivers [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]
            puts $f "$name\tDRIVERS\t$drivers"
            foreach driver $drivers {
                set cell [get_cells -of_objects $driver]
                foreach prop {REF_NAME INIT IS_CLR_INVERTED ASYNC_REG} {
                    if {$prop in [list_property $cell]} {puts $f "$cell\t$prop\t[get_property $prop $cell]"}
                }
                set ref [get_property REF_NAME $cell]
                if {[string match LUT* $ref]} {
                    foreach input [get_pins -of_objects $cell -filter {DIRECTION == IN}] {lappend next $input}
                }
            }
        }
        set pending $next
    }
    set cell [get_cells loader_readiness/ready_reg]
    foreach prop {REF_NAME INIT IS_CLR_INVERTED ASYNC_REG} {
        puts $f "$cell\t$prop\t[get_property $prop $cell]"
    }
    close $f
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f CAPTURE_RESET_PROBE_EXPORTED;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
