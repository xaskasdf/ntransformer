# Audit the synthesized board-owned reference tree without modifying constraints.
proc ro_one {items label} {
    if {[llength $items] != 1} {error "Expected exactly one $label: $items"}
    return [lindex $items 0]
}
proc ro_driver {pin} {
    return [ro_one [get_pins -leaf -of_objects [get_nets -segments -of_objects $pin] -filter {DIRECTION == OUT}] "driver of $pin"]
}
proc ro_audit {out} {
    set f [open $out/reference_tree.tsv {WRONLY CREAT EXCL}]
    puts $f "stack\tpad\tinput_buffer\tinput_leaf_driver\tglobal_buffer\tip_pin\tclock\tperiod_ns\tclock_source\tleaf_loads"
    foreach i {0 1} {
        set pad hbm_ref${i}_clk_p
        set neg hbm_ref${i}_clk_n
        set ib board_inputs/hbm_reference${i}_input
        set gb hbm_reference${i}
        set hp hbm_binding/memory/HBM_REF_CLK_${i}
        set clock hbm_reference${i}_clk
        foreach {cell ref} [list $ib IBUFDS $gb BUFGCE] {
            if {[get_property REF_NAME [ro_one [get_cells -quiet $cell] $cell]] ne $ref} {
                error "Wrong primitive for $cell"
            }
        }
        foreach {pin port} [list $ib/I $pad $ib/IB $neg] {
            set ports [get_ports -quiet -of_objects [get_nets -segments -of_objects [get_pins $pin]]]
            if {$ports ne $port} {error "Wrong input pad for $pin: $ports"}
        }
        # Vivado expands IBUFDS into DIFFINBUF + IBUFCTRL. The wrapper's O is
        # not a leaf pin; require the exact expanded driver at both endpoints.
        set input_leaf $ib/IBUFCTRL_INST/O
        if {[get_property REF_NAME [get_cells $ib/IBUFCTRL_INST]] ne "IBUFCTRL" ||
            [ro_driver [get_pins $ib/O]] ne $input_leaf ||
            [ro_driver [get_pins $gb/I]] ne $input_leaf} {error "Reference input-buffer driver differs"}
        set endpoint [ro_one [get_pins -quiet $hp] $hp]
        if {[ro_driver $endpoint] ne "$gb/O"} {error "HBM reference driver differs"}
        set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $gb/O]] -filter {DIRECTION == IN}]
        if {![llength $loads]} {error "HBM reference has no leaf loads"}
        foreach load $loads {
            set ref [get_property REF_NAME [get_cells -of_objects $load]]
            if {[string match *BUF* $ref]} {error "Cascaded buffer at $load"}
        }
        foreach pin [concat [list $endpoint] $loads] {
            if {[get_clocks -quiet -of_objects $pin] ne $clock} {error "Clock redefined or missing at $pin"}
        }
        set c [ro_one [get_clocks -quiet $clock] $clock]
        if {[get_property IS_GENERATED $c] || abs([get_property PERIOD $c]-10.0)>0.001 ||
            [get_property SOURCE_PINS $c] ne $pad} {error "Wrong reference primary clock $clock"}
        puts $f [join [list $i $pad $ib $input_leaf $gb $hp $clock [get_property PERIOD $c] [get_property SOURCE_PINS $c] [join [lsort $loads] |]] "\t"]
    };close $f
    set hidden [get_cells -hier -quiet -filter {NAME =~ hbm_binding/* && (REF_NAME =~ BUFG* || REF_NAME =~ IBUF*)}]
    if {[llength $hidden]} {error "Native HBM still owns input/global reference buffers: $hidden"}
    set f [open $out/reference_clock_inventory.tsv {WRONLY CREAT EXCL}]
    puts $f "clock\tperiod_ns\tgenerated\tmaster\tsources"
    foreach clock [lsort [get_clocks]] {
        set c [get_clocks $clock]
        puts $f [join [list $clock [get_property PERIOD $c] [get_property IS_GENERATED $c] [get_property MASTER_CLOCK $c] [get_property SOURCE_PINS $c]] "\t"]
    };close $f
    report_methodology -file $out/reference_methodology.rpt
    report_clock_utilization -file $out/reference_clock_utilization.rpt
    report_waivers -file $out/reference_waivers.rpt
    set f [open $out/REFERENCE_STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_REFERENCE_OWNERSHIP_SYNTHESIZED_REVIEW_REQUIRED
    close $f
}
