# Query the frozen routed reset tree; no netlist or constraint edits.
set_param general.maxThreads 2
proc rf_one {objects label} {
    if {[llength $objects]!=1} {error "Expected one $label, got [llength $objects]"}
    return [lindex $objects 0]
}
proc rf_run {checkpoint out} {
    if {[file exists $out]} {error "Fresh reset probe output required"}
    file mkdir $out
    open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
        [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong board checkpoint"}
    write_xdc -constraints all $out/before.xdc
    set name {core/reader/startup/memory_distribution_reg}
    set cell [rf_one [get_cells $name] reset_register]
    set f [open $out/source.tsv {WRONLY CREAT EXCL}];puts $f "object\tproperty\tvalue"
    foreach k {REF_NAME LOC BEL INIT ASYNC_REG DONT_TOUCH IS_CLR_INVERTED} {puts $f "$cell\t$k\t[get_property $k $cell]"}
    foreach pin {C CE CLR D Q} {
        set p [rf_one [get_pins $name/$pin] register_pin]
        set nets [get_nets -segments -of_objects $p]
        puts $f "$p\tDRIVERS\t[lsort -unique [get_pins -leaf -of_objects $nets -filter {DIRECTION == OUT}]]"
    };close $f
    set nets [get_nets -segments -of_objects [get_pins $name/Q]]
    set loads [lsort -unique [get_pins -leaf -of_objects $nets -filter {DIRECTION == IN}]]
    set f [open $out/loads.tsv {WRONLY CREAT EXCL}]
    puts $f "pin\tcell\treference\tlocation\tslr\tclocks\tasync_reg\tdont_touch"
    foreach pin $loads {
        set cell [rf_one [get_cells -of_objects [get_pins $pin]] load_cell]
        set clk [get_clocks -quiet -of_objects [get_pins -of_objects $cell -filter {IS_CLOCK == 1}]]
        puts $f "$pin\t$cell\t[get_property REF_NAME $cell]\t[get_property LOC $cell]\t[get_slrs -of_objects $cell]\t$clk\t[get_property ASYNC_REG $cell]\t[get_property DONT_TOUCH $cell]"
    };close $f
    set paths [get_timing_paths -from [get_pins $name/C] -max_paths 512 -nworst 1 -delay_type max]
    set f [open $out/worst_paths.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\tgroup\tclock\trequirement_ns\tslack_ns\tdatapath_ns"
    foreach p $paths {
        puts $f "[get_property STARTPOINT_PIN $p]\t[get_property ENDPOINT_PIN $p]\t[get_property GROUP $p]\t[get_property ENDPOINT_CLOCK $p]\t[get_property REQUIREMENT $p]\t[get_property SLACK $p]\t[get_property DATAPATH_DELAY $p]"
    };close $f
    report_timing -from [get_pins $name/C] -max_paths 8 -nworst 1 -delay_type max -path_type full_clock_expanded -input_pins -file $out/reset_timing.rpt
    rb_probe $out
    help phys_opt_design
    report_timing_summary -delay_type min_max -file $out/timing.rpt
    report_route_status -file $out/route_status.rpt
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_RESET_BREADY_PROBE_EXPORTED;close $f
    close_design
}


proc rb_driver {pin} {
    return [rf_one [get_pins -leaf -of_objects [get_nets -of_objects [rf_one [get_pins $pin] pin]] -filter {DIRECTION == OUT}] driver]
}
proc rb_paths {out label from to kind} {
    set args [list -to [get_pins $to] -delay_type $kind -max_paths 1 -nworst 1]
    if {$from ne ""} {lappend args -from [get_pins $from]}
    set path [rf_one [get_timing_paths {*}$args] timing_path]
    set fields {STARTPOINT_PIN ENDPOINT_PIN STARTPOINT_CLOCK ENDPOINT_CLOCK GROUP DELAY_TYPE SLACK REQUIREMENT DATAPATH_DELAY SKEW LOGIC_LEVELS EXCEPTION}
    set f [open $out/${label}.tsv {WRONLY CREAT EXCL}];puts $f [join $fields "\t"]
    set row {};foreach k $fields {lappend row [get_property $k $path]};puts $f [join $row "\t"];close $f
    report_timing -of_objects $path -file $out/${label}.rpt
}
proc rb_probe {out} {
    set f [open $out/bready.tsv {WRONLY CREAT EXCL}];puts $f "channel\tobject\tproperty\tvalue"
    set dist core/reader/startup/memory_distribution_reg
    for {set ch 0} {$ch<2} {incr ch} {
        set loader [format {weight_loader/h_bready[%d]} $ch]
        set axi [format {hbm_binding/memory/AXI_%02d_BREADY} [expr {$ch*16}]]
        set hbm [format {hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<%d>_INST/HBM_SNGLBLI_INTF_AXI<0>_INST} $ch]
        foreach pin [list $loader $axi $hbm/BREADY] {
            set driver [rb_driver $pin];set cell [rf_one [get_cells -of_objects [get_pins $driver]] driver_cell]
            puts $f "$ch\t$pin\tDRIVER\t$driver"
            foreach k {REF_NAME INIT LOC BEL ASYNC_REG DONT_TOUCH} {puts $f "$ch\t$cell\t$k\t[get_property $k $cell]"}
            puts $f "$ch\t$pin\tSTARTPOINTS\t[join [lsort [all_fanin -flat -startpoints_only -to [get_pins $pin]]] ,]"
            foreach input [lsort [get_pins -of_objects $cell -filter {DIRECTION == IN}]] {
                puts $f "$ch\t$input\tDRIVER\t[rb_driver $input]"
            }
        }
        set clear [format {core/reader/channels[%d].adapter/tracker/validated_cut/valid_reg_reg/CLR} $ch]
        foreach kind {max min} {
            rb_paths $out ch${ch}_bready_$kind "" $hbm/BREADY $kind
            rb_paths $out ch${ch}_validated_clear_$kind $dist/C $clear $kind
        }
    };close $f
}
