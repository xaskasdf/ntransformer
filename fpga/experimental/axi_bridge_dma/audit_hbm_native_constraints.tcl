# Shared apply/reopen audit of an immutable native donor and a fresh result.
set_param general.maxThreads 2
source [file join [file dirname [info script]] constrain_hbm_native_constraints.tcl]
proc nc_clock_map {out} {
    set ::nc_clocks [dict create]
    set f [open $out/domain_clocks.tsv {WRONLY CREAT EXCL}]
    puts $f "logical\tpin\tclock\tperiod\tdriver\treference"
    foreach {logical port period} {pcie_clk clk 4.0 reader_clk reader_clk 5.0 mgmt_clk mgmt_clk 10.0 mem_clk mem_clk 2.222222} {
        set pin [nc_pin core/$port]
        set clock [nc_one [get_clocks -of_objects $pin] "$port clock"]
        set value [get_property PERIOD [get_clocks $clock]]
        if {abs($value-$period)>0.001} {error "Clock period changed $port: $value"}
        set driver [nc_one [get_pins -leaf -of_objects [get_nets -segments -of_objects $pin] -filter {DIRECTION == OUT}] "$port driver"]
        set ref [get_property REF_NAME [get_cells -of_objects $driver]]
        if {![string match BUFG* $ref]} {error "No native buffer for $port"}
        dict set ::nc_clocks $logical $clock
        puts $f "$logical\t$pin\t$clock\t$value\t$driver\t$ref"
    };close $f
}
proc nc_topology {out} {
    set f [open $out/capture_topology.tsv {WRONLY CREAT EXCL}]
    puts $f "from\tto\tleaf_loads\tdriver_ref"
    foreach group [nc_groups] {
        lassign $group kind name sources destinations bound skew count
        foreach to $destinations {
            set pin [nc_pin $to]
            set drivers [get_pins -leaf -of_objects [get_nets -segments -of_objects $pin] -filter {DIRECTION == OUT}]
            set driver [nc_one $drivers "$to driver"]
            set cell [nc_one [get_cells -of_objects $driver] "$to source cell"]
            if {[lsearch -exact $sources $cell]<0} {error "Unexpected CDC input logic $to"}
            set dest [get_cells -of_objects $pin]
            if {$kind eq "capture"} {
                set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects $driver] -filter {DIRECTION == IN}]
                if {$loads ne $to || [get_property REF_NAME $dest] ne "FDRE"} {error "Capture bypass $to"}
                puts $f "$driver\t$to\t[llength $loads]\t[get_property REF_NAME $cell]"
            } elseif {[get_property ASYNC_REG $dest] ni {1 TRUE true}} {error "Unmarked CDC first stage $to"}
        }
    };close $f
    set private {}
    foreach cell [get_cells -hier -quiet -filter {ASYNC_REG == TRUE}] {
        if {[regexp {^(core/(transport/|endpoint/guard/|memory_watch/|reader/startup/)|availability/)} $cell]} {lappend private $cell}
    }
    set f [open $out/synchronizers.tsv {WRONLY CREAT EXCL}];puts $f "cell\tclock\tasync_reg"
    foreach cell [lsort $private] {puts $f "$cell\t[get_clocks -of_objects [get_pins $cell/C]]\t[get_property ASYNC_REG $cell]"};close $f
    if {[llength $private]!=147} {error "Expected 147 private synchronizer registers, got [llength $private]"}
    set f [open $out/interstage.tsv {WRONLY CREAT EXCL}];puts $f "from\tto\ttype\tstart_clock\tend_clock\trequirement_ns\tslack_ns"
    set pairs 0
    foreach cell $private {
        set loads [get_pins -leaf -of_objects [get_nets -segments -of_objects [get_pins $cell/Q]] -filter {DIRECTION == IN}]
        foreach to $loads {
            set dest [nc_one [get_cells -of_objects $to] destination]
            if {[lsearch -exact $private $dest]<0 || [get_property REF_PIN_NAME $to] ne "D"} {continue}
            set sc [get_clocks -of_objects [get_pins $cell/C]];set dc [get_clocks -of_objects [get_pins $dest/C]]
            if {$sc ne $dc} {continue}
            if {[llength $loads]!=1} {error "Intermediate synchronizer has extra consumers $cell"}
            incr pairs
            foreach type {max min} {
                set path [nc_one [get_timing_paths -quiet -from [get_pins $cell/C] -to $to -delay_type $type -max_paths 1] "$cell $to $type"]
                puts $f "$cell\t$dest\t$type\t[get_property STARTPOINT_CLOCK $path]\t[get_property ENDPOINT_CLOCK $path]\t[get_property REQUIREMENT $path]\t[get_property SLACK $path]"
            }
        }
    };close $f
    if {$pairs!=90} {error "Expected 90 retained interstage paths, got $pairs"}
    set f [open $out/raw_lock.tsv {WRONLY CREAT EXCL}];puts $f "source\tfirst_stage\tperiod_ns\tbounded_by_this_stage"
    set pin [nc_pin {availability/lock_sync_reg[0]/D}]
    set driver [nc_one [get_pins -leaf -of_objects [get_nets -segments -of_objects $pin] -filter {DIRECTION == OUT}] lock]
    if {[get_property REF_NAME [get_cells -of_objects $driver]] ne "MMCME4_ADV" || [get_property REF_PIN_NAME $driver] ne "LOCKED"} {error "Raw lock source changed"}
    puts $f "$driver\t$pin\t10.000\t0";close $f
}
proc nc_paths {out} {
    set f [open $out/paths.tsv {WRONLY CREAT EXCL}]
    puts $f "kind\tgroup\tfrom\tto\tstart_clock\tend_clock\trequirement_ns\tslack_ns\tdatapath_ns"
    foreach group [nc_groups] {
        lassign $group kind name from to bound skew count
        set paths [get_timing_paths -quiet -from $from -to $to -delay_type max -max_paths $count -nworst 1]
        if {[llength $paths]!=$count} {error "Incomplete constrained paths $name: [llength $paths]/$count"}
        set ends {}
        foreach path $paths {
            set end [get_property ENDPOINT_PIN $path];lappend ends $end
            if {abs([get_property REQUIREMENT $path]-$bound)>0.001} {error "Missing/overridden max delay $name"}
            puts $f [join [list $kind $name [get_property STARTPOINT_PIN $path] $end [get_property STARTPOINT_CLOCK $path] [get_property ENDPOINT_CLOCK $path] [get_property REQUIREMENT $path] [get_property SLACK $path] [get_property DATAPATH_DELAY $path]] "\t"]
        }
        if {[lsort -unique $ends] ne [lsort -unique $to]} {error "Endpoint coverage changed $name"}
    };close $f
}
proc nc_run {checkpoint out mode} {
    if {[file exists $out] || $mode ni {apply reopen}} {error "New output and apply/reopen required"}
    file mkdir $out;open_checkpoint $checkpoint
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_native_join"} {error "Wrong native donor"}
    write_xdc -constraints all $out/before.xdc
    nc_clock_map $out
    nc_constrain $out [expr {$mode eq "apply"}]
    update_timing
    nc_topology $out
    nc_paths $out
    write_xdc -constraints all $out/after.xdc
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/exceptions_ignored.rpt
    report_bus_skew -warn_on_violation -file $out/bus_skew.rpt
    report_cdc -details -file $out/cdc.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing.rpt
    check_timing -verbose -file $out/check_timing.rpt
    foreach {domain clock} $::nc_clocks {
        report_timing -from [get_clocks $clock] -to [get_clocks $clock] -delay_type max -max_paths 2 -file $out/${domain}_setup.rpt
        report_timing -from [get_clocks $clock] -to [get_clocks $clock] -delay_type min -max_paths 2 -file $out/${domain}_hold.rpt
    }
    if {$mode eq "apply"} {write_checkpoint $out/constrained.dcp}
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f NATIVE_CONSTRAINTS_${mode}_EXPORTED_PENDING_REVIEW;close $f
    close_design
}
