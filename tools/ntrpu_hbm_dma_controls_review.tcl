# Read-only donor checkpoint, new constrained checkpoint; no place/route or waiver.
set root [file normalize [file join [file dirname [info script]] ..]]
proc exact_cells {pattern expected} {
    set result [get_cells -quiet -hier -regexp $pattern]
    if {[llength $result] != $expected} {error "Expected $expected cells: $pattern; got $result"}
    return $result
}
proc bind_control {name src_pattern dst_pattern expected source_clock dest_clock gray} {
    global bindings
    set sources [exact_cells $src_pattern $expected]
    set destinations [exact_cells $dst_pattern $expected]
    foreach cell $sources {
        set clock [get_clocks -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]]
        if {$clock ne $source_clock} {error "Wrong source clock on $cell: $clock"}
    }
    foreach cell $destinations {
        if {![get_property ASYNC_REG $cell]} {error "First stage lacks ASYNC_REG: $cell"}
        set dpin [get_pins -of_objects $cell -filter {REF_PIN_NAME == D}]
        set net [get_nets -of_objects $dpin]
        set drivers [get_pins -of_objects $net -filter {DIRECTION == OUT}]
        # A net can list both the driving primitive and a hierarchy boundary
        # output. Count real logic drivers, not the boundary alias.
        set driver_cells [get_cells -of_objects $drivers -filter {IS_PRIMITIVE == 1}]
        if {[llength $driver_cells]!=1 || [lsearch -exact $sources $driver_cells]<0} {
            error "Unexpected logic/driver before $cell: $drivers"
        }
        set clock [get_clocks -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]]
        if {$clock ne $dest_clock} {error "Wrong destination clock on $cell: $clock"}
        set qpin [get_pins -of_objects $cell -filter {REF_PIN_NAME == Q}]
        set loads [get_pins -of_objects [get_nets -of_objects $qpin] -filter {DIRECTION == IN}]
        if {[llength $loads]!=1 || [get_property REF_PIN_NAME $loads] ne "D"} {
            error "First-stage fanout is not a single D pin: $cell: $loads"
        }
        set second [get_cells -of_objects $loads]
        if {![get_property ASYNC_REG $second]} {error "Second stage lacks ASYNC_REG: $second"}
        set second_clock [get_clocks -of_objects [get_pins -of_objects $second -filter {REF_PIN_NAME == C}]]
        if {$second_clock ne $dest_clock} {error "Wrong second-stage clock"}
        puts $bindings "$name\t$driver_cells\t$cell\t$second\t$source_clock\t$dest_clock\t$gray"
    }
}
proc review {root out} {
    global bindings
    file mkdir $out
    set_param general.maxThreads 2
    open_checkpoint [file join $root evidence synth.dcp]
    foreach {clock period} {reader_clk 5.0 dma_clk 4.0} {
        if {[llength [get_clocks $clock]]!=1 || [get_property PERIOD [get_clocks $clock]]!=$period} {error "Wrong profile clock: $clock"}
    }
    if {[llength [get_cells -hier -filter {ASYNC_REG == 1}]]!=98} {error "Unexpected synchronizer count"}
    set bindings [open $out/bindings.tsv w]
    puts $bindings "group\tsource\tfirst\tsecond\tsource_clock\tdest_clock\tgray"
    foreach lane {0 1} {
        foreach fifo {commands payloads completions} {
            foreach {src dst} {wg ws1 rg rs1} {
                set forward [expr {($fifo ne "completions") == ($src eq "wg")}]
                set sc [expr {$forward ? "reader_clk" : "dma_clk"}]
                set dc [expr {$forward ? "dma_clk" : "reader_clk"}]
                set sp [format {lanes\[%d\]\.bridge/%s/%s_reg\[[0-2]\]} $lane $fifo $src]
                set dp [format {lanes\[%d\]\.bridge/%s/%s_reg\[[0-2]\]} $lane $fifo $dst]
                bind_control l${lane}_${fifo}_${src} $sp $dp 3 $sc $dc 1
            }
        }
        bind_control l${lane}_fault [format {lanes\[%d\]\.bridge/dma_fault_reg} $lane] [format {lanes\[%d\]\.bridge/fault_s1_reg} $lane] 1 dma_clk reader_clk 0
        bind_control l${lane}_sent [format {lanes\[%d\]\.bridge/sent_generation_reg\[[0-1]\]} $lane] [format {lanes\[%d\]\.bridge/sent_s1_reg\[[0-1]\]} $lane] 2 dma_clk reader_clk 0
    }
    bind_control ready {dr_reg\[2\]} {ready_s1_reg} 1 dma_clk reader_clk 0
    close $bindings
    read_xdc [file join $root fpga constraints ntrpu_hbm_dma_bridge_controls_cdc.xdc]
    report_exceptions -coverage -file $out/exceptions.rpt
    report_exceptions -ignored -file $out/ignored_exceptions.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/cdc.rpt
    report_timing_summary -report_unconstrained -file $out/timing.rpt
    report_clock_interaction -file $out/clock_interaction.rpt
    report_bus_skew -file $out/bus_skew_estimated.rpt
    check_timing -verbose -file $out/check_timing.rpt
    set f [open $out/metrics.tsv w]
    foreach clock {reader_clk dma_clk} {
        foreach {kind delay} {setup max hold min} {
            set path [get_timing_paths -from [get_clocks $clock] -to [get_clocks $clock] -delay_type $delay -max_paths 1]
            puts $f "${clock}_${kind}\t[get_property SLACK $path]"
        }
    }
    close $f
    write_checkpoint $out/controls_constrained.dcp
    write_xdc -constraints all $out/effective_constraints.xdc
    puts NTRPU_BRIDGE_CONTROLS_REVIEW_DONE
}
if {[catch {review $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    puts stderr [dict get $options -errorinfo]
    exit 1
}
exit 0
