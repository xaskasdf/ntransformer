# Isolated rev3 flow. Existing boundary/CDC helpers remain frozen dependencies.
set root [file normalize [file join [file dirname [info script]] ..]]
set out [file normalize [lindex $argv 0]]
proc reset_scope_rev3 {out constrain} {
    # Pre-opt FDCE instances can omit properties that have library defaults.
    # Resolve only this exact installed model; retain raw values in evidence.
    set lib [file join $::env(XILINX_VIVADO) data verilog src unisims FDCE.v]
    set sha [lindex [exec sha256sum $lib] 0]
    if {$sha ne "74d30d7c788e7db47f12b92d5b4b57b707974a0962708b309e4cc081d5f98b04"} {error "Unreviewed FDCE model"}
    set f [open $lib];set model [read $f];close $f
    foreach name {INIT IS_CLR_INVERTED} {
        set pattern [format {parameter \[0:0\] %s = 1'b0,} $name]
        if {![regexp $pattern $model]} {error "Changed FDCE default: $name"}
    }
    set f [open $out/reset_library.tsv w]
    puts $f "path\tsha256\tinit_default\tclr_inverted_default"
    puts $f "$lib\t$sha\t0\t0";close $f
    set raw [leaf_pins [get_nets -of_objects [get_ports cold_reset_n]] IN]
    set cold [all_fanout -flat -endpoints_only [get_ports cold_reset_n]]
    if {[llength $cold]!=10} {error "Unexpected cold-reset scope"}
    if {[llength $raw]==1} {
        set inv [get_cells -of_objects $raw]
        if {[get_property REF_NAME $inv] ne "LUT1" || [get_property INIT $inv] ne "2'h1"} {error "Unexpected reset inversion"}
        set op [get_pins -of_objects $inv -filter {DIRECTION == OUT}]
        if {[leaf_pins [get_nets -of_objects $op] IN] ne [lsort $cold]} {error "Reset inversion fanout changed"}
        set inverted 0;set representation LUT1
    } elseif {[lsort $raw] eq [lsort $cold]} {
        set inverted 1;set representation CLR_ATTRIBUTE
    } else {error "Unexpected raw reset loads"}
    set f [open $out/reset_polarity.tsv w]
    puts $f "endpoint\trepresentation\tclr_inverted\tclock\tref\tasync_reg\tinit\traw_clr_inverted\traw_init"
    foreach p $cold {
        set c [get_cells -of_objects $p]
        if {[get_property REF_PIN_NAME $p] ne "CLR" || [get_property REF_NAME $c] ne "FDCE" || ![get_property ASYNC_REG $c]} {error "Unexpected reset endpoint"}
        set v [get_property IS_CLR_INVERTED $c];set raw_v $v
        if {$v eq ""} {set v 0}
        if {$v in {1 1'b1 true TRUE}} {set bit 1} elseif {$v in {0 1'b0 false FALSE}} {set bit 0} else {error "Unknown CLR polarity: $c = $v"}
        set init [get_property INIT $c];set raw_init $init
        if {$init eq ""} {set init 1'b0}
        if {$bit!=$inverted || $init ne "1'b0"} {error "Reset is not active-low/initially clear"}
        set clock [get_clocks -of_objects [get_pins -of_objects $c -filter {REF_PIN_NAME == C}]]
        set expected [expr {[string match dr_reg* $c] ? "dma_clk" : "reader_clk"}]
        if {$clock ne $expected} {error "Reset synchronizer clock changed"}
        puts $f "$p\t$representation\t$bit\t$clock\tFDCE\t1\t$init\t$raw_v\t$raw_init"
    }
    close $f
    if {$constrain} {set_false_path -from [get_ports cold_reset_n] -to $cold}
    set f [open $out/cold_reset_endpoints.txt w];puts $f [join [lsort $cold] \n];close $f
}
proc cache_inventory {out timed} {
    set f [open $out/cache_registers.tsv w]
    puts $f "group\tcell\tref\tclock\tloc"
    set m [open $out/cache_metrics.tsv w]
    puts $m "group\tregisters\tsetup_ns\thold_ns"
    foreach group {window_end published_end source_capacity target_capacity} {
        set pattern [format {^%s_reg(\[[0-9]+\])?(_replica.*)?$} $group]
        set cells [get_cells -quiet -hier -regexp $pattern]
        if {![llength $cells]} {error "Missing precomputed bound: $group"}
        foreach c $cells {
            set clock [get_clocks -of_objects [get_pins -of_objects $c -filter {REF_PIN_NAME == C}]]
            if {$clock ne "reader_clk" || ![string match FD* [get_property REF_NAME $c]]} {
                error "Wrong cache register/clock: $c"
            }
            puts $f "$group\t$c\t[get_property REF_NAME $c]\t$clock\t[get_property LOC $c]"
        }
        set setup NA;set hold NA
        if {$timed} {
            set sources [all_registers -clock reader_clk]
            foreach {kind delay} {setup max hold min} {
                set p [get_timing_paths -from $sources -to $cells -delay_type $delay -max_paths 1]
                if {[llength $p]!=1} {error "Missing cache timing: $group $kind"}
                set $kind [get_property SLACK $p]
                report_timing -from $sources -to $cells -delay_type $delay -max_paths 5 -input_pins -file $out/${group}_${kind}.rpt
            }
        }
        puts $m "$group\t[llength $cells]\t$setup\t$hold"
    }
    close $f;close $m
}
proc load_boundary_helpers {root} {
    set f [open $root/tools/ntrpu_hbm_dma_boundary.tcl];set text [read $f];close $f
    set at [string first "\nif \{\[catch \{boundary_run" $text]
    if {$at<0} {error "Boundary helper entrypoint changed"}
    uplevel #0 [string range $text 0 [expr {$at-1}]]
    rename export_boundary export_boundary_original
    rename reset_scope reset_scope_original
    proc ::reset_scope {out constrain} {reset_scope_rev3 $out $constrain}
    proc ::export_boundary {root out manifest} {
        export_boundary_original $root $out $manifest
        cache_inventory $out 1
    }
}
proc load_physical_helpers {root} {
    set f [open $root/tools/ntrpu_hbm_dma_rev2_physical.tcl];set text [read $f];close $f
    set at [string first "\nif \{\[catch \{physical" $text]
    if {$at<0} {error "Physical helper entrypoint changed"}
    uplevel #0 [string range $text 0 [expr {$at-1}]]
}
proc finish_prepare {root out} {
    load_boundary_helpers $root
    load_physical_helpers $root
    set_param general.maxThreads 4
    open_checkpoint $root/evidence/anchors_unconstrained.dcp
    file copy $root/evidence/boundary.tsv $out/boundary.tsv
    set f [open $out/boundary.tsv];set lines [split [read $f] \n];close $f
    set inputs {};set outputs {};set launch_d {};set capture_cells {}
    set f [open $out/connectivity.tsv w];puts $f "port\tunchanged_original_pins"
    foreach line [lrange $lines 1 end] {
        if {[string trim $line] eq ""} {continue}
        lassign [split $line \t] name direction clock anchor net original
        if {[join [original_pins [get_nets [list $net]] $direction] |] ne $original} {error "Recovered connectivity differs: $name"}
        puts $f "$name\t$original"
        set port [get_ports [list $name]]
        if {$direction eq "IN"} {
            lappend inputs $port;lappend launch_d [get_pins $anchor/D]
        } else {
            lappend outputs $port;lappend capture_cells [get_cells $anchor]
        }
        if {$name eq "dma_link_up"} {set link [get_cells $anchor]}
    }
    close $f
    if {[llength $inputs]!=7662 || [llength $outputs]!=3713} {error "Wrong recovered boundary size"}
    set_false_path -from $inputs -to $launch_d
    set_false_path -from $capture_cells -to $outputs
    set_max_delay -datapath_only 4.0 -from $link -to [get_pins link_s1_reg/D]
    reset_scope $out 1
    write_checkpoint $out/prepared.dcp
    export_boundary $root $out $out/boundary.tsv
}
proc run_rev3 {root out} {
    file mkdir $out
    set f [open $root/evidence/mode.txt];set mode [string trim [read $f]];close $f
    if {$mode eq "synth"} {
        # This script reads exactly four production RTL files, never the reference TB.
        uplevel #0 [list source $root/tools/ntrpu_hbm_dma_cdc_probe.tcl]
        cache_inventory $out 0
    } elseif {$mode eq "capture"} {
        # Its entrypoint independently opens the new synth.dcp and checks all bindings.
        uplevel #0 [list source $root/tools/ntrpu_hbm_dma_rev2_capture_review.tcl]
    } elseif {$mode eq "finish_prepare"} {
        finish_prepare $root $out
    } elseif {$mode eq "place"} {
        set_param general.maxThreads 4
        open_checkpoint $root/evidence/prepared.dcp
        file copy $root/evidence/boundary.tsv $out/boundary.tsv
        opt_design
        write_checkpoint $out/optimized.dcp
        place_design
        write_checkpoint $out/placed.dcp
        report_timing_summary -report_unconstrained -file $out/placed_timing.rpt
        cache_inventory $out 1
    } else {
        load_boundary_helpers $root
        boundary_run $root $out
    }
    puts NTRPU_REV3_PHYSICAL_STAGE_DONE
}
if {[catch {run_rev3 $root $out} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
