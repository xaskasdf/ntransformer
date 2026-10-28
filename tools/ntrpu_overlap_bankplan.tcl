# Eight disjoint soft regions selected from leaf-pin connectivity.
if {[lindex $argv 1] eq "netlist"} {
    source [file join [file dirname [info script]] ntrpu_overlap_physical.tcl]
    exit 0
}
if {[lindex $argv 1] ni {place route}} {error "Expected place, route or netlist"}
proc save_bank_members {path} {
    set f [open $path w];puts $f "bank\tcell\tref\tloc\tbel"
    for {set bank 0} {$bank<8} {incr bank} {
        set cells [lsort [get_cells -of_objects [get_pblocks ntrpu_bank_$bank]]]
        foreach c $cells ref [get_property REF_NAME $cells] loc [get_property LOC $cells] bel [get_property BEL $cells] {
            puts $f "$bank\t$c\t$ref\t$loc\t$bel"
        }
    }
    close $f
}
proc add_bank_regions {output} {
    if {[llength [get_pblocks -quiet]]} {error "Unexpected pre-existing Pblocks"}
    set allcells [dict create]
    foreach c [get_cells -hier -filter {IS_PRIMITIVE == 1}] {dict set allcells $c $c}
    set f [open [file join $output .. plan_evidence groups.tsv] r]
    set lines [split [string trim [read $f]] "\n"];close $f
    if {[lindex $lines 0] ne "bank\tcell\trange"} {error "Bank-plan schema changed"}
    set seen [dict create];set ranges [dict create];set groups [dict create]
    foreach line [lrange $lines 1 end] {
        lassign [split $line "\t"] bank name range
        if {![string is integer -strict $bank] || $bank<0 || $bank>7 || [dict exists $seen $name]} {error "Invalid/duplicate bank member"}
        if {![dict exists $allcells $name]} {error "Selected primitive missing: $name"}
        if {[dict exists $ranges $bank] && [dict get $ranges $bank] ne $range} {error "Inconsistent region"}
        dict set seen $name 1;dict set ranges $bank $range
        dict lappend groups $bank [dict get $allcells $name]
    }
    if {[dict size $groups]!=8} {error "Expected eight bank groups"}
    for {set bank 0} {$bank<8} {incr bank} {
        set pb [create_pblock ntrpu_bank_$bank]
        resize_pblock $pb -add [dict get $ranges $bank]
        set_property IS_SOFT TRUE $pb
        set_property EXCLUDE_PLACEMENT FALSE $pb
        set_property CONTAIN_ROUTING FALSE $pb
        add_cells_to_pblock $pb [dict get $groups $bank]
        if {[llength [get_sites -of_objects $pb -filter {SITE_TYPE == SLICEL || SITE_TYPE == SLICEM}]]<80} {error "Insufficient slice sites in bank $bank"}
    }
    save_bank_members [file join $output selected_before_place.tsv]
    write_xdc -constraints all [file join $output bankplan_before.xdc]
}
proc report_bank_regions {output} {
    if {[llength [get_pblocks]]!=8} {error "Unexpected Pblock count"}
    set f [open [file join $output bankplan.tsv] w]
    puts $f "bank\trange\tsoft\texclude\tcontain\tslice_sites"
    for {set bank 0} {$bank<8} {incr bank} {
        set pb [get_pblocks ntrpu_bank_$bank]
        puts $f "$bank\t[get_property GRID_RANGES $pb]\t[get_property IS_SOFT $pb]\t[get_property EXCLUDE_PLACEMENT $pb]\t[get_property CONTAIN_ROUTING $pb]\t[llength [get_sites -of_objects $pb -filter {SITE_TYPE == SLICEL || SITE_TYPE == SLICEM}]]"
        report_utilization -pblocks $pb -file [file join $output bank_${bank}_utilization.rpt]
    }
    close $f
    save_bank_members [file join $output members_after.tsv]
}
set f [open [file join [file dirname [info script]] ntrpu_transport_study.tcl] r]
set script [read $f];close $f
set needle {opt_design;place_design}
if {[string first $needle $script]<0} {error "Placement flow changed"}
set script [string map [list $needle {opt_design;add_bank_regions $output;place_design}] $script]
set needle {        write_checkpoint [file join $output physical_${stage}.dcp]}
if {[string first $needle $script]<0} {error "Checkpoint flow changed"}
set script [string map [list $needle "$needle\n        report_bank_regions \$output"] $script]
eval $script
