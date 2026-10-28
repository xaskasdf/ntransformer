# Inspect two archived synthesized DCPs. No optimization or constraint changes.
proc stage_timing {candidate reference output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach command {open_checkpoint report_timing get_timing_paths report_utilization report_high_fanout_nets} {
        set f [open [file join $output help_${command}.txt] {WRONLY CREAT EXCL}]
        puts $f [help $command]
        close $f
    }
    set f [open [file join $output VERSION.txt] {WRONLY CREAT EXCL}]
    puts $f [version]; close $f
    foreach label {candidate reference} checkpoint [list $candidate $reference] {
        set folder [file join $output $label]
        file mkdir $folder
        file copy $checkpoint [file join $folder input.dcp]
        open_checkpoint [file join $folder input.dcp]
        set registers [get_cells -hierarchical -filter {REF_NAME == FDRE}]
        if {![llength $registers]} {error "Empty register inventory"}
        set groups [dict create C {} E {} D {} A {} B {} R {} TOP {}]
        set f [open [file join $folder registers.tsv] {WRONLY CREAT EXCL}]
        puts $f "group\tregister\tprimitive"
        foreach cell [lsort $registers] {
            set matches {}
            foreach {group prefix} {C c_stage/ E e_stage/ D d_stage/ A apply_stage/ B delivery_stage/ R retirement_i/} {
                if {[string first $prefix $cell]==0} {lappend matches $group}
            }
            if {[llength $matches]>1} {error "Overlapping register groups"}
            if {[llength $matches]} {
                set group [lindex $matches 0]
            } else {
                if {[string first / $cell]>=0} {error "Unclassified hierarchy: $cell"}
                set group TOP
            }
            dict lappend groups $group $cell
            puts $f "$group\t$cell\t[get_property REF_NAME $cell]"
        }
        close $f
        foreach group [dict keys $groups] {
            if {![llength [dict get $groups $group]]} {error "Empty group $group"}
        }
        report_clocks -file [file join $folder clocks.rpt]
        report_utilization -file [file join $folder utilization.rpt]
        report_utilization -hierarchical -file [file join $folder hierarchy.rpt]
        report_timing_summary -report_unconstrained -max_paths 20 -file [file join $folder timing.rpt]
        report_high_fanout_nets -timing -load_types -max_nets 30 -file [file join $folder high_fanout_timing.rpt]
        report_exceptions -file [file join $folder exceptions.rpt]
        check_timing -verbose -file [file join $folder check_timing.rpt]
        write_xdc [file join $folder constraints.xdc]
        set f [open [file join $folder paths.tsv] {WRONLY CREAT EXCL}]
        puts $f "from\tto\tkind\tpath_exists\tstart\tend\tslack\tdata_delay"
        foreach from [dict keys $groups] {
            foreach to [dict keys $groups] {
                foreach {kind delay} {setup max hold min} {
                    set sources [dict get $groups $from]
                    set destinations [dict get $groups $to]
                    set paths [get_timing_paths -quiet -from $sources -to $destinations -delay_type $delay -max_paths 1]
                    set report [file join $folder ${from}_${to}_${kind}.rpt]
                    if {[llength $paths]==0} {
                        set empty [open $report {WRONLY CREAT EXCL}]
                        puts $empty "NTRPU_NO_STAGE_TIMING_PATH $from $to $kind"
                        close $empty
                        puts $f "$from\t$to\t$kind\t0\t\t\t\t"
                    } elseif {[llength $paths]==1} {
                        set path [lindex $paths 0]
                        report_timing -from $sources -to $destinations -delay_type $delay -max_paths 20 -input_pins -file $report
                        puts $f "$from\t$to\t$kind\t1\t[get_property STARTPOINT_PIN $path]\t[get_property ENDPOINT_PIN $path]\t[get_property SLACK $path]\t[get_property DATAPATH_DELAY $path]"
                    } else {error "Expected at most one worst path"}
                }
            }
        }
        close $f
        close_project
        puts "NTRPU_STAGE_TIMING_REVIEW_READY $label"
    }
    puts NTRPU_MONITOR_PREDECODE_STAGE_TIMING_EXPORTED
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {stage_timing {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
