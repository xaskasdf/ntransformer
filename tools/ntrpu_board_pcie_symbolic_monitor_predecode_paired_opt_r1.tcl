# Compare ordinary opt_design on two reviewed checkpoints. No placement or route.
proc paired_opt_inventory {folder prefix} {
    set f [open [file join $folder ${prefix}_primitives.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tprimitive"
    foreach cell [lsort [get_cells -hierarchical -filter {IS_PRIMITIVE}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]"
    }
    close $f
    set f [open [file join $folder ${prefix}_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "register\tprimitive\tinit"
    foreach cell [lsort [get_cells -hierarchical -filter {REF_NAME == FDRE}]] {
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property INIT $cell]"
    }
    close $f
    set f [open [file join $folder ${prefix}_ports.tsv] {WRONLY CREAT EXCL}]
    puts $f "port\tdirection"
    foreach port [lsort [get_ports]] {puts $f "$port\t[get_property DIRECTION $port]"}
    close $f
    write_xdc -constraints all [file join $folder ${prefix}.xdc]
}

proc paired_opt_check {top} {
    if {[get_property TOP [current_design]] ne $top ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i"} {error "Wrong checkpoint identity"}
    if {[llength [get_clocks]]!=1 || [get_property PERIOD [get_clocks pcie_user_clk]]!=4.0} {
        error "Wrong clock inventory or period"
    }
    if {[get_property HD.CLK_SRC [get_ports clk]] ne ""} {error "Unexpected external clock source assumption"}
    if {[llength [get_cells -quiet -hierarchical -filter {IS_BLACKBOX == 1 || REF_NAME == LDCE || REF_NAME == LDPE || REF_NAME =~ BUFG*}]]} {
        error "Unexpected blackbox, latch or clock buffer"
    }
}

proc paired_opt {candidate reference output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    foreach command {open_checkpoint opt_design write_checkpoint write_edif write_verilog report_timing report_timing_summary} {
        set f [open [file join $output help_${command}.txt] {WRONLY CREAT EXCL}]
        puts $f [help $command]; close $f
    }
    set f [open [file join $output VERSION.txt] {WRONLY CREAT EXCL}]
    puts $f [version]; close $f
    foreach label {candidate reference} variant {winner prefix} checkpoint [list $candidate $reference] {
        set folder [file join $output $label]; file mkdir $folder
        file copy $checkpoint [file join $folder input.dcp]
        open_checkpoint [file join $folder input.dcp]
        set top ntrpu_pcie_symbolic_monitor_predecode_fault_leaves_${variant}_r1
        paired_opt_check $top
        paired_opt_inventory $folder before
        set started [clock milliseconds]
        opt_design
        set elapsed [expr {[clock milliseconds]-$started}]
        # Preserve all transformed artifacts before the structural checks.
        write_checkpoint [file join $folder optimized.dcp]
        write_edif [file join $folder optimized.edf]
        write_verilog -mode funcsim [file join $folder optimized_funcsim.v]
        paired_opt_inventory $folder after
        report_timing_summary -report_unconstrained -max_paths 20 -file [file join $folder timing.rpt]
        report_utilization -file [file join $folder utilization.rpt]
        report_utilization -hierarchical -file [file join $folder hierarchy.rpt]
        report_drc -file [file join $folder drc.rpt]
        report_exceptions -file [file join $folder exceptions.rpt]
        check_timing -verbose -file [file join $folder check_timing.rpt]
        foreach {name source destination} {B_R delivery_stage/* retirement_i/* E_D e_stage/* d_stage/*} {
            set sources [get_cells -hierarchical -filter "REF_NAME == FDRE && NAME =~ $source"]
            set destinations [get_cells -hierarchical -filter "REF_NAME == FDRE && NAME =~ $destination"]
            if {![llength $sources] || ![llength $destinations]} {error "Empty boundary $name"}
            foreach {kind delay} {setup max hold min} {
                report_timing -from $sources -to $destinations -delay_type $delay -max_paths 20 -input_pins -file [file join $folder ${name}_${kind}.rpt]
            }
        }
        paired_opt_check $top
        set f [open [file join $folder optimization.tsv] {WRONLY CREAT EXCL}]
        puts $f "label\ttop\tpart\tcommand\tthreads\telapsed_ms"
        puts $f "$label\t$top\t[get_property PART [current_design]]\topt_design\t2\t$elapsed"
        close $f
        close_project
        puts "NTRPU_PAIRED_OPT_REVIEW_READY $label"
    }
    puts NTRPU_MONITOR_PREDECODE_PAIRED_OPT_EXPORTED
}
if {[llength $argv]!=3 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {paired_opt {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
