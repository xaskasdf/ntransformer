# Read-only audit of guard release chains and all timed clock crossings.
proc audit {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected SYNTH_DCP NEW_OUTPUT in a fresh process"
    }
    lassign $arguments checkpoint output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    file copy $checkpoint [file join $output input.dcp]
    open_checkpoint [file join $output input.dcp]
    if {[get_property TOP [current_design]] ne "ntrpu_host_loss_guard" ||
        [get_property PART [current_design]] ne "xcvu5p-flva2104-2-i" || [llength [get_clocks]] != 3} {
        error "Unexpected input checkpoint"
    }
    set chains {link_filter endpoint_filter host_count_sync source_count_sync host_release source_release host_up_sync source_up_sync host_admission source_admission}
    set releases [get_cells -quiet -hierarchical -filter {NAME =~ *release_ff_reg*}]
    set f [open [file join $output release_registers.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tref_name\tinit\tasync_reg\tclock\tdata_driver\tclear_drivers\tclear_ports\tclear_startpoints"
    set total 0
    foreach suffix $chains {
        set chain $suffix
        set cells {}
        foreach cell $releases {
            if {[file dirname $cell] eq $chain} {lappend cells $cell}
        }
        if {[llength $cells] != 3} {error "Expected three release stages: $chain"}
        set previous ""
        foreach cell [lsort $cells] {
            set ref [get_property REF_NAME $cell]
            set init [get_property INIT $cell]
            set async [get_property ASYNC_REG $cell]
            set clk [get_clocks -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == C}]]
            set dn [get_nets -segments -of_objects [get_pins -of_objects $cell -filter {REF_PIN_NAME == D}]]
            set dd [get_pins -quiet -leaf -of_objects $dn -filter {DIRECTION == OUT}]
            set clr [get_pins -of_objects $cell -filter {REF_PIN_NAME == CLR}]
            set cn [get_nets -segments -of_objects $clr]
            set cd [get_pins -quiet -leaf -of_objects $cn -filter {DIRECTION == OUT}]
            set cp [get_ports -quiet -of_objects $cn]
            set starts [all_fanin -flat -startpoints_only -to $clr]
            puts $f "$cell\t$ref\t$init\t$async\t$clk\t$dd\t$cd\t$cp\t$starts"
            if {$ref ne "FDCE" || $init ne "1'b0" || $async ni {1 TRUE true} || [llength $clk] != 1} {
                error "Unrecognized reset release primitive: $cell"
            }
            if {$previous eq ""} {
                if {[llength $dd] != 1 || [get_property REF_NAME [get_cells -of_objects $dd]] ne "VCC"} {
                    error "First release stage must shift in one: $cell"
                }
            } elseif {$dd ne "$previous/Q"} {error "Broken D/Q release chain: $cell"}
            set previous $cell
            incr total
        }
    }
    close $f
    if {$total != 30 || [llength $releases] != 30} {error "Unexpected release register inventory"}
    set f [open [file join $output interclock_paths.tsv] {WRONLY CREAT EXCL}]
    puts $f "from_clock\tto_clock\tdelay_type\tstartpoint\tendpoint\tslack\tdatapath_delay"
    foreach {from to} {mgmt_clk source_clk source_clk mgmt_clk mgmt_clk host_clk host_clk mgmt_clk} {
        foreach delay {max min} {
            set paths [get_timing_paths -quiet -from [get_clocks $from] -to [get_clocks $to] -delay_type $delay -max_paths 1000 -nworst 1]
            if {[llength $paths] == 1000} {error "Timing path inventory truncated"}
            foreach p $paths {
                puts $f "$from\t$to\t$delay\t[get_property STARTPOINT_PIN $p]\t[get_property ENDPOINT_PIN $p]\t[get_property SLACK $p]\t[get_property DATAPATH_DELAY $p]"
            }
            report_timing -from [get_clocks $from] -to [get_clocks $to] -delay_type $delay -max_paths 1000 -nworst 1 -file [file join $output ${from}_${to}_${delay}.rpt]
        }
    }
    close $f
    report_cdc -details -all_checks_per_endpoint -file [file join $output cdc_all.rpt]
    report_methodology -file [file join $output methodology.rpt]
    report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
    write_xdc -constraints all [file join $output applied.xdc]
    puts "NTRPU_HOST_GUARD_RESET_AUDIT_EXPORTED release_registers=$total"
    close_project
}
if {[catch {audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
