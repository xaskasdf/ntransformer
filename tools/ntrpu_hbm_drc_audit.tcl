# Inspect a copy of the routed checkpoint. No waivers or circuit edits.
set_param general.maxThreads 2
proc clean {value} {return [string map [list "\t" { } "\n" { } "\r" { }] $value]}
proc prop {obj name} {
    if {$name ni [list_property $obj]} {return {}}
    return [get_property $name $obj]
}
if {[catch {
    open_checkpoint [lindex $argv 0]
    report_drc -name audit -no_waivers -file drc.rpt
    report_route_status -file route_status.rpt
    set vf [open violations.tsv w]
    set cf [open cells.tsv w]
    set pf [open pins.tsv w]
    set nf [open nets.tsv w]
    set lf [open loads.tsv w]
    puts $vf "violation\tproperty\tvalue"
    puts $cf "violation\tcell\tproperty\tvalue"
    puts $pf "violation\tcell\tpin\tdirection\tnet\tbel_pins"
    puts $nf "violation\tnet\troute_status\ttype\tleaf_pins\tports"
    puts $lf "violation\tnet\tpin\tdirection\tcell\tref_name\tinit\tbel_pins"
    foreach v [lsort [get_drc_violations -name audit]] {
        foreach key [lsort [list_property $v]] {puts $vf "$v\t$key\t[clean [get_property $key $v]]"}
        foreach c [lsort [get_cells -quiet -of_objects $v]] {
            foreach key [lsort [list_property $c]] {puts $cf "$v\t$c\t$key\t[clean [get_property $key $c]]"}
            foreach pin [lsort [get_pins -quiet -of_objects $c]] {
                puts $pf "$v\t$c\t$pin\t[get_property DIRECTION $pin]\t[clean [get_nets -quiet -of_objects $pin]]\t[clean [get_bel_pins -quiet -of_objects $pin]]"
            }
        }
        foreach n [lsort [get_nets -quiet -of_objects $v]] {
            set pins [get_pins -quiet -leaf -of_objects $n]
            puts $nf "$v\t$n\t[prop $n ROUTE_STATUS]\t[prop $n TYPE]\t[llength $pins]\t[clean [get_ports -quiet -of_objects $n]]"
            foreach pin [lsort $pins] {
                set c [get_cells -quiet -of_objects $pin]
                puts $lf "$v\t$n\t$pin\t[get_property DIRECTION $pin]\t$c\t[prop $c REF_NAME]\t[prop $c INIT]\t[clean [get_bel_pins -quiet -of_objects $pin]]"
            }
        }
    }
    foreach f [list $vf $cf $pf $nf $lf] {close $f}
    close_design
} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
puts NTRPU_HBM_DRC_AUDIT_COMPLETE
exit 0
