# Reopen the saved design without changing its constraints or implementation.
set script_dir [file dirname [file normalize [info script]]]
source [file join $script_dir ntrpu_dual_physical_reports.tcl]
proc audit {arguments} {
    global script_dir
    if {[llength $arguments]!=1 || [llength [get_projects -quiet]]} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    open_checkpoint [file normalize [file join $script_dir .. evidence routed.dcp]]
    dual_physical_reports $output
    set f [open [file join $output cross_clock_classification.tsv] w]
    puts $f "source\tdestination\tpaths_returned\tslack\trequirement\tendpoint"
    foreach a [get_clocks] {
        foreach b [get_clocks] {
            if {$a eq $b} {continue}
            set paths [get_timing_paths -quiet -from $a -to $b -max_paths 1]
            set slack {};set requirement {};set endpoint {}
            if {[llength $paths]} {
                set slack [get_property SLACK $paths]
                set requirement [get_property REQUIREMENT $paths]
                set endpoint [get_property ENDPOINT_PIN $paths]
            }
            puts $f "$a\t$b\t[llength $paths]\t$slack\t$requirement\t$endpoint"
        }
    }
    close $f
    set f [open [file join $output startup_input_cones.tsv] w]
    puts $f "cell\tpin\tdriver\tdriver_ref\tdriver_init\tdriver_clock\tinput_drivers\tinput_driver_refs\tports\tpin_inverted"
    foreach cell [lsort [get_cells -hier -filter {NAME =~ core/startup/* && IS_PRIMITIVE==1 && REF_NAME =~ FD*}]] {
        foreach pin [get_pins -of_objects $cell -filter {REF_PIN_NAME==D || REF_PIN_NAME==CLR || REF_PIN_NAME==PRE || REF_PIN_NAME==CE}] {
            set nets [get_nets -segments -of_objects $pin]
            set ports [get_ports -quiet -of_objects $nets]
            set drivers [get_pins -quiet -leaf -of_objects $nets -filter {DIRECTION==OUT}]
            if {[llength $drivers]>1} {error "Multiple drivers on $pin"}
            set ref {};set init {};set clocks {};set upstream {};set refs {}
            if {[llength $drivers]} {
                set driver_cell [get_cells -of_objects $drivers];set ref [get_property REF_NAME $driver_cell]
                if {[string match FD* $ref]} {
                    set init [get_property INIT $driver_cell]
                    set clocks [get_clocks -of_objects [get_pins $driver_cell/C]]
                } elseif {[string match LUT* $ref]} {
                    set init [get_property INIT $driver_cell]
                    foreach ip [get_pins -of_objects $driver_cell -filter {DIRECTION==IN}] {
                        set dd [get_pins -quiet -leaf -of_objects [get_nets -segments -of_objects $ip] -filter {DIRECTION==OUT}]
                        lappend upstream $dd
                        if {[llength $dd]} {
                            lappend refs [get_property REF_NAME [get_cells -of_objects $dd]]
                        } else {
                            set pp [get_ports -quiet -of_objects [get_nets -segments -of_objects $ip]]
                            lappend refs "PORT:$pp"
                        }
                    }
                }
            }
            set inverted {}
            if {[lsearch -exact [list_property $pin] IS_INVERTED]>=0} {set inverted [get_property IS_INVERTED $pin]}
            puts $f "$cell\t[get_property REF_PIN_NAME $pin]\t$drivers\t$ref\t$init\t$clocks\t$upstream\t$refs\t$ports\t$inverted"
        }
    }
    close $f
    close_design
    puts NTRPU_DUAL_PHYSICAL_AUDIT_EXPORTED
}
if {[catch {audit $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
