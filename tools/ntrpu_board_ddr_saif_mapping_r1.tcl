# Correspondence diagnostic only. Independent fresh checkpoint per window.
proc mapping_row {stream values} {
    set fields {}
    foreach value $values {
        if {[regexp {[\t\r\n]} $value]} {error "TSV field contains a delimiter"}
        lappend fields $value
    }
    puts $stream [join $fields "\t"]
}
proc ntrpu_ddr_saif_mapping {arguments} {
    if {[llength $arguments] != 4 || [llength [get_projects -quiet]]} {
        error "Expected fresh process: DCP SAIF0 SAIF1 NEW_OUTPUT"
    }
    lassign $arguments checkpoint saif0 saif1 output
    foreach path [list $checkpoint $saif0 $saif1] {
        if {![file isfile $path]} {error "Missing frozen input: $path"}
    }
    if {[file exists $output]} {error "Output already exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set strip_path ntrpu_ddr_example_activity_tb/dut
    foreach window {0 1} saif [list $saif0 $saif1] {
        set folder [file join $output prbs_$window]
        file mkdir $folder
        # Do not reuse a design carrying the previous window's annotation.
        open_checkpoint $checkpoint
        set part [get_property PART [current_design]]
        set top [get_property TOP [current_design]]
        set ports [get_ports]
        if {$part ne "xcvu5p-flva2104-2-i" || $top ne "example_top" || [llength $ports] != 154} {
            error "Unexpected checkpoint identity"
        }
        set f [open [file join $folder identity.tsv] {WRONLY CREAT EXCL}]
        foreach property {NAME TOP PART} {
            mapping_row $f [list $property [get_property $property [current_design]]]
        }
        mapping_row $f [list PORTS [llength $ports]]
        mapping_row $f [list STRIP_PATH $strip_path]
        mapping_row $f [list WINDOW prbs_$window]
        mapping_row $f [list VIVADO [version -short]]
        close $f
        report_clocks -file [file join $folder clocks.rpt]
        report_route_status -file [file join $folder route_status.rpt]
        report_utilization -file [file join $folder utilization.rpt]
        set nets [get_nets -hierarchical]
        if {![llength $nets]} {error "Empty design net inventory"}
        set f [open [file join $folder nets.tsv] {WRONLY CREAT EXCL}]
        puts $f "name\ttype"
        foreach net $nets {mapping_row $f [list [get_property NAME $net] [get_property TYPE $net]]}
        close $f
        set f [open [file join $folder ports.tsv] {WRONLY CREAT EXCL}]
        puts $f "name\tdirection\tnets"
        foreach port $ports {
            mapping_row $f [list [get_property NAME $port] [get_property DIRECTION $port] [join [get_property NAME [get_nets -of_objects $port]] ";"]]
        }
        close $f
        set f [open [file join $folder cells.tsv] {WRONLY CREAT EXCL}]
        puts $f "name\tref_name\tprimitive"
        foreach cell [get_cells -hierarchical] {
            mapping_row $f [list [get_property NAME $cell] [get_property REF_NAME $cell] [get_property IS_PRIMITIVE $cell]]
        }
        close $f
        report_switching_activity -file [file join $folder activity_inherited.rpt] $nets
        reset_switching_activity -all
        report_switching_activity -file [file join $folder activity_reset.rpt] $nets
        redirect -file [file join $folder annotation.rpt] {
            read_saif -strip_path $strip_path -out_file [file join $folder unmatched.rpt] $saif
        }
        report_switching_activity -file [file join $folder activity_annotated.rpt] $nets
        report_switching_activity -file [file join $folder port_activity.rpt] $ports
        set f [open [file join $folder scope.txt] {WRONLY CREAT EXCL}]
        puts $f "Single generic DDR example, window prbs_$window, 20.48 us"
        puts $f "SAIF mapping and activity provenance only; no report_power executed"
        puts $f "Fresh checkpoint; inherited activity exported, reset, then one SAIF imported"
        puts $f "X/Z classification, source equivalence and representative activity remain under review"
        puts $f "NOT_POWER_QUALIFIED; NOT_DIMM_QUALIFIED; NOT_BOARD_QUALIFIED"
        close $f
        close_project
        puts "NTRPU_DDR_SAIF_MAPPING_WINDOW_EXPORTED window=$window"
    }
    puts NTRPU_DDR_SAIF_MAPPING_R1_EXPORTED
}
if {[catch {ntrpu_ddr_saif_mapping $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
