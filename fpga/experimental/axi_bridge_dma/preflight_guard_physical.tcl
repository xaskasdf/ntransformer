# Inspect the exact native checkpoint before proposing any physical assignment.
set_param general.maxThreads 2
proc export_props {path objects} {
    set f [open $path {WRONLY CREAT EXCL}];puts $f "object\tproperty\tvalue"
    foreach o $objects {
        foreach p [lsort [list_property $o]] {
            puts $f [join [list $o $p [string map [list "\t" " " "\n" " " "\r" " "] [get_property $p $o]]] "\t"]
        }
    }
    close $f
}
if {[catch {
    if {[llength $argv]!=2} {error "CHECKPOINT NEW_OUTPUT required"}
    lassign $argv dcp out
    if {[file exists $out]} {error "New output required"}
    file mkdir $out
    open_checkpoint $dcp
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Unexpected device"}
    write_xdc -constraints all $out/before.xdc
    export_props $out/package.tsv [get_package_pins]
    export_props $out/banks.tsv [get_iobanks]
    export_props $out/physical_cells.tsv [get_cells -hierarchical -filter {REF_NAME =~ GTYE4* || REF_NAME =~ PCIE4* || REF_NAME =~ IBUFDS_GTE4 || REF_NAME =~ BUFG*}]
    export_props $out/reference_sites.tsv [get_sites -filter {SITE_TYPE =~ IBUFDS_GTE4 || SITE_TYPE =~ GTYE4_COMMON}]
    export_props $out/external_ports.tsv [get_ports {sys_clk_p sys_clk_n sys_rst_n mgmt_clk_in configuration_rst_n}]
    set f [open $out/serial_lanes.tsv {WRONLY CREAT EXCL}]
    puts $f "port\tdirection\tprimitive_pin\tcell\tref\tloc\tpackage_candidates"
    foreach p [lsort [get_ports {pci_exp_*}]] {
        set nets [get_nets -segments -of_objects $p]
        set pins [get_pins -leaf -of_objects $nets]
        foreach pin $pins {
            set c [get_cells -of_objects $pin]
            if {[get_property REF_NAME $c] ne "GTYE4_CHANNEL"} {continue}
            set loc [get_property LOC $c]
            set balls [get_package_pins -quiet -of_objects [get_sites $loc]]
            puts $f [join [list $p [get_property DIRECTION $p] [get_property REF_PIN_NAME $pin] $c [get_property REF_NAME $c] $loc [join $balls ,]] "\t"]
        }
    }
    close $f
    # Record reference pin connectivity from the actual native top, not a board XDC.
    set f [open $out/reference_network.tsv {WRONLY CREAT EXCL}]
    puts $f "driver\tloads"
    foreach p [get_pins -of_objects [get_cells refclk_ibuf] -filter {DIRECTION == OUT}] {
        set nets [get_nets -segments -of_objects $p]
        puts $f [join [list $p [join [get_pins -leaf -of_objects $nets -filter {DIRECTION == IN}] ,]] "\t"]
    }
    close $f
    check_timing -verbose -file $out/check_timing.rpt
    report_cdc -details -all_checks_per_endpoint -no_waiver -file $out/full_cdc.rpt
    report_cdc -cells [get_cells guarded] -details -all_checks_per_endpoint -no_waiver -file $out/guarded_cdc.rpt
    report_methodology -file $out/methodology.rpt
    report_waivers -file $out/waivers.rpt
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f NATIVE_PHYSICAL_PREFLIGHT_EXPORTED_NO_ASSIGNMENTS
    close $f
    close_design
} message options]} {
    puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1
}
exit 0
