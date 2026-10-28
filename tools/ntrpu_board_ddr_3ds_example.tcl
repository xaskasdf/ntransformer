# Generate a separate catalog 3DS example; preserve its source project.
proc ntrpu_3ds_example {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected COMPLETED_3DS_OUTPUT NEW_DIRECTORY"
    }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    if {![file exists [file join $input project ntrpu_ddr_3ds_catalog.xpr]]} {
        error "Missing generated catalog project"
    }
    file copy $input $output
    open_project [file join $output project ntrpu_ddr_3ds_catalog.xpr]
    foreach file [get_files -all] {
        if {[string first $input [file normalize $file]] == 0} {
            error "Copied project references immutable source: $file"
        }
    }
    set ip [get_ips ntrpu_ddr_3ds_catalog]
    foreach {property expected} {
        CONFIG.C0.DDR4_MemoryPart M393AAK40B41-CTC
        CONFIG.C0.DDR4_isCustom false CONFIG.C0.DDR4_AxiAddressWidth 37
        CONFIG.C0.DDR4_Ecc true CONFIG.C0.DDR4_AxiDataWidth 512
        CONFIG.C0.DDR4_Slot Single CONFIG.C0.DDR4_TimePeriod 1250
    } {
        if {[get_property $property $ip] ne $expected} { error "Unexpected $property" }
    }
    report_ip_status -file [file join $output example_ip_status.rpt]
    open_example_project -dir [file join $output example] $ip
    puts "NTRPU_3DS_EXAMPLE_GENERATED; inspect model/CID/ranks before simulation"
    close_project
}
if {[catch {ntrpu_3ds_example $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
