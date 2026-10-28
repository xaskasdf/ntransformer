# Explore the generic 128 GiB controller example. NOT exact-DIMM qualification.
# Usage: -tclargs COMPLETED_GEOMETRY_OUTPUT NEW_DIRECTORY
proc ntrpu_ddr_example {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected fresh process and COMPLETED_GEOMETRY_OUTPUT NEW_DIRECTORY"
    }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    if {![file exists [file join $input project ntrpu_ddr_geometry.xpr]]} {
        error "Missing generated generic geometry project"
    }
    file copy $input $output
    open_project [file join $output project ntrpu_ddr_geometry.xpr]
    foreach file [get_files -all] {
        if {[string first $input [file normalize $file]] == 0} {
            error "Copied project references immutable source: $file"
        }
    }
    set ip [get_ips ntrpu_ddr_geometry]
    foreach {property expected} {
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY
        CONFIG.C0.DDR4_AxiAddressWidth 37 CONFIG.C0.DDR4_Ecc true
        CONFIG.C0.DDR4_AxiDataWidth 512 CONFIG.C0.DDR4_Slot Dual
    } {
        if {[get_property $property $ip] ne $expected} { error "Unexpected $property" }
    }
    report_ip_status -file [file join $output example_ip_status.rpt]
    open_example_project -dir [file join $output example] $ip
    puts NTRPU_GENERIC_DDR_EXAMPLE_GENERATED
    puts "Inspect models and geometry before any simulation; generation is not qualification."
    close_project
}
if {[catch {ntrpu_ddr_example $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
