# V2 candidate probe. No fallback to a smaller or different memory module.
# Usage: -tclargs EXACT_FPGA_PART NEW_DIRECTORY
proc ntrpu_candidate {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Use a fresh process: -tclargs EXACT_FPGA_PART NEW_DIRECTORY"
    }
    lassign $arguments part output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists: $output" }
    file mkdir $output
    create_project ntrpu_ddr_candidate [file join $output project] -part $part
    create_ip -name ddr4 -vendor xilinx.com -library ip -version 2.2 -module_name ntrpu_ddr_candidate
    set ip [get_ips ntrpu_ddr_candidate]
    set requested [list CONFIG.C0.DDR4_MemoryType RDIMMs \
        CONFIG.C0.DDR4_MemoryPart MTA36ASF8G72PZ-3G2F1 \
        CONFIG.C0.DDR4_Slot Dual CONFIG.C0.DDR4_DataWidth 72 \
        CONFIG.C0.DDR4_Ecc true CONFIG.C0.DDR4_DataMask NO_DM_NO_DBI \
        CONFIG.C0.DDR4_AxiSelection true CONFIG.C0.DDR4_AxiDataWidth 512 \
        CONFIG.C0.DDR4_TimePeriod 833 CONFIG.C0.DDR4_InputClockPeriod 4000]
    set stream [open [file join $output requested.tcl] {WRONLY CREAT EXCL}]
    puts $stream "set_property -dict [list $requested] \[get_ips ntrpu_ddr_candidate\]"
    puts $stream {# Target: 137438953472 useful bytes, 37-bit AXI byte address; 833 ps is exploratory only.}
    close $stream
    set failed [catch {set_property -dict $requested $ip} message options]
    report_property -all -file [file join $output resulting_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    if {$failed} {
        set stream [open [file join $output CONFIGURATION_REJECTED.txt] {WRONLY CREAT EXCL}]
        puts $stream $message
        if {[dict exists $options -errorinfo]} { puts $stream [dict get $options -errorinfo] }
        close $stream
        error "Exact DIMM request rejected; no alternate part substituted. $message"
    }
    foreach {property expected} $requested {
        if {[get_property $property $ip] ne $expected} {
            error "Requested $property=$expected, got [get_property $property $ip]"
        }
    }
    if {[get_property CONFIG.C0.DDR4_AxiAddressWidth $ip] != 37} {
        error "Target capacity requires exactly 37 AXI byte-address bits"
    }
    puts NTRPU_DDR_CANDIDATE_PARAMETERS_ACCEPTED
    puts "Exact datasheet/SPD/model still required before generation or qualification."
    close_project
}
if {[catch {ntrpu_candidate $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
