# Separate V2 catalog alternative: one 128 GiB 3DS RDIMM per channel.
# Does not replace the two-64-GiB profile or qualify physical memory.
proc ntrpu_ddr_3ds_probe {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected EXACT_FPGA_PART NEW_DIRECTORY"
    }
    lassign $arguments part output
    if {$part ne "xcvu5p-flva2104-2-i"} { error "Unexpected exploratory part" }
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    set stream [open [file join $output SCOPE.txt] {WRONLY CREAT EXCL}]
    puts $stream "SEPARATE CATALOG ALTERNATIVE: M393AAK40B41-CTC, one 128 GiB 3DS RDIMM, two package ranks, stack height four, 8 Gb x4 chips."
    puts $stream "Not the Micron dual-slot 16 Gb profile. No SPD/model/electrical qualification or simulation PASS."
    close $stream
    create_project ntrpu_ddr_3ds_catalog [file join $output project] -part $part
    create_ip -name ddr4 -vendor xilinx.com -library ip -version 2.2 -module_name ntrpu_ddr_3ds_catalog
    set ip [get_ips ntrpu_ddr_3ds_catalog]
    set requested [list CONFIG.C0.DDR4_isCustom false \
        CONFIG.C0.DDR4_MemoryType RDIMMs CONFIG.C0.DDR4_MemoryPart M393AAK40B41-CTC \
        CONFIG.C0.DDR4_Slot Single CONFIG.C0.DDR4_DataWidth 72 \
        CONFIG.C0.DDR4_Ecc true \
        CONFIG.C0.DDR4_AxiSelection true CONFIG.C0.DDR4_AxiDataWidth 512 \
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_InputClockPeriod 4000]
    set stream [open [file join $output requested.tcl] {WRONLY CREAT EXCL}]
    puts $stream "set_property -dict [list $requested] \[get_ips ntrpu_ddr_3ds_catalog\]"
    close $stream
    set failed [catch {set_property -dict $requested $ip} message options]
    report_property -all -file [file join $output resulting_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    if {$failed} { error "Catalog alternative rejected: $message" }
    foreach {property expected} $requested {
        if {[get_property $property $ip] ne $expected} { error "Unexpected $property" }
    }
    # The catalog part disables this selector and fixes it to NONE for x4.
    if {[get_property CONFIG.C0.DDR4_DataMask $ip] ne "NONE"} {
        error "Unexpected data-mask mode for the x4 catalog part"
    }
    if {[get_property CONFIG.C0.DDR4_AxiAddressWidth $ip] != 37} {
        error "Catalog alternative does not expose 128 GiB through AXI"
    }
    generate_target all $ip
    puts "NTRPU_3DS_CATALOG_TARGETS_EXPORTED; independently inspect generated geometry; not DIMM qualification or simulation PASS"
    close_project
}
if {[catch {ntrpu_ddr_3ds_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
