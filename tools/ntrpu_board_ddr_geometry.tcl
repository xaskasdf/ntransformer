# V2 exploratory IP geometry only: NOT a qualified Micron configuration.
# See MEMORY_QUALIFICATION.md for field provenance and limitations.
# Usage: -tclargs EXACT_FPGA_PART NEW_DIRECTORY
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_geometry {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Use fresh process: -tclargs EXACT_FPGA_PART NEW_DIRECTORY"
    }
    lassign $arguments part output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    file copy [file join $ntrpu_script_dir .. docs fpga board ddr4_geometry_exploratory.csv] [file join $output geometry.csv]
    set stream [open [file join $output SCOPE.txt] {WRONLY CREAT EXCL}]
    puts $stream "EXPLORATORY GEOMETRY ONLY. Generic AMD 16Gb sg125E x4 timings."
    puts $stream "NOT MTA36ASF8G72PZ-3G2F1 qualification, SPD, RCD, electrical or calibration validation."
    close $stream
    create_project ntrpu_ddr_geometry [file join $output project] -part $part
    create_ip -name ddr4 -vendor xilinx.com -library ip -version 2.2 -module_name ntrpu_ddr_geometry
    set ip [get_ips ntrpu_ddr_geometry]
    set requested [list CONFIG.C0.DDR4_isCustom true \
        CONFIG.C0.DDR4_CustomParts [file join $output geometry.csv] \
        CONFIG.C0.DDR4_MemoryType RDIMMs \
        CONFIG.C0.DDR4_MemoryPart NTRPU_RDIMM64_GEOMETRY_ONLY \
        CONFIG.C0.DDR4_Slot Dual CONFIG.C0.DDR4_DataWidth 72 \
        CONFIG.C0.DDR4_Ecc true \
        CONFIG.C0.DDR4_AxiSelection true CONFIG.C0.DDR4_AxiDataWidth 512 \
        CONFIG.C0.DDR4_TimePeriod 1250 CONFIG.C0.DDR4_InputClockPeriod 4000]
    set stream [open [file join $output requested.tcl] {WRONLY CREAT EXCL}]
    puts $stream "set_property -dict [list $requested] \[get_ips ntrpu_ddr_geometry\]"
    close $stream
    set failed [catch {set_property -dict $requested $ip} message options]
    report_property -all -file [file join $output resulting_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    if {$failed} { error "Custom geometry rejected: $message" }
    foreach {property expected} $requested {
        if {$property eq "CONFIG.C0.DDR4_CustomParts"} {
            set actual [get_property $property $ip]
            if {[file normalize [file join [file dirname [get_property IP_FILE $ip]] $actual]] ne $expected} {
                error "Custom CSV path does not resolve to this run's file"
            }
            continue
        }
        if {[get_property $property $ip] ne $expected} {
            error "Requested $property=$expected, got [get_property $property $ip]"
        }
    }
    if {[get_property CONFIG.C0.DDR4_DataMask $ip] ni {NONE NO_DM_NO_DBI}} {
        error "Unexpected physical DM/DBI mode for x4 geometry"
    }
    if {[get_property CONFIG.C0.DDR4_AxiAddressWidth $ip] != 37} {
        error "128 GiB useful capacity requires 37 AXI byte-address bits"
    }
    puts NTRPU_DDR_GEOMETRY_PARAMETERS_ACCEPTED
    generate_target all $ip
    puts NTRPU_DDR_GEOMETRY_TARGETS_GENERATED
    puts "Inspect generated geometry; this is NOT DIMM qualification or simulation PASS."
    close_project
}
if {[catch {ntrpu_geometry $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
