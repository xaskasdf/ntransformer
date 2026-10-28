# Inspect installed IP properties before choosing a memory configuration.
# Usage: vivado -mode batch -source tools/ntrpu_board_ip_probe.tcl \
#          -tclargs EXACT_PART NEW_DIRECTORY
proc ntrpu_probe {arguments} {
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Use a fresh process: -tclargs EXACT_PART NEW_DIRECTORY"
    }
    lassign $arguments part output
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists: $output" }
    file mkdir $output
    create_project ntrpu_ip_probe [file join $output project] -part $part
    foreach {name version} {ddr4 2.2 pcie4_uscale_plus 1.3 xdma 4.2} {
        create_ip -name $name -vendor xilinx.com -library ip -version $version -module_name ntrpu_$name
        set ip [get_ips ntrpu_$name]
        report_property -all -file [file join $output ${name}_properties.rpt] $ip
        set stream [open [file join $output ${name}_values.txt] {WRONLY CREAT EXCL}]
        foreach property [lsort [list_property $ip]] {
            if {![string match CONFIG.* $property]} { continue }
            puts $stream [list $property value [get_property $property $ip]]
            if {[catch {list_property_value $property $ip} values]} {
                puts $stream [list $property allowed_values_unavailable $values]
            } else {
                puts $stream [list $property allowed_values $values]
            }
        }
        close $stream
    }
    report_ip_status -file [file join $output ip_status.rpt]
    # Tiny out-of-context design checks device synthesis license, not DDR/PCIe.
    set stream [open [file join $output license_smoke.v] {WRONLY CREAT EXCL}]
    puts $stream {module license_smoke(input clk, input d, output reg q); always @(posedge clk) q <= d; endmodule}
    close $stream
    read_verilog [file join $output license_smoke.v]
    synth_design -top license_smoke -mode out_of_context -part $part
    report_utilization -file [file join $output license_smoke_utilization.rpt]
    write_checkpoint [file join $output license_smoke.dcp]
    if {[llength [get_cells -hier -filter {REF_NAME == FDRE}]] != 1} {
        error "Expected one retained flip-flop in license smoke"
    }
    close_project
    puts NTRPU_IP_PROBE_COMPLETE
}
if {[catch {ntrpu_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
