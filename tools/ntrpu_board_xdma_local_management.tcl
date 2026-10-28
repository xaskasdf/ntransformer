# Generate an isolated XDMA variant exposing local register access and status.
# Generation only; native traffic, drain certification and reset remain untested.
# Usage: EXACT_PART SEALED_RESET_ENABLED_XCI EXPECTED_SHA256 NEW_OUTPUT
set_param general.maxThreads 2
proc ntrpu_xdma_local_management {arguments} {
    if {[llength $arguments] != 4 || [llength [get_projects -quiet]]} {
        error "Use fresh Vivado: EXACT_PART SEALED_RESET_ENABLED_XCI EXPECTED_SHA256 NEW_OUTPUT"
    }
    lassign $arguments part seed expected_sha output
    set seed [file normalize $seed]
    set output [file normalize $output]
    if {$part ne "xcvu5p-flva2104-2-i" || ![file isfile $seed] || [file exists $output]} {
        error "Invalid part, seed, or occupied output"
    }
    if {![regexp {^[0-9a-f]{64}$} $expected_sha]} {error "Invalid seed SHA-256"}
    set hash_code {import hashlib,sys;from pathlib import Path;print(hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())}
    if {[exec python3 -c $hash_code $seed] ne $expected_sha} {error "Seed hash mismatch"}
    file mkdir [file join $output seed]
    set copy [file join $output seed ntrpu_xdma.xci]
    file copy $seed $copy
    create_project ntrpu_xdma_local_management [file join $output project] -part $part
    import_ip $copy
    set ip [get_ips ntrpu_xdma]
    if {[llength $ip] != 1 || [get_property IPDEF $ip] ne "xilinx.com:ip:xdma:4.2"} {
        error "Expected XDMA 4.2"
    }
    set generated_dir [exec python3 -c {import json,sys;from pathlib import Path;p=Path(sys.argv[1]);print((p.parent/json.loads(p.read_text())["ip_inst"]["gen_directory"]).resolve())} [get_property IP_FILE $ip]]
    if {![string match "$output/*" $generated_dir]} {error "Generated IP escapes isolated output"}
    set invariant_properties {
        functional_mode DMA soft_reset_en true pipe_sim true
        axist_bypass_en true axilite_master_en true xdma_dsc_bypass false
        xdma_rnum_chnl 1 xdma_wnum_chnl 1 comp_timeout 50ms
        pl_link_cap_max_link_width X16 axi_data_width 512_bit
    }
    foreach {key value} [concat $invariant_properties {xdma_axilite_slave false xdma_sts_ports false}] {
        if {[get_property CONFIG.$key $ip] ne $value} {error "Unexpected seed $key"}
    }
    report_property -all -file [file join $output before.rpt] $ip
    set before [dict create]
    foreach property [lsort [list_property $ip]] {
        if {[string match CONFIG.* $property]} {dict set before $property [get_property $property $ip]}
    }
    set_property -dict [list CONFIG.xdma_axilite_slave true CONFIG.xdma_sts_ports true] $ip
    foreach {key value} [concat $invariant_properties {xdma_axilite_slave true xdma_sts_ports true}] {
        if {[get_property CONFIG.$key $ip] ne $value} {error "Generated configuration changed $key"}
    }
    report_property -all -file [file join $output after.rpt] $ip
    set f [open [file join $output config_changes.tcl] {WRONLY CREAT EXCL}]
    foreach property [lsort [dict keys $before]] {
        set after [get_property $property $ip]
        if {[dict get $before $property] ne $after} {
            puts $f [list $property [dict get $before $property] $after]
        }
    }
    close $f
    report_ip_status -file [file join $output ip_status.rpt]
    generate_target all $ip
    open_example_project -dir [file join $output example] $ip
    if {[get_property PART [current_project]] ne $part} {error "Example part changed"}
    report_ip_status -file [file join $output example_ip_status.rpt]
    set f [open [file join $output example_files.txt] {WRONLY CREAT EXCL}]
    foreach path [lsort [get_files -all]] {
        set absolute [file normalize $path]
        if {[string first [file dirname $seed] $absolute] == 0} {
            error "Example still references immutable seed"
        }
        puts $f $absolute
    }
    close $f
    close_project
    if {[exec python3 -c $hash_code $seed] ne $expected_sha} {error "Original seed changed"}
    puts NTRPU_XDMA_LOCAL_MANAGEMENT_EXAMPLE_GENERATED
    puts "Generation only. No native drain, host receipt, operational reset or board qualification."
}
if {[catch {ntrpu_xdma_local_management $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
