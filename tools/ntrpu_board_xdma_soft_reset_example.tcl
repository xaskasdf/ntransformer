# Generate a separate native XDMA example with its optional reset port.
# Generation only: no simulator, synthesis, route, or runtime qualification.
# Usage: -tclargs EXACT_PART SEALED_SEED_XCI NEW_OUTPUT
proc ntrpu_soft_reset_example {arguments} {
    if {[llength $arguments]!=3 || [llength [get_projects -quiet]]} {
        error "Use fresh Vivado: EXACT_PART SEALED_SEED_XCI NEW_OUTPUT"
    }
    lassign $arguments part seed output
    set output [file normalize $output]
    if {[file exists $output] || ![file isfile $seed]} {error "Output exists or seed absent"}
    file mkdir [file join $output seed]
    set copy [file join $output seed ntrpu_xdma.xci]
    file copy $seed $copy
    create_project ntrpu_soft_reset [file join $output project] -part $part
    import_ip $copy
    set ip [get_ips ntrpu_xdma]
    if {[llength $ip]!=1} {error "Expected one ntrpu_xdma"}
    set generated_dir [exec python3 -c {import json,sys;from pathlib import Path;p=Path(sys.argv[1]);print((p.parent/json.loads(p.read_text())["ip_inst"]["gen_directory"]).resolve())} [get_property IP_FILE $ip]]
    if {![string match "$output/*" $generated_dir]} {error "Generated IP escapes new output: $generated_dir"}
    foreach {key value} {functional_mode DMA soft_reset_en false pipe_sim true axist_bypass_en true axilite_master_en true} {
        if {[get_property CONFIG.$key $ip] ne $value} {error "Unexpected seed $key"}
    }
    report_property -all -file [file join $output before.rpt] $ip
    set_property CONFIG.soft_reset_en true $ip
    report_property -all -file [file join $output after.rpt] $ip
    if {[get_property CONFIG.functional_mode $ip] ne "DMA" || [get_property CONFIG.soft_reset_en $ip] ne "true"} {error "Reset property not preserved"}
    report_ip_status -file [file join $output ip_status.rpt]
    generate_target all $ip
    open_example_project -dir [file join $output example] $ip
    puts NTRPU_XDMA_SOFT_RESET_EXAMPLE_GENERATED
    puts "DMA optional reset is unqualified. No simulator or physical implementation was run."
    close_project
}
if {[catch {ntrpu_soft_reset_example $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
