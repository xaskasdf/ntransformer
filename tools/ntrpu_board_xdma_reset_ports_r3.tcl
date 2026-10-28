# V4 bounded interface audit. Copies an existing XCI into two isolated projects.
# Generates instantiation templates only: no simulation/synthesis/implementation.
# A generated optional port is NOT evidence of supported runtime semantics.
# Usage: -tclargs EXACT_PART SEED_XCI NEW_OUTPUT
proc ntrpu_reset_ports {arguments} {
    if {[llength $arguments] != 3 || [llength [get_projects -quiet]]} {
        error "Use fresh Vivado: -tclargs EXACT_PART SEED_XCI NEW_OUTPUT"
    }
    lassign $arguments part seed output
    set output [file normalize $output]
    if {[file exists $output] || ![file isfile $seed]} {error "Output exists or seed absent"}
    file mkdir $output
    help generate_target
    help import_ip
    set audit [open [file join $output reset_ports.tsv] w]
    puts $audit "variant\tproperty\tvalue"
    foreach variant {baseline soft_request} {
        set path [file join $output $variant]
        file mkdir [file join $path seed]
        set copy [file join $path seed ntrpu_xdma.xci]
        file copy $seed $copy
        create_project ntrpu_reset_ports [file join $path project] -part $part
        import_ip $copy
        set ip [get_ips ntrpu_xdma]
        if {[llength $ip] != 1} {error "Expected one ntrpu_xdma"}
        if {[get_property CONFIG.functional_mode $ip] ne "DMA"} {error "Seed is not DMA"}
        set generated_dir [exec python3 -c {import json,sys;from pathlib import Path;p=Path(sys.argv[1]);print((p.parent/json.loads(p.read_text())["ip_inst"]["gen_directory"]).resolve())} [get_property IP_FILE $ip]]
        if {![string match "[file normalize $path]/*" $generated_dir]} {
            error "IP output outside isolated variant: $generated_dir"
        }
        puts $audit "$variant\tIP_OUTPUT_DIR\t$generated_dir"
        report_property -all -file [file join $path before.rpt] $ip
        set rc 0
        set diagnostic "unchanged"
        if {$variant eq "soft_request"} {
            set rc [catch {set_property CONFIG.soft_reset_en true $ip} diagnostic]
        }
        puts $audit "$variant\tset_return_code\t$rc"
        set f [open [file join $path set_diagnostic.txt] w]
        puts $f $diagnostic
        close $f
        foreach property {IPDEF CONFIG.functional_mode CONFIG.soft_reset_en CONFIG.dma_reset_source_sel CONFIG.axist_bypass_en CONFIG.axilite_master_en CONFIG.axi_data_width CONFIG.axi_addr_width CONFIG.pipe_sim CONFIG.pcie_blk_locn CONFIG.select_quad} {
            puts $audit "$variant\t$property\t[get_property $property $ip]"
        }
        set generated [catch {generate_target instantiation_template $ip} diagnostic]
        puts $audit "$variant\tgeneration_return_code\t$generated"
        set f [open [file join $path generation_diagnostic.txt] w]
        puts $f $diagnostic
        close $f
        report_property -all -file [file join $path after.rpt] $ip
        report_ip_status -file [file join $path ip_status.rpt]
        flush $audit
        close_project
    }
    close $audit
    puts "NTRPU_XDMA_RESET_PORT_AUDIT_COMPLETE"
    puts "No runtime, reset coordination, native traffic or physical qualification."
}
if {[catch {ntrpu_reset_ports $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
