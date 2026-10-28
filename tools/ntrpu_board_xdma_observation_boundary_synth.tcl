# First production-boundary probe: synthesize the unchanged XDMA IP in a copy.
# Export actual hierarchy/pins/nets. This does not attach an observer or approve reset.
proc observation_boundary {input output} {
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output] || ![file isfile [file join $input ntrpu_xdma_ex.xpr]]} {
        error "Use immutable seed example and a new output directory"
    }
    file mkdir $output
    file copy $input [file join $output example]
    set_param general.maxThreads 2
    open_project [file join $output example ntrpu_xdma_ex.xpr]
    if {[get_property PART [current_project]] ne "xcvu5p-flva2104-2-i"} {error "Wrong part"}
    set ip [get_ips ntrpu_xdma]
    if {[llength $ip]!=1} {error "Missing unique XDMA IP"}
    set copied_xci [file normalize [file join $output example ntrpu_xdma_ex.srcs sources_1 ip ntrpu_xdma ntrpu_xdma.xci]]
    set ip_xci [get_files $copied_xci]
    if {[llength $ip_xci]!=1 || [file normalize [get_property IP_FILE $ip]] ne $copied_xci} {
        error "XDMA must resolve to the copied XCI"
    }
    foreach {key value} {CONFIG.functional_mode DMA CONFIG.soft_reset_en true CONFIG.pipe_sim true CONFIG.axist_bypass_en true CONFIG.axilite_master_en true CONFIG.xdma_dsc_bypass false CONFIG.xdma_sts_ports true CONFIG.xdma_axilite_slave true CONFIG.xdma_rnum_chnl 1 CONFIG.xdma_wnum_chnl 1 CONFIG.comp_timeout 50ms} {
        if {[get_property $key $ip] ne $value} {error "Changed native profile: $key"}
    }
    report_property -all -file [file join $output xdma_properties.rpt] $ip
    report_ip_status -file [file join $output ip_status.rpt]
    if {[llength [get_runs -quiet ntrpu_xdma_synth_1]]} {error "Unexpected pre-existing synthesis run in seed copy"}
    generate_target synthesis $ip
    # UG835 get_files: scope compile order to this IP's composite XCI only.
    set f [open [file join $output ip_synthesis_sources.tsv] {WRONLY CREAT EXCL}]
    puts $f "kind\tpath"
    foreach kind {sources constraints} {
        foreach source [get_files -compile_order $kind -used_in synthesis -of_objects $ip_xci] {
            set path [file normalize $source]
            puts $f "$kind\t$path"
            if {![string match "$output/example/*" $path]} {
                close $f
                error "IP synthesis source escapes the copied example: $path"
            }
        }
    }
    close $f
    set run [create_ip_run $ip]
    if {[llength $run]!=1 || $run ne "ntrpu_xdma_synth_1"} {error "Unexpected IP synthesis run: $run"}
    set hook [file join $output synthesis_threads.tcl]
    set f [open $hook {WRONLY CREAT EXCL}]
    puts $f {set_param general.maxThreads 2}
    close $f
    set_property STEPS.SYNTH_DESIGN.TCL.PRE $hook [get_runs $run]
    launch_runs $run -jobs 1
    wait_on_run $run
    report_property -all -file [file join $output run_properties.rpt] [get_runs $run]
    if {[get_property PROGRESS [get_runs $run]] ne "100%" ||
        [string first "Complete" [get_property STATUS [get_runs $run]]]<0} {error "IP synthesis did not complete"}
    open_run $run
    report_utilization -hierarchical -file [file join $output utilization.rpt]
    report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
    report_drc -file [file join $output drc.rpt]
    set f [open [file join $output cells.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\treference\tprimitive\tblackbox"
    foreach cell [lsort [get_cells -hierarchical]] {
        puts $f "$cell\t[get_property REF_NAME $cell]\t[get_property IS_PRIMITIVE $cell]\t[get_property IS_BLACKBOX $cell]"
    }
    close $f
    set f [open [file join $output pcie_boundary_pins.tsv] {WRONLY CREAT EXCL}]
    puts $f "cell\tpin\tdirection\tnets"
    foreach cell [lsort [get_cells -hierarchical -filter {REF_NAME =~ *pcie4* || NAME =~ *pcie4_ip_i}]] {
        foreach pin [lsort [get_pins -of_objects $cell]] {
            puts $f "$cell\t$pin\t[get_property DIRECTION $pin]\t[join [lsort [get_nets -of_objects $pin]] ,]"
        }
    }
    close $f
    write_checkpoint [file join $output xdma_observation_boundary.dcp]
    write_verilog -mode funcsim [file join $output xdma_observation_boundary_funcsim.v]
    close_project
    puts NTRPU_XDMA_OBSERVATION_BOUNDARY_EXPORTED
}
if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {observation_boundary {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
