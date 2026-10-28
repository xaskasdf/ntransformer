# Inventory the installed AMD clock converters before connecting XDMA to FFN.
set script_dir [file dirname [file normalize [info script]]]
proc ntrpu_host_recode {arguments} {
    global script_dir
    if {[llength $arguments] != 2 || [llength [get_projects -quiet]]} {
        error "Expected probe|sim|sim_corrupt|synth_axi|synth_axil NEW_DIRECTORY in a fresh Vivado process"
    }
    lassign $arguments mode output
    if {$mode ni {sim sim_corrupt}} {error "Unsupported mode"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    create_project host_clock [file join $output project] -part xcvu5p-flva2104-2-i
    set defs [get_ipdefs -all -quiet xilinx.com:ip:axi_clock_converter:2.1]
    if {[llength $defs] != 1} {error "Expected installed clock converter 2.1"}
    report_property -all $defs -file [file join $output ip_definition.rpt]
    set synthesis [expr {$mode in {synth_axi synth_axil}}]
    set top [expr {$mode eq "synth_axil" ? "ntrpu_host_axil_cdc":"ntrpu_host_axi_cdc"}]
    foreach {name protocol width address} {
        ntrpu_host_axi_cdc AXI4 512 64
        ntrpu_host_axil_cdc AXI4LITE 32 32
    } {
        if {$synthesis && $name ne $top} {continue}
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name $name
        set ip [get_ips $name]
        if {$synthesis} {set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]}
        report_property -all $ip -file [file join $output ${name}_defaults.rpt]
        set stream [open [file join $output ${name}_allowed.txt] {WRONLY CREAT EXCL}]
        foreach property [lsort [list_property $ip]] {
            if {![string match CONFIG.* $property]} {continue}
            puts $stream [list $property current [get_property $property $ip]]
            if {[catch {list_property_value $property $ip} values]} {
                puts $stream [list $property unavailable $values]
            } else {puts $stream [list $property allowed $values]}
        }
        close $stream
        set config [list CONFIG.PROTOCOL $protocol CONFIG.DATA_WIDTH $width CONFIG.ADDR_WIDTH $address \
                         CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3]
        if {$protocol eq "AXI4"} {lappend config CONFIG.ID_WIDTH 4}
        foreach {property value} $config {
            if {$property ni [list_property $ip]} {error "Missing $property"}
        }
        set_property -dict $config $ip
        foreach {property value} $config {
            if {[get_property $property $ip] ne $value} {error "Unexpected $property after customization"}
        }
        report_property -all $ip -file [file join $output ${name}_configured.rpt]
        generate_target all $ip
        puts "NTRPU_HOST_CLOCK_CONFIG name=$name protocol=$protocol data=$width address=$address async=1 sync_stages=3"
    }
    add_files [file normalize [file join $script_dir .. fpga rtl ntrpu_axil_cdc_recode.sv]]
    report_ip_status -file [file join $output ip_status.rpt]
    if {$mode in {sim sim_corrupt}} {
        add_files -fileset sim_1 [file normalize [file join $script_dir .. fpga tb tb_ntrpu_host_clock_recode.sv]]
        set_property top tb_ntrpu_host_clock_recode [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        foreach profile {0 1 2} {
            set options "-testplusarg PROFILE=$profile"
            if {$mode eq "sim_corrupt"} {append options " -testplusarg INJECT_RRESP"}
            set_property -dict [list xsim.simulate.xsim.more_options $options] [get_filesets sim_1]
            launch_simulation
            run all
            close_sim
            set log [file join $output project host_clock.sim sim_1 behav xsim simulate.log]
            file copy $log [file join $output profile-${profile}.log]
            set stream [open $log r];set text [read $stream];close $stream
            if {[regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Host clock bank failed profile=$profile"}
            if {![string match "*NTRPU_HOST_CLOCK_PASS banks=18 resets=8*" $text]} {error "Host clock bank incomplete"}
        }
        puts NTRPU_HOST_CLOCK_SUITE_RETURNED
    } elseif {$mode in {synth_axi synth_axil}} {
        # Replace only the standalone 100ns placeholder clocks, without
        # changing AMD's internal CDC constraints or generated files.
        set placeholders [get_files -all -quiet *_ooc.xdc]
        if {[llength $placeholders]!=1} {error "Expected one placeholder clock XDC file"}
        set_property IS_ENABLED false $placeholders
        set f [open [file join $output clocks.xdc] {WRONLY CREAT EXCL}]
        puts $f {create_clock -name host_clk -period 4.000 [get_ports s_axi_aclk]}
        puts $f {create_clock -name device_clk -period 5.000 [get_ports m_axi_aclk]}
        close $f
        read_xdc [file join $output clocks.xdc]
        set_property top $top [get_filesets sources_1]
        synth_design -mode out_of_context -top $top -part xcvu5p-flva2104-2-i
        if {[llength [get_clocks]]!=2 || [get_property PERIOD [get_clocks host_clk]]!=4.0 ||
            [get_property PERIOD [get_clocks device_clk]]!=5.0} {error "Clock identity"}
        if {[llength [get_cells -quiet -hier -filter {IS_BLACKBOX == 1}]]} {error "Unresolved black boxes"}
        if {[llength [get_cells -quiet -hier -filter {REF_NAME == LDCE || REF_NAME == LDPE}]]} {error "Unexpected latches"}
        write_checkpoint [file join $output ${top}_synth.dcp]
        report_utilization -hierarchical -file [file join $output hierarchy.rpt]
        report_timing_summary -report_unconstrained -file [file join $output timing.rpt]
        check_timing -verbose -file [file join $output check_timing.rpt]
        report_clocks -file [file join $output clocks.rpt]
        report_clock_interaction -file [file join $output clock_interaction.rpt]
        report_cdc -details -file [file join $output cdc.rpt]
        report_bus_skew -file [file join $output bus_skew.rpt]
        report_exceptions -coverage -file [file join $output exceptions.rpt]
        report_exceptions -ignored -file [file join $output exceptions_ignored.rpt]
        report_drc -file [file join $output drc.rpt]
        report_methodology -file [file join $output methodology.rpt]
        write_xdc -constraints all [file join $output applied.xdc]
        puts "NTRPU_HOST_CLOCK_SYNTH_EXPORTED top=$top host_period_ns=4 device_period_ns=5"
    }
    close_project
    puts NTRPU_HOST_CLOCK_PROBE_EXPORTED
}
if {[catch {ntrpu_host_recode $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
