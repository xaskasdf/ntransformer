# Configuration-only investigation. No implementation or driver changes.
set root [file dirname [file dirname [file normalize [info script]]]]
set out [file normalize [lindex $argv 0]]
set variant [lindex $argv 1]
if {$variant ni {baseline no_host_masters no_declared_bars user_bar_only user_type_only}} {error "Unknown variant"}
set_param general.maxThreads 1
if {[catch {
    file mkdir $out
    foreach variant [list $variant] {
        set work $out/$variant
        file mkdir $work/ip
        create_project probe $work/project -part xcvu47p-fsvh2892-2-e
        file copy $root/evidence/xdma.xci $work/ip/ntrpu_vu47p_xdma.xci
        exec python3 -B $root/tools/relocate_ntrpu_hbm_xci.py $work/ip/ntrpu_vu47p_xdma.xci
        read_ip $work/ip/ntrpu_vu47p_xdma.xci
        set ip [get_ips ntrpu_vu47p_xdma]
        if {$variant in {no_host_masters no_declared_bars}} {
            set_property -dict [list CONFIG.pf0_msix_enabled false CONFIG.axilite_master_en false CONFIG.en_axi_slave_if false] $ip
        }
        if {$variant eq "no_declared_bars"} {
            set_property -dict [list CONFIG.pf0_bar0_enabled false CONFIG.pf0_bar1_enabled false CONFIG.pf0_bar0_enabled_mqdma false] $ip
        }
        if {$variant eq "user_bar_only"} {
            set_property -dict [list CONFIG.pf0_msix_enabled false CONFIG.pf0_bar0_enabled false CONFIG.pf0_bar0_enabled_mqdma false CONFIG.pf0_bar1_enabled true CONFIG.pf0_bar1_enabled_mqdma true] $ip
        }
        if {$variant eq "user_type_only"} {
            set values [list_property_value CONFIG.pf0_bar0_type_mqdma $ip]
            puts "HOST_BAR_TYPES $values";flush stdout
            set choice {}
            foreach v $values {if {[regexp -nocase {axi.*lite} $v]} {set choice $v}}
            if {$choice eq ""} {error "No user AXI-Lite type exposed"}
            set_property CONFIG.pf0_msix_enabled false $ip
            set_property CONFIG.pf0_bar0_type_mqdma $choice $ip
        }
        generate_target instantiation_template $ip
        report_property -all -file $work/properties.rpt $ip
        puts "HOST_BAR_PROBE_VARIANT $variant";flush stdout
        close_project
    }
    puts HOST_BAR_PROBE_FINISHED
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
