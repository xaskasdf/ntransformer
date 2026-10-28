set_param general.maxThreads 2
if {[catch {
    if {[llength $argv]!=3} {error "SOURCE SYNTH_DCP NEW_OUTPUT required"}
    lassign $argv source dcp out
    if {[file exists $out]} {error "New output required"};file mkdir $out
    open_checkpoint $dcp
    if {[get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e" || [get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_cdc"} {error "Unexpected donor"}
    foreach {name period} {pcie_clk 4.0 reader_clk 5.0 mgmt_clk 10.0} {
        if {[get_property PERIOD [get_clocks $name]]!=$period} {error "Clock mismatch $name"}
    }
    set f [open $out/context.tsv {WRONLY CREAT EXCL}];puts $f "port\tclock\tperiod_ns\tassumed_external_bufg"
    foreach {port clock period site} {p_clk pcie_clk 4.0 BUFGCE_X0Y60 r_clk reader_clk 5.0 BUFGCE_X0Y61 mgmt_clk mgmt_clk 10.0 BUFGCE_X0Y62} {
        if {[llength [get_sites -quiet $site]]!=1 || [get_property SITE_TYPE [get_sites $site]] ne "BUFGCE"} {error "Unavailable OOC clock site $site"}
        set_property HD.CLK_SRC $site [get_ports $port]
        puts $f "$port\t$clock\t$period\t$site"
    }
    close $f
    source $source/constrain_hbm_cdc_physical.tcl
    source $source/audit_hbm_cdc_physical.tcl
    hc_constrain $out
    write_xdc -constraints all $out/before_opt.xdc
    opt_design
    place_design
    write_checkpoint $out/placed.dcp
    route_design
    write_checkpoint $out/routed.dcp
    hc_audit $out/audit
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_CDC_OOC_ROUTE_EXPORTED_PENDING_REVIEW;close $f
    close_design
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
