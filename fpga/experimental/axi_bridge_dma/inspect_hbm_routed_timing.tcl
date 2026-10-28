# Read-only diagnosis of the sealed, fully routed VU47P experiment.
set_param general.maxThreads 2

proc ht_table {out label paths} {
    set f [open $out/${label}.tsv {WRONLY CREAT EXCL}]
    set fields {STARTPOINT_PIN ENDPOINT_PIN STARTPOINT_CLOCK ENDPOINT_CLOCK GROUP DELAY_TYPE SLACK REQUIREMENT DATAPATH_DELAY SKEW CLOCK_PESSIMISM LOGIC_LEVELS EXCEPTION}
    puts $f [join $fields "\t"]
    foreach p $paths {
        set values {}
        foreach k $fields {lappend values [get_property $k $p]}
        puts $f [join $values "\t"]
    }
    close $f
    if {[llength $paths]} {
        # Path objects carry their report options. Do not add ignored/deprecated flags.
        report_timing -of_objects $paths -file $out/${label}.rpt
        set f [open $out/${label}_properties.tsv {WRONLY CREAT EXCL}]
        puts $f "property\tvalue"
        foreach k [lsort [list_property [lindex $paths 0]]] {
            puts $f "$k\t[get_property $k [lindex $paths 0]]"
        }
        close $f
    }
}

proc ht_run {checkpoint out} {
    if {[file exists $out]} {error "Fresh timing output required"}
    file mkdir $out
    open_checkpoint $checkpoint
    if {[get_property TOP [current_design]] ne "ntrpu_axi_bridge_hbm_controlled" ||
        [get_property PART [current_design]] ne "xcvu47p-fsvh2892-2-e"} {error "Wrong board checkpoint"}
    write_xdc -constraints all $out/before.xdc
    # Vivado writes help to the captured tool log; it has no generic redirect command.
    help get_timing_paths
    help phys_opt_design
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing_before_methodology.rpt
    ht_table $out worst_setup [get_timing_paths -delay_type max -max_paths 128 -nworst 1]
    ht_table $out worst_hold [get_timing_paths -delay_type min -max_paths 8 -nworst 1]
    set pcie {ntrpu_bridge_i/inst/pcie4c_ip_i/inst/ntrpu_bridge_pcie4c_ip_pcie_4_0_pipe_inst/pcie_4_c_e4_inst/CORECLK}
    set targets [list \
        hbm_response {} {core/reader/channels[0].adapter/tracker/response_cut/slot1_reg[230]/CE} \
        hbm_recovery {core/reader/startup/memory_release_reg[2]/C} {hbm_binding/memory/inst/TWO_STACK.u_hbm_top/TWO_STACK_HBM.hbm_two_stack_intf/HBM_ONE_STACK_INTF<1>_INST/HBM_SNGLBLI_INTF_AXI<15>_INST/ARESET_N} \
        transport_return {core/endpoint/source_join/tag_reg[30]/C} {core/transport/lanes[1].returns/output_data_reg[206]/CE} \
        pcie_msi23 $pcie {ntrpu_bridge_i/inst/udma_wrapper/dma_top/base/IRQ_INST/clk_msi_req_reg[23]/R} \
        pcie_msi10 $pcie {ntrpu_bridge_i/inst/udma_wrapper/dma_top/base/IRQ_INST/clk_msi_req_reg[10]/R} \
        pcie_slave $pcie {ntrpu_bridge_i/inst/udma_wrapper/dma_top/axi4mm_bridge_top_inst/mm_slave_port.axi4mm_axi_slave_read_inst/FSM_sequential_read_reqSM_cs_reg[0]/CE}]
    foreach {label from to} $targets {
        set dst [get_pins $to]
        if {[llength $dst]!=1} {error "Missing exact endpoint $to"}
        set query [list -to $dst -delay_type max -max_paths 4 -nworst 2]
        if {$from ne ""} {
            set src [get_pins $from]
            if {[llength $src]!=1} {error "Missing exact startpoint $from"}
            lappend query -from $src
        }
        ht_table $out $label [get_timing_paths {*}$query]
        if {[string match pcie_* $label]} {
            ht_table $out ${label}_ignored [get_timing_paths -user_ignored {*}$query]
        }
        set f [open $out/${label}_endpoint.tsv {WRONLY CREAT EXCL}]
        puts $f "object\tproperty\tvalue"
        foreach obj [get_cells -of_objects $dst] {
            foreach k {REF_NAME LOC BEL DONT_TOUCH ASYNC_REG} {puts $f "$obj\t$k\t[get_property $k $obj]"}
        }
        foreach net [get_nets -of_objects $dst] {
            puts $f "$net\tLEAF_PIN_COUNT\t[llength [get_pins -leaf -of_objects $net]]"
            puts $f "$net\tLEAF_DRIVERS\t[get_pins -leaf -of_objects $net -filter {DIRECTION == OUT}]"
        }
        close $f
    }
    set f [open $out/clocks.tsv {WRONLY CREAT EXCL}]
    puts $f "clock\tproperty\tvalue"
    foreach clk [get_clocks] {
        foreach k [lsort [list_property $clk]] {puts $f "$clk\t$k\t[get_property $k $clk]"}
    }
    close $f
    report_methodology -file $out/methodology_fresh.rpt
    report_timing_summary -delay_type min_max -report_unconstrained -file $out/timing_after_methodology.rpt
    ht_table $out pcie_msi23_after [get_timing_paths -from [get_pins $pcie] -to [get_pins {ntrpu_bridge_i/inst/udma_wrapper/dma_top/base/IRQ_INST/clk_msi_req_reg[23]/R}] -delay_type max -max_paths 4 -nworst 2]
    report_route_status -file $out/route_status.rpt
    write_xdc -constraints all $out/after.xdc
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}]
    puts $f HBM_ROUTED_TIMING_DIAGNOSTICS_EXPORTED;close $f
    close_design
}
