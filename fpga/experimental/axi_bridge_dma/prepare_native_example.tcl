# Private example only. Arguments: immutable seed XCI, NEW output directory.
set_param general.maxThreads 2
if {[catch {
    if {[llength $argv]!=2 || [llength [get_projects -quiet]]} {error "Use fresh process: SEED_XCI NEW_OUTPUT"}
    lassign $argv seed out
    set seed [file normalize $seed];set out [file normalize $out]
    if {![file isfile $seed] || [file exists $out]} {error "Invalid seed/output"}
    create_project bridge_native $out/project -part xcvu47p-fsvh2892-2-e
    import_ip -files $seed
    set ip [get_ips ntrpu_bridge]
    set owned [file normalize [get_property IP_FILE $ip]]
    if {[string first "$out/" $owned]!=0} {error "Imported IP outside private output: $owned"}
    foreach {key value} {CONFIG.functional_mode AXI_Bridge CONFIG.axi_addr_width 64 CONFIG.axi_data_width 512_bit CONFIG.pipe_sim true CONFIG.xdma_axilite_slave true} {
        if {[get_property $key $ip] ne $value} {error "Wrong $key"}
    }
    generate_target {synthesis simulation} $ip
    open_example_project -dir $out/example $ip
    # AMD creates the example in a separate process; current_project remains
    # the generator project. Record the actual example, not current_project.
    set example $out/example/ntrpu_bridge_ex/ntrpu_bridge_ex.xpr
    if {![file isfile $example]} {error "Example project was not created"}
    set f [open $out/example_project.txt {WRONLY CREAT EXCL}]
    puts $f $example
    close $f
    report_compile_order -used_in simulation -file $out/compile_order.rpt
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
puts BRIDGE_NATIVE_EXAMPLE_PREPARED
exit 0
