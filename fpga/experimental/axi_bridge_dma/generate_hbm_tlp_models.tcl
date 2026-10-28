# Only generate simulation models in a new private project; do not open a donor.
set_param general.maxThreads 2
if {[catch {
    lassign $argv source out
    if {[llength $argv]!=2 || [file exists $out]} {error "SOURCE NEW_OUTPUT required"}
    file mkdir $out
    create_project hbm_tlp_models $out/project -part xcvu47p-fsvh2892-2-e
    foreach name {native_hbm fabric_clocks hbm_read_cdc hbm_read_width hbm_read_protocol} {
        import_ip -files $source/$name.xci
        set ip [get_ips $name]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        generate_target simulation $ip
        set f [open $out/${name}_config.tsv {WRONLY CREAT EXCL}]
        foreach key [lsort [list_property $ip]] {
            if {[string match CONFIG.* $key]} {puts $f "$key\t[get_property $key $ip]"}
        };close $f
    }
    # Explicit roots prevent automatic top selection from retaining only clocks.
    set f [open $out/model_inventory.v {WRONLY CREAT EXCL}]
    puts $f {module model_inventory; native_hbm memory(); fabric_clocks clocks(); hbm_read_cdc cdc(); hbm_read_width width_converter(); hbm_read_protocol protocol_converter(); endmodule}
    close $f
    add_files $out/model_inventory.v
    set_property top model_inventory [get_filesets sources_1]
    set_property top model_inventory [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    set f [open $out/compile_order.tsv {WRONLY CREAT EXCL}]
    foreach path [get_files -compile_order sources -used_in simulation] {
        puts $f "[get_property LIBRARY $path]\t[get_property FILE_TYPE $path]\t[file normalize $path]"
    };close $f
    report_compile_order -used_in simulation -file $out/compile_order.rpt
    set f [open $out/STATUS.txt {WRONLY CREAT EXCL}];puts $f HBM_TLP_MODELS_EXPORTED;close $f
    close_project
} message options]} {puts stderr $message;puts stderr [dict get $options -errorinfo];exit 1}
exit 0
