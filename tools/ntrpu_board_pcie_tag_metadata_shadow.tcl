set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc metadata_shadow {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project metadata_shadow $output -part xcvu5p-flva2104-2-i
    foreach name {ntrpu_pcie_request_inventory_r7.sv ntrpu_pcie_tag_metadata.sv} {
        add_files [file join $ntrpu_script_dir .. fpga rtl $name]
    }
    add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_pcie_tag_metadata_shadow.sv]
    set_property top tb_ntrpu_pcie_tag_metadata_shadow [get_filesets sim_1]
    set vectors [file normalize [file join $ntrpu_script_dir .. fpga tb data pcie_request_inventory_r7.txt]]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    foreach mode {normal omit_rq reverse_writes} {
        set args "-testplusarg VECTORS=$vectors"
        if {$mode eq "omit_rq"} {append args " -testplusarg OMIT_RQ"}
        if {$mode eq "reverse_writes"} {append args " -testplusarg REVERSE_WRITES"}
        set_property xsim.simulate.xsim.more_options $args [get_filesets sim_1]
        launch_simulation
        set caught [catch {run all} message]
        close_sim
        set target [file join $output $mode]
        file mkdir $target
        set logfile [file join $output metadata_shadow.sim sim_1 behav xsim simulate.log]
        file copy $logfile [file join $target simulate.log]
        set f [open $logfile r];set text [read $f];close $f
        if {$mode eq "normal"} {
            if {$caught || [string first "META_SHADOW_SUITE_PASS rows=22339 scenarios=68 mode=0" $text]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:|FAIL )} $text]} {error "Normal metadata shadow failed: $message"}
        } else {
            if {[string first "META_SHADOW_SUITE_PASS" $text]>=0 || ![regexp {Fatal:.*META_SHADOW_(LOOKUP_)?MISMATCH} $text]} {error "Expected shadow mutation rejection missing: $mode"}
        }
        puts "NTRPU_METADATA_SHADOW_CASE_REVIEWED $mode"
    }
    close_project
    puts NTRPU_PCIE_TAG_METADATA_SHADOW_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {metadata_shadow [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
