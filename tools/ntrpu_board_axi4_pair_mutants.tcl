set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc replace_once {text before after} {
    set at [string first $before $text]
    if {$at<0 || [string first $before $text [expr {$at+1}]]>=0} {error "Mutation anchor not unique"}
    return [string replace $text $at [expr {$at+[string length $before]-1}] $after]
}
proc axi4_pair_mutants {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set mutations [list \
        [list lost_device_r ntrpu_axi4_pair_drain {.src_in(device_fault_sources[i])} {.src_in((i==2)?1'b0:device_fault_sources[i])} {phase=3 device fault must report with source clock stopped}] \
        [list stale_mailbox ntrpu_tagged_reset_mailbox {assign ack_valid=certificate && current_request;} {assign ack_valid=certificate && req_valid;} {phase=1 new tag immediately masks cached proof}]]
    foreach mutation $mutations {
        lassign $mutation name target before after expected
        set f [open [file join $ntrpu_script_dir .. fpga rtl $target.sv] r]
        set original [read $f];close $f
        set mutated [file join $output $name.sv]
        set f [open $mutated {WRONLY CREAT EXCL}]
        puts -nonewline $f [replace_once $original $before $after];close $f
        set work [file join $output $name]
        create_project pair $work -part xcvu5p-flva2104-2-i
        set_property XPM_LIBRARIES {XPM_CDC} [current_project]
        foreach module {ntrpu_axi4_pair_drain ntrpu_axi4_drain_monitor ntrpu_tagged_reset_mailbox} {
            if {$module eq $target} {add_files $mutated} else {add_files [file join $ntrpu_script_dir .. fpga rtl $module.sv]}
        }
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_axi4_cdc_probe
        set ip [get_ips ntrpu_axi4_cdc_probe]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set_property -dict {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $ip
        generate_target all $ip
        report_property -all -file [file join $work cdc_properties.rpt] $ip
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_pair_drain_r2.sv]
        set_property top tb_ntrpu_axi4_pair_drain [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property generic {DEVICE_PERIOD_PS=5000} [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        catch {run all} diagnostic
        close_sim
        set log [file join $work pair.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output $name.log]
        set f [open $log r];set text [read $f];close $f
        if {[string first "Fatal: AXI4_PAIR_ASSERT $expected" $text]<0 || [string first AXI4_PAIR_SUITE_PASS $text]>=0} {error "Mutant $name did not fail at intended assertion"}
        close_project
        puts "AXI4_PAIR_MUTANT_REJECTED name=$name"
    }
    puts NTRPU_AXI4_PAIR_MUTANTS_REJECTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {axi4_pair_mutants [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
