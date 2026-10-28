set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc replace_once {text before after} {
    set at [string first $before $text]
    if {$at<0 || [string first $before $text [expr {$at+1}]]>=0} {error "Mutation anchor not unique"}
    return [string replace $text $at [expr {$at+[string length $before]-1}] $after]
}
proc section_mutants {output} {
    global ntrpu_script_dir
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set_param general.maxThreads 2
    set f [open [file join $ntrpu_script_dir .. fpga rtl ntrpu_axi4_section_certificate.sv] r]
    set original [read $f];close $f
    set mutations [list [list bypass_pair {assign section_ack=upstream_certificate && pair_certificate;} {assign section_ack=upstream_certificate;} {reviewer_only}] \
        [list extra_source_edge {.SIM_ASSERT_CHK(1),.SRC_INPUT_REG(0)) synchronize_fault(
            .src_clk(1'b0),.src_in(admission_fault_sources[i]),} {.SIM_ASSERT_CHK(1),.SRC_INPUT_REG(1)) synchronize_fault(
            .src_clk(host_clk),.src_in(admission_fault_sources[i]),} {registered admission alarm crosses without additional host edge}]]
    foreach mutation $mutations {
        lassign $mutation name before after expected
        set file [file join $output $name.sv]
        set f [open $file {WRONLY CREAT EXCL}]
        puts -nonewline $f [replace_once $original $before $after];close $f
        create_project pair [file join $output $name] -part xcvu5p-flva2104-2-i
        set_property XPM_LIBRARIES {XPM_CDC} [current_project]
        add_files $file
        foreach module {ntrpu_axi4_retained_admission ntrpu_axi4_response_guard ntrpu_axi4_pair_drain ntrpu_axi4_drain_monitor ntrpu_tagged_reset_mailbox} {
            add_files [file join $ntrpu_script_dir .. fpga rtl $module.sv]
        }
        create_ip -vlnv xilinx.com:ip:axi_clock_converter:2.1 -module_name ntrpu_axi4_cdc_probe
        set ip [get_ips ntrpu_axi4_cdc_probe]
        set_property generate_synth_checkpoint false [get_files [get_property IP_FILE $ip]]
        set_property -dict {CONFIG.PROTOCOL AXI4 CONFIG.DATA_WIDTH 512 CONFIG.ADDR_WIDTH 64 CONFIG.ID_WIDTH 4 CONFIG.ACLK_ASYNC 1 CONFIG.SYNCHRONIZATION_STAGES 3} $ip
        generate_target all $ip
        report_property -all -file [file join $output $name-properties.rpt] $ip
        report_ip_status -file [file join $output $name-ip.rpt]
        add_files -fileset sim_1 [file join $ntrpu_script_dir .. fpga tb tb_ntrpu_axi4_section.sv]
        set_property top tb_ntrpu_axi4_section [get_filesets sim_1]
        set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
        set_property generic {DEVICE_PERIOD_PS=5000} [get_filesets sim_1]
        update_compile_order -fileset sim_1
        launch_simulation
        catch {run all} diagnostic
        close_sim
        set log [file join $output $name pair.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output $name.log]
        set f [open $log r];set text [read $f];close $f
        if {$expected eq "reviewer_only"} {
            if {[string first "AXI4_SECTION_SUITE_PASS device_period_ps=5000" $text]<0 || [regexp {(^|\n)[ \t]*Fatal:} $text]} {error "Review-only mutant did not finish HDL bank"}
            puts "AXI4_SECTION_MUTANT_AWAITING_REVIEW name=$name"
        } else {
            if {[string first "Fatal: AXI4_SECTION_ASSERT phase=3 $expected" $text]<0 || [string first AXI4_SECTION_SUITE_PASS $text]>=0} {error "Mutant did not fail intended assertion"}
            puts "AXI4_SECTION_MUTANT_REJECTED name=$name"
        }
        close_project
    }
    puts NTRPU_AXI4_SECTION_MUTANTS_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {section_mutants [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
