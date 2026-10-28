# Full vocabulary arithmetic using the original generated dual-MAC datapath.
# Synthesis belongs to the already queued parent; this batch is behavioral RTL.
set vocab_script_dir [file dirname [file normalize [info script]]]
proc vocab_arithmetic_bank {output fixture} {
    global vocab_script_dir
    set output [file normalize $output]
    set fixture [file normalize $fixture]
    if {[file exists $output] || ![file isfile [file join $fixture manifest.json]]} {error "Output exists or fixture missing"}
    file mkdir $output
    set_param general.maxThreads 2
    create_project vocab_arithmetic [file join $output project] -part xcvu5p-flva2104-2-i
    set rtl [file join $output mac_rtl]
    file copy [file join $vocab_script_dir .. generated mac] $rtl
    cd $rtl
    add_files [lsort [glob *.v]]
    foreach script [lsort [glob *_ip.tcl]] {source $script}
    foreach name {ntrpu_fp64_to_fp32 ntrpu_axi_read_master ntrpu_weight_tile_buffer ntrpu_q8_unpacker ntrpu_q8_weight_stream ntrpu_q8_gemv_parallel ntrpu_activation_loader ntrpu_q8_gemv_memory ntrpu_result_writer ntrpu_q8_gemv_store ntrpu_matrix_sequencer ntrpu_q8_matrix_memory ntrpu_llama32_vocab_control_r1 ntrpu_llama32_vocab_memory_r1} {
        add_files [file normalize [file join $vocab_script_dir .. fpga rtl ${name}.sv]]
    }
    add_files -fileset sim_1 [file normalize [file join $vocab_script_dir .. fpga tb tb_ntrpu_llama32_vocab_memory_r1.sv]]
    set_property top ntrpu_llama32_vocab_memory_r1 [get_filesets sources_1]
    set_property top tb_ntrpu_llama32_vocab_memory_r1 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.elaborate.debug_level off [get_filesets sim_1]
    set_property xsim.simulate.custom_tcl [file join $vocab_script_dir ntrpu_xsim_no_autowave_r1.tcl] [get_filesets sim_1]
    set_property xsim.simulate.xsim.more_options "-testplusarg SEED=1 -testplusarg FIXTURE=$fixture" [get_filesets sim_1]
    update_compile_order -fileset sources_1
    update_compile_order -fileset sim_1
    report_ip_status -file [file join $output ip_status.rpt]
    launch_simulation
    run all
    close_sim
    set sim [file join $output project vocab_arithmetic.sim sim_1 behav xsim]
    file copy [file join $sim simulate.log] [file join $output arithmetic.log]
    foreach name [concat {vocab-result.hex} [lmap phase {0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15} {format "vocab-phase-%02d.hex" $phase}]] {
        if {![file isfile [file join $sim $name]] || [file size [file join $sim $name]] == 0} {error "Missing output $name"}
        file copy [file join $sim $name] [file join $output $name]
    }
    set f [open [file join $output arithmetic.log] r]
    set text [read $f]
    close $f
    if {[string first "VOCABM_PASS seed=1 jobs=3 publics=3 source_acks=3 cancels=1 matrices=17 groups=4009 results=128256 " $text] < 0 ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL |VOCABM_FAIL)} $text]} {error "Full vocabulary arithmetic failed"}
    close_project
    puts NTRPU_VOCAB_ARITHMETIC_R1_EXPORTED
}
if {[llength $argv] != 2 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {vocab_arithmetic_bank {*}$argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
