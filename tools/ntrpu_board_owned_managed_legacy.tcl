set root [file normalize [file join [file dirname [info script]] ..]]
proc legacy_run {root output variant seed} {
    create_project legacy $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach module {ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_fragment_pool_bridge ntrpu_pcie_fragment_pool ntrpu_tagged_reset_mailbox ntrpu_ownership_certificate ntrpu_pcie_fragment_certified_owner ntrpu_drain_certificate_join ntrpu_owned_drain_coordinator ntrpu_pcie_fragment_dual_control ntrpu_pcie_fragment_dual_managed ntrpu_pcie_fragment_dual_coordinated} {add_files [file join $root fpga rtl $module.sv]}
    set top [expr {$variant eq "original" ? "tb_ntrpu_pcie_fragment_dual_managed" : "tb_ntrpu_dual_coordinated_legacy"}]
    add_files -fileset sim_1 [file join $root fpga tb $top.sv]
    set_property top $top [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    set_property xsim.simulate.xsim.more_options "-testplusarg SEED=$seed" [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation -scripts_only
    set simdir [file join $output legacy.sim sim_1 behav xsim]
    set f [open [file join $simdir $top.tcl] w];puts $f {run all;quit};close $f
    set previous [pwd];cd $simdir
    exec bash compile.sh >[file join $output compile-console.log] 2>@1
    exec bash elaborate.sh >[file join $output elaborate-console.log] 2>@1
    exec bash simulate.sh >[file join $output simulate-console.log] 2>@1
    cd $previous
    foreach leaf {simulate.log elaborate.log xvlog.log} {file copy [file join $simdir $leaf] [file join $output $leaf]}
    set f [open [file join $output simulate.log] r];set text [read $f];close $f
    if {![regexp -line {^CTL_PASS } $text] || ![regexp -line {^OWN_PASS } $text] || [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Managed legacy failed: $variant $seed"}
    puts "OWNED_MANAGED_LEGACY_PASS variant=$variant seed=$seed"
    close_project
}
proc main {root output} {
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    file mkdir $output
    foreach seed {1 7 42} {
        foreach variant {original coordinated} {
            legacy_run $root [file join $output $variant-$seed] $variant $seed
        }
    }
    puts NTRPU_OWNED_MANAGED_LEGACY_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {main $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
