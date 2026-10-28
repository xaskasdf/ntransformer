set root [file normalize [file join [file dirname [info script]] ..]]
proc ownership_certificate {root output} {
    if {[file exists $output]} {error "Output exists"}
    set_param general.maxThreads 2
    create_project ownership $output -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    foreach module {ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_fragment_pool_bridge ntrpu_pcie_fragment_pool ntrpu_tagged_reset_mailbox ntrpu_ownership_certificate ntrpu_pcie_fragment_certified_owner} {
        add_files [file join $root fpga rtl $module.sv]
    }
    foreach module {ntrpu_ownership_certificate_case_r2 tb_ntrpu_ownership_certificate_r2} {
        add_files -fileset sim_1 [file join $root fpga tb $module.sv]
    }
    set_property top tb_ntrpu_ownership_certificate_r2 [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    update_compile_order -fileset sim_1
    launch_simulation -scripts_only
    set simdir [file join $output ownership.sim sim_1 behav xsim]
    set f [open [file join $simdir tb_ntrpu_ownership_certificate_r2.tcl] w]
    puts $f [list open_vcd [file join $output ownership.vcd]]
    puts $f {if {[catch {
        set objects [get_objects -r /tb_ntrpu_ownership_certificate_r2/*]
        if {[llength $objects]<500} {error "Incomplete ownership VCD scope"}
        puts "OWNERSHIP_CAPTURE_OBJECTS [llength $objects]"
        log_vcd $objects
        run all
        flush_vcd
        close_vcd
    } message]} {puts stderr "OWNERSHIP_CAPTURE_FATAL $message";catch {close_vcd};quit}}
    puts $f {quit}
    close $f
    set previous [pwd];cd $simdir
    exec bash compile.sh >[file join $output compile-console.log] 2>@1
    exec bash elaborate.sh >[file join $output elaborate-console.log] 2>@1
    exec bash simulate.sh >[file join $output simulate-console.log] 2>@1
    cd $previous
    foreach leaf {simulate.log elaborate.log xvlog.log} {file copy [file join $simdir $leaf] [file join $output $leaf]}
    set f [open [file join $output simulate.log] r];set text [read $f];close $f
    set marker "OWNERSHIP_CERTIFICATE_SUITE_PASS cases=12 real_owner=1 native_xpm=1 reset_driven=0"
    if {[lsearch -exact [split $text \n] $marker]<0 || [regexp -nocase {(^|\n)[ \t]*(Fatal:|Failure:|ERROR:)} $text]} {error "Ownership certificate simulation failed"}
    set f [open [file join $output ownership.vcd] r];set wave [read $f];close $f
    if {[regexp -all {\$var } $wave]<500 || [file size [file join $output ownership.vcd]]<10000} {error "Empty VCD"}
    close_project
    puts NTRPU_OWNERSHIP_CERTIFICATE_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {ownership_certificate $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
