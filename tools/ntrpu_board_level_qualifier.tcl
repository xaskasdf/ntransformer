set script_dir [file dirname [file normalize [info script]]]
proc level_test {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected sim|early|late NEW_DIRECTORY"}
    lassign $arguments mode output
    if {$mode ni {sim early late}} {error "Invalid mode"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    create_project level_test [file join $output project] -part xcvu5p-flva2104-2-i
    set_property XPM_LIBRARIES {XPM_CDC} [current_project]
    set original [file join $script_dir .. fpga rtl ntrpu_irq_transport.sv]
    set f [open $original r];set text [read $f];close $f
    if {$mode eq "early"} {
        if {[regsub {dest_level=level_ff\[2\]} $text {dest_level=level_ff[0]} text]!=1} {error "Negative edit mismatch"}
    }
    if {$mode eq "late"} {
        if {[regsub {posedge dest_clk or negedge src_level} $text {posedge dest_clk} text]!=1} {error "Negative edit mismatch"}
    }
    set copy [file join $output irq_transport_test.sv]
    set f [open $copy {WRONLY CREAT EXCL}];puts -nonewline $f $text;close $f
    add_files $copy
    add_files -fileset sim_1 [file join $script_dir .. fpga tb tb_ntrpu_level_qualifier.sv]
    set_property top tb_ntrpu_level_qualifier [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    foreach profile {0 1 2} {
        set_property generic PROFILE=$profile [get_filesets sim_1]
        launch_simulation
        run all
        close_sim
        set log [file join $output project level_test.sim sim_1 behav xsim simulate.log]
        file copy $log [file join $output profile-$profile.log]
        set f [open $log r];set text [read $f];close $f
        if {$mode eq "sim"} {
            if {![string match "*NTRPU_LEVEL_PASS profile=$profile pulses=96 clears=99*" $text] || [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} {error "Level qualifier failed"}
        } else {
            if {![string match {*Fatal: FAIL LEVEL*} $text] || [string match *NTRPU_LEVEL_PASS* $text]} {error "Negative did not reject modified RTL"}
        }
    }
    puts "NTRPU_LEVEL_SUITE mode=$mode profiles=3"
    close_project
}
if {[catch {level_test $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
