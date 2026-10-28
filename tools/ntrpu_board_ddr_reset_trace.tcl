# Replay an existing DDR XSIM snapshot and trace only public reset/clock signals.
# A diagnostic window, not a completed functional test. Do not alter the model.
proc ntrpu_ddr_reset_trace {arguments} {
    if {[llength $arguments] ni {2 3}} { error "Expected INPUT_XSIM_DIRECTORY NEW_DIRECTORY ?commands?" }
    lassign $arguments input output mode
    if {$mode ni {{} commands}} { error "Unknown trace mode" }
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    if {![file exists [file join $input xsim.dir ntrpu_ddr_capacity_tb_behav xsimk]]} {
        error "Missing completed DDR simulation snapshot"
    }
    file mkdir $output
    foreach name {xsim.dir xsim.ini temp_mem.txt temp_second_mem.txt} {
        file copy [file join $input $name] [file join $output $name]
    }
    set stream [open [file join $output origin.txt] {WRONLY CREAT EXCL}]
    puts $stream $input
    puts $stream "Reuses compiled snapshot without HDL or model changes; stops at 55 us for diagnosis."
    close $stream
    set stream [open [file join $output trace.tcl] {WRONLY CREAT EXCL}]
    puts $stream "set capture_commands [expr {$mode eq {commands}}]"
    puts $stream {
        if {[catch {
        set root [current_scope]
        puts "NTRPU_DDR_TRACE_ROOT $root"
        open_vcd reset_trace.vcd
        foreach name {sys_rst reset_n ck_t cke calibrated ui_rst} {
            set objects [get_objects $name]
            if {[llength $objects] != 1} { error "Missing/ambiguous public trace signal: $name" }
            log_vcd $objects
            puts "NTRPU_DDR_TRACE_SIGNAL $objects"
        }
        set children [get_scopes]
        foreach suffix {{dimms[0].memory} {dimms[1].memory}} {
            set candidates {}
            foreach scope $children {
                if {[string first $suffix $scope] >= 0} { lappend candidates $scope }
            }
            if {[llength $candidates] != 1} { error "Missing/ambiguous public memory scope: $suffix" }
            current_scope [lindex $candidates 0]
            foreach name {reset_n ck} {
                set objects [get_objects $name]
                if {[llength $objects] != 1} { error "Missing/ambiguous public signal: $suffix/$name" }
                log_vcd $objects
                puts "NTRPU_DDR_TRACE_SIGNAL $objects"
            }
            foreach name {RESET_n CK CKE} {
                set public_objects {}
                foreach object [get_objects -r $name] {
                    if {[string first iDDR4 $object] >= 0} { lappend public_objects $object }
                }
                puts "NTRPU_DDR_TRACE_COMPONENTS slot=$suffix signal=$name count=[llength $public_objects]"
                if {[llength $public_objects] != 36} { error "Expected 36 public component interfaces per slot: $name" }
                log_vcd $public_objects
            }
            if {$capture_commands && $suffix eq {dimms[0].memory}} {
                set matches 0
                foreach object $public_objects {
                    if {[string first {rank_instances[0]} $object] >= 0 &&
                        [string first {instance_of_sdram_devices[0]} $object] >= 0} {
                        incr matches
                        if {![string match */CKE $object]} { error "Unexpected component object path" }
                        current_scope [string range $object 0 end-4]
                        foreach name {CS_n ACT_n RAS_n_A16 CAS_n_A15 WE_n_A14 BG BA ADDR ADDR_17 ODT TEN PARITY} {
                            set signals [get_objects $name]
                            if {[llength $signals] != 1} { error "Missing command signal $name" }
                            log_vcd $signals
                        }
                        puts "NTRPU_DDR_COMMAND_TRACE_SCOPE [current_scope]"
                    }
                }
                if {$matches != 1} { error "Expected one representative chip command interface" }
            }
            current_scope $root
        }
        run 55 us
        flush_vcd
        close_vcd
        puts "NTRPU_DDR_TRACE_WINDOW_COMPLETE time=[current_time -s]"
        set done [open trace_complete.txt {WRONLY CREAT EXCL}]
        puts $done [current_time -s]
        close $done
        } message]} {
            puts "NTRPU_DDR_TRACE_ERROR $message"
        }
        quit
    }
    close $stream
    cd $output
    exec xsim ntrpu_ddr_capacity_tb_behav -tclbatch trace.tcl -log simulate.log > trace.console.log 2>@1
    if {![file exists trace_complete.txt]} { error "Trace incomplete; inspect trace.console.log" }
    puts "NTRPU_DDR_TRACE_RETURNED; inspect model diagnostics and VCD, not functional PASS"
}
if {[catch {ntrpu_ddr_reset_trace $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
