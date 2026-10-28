set script_dir [file dirname [file normalize [info script]]]
proc capacity_guard {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected sim|synth NEW_DIRECTORY"}
    lassign $arguments mode output
    if {$mode ni {sim synth}} {error "Unknown mode"}
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output;cd $output
    set rtl [file normalize [file join $script_dir .. fpga rtl ntrpu_axi_capacity_guard.sv]]
    set tb [file normalize [file join $script_dir .. fpga tb tb_ntrpu_axi_capacity_guard.sv]]
    if {$mode eq "sim"} {
        set f [open $rtl r];set original [read $f];close $f
        foreach variant {normal range strobes admit} {
            set path [file join $output $variant];file mkdir $path;cd $path
            set source $rtl
            if {$variant ne "normal"} {
                switch $variant {
                    range {
                        set before {page_end<=15'd4096 && {1'b0,address}<CAPACITY && end_address<=CAPACITY}
                        set after {page_end<=15'd4096}
                    }
                    strobes {
                        set before {w_strb[w_tail]<=s_axi_wstrb;}
                        set after {w_strb[w_tail]<='1;}
                    }
                    admit {
                        set before {m_axi_awvalid=rst_n && aw_pending;}
                        set after {m_axi_awvalid=rst_n && aw_pending && admit;}
                    }
                }
                if {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]!=2} {error "Mutation match count: $variant"}
                set source [file join $path mutant.sv]
                set f [open $source {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
            }
            exec xvlog --sv $source $tb > compile.log 2>@1
            exec xelab tb_ntrpu_axi_capacity_guard --debug typical --mt 4 -s capacity_guard > elaborate.log 2>@1
            set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
            set seeds [expr {$variant eq "normal" ? {1 7 42}:{1}}]
            foreach seed $seeds {
                set code [catch {exec xsim capacity_guard -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
                set f [open seed-${seed}.console.log r];set text [read $f];close $f
                if {$variant eq "normal"} {
                    if {$code || ![regexp -line {^NTRPU_CAP_PASS cases=20 allowed=10 rejected=10 sw=429 mw=150 sr=429 mr=150 } $text] ||
                        [regexp -nocase -line {^(Fatal:|ERROR:|FAIL |CRITICAL WARNING:)} $text]} {error "Capacity positive failed seed=$seed: $message"}
                } else {
                    set expected [dict get {range CAP_REJECT_EXPOSED strobes CAP_WRITE_PAYLOAD admit CAP_STABILITY} $variant]
                    set pattern [format {^Fatal: FAIL CAP .*%s} $expected]
                    if {![regexp -line $pattern $text] || [regexp -line {^NTRPU_CAP_PASS } $text]} {
                        error "Capacity mutant was not detected: $variant"
                    }
                }
                puts "NTRPU_CAP_REVIEW variant=$variant seed=$seed PASS"
            }
        }
        puts NTRPU_CAP_SUITE_PASS
    } else {
        create_project capacity_guard [file join $output project] -part xcvu5p-flva2104-2-i
        add_files $rtl
        set_property top ntrpu_axi_capacity_guard [current_fileset]
        set f [open clock.xdc {WRONLY CREAT EXCL}]
        puts $f {create_clock -name clk -period 5.000 [get_ports clk]};close $f
        read_xdc clock.xdc
        synth_design -top ntrpu_axi_capacity_guard -mode out_of_context -part xcvu5p-flva2104-2-i
        if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE || IS_BLACKBOX == 1}]]} {error "Latch or unresolved cell"}
        write_checkpoint [file join $output capacity_guard_synth.dcp]
        report_utilization -hierarchical -file utilization.rpt
        report_timing_summary -report_unconstrained -file timing.rpt
        check_timing -verbose -file check_timing.rpt
        report_drc -file drc.rpt
        report_methodology -file methodology.rpt
        write_xdc -constraints all applied.xdc
        puts NTRPU_CAP_SYNTH_EXPORTED
        close_project
    }
}
if {[catch {capacity_guard $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
