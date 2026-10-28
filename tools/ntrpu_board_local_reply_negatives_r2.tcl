# Native controls against private mutations. Original sources remain untouched.
set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set text [read $f];close $f;return $text}
proc write_text {path text} {set f [open $path {WRONLY CREAT EXCL}];puts -nonewline $f $text;close $f}
proc replace_once {text old new} {
    set i [string first $old $text]
    if {$i < 0 || [string first $old $text [expr {$i+1}]] >= 0} {error "Mutation target is not unique: $old"}
    return [string replace $text $i [expr {$i+[string length $old]-1}] $new]
}
proc load_profile {root script} {
    set body [read_text [file join $root tools $script]]
    set cut [string first "\nif " $body]
    if {$cut < 0} {error "Cannot find isolated profile boundary"}
    # Evaluate only the immutable profile procedure, excluding its main loop.
    uplevel #0 [string range $body 0 [expr {$cut-1}]]
}
proc negatives {root output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    foreach case {baseline ignore_pending wrong_ack_tag shared_disconnected} {
        set private [file join $output inputs $case]
        foreach dir {rtl tb} {
            file mkdir [file join $private fpga $dir]
            foreach file [glob -directory [file join $root fpga $dir] *.sv] {
                file copy $file [file join $private fpga $dir [file tail $file]]
            }
        }
        set path [file join $private fpga rtl ntrpu_local_reply_certificate_r2.sv]
        set original [read_text $path]
        set rtl $original
        set expected {}
        if {$case eq "ignore_pending"} {
            set rtl [replace_once $rtl {wire observed_idle=local_idle && !request_present && !reply_pending && !(|source_faults);} {wire observed_idle=!request_present && !(|source_faults);}]
            set expected {NEGATIVE_COMPLETE_WITH_LOCAL_OBLIGATION}
        } elseif {$case eq "wrong_ack_tag"} {
            set rtl [replace_once $rtl {.ack_tag(ack_tag)} {.ack_tag()}]
            set rtl [replace_once $rtl {endmodule} {assign ack_tag=64'hbad;
endmodule}]
            set expected {NEGATIVE_ACK_IDENTITY}
        } elseif {$case eq "shared_disconnected"} {
            set rtl [replace_once $rtl {wire [1:0] synchronized_faults={synchronized_revocation,section_fault_mgmt};} {wire [1:0] synchronized_faults={synchronized_revocation,1'b0};}]
            set expected {shared section fault reaches local provider with host stopped}
        }
        if {$case ne "baseline" && $rtl eq $original} {error "Mutation made no change"}
        set f [open $path w];puts -nonewline $f $rtl;close $f
        if {$case eq "shared_disconnected"} {
            load_profile $root ntrpu_board_drain_shared_fault_r2.tcl
            set external 0
        } else {
            load_profile $root ntrpu_board_drain_sections_r6.tcl
            set external 1
            set tbpath [file join $private fpga tb tb_ntrpu_drain_sections_r2.sv]
            set tb [read_text $tbpath]
            set monitor {    // Negative-control monitors: public obligations and identity only.
    // The certificate must be empty when first asserted. A later source
    // violation crosses the real three-stage CDC; the existing bank checks
    // that bounded revocation and the independent VCD review measures it.
    always @(posedge complete)begin
        #0.001;
        if(cold)must(local_idle==3 && requests_present==0 && replies_pending==0,
            "NEGATIVE_COMPLETE_WITH_LOCAL_OBLIGATION");
    end
    always @(posedge mgmt_clk)if(cold)begin
        for(integer slot=5;slot<7;slot=slot+1)if(ack[slot])
            must(ack_gen[slot*64+:64]==pg && ack_tag[slot*64+:64]==pt,
                "NEGATIVE_ACK_IDENTITY");
    end
endmodule}
            set tb [replace_once $tb {endmodule} $monitor]
            set f [open $tbpath w];puts -nonewline $f $tb;close $f
        }
        set profile [file join $output $case]
        set code [catch {drain_profile $private $profile 5000 $external} message options]
        set sim [file join $profile sections.sim sim_1 behav xsim simulate.log]
        if {![file exists $sim]} {error "Native simulation log missing for $case: $message"}
        set text [read_text $sim]
        if {$case eq "baseline"} {
            if {$code} {error "Baseline failed: $message"}
            puts "LOCAL_REPLY_NEGATIVE_BASELINE_PASS"
        } else {
            if {!$code} {error "Mutant was not rejected: $case"}
            if {[string first $expected $text] < 0 || ![regexp -line {^Fatal:.*DRAIN_SECTIONS_ASSERT} $text]} {
                error "Mutant failed for an unintended reason: $case: $message"
            }
            if {[regexp -line {^DRAIN_(SECTIONS_R2|SHARED_FAULT)_PASS } $text]} {error "Mutant reported PASS"}
            puts "LOCAL_REPLY_MUTANT_DETECTED case=$case reason=$expected"
        }
        write_text [file join $output $case-result.txt] "case=$case\nexpected=$expected\nprofile_return=$code\nmessage=$message\n"
        if {[llength [get_projects -quiet]]} {close_project}
    }
    puts NTRPU_LOCAL_REPLY_NEGATIVES_R2_EXPORTED
}
if {[llength $argv] != 1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {negatives $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
