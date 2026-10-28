# Three deliberate defects in private RTL copies, plus integrated-owner baseline.
set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set text [read $f];close $f;return $text}
proc replace_once {text old new} {
    set i [string first $old $text]
    if {$i<0 || [string first $old $text [expr {$i+1}]]>=0} {error "Mutation target not unique"}
    return [string replace $text $i [expr {$i+[string length $old]-1}] $new]
}
set body [read_text [file join $root tools ntrpu_board_ownership_certificate_r3.tcl]]
set cut [string first "\nif " $body]
if {$cut<0} {error "Missing profile boundary"}
eval [string range $body 0 [expr {$cut-1}]]
proc ownership_negatives {root output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    foreach case {baseline wrong_epoch live_owner no_revocation} {
        set private [file join $output inputs $case]
        foreach dir {rtl tb} {
            file mkdir [file join $private fpga $dir]
            foreach path [glob -directory [file join $root fpga $dir] *.sv] {
                file copy $path [file join $private fpga $dir [file tail $path]]
            }
        }
        set path [file join $private fpga rtl ntrpu_ownership_certificate.sv]
        set original [read_text $path];set rtl $original;set expected {}
        if {$case eq "wrong_epoch"} {
            set rtl [replace_once $rtl {wire identity_matches=peer_generation==collector_epoch && peer_tag!=0;} {wire identity_matches=peer_tag!=0;}]
            set expected CERTIFIED_WRONG_EPOCH
        } elseif {$case eq "live_owner"} {
            set rtl [replace_once $rtl {wire observed_invalid=session_lost && session_invalid && collector_epoch!=0 &&
        collector_epoch==bridge_epoch && !arm_ready && !retire_ready &&
        !line_ready && !(|cpl_ready);} {wire observed_invalid=collector_epoch!=0 && collector_epoch==bridge_epoch;}]
            set expected CERTIFIED_LIVE_OWNER
        } elseif {$case eq "no_revocation"} {
            set rtl [replace_once $rtl {if(ever_certified && (!observed_invalid || !ownership_stable))revoked_source<=1;} {if(ever_certified && (!observed_invalid || !ownership_stable))revoked_source<=revoked_source;}]
            set expected REVOCATION_NOT_PROPAGATED
        }
        if {$case ne "baseline" && $rtl eq $original} {error "Unchanged mutant"}
        set f [open $path w];puts -nonewline $f $rtl;close $f
        set profile [file join $output $case]
        set code [catch {ownership_certificate $private $profile} message options]
        set sim [file join $profile ownership.sim sim_1 behav xsim simulate.log]
        if {![file exists $sim]} {error "Simulation missing for $case: $message"}
        set text [read_text $sim]
        if {$case eq "baseline"} {
            if {$code} {error "Baseline failed: $message"}
            puts OWNERSHIP_NEGATIVE_BASELINE_PASS
        } else {
            if {!$code} {error "Undetected mutant $case"}
            set fatal [regexp -all -inline -line {^Fatal:.*$} $text]
            if {[llength $fatal]!=1 || ![regexp "^Fatal: OWN_CERT_CHECK id=\[0-9\]+ cycle=\[0-9\]+ $expected$" [lindex $fatal 0]]} {error "Unexpected mutant failure $case: $fatal"}
            if {[regexp -line {^OWNERSHIP_CERTIFICATE_SUITE_PASS } $text]} {error "Mutant reported PASS"}
            puts "OWNERSHIP_MUTANT_DETECTED case=$case reason=$expected"
        }
        set f [open [file join $output $case-result.txt] {WRONLY CREAT EXCL}]
        puts $f "case=$case\nexpected=$expected\nprofile_return=$code\nmessage=$message";close $f
        if {[llength [get_projects -quiet]]} {close_project}
    }
    puts NTRPU_OWNERSHIP_NEGATIVES_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {ownership_negatives $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
