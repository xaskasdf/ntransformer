# Native controls for actual ownership acceptance and independently absent peers.
set root [file normalize [file join [file dirname [info script]] ..]]
proc read_text {path} {set f [open $path r];set text [read $f];close $f;return $text}
proc replace_once {text old new} {
    set i [string first $old $text]
    if {$i<0 || [string first $old $text [expr {$i+1}]]>=0} {error "Nonunique mutation target"}
    return [string replace $text $i [expr {$i+[string length $old]-1}] $new]
}
set body [read_text [file join $root tools ntrpu_board_drain_owned_sections.tcl]]
set cut [string first "\nif " $body]
if {$cut<0} {error "Missing profile boundary"}
eval [string range $body 0 [expr {$cut-1}]]
proc owned_negatives {root output} {
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    foreach case {baseline_missing baseline_control ignore_owner ignore_host ignore_xdma} {
        set mode [expr {$case in {baseline_control ignore_owner} ? 3 : 0}]
        set private [file join $output inputs $case]
        foreach dir {rtl tb} {
            file mkdir [file join $private fpga $dir]
            foreach path [glob -directory [file join $root fpga $dir] *.sv] {
                file copy $path [file join $private fpga $dir [file tail $path]]
            }
        }
        set path [file join $private fpga rtl ntrpu_drain_certificate_join.sv]
        set rtl [read_text $path];set expected {}
        if {$case eq "ignore_owner"} {
            set rtl [replace_once $rtl {        for(integer i=5;i<7;i=i+1) begin} {        if(proof_request[0])good[0]=1;
        for(integer i=5;i<7;i=i+1) begin}]
            set expected UNPROVEN_OWNER_ACCEPTED
        } elseif {$case eq "ignore_host"} {
            set rtl [replace_once $rtl {if(current && contract_present[i] && peer_ready[i]) proof_request[i]=1;} {if(current && (contract_present[i] || i==1) && peer_ready[i]) proof_request[i]=1;}]
            set expected ABSENT_HOST_PROOF
        } elseif {$case eq "ignore_xdma"} {
            set rtl [replace_once $rtl {if(current && contract_present[i] && peer_ready[i]) proof_request[i]=1;} {if(current && (contract_present[i] || i==2) && peer_ready[i]) proof_request[i]=1;}]
            set expected ABSENT_XDMA_PROOF
        }
        set f [open $path w];puts -nonewline $f $rtl;close $f
        set path [file join $private fpga tb tb_ntrpu_drain_owned_sections.sv]
        set tb [read_text $path]
        set monitor {    // Public composition monitors, identical in baseline and mutants.
    always @(posedge mgmt_clk)if(cold)begin
        if(accepted[0])must(ack[0] && owner_lost && owner_invalid &&
            ack_gen[63:0]==pg && ack_tag[63:0]==pt,"UNPROVEN_OWNER_ACCEPTED");
        if(!present[1])must(!(proof_req[1] || ack[1] || accepted[1]),"ABSENT_HOST_PROOF");
        if(!present[2])must(!(proof_req[2] || ack[2] || accepted[2]),"ABSENT_XDMA_PROOF");
    end
endmodule}
        set tb [replace_once $tb {endmodule} $monitor]
        set f [open $path w];puts -nonewline $f $tb;close $f
        set profile [file join $output $case]
        set code [catch {owned_profile $private $profile 5000 $mode} message options]
        set sim [file join $profile sections.sim sim_1 behav xsim simulate.log]
        if {![file exists $sim]} {error "No native simulation for $case: $message"}
        set text [read_text $sim]
        if {$expected eq ""} {
            if {$code} {error "Baseline failed $case: $message"}
            puts "OWNED_BASELINE_PASS case=$case mode=$mode"
        } else {
            if {!$code} {error "Mutant survived $case"}
            set fatal [regexp -all -inline -line {^Fatal:.*$} $text]
            if {[llength $fatal]!=1 || [lindex $fatal 0] ne "Fatal: DRAIN_OWNED_ASSERT $expected"} {error "Unexpected mutant failure $case: $fatal"}
            if {[regexp -line {^DRAIN_OWNED_PASS } $text]} {error "Mutant reported PASS"}
            puts "OWNED_MUTANT_DETECTED case=$case reason=$expected"
        }
        set f [open [file join $output $case-result.txt] {WRONLY CREAT EXCL}]
        puts $f "case=$case\nmode=$mode\nexpected=$expected\nprofile_return=$code\nmessage=$message";close $f
        if {[llength [get_projects -quiet]]} {close_project}
    }
    puts NTRPU_DRAIN_OWNED_NEGATIVES_EXPORTED
}
if {[llength $argv]!=1 || [llength [get_projects -quiet]]} {exit 1}
if {[catch {owned_negatives $root [file normalize [lindex $argv 0]]} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
