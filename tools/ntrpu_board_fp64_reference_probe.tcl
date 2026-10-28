# Isolate simulator real rounding before accepting an adversarial MAC oracle.
proc fp64_probe {arguments} {
    if {[llength $arguments] != 1} { error "Expected NEW_DIRECTORY" }
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir $output
    set stream [open [file join $output probe.sv] {WRONLY CREAT EXCL}]
    puts $stream {
module probe;
    real sum, term, stored, exact;
    logic [63:0] bits;
    initial begin
        exact=$bitstoreal(64'h4340000000000000);
        $display("FP64_LITERAL decimal_bits=%016h exact_bits=%016h",$realtobits(9007199254740992.0),$realtobits(exact));
        sum=0;
        for (int block=0;block<64;block++) begin
            term=block==0 ? exact : block==31 ? 1.0 : block==32 ? -exact : 0.0;
            sum+=term;
            bits=$realtobits(sum);
            stored=$bitstoreal(bits);
            if (block inside {0,31,32,63})
                $display("FP64_PROBE block=%0d term=%g sum=%g bits=%016h stored=%g",block,term,sum,bits,stored);
        end
        sum=0;
        for (int block=0;block<64;block++) begin
            term=block==0 ? exact : block==31 ? 1.0 : block==32 ? -exact : 0.0;
            bits=$realtobits(sum+term);
            sum=$bitstoreal(bits);
        end
        $display("FP64_STORED sum=%g bits=%016h",sum,$realtobits(sum));
        if ($realtobits(sum)!=0) $fatal(1,"Stored FP64 oracle disagrees with IEEE round-to-nearest-even");
        $display("NTRPU_FP64_REFERENCE_PROBE_PASS");
        $finish;
    end
endmodule
    }
    close $stream
    create_project fp64_probe [file join $output project] -part xcvu5p-flva2104-2-i
    add_files -fileset sim_1 [file join $output probe.sv]
    set_property top probe [get_filesets sim_1]
    set_property xsim.simulate.runtime 0ns [get_filesets sim_1]
    launch_simulation
    run all
    close_sim
    file copy [file join $output project fp64_probe.sim sim_1 behav xsim simulate.log] [file join $output probe.log]
    set stream [open [file join $output probe.log] r]; set text [read $stream]; close $stream
    if {![string match "*NTRPU_FP64_REFERENCE_PROBE_PASS*" $text] ||
        [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|FAIL )} $text]} { error "FP64 probe failed" }
    close_project
}
if {[catch {fp64_probe $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
