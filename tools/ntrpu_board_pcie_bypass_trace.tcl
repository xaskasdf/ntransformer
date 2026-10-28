# Passive native-interface witness on an immutable compiled simulation copy.
# Usage: -tclargs TERMINAL_XSIM_DIRECTORY NEW_DIRECTORY
proc ntrpu_bypass_trace {arguments} {
    if {[llength $arguments]!=2} {error "Expected TERMINAL_XSIM_DIRECTORY NEW_DIRECTORY"}
    lassign $arguments input output
    set input [file normalize $input];set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    if {![file exists [file join $input xsim.dir board_behav xsimk]]} {error "Missing compiled snapshot"}
    file mkdir $output
    file copy [file join $input xsim.dir] [file join $output xsim.dir]
    foreach p [glob -nocomplain -directory $input *] {
        if {[file isfile $p] && [file extension $p] in {.dat .mem .hex .sveo}} {file copy $p $output}
    }
    exec python3 -c {
import hashlib,json,pathlib,sys
src,dst=map(pathlib.Path,sys.argv[1:]); rows={}
for p in sorted(dst.rglob('*')):
    if not p.is_file():continue
    name=p.relative_to(dst).as_posix()
    sha=hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha==hashlib.sha256((src/name).read_bytes()).hexdigest(),name
    rows[name]=sha
assert 'xsim.dir/board_behav/xsimk' in rows
(dst/'copied_snapshot.json').write_text(json.dumps(rows,indent=2)+'\n')
} $input $output
    set f [open [file join $output origin.txt] {WRONLY CREAT EXCL}]
    puts $f $input;puts $f "Exact compiled-snapshot copy; native VCD observation only; no HDL edits or signal forcing.";close $f
    set f [open [file join $output replay.tcl] {WRONLY CREAT EXCL}]
    puts $f {
set names {/board/RP/user_clk /board/EP/user_clk /board/EP/user_resetn}
foreach prefix {s_axis_rq m_axis_rc} {
    foreach suffix {tdata tkeep tuser tvalid tready tlast} {lappend names /board/RP/${prefix}_${suffix}}
}
foreach name {pcie_rq_tag0 pcie_rq_tag_vld0 pcie_rq_tag1 pcie_rq_tag_vld1} {
    lappend names /board/RP/pcie_4_0_rport/$name
}
foreach suffix {awid awaddr awlen awsize awburst awvalid awready wdata wstrb wlast wvalid wready bid bresp bvalid bready arid araddr arlen arsize arburst arvalid arready rid rdata rresp rlast rvalid rready} {
    lappend names /board/EP/m_axib_$suffix
}
set objects {}
set f [open trace_objects.txt {WRONLY CREAT EXCL}]
foreach name $names {
    set found [get_objects $name]
    if {[llength $found]!=1} {error "Expected one native signal: $name"}
    puts $f $name
    lappend objects {*}$found
}
close $f
open_vcd native.vcd
log_vcd $objects
set previous [current_time -s]
for {set i 0} {$i<205} {incr i} {
    run 1 us
    set now [current_time -s]
    puts "NTRPU_BYPASS_TRACE_PROGRESS time=$now"
    flush stdout
    if {$now eq $previous} {break}
    set previous $now
}
close_vcd
puts NTRPU_BYPASS_NATIVE_TRACE_EXPORTED
quit
}
    close $f
    cd $output
    exec xsim board_behav -tclbatch replay.tcl -log simulate.log -testplusarg TESTNAME=ntrpu_bypass > replay.console.log 2>@1
    puts NTRPU_BYPASS_NATIVE_REPLAY_RETURNED
    puts "Witness exported; exact copied binary/source hashes and native handshakes still require review."
}
if {[catch {ntrpu_bypass_trace $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
