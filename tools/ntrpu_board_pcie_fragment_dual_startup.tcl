set script_dir [file dirname [file normalize [info script]]]
proc simulate {directory sources tb netlist variant} {
    file mkdir $directory;cd $directory
    set defines [expr {$netlist ? {-d NTRD_GSR_TRACE} : {}}]
    exec xvlog --sv {*}$defines {*}$sources $tb > compile.log 2>@1
    set libs [expr {$netlist ? {-L unisims_ver -L secureip} : {}}]
    set tops [expr {$netlist ? {tb_ntrpu_pcie_fragment_dual_managed glbl} : {tb_ntrpu_pcie_fragment_dual_managed}}]
    exec xelab {*}$tops --debug typical --mt 4 {*}$libs -s ownership > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    set seeds [expr {$variant eq "normal" ? {1 7 42} : {1}}]
    foreach seed $seeds {
        set code [catch {exec xsim ownership -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
        set f [open seed-${seed}.console.log r];set text [read $f];close $f
        if {$variant eq "normal"} {
            if {$code || ![regexp -line {^OWN_PASS } $text] || [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Ownership positive failed seed=$seed: $message"}
        } else {
            set expected [dict get {profile CONTROL_READ_RESPONSE caps CONTROL_READ_RESPONSE range CONTROL_WRITE_RESPONSE capture CONTROL_READ_RESPONSE ack CONTROL_WRITE_RESPONSE high CONTROL_WRITE_RESPONSE lock CONTROL_WRITE_RESPONSE snapshot CONTROL_READ_RESPONSE unavailable CONTROL_WRITE_RESPONSE} $variant]
            if {![regexp -line [format {^Fatal: .*%s.*$} $expected] $text] || [regexp -line {^OWN_PASS } $text]} {error "Ownership mutant not detected: $variant"}
        }
        puts "NTRPU_OWNERSHIP_REVIEW netlist=$netlist variant=$variant seed=$seed PASS"
    }
}
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected NEW_OUTPUT VERIFIED_SYNTH_DIRECTORY"}
    set output [file normalize [lindex $arguments 0]]
    set seed [file normalize [lindex $arguments 1]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set root [file normalize [file join $script_dir ..]]
    set rtl {};foreach n {ntrpu_pcie_fragment_decode ntrpu_pcie_fragment_ingress ntrpu_fragment_pool_bridge ntrpu_pcie_fragment_pool ntrpu_pcie_fragment_dual_managed ntrpu_pcie_fragment_dual_control} {lappend rtl [file join $root fpga rtl $n.sv]}
    set tb [file join $root fpga tb tb_ntrpu_pcie_fragment_dual_startup.sv]
    set imports [file join $output imported];file mkdir $imports
    foreach name {ownership_funcsim.v ownership_synth.dcp utilization.rpt timing.rpt check_timing.rpt drc.rpt applied.xdc} {
        file copy [file join $seed $name] [file join $imports $name]
    }
    # Explicit, unchanged installed global primitive model. Observe, never force GSR.
    set glbl [file join $output glbl.v]
    file copy [file join $::env(XILINX_VIVADO) data verilog src glbl.v] $glbl
    simulate [file join $output normal] $rtl $tb 0 normal
    set f [open $tb r];set net_tb [string map [list { #(.FIRST_COOKIE(64'hffffffff))} {}] [read $f]];close $f
    set n [file join $output netlist_tb.sv];set f [open $n {WRONLY CREAT EXCL}];puts -nonewline $f $net_tb;close $f
    simulate [file join $output netlist] [list [file join $imports ownership_funcsim.v] $glbl] $n 1 normal
    puts NTRPU_FRAGMENT_DUAL_STARTUP_REPLAY_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
