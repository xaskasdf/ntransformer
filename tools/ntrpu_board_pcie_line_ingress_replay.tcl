set script_dir [file dirname [file normalize [info script]]]
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected PRIOR_SYNTH_OUTPUT NEW_DIRECTORY"}
    lassign $arguments prior output
    set prior [file normalize $prior];set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    foreach name {ingress_funcsim.v ingress_synth.dcp} {
        file copy [file join $prior $name] [file join $output $name]
    }
    set tb [file normalize [file join $script_dir .. fpga tb tb_ntrpu_pcie_line_ingress.sv]]
    file mkdir [file join $output netlist];cd [file join $output netlist]
    exec xvlog --sv [file join $output ingress_funcsim.v] $tb > compile.log 2>@1
    exec xelab tb_ntrpu_pcie_line_ingress glbl --debug typical --mt 4 -L unisims_ver -L secureip -s ingress > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    foreach seed {1 7 42} {
        set code [catch {exec xsim ingress -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
        set f [open seed-${seed}.console.log r];set text [read $f];close $f
        if {$code || ![regexp -line {^NTRPU_INGRESS_PASS jobs=19 results=19 lines=7 } $text] ||
            [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Ingress netlist positive failed seed=$seed: $message"}
        puts "NTRPU_INGRESS_REVIEW variant=normal netlist=1 seed=$seed PASS"
    }
    puts NTRPU_INGRESS_REPLAY_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
