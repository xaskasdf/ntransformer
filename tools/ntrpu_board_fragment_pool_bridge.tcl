set script_dir [file dirname [file normalize [info script]]]
proc simulate_bridge {directory source tb boundary netlist variant} {
    file mkdir $directory;cd $directory
    set defines {}
    if {$boundary} {lappend defines -d NTRPU_BOUNDARY_TEST}
    if {$netlist} {lappend defines -d NTRPU_NETLIST}
    exec xvlog --sv {*}$defines $source $tb > compile.log 2>@1
    set libraries [expr {$netlist ? {-L unisims_ver -L secureip} : {}}]
    set tops [expr {$netlist ? {tb_ntrpu_fragment_pool_bridge glbl} : {tb_ntrpu_fragment_pool_bridge}}]
    exec xelab {*}$tops --debug typical --mt 4 {*}$libraries -s bridge > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all; quit};close $f
    set seeds [expr {$variant eq "normal" ? {1 7 42} : {1}}]
    foreach seed $seeds {
        set code [catch {exec xsim bridge -tclbatch run.tcl -testplusarg SEED=$seed -log seed-${seed}.log > seed-${seed}.console.log 2>@1} message]
        set f [open seed-${seed}.console.log r];set text [read $f];close $f
        if {$variant eq "normal"} {
            if {$code || ![regexp -line {^POOL_BRIDGE_PASS } $text] || [regexp -line {^(Fatal:|ERROR:|CRITICAL WARNING:)} $text]} {error "Bridge positive failed boundary=$boundary seed=$seed: $message"}
        } else {
            set expected [dict get {cookie COMMAND_CONTEXT identity BAD_LINE_ACCEPTED drain EARLY_DONE_BEFORE_DRAIN retire FORGED_RETIRE_TAG session SESSION_REOPENED stop DONE_CONTEXT wrap COOKIE_WRAP} $variant]
            if {![regexp -line [format {^Fatal: .*%s.*$} $expected] $text] || [regexp -line {^POOL_BRIDGE_PASS } $text]} {error "Bridge mutant not detected: $variant"}
        }
        puts "NTRPU_POOL_BRIDGE_REVIEW boundary=$boundary netlist=$netlist variant=$variant seed=$seed PASS"
    }
}
proc main {arguments} {
    global script_dir
    if {[llength $arguments]!=1} {error "Expected NEW_OUTPUT"}
    set output [file normalize [lindex $arguments 0]]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output
    set rtl [file normalize [file join $script_dir .. fpga rtl ntrpu_fragment_pool_bridge.sv]]
    set tb [file normalize [file join $script_dir .. fpga tb tb_ntrpu_fragment_pool_bridge.sv]]
    set f [open $rtl r];set original [read $f];close $f
    foreach variant {normal cookie identity drain retire session stop wrap} {
        set source $rtl;set boundary [expr {$variant eq "wrap"}]
        if {$variant ne "normal"} {
            switch $variant {
                cookie {set before {active_cookie<=next_cookie};set after {active_cookie<=saved_tag}}
                identity {set before {line_epoch!=epoch};set after {line_epoch[31:0]!=epoch[31:0]}}
                drain {set before {!session_invalid && pool_drained && !(|cpl_valid) && completion_seen};set after {!session_invalid && !(|cpl_valid) && completion_seen}}
                retire {set before {retire_tag==saved_tag};set after {retire_tag[31:0]==saved_tag[31:0]}}
                session {set before {if(loss_now) session_lost<=1};set after {if(1'b0) session_lost<=1}}
                stop {set before {stopped || stop ? STOPPED};set after {1'b0 ? STOPPED}}
                wrap {set before {if(next_cookie==64'hffffffffffffffff) exhausted<=1};set after {if(next_cookie==64'hffffffffffffffff) exhausted<=0}}
            }
            if {[llength [split [string map [list $before "\u0001"] $original] "\u0001"]]!=2} {error "Mutation match count: $variant"}
            set source [file join $output ${variant}.sv]
            set f [open $source {WRONLY CREAT EXCL}];puts -nonewline $f [string map [list $before $after] $original];close $f
        }
        simulate_bridge [file join $output $variant] $source $tb $boundary 0 $variant
    }
    simulate_bridge [file join $output boundary] $rtl $tb 1 0 normal
    foreach boundary {0 1} {
        set synth [file join $output synth-$boundary];file mkdir $synth;cd $synth
        create_project bridge [file join $synth project] -part xcvu5p-flva2104-2-i
        add_files $rtl
        set_property top ntrpu_fragment_pool_bridge [current_fileset]
        set extra {}
        if {$boundary} {lappend extra -generic {FIRST_COOKIE=64'hfffffffffffffffe}}
        synth_design -top ntrpu_fragment_pool_bridge -mode out_of_context -part xcvu5p-flva2104-2-i {*}$extra
        if {[llength [get_cells -quiet -hierarchical -filter {REF_NAME == LDCE || REF_NAME == LDPE || IS_BLACKBOX == 1}]]} {error "Latch or unresolved cell"}
        create_clock -name clk -period 5.000 [get_ports clk]
        report_utilization -hierarchical -file utilization.rpt
        report_timing_summary -report_unconstrained -file timing.rpt
        check_timing -verbose -file check_timing.rpt
        report_drc -file drc.rpt
        write_checkpoint [file join $synth bridge_synth.dcp]
        write_xdc -constraints all [file join $synth applied.xdc]
        write_verilog -mode funcsim [file join $synth bridge_funcsim.v]
        close_project
        simulate_bridge [file join $output netlist-$boundary] [file join $synth bridge_funcsim.v] $tb $boundary 1 normal
    }
    puts NTRPU_POOL_BRIDGE_SUITE_PASS
}
if {[catch {main $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
