set script_dir [file dirname [file normalize [info script]]]
proc brandon_host {arguments} {
    global script_dir
    if {[llength $arguments]!=2} {error "Expected FIXTURE NEW_OUTPUT"}
    lassign $arguments fixture output
    set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    file mkdir $output;cd $output
    exec python3 [file join $script_dir prepare_ntrpu_pcie_ffn_brandon.py] $fixture bench > prepare.log 2>@1
    exec xvlog --sv -i bench [file join $output bench ntrpu_pcie_ffn_host.sv] [file join $output bench ntrpu_pcie_ffn_app.sv] [file join $script_dir .. fpga tb ntrpu_pcie_brandon_host_tb.sv] > compile.log 2>@1
    exec xelab board --debug typical --mt 2 -s brandon_host > elaborate.log 2>@1
    set f [open run.tcl {WRONLY CREAT EXCL}];puts $f {run all;quit};close $f
    set expected {PASS HOST_READ_UNMAPPED HOST_READ_UNMAPPED HOST_READ_BOUNDS HOST_READ_UNMAPPED HOST_DOUBLE_INIT HOST_NEGATIVE_INDEX HOST_READ_BOUNDS HOST_READ_UNMAPPED}
    for {set mode 0} {$mode<9} {incr mode} {
        set code [catch {exec xsim brandon_host -tclbatch run.tcl -testplusarg MODE=$mode -log mode${mode}.log > mode${mode}.console.log 2>@1} message]
        set f [open mode${mode}.console.log r];set text [read $f];close $f
        if {$mode==0} {
            if {$code || ![string match {*NTRPU_BRANDON_HOST_PASS bytes=592544 requests=* output_reference_bytes=1024*} $text] ||
                [regexp -nocase {(^|\n)[ \t]*(Fatal:|ERROR:|WARNING:)} $text]} {error "Brandon host fixture rejected"}
        } elseif {![string match "*Fatal: [lindex $expected $mode]*" $text] || [string match {*NTRPU_BRANDON_HOST_PASS*} $text]} {error "Expected diagnostic absent: $mode"}
        puts "NTRPU_BRANDON_HOST_REVIEW mode=$mode expected=[lindex $expected $mode] PASS"
    }
    puts NTRPU_BRANDON_HOST_SUITE_PASS
}
if {[catch {brandon_host $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
