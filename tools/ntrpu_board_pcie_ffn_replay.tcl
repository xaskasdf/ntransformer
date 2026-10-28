# Read-only diagnosis of a completed compiled snapshot, copied into a new run.
proc diagnose {arguments} {
    if {[llength $arguments]!=2} {error "Expected COMPILED_XSIM NEW_DIRECTORY"}
    lassign $arguments input output
    set input [file normalize $input];set output [file normalize $output]
    if {[file exists $output]} {error "Output exists"}
    if {![file exists [file join $input xsim.dir board_behav xsimk]]} {error "Missing compiled snapshot"}
    file mkdir $output
    file copy [file join $input xsim.dir] [file join $output xsim.dir]
    foreach p [glob -nocomplain -directory $input *] {
        if {[file isfile $p] && [file extension $p] in {.dat .mem .hex .sveo}} {file copy $p $output}
    }
    set f [open [file join $output origin.txt] {WRONLY CREAT EXCL}]
    puts $f $input;puts $f "Compiled snapshot replay, no source or signal changes.";close $f
    set f [open [file join $output replay.tcl] {WRONLY CREAT EXCL}]
    puts $f {set previous [current_time -s]
for {set i 0} {$i<1000} {incr i} {
    run 1 us
    set now [current_time -s]
    puts "NTRPU_PCIE_FFN_REPLAY time=$now"
    flush stdout
    if {$now eq $previous} {break}
    set previous $now
}
foreach name {m_axi_awaddr m_axi_awlen m_axi_awsize m_axi_awburst m_axi_awlock m_axi_awid m_axi_awvalid m_axi_awready writing b_pending m_axi_bvalid write_left s_host_axi_awaddr s_host_axi_awlen s_host_axi_awsize s_host_axi_awburst s_host_axi_awlock} {
    puts "NTRPU_PCIE_FFN_DIAG signal=$name value=[get_value -radix hex /board/EP/ntrpu_app/$name]"
}
quit}
    close $f
    cd $output
    exec xsim board_behav -tclbatch replay.tcl -log simulation.log -testplusarg TESTNAME=ntrpu_ffn_msix > replay.console.log 2>@1
    puts NTRPU_PCIE_FFN_DIAG_EXPORTED
}
if {[catch {diagnose $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} {puts stderr [dict get $options -errorinfo]}
    exit 1
}
exit 0
