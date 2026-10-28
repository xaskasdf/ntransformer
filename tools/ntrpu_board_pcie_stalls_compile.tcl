# Compile only the public example wrapper after a declaration-order correction.
# This is a syntax/diagnostic check, not another functional DMA result.
set ntrpu_script_dir [file dirname [file normalize [info script]]]
proc ntrpu_stalls_compile {arguments} {
    global ntrpu_script_dir
    if {[llength $arguments] != 2} { error "Expected CLEAN_EP_SV NEW_DIRECTORY" }
    lassign $arguments input output
    set input [file normalize $input]
    set output [file normalize $output]
    if {[file exists $output]} { error "Output already exists" }
    file mkdir [file join $output example imports]
    file copy $input [file join $output example imports xilinx_dma_pcie_ep.sv]
    exec python3 [file join $ntrpu_script_dir patch_ntrpu_pcie_stalls.py] [file join $output example] > [file join $output patch.log] 2>@1
    cd $output
    exec xvlog --sv [file join $ntrpu_script_dir .. fpga tb ntrpu_sim_ready_gate.sv] \
        [file join $output example imports xilinx_dma_pcie_ep.sv] > compile.log 2>@1
    set stream [open compile.log r]; set text [read $stream]; close $stream
    if {[regexp -nocase {\m(error|warning)\M} $text]} { error "Wrapper compilation has diagnostics; review compile.log" }
    puts NTRPU_STALL_WRAPPER_COMPILE_PASS
    puts "Scope: wrapper syntax only; no elaboration or simulation in this batch."
}
if {[catch {ntrpu_stalls_compile $argv} message options]} {
    puts stderr $message
    if {[dict exists $options -errorinfo]} { puts stderr [dict get $options -errorinfo] }
    exit 1
}
exit 0
