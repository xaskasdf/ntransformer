# Run only inside the prepared private XSIM snapshot.
set out [file normalize $::env(NTRPU_NATIVE_OUT)]
if {[file exists $out]} {error "Expected new clock-loss report directory"}
file mkdir $out
set names {/board/sys_rst_n /board/AXI_PCIE_EP/axi_aclk_out /board/AXI_PCIE_EP/axi_aresetn /board/AXI_PCIE_EP/axi_ctl_aresetn /board/AXI_PCIE_EP/user_lnk_up /board/probe/guard/axi_clk /board/probe/dut/clk}
foreach name {clk mgmt_clk phase configuration_rst_n cold_rst_n link_up stop guard_fault guard_code guard_armed guard_alive owned fault mapper_fault cmd_valid cmd_ready cmd_tag cmd_generation cmd_source cmd_bytes cfg_ready src_req_valid src_req_ready src_req_tag src_req_address src_req_bytes src_valid src_cpl_valid ack_valid result_valid pause_active pause_complete clock_case_done stop_level} {
    lappend names /board/probe/$name
}
set objects {}
set f [open $out/trace_objects.txt {WRONLY CREAT EXCL}]
foreach name $names {
    set found [get_objects $name]
    if {[llength $found]!=1} {error "Missing clock-loss signal: $name"}
    puts $f $name
    lappend objects {*}$found
}
close $f
open_vcd $out/clock_loss.vcd
log_vcd $objects
set previous [current_time -s]
for {set i 0} {$i<251} {incr i} {
    run 1 us
    set now [current_time -s]
    if {$i%10==0} {puts "CL PROGRESS time=$now";flush stdout}
    if {$now eq $previous} {break}
    set previous $now
}
close_vcd
puts CLOCK_LOSS_SIMULATOR_RETURNED_REVIEW_REQUIRED
quit
