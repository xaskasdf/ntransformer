"""Derive an isolated cold native endpoint/guard/two-MIG harness.

No protected AMD implementation is edited. The prior native packet stimulus,
memory fixture and reference runs remain intact. Runtime reset is a later gate.
"""
from pathlib import Path


def once(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new,1)


def bench(text):
    text=once(text,'module ntrpu_pcie_managed_native_harness;',
              'module ntrpu_pcie_guarded_native_harness;')
    text=once(text,'    reg host_rst_n=0,transport_rst_n=0;', '''    wire link_up_async=board.EP.user_lnk_up;
    wire endpoint_reset_async=!board.EP.user_resetn || !board.sys_rst_n;
    reg guard_arm=0,guard_peers_held=0,guard_peers_ready=0;
    wire host_reset_n,transport_reset_n,host_reset_req;
    wire host_admit,source_admit,host_alive,guard_source_alive;
    wire guard_invalidate,loss_mgmt,waiting_ack,guard_running;
    wire [4:0] loss_causes;
    wire [63:0] first_loss_tick;
    wire [3:0] guard_state;
    wire owner_invalid_ack,boundary_invalid;
    wire [7:0] boundary_fenced;
    wire [15:0] boundary_pending,boundary_fault;
    wire [511:0] boundary_accepted,boundary_delivered,boundary_abandoned;
    wire host_rst_n=host_reset_n,transport_rst_n=transport_reset_n;
    // Cold-only peer observation: actual native requests must be inactive.
    // This does not make the native endpoint obey the guard's runtime request.
    wire native_requests_idle=(s_axi_awvalid===1'b0 && s_axi_wvalid===1'b0 &&
        s_axil_awvalid===1'b0 && s_axil_wvalid===1'b0 && s_axil_arvalid===1'b0);
    always @(posedge mgmt_clk) begin
        guard_peers_held<=!host_reset_n && !transport_reset_n && native_requests_idle;
        guard_peers_ready<=host_reset_n && transport_reset_n &&
            board.EP.user_resetn && board.EP.user_lnk_up;
        if(!passed) $display("NTRD_GUARD_NATIVE ps=%0d cold=%b link=%b epreset=%b perst=%b loss=%b invalid=%b ack=%b req=%b hrst=%b srst=%b ha=%b sa=%b state=%h held=%b ready=%b idle=%b",
            ps(),lifetime_reset_n,board.EP.user_lnk_up,board.EP.user_resetn,board.sys_rst_n,
            loss_mgmt,guard_invalidate,owner_invalid_ack,host_reset_req,host_reset_n,transport_reset_n,
            host_admit,source_admit,guard_state,guard_peers_held,guard_peers_ready,native_requests_idle);
    end''')
    text=once(text,'''        wait(lifetime_reset_n);repeat(5) @(negedge source_clk);transport_rst_n=1;
        wait(board.EP.user_resetn===1'b1);
        repeat(5) @(negedge host_clk);host_rst_n=1;''',
        '        wait(lifetime_reset_n);@(negedge mgmt_clk);guard_arm=1;')
    text=once(text,'ntrpu_pcie_fragment_dual_host_cold #','ntrpu_pcie_fragment_dual_host_guarded #')
    text=text.replace('dut.d_','dut.host.k_')
    text=once(text,'        wait(host_rst_n);idle(8);',
              '        wait(guard_running && host_admit && source_admit);idle(8);')
    text=once(text,'        passed=1;', '''        need(guard_running && host_admit && source_admit && !loss_mgmt && !boundary_invalid &&
             boundary_pending==0 && boundary_fault==0 && boundary_abandoned==0,"GUARDED_NATIVE_FINAL_BOUNDARY");
        $display("NTRD_GUARDED_NATIVE_PASS ps=%0d accepted=%h delivered=%h abandoned=%h",ps(),boundary_accepted,boundary_delivered,boundary_abandoned);
        passed=1;''')
    text=once(text,'    always @(posedge host_clk) begin\n        host_cycle++;', '''    always @(posedge host_clk) begin
        if(!host_rst_n && lifetime_reset_n)
            need(native_requests_idle,"NATIVE_REQUEST_DURING_COLD_TRANSPORT_RESET");
        host_cycle++;''')
    return text


def tcl(text):
    text=text.replace('patch_ntrpu_pcie_managed_native.py','patch_ntrpu_guarded_native.py')
    text=text.replace('ntrpu_pcie_managed_native.sv','ntrpu_pcie_guarded_native.sv')
    text=once(text,'ntrpu_pcie_fragment_dual_host_cold} {',
        'ntrpu_retained_channel ntrpu_host_boundary_cut ntrpu_host_loss_guard ntrpu_pcie_fragment_guarded_host ntrpu_pcie_fragment_dual_host_guarded} {')
    text=once(text,'set objects {}', '''lappend names /board/sys_rst_n /board/EP/user_lnk_up
foreach suffix {mgmt_clk guard_arm guard_peers_held guard_peers_ready native_requests_idle host_reset_req host_admit source_admit guard_state loss_mgmt guard_invalidate owner_invalid_ack} {
    lappend names /board/ntrpu_bypass/$suffix
}
set objects {}''')
    text=text.replace('NTRD_NATIVE_DUAL_MIG_SIMULATOR_RETURNED','NTRD_GUARDED_NATIVE_SIMULATOR_RETURNED')
    return text


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    for src,dst,transform in [
        ('fpga/tb/ntrpu_pcie_managed_native.sv','fpga/tb/ntrpu_pcie_guarded_native.sv',bench),
        ('tools/ntrpu_board_pcie_managed_native.tcl','tools/ntrpu_board_guarded_native.tcl',tcl)]:
        with (root/dst).open('x',newline='\n') as f:f.write(transform((root/src).read_text()))
