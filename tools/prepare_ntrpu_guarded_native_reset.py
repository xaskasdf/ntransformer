"""Prepare a separate native PERST retention experiment, pending the cold gate.

Runtime peer acknowledgements are deliberately withheld: the current wrapper
has no endpoint reset coordinator. Passing this bank will qualify retention
during that blocked state, not runtime reuse or physical PERST timing.
"""
from pathlib import Path
from prepare_ntrpu_guarded_native import once


def bench(text,loss_reference):
    text=once(text,'module ntrpu_pcie_guarded_native_harness;','module ntrpu_pcie_guarded_native_reset_harness;')
    text=once(text,'    reg guard_arm=0,guard_peers_held=0,guard_peers_ready=0;',
        '''    reg guard_arm=0,guard_peers_held=0,guard_peers_ready=0;
    reg healthy_done=0,loss_test_started=0;
    reg [63:0] loss_cookie,loss_epoch;
    integer loss_window=0;
    // Runtime coordination is absent. Never fabricate its held/ready ACK.
    // A warm link may return while the guard keeps local transport reset.''')
    text=once(text,'guard_peers_held<=!host_reset_n && !transport_reset_n && native_requests_idle;',
        'guard_peers_held<=!loss_test_started && !host_reset_n && !transport_reset_n && native_requests_idle;')
    text=once(text,'guard_peers_ready<=host_reset_n && transport_reset_n &&',
        'guard_peers_ready<=!loss_test_started && host_reset_n && transport_reset_n &&')
    assert text.count('[0:17]')==6
    text=text.replace('[0:17]','[0:18]')
    text=once(text,'for(integer j=0;j<18;j++) begin commanded[j]', 'for(integer j=0;j<19;j++) begin commanded[j]')
    text=once(text,'        passed=1;','        healthy_done=1;')
    text=once(text,'        $display("NTRD_HOST_PASS cycles=',
        '''        // Place the healthy boundary between native/management rising
        // edges so its prefix is independently replayable without edge races.
        while(ps()%20000!=10000) @(negedge clk);
        $display("NTRD_HOST_PASS cycles=''')
    text=once(text,'need(board.EP.user_resetn===1\'b1,"RUNTIME_PCIE_RESET_OUTSIDE_COLD_CONTRACT");',
        'if(!loss_test_started) need(board.EP.user_resetn===1\'b1,"UNPLANNED_PCIE_RESET");')
    text=once(text,'if(!host_rst_n && lifetime_reset_n)', 'if(!host_rst_n && lifetime_reset_n && !loss_test_started)')
    loss_trace=loss_reference[loss_reference.index('    always @(posedge clk) if(loss_test_started && !passed) begin'):]
    loss_trace=loss_trace[:loss_trace.index('    always @(posedge host_clk) begin')]
    loss_trace=loss_trace.replace('GUARDED_MIG_LOSS','NTRD_NATIVE_RESET_LOSS')
    task='''    task automatic run_native_reset;
        need(healthy_done && !passed && guard_running && native_requests_idle,"NATIVE_RESET_PREREQUISITE");
        $display("NTRD_NATIVE_RESET_HEALTHY_END ps=%0d",ps());
        @(negedge ui_clk[0]);hold_b[0]=1;
        setup(18);fill();wait(aw_seen[18] && w_seen[18]);
        drain_control();repeat(40) @(negedge clk);
        need(!b_seen[18] && !result_valid && boundary_pending==0 && control_quiet() &&
             native_requests_idle && host_data_b==37,"NATIVE_RESET_READY");
        loss_cookie=next_cookie;loss_epoch=epoch;loss_test_started=1;
        $display("NTRD_NATIVE_RESET_EVENT name=ready ps=%0d cookie=%h epoch=%h",ps(),loss_cookie,loss_epoch);
        @(negedge board.rp_sys_clk_p);
        board.ntrpu_set_perst(1'b0);
        $display("NTRD_NATIVE_RESET_EVENT name=perst_assert ps=%0d",ps());
        fork
            begin
                // Same 500 reference-clock edges as the public cold reset.
                // This is a simulation stimulus, not a board pulse-width spec.
                repeat(500) @(posedge board.rp_sys_clk_p);
            end
            begin
                wait(board.EP.user_resetn===1'b0 && board.EP.user_lnk_up===1'b0);
                $display("NTRD_NATIVE_RESET_EVENT name=endpoint_down ps=%0d",ps());
                wait(guard_state==6);repeat(40) @(negedge clk);
                need(loss_mgmt && owner_invalid_ack && boundary_invalid && session_lost && session_invalid,"NATIVE_RESET_OWNER_ACK");
                need(host_reset_req && !host_reset_n && !transport_reset_n &&
                     !host_admit && !source_admit && !guard_running && !guard_peers_held && !guard_peers_ready,"NATIVE_RESET_WAITING_COORDINATOR");
                need(bridge_lease_held && lease_held && bridge_epoch==loss_epoch && epoch==loss_epoch && next_cookie==loss_cookie,"NATIVE_RESET_OWNER_PRESERVED");
                need(!ack_ready && !arm_ready && !result_ready && cpl_ready==0 && !done_valid && !result_valid,"NATIVE_RESET_PUBLICATION_BLOCKED");
                $display("NTRD_NATIVE_RESET_EVENT name=release_memory_b ps=%0d",ps());
                @(negedge ui_clk[0]);hold_b[0]=0;
                wait(cpl_valid[0]);repeat(80) @(negedge clk);
                need(cpl_valid==1 && cpl_ready==0 && cpl_status[1:0]==0 && cpl_tag[63:0]==loss_cookie-1,"NATIVE_RESET_CPL_QUARANTINED");
                mem[0].readback(18);
                $display("NTRD_NATIVE_RESET_EVENT name=memory_verified ps=%0d",ps());
            end
        join
        @(negedge board.rp_sys_clk_p);
        board.ntrpu_set_perst(1'b1);
        $display("NTRD_NATIVE_RESET_EVENT name=perst_release ps=%0d",ps());
        repeat(10000) begin
            @(negedge mgmt_clk);
            if(board.EP.user_resetn===1'b1 && board.EP.user_lnk_up===1'b1) break;
        end
        need(board.EP.user_resetn===1'b1 && board.EP.user_lnk_up===1'b1,"NATIVE_LINK_RETURN_TIMEOUT");
        $display("NTRD_NATIVE_RESET_EVENT name=link_return ps=%0d",ps());
        // Exercise ARM again without authorizing the missing peer coordinator.
        guard_arm=0;repeat(8) @(negedge mgmt_clk);guard_arm=1;
        repeat(100) @(negedge clk);
        need(guard_state==6 && host_reset_req && !host_reset_n && !transport_reset_n &&
             !host_admit && !source_admit && !guard_running && !guard_peers_held && !guard_peers_ready,"NATIVE_LINK_RETURN_STILL_FENCED");
        need(boundary_pending==0 && boundary_fault==0 && native_requests_idle,"NATIVE_RESET_BOUNDARY_DRAINED");
        need(bridge_lease_held && !bridge_idle && !pool_drained && banks_used==0 && pending_counts==0 &&
             dut.receiver.receiver.receiver.core.occupied[0]==1 && dut.receiver.receiver.receiver.core.occupied[1]==0 &&
             !pool_fault && !restart_required_source,"NATIVE_RESET_DESCRIPTOR_RETAINED");
        need(arms==19 && lines==17 && commands==17 && writes==17 && returns==16 && dones==16 && acks==18 &&
             aw_count==17 && w_count==17 && b_count==17 && reads==29 && bcount==37 && rejections==1 &&
             mmio_writes==304 && mmio_reads==9 && host_data_b==37 && host_control_b==304 && host_control_r==9,"NATIVE_RESET_CONSERVATION");
        fork mem[0].model.verify_periodic_complete();mem[1].model.verify_periodic_complete();join
        need(loss_window>100 && next_cookie==loss_cookie && epoch==loss_epoch && !result_valid && !done_valid && !ack_ready,"NATIVE_RESET_END");
        @(negedge clk);
        $display("NTRD_NATIVE_RESET_RETENTION_PASS ps=%0d arms=%0d commands=%0d data=%0d returns=%0d done=%0d ack=%0d aw=%0d w=%0d b=%0d reads=%0d frames=%0d cookie=%h epoch=%h banks=%h pending=%h occupied0=%0d occupied1=%0d loss_samples=%0d coordination=blocked",
            ps(),arms,commands,writes,returns,dones,acks,aw_count,w_count,b_count,reads,bcount,next_cookie,epoch,banks_used,pending_counts,
            dut.receiver.receiver.receiver.core.occupied[0],dut.receiver.receiver.receiver.core.occupied[1],loss_window);
        passed=1;
    endtask
'''
    return once(text,'\nendmodule','\n'+task+loss_trace+'\nendmodule')


def tasks(text):
    marker='        $display("NTRD_NATIVE_PASS writes=%0d reads=%0d frames=%0d ps=%0.0f",ntrpu_control_writes,ntrpu_control_reads,ntrpu_data_writes,$realtime/1ps);'
    return once(text,marker,marker+'''
        board.ntrpu_bypass.run_native_reset();
        if(ntrpu_control_writes!=304 || ntrpu_control_reads!=8 || ntrpu_data_writes!=37)
            $fatal(1,"NTRD_NATIVE_RESET_STIMULUS_COUNT");
        $display("NTRD_NATIVE_RESET_ROOT_PASS writes=%0d reads=%0d frames=%0d ps=%0.0f",ntrpu_control_writes,ntrpu_control_reads,ntrpu_data_writes,$realtime/1ps);''')


def tcl(text):
    text=text.replace('patch_ntrpu_guarded_native.py','patch_ntrpu_guarded_native_reset.py')
    text=text.replace('ntrpu_pcie_guarded_native.sv','ntrpu_pcie_guarded_native_reset.sv')
    text=text.replace('NTRD_GUARDED_NATIVE_SIMULATOR_RETURNED','NTRD_GUARDED_NATIVE_RESET_SIMULATOR_RETURNED')
    return once(text,'set objects {}','lappend names /board/rp_sys_clk_p /board/ep_sys_clk_p\nset objects {}')


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    reference=(root/'fpga/tb/tb_ntrpu_pcie_guarded_mig_r2.sv').read_text()
    products={
        'fpga/tb/ntrpu_pcie_guarded_native_reset.sv':bench((root/'fpga/tb/ntrpu_pcie_guarded_native.sv').read_text(),reference),
        'fpga/tb/ntrpu_pcie_guarded_native_reset_tasks.vh':tasks((root/'fpga/tb/ntrpu_pcie_managed_native_tasks.vh').read_text()),
        'tools/ntrpu_board_guarded_native_reset.tcl':tcl((root/'tools/ntrpu_board_guarded_native.tcl').read_text())}
    assert all(not (root/n).exists() for n in products)
    for name,value in products.items():
        with (root/name).open('x',newline='\n') as f:f.write(value)
