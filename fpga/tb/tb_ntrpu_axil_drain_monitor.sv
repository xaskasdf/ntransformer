`timescale 1ns / 1ps
module tb_ntrpu_axil_drain_monitor;
    reg clk=0;always #5 clk=~clk;
    reg cold=0;
    reg av=0,ar=0,wv=0,wr=0,bv=0,br=0,rv=0,rr=0,qv=0,qr=0;
    reg request=0,fenced=0;
    reg [63:0] tag=0;
    wire ack,fault;
    wire [63:0] ack_tag;
    wire [1:0] na,nw,nr;
    integer scenario=-1,passed=0;
    ntrpu_axil_drain_monitor #(.COUNT_WIDTH(2)) dut(
        .clk(clk),.cold_rst_n(cold),.awvalid(av),.awready(ar),.wvalid(wv),.wready(wr),
        .bvalid(bv),.bready(br),.arvalid(qv),.arready(qr),.rvalid(rv),.rready(rr),
        .req_valid(request),.source_fenced(fenced),.req_tag(tag),
        .ack_valid(ack),.ack_tag(ack_tag),.protocol_fault(fault),
        .aw_pending(na),.w_pending(nw),.ar_pending(nr));
    task tick(input integer n);repeat(n)@(negedge clk);endtask
    task check(input bit ok,input string name);if(!ok)$fatal(1,"DRAIN_ASSERT case=%0d %s",scenario,name);endtask
    task mark(input string name);
        $display("DRAIN_EVENT case=%0d name=%s ps=%0.0f aw=%0d w=%0d ar=%0d ack=%b fault=%b req=%b fenced=%b tag=%h ack_tag=%h",scenario,name,$realtime*1000,na,nw,nr,ack,fault,request,fenced,tag,ack_tag);
    endtask
    task boot(input integer number);
        cold=0;av=0;ar=0;wv=0;wr=0;bv=0;br=0;qv=0;qr=0;rv=0;rr=0;request=0;fenced=0;tag=0;
        tick(3);cold=1;tick(1);scenario=number;
        check(!ack && !fault && na==0 && nw==0 && nr==0,"cold inventory");
    endtask
    task ask(input [63:0] id);
        request=1;tag=id;fenced=1;#0.001;check(!ack,"new request cannot inherit ACK");
    endtask
    task ready_ack;
        tick(3);check(ack && ack_tag==tag && !fault,"fresh drained ACK");mark("ACK");
    endtask
    task pass_case(input string name);
        passed=passed+1;$display("DRAIN_CASE_PASS case=%0d name=%s ps=%0.0f",scenario,name,$realtime*1000);
    endtask
    task take_aw;av=1;ar=1;tick(1);av=0;ar=0;endtask
    task take_w;wv=1;wr=1;tick(1);wv=0;wr=0;endtask
    task take_b;bv=1;br=1;tick(1);bv=0;br=0;endtask
    task take_ar;qv=1;qr=1;tick(1);qv=0;qr=0;endtask
    task take_r;rv=1;rr=1;tick(1);rv=0;rr=0;endtask

    // Same-domain integration fixture: the monitor owns ONLY the QUIESCE ACK.
    // Other sequencer actions are controlled bench stimuli; no XDMA/CDC/MIG.
    reg scold=0,start=0,other_ack=0;
    wire req,seq_local,seq_ep,seq_retained,seq_blocked;
    wire [2:0] kind;
    wire [63:0] st,monitor_tag;
    wire [3:0] state;
    reg saw=0,sw=0,sb=0,sbready=0;
    wire monitor_ack,monitor_fault;
    wire [3:0] sa_count,sw_count,sr_count;
    wire seq_ack=(kind==1) ? monitor_ack : other_ack;
    ntrpu_endpoint_reset_sequencer #(.HOLD_CYCLES(7)) sequence_dut(
        .mgmt_clk(clk),.cold_rst_n(scold),.start(start),.retry(1'b0),.external_hold(1'b0),
        .endpoint_held(1'b0),.endpoint_available(1'b1),.ack_valid(seq_ack),.ack_kind(kind),
        .ack_tag(kind==1 ? monitor_tag : st),.req_valid(req),.req_kind(kind),.req_tag(st),
        .invalidate_req(),.block_admission(seq_blocked),.endpoint_reset_req(seq_ep),
        .local_reset_req(seq_local),.owner_retained(seq_retained),.complete(),.exhausted(),
        .rejected_ack_cycles(),.state_debug(state),.hold_elapsed());
    ntrpu_axil_drain_monitor integrated(
        .clk(clk),.cold_rst_n(scold),.awvalid(saw),.awready(1'b1),.wvalid(sw),.wready(1'b1),
        .bvalid(sb),.bready(sbready),.arvalid(1'b0),.arready(1'b0),.rvalid(1'b0),.rready(1'b0),
        .req_valid(req && kind==1),.source_fenced(req && kind==1),.req_tag(st),
        .ack_valid(monitor_ack),.ack_tag(monitor_tag),.protocol_fault(monitor_fault),
        .aw_pending(sa_count),.w_pending(sw_count),.ar_pending(sr_count));

    initial begin
        boot(0);ask(64'h100000001);tick(1);check(!ack,"capture first");
        tick(1);check(!ack,"first quiet edge insufficient");tick(1);check(ack,"second quiet edge");mark("ACK");
        tag=64'h200000001;#0.001;check(!ack,"full 64-bit identity change revokes ACK");ready_ack();
        request=0;#0.001;check(!ack,"request removal revokes ACK");tick(1);ask(tag);ready_ack();
        pass_case("fresh_request_and_identity");

        boot(1);take_aw();ask(2);tick(6);check(na==1 && nw==0 && !ack && !fault,"AW-only must wait despite idle pins");mark("AW_ONLY");
        take_w();bv=1;tick(5);check(na==1 && nw==1 && !ack,"blocked B retains both halves");mark("B_BLOCKED");
        br=1;tick(1);bv=0;br=0;ready_ack();pass_case("aw_before_w_and_blocked_b");

        boot(2);take_w();ask(3);tick(6);check(na==0 && nw==1 && !ack && !fault,"W-only waits");mark("W_ONLY");
        take_aw();take_b();ready_ack();pass_case("w_before_aw");

        boot(3);take_ar();ask(4);tick(6);check(nr==1 && !ack,"AR with no R waits");mark("AR_PENDING");
        rv=1;tick(5);check(nr==1 && !ack,"R backpressure");mark("R_BLOCKED");
        rr=1;tick(1);rv=0;rr=0;ready_ack();pass_case("read_and_blocked_r");

        boot(4);ask(5);av=1;tick(5);check(na==0 && !ack && !fault,"unaccepted held AW cannot ACK");mark("VALID_HELD");
        ar=1;tick(1);av=0;ar=0;take_w();take_b();ready_ack();
        fenced=0;#0.001;check(!ack,"fence withdrawal revokes ACK");tick(2);fenced=1;tick(1);check(!ack,"fence must requalify");
        tick(1);check(ack,"fence requalified");pass_case("held_valid_and_external_fence");

        boot(5);repeat(3)begin take_aw();take_w();take_ar();end
        ask(6);check(na==3 && nw==3 && nr==3,"maximum inventory");
        av=1;ar=1;wv=1;wr=1;bv=1;br=1;qv=1;qr=1;rv=1;rr=1;tick(1);
        av=0;ar=0;wv=0;wr=0;bv=0;br=0;qv=0;qr=0;rv=0;rr=0;
        check(na==3 && nw==3 && nr==3 && !fault && !ack,"simultaneous retire/accept at capacity");mark("FULL_BALANCED");
        repeat(3)begin take_b();take_r();end
        ready_ack();pass_case("multiple_and_simultaneous_transactions");

        boot(6);ask(7);take_b();tick(5);check(fault && !ack && na==0 && nw==0,"unexpected B sticky fault");
        request=0;tick(1);ask(8);tick(5);check(fault && !ack,"new tag cannot clear fault");mark("FAULT");pass_case("unexpected_b");
        boot(7);take_aw();ask(9);take_b();tick(5);check(fault && !ack,"B before W fault");mark("FAULT");pass_case("b_before_both_halves");
        boot(8);ask(10);take_r();tick(5);check(fault && !ack && nr==0,"unexpected R sticky fault");mark("FAULT");pass_case("unexpected_r");
        boot(9);repeat(4)take_aw();ask(11);tick(5);check(fault && !ack && na==3,"AW overflow saturates");mark("FAULT");pass_case("aw_overflow");
        boot(10);repeat(4)take_w();ask(12);tick(5);check(fault && !ack && nw==3,"W overflow saturates");mark("FAULT");pass_case("w_overflow");
        boot(11);repeat(4)take_ar();ask(13);tick(5);check(fault && !ack && nr==3,"AR overflow saturates");mark("FAULT");pass_case("ar_overflow");

        scenario=12;tick(3);scold=1;saw=1;tick(1);saw=0;start=1;tick(1);start=0;
        check(state==1 && !seq_local && !seq_ep,"ownership first");other_ack=1;tick(1);other_ack=0;
        tick(10);check(state==2 && seq_retained && seq_blocked && !seq_local && !seq_ep && !monitor_ack,"sequencer waits for AW-only transaction");
        $display("DRAIN_INTEGRATION event=AW_ONLY state=%0d local_reset=%b endpoint_reset=%b aw=%0d w=%0d",state,seq_local,seq_ep,sa_count,sw_count);
        sw=1;tick(1);sw=0;sb=1;tick(10);check(state==2 && !seq_local && !seq_ep && !monitor_ack,"sequencer waits for blocked B");
        $display("DRAIN_INTEGRATION event=B_BLOCKED state=%0d local_reset=%b endpoint_reset=%b aw=%0d w=%0d",state,seq_local,seq_ep,sa_count,sw_count);
        sbready=1;tick(1);sb=0;sbready=0;tick(2);check(state==2 && !seq_local,"fresh observation before consumption");
        tick(1);check(state==3 && seq_local && !seq_ep && !monitor_fault,"local reset only after drained tagged ACK");
        $display("DRAIN_INTEGRATION event=LOCAL_HELD state=%0d local_reset=%b endpoint_reset=%b aw=%0d w=%0d",state,seq_local,seq_ep,sa_count,sw_count);
        pass_case("sequencer_waits_for_real_ledger");
        check(passed==13,"case count");$display("DRAIN_SUITE_PASS cases=13 native_endpoint=0 cdc=0 scope=axil_only");$finish;
    end
    initial begin #100000;$fatal(1,"DRAIN_TIMEOUT");end
endmodule
