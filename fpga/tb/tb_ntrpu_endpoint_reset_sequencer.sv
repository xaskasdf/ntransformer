`timescale 1ns/1ps
// Contract fixtures for a management sequencer, not a native PCIe/MIG test.
module ntrpu_endpoint_seq_fixture #(
    parameter integer SPAN=7, PROBE=0,
    parameter [63:0] FIRST=64'hfffffffe
)(input wire clk);
    reg cold=0,start=0,retry=0,external_hold=0,held=0,available_enable=1;
    reg av=0;
    reg [2:0] ak=0;
    reg [63:0] atag=0;
    wire rv,invalid,blocked,er,lr,retained,complete,exhausted,alive;
    wire [2:0] rk;
    wire [63:0] rt,count;
    wire [31:0] rejected;
    wire [3:0] state;
    reg endpoint_clk=0,endpoint_run=1;
    integer scenario=-1;
    reg [63:0] cycles=0;
    reg [3:0] last_state=15;
    reg [63:0] last_tag=0;
    wire available=available_enable && alive && !held;
    initial if(PROBE) forever begin wait(endpoint_run); #3.5; if(endpoint_run)endpoint_clk=~endpoint_clk; end
    generate if(PROBE) begin
        ntrpu_clock_probe #(.TIMEOUT_CYCLES(16)) probe(
            .mgmt_clk(clk),.observed_clk(endpoint_clk),.alive(alive));
    end else assign alive=1'b1; endgenerate
    ntrpu_endpoint_reset_sequencer #(.HOLD_CYCLES(SPAN),.FIRST_TAG(FIRST)) dut(
        .mgmt_clk(clk),.cold_rst_n(cold),.start(start),.retry(retry),.external_hold(external_hold),
        .endpoint_held(held),.endpoint_available(available),.ack_valid(av),.ack_kind(ak),.ack_tag(atag),
        .req_valid(rv),.req_kind(rk),.req_tag(rt),.invalidate_req(invalid),.block_admission(blocked),
        .endpoint_reset_req(er),.local_reset_req(lr),.owner_retained(retained),.complete(complete),
        .exhausted(exhausted),.rejected_ack_cycles(rejected),.state_debug(state),.hold_elapsed(count));
    always @(posedge clk) begin
        cycles=cycles+1;
        #0.001;
        if(scenario>=0) begin
            if(cold && (er || lr) && !retained) $fatal(1,"SEQ_RESET_BEFORE_OWNER");
            if(cold && retained && (!invalid || !blocked)) $fatal(1,"SEQ_REOPENED_OWNER");
            if(cold && complete && (er || lr || !retained || !blocked)) $fatal(1,"SEQ_BAD_COMPLETE");
            if(cold && count>SPAN) $fatal(1,"SEQ_COUNTER_OVERFLOW");
            if(state!=last_state || rt!=last_tag)
                $display("SEQ_TRACE case=%0d ps=%0.0f cycle=%0d state=%0d rv=%b kind=%0d tag=%h count=%0d er=%b lr=%b owner=%b invalid=%b blocked=%b complete=%b held=%b available=%b ext=%b ack=%b ak=%0d atag=%h",scenario,$realtime*1000,cycles,state,rv,rk,rt,count,er,lr,retained,invalid,blocked,complete,held,available,external_hold,av,ak,atag);
        end
        last_state=state;last_tag=rt;
    end
    task tick(input integer n);repeat(n) @(negedge clk);endtask
    task check(input bit condition,input string label);
        if(!condition)$fatal(1,"SEQ_ASSERT case=%0d %s",scenario,label);
    endtask
    task mark(input string name);
        $display("SEQ_EVENT case=%0d name=%s ps=%0.0f cycle=%0d state=%0d tag=%h count=%0d er=%b lr=%b owner=%b blocked=%b complete=%b alive=%b rejected=%0d",scenario,name,$realtime*1000,cycles,state,rt,count,er,lr,retained,blocked,complete,alive,rejected);
    endtask
    task boot(input integer number);
        scenario=-1;cold=0;start=0;retry=0;external_hold=0;held=0;available_enable=1;av=0;endpoint_run=1;
        tick(3);cold=1;tick(100);check(alive,"clock alive before request");scenario=number;
        start=1;tick(1);start=0;check(state==1 && rv && rk==0,"initial owner request");mark("START");
    endtask
    task acknowledge(input [2:0] kind,input [63:0] tag);
        av=1;ak=kind;atag=tag;tick(1);av=0;
    endtask
    task good(input [2:0] kind);
        check(rv && rk==kind,"expected request kind");acknowledge(kind,rt);
    endtask
    task stale(input [2:0] kind,input [63:0] tag);
        reg [3:0] old_state;reg [63:0] old_tag;
        begin old_state=state;old_tag=rt;acknowledge(kind,tag);
            check(state==old_state && rt==old_tag,"stale ACK advanced phase");mark("STALE_REJECTED");end
    endtask
    task enter_hold;
        good(0);good(1);check(lr && !er,"local held before endpoint");good(2);
        held=1;good(3);check(state==5 && er && lr && count==0,"hold begins at zero");mark("HOLD_START");
    endtask
    task await_release;
        integer wait_cycles;
        begin wait_cycles=0;
            while(state==5) begin tick(1);wait_cycles=wait_cycles+1;check(wait_cycles<=SPAN+4,"bounded release");end
            check(state==6 && !er && lr && count==SPAN,"endpoint released with local held");mark("EP_RELEASE");
        end
    endtask
    task finish_ready;
        held=0;available_enable=1;good(4);check(state==7 && !lr && !er,"local release after endpoint ready");
        good(5);check(complete && state==8 && invalid && blocked,"completed but retained");mark("DONE");
    endtask
    task pass_case(input string name);
        $display("SEQ_CASE_PASS case=%0d name=%s ps=%0.0f",scenario,name,$realtime*1000);
    endtask
endmodule

module tb_ntrpu_endpoint_reset_sequencer;
    reg clk=0;always #5 clk=~clk;
    ntrpu_endpoint_seq_fixture #(.PROBE(1)) fast(clk);
    ntrpu_endpoint_seq_fixture #(.SPAN(5000001),.FIRST(1)) full(clk);
    ntrpu_endpoint_seq_fixture #(.FIRST(64'hffffffffffffffff)) wrap(clk);
    reg [63:0] old_tag,hold_cycle;
    realtime hold_ps;
    initial begin
        fast.boot(0);
        fast.stale(0,0);fast.stale(1,fast.rt);fast.good(0);
        fast.stale(0,64'hfffffffe);fast.good(1);
        fast.check(fast.lr && !fast.er,"drain before local reset");fast.stale(3,fast.rt);fast.good(2);
        fast.good(3);fast.check(fast.state==4,"held observation required even with matching ACK");
        fast.held=1;fast.good(3);fast.mark("HOLD_START");hold_cycle=fast.cycles;
        fast.await_release();fast.check(fast.cycles-hold_cycle==7,"seven complete intervals");
        fast.held=0;fast.available_enable=0;old_tag=fast.rt;fast.good(4);
        fast.check(fast.state==6 && fast.lr,"ACK without available cannot release local");
        fast.available_enable=1;fast.stale(4,old_tag-1);fast.good(4);
        fast.stale(5,old_tag);fast.good(5);fast.mark("DONE");
        fast.start=1;fast.tick(10);fast.start=0;fast.check(fast.complete && fast.blocked,"start cannot rearm");
        fast.check(fast.rt>64'hffffffff,"64-bit tag carry");fast.pass_case("normal_and_stale_ack");

        fast.boot(1);fast.tick(20);fast.check(fast.state==1 && !fast.er && !fast.lr,"missing owner cannot reset");
        fast.mark("WAIT_OWNER");fast.pass_case("missing_owner_ack");
        fast.boot(2);fast.good(0);fast.external_hold=1;fast.tick(20);
        fast.check(fast.state==2 && !fast.er && !fast.lr,"missing drain cannot reset");
        fast.mark("WAIT_DRAIN");fast.pass_case("missing_quiesce_ack");

        fast.boot(3);fast.enter_hold();old_tag=fast.rt;fast.tick(4);fast.held=0;fast.tick(1);
        fast.check(fast.state==4 && fast.er && fast.lr && fast.count==0,"hold observation loss restarts qualification");
        fast.held=1;fast.stale(3,old_tag);fast.good(3);fast.mark("HOLD_RESTART");hold_cycle=fast.cycles;
        fast.await_release();fast.check(fast.cycles-hold_cycle==7,"restarted full span");fast.finish_ready();
        fast.pass_case("held_observation_lost");

        fast.boot(4);fast.external_hold=1;fast.enter_hold();fast.tick(30);
        fast.check(fast.state==5 && fast.count==7 && fast.er && fast.lr,"external hold and counter saturation");
        fast.mark("EXT_HELD");fast.external_hold=0;fast.tick(1);
        fast.check(fast.state==6 && !fast.er && fast.lr,"release after external hold");
        old_tag=fast.rt;fast.external_hold=1;fast.tick(1);fast.external_hold=0;
        fast.check(fast.state==2 && fast.lr && !fast.er,"repeat PERST needs fresh drain before reassertion");
        fast.stale(1,64'hffffffff);fast.mark("REPEAT_PERST");fast.pass_case("external_hold_and_repeat_perst");

        fast.boot(5);fast.enter_hold();fast.endpoint_run=0;fast.tick(60);
        fast.check(!fast.alive && fast.state==6 && !fast.er && fast.lr,"independent timer releases endpoint while clock absent");
        fast.held=0;fast.good(4);fast.check(fast.state==6 && fast.lr,"stopped clock blocks readiness");
        fast.mark("CLOCK_STOPPED");fast.endpoint_run=1;fast.tick(100);fast.check(fast.alive,"clock returned");
        fast.finish_ready();fast.pass_case("actual_clock_probe_stop_return");

        fast.boot(6);fast.enter_hold();fast.await_release();fast.held=0;fast.good(4);old_tag=fast.rt;
        fast.available_enable=0;fast.tick(1);
        fast.check(fast.state==2 && !fast.er && !fast.lr && !fast.complete,"loss during local release returns to drain");
        fast.stale(5,old_tag);fast.mark("LOSS_DURING_RELEASE");fast.good(1);fast.good(2);fast.held=1;fast.good(3);
        fast.mark("HOLD_RETRY");fast.await_release();fast.finish_ready();
        fast.retry=1;fast.tick(1);fast.retry=0;fast.check(fast.state==2 && !fast.complete,"explicit retry after completion");
        fast.mark("RETRY_COMPLETE");fast.pass_case("loss_during_release_and_retry");

        wrap.boot(7);wrap.good(0);wrap.check(wrap.state==9 && wrap.exhausted && !wrap.rv && !wrap.er && !wrap.lr && wrap.blocked,"no tag wrap");
        wrap.retry=1;wrap.tick(10);wrap.check(wrap.state==9 && !wrap.complete,"exhaustion locked");
        wrap.mark("EXHAUSTED");wrap.pass_case("tag_exhaustion");
        fast.scenario=-1;fast.endpoint_run=0;wrap.scenario=-1;

        full.boot(8);full.enter_hold();hold_cycle=full.cycles;hold_ps=$realtime*1000;
        full.tick(5000000);
        full.check(full.state==5 && full.er && full.lr && full.count==5000000,"last full interval still held");
        full.mark("FULL_LAST_HELD");full.tick(1);
        full.check(full.state==6 && !full.er && full.lr && full.count==5000001,"full timer release boundary");
        full.check(full.cycles-hold_cycle==5000001,"full interval count");
        full.check($realtime*1000-hold_ps==50000010000.0,"full 50ms span in simulator time");
        full.mark("EP_RELEASE");full.finish_ready();full.pass_case("full_50ms_counter");
        $display("SEQ_SUITE_PASS cases=9 full_cycles=5000001 period_ps=10000 hold_ps=50000010000 native_endpoint=0");
        $finish;
    end
    initial begin #51000000;$fatal(1,"SEQ_SUITE_TIMEOUT");end
endmodule
