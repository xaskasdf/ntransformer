`timescale 1ns / 1ps
module tb_ntrpu_tagged_reset_mailbox;
    localparam [63:0] GENERATION=64'h000000010000abcd;
    reg mclk=0,pclk=0,mrun=1,prun=1;
    always #5 if(mrun)mclk=~mclk;
    initial begin #1.3;forever begin #3.5;if(prun)pclk=~pclk;end end
    reg request=0;
    reg [2:0] kind=0;
    reg [63:0] generation=GENERATION,tag=0;
    reg pav=0;
    reg [2:0] pak=0;
    reg [63:0] pag=0,pat=0;
    wire ack,busy,peer_req;
    wire [2:0] ack_kind,pk;
    wire [63:0] ack_gen,ack_tag,pg,pt;
    wire [31:0] obsolete,rejected;
    integer scenario=0,passed=0,deliveries=0,answers=0;
    reg last_peer=0,last_ack=0,last_accept=0;

    // Real ledger runs at peer_clk; only its external isolation contract and
    // the sequencer's ownership action remain controlled stimuli.
    reg integration=0,ledger_cold=0,seq_cold=0,seq_start=0,owner_ack=0;
    reg saw=0,sw=0,sb=0,sbready=0;
    wire seq_req,seq_local,seq_ep,seq_blocked;
    wire [2:0] seq_kind;
    wire [63:0] seq_tag;
    wire [3:0] seq_state;
    wire ledger_ack,ledger_fault;
    wire [63:0] ledger_tag;
    wire [3:0] aw_pending,w_pending,ar_pending;
    wire input_req=integration ? seq_req && seq_kind==1 : request;
    wire [2:0] input_kind=integration ? seq_kind : kind;
    wire [63:0] input_generation=integration ? GENERATION : generation;
    wire [63:0] input_tag=integration ? seq_tag : tag;

    ntrpu_tagged_reset_mailbox dut(
        .mgmt_clk(mclk),.peer_clk(pclk),.req_valid(input_req),.req_kind(input_kind),
        .req_generation(input_generation),.req_tag(input_tag),.ack_valid(ack),.ack_kind(ack_kind),
        .ack_generation(ack_gen),.ack_tag(ack_tag),.busy(busy),.obsolete_completions(obsolete),
        .peer_req_valid(peer_req),.peer_req_kind(pk),.peer_req_generation(pg),.peer_req_tag(pt),
        .peer_ack_valid(integration ? ledger_ack : pav),.peer_ack_kind(integration ? 3'd1 : pak),
        .peer_ack_generation(integration ? GENERATION : pag),.peer_ack_tag(integration ? ledger_tag : pat),
        .rejected_peer_ack_cycles(rejected));
    ntrpu_axil_drain_monitor ledger(
        .clk(pclk),.cold_rst_n(ledger_cold),.awvalid(saw),.awready(1'b1),.wvalid(sw),.wready(1'b1),
        .bvalid(sb),.bready(sbready),.arvalid(1'b0),.arready(1'b0),.rvalid(1'b0),.rready(1'b0),
        .req_valid(integration && peer_req && pk==1 && pg==GENERATION),.source_fenced(integration),.req_tag(pt),
        .ack_valid(ledger_ack),.ack_tag(ledger_tag),.protocol_fault(ledger_fault),
        .aw_pending(aw_pending),.w_pending(w_pending),.ar_pending(ar_pending));
    ntrpu_endpoint_reset_sequencer #(.HOLD_CYCLES(7)) sequencer(
        .mgmt_clk(mclk),.cold_rst_n(seq_cold),.start(seq_start),.retry(1'b0),.external_hold(1'b0),
        .endpoint_held(1'b0),.endpoint_available(1'b1),.ack_valid(seq_kind==1 ? ack : owner_ack),
        .ack_kind(seq_kind==1 ? ack_kind : seq_kind),.ack_tag(seq_kind==1 ? ack_tag : seq_tag),
        .req_valid(seq_req),.req_kind(seq_kind),.req_tag(seq_tag),.invalidate_req(),
        .block_admission(seq_blocked),.endpoint_reset_req(seq_ep),.local_reset_req(seq_local),
        .owner_retained(),.complete(),.exhausted(),.rejected_ack_cycles(),.state_debug(seq_state),.hold_elapsed());

    task mtick(input integer n);repeat(n)@(negedge mclk);endtask
    task ptick(input integer n);repeat(n)@(negedge pclk);endtask
    task check(input bit ok,input string name);if(!ok)$fatal(1,"MAILBOX_ASSERT case=%0d %s",scenario,name);endtask
    task mark(input string name);
        $display("MAILBOX_EVENT case=%0d name=%s ps=%0.0f busy=%b ack=%b peer=%b obsolete=%0d rejected=%0d deliveries=%0d answers=%0d",scenario,name,$realtime*1000,busy,ack,peer_req,obsolete,rejected,deliveries,answers);
    endtask
    task set_request(input [63:0] g,input [63:0] t,input [2:0] k);
        @(negedge mclk);generation=g;tag=t;kind=k;request=1;
        $display("MAILBOX_INTENT case=%0d ps=%0.0f generation=%h tag=%h kind=%0d",scenario,$realtime*1000,g,t,k);
    endtask
    task wait_peer(input [63:0] g,input [63:0] t,input [2:0] k);
        integer n;
        begin n=0;while(!(peer_req && pg==g && pt==t && pk==k))begin ptick(1);n=n+1;check(n<200,"bounded peer delivery");end end
    endtask
    task answer(input [63:0] g,input [63:0] t,input [2:0] k);
        @(negedge pclk);pag=g;pat=t;pak=k;pav=1;ptick(1);pav=0;
    endtask
    task wait_ack;
        integer n;
        begin n=0;while(!ack)begin mtick(1);n=n+1;check(n<200,"bounded management ACK");end
            check({ack_gen,ack_tag,ack_kind}=={generation,tag,kind},"management identity");end
    endtask
    task wait_idle;
        integer n;
        begin n=0;while(busy)begin mtick(1);n=n+1;check(n<200,"complete four-phase drain");end end
    endtask
    task finish_request;
        wait_ack();wait_idle();@(negedge mclk);request=0;mtick(1);check(!ack,"withdrawn request cannot retain ACK");
    endtask
    task pass_case(input string name);
        passed=passed+1;$display("MAILBOX_CASE_PASS case=%0d name=%s ps=%0.0f",scenario,name,$realtime*1000);
    endtask
    always @(posedge pclk)begin
        #0.001;
        if(peer_req && !last_peer)begin
            deliveries=deliveries+1;
            $display("MAILBOX_DELIVER case=%0d ps=%0.0f generation=%h tag=%h kind=%0d",scenario,$realtime*1000,pg,pt,pk);
        end
        if(dut.dest_ack && !last_accept)
            $display("MAILBOX_ACCEPT case=%0d ps=%0.0f generation=%h tag=%h kind=%0d",scenario,$realtime*1000,pg,pt,pk);
        last_accept=dut.dest_ack;
        last_peer=peer_req;
    end
    always @(posedge mclk)begin
        #0.001;
        if(ack && !last_ack)begin
            answers=answers+1;
            $display("MAILBOX_ACK case=%0d ps=%0.0f generation=%h tag=%h kind=%0d",scenario,$realtime*1000,ack_gen,ack_tag,ack_kind);
        end
        last_ack=ack;
    end

    initial begin
        #200;mtick(2);ledger_cold=1;
        set_request(GENERATION,64'h100000001,0);wait_peer(generation,tag,kind);ptick(8);check(!ack,"no action ACK no certificate");
        answer(generation,tag,kind);wait_ack();wait_idle();mtick(10);check(deliveries==1 && answers==1,"held request once");
        request=0;mtick(1);pass_case("normal_delayed_ack");

        scenario=1;set_request(GENERATION,64'h100000002,2);wait_peer(generation,tag,kind);ptick(2);
        answer(generation,tag-1,kind);answer(generation,tag,3);answer(generation^64'h100000000,tag,kind);
        mtick(10);check(peer_req && !ack && rejected>=3,"wrong tag kind generation rejected");mark("WRONG_FIELDS");
        answer(generation,tag,kind);finish_request();pass_case("wrong_identity_fields");

        scenario=2;pag=GENERATION;pat=64'h100000003;pak=4;pav=1;
        set_request(GENERATION,pat,pak);wait_peer(generation,tag,kind);ptick(12);
        check(peer_req && !ack,"preasserted matching ACK rejected");mark("PREASSERTED_ACK");pav=0;ptick(2);
        answer(generation,tag,kind);finish_request();pass_case("preasserted_matching_ack");

        scenario=3;@(negedge pclk);prun=0;
        set_request(GENERATION,64'h100000004,1);mtick(20);
        check(busy && !ack && deliveries==3,"stopped destination cannot accept");mark("PEER_STOP_BEFORE");
        prun=1;wait_peer(generation,tag,kind);ptick(2);answer(generation,tag,kind);finish_request();pass_case("peer_stopped_before_request");

        scenario=4;set_request(GENERATION,64'h100000005,3);wait_peer(generation,tag,kind);ptick(2);prun=0;
        pag=generation;pat=tag;pak=kind;pav=1;mtick(20);check(peer_req && busy && !ack,"stopped peer cannot sample action ACK");mark("PEER_STOP_DURING");
        prun=1;ptick(2);pav=0;finish_request();pass_case("peer_stopped_during_action");

        scenario=5;set_request(GENERATION,64'h100000006,5);wait_peer(generation,tag,kind);ptick(2);
        @(negedge mclk);mrun=0;answer(generation,tag,kind);ptick(30);check(busy && !ack,"stopped management cannot sample return");mark("MGMT_STOPPED");
        mrun=1;finish_request();pass_case("management_stopped_during_return");

        scenario=6;set_request(GENERATION,64'h100000007,0);wait_peer(generation,tag,kind);ptick(2);
        @(negedge mclk);request=0;answer(generation,tag,kind);wait_idle();
        check(!ack && obsolete==1 && answers==6,"canceled response discarded");mark("CANCELED");pass_case("canceled_request_drains");

        scenario=7;set_request(GENERATION,64'h100000008,1);wait_peer(generation,tag,kind);ptick(2);
        set_request(GENERATION,64'h100000009,2);answer(GENERATION,64'h100000008,1);
        wait_peer(generation,tag,kind);check(!ack && obsolete==2,"replacement did not relabel old return");mark("REPLACED");
        ptick(2);answer(generation,tag,kind);finish_request();pass_case("replacement_while_busy");

        scenario=8;set_request(GENERATION,64'h10000000a,3);wait_peer(generation,tag,kind);ptick(2);
        set_request(GENERATION,64'h10000000b,4);set_request(GENERATION,64'h10000000c,5);
        answer(GENERATION,64'h10000000a,3);wait_peer(generation,tag,kind);
        check(obsolete==3 && deliveries==11 && !ack,"latest request retained without invented queue");mark("COALESCED");
        ptick(2);answer(generation,tag,kind);finish_request();pass_case("multiple_replacements_coalesce");

        scenario=9;set_request(GENERATION^64'h100000000,64'h10000000c,5);wait_peer(generation,tag,kind);ptick(2);
        answer(GENERATION,tag,kind);mtick(10);check(!ack && peer_req,"prior generation cannot approve same tag");mark("OLD_GENERATION");
        answer(generation,tag,kind);finish_request();pass_case("generation_identity");

        scenario=10;set_request(generation,tag,kind);mtick(30);
        check(!ack && !busy && deliveries==12 && answers==9,"last identity never reissued after withdrawal");mark("REPLAY_SUPPRESSED");pass_case("same_identity_not_rearmed");

        scenario=11;@(negedge mclk);request=0;integration=1;seq_cold=1;
        @(negedge pclk);saw=1;ptick(1);saw=0;
        @(negedge mclk);seq_start=1;mtick(1);seq_start=0;check(seq_state==1,"sequence owner first");
        owner_ack=1;mtick(1);owner_ack=0;
        $display("MAILBOX_INTENT case=11 ps=%0.0f generation=%h tag=%h kind=%0d",$realtime*1000,GENERATION,seq_tag,seq_kind);
        wait_peer(GENERATION,seq_tag,1);mtick(20);
        check(seq_state==2 && !seq_local && !seq_ep && aw_pending==1 && w_pending==0 && !ack,"AW-only prevents reset across clocks");
        $display("MAILBOX_INTEGRATION event=AW_ONLY state=%0d local_reset=%b endpoint_reset=%b aw=%0d w=%0d",seq_state,seq_local,seq_ep,aw_pending,w_pending);
        @(negedge pclk);sw=1;ptick(1);sw=0;sb=1;mtick(20);
        check(seq_state==2 && !seq_local && !seq_ep && aw_pending==1 && w_pending==1 && !ack,"blocked B prevents reset across clocks");
        $display("MAILBOX_INTEGRATION event=B_BLOCKED state=%0d local_reset=%b endpoint_reset=%b aw=%0d w=%0d",seq_state,seq_local,seq_ep,aw_pending,w_pending);
        @(negedge pclk);sbready=1;ptick(1);sb=0;sbready=0;
        begin:await_local integer n;n=0;while(seq_state==2)begin mtick(1);n=n+1;check(n<200,"bounded sequence advance");end end
        check(seq_state==3 && seq_local && !seq_ep && seq_blocked && !ledger_fault,"fresh CDC drain proof releases sequence");
        $display("MAILBOX_INTEGRATION event=LOCAL_HELD state=%0d local_reset=%b endpoint_reset=%b aw=%0d w=%0d",seq_state,seq_local,seq_ep,aw_pending,w_pending);
        wait_idle();mtick(2);check(deliveries==13 && answers==10 && obsolete==3,"conserved terminal counts");
        pass_case("sequencer_axil_ledger_across_clocks");
        check(passed==12,"twelve cases");
        $display("MAILBOX_SUITE_PASS cases=12 deliveries=13 answers=10 obsolete=3 mgmt_period_ps=10000 peer_period_ps=7000 native_endpoint=0");$finish;
    end
    initial begin #100000;$fatal(1,"MAILBOX_TIMEOUT");end
endmodule
