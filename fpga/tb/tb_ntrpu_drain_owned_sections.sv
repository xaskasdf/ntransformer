`timescale 1ns / 1ps
module tb_ntrpu_drain_owned_sections #(
    parameter integer DEVICE_PERIOD_PS=5000,CONTRACT_MODE=0
);
    localparam integer EXTERNAL_FIXTURE=(CONTRACT_MODE==3);
    localparam integer OWNER_PHASE=(CONTRACT_MODE==1 ? 1 : CONTRACT_MODE==2 ? 2 : 0);
    reg mgmt_clk=0;always #5 mgmt_clk=~mgmt_clk;
    reg cold=0,req=0;
    reg [63:0] generation=64'h100000001,tag=1;
    wire [6:0] present=7'd121 | ((CONTRACT_MODE & 1)?7'd2:7'd0) | ((CONTRACT_MODE & 2)?7'd4:7'd0);
    reg [6:0] ready=127;
    wire [6:0] proof_req,ack,accepted,armed,retained_faults,fault;
    wire [447:0] ack_gen,ack_tag;
    wire [63:0] pg,pt,cg,ct;
    wire complete,revoked,replay;
    wire [1:0] ready_close,forward_held,local_held,drained,host_clock,peer_cold;
    wire [1:0] section_ack,local_idle,requests_present,replies_pending,section_fault;
    reg release_forward=0,late_offer=0;
    reg [1:0] release_local=0,host_enable=3;
    wire [1:0] lite_sources;
    wire [6:0] axi4_sources;
    wire late_accepted;
    wire [1:0] local_busy,local_peer_req,local_peer_ack,local_revoked;
    wire [1:0] lite_fault;
    wire [1:0] axi4_fault;
    wire [63:0] obsolete;
    reg [2:0] fixture_ack=0;
    reg [191:0] fixture_gen=0,fixture_tag=0;
    ntrpu_drain_certificate_join joiner(.mgmt_clk(mgmt_clk),.cold_rst_n(cold),.req_valid(req),
        .generation(generation),.tag(tag),.contract_present(present),.peer_ready(ready),.proof_fault(fault),
        .ack_valid(ack),.ack_generation(ack_gen),.ack_tag(ack_tag),.proof_request(proof_req),
        .proof_generation(pg),.proof_tag(pt),.complete_ack(complete),.complete_generation(cg),.complete_tag(ct),
        .accepted_mask(accepted),.armed_mask(armed),.retained_faults(retained_faults),.revoked(revoked),.identity_fault(replay));
    // Slot 0 is the real owner. Only slots 1/2 remain explicit fixtures,
    // independently absent in modes 0/1/2. Mode 3 is a control experiment.
    always @(posedge mgmt_clk)for(integer i=1;i<3;i=i+1)begin
        if(!proof_req[i])fixture_ack[i]<=0;
        else if(fixture_gen[i*64+:64]!=pg || fixture_tag[i*64+:64]!=pt)begin
            fixture_ack[i]<=0;fixture_gen[i*64+:64]<=pg;fixture_tag[i*64+:64]<=pt;
        end else if(i!=1 || (&local_held))fixture_ack[i]<=1;
    end
    assign ack[2:1]=fixture_ack[2:1];assign ack_gen[191:64]=fixture_gen[191:64];assign ack_tag[191:64]=fixture_tag[191:64];
    assign fault[2:1]=0;
    reg owner_enable=1,invalidate_owner=0,release_owner_cmd=0,release_owner_payload=0,try_owner_reuse=0;
    wire owner_ready,owner_done,owner_reuse_tested,owner_clock,owner_cold,owner_lost,owner_invalid;
    wire owner_cmd,owner_data,owner_result,owner_busy,owner_peer_req,owner_peer_ack,owner_revoked;
    wire [63:0] owner_peer_gen,owner_peer_tag;
    wire [31:0] owner_obsolete;
    ntrpu_drain_owner_fixture #(.SOURCE_PERIOD_PS(DEVICE_PERIOD_PS),.LOSS_PHASE(OWNER_PHASE)) owner(
        .mgmt_clk(mgmt_clk),.source_enable(owner_enable),.proof_request(proof_req[0]),
        .invalidate_request(invalidate_owner),.release_cmd(release_owner_cmd),.release_payload(release_owner_payload),
        .try_reuse(try_owner_reuse),.generation(pg),.proof_tag(pt),
        .proof_ack(ack[0]),.proof_generation(ack_gen[63:0]),.proof_ack_tag(ack_tag[63:0]),
        .busy(owner_busy),.peer_request(owner_peer_req),.peer_answer(owner_peer_ack),
        .peer_generation(owner_peer_gen),.peer_tag(owner_peer_tag),.fault_mgmt(fault[0]),
        .revoked_source(owner_revoked),.obsolete(owner_obsolete),
        .source_clk_o(owner_clock),.cold_o(owner_cold),.lost_o(owner_lost),.invalid_o(owner_invalid),
        .cmd_held_o(owner_cmd),.data_held_o(owner_data),.result_held_o(owner_result),
        .owner_ready(owner_ready),.owner_done(owner_done),.reuse_tested(owner_reuse_tested));
    assign ack[4:3]=section_ack;assign fault[4:3]=section_fault;
    // These section modules already mask ACK against the current request's
    // generation/tag. The integration tests replacement while host is stopped.
    assign ack_gen[3*64+:128]={pg,pg};assign ack_tag[3*64+:128]={pt,pt};
    assign fault[5]=|lite_fault;assign fault[6]=|axi4_fault;
    ntrpu_drain_axil_fixture #(.DEVICE_PERIOD_PS(DEVICE_PERIOD_PS)) lite(
        .mgmt_clk(mgmt_clk),.section_request(proof_req[3]),.release_forward(release_forward),
        .release_local(release_local[0]),.host_enable(host_enable[0]),.generation(pg),.tag(pt),
        .ready_to_close(ready_close[0]),.forward_held(forward_held[0]),.local_held(local_held[0]),.drained(drained[0]),
        .host_clk_o(host_clock[0]),.cold_o(peer_cold[0]),.section_ack_o(section_ack[0]),.local_idle_o(local_idle[0]),
        .request_present_o(requests_present[0]),.reply_pending_o(replies_pending[0]),
        .source_faults_o(lite_sources),.section_fault_o(section_fault[0]));
    ntrpu_drain_axi4_fixture #(.DEVICE_PERIOD_PS(DEVICE_PERIOD_PS)) axi4(
        .mgmt_clk(mgmt_clk),.section_request(proof_req[4]),.release_forward(release_forward),
        .release_local(release_local[1]),.host_enable(host_enable[1]),.late_offer(late_offer),.generation(pg),.tag(pt),
        .ready_to_close(ready_close[1]),.forward_held(forward_held[1]),.local_held(local_held[1]),.drained(drained[1]),
        .late_accepted(late_accepted),.host_clk_o(host_clock[1]),.cold_o(peer_cold[1]),
        .section_ack_o(section_ack[1]),.local_idle_o(local_idle[1]),.request_present_o(requests_present[1]),
        .reply_pending_o(replies_pending[1]),.source_faults_o(axi4_sources),.section_fault_o(section_fault[1]));
    ntrpu_local_reply_certificate_r2 #(.FAULT_WIDTH(2)) local_lite(
        .mgmt_clk(mgmt_clk),.host_clk(host_clock[0]),.cold_rst_n(peer_cold[0]),.req_valid(proof_req[5]),.generation(pg),.tag(pt),
        .local_idle(local_idle[0]),.request_present(requests_present[0]),.reply_pending(replies_pending[0]),.source_faults(lite_sources),.section_fault_mgmt(section_fault[0]),
        .ack_valid(ack[5]),.ack_generation(ack_gen[5*64+:64]),.ack_tag(ack_tag[5*64+:64]),.busy(local_busy[0]),
        .peer_request(local_peer_req[0]),.peer_answer(local_peer_ack[0]),.peer_generation(),.peer_tag(),
        .fault_mgmt(lite_fault),.revoked_source(local_revoked[0]),.obsolete_completions(obsolete[31:0]));
    ntrpu_local_reply_certificate_r2 #(.FAULT_WIDTH(7)) local_axi4(
        .mgmt_clk(mgmt_clk),.host_clk(host_clock[1]),.cold_rst_n(peer_cold[1]),.req_valid(proof_req[6]),.generation(pg),.tag(pt),
        .local_idle(local_idle[1]),.request_present(requests_present[1]),.reply_pending(replies_pending[1]),.source_faults(axi4_sources),.section_fault_mgmt(section_fault[1]),
        .ack_valid(ack[6]),.ack_generation(ack_gen[6*64+:64]),.ack_tag(ack_tag[6*64+:64]),.busy(local_busy[1]),
        .peer_request(local_peer_req[1]),.peer_answer(local_peer_ack[1]),.peer_generation(),.peer_tag(),
        .fault_mgmt(axi4_fault),.revoked_source(local_revoked[1]),.obsolete_completions(obsolete[63:32]));
    task must(input bit ok,input string name);if(!ok)$fatal(1,"DRAIN_OWNED_ASSERT %s",name);endtask
    task mark(input string name);
        begin #0.001;
            $display("DRAIN_OWNED_EVENT name=%s ps=%0.0f external=%0d period=%0d gen=%h tag=%h pg=%h pt=%h req=%b present=%b ready=%b proof=%b ack=%b accepted=%b complete=%b revoked=%b fault=%b sections=%b idle=%b pending=%b localbusy=%b localreq=%b localack=%b localfault=%b enable=%b obsolete_l=%0d obsolete_a=%0d mode=%0d own_phase=%0d own_enable=%b own_cold=%b own_lost=%b own_invalid=%b own_cmd=%b own_data=%b own_result=%b own_ack=%b own_busy=%b own_obsolete=%0d own_done=%b own_reuse=%b",name,$realtime*1000,EXTERNAL_FIXTURE,DEVICE_PERIOD_PS,generation,tag,pg,pt,req,present,ready,proof_req,ack,accepted,complete,revoked,retained_faults,section_ack,local_idle,replies_pending,local_busy,local_peer_req,local_peer_ack,local_revoked,host_enable,obsolete[31:0],obsolete[63:32],CONTRACT_MODE,OWNER_PHASE,owner_enable,owner_cold,owner_lost,owner_invalid,owner_cmd,owner_data,owner_result,ack[0],owner_busy,owner_obsolete,owner_done,owner_reuse_tested);
            #0.001;
        end
    endtask
    task owner_held;
        if(OWNER_PHASE==0)must(owner_cmd,"owner CMD remains held");
        if(OWNER_PHASE==1)must(owner_data,"owner DATA remains held");
        if(OWNER_PHASE==2)must(owner_result,"owner result remains held");
    endtask
    initial begin
        wait(&ready_close && owner_ready);@(negedge mgmt_clk);cold=1;req=1;mark("BEGIN_CLOSE");
        wait(owner_peer_req);repeat(30)@(negedge mgmt_clk);
        must(!ack[0] && !owner_lost && !complete && proof_req[6:5]==0,"live owner blocks join");owner_held();mark("LIVE_OWNER_BLOCKED");
        @(negedge owner_clock);owner_enable=0;invalidate_owner=1;
        repeat(30)@(negedge mgmt_clk);
        must(owner_invalid && !owner_lost && !ack[0] && !complete,"no source edge to register loss");mark("OWNER_STOPPED_BEFORE_LOSS");
        tag=tag+1;repeat(5)@(negedge mgmt_clk);mark("OWNER_REPLACED_WHILE_STOPPED");
        owner_enable=1;@(posedge owner_clock);@(negedge owner_clock);owner_enable=0;
        repeat(30)@(negedge mgmt_clk);
        must(owner_lost && !ack[0] && !complete,"one owner edge cannot complete proof");mark("OWNER_LOSS_REGISTERED_STOPPED");
        owner_enable=1;wait(ack[0]);must(owner_obsolete==1,"owner obsolete completion");owner_held();mark("OWNER_CERTIFIED");
        invalidate_owner=0;try_owner_reuse=1;wait(owner_reuse_tested);owner_held();mark("OWNER_REUSE_BLOCKED");
        wait(&forward_held);must(section_ack==0 && !complete,"forwarded B/R still held");mark("FORWARD_HELD");
        release_forward=1;wait(&local_held);repeat(8)@(negedge mgmt_clk);
        must(section_ack==3 && local_idle==0 && replies_pending==3 && !complete,"sections exclude local replies");mark("LOCAL_HELD");
        if(EXTERNAL_FIXTURE)begin wait(local_busy==3 && local_peer_req==3);mark("LOCAL_PROOFS_PENDING");end
        @(posedge host_clock[1]);#0.001;host_enable[1]=0;
        @(negedge owner_clock);owner_enable=0;
        @(negedge mgmt_clk);tag=tag+1;mark("REPLACE_IDENTITY");repeat(48)@(negedge mgmt_clk);
        must(!ack[0] && !section_ack[1] && !complete,"two stopped peers cannot renew");mark("STOPPED_NEW_ID");
        owner_enable=1;wait(ack[0]);must(!section_ack[1] && !complete,"owner alone cannot certify section");mark("OWNER_ONLY_RENEWED");
        host_enable[1]=1;wait(section_ack==3);mark("SECTIONS_RENEWED");
        release_owner_cmd=1;
        if(OWNER_PHASE<2)wait(owner_data);else must(owner_result,"held owner result");
        mark("OWNER_OBLIGATION_HELD");
        release_local[0]=1;wait(drained[0]);repeat(12)@(negedge mgmt_clk);
        must(local_idle[0] && !local_idle[1] && replies_pending[1] && !complete,"AXI4 local replies still pending");mark("AXIL_LOCAL_DRAINED");
        release_owner_payload=1;wait(owner_done);must(ack[0],"owner proof survives retained data transfer");mark("OWNER_DATA_TRANSFERRED");
        release_local[1]=1;wait(&drained);
        if(EXTERNAL_FIXTURE)begin
            wait(complete);must(obsolete[31:0]>=1 && obsolete[63:32]>=1,"old local proofs retired without relabeling");
        end else begin
            repeat(32)@(negedge mgmt_clk);
            must(!complete && (proof_req & ~present)==0 && (ack & ~present)==0,"missing host/XDMA contracts block");
        end
        mark("LOCAL_DRAINED");
        if(EXTERNAL_FIXTURE)begin
            @(negedge mgmt_clk);ready[0]=0;mark("READINESS_LOST");repeat(2)@(negedge mgmt_clk);ready[0]=1;
            must(!complete && revoked,"restoring owner readiness cannot revive round");tag=tag+1;wait(complete);mark("READINESS_REPROVEN");
            late_offer=1;wait(late_accepted && local_revoked[1]);host_enable[1]=0;mark("LATE_REQUEST_SOURCE_STOPPED");
            repeat(8)@(negedge mgmt_clk);
            must(retained_faults[6] && !complete,"local late fault crosses after host stops");mark("LATE_REQUEST_REVOKED");
            req=0;repeat(2)@(negedge mgmt_clk);tag=tag+1;req=1;repeat(12)@(negedge mgmt_clk);
            must(retained_faults[6] && !complete,"late obligation forbids new certificate");mark("FAULT_RETAINED");
        end
        mark("TERMINAL");
        $display("DRAIN_OWNED_PASS mode=%0d period=%0d owner_phase=%0d actual_ownership=1 actual_sections=2 local_providers=2 native_endpoint=0 host_xdma_qualified=0 reset_driven=0",CONTRACT_MODE,DEVICE_PERIOD_PS,OWNER_PHASE);
        $finish;
    end
    initial begin #400000;$fatal(1,"DRAIN_OWNED_TIMEOUT");end
endmodule
