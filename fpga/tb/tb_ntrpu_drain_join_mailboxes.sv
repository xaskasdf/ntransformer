`timescale 1ns / 1ps
// Actual AMD XPM handshakes; certificate providers are explicit test fixtures.
module tb_ntrpu_drain_join_mailboxes;
    reg clk=0;always #5 clk=~clk;
    reg cold=0,req=0;
    reg [63:0] gen=2,tag=1;
    reg [6:0] present=7'b1111011,ready=127,fault=0,run_peer=7'b0111111;
    wire [6:0] ack,request,accepted,armed,retained,busy,peer_req,peer_ack;
    wire [447:0] ag,at;
    wire [223:0] obsolete;
    wire [63:0] pg,pt,cg,ct;
    wire complete,revoked,replay;
    ntrpu_drain_certificate_join dut(.mgmt_clk(clk),.cold_rst_n(cold),.req_valid(req),
        .generation(gen),.tag(tag),.contract_present(present),.peer_ready(ready),
        .proof_fault(fault),.ack_valid(ack),.ack_generation(ag),.ack_tag(at),
        .proof_request(request),.proof_generation(pg),.proof_tag(pt),
        .complete_ack(complete),.complete_generation(cg),.complete_tag(ct),
        .accepted_mask(accepted),.armed_mask(armed),.retained_faults(retained),
        .revoked(revoked),.identity_fault(replay));
    for(genvar i=0;i<7;i=i+1)begin:providers
        reg pclk=0,p_ack=0;
        wire [2:0] kind;
        wire [63:0] peer_gen,peer_tag;
        always #(i+3)if(run_peer[i])pclk=~pclk;
        assign peer_ack[i]=p_ack;
        always @(posedge pclk)begin
            if(!peer_req[i])p_ack<=0;
            else p_ack<=1;
        end
        ntrpu_tagged_reset_mailbox mailbox(.mgmt_clk(clk),.peer_clk(pclk),
            .req_valid(request[i]),.req_kind(3'd1),.req_generation(pg),.req_tag(pt),
            .ack_valid(ack[i]),.ack_kind(),.ack_generation(ag[i*64+:64]),.ack_tag(at[i*64+:64]),
            .busy(busy[i]),.obsolete_completions(obsolete[i*32+:32]),
            .peer_req_valid(peer_req[i]),.peer_req_kind(kind),.peer_req_generation(peer_gen),.peer_req_tag(peer_tag),
            .peer_ack_valid(p_ack),.peer_ack_kind(kind),.peer_ack_generation(peer_gen),.peer_ack_tag(peer_tag),
            .rejected_peer_ack_cycles());
    end
    task tick;begin @(posedge clk);#1;end endtask
    task observe(input integer label,input bit expected);
        begin #1;
            $display("JOIN_ASYNC_CHECK label=%0d t=%0t expected=%0d complete=%b req=%b gen=%h tag=%h pg=%h pt=%h present=%b ready=%b run=%b request=%b ack=%b accepted=%b armed=%b busy=%b revoked=%b faults=%b obsolete4=%0d",label,$time,expected,complete,req,gen,tag,pg,pt,present,ready,run_peer,request,ack,accepted,armed,busy,revoked,retained,obsolete[4*32+:32]);
            if(complete!==expected)$fatal(1,"JOIN_ASYNC_FAILED label=%0d",label);
            if(complete && (cg!==gen || ct!==tag))$fatal(1,"JOIN_ASYNC_IDENTITY");
            #1;
        end
    endtask
    initial begin
        repeat(4)tick();@(negedge clk);cold=1;req=1;
        wait((ack & 7'b0011011)==7'b0011011);repeat(16)tick();observe(1,0);
        if(request[6:5]!==0 || request[2]!==0)$fatal(1,"JOIN_ASYNC_MISSING_CONTRACT");
        @(negedge clk);present[2]=1;
        wait(ack[5:0]==63);repeat(16)tick();observe(2,0);
        if(!busy[6] || ack[6])$fatal(1,"JOIN_ASYNC_STOPPED_CLOCK");
        @(negedge clk);run_peer[6]=1;wait(complete);tick();observe(3,1);
        @(negedge clk);ready[3]=0;observe(4,0);tick();ready[3]=1;tick();observe(5,0);
        if(!revoked)$fatal(1,"JOIN_ASYNC_NOT_REVOKED");
        @(negedge clk);tag=2;wait(complete);tick();observe(6,1);
        wait(busy==0);
        @(negedge clk);run_peer[4]=0;tag=3;
        wait(busy[4]);repeat(12)tick();observe(7,0);
        @(negedge clk);req=0;tick();observe(8,0);
        @(negedge clk);gen=3;tag=1;req=1;repeat(12)tick();observe(9,0);
        @(negedge clk);run_peer[4]=1;wait(complete);tick();observe(10,1);
        if(obsolete[4*32+:32]<1)$fatal(1,"JOIN_ASYNC_OLD_COMPLETION_NOT_OBSERVED");
        @(negedge clk);run_peer[4]=0;fault[4]=1;observe(11,0);tick();
        fault=0;req=0;tick();@(negedge clk);tag=2;req=1;repeat(12)tick();observe(12,0);
        if(!retained[4])$fatal(1,"JOIN_ASYNC_FAULT_NOT_RETAINED");
        $display("JOIN_ASYNC_SUITE_PASS observations=12 providers=7 real_xpm=1 native_endpoint=0 external_providers_qualified=0 reset_driven=0");
        $finish;
    end
    initial begin #50000;$fatal(1,"JOIN_ASYNC_TIMEOUT");end
endmodule
