`timescale 1ns / 1ps
module tb_ntrpu_drain_certificate_join;
    reg clk=0;always #5 clk=~clk;
    reg cold=0,req=0;
    reg [63:0] gen=1,tag=1;
    reg [6:0] present=127,ready=127,fault=0,ack=0;
    reg [447:0] ag=0,at=0;
    wire [6:0] request,accepted,armed,retained;
    wire [63:0] pg,pt,cg,ct;
    wire complete,revoked,replay;
    integer checks=0,case_id=0;
    ntrpu_drain_certificate_join dut(.mgmt_clk(clk),.cold_rst_n(cold),.req_valid(req),
        .generation(gen),.tag(tag),.contract_present(present),.peer_ready(ready),
        .proof_fault(fault),.ack_valid(ack),.ack_generation(ag),.ack_tag(at),
        .proof_request(request),.proof_generation(pg),.proof_tag(pt),
        .complete_ack(complete),.complete_generation(cg),.complete_tag(ct),
        .accepted_mask(accepted),.armed_mask(armed),.retained_faults(retained),
        .revoked(revoked),.identity_fault(replay));
    task tick;begin @(posedge clk);#1;end endtask
    task ids;begin for(integer i=0;i<7;i=i+1)begin ag[i*64+:64]=gen;at[i*64+:64]=tag;end end endtask
    task check(input integer label,input bit expected);
        begin #1;checks=checks+1;
            $display("JOIN_CHECK case=%0d label=%0d t=%0t expected=%0d complete=%b request=%b accepted=%b armed=%b revoked=%b replay=%b faults=%b gen=%h tag=%h ack=%b present=%b ready=%b",case_id,label,$time,expected,complete,request,accepted,armed,revoked,replay,retained,gen,tag,ack,present,ready);
            if(complete!==expected)$fatal(1,"JOIN_CHECK_FAILED case=%0d label=%0d",case_id,label);
            if(complete && (cg!==gen || ct!==tag))$fatal(1,"JOIN_IDENTITY_FAILED");
        end
    endtask
    task init_case(input integer number);
        begin @(negedge clk);case_id=number;cold=0;req=0;gen=1;tag=number+1;
            present=127;ready=127;fault=0;ack=0;ids();repeat(3)tick();
            @(negedge clk);cold=1;req=1;tick();tick();check(0,0);
        end
    endtask
    task certify;
        begin @(negedge clk);tag=tag+1;ack=0;ids();tick();tick();@(negedge clk);ack[4:0]=31;
            tick();tick();check(1,0);@(negedge clk);ack[6:5]=3;tick();check(2,1);
        end
    endtask
    initial begin
        // Missing contract must dominate even forged matching ACK levels.
        for(integer slot=0;slot<7;slot=slot+1)begin
            init_case(10+slot);present[slot]=0;ack=127;repeat(3)tick();check(10,0);
            present[slot]=1;repeat(3)tick();check(11,0);certify();
        end
        // A stale level cannot acquire a new identity without a fresh low.
        init_case(20);certify();@(negedge clk);tag=tag+1;ids();check(20,0);
        repeat(3)tick();check(21,0);certify();
        // Each bad identity is independently denied after arming.
        for(integer slot=0;slot<7;slot=slot+1)begin
            init_case(30+slot);at[slot*64+:64]=tag+1;
            ack[4:0]=31;tick();tick();ack[6:5]=3;check(30,0);tick();
            ag[slot*64+:64]=gen+1;at[slot*64+:64]=tag;check(31,0);tick();
            if(slot<5)ack[6:5]=0;
            ids();tick();tick();ack[6:5]=3;check(32,1);tick();
        end
        // Readiness and certificate revocation forbid resurrection this round.
        for(integer slot=0;slot<7;slot=slot+1)begin
            init_case(40+slot);certify();ready[slot]=0;check(40,0);tick();
            ready[slot]=1;check(41,0);if(!revoked)$fatal(1,"JOIN_REVOKE_MISSING");
            @(negedge clk);tag=tag+1;ack=0;ids();tick();certify();
            ack[slot]=0;check(42,0);tick();ack[slot]=1;check(43,0);
        end
        // Faults survive withdrawal and identity replacement.
        for(integer slot=0;slot<7;slot=slot+1)begin
            init_case(50+slot);certify();fault[slot]=1;check(50,0);tick();
            fault=0;req=0;tick();tag=tag+1;ids();req=1;tick();check(51,0);
            if(!retained[slot])$fatal(1,"JOIN_FAULT_NOT_RETAINED");
        end
        init_case(60);certify();req=0;check(60,0);tick();req=1;tick();check(61,0);
        if(!replay)$fatal(1,"JOIN_REPLAY_ACCEPTED");
        init_case(61);certify();tag=tag-1;tick();check(62,0);
        if(!replay)$fatal(1,"JOIN_BACKWARD_ACCEPTED");
        // Unknown public inputs never produce a high certificate.
        for(integer slot=0;slot<7;slot=slot+1)begin
            init_case(70+slot);certify();ack[slot]=1'bx;check(70,0);ack[slot]=1;
            ag[slot*64]=1'bx;check(71,0);ids();at[slot*64]=1'bx;check(72,0);ids();
            present[slot]=1'bx;check(73,0);present[slot]=1;ready[slot]=1'bx;check(74,0);ready[slot]=1;
        end
        // Local reply proof is separate and cannot start before all upstream proofs.
        init_case(80);ack=0;tick();ack[3:0]=15;tick();check(80,0);
        if(request[6:5]!==0)$fatal(1,"JOIN_LOCAL_REQUEST_EARLY");
        ack[4]=1;tick();tick();ack[5]=1;tick();check(81,0);
        repeat(16)tick();check(82,0);ack[6]=1;tick();check(83,1);
        $display("JOIN_SUITE_PASS checks=%0d cases=39 native_endpoint=0 reset_driven=0",checks);
        $finish;
    end
    initial begin #100000;$fatal(1,"JOIN_TIMEOUT");end
endmodule
