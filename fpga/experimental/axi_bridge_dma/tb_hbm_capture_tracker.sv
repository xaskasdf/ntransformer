`timescale 1ns/1ps
module capture_tracker_case #(parameter integer MAX_PENDING=8)(output reg done=0);
    reg clk=0,rst_n=0,p_arvalid=0,p_rready=0,h_arready=0,h_rvalid=0;
    reg [33:0] p_araddr=0;
    reg [3:0] p_arlen=0,p_arid=0;
    reg [2:0] p_arsize=5;
    reg [1:0] p_arburst=1,h_rresp=0;
    reg [255:0] h_rdata=0;
    reg [5:0] h_rid=0;
    reg h_rlast=0;
    wire p_arready,p_rvalid,h_arvalid,h_rready,pending,fault;
    wire [255:0] validated_data;
    wire [5:0] validated_rid,h_arid;
    wire [1:0] validated_resp,h_arburst;
    wire validated_last;
    wire [33:0] h_araddr;
    wire [3:0] h_arlen;
    wire [2:0] h_arsize;
    wire r_p_arready,r_p_rvalid,r_h_arvalid,r_h_rready,r_pending,r_fault;
    wire [255:0] r_data;
    wire [5:0] r_rid,r_arid;
    wire [1:0] r_resp,r_arburst;
    wire r_last;
    wire [33:0] r_araddr;
    wire [3:0] r_arlen;
    wire [2:0] r_arsize;
    ntrpu_hbm_read_capture_tracker #(.MAX_PENDING(MAX_PENDING)) dut(.*);
    ntrpu_hbm_read_tracker #(.MAX_PENDING(MAX_PENDING)) reference(
        .clk(clk),.rst_n(rst_n),.p_arvalid(p_arvalid),.p_arready(r_p_arready),
        .p_araddr(p_araddr),.p_arlen(p_arlen),.p_arid(p_arid),.p_arsize(p_arsize),.p_arburst(p_arburst),
        .p_rvalid(r_p_rvalid),.p_rready(p_rready),.validated_data(r_data),.validated_rid(r_rid),
        .validated_resp(r_resp),.validated_last(r_last),.h_arvalid(r_h_arvalid),.h_arready(h_arready),
        .h_araddr(r_araddr),.h_arlen(r_arlen),.h_arid(r_arid),.h_arsize(r_arsize),.h_arburst(r_arburst),
        .h_rvalid(h_rvalid),.h_rready(r_h_rready),.h_rdata(h_rdata),.h_rid(h_rid),
        .h_rresp(h_rresp),.h_rlast(h_rlast),.pending(r_pending),.fault(r_fault));
    always #2 clk=~clk;
    reg [264:0] expected[0:16383];
    integer head=0,tail=0,checks=0,delivered=0,accepted_ar=0,quarantines=0,stalls=0;
    integer clean_beats=0,cycles=0,cold_epochs=0;
    bit fault_test=0,took_ar,took_r,took_reserve;
    reg [31:0] rng=32'h10203040+MAX_PENDING;
    function [31:0] next_random(input [31:0] v);
        reg [31:0] x;
        begin x=v^(v<<13);x=x^(x>>17);next_random=x^(x<<5);end
    endfunction
    task check(input bit ok,input string why);
        begin checks=checks+1;if(!ok)$fatal(1,"CAPTURE_TRACKER pending=%0d %s",MAX_PENDING,why);end
    endtask
    task tick;
        begin
            @(negedge clk);#0.1;cycles=cycles+1;
            check({p_arready,p_rvalid,h_arvalid,h_rready,pending,fault}===
                {r_p_arready,r_p_rvalid,r_h_arvalid,r_h_rready,r_pending,r_fault},"reference controls");
            if(h_arvalid)check({h_araddr,h_arlen,h_arid,h_arsize,h_arburst}===
                {r_araddr,r_arlen,r_arid,r_arsize,r_arburst},"offered AR retained");
            if(p_rvalid)check({validated_rid,validated_resp,validated_last,validated_data}===
                {r_rid,r_resp,r_last,r_data},"reference validated payload");
            check({dut.q_valid,dut.response_cut.count,dut.response_cut.read_slot,dut.response_cut.write_slot}===
                {reference.q_valid,reference.response_cut.count,reference.response_cut.read_slot,reference.response_cut.write_slot},"response ownership");
            if(dut.q_valid)check({dut.q_rid,dut.q_resp,dut.q_last,dut.q_data}===
                {reference.q_rid,reference.q_resp,reference.q_last,reference.q_data},"quarantined/active head retained");
            check({dut.reserved_count,dut.accepted_count,dut.capture_remaining,dut.validate_index,dut.head_length}===
                {reference.reserved_count,reference.accepted_count,reference.capture_remaining,reference.validate_index,reference.head_length},"credits and descriptor state");
            took_ar=h_arvalid&&h_arready;took_r=h_rvalid&&h_rready;took_reserve=p_arvalid&&p_arready;
            if(rst_n)begin
                if(took_ar)accepted_ar=accepted_ar+1;
                if(h_rvalid&&!h_rready)stalls=stalls+1;
                if(!fault_test)begin
                    check(!fault,"unexpected clean fault");
                    if(p_rvalid&&p_rready)begin
                        check(head<tail,"no fabricated response");
                        check({validated_rid,validated_resp,validated_last,validated_data}===expected[head],"end-to-end scoreboard");
                        head=head+1;delivered=delivered+1;
                    end
                    if(took_r)begin
                        check(tail<16384,"scoreboard capacity");
                        expected[tail]={h_rid,h_rresp,h_rlast,h_rdata};tail=tail+1;clean_beats=clean_beats+1;
                    end
                end
            end
            @(posedge clk);#0.1;
        end
    endtask
    task cold_reset;
        begin
            rst_n=0;p_arvalid=0;h_arready=0;h_rvalid=0;p_rready=0;
            tick();tick();head=0;tail=0;fault_test=0;
            check(!fault&&!pending&&!p_rvalid,"cold reset releases ownership");
            cold_epochs=cold_epochs+1;rst_n=1;tick();
        end
    endtask
    task request(input integer length);
        integer tries;
        begin
            p_arvalid=1;p_arlen=length;p_araddr=p_araddr+512;h_arready=0;tries=0;
            do begin tick();tries=tries+1;check(tries<100,"reserve timeout");end while(!took_reserve);
            p_arvalid=0;
            repeat(3)tick(); // The offered AR must remain stable while blocked.
            h_arready=1;tick();check(took_ar,"AR acceptance");h_arready=0;
        end
    endtask
    task send_beat(input bit last,input integer bad_kind);
        integer tries,i;
        begin
            h_rvalid=0;
            rng=next_random(rng);repeat(rng[1:0])begin
                rng=next_random(rng);p_rready=rng[3];h_rdata={8{rng}};tick();
            end
            for(i=0;i<8;i=i+1)begin rng=next_random(rng);h_rdata[i*32+:32]=rng;end
            h_rlast=last;h_rid=(bad_kind==1)?6'd1:0;h_rresp=(bad_kind==2)?2'b01:2'b00;
            h_rvalid=1;tries=0;
            do begin
                rng=next_random(rng);p_rready=(tries>20)||rng[4];tick();
                tries=tries+1;check(tries<100,"response timeout");
            end while(!took_r);
            h_rvalid=0;h_rid=0;h_rresp=0;
        end
    endtask
    task drain;
        integer tries;
        begin
            tries=0;p_rready=1;h_rvalid=0;
            while(pending)begin tick();tries=tries+1;check(tries<1000,"drain timeout");end
            check(head==tail,"all accepted responses delivered");tick();
        end
    endtask
    integer round,d,b,lengths[0:7],kind;
    reg [264:0] held;
    initial begin
        repeat(2)@(posedge clk);#0.1;cold_reset();
        // Zero-latency slave offer at first AR: payload is not owned until READY.
        p_arvalid=1;p_arlen=0;tick();check(took_reserve,"zero-latency reservation");
        p_arvalid=0;h_arready=1;h_rvalid=1;h_rlast=1;h_rdata=256'hfeed;
        tick();check(took_ar&&!took_r&&!fault,"registered capture credit");
        h_arready=0;tick();check(took_r,"held zero-latency response accepted");h_rvalid=0;drain();
        for(round=0;round<32;round=round+1)begin
            for(d=0;d<MAX_PENDING;d=d+1)begin
                case((round+d)%4)0:lengths[d]=0;1:lengths[d]=3;2:lengths[d]=15;default:lengths[d]=1;endcase
                request(lengths[d]);
            end
            check(dut.reserved_count==MAX_PENDING&&!p_arready,"credit limit");
            for(d=0;d<MAX_PENDING;d=d+1)
                for(b=0;b<=lengths[d];b=b+1)send_beat(b==lengths[d],0);
            drain();
        end
        // Bad RID, EXOKAY and early RLAST: retain faulted head despite changing input.
        for(kind=1;kind<=3;kind=kind+1)begin
            cold_reset();request(3);send_beat(0,0);p_rready=1;repeat(6)tick();
            check(head==tail,"good prefix delivered before quarantine");fault_test=1;
            send_beat(kind==3,kind);h_rvalid=0;p_rready=0;repeat(6)tick();
            check(fault&&pending&&dut.q_valid,"bad response quarantined");
            held={dut.q_rid,dut.q_resp,dut.q_last,dut.q_data};
            repeat(64)begin
                rng=next_random(rng);h_rdata={8{rng}};h_rvalid=rng[0];p_rready=rng[1];tick();
                check(fault&&pending&&!h_rready&&!p_arready,"fault must retain obligations");
                check({dut.q_rid,dut.q_resp,dut.q_last,dut.q_data}===held,"fault head cannot be overwritten");
            end
            quarantines=quarantines+1;
        end
        cold_reset();fault_test=1;h_rvalid=1;h_rlast=1;tick();
        check(fault&&!h_rready,"unsolicited response fault");h_rvalid=0;repeat(8)tick();
        check(fault,"unsolicited fault sticky");cold_reset();
        check(delivered==clean_beats,"clean traffic conservation");
        check(quarantines==3&&stalls>0&&accepted_ar>32,"scenario coverage");
        $display("CAPTURE_TRACKER PASS pending=%0d checks=%0d delivered=%0d descriptors=%0d quarantines=%0d stalls=%0d cold_epochs=%0d cycles=%0d",MAX_PENDING,checks,delivered,accepted_ar,quarantines,stalls,cold_epochs,cycles);
        done=1;
    end
endmodule
module tb_hbm_capture_tracker;
    wire a,b,c,d;
    capture_tracker_case #(.MAX_PENDING(1)) c1(a);
    capture_tracker_case #(.MAX_PENDING(2)) c2(b);
    capture_tracker_case #(.MAX_PENDING(4)) c4(c);
    capture_tracker_case #(.MAX_PENDING(8)) c8(d);
    initial begin wait(a&&b&&c&&d);$display("CAPTURE_TRACKER ALL PASS");$finish;end
    initial begin #1000000;$fatal(1,"CAPTURE_TRACKER watchdog");end
endmodule
