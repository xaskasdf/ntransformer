`timescale 1ns / 1ps
module tb_ntrpu_axi4_response_guard;
    reg clk=0;always #5 clk=~clk;
    integer case_id=-1,cycle=0;reg running=0,auto_ready=0;
    reg cold_rst_n=0;
    reg awvalid=0;
    reg awready=0;
    reg wvalid=0;
    reg wready=0;
    reg wlast=0;
    reg arvalid=0;
    reg arready=0;
    reg [3:0] awid=0;
    reg [3:0] arid=0;
    reg [7:0] awlen=0;
    reg [7:0] arlen=0;
    reg c_bvalid=0;
    reg c_rvalid=0;
    reg c_rlast=0;
    reg [3:0] c_bid=0;
    reg [3:0] c_rid=0;
    reg [1:0] c_bresp=0;
    reg [1:0] c_rresp=0;
    reg [511:0] c_rdata=0;
    reg h_bready=0;
    reg h_rready=0;
    wire c_bready;
    wire c_rready;
    wire h_bvalid;
    wire h_rvalid;
    wire h_rlast;
    wire [3:0] h_bid;
    wire [3:0] h_rid;
    wire [1:0] h_bresp;
    wire [1:0] h_rresp;
    wire [511:0] h_rdata;
    wire [3:0] fault_sources;
    wire aw_capacity;
    wire w_capacity;
    wire ar_capacity;
    wire local_idle;
    wire [5:0] aw_pending;
    wire [5:0] w_pending;
    wire [7:0] ar_pending;
    wire [8:0] w_partial;
    ntrpu_axi4_response_guard dut(.clk(clk),.cold_rst_n(cold_rst_n),.awvalid(awvalid),.awready(awready),.wvalid(wvalid),.wready(wready),.wlast(wlast),.arvalid(arvalid),.arready(arready),.awid(awid),.arid(arid),.awlen(awlen),.arlen(arlen),.c_bvalid(c_bvalid),.c_rvalid(c_rvalid),.c_rlast(c_rlast),.c_bid(c_bid),.c_rid(c_rid),.c_bresp(c_bresp),.c_rresp(c_rresp),.c_rdata(c_rdata),.h_bready(h_bready),.h_rready(h_rready),.c_bready(c_bready),.c_rready(c_rready),.h_bvalid(h_bvalid),.h_rvalid(h_rvalid),.h_rlast(h_rlast),.h_bid(h_bid),.h_rid(h_rid),.h_bresp(h_bresp),.h_rresp(h_rresp),.h_rdata(h_rdata),.fault_sources(fault_sources),.aw_capacity(aw_capacity),.w_capacity(w_capacity),.ar_capacity(ar_capacity),.local_idle(local_idle),.aw_pending(aw_pending),.w_pending(w_pending),.ar_pending(ar_pending),.w_partial(w_partial));
    always @(posedge clk)if(running)begin
        $display("AXI4_GUARD_SAMPLE case=%0d cycle=%0d time=%0t cold_rst_n=%h awvalid=%h awready=%h wvalid=%h wready=%h wlast=%h arvalid=%h arready=%h awid=%h arid=%h awlen=%h arlen=%h c_bvalid=%h c_rvalid=%h c_rlast=%h c_bid=%h c_rid=%h c_bresp=%h c_rresp=%h c_rdata=%h h_bready=%h h_rready=%h c_bready=%h c_rready=%h h_bvalid=%h h_rvalid=%h h_rlast=%h h_bid=%h h_rid=%h h_bresp=%h h_rresp=%h h_rdata=%h fault_sources=%h aw_capacity=%h w_capacity=%h ar_capacity=%h local_idle=%h aw_pending=%h w_pending=%h ar_pending=%h w_partial=%h",case_id,cycle,$time,cold_rst_n,awvalid,awready,wvalid,wready,wlast,arvalid,arready,awid,arid,awlen,arlen,c_bvalid,c_rvalid,c_rlast,c_bid,c_rid,c_bresp,c_rresp,c_rdata,h_bready,h_rready,c_bready,c_rready,h_bvalid,h_rvalid,h_rlast,h_bid,h_rid,h_bresp,h_rresp,h_rdata,fault_sources,aw_capacity,w_capacity,ar_capacity,local_idle,aw_pending,w_pending,ar_pending,w_partial);
        cycle=cycle+1;
    end

    always @(negedge clk)if(auto_ready)begin h_bready=(cycle%3!=0);h_rready=(cycle%5!=0);end
    function integer length(input integer n);
        case(n%8)0:length=1;1:length=2;2:length=4;3:length=16;4:length=64;5:length=256;6:length=3;default:length=1;endcase
    endfunction
    function [511:0] pattern(input integer seed);
        for(integer j=0;j<16;j=j+1)pattern[j*32+:32]=32'h9e3779b9*(seed+j+1)^32'hd00dbeef;
    endfunction
    task step;begin @(posedge clk);#1;end endtask
    task idle(input integer n);begin repeat(n)step;end endtask
    task check(input bit good,input [511:0] message);
        if(!good)$fatal(1,"AXI4_GUARD_ASSERT case=%0d cycle=%0d %0s",case_id,cycle,message);
    endtask
    task start_case(input integer n);
        begin
            @(negedge clk);running=0;auto_ready=0;cold_rst_n=0;
            awvalid=0;wvalid=0;arvalid=0;c_bvalid=0;c_rvalid=0;
            awready=1;wready=1;arready=1;h_bready=0;h_rready=0;
            awid=0;arid=0;awlen=0;arlen=0;wlast=0;
            c_bid=0;c_rid=0;c_bresp=0;c_rresp=0;c_rdata=0;c_rlast=0;
            idle(3);@(negedge clk);cold_rst_n=1;case_id=n;cycle=0;running=1;
            $display("AXI4_GUARD_START case=%0d",n);idle(2);
        end
    endtask
    task aw(input integer id,input integer beats);
        begin @(negedge clk);awvalid=1;awid=id;awlen=beats-1;step;@(negedge clk);awvalid=0;end
    endtask
    task w(input bit last);
        begin @(negedge clk);wvalid=1;wlast=last;step;@(negedge clk);wvalid=0;end
    endtask
    task ar(input integer id,input integer beats);
        begin @(negedge clk);arvalid=1;arid=id;arlen=beats-1;step;@(negedge clk);arvalid=0;end
    endtask
    task send_b(input integer id,input integer resp);
        integer count;bit accepted;
        begin
            @(negedge clk);c_bid=id;c_bresp=resp;c_bvalid=1;count=0;accepted=0;
            while(!accepted)begin @(posedge clk);accepted=c_bready;#1;count=count+1;check(count<40,"B capture timeout");end
            @(negedge clk);c_bvalid=0;
        end
    endtask
    task send_r(input integer id,input bit last,input integer seed);
        integer count;bit accepted;
        begin
            @(negedge clk);c_rid=id;c_rlast=last;c_rdata=pattern(seed);c_rresp=seed%4;c_rvalid=1;count=0;accepted=0;
            while(!accepted)begin @(posedge clk);accepted=c_rready;#1;count=count+1;check(count<40,"R capture timeout");end
            @(negedge clk);c_rvalid=0;
        end
    endtask
    task bad_b(input integer id);
        begin @(negedge clk);c_bvalid=1;c_bid=id;c_bresp=2;idle(3);check(!c_bready && fault_sources[0],"B must quarantine");end
    endtask
    task bad_r(input integer id,input bit last);
        begin @(negedge clk);c_rvalid=1;c_rid=id;c_rlast=last;c_rdata=pattern(999);c_rresp=2;
            idle(3);check(!c_rready && fault_sources[1],"R must quarantine");end
    endtask
    task finish_case(input integer expected_fault);
        begin
            idle(3);check(fault_sources==expected_fault,"exact retained fault sources");
            $display("AXI4_GUARD_CASE_PASS case=%0d faults=%0d cycles=%0d",case_id,fault_sources,cycle);
            @(negedge clk);running=0;
        end
    endtask
    task drained;
        begin @(negedge clk);auto_ready=0;h_bready=1;h_rready=1;idle(5);check(local_idle && fault_sources==0,"healthy inventory drain");end
    endtask
    integer i,j,k,beat;
    initial begin
        // Full depth, repeated wrap, non-ID-order writes, mixed burst lengths.
        start_case(0);auto_ready=1;
        for(k=0;k<3;k=k+1)begin
            for(i=0;i<32;i=i+1)aw(i%16,length(i));
            for(i=0;i<32;i=i+1)for(j=0;j<length(i);j=j+1)w(j==length(i)-1);
            check(aw_pending==32 && w_pending==32 && !aw_capacity && !w_capacity,"32 write slots full");
            for(i=15;i>=0;i=i-1)begin send_b(i,(i+k)%4);send_b(i,(i+k+1)%4);end
            idle(5);
        end
        drained;finish_case(0);
        // Thirty-one complete W before AW, plus partial final burst.
        start_case(1);auto_ready=1;
        for(i=0;i<31;i=i+1)for(j=0;j<length(i);j=j+1)w(j==length(i)-1);
        for(j=0;j<3;j=j+1)w(0);
        check(w_partial==3 && w_pending==31 && w_capacity,"partial W reservation");
        for(i=0;i<32;i=i+1)aw(i%16,i==31?4:length(i));w(1);
        for(i=15;i>=0;i=i-1)begin send_b(i,0);send_b(i,2);end
        drained;finish_case(0);
        // All128 read descriptors, interleaved16 IDs and full512-bit payloads.
        start_case(2);auto_ready=1;
        for(i=0;i<16;i=i+1)for(j=0;j<8;j=j+1)ar(i,length(j));
        check(ar_pending==128 && !ar_capacity,"all128 read obligations full");
        for(j=0;j<8;j=j+1)for(beat=0;beat<length(j);beat=beat+1)
            for(i=0;i<16;i=i+1)send_r(i,beat==length(j)-1,4096*j+256*beat+i);
        drained;finish_case(0);
        // All five channels active, continuous output-slot replacement.
        start_case(3);h_bready=1;h_rready=1;
        for(i=0;i<98;i=i+1)begin
            @(negedge clk);awvalid=(i<96);wvalid=(i<96);arvalid=(i<96);wlast=1;
            awid=i%16;arid=i%16;awlen=0;arlen=0;
            c_bvalid=(i>0 && i<=96);c_rvalid=(i>0 && i<=96);
            c_bid=(i-1)%16;c_rid=(i-1)%16;c_bresp=i%4;c_rresp=i%4;c_rlast=1;c_rdata=pattern(i);
            step;check(fault_sources==0,"duplex pipeline must stay healthy");
        end
        @(negedge clk);awvalid=0;wvalid=0;arvalid=0;c_bvalid=0;c_rvalid=0;
        drained;finish_case(0);
        start_case(4);h_bready=1;bad_b(3);aw(3,1);w(1);idle(5);
        check(!c_bready && !h_bvalid,"late requests cannot reauthorize orphan B");finish_case(1);
        start_case(5);aw(1,1);w(1);aw(2,1);w(1);send_b(1,0);bad_b(3);
        check(h_bvalid && h_bid==1,"healthy held B survives wrong ID");
        @(negedge clk);h_bready=1;idle(4);check(!h_bvalid && aw_pending==1,"only healthy held B retires");finish_case(1);
        start_case(6);aw(5,4);w(0);w(0);ar(2,1);send_r(2,1,2);bad_b(5);
        check(h_rvalid && h_rid==2,"R retained during premature B");
        @(negedge clk);h_rready=1;w(0);w(1);idle(4);
        check(ar_pending==0 && !c_bready,"R drains, B remains quarantined after WLAST");finish_case(1);
        start_case(7);aw(1,1);w(1);send_b(1,0);bad_r(3,1);ar(3,1);
        check(h_bvalid && !c_rready,"B retained, late AR cannot authorize orphan R");
        @(negedge clk);h_bready=1;idle(4);check(aw_pending==0 && !h_rvalid,"healthy B drains");finish_case(2);
        start_case(8);ar(1,2);send_r(1,0,1);bad_r(3,1);
        check(h_rvalid && h_rid==1 && !h_rlast && h_rdata==pattern(1),"healthy R survives wrong RID");
        @(negedge clk);h_rready=1;idle(4);check(!h_rvalid && ar_pending==1,"partial healthy R retained obligation");finish_case(2);
        start_case(9);ar(1,4);bad_r(1,1);aw(2,1);w(1);h_bready=1;send_b(2,0);idle(4);
        check(!h_rvalid && aw_pending==0,"early RLAST blocked, healthy B completes");finish_case(2);
        start_case(10);ar(1,1);bad_r(1,0);check(!h_rvalid,"missing RLAST blocked");finish_case(2);
        start_case(11);aw(0,1);w(1);send_b(0,0);aw(1,2);w(1);idle(4);
        check(fault_sources==4 && h_bvalid,"W length mismatch preserves held B");
        @(negedge clk);h_bready=1;idle(4);check(!h_bvalid,"held B retires despite corrupt W inventory");finish_case(4);
        start_case(12);ar(0,1);send_r(0,1,0);
        for(i=0;i<9;i=i+1)ar(1,1);
        check(fault_sources==8 && h_rvalid,"read overflow preserves held R");
        @(negedge clk);h_rready=1;aw(2,1);w(1);h_bready=1;send_b(2,0);idle(4);
        check(!h_rvalid && aw_pending==0,"healthy held R and independent B retire");finish_case(8);
        start_case(13);for(i=0;i<33;i=i+1)aw(i%16,1);
        check(aw_pending==32 && fault_sources==4,"AW overflow freezes without wrap");finish_case(4);
        start_case(14);for(i=0;i<256;i=i+1)w(0);
        check(w_partial==255 && fault_sources==4,"missing WLAST256 freezes partial");finish_case(4);
        // Stalled input corruption cannot alter a protected held response.
        start_case(15);aw(0,1);w(1);aw(1,1);w(1);send_b(0,0);
        @(negedge clk);c_bvalid=1;c_bid=1;c_bresp=0;idle(2);
        @(negedge clk);c_bresp=1;idle(3);check(fault_sources==1 && h_bid==0 && h_bresp==0,"B input changed");
        @(negedge clk);h_bready=1;idle(4);check(!h_bvalid,"corrupt second B never offered");finish_case(1);
        start_case(16);ar(0,1);ar(1,1);send_r(0,1,0);
        @(negedge clk);c_rvalid=1;c_rid=1;c_rlast=1;c_rdata=pattern(1);c_rresp=0;idle(2);
        @(negedge clk);c_rdata[511]=~c_rdata[511];idle(3);
        check(fault_sources==2 && h_rdata==pattern(0),"R input high bit changed");
        @(negedge clk);h_rready=1;idle(4);check(!h_rvalid,"corrupt second R never offered");finish_case(2);
        start_case(17);aw(0,1);w(1);aw(1,1);w(1);send_b(0,0);
        @(negedge clk);c_bvalid=1;c_bid=1;c_bresp=0;idle(2);
        @(negedge clk);c_bvalid=0;idle(3);check(fault_sources==1 && h_bvalid,"B input withdrawal");finish_case(1);
        start_case(18);ar(0,1);ar(1,1);send_r(0,1,0);
        @(negedge clk);c_rvalid=1;c_rid=1;c_rlast=1;c_rdata=pattern(1);idle(2);
        @(negedge clk);c_rvalid=0;idle(3);check(fault_sources==2 && h_rvalid,"R input withdrawal");finish_case(2);
        $display("AXI4_GUARD_SUITE_PASS cases=19 write_depth=32 read_total=128 native_endpoint=0");$finish;
    end
    initial begin #2000000;$fatal(1,"AXI4_GUARD_TIMEOUT case=%0d",case_id);end
endmodule
