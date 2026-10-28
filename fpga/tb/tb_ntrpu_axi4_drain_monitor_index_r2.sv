`timescale 1ns / 1ps
module tb_ntrpu_axi4_drain_monitor_index_r2;
    reg clk=0;always #5 clk=~clk;
    reg cold_rst_n=0, active=0;
    reg awvalid=0,awready=1,wvalid=0,wready=1,bvalid=0,bready=1;
    reg arvalid=0,arready=1,rvalid=0,rready=1;
    reg [63:0] awaddr=0,araddr=0;
    reg [3:0] awid=0,bid=0,arid=0,rid=0;
    reg [7:0] awlen=0,arlen=0;
    reg [2:0] awsize=6,arsize=6,awprot=0,arprot=0;
    reg [1:0] awburst=1,arburst=1,bresp=0,rresp=0;
    reg [3:0] awcache=3,arcache=3;
    reg awlock=0,arlock=0,wlast=0,rlast=0;
    reg [511:0] wdata=0,rdata=0;
    reg [63:0] wstrb=0;
    reg req_valid=0,source_fenced=0;
    reg [63:0] req_tag=0;
    wire ack_valid,protocol_fault;
    wire [63:0] ack_tag;
    wire [5:0] fault_sources;
    wire [3:0] aw_pending,w_pending;
    wire [7:0] ar_pending;
    wire [8:0] w_partial;
    ntrpu_axi4_drain_monitor_r2 dut(.*);
    integer test_case=-1,cycle=0;
    integer lens[0:5]='{1,2,4,16,64,256};
    function [511:0] pattern(input [31:0] seed);
        for(integer j=0;j<16;j=j+1) pattern[j*32+:32]=seed^(32'h9e3779b9*j);
    endfunction
    function [63:0] mask(input integer beat,input integer size);
        mask=(size==4)?(64'h000000000000a5f3<<((beat%4)*16)):
             (64'hf0ff55aa12345678^(64'h0101010101010101*(beat%16)));
    endfunction
    task tick;
        @(posedge clk);#2;@(negedge clk);
    endtask
    task must(input bit ok,input string label);
        if(!ok) $fatal(1,"AXI4_ASSERT case=%0d %s",test_case,label);
    endtask
    task start_case(input integer n);
        active=0;cold_rst_n=0;
        awvalid=0;wvalid=0;bvalid=0;arvalid=0;rvalid=0;
        awready=1;wready=1;bready=1;arready=1;rready=1;
        req_valid=0;source_fenced=0;req_tag=64'h100000000+n;
        repeat(3) tick();test_case=n;cycle=0;cold_rst_n=1;active=1;
        $display("AXI4_CASE case=%0d write_depth=8 read_depth_per_id=8 ids=16 data_bits=512",n);
    endtask
    task address_w(input integer id,input integer beats,input integer size,input [63:0] addr);
        awid=id;awlen=beats-1;awsize=size;awaddr=addr;awvalid=1;tick();awvalid=0;
    endtask
    task address_r(input integer id,input integer beats,input integer size,input [63:0] addr);
        arid=id;arlen=beats-1;arsize=size;araddr=addr;arvalid=1;tick();arvalid=0;
    endtask
    task write_beat(input integer beat,input integer size,input [31:0] seed,input bit last);
        wdata=pattern(seed^(beat<<8));wstrb=mask(beat,size);wlast=last;wvalid=1;
        tick();wvalid=0;
    endtask
    task write_burst(input integer beats,input integer size,input [31:0] seed);
        for(integer j=0;j<beats;j=j+1) write_beat(j,size,seed,j==beats-1);
    endtask
    task response_b(input integer id,input integer resp,input integer stall);
        bid=id;bresp=resp;bvalid=1;bready=0;
        repeat(stall) tick();bready=1;tick();bvalid=0;
    endtask
    task read_beat(input integer id,input integer beat,input [31:0] seed,input bit last,input integer stall);
        rid=id;rresp=beat%4;rdata=pattern(seed^(beat<<8)^id);rlast=last;rvalid=1;rready=0;
        repeat(stall) tick();rready=1;tick();rvalid=0;
    endtask
    task drain;
        must(!protocol_fault,"healthy before drain");
        req_valid=1;source_fenced=1;tick();
        must(!ack_valid,"capture is not two quiet edges");tick();
        must(!ack_valid,"one quiet edge insufficient");tick();
        must(ack_valid && ack_tag==req_tag,"fresh drain proof");
        req_tag=req_tag+64'h100000001;#1;must(!ack_valid,"tag immediately revokes old ACK");
        tick();must(!ack_valid,"new tag capture");tick();must(!ack_valid,"new tag one edge");tick();
        must(ack_valid && ack_tag==req_tag,"new tag independent proof");
        source_fenced=0;#1;must(!ack_valid,"fence revokes immediately");tick();
        source_fenced=1;tick();must(!ack_valid,"fence one edge");tick();must(ack_valid,"fence two edges");
        req_valid=0;#1;must(!ack_valid,"withdraw request");tick();
        $display("AXI4_CASE_PASS case=%0d fault=0",test_case);
    endtask
    task fault_done(input integer expected);
        repeat(3) tick();
        must(fault_sources==expected,"specific sticky fault");
        req_valid=1;source_fenced=1;
        repeat(4) begin tick();must(!ack_valid,"corrupt ledger cannot certify");end
        req_tag=req_tag+64'h100000001;repeat(4) tick();
        must(!ack_valid && fault_sources==expected,"fresh tag does not erase fault");
        $display("AXI4_CASE_PASS case=%0d fault=%0d",test_case,fault_sources);
    endtask
    always @(posedge clk) if(active && cold_rst_n) begin
        cycle=cycle+1;
        #1;
        $display("AXI4_SAMPLE case=%0d cycle=%0d awv=%0d awr=%0d awid=%0d awlen=%0d awsize=%0d awaddr=%h wv=%0d wr=%0d wl=%0d wd=%h ws=%h bv=%0d br=%0d bid=%0d bp=%0d arv=%0d arr=%0d arid=%0d arlen=%0d arsize=%0d araddr=%h rv=%0d rr=%0d rid=%0d rl=%0d rd=%h rp=%0d req=%0d fence=%0d tag=%h ack=%0d acktag=%h faults=%0d ap=%0d wp=%0d rpnd=%0d partial=%0d",
            test_case,cycle,awvalid,awready,awid,awlen,awsize,awaddr,wvalid,wready,wlast,wdata,wstrb,bvalid,bready,bid,bresp,
            arvalid,arready,arid,arlen,arsize,araddr,rvalid,rready,rid,rlast,rdata,rresp,req_valid,source_fenced,req_tag,ack_valid,ack_tag,
            fault_sources,aw_pending,w_pending,ar_pending,w_partial);
    end
    initial begin
        #10;
        start_case(0);
        for(integer n=0;n<6;n=n+1) begin
            if(n%2==0) begin
                address_w(n+8,lens[n],(n==5)?4:6,64'h100000000+n*4096);
                write_burst(lens[n],(n==5)?4:6,n*4096);
            end else begin
                write_burst(lens[n],(n==5)?4:6,n*4096);
                address_w(n+8,lens[n],(n==5)?4:6,64'h100000000+n*4096);
            end
            // No idle edge before B: AW/W pairing must not add a false orphan.
            response_b(n+8,n%4,3);
            address_r(n+8,lens[n],(n==5)?4:6,64'h100000000+n*4096);
            for(integer j=0;j<lens[n];j=j+1) read_beat(n+8,j,n*4096,j==lens[n]-1,(j==0)?3:0);
        end
        drain();
        start_case(1);
        for(integer n=0;n<8;n=n+1) address_w(n,4,6,64'h200000000+n*4096);
        for(integer n=0;n<8;n=n+1) write_burst(4,6,n*4096);
        for(integer n=7;n>=0;n=n-1) response_b(n,n%4,n%3);
        for(integer n=0;n<8;n=n+1) address_r(n,4,6,64'h200000000+n*4096);
        for(integer j=0;j<4;j=j+1)
            for(integer n=7;n>=0;n=n-1) read_beat(n,j,n*4096,j==3,n%2);
        drain();
        start_case(2);
        for(integer n=0;n<8;n=n+1) address_r(15,1,6,64'h300000000+n*4096);
        for(integer n=0;n<24;n=n+1) begin
            arvalid=1;arid=15;arlen=0;arsize=6;araddr=64'h300000000+(n+8)*4096;
            read_beat(15,0,n*4096,1,0);arvalid=0;
        end
        for(integer n=24;n<32;n=n+1) read_beat(15,0,n*4096,1,0);
        drain();
        start_case(3);
        for(integer n=0;n<65;n=n+1) begin
            awvalid=n<64;wvalid=n<64;arvalid=n<64;
            awid=n%16;arid=n%16;awlen=0;arlen=0;awsize=6;arsize=6;
            awaddr=64'h400000000+n*4096;araddr=awaddr;
            wdata=pattern(n*4096);wstrb=mask(0,6);wlast=1;
            bvalid=n>0;rvalid=n>0;bid=(n-1)%16;rid=bid;bresp=(n-1)%4;
            rresp=0;rlast=1;rdata=pattern((n-1)*4096^rid);
            tick();
        end
        awvalid=0;wvalid=0;arvalid=0;bvalid=0;rvalid=0;drain();
        start_case(4);
        write_beat(0,6,0,0);req_valid=1;source_fenced=1;
        repeat(5) begin tick();must(!ack_valid,"partial W blocks proof");end
        address_w(14,2,6,64'h500000000);write_beat(1,6,0,1);
        repeat(3) begin tick();must(!ack_valid,"pending B blocks proof");end
        response_b(14,2,4);req_valid=0;tick();drain();
        start_case(5);bvalid=1;bready=0;bid=2;tick();fault_done(2);
        start_case(6);rvalid=1;rready=0;rid=3;rlast=1;tick();fault_done(4);
        start_case(7);awvalid=1;awid=2;awlen=0;wvalid=1;wlast=1;bvalid=1;bid=2;
        tick();awvalid=0;wvalid=0;bvalid=0;fault_done(2);
        start_case(8);arvalid=1;arid=3;arlen=0;rvalid=1;rid=3;rlast=1;
        tick();arvalid=0;rvalid=0;fault_done(4);
        start_case(9);address_w(1,4,6,64'h600000000);write_burst(2,6,0);fault_done(8);
        start_case(10);address_w(1,2,6,64'h600000000);write_beat(0,6,0,0);write_beat(1,6,0,0);fault_done(8);
        start_case(11);address_r(2,4,6,64'h600000000);read_beat(2,0,0,1,0);fault_done(16);
        start_case(12);address_r(2,1,6,64'h600000000);read_beat(2,0,0,0,0);fault_done(16);
        start_case(13);for(integer n=0;n<9;n=n+1) address_w(n,1,6,64'h700000000+n*4096);fault_done(1);
        start_case(14);for(integer n=0;n<9;n=n+1) write_burst(1,6,n*4096);fault_done(1);
        start_case(15);for(integer n=0;n<9;n=n+1) address_r(4,1,6,64'h700000000+n*4096);fault_done(1);
        start_case(16);awvalid=1;awready=0;tick();awaddr=awaddr^64'h100000000;tick();fault_done(32);
        start_case(17);wvalid=1;wready=0;tick();wstrb=wstrb^64'h8000000000000000;tick();fault_done(32);
        start_case(18);address_w(4,1,6,64'h800000000);write_burst(1,6,0);
        bvalid=1;bid=4;bready=0;bresp=0;tick();bresp=3;tick();fault_done(32);
        start_case(19);arvalid=1;arready=0;tick();arid=arid^4'h8;tick();fault_done(32);
        start_case(20);address_r(4,1,6,64'h800000000);
        rvalid=1;rid=4;rready=0;rlast=1;rdata=0;tick();rdata[511]=1;tick();fault_done(32);
        start_case(21);awvalid=1;awready=0;tick();awvalid=0;tick();fault_done(32);
        active=0;$display("AXI4_DRAIN_SUITE_PASS cases=22 native_endpoint=0");$finish;
    end
    initial begin #1000000;$fatal(1,"AXI4_TIMEOUT");end
endmodule
