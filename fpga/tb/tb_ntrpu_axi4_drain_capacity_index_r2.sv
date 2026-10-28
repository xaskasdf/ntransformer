`timescale 1ns / 1ps
module tb_ntrpu_axi4_drain_capacity_index_r2;
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
    wire [5:0] aw_pending,w_pending;
    wire [7:0] ar_pending;
    wire [8:0] w_partial;
    ntrpu_axi4_drain_monitor_r2 #(.WRITE_DEPTH(32)) dut(.*);
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
        $display("AXI4_CASE case=%0d write_depth=32 read_depth_per_id=8 ids=16 data_bits=512",n);
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

    task hold_inventory;
        req_valid=1;source_fenced=1;
        repeat(5) begin tick();must(!ack_valid,"nonempty inventory cannot certify");end
        req_valid=0;source_fenced=0;tick();
    endtask
    initial begin
        #10;
        start_case(0);
        for(integer n=0;n<32;n=n+1) address_w(n%16,lens[n%4],6,64'h100000000+n*4096);
        for(integer n=0;n<32;n=n+1) write_burst(lens[n%4],6,n*4096);
        must(aw_pending==32 && w_pending==32,"full32 writes");hold_inventory();
        for(integer n=0;n<96;n=n+1) begin
            awvalid=1;awid=n%16;awlen=0;awsize=6;awaddr=64'h100000000+(n+32)*4096;
            wvalid=1;wlast=1;wdata=pattern((n+32)*4096);wstrb=mask(0,6);
            response_b(n%16,n%4,0);
            must(!protocol_fault && aw_pending==32 && w_pending==32,"full write replacement");
        end
        awvalid=0;wvalid=0;
        for(integer round=0;round<2;round=round+1)
            for(integer id=15;id>=0;id=id-1) response_b(id,id%4,id%3);
        drain();

        start_case(1);
        for(integer batch=0;batch<4;batch=batch+1) begin
            for(integer n=0;n<31;n=n+1) write_burst(lens[n%6],(n%6==5)?4:6,(batch*32+n)*4096);
            for(integer beat=0;beat<3;beat=beat+1) write_beat(beat,6,(batch*32+31)*4096,0);
            must(w_pending==31 && w_partial==3 && aw_pending==0,"complete and partial W before AW");
            hold_inventory();
            for(integer n=0;n<32;n=n+1)
                address_w(n%16,(n==31)?4:lens[n%6],(n%6==5 && n!=31)?4:6,64'h200000000+(batch*32+n)*4096);
            write_beat(3,6,(batch*32+31)*4096,1);
            for(integer round=0;round<2;round=round+1)
                for(integer id=15;id>=0;id=id-1) response_b(id,id%4,0);
            must(!protocol_fault && aw_pending==0 && w_pending==0,"wrapped write batch drained");
            $display("CAPACITY_PHASE case=1 batch=%0d wrapped_transactions=%0d",batch,(batch+1)*32);
        end
        drain();

        start_case(2);
        for(integer q=0;q<8;q=q+1)
            for(integer id=0;id<16;id=id+1) address_r(id,lens[q%4],6,64'h300000000+(q*16+id)*4096);
        must(ar_pending==128,"all128 read slots populated");hold_inventory();
        for(integer q=0;q<8;q=q+1)
            for(integer beat=0;beat<lens[q%4];beat=beat+1)
                for(integer id=15;id>=0;id=id-1) begin
                    arvalid=beat==lens[q%4]-1;arid=id;arlen=2;arsize=6;
                    araddr=64'h300000000+((q+8)*16+id)*4096;
                    read_beat(id,beat,(q*16+id)*4096,beat==lens[q%4]-1,
                        (beat==0 && lens[q%4]>1 && (id+q)%7==0)?2:0);
                    arvalid=0;must(!protocol_fault && ar_pending==128,"full read replacement");
                end
        for(integer q=8;q<16;q=q+1)
            for(integer beat=0;beat<3;beat=beat+1)
                for(integer id=15;id>=0;id=id-1) read_beat(id,beat,(q*16+id)*4096,beat==2,0);
        drain();

        start_case(3);
        for(integer n=0;n<32;n=n+1) address_w(n%16,1,6,64'h400000000+n*4096);
        for(integer n=0;n<32;n=n+1) write_burst(1,6,n*4096);
        for(integer n=0;n<128;n=n+1) address_r(n%16,1,6,64'h400000000+n*4096);
        must(aw_pending==32 && w_pending==32 && ar_pending==128,"all inventories full");hold_inventory();
        for(integer n=0;n<96;n=n+1) begin
            awvalid=1;awid=n%16;awlen=0;awsize=6;awaddr=64'h400000000+(n+32)*4096;
            wvalid=1;wlast=1;wdata=pattern((n+32)*4096);wstrb=mask(0,6);
            bvalid=1;bid=n%16;bresp=n%4;
            arvalid=1;arid=n%16;arlen=0;arsize=6;araddr=64'h400000000+(n+128)*4096;
            read_beat(n%16,0,n*4096,1,0);
            must(!protocol_fault && aw_pending==32 && w_pending==32 && ar_pending==128,"all-channel full replacement");
        end
        awvalid=0;wvalid=0;bvalid=0;arvalid=0;
        for(integer n=96;n<128;n=n+1) response_b(n%16,n%4,0);
        for(integer n=96;n<224;n=n+1) read_beat(n%16,0,n*4096,1,0);
        drain();

        start_case(4);for(integer n=0;n<33;n=n+1) address_w(n%16,1,6,64'h500000000+n*4096);fault_done(1);
        start_case(5);for(integer n=0;n<33;n=n+1) write_burst(1,6,n*4096);fault_done(1);
        start_case(6);
        for(integer n=0;n<128;n=n+1) address_r(n%16,1,6,64'h500000000+n*4096);
        address_r(15,1,6,64'h500080000);fault_done(1);
        start_case(7);for(integer n=0;n<256;n=n+1) write_beat(n,4,0,0);fault_done(8);
        start_case(8);
        for(integer n=0;n<31;n=n+1) write_burst(2,6,n*4096);
        write_burst(3,6,31*4096);
        for(integer n=0;n<32;n=n+1) address_w(n%16,2,6,64'h600000000+n*4096);
        fault_done(8);
        start_case(9);address_w(7,4,6,64'h600000000);write_beat(0,6,0,0);
        bid=7;bvalid=1;bready=0;tick();fault_done(2);
        start_case(10);
        for(integer n=0;n<128;n=n+1) address_r(n%16,4,6,64'h600000000+n*4096);
        read_beat(15,0,15*4096,1,0);fault_done(16);
        active=0;$display("AXI4_CAPACITY_SUITE_PASS cases=11 write_depth=32 read_total=128 native_endpoint=0");$finish;
    end
    initial begin #1000000;$fatal(1,"AXI4_CAPACITY_TIMEOUT");end
endmodule
