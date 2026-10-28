`timescale 1ns/1ps
`define CHECK(C,M) if((C)!==1'b1) $fatal(1,"LREAL_FAIL cycle=%0d %s",cycle,M);
module tb_ntrpu_llama32_layer_memory_r1;
    localparam integer FRAMES=1017348;reg clk=0,rst_n=0;reg [1:0] tail_selected=0;integer mode=0;
    reg [63:0] bus_araddr[0:4];
    reg [7:0] bus_arlen[0:4];
    reg [2:0] bus_arsize[0:4];
    reg [2:0] bus_arprot[0:4];
    reg [1:0] bus_arburst[0:4];
    reg [3:0] bus_arid[0:4];
    reg [3:0] bus_arcache[0:4];
    reg  bus_arlock[0:4];
    reg  bus_arvalid[0:4];
    reg  bus_arready[0:4];
    reg [511:0] bus_rdata[0:4];
    reg [1:0] bus_rresp[0:4];
    reg [3:0] bus_rid[0:4];
    reg  bus_rlast[0:4];
    reg  bus_rvalid[0:4];
    reg  bus_rready[0:4];
    reg [63:0] bus_awaddr[0:4];
    reg [7:0] bus_awlen[0:4];
    reg [2:0] bus_awsize[0:4];
    reg [2:0] bus_awprot[0:4];
    reg [1:0] bus_awburst[0:4];
    reg [3:0] bus_awid[0:4];
    reg [3:0] bus_awcache[0:4];
    reg  bus_awlock[0:4];
    reg  bus_awvalid[0:4];
    reg  bus_awready[0:4];
    reg [511:0] bus_wdata[0:4];
    reg [63:0] bus_wstrb[0:4];
    reg  bus_wlast[0:4];
    reg  bus_wvalid[0:4];
    reg  bus_wready[0:4];
    reg [3:0] bus_bid[0:4];
    reg [1:0] bus_bresp[0:4];
    reg  bus_bvalid[0:4];
    reg  bus_bready[0:4];
    localparam [63:0] SEQ=64'h1234567800000001,EPOCH=64'hfedcba9800000001,LEASE=64'h8000000012340000;
    reg [511:0] memory[0:FRAMES-1],golden[0:FRAMES-1];
    bit valid_frame[0:FRAMES-1],written[0:FRAMES-1];
    integer starts[0:24],sizes[0:24];bit writable[0:24];
    reg [31:0] rope_expected[0:2559],attention_expected[0:2047];
    reg [63:0] read_hash[0:4][0:24];integer region_reads[0:4][0:24];
    integer writes[0:24];integer cycle=0,seed=1,limit,public_cycle,total_writes=0;
    integer r_count[0:4],r_index[0:4],w_count[0:4],w_beats[0:4];
    integer ars[0:4],rs[0:4],aws[0:4],ws[0:4],bs[0:4],r_stalls[0:4],w_stalls[0:4];
    bit read_active[0:4],aw_active[0:4],ar_stalled[0:4],aw_stalled[0:4],w_stalled[0:4];
    reg [63:0] r_base[0:4],w_base[0:4];reg [3:0] read_id[0:4],write_id[0:4];
    reg [511:0] w_buffer[0:4][0:63];bit w_last[0:4][0:63];
    reg [88:0] old_ar[0:4],old_aw[0:4];reg [576:0] old_w[0:4];
    reg [3:0] stage_seen=0,stage_started=0,stage_acked=0;
    integer heads=0,head_acks=0,head=-1,rope_words=0,at_starts=0,at_dones=0,attention_words=0;
    integer source_reads=0,hls_inputs=0;bit source_pending=0;integer source_address=0;
    bit hls_input_pending=0;integer hls_input_address=0;reg [2047:0] source_seen=0;
    bit rp_running=0,at_running=0;reg [103:0] state_seen=0;reg [63:0] output_seen=0;
    string fixture;
    always #2.5 clk=~clk;
    always @(posedge clk)cycle<=cycle+1;
    function automatic [63:0] region_address(input integer r);return (64'(r+1)<<40)+4032;endfunction
    function automatic integer region_index(input [63:0] a);
        integer r;r=int'(a>>40)-1;
        if(r<0 || r>24 || a[5:0]!=0)$fatal(1,"LREAL_FAIL unmapped or unaligned address %016h",a);
        if(a<region_address(r) || a+64>region_address(r)+64'(sizes[r]*64))$fatal(1,"LREAL_FAIL address outside payload %016h",a);
        return r;
    endfunction
    function automatic integer frame_index(input [63:0] a);
        integer r;r=region_index(a);return starts[r]+int'((a-region_address(r))/64);
    endfunction
    function automatic [511:0] read_frame(input [63:0] a);
        integer index;index=frame_index(a);
        if(!valid_frame[index])$fatal(1,"LREAL_FAIL read before actual B commit %016h",a);
        return memory[index];
    endfunction
    function automatic bit read_permission(input integer p,r);
        case(p)
            0:return r<=5;
            1:return r==6 || r==7 || r==9;
            2:return r==8;
            3:return r==24;
            4:case(tail_selected)
                0:return r==10 || r==11;
                1:return r==0 || r==12 || r==13;
                2:return r==14 || (r>=15 && r<=22);
                default:return 0;
            endcase
            default:return 0;
        endcase
    endfunction
    function automatic bit write_permission(input integer p,r);
        case(p)
            0:return r==2 || (r>=6 && r<=8);
            3:return r==24;
            4:case(tail_selected)
                0:return r==11 || r==12;
                1:return r==14 || r==15;
                2:return r>=19 && r<=23;
                default:return 0;
            endcase
            default:return 0;
        endcase
    endfunction
    // Only the actual DUT write payload is installed, and only after B accepts.
    // Expected images are assertions; read_frame never returns them.
    always @(negedge clk)begin
        for(integer p=0;p<5;p=p+1)begin
            bus_arready[p]=rst_n && !read_active[p] && (cycle+p+seed)%5!=0;
            bus_awready[p]=rst_n && !aw_active[p] && (cycle+p+seed)%4!=0;
            bus_wready[p]=rst_n && w_count[p]<64 && (cycle+p+seed)%3!=0;
            if(rst_n)begin
                if(!bus_rvalid[p] && read_active[p] && (cycle+p+seed)%4!=0)begin
                    bus_rdata[p]=read_frame(r_base[p]+64'(r_index[p]*64));bus_rid[p]=read_id[p];
                    bus_rresp[p]=0;bus_rlast[p]=r_index[p]==r_count[p]-1;bus_rvalid[p]=1;
                end
                if(!bus_bvalid[p] && aw_active[p] && w_count[p]==w_beats[p] && (cycle+p+seed)%3!=0)begin
                    bus_bid[p]=write_id[p];bus_bresp[p]=0;bus_bvalid[p]=1;
                end
            end
        end
    end
    always @(posedge clk)begin: memory_service
        integer r,index,off;reg [63:0] a;reg [511:0] data;
        if(rst_n)for(integer p=0;p<5;p=p+1)begin
            if(ar_stalled[p])`CHECK(bus_arvalid[p] && {bus_araddr[p],bus_arlen[p],bus_arsize[p],bus_arburst[p],bus_arid[p],bus_arprot[p],bus_arcache[p],bus_arlock[p]}===old_ar[p],"AR changed under backpressure")
            if(aw_stalled[p])`CHECK(bus_awvalid[p] && {bus_awaddr[p],bus_awlen[p],bus_awsize[p],bus_awburst[p],bus_awid[p],bus_awprot[p],bus_awcache[p],bus_awlock[p]}===old_aw[p],"AW changed under backpressure")
            if(w_stalled[p])`CHECK(bus_wvalid[p] && {bus_wdata[p],bus_wstrb[p],bus_wlast[p]}===old_w[p],"W changed under backpressure")
            ar_stalled[p]=bus_arvalid[p] && !bus_arready[p];aw_stalled[p]=bus_awvalid[p] && !bus_awready[p];w_stalled[p]=bus_wvalid[p] && !bus_wready[p];
            old_ar[p]={bus_araddr[p],bus_arlen[p],bus_arsize[p],bus_arburst[p],bus_arid[p],bus_arprot[p],bus_arcache[p],bus_arlock[p]};
            old_aw[p]={bus_awaddr[p],bus_awlen[p],bus_awsize[p],bus_awburst[p],bus_awid[p],bus_awprot[p],bus_awcache[p],bus_awlock[p]};
            old_w[p]={bus_wdata[p],bus_wstrb[p],bus_wlast[p]};
            if(ar_stalled[p])r_stalls[p]++;if(w_stalled[p])w_stalls[p]++;
            if(bus_arvalid[p] && bus_arready[p])begin
                a=bus_araddr[p];r=region_index(a);off=int'((a-region_address(r))/64);
                `CHECK(!read_active[p] && bus_arlen[p]<64 && bus_arsize[p]==6 && bus_arburst[p]==1,"AR protocol")
                `CHECK(off+int'(bus_arlen[p])+1<=sizes[r] && int'(a[11:0])+(int'(bus_arlen[p])+1)*64<=4096,"AR bounds/page")
                `CHECK(read_permission(p,r),"Wrong master or tail owner read")
                for(integer i=0;i<=int'(bus_arlen[p]);i++)begin `CHECK(valid_frame[starts[r]+off+i],"AR before B commit") end
                r_base[p]=a;r_count[p]=int'(bus_arlen[p])+1;r_index[p]=0;read_id[p]=bus_arid[p];read_active[p]=1;
                $display("LREAL_AR cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,ars[p],r,off*64,r_count[p]);ars[p]++;
            end
            if(bus_rvalid[p] && bus_rready[p])begin
                a=r_base[p]+64'(r_index[p]*64);r=region_index(a);data=read_frame(a);
                `CHECK(read_active[p] && bus_rdata[p]===data && bus_rid[p]==read_id[p] && bus_rresp[p]==0 && bus_rlast[p]==(r_index[p]==r_count[p]-1),"R payload/framing")
                for(integer lane=0;lane<8;lane++)read_hash[p][r]=(read_hash[p][r]^data[lane*64+:64])*64'h100000001b3;
                if(writable[r])$display("LREAL_R cycle=%0d port=%0d region=%0d offset=%0d data=%0128h",cycle,p,r,int'(a-region_address(r)),data);
                region_reads[p][r]++;rs[p]++;r_index[p]++;bus_rvalid[p]<=0;
                if(r_index[p]==r_count[p])read_active[p]=0;
            end
            if(bus_awvalid[p] && bus_awready[p])begin
                a=bus_awaddr[p];r=region_index(a);off=int'((a-region_address(r))/64);
                `CHECK(!aw_active[p] && writable[r] && write_permission(p,r),"AW ownership")
                `CHECK(bus_awlen[p]<64 && bus_awsize[p]==6 && bus_awburst[p]==1,"AW protocol")
                `CHECK(off+int'(bus_awlen[p])+1<=sizes[r] && int'(a[11:0])+(int'(bus_awlen[p])+1)*64<=4096,"AW bounds/page")
                w_base[p]=a;w_beats[p]=int'(bus_awlen[p])+1;write_id[p]=bus_awid[p];aw_active[p]=1;
                $display("LREAL_AW cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,aws[p],r,off*64,w_beats[p]);aws[p]++;
            end
            if(bus_wvalid[p] && bus_wready[p])begin
                `CHECK(w_count[p]<64 && bus_wstrb[p]==64'hffffffffffffffff,"W extent/strobe")
                w_buffer[p][w_count[p]]=bus_wdata[p];w_last[p][w_count[p]]=bus_wlast[p];w_count[p]++;ws[p]++;
                $display("LREAL_W cycle=%0d port=%0d index=%0d last=%0d data=%0128h",cycle,p,ws[p]-1,bus_wlast[p],bus_wdata[p]);
            end
            if(bus_bvalid[p] && bus_bready[p])begin
                `CHECK(aw_active[p] && w_count[p]==w_beats[p] && bus_bresp[p]==0 && bus_bid[p]==write_id[p],"B before full write")
                r=region_index(w_base[p]);off=int'((w_base[p]-region_address(r))/64);
                for(integer i=0;i<w_beats[p];i++)begin
                    index=starts[r]+off+i;data=w_buffer[p][i];
                    `CHECK(!written[index] && w_last[p][i]==(i==w_beats[p]-1),"Duplicate commit or WLAST")
                    `CHECK(data===golden[index],"Actual intermediate differs from immutable reference")
                    memory[index]=data;valid_frame[index]=1;written[index]=1;writes[r]++;total_writes++;
                end
                $display("LREAL_B cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,bs[p],r,off*64,w_beats[p]);
                bs[p]++;aw_active[p]=0;w_count[p]=0;bus_bvalid[p]<=0;
            end
        end
    end


    task step;begin @(posedge clk);#1;@(negedge clk);#0.1;end endtask
    task write_one(input integer p,r,before_aw,input bit corrupt);
        integer count;reg [511:0] value;
        begin
            value=golden[starts[r]];if(corrupt)value[0]=!value[0];
            bus_awaddr[p]=region_address(r);bus_awlen[p]=0;bus_awsize[p]=6;bus_awburst[p]=1;
            bus_wdata[p]=value;bus_wstrb[p]='1;bus_wlast[p]=mode!=7;
            if(before_aw)begin
                bus_wvalid[p]=1;count=0;while(!bus_wready[p])begin step();count++;`CHECK(count<20,"test W deadline") end
                step();bus_wvalid[p]=0;
            end
            bus_awvalid[p]=1;count=0;while(!bus_awready[p])begin step();count++;`CHECK(count<20,"test AW deadline") end
            step();bus_awvalid[p]=0;
            if(!before_aw)begin
                bus_wvalid[p]=1;count=0;while(!bus_wready[p])begin step();count++;`CHECK(count<20,"test W deadline") end
                step();bus_wvalid[p]=0;
            end
            count=0;while(!bus_bvalid[p])begin step();count++;`CHECK(count<20,"test B deadline") end
            bus_bready[p]=1;step();bus_bready[p]=0;
        end
    endtask
    task read_one(input integer p,r,input bit crossing);
        integer count;begin
            bus_araddr[p]=region_address(r);bus_arlen[p]=crossing ? 1:0;bus_arsize[p]=6;bus_arburst[p]=1;
            bus_arvalid[p]=1;count=0;while(!bus_arready[p])begin step();count++;`CHECK(count<20,"test AR deadline") end
            step();bus_arvalid[p]=0;count=0;
            while(!bus_rvalid[p])begin step();count++;`CHECK(count<20,"test R deadline") end
            `CHECK(bus_rdata[p]===memory[starts[r]],"memory test returned incorrect bytes")
            // Keep the memory response stalled for three clocks, then accept.
            repeat(3)begin step();`CHECK(bus_rvalid[p] && bus_rdata[p]===memory[starts[r]],"stalled R changed") end
            bus_rready[p]=1;step();bus_rready[p]=0;
        end
    endtask
    initial begin
        if(!$value$plusargs("MODE=%d",mode))mode=0;
        for(integer p=0;p<5;p++)bus_araddr[p]='0;
        for(integer p=0;p<5;p++)bus_arlen[p]='0;
        for(integer p=0;p<5;p++)bus_arsize[p]='0;
        for(integer p=0;p<5;p++)bus_arprot[p]='0;
        for(integer p=0;p<5;p++)bus_arburst[p]='0;
        for(integer p=0;p<5;p++)bus_arid[p]='0;
        for(integer p=0;p<5;p++)bus_arcache[p]='0;
        for(integer p=0;p<5;p++)bus_arlock[p]='0;
        for(integer p=0;p<5;p++)bus_arvalid[p]='0;
        for(integer p=0;p<5;p++)bus_arready[p]='0;
        for(integer p=0;p<5;p++)bus_rdata[p]='0;
        for(integer p=0;p<5;p++)bus_rresp[p]='0;
        for(integer p=0;p<5;p++)bus_rid[p]='0;
        for(integer p=0;p<5;p++)bus_rlast[p]='0;
        for(integer p=0;p<5;p++)bus_rvalid[p]='0;
        for(integer p=0;p<5;p++)bus_rready[p]='0;
        for(integer p=0;p<5;p++)bus_awaddr[p]='0;
        for(integer p=0;p<5;p++)bus_awlen[p]='0;
        for(integer p=0;p<5;p++)bus_awsize[p]='0;
        for(integer p=0;p<5;p++)bus_awprot[p]='0;
        for(integer p=0;p<5;p++)bus_awburst[p]='0;
        for(integer p=0;p<5;p++)bus_awid[p]='0;
        for(integer p=0;p<5;p++)bus_awcache[p]='0;
        for(integer p=0;p<5;p++)bus_awlock[p]='0;
        for(integer p=0;p<5;p++)bus_awvalid[p]='0;
        for(integer p=0;p<5;p++)bus_awready[p]='0;
        for(integer p=0;p<5;p++)bus_wdata[p]='0;
        for(integer p=0;p<5;p++)bus_wstrb[p]='0;
        for(integer p=0;p<5;p++)bus_wlast[p]='0;
        for(integer p=0;p<5;p++)bus_wvalid[p]='0;
        for(integer p=0;p<5;p++)bus_wready[p]='0;
        for(integer p=0;p<5;p++)bus_bid[p]='0;
        for(integer p=0;p<5;p++)bus_bresp[p]='0;
        for(integer p=0;p<5;p++)bus_bvalid[p]='0;
        for(integer p=0;p<5;p++)bus_bready[p]='0;
        for(integer p=0;p<5;p++)begin
            read_active[p]=0;aw_active[p]=0;r_count[p]=0;r_index[p]=0;w_count[p]=0;w_beats[p]=0;
            ars[p]=0;rs[p]=0;aws[p]=0;ws[p]=0;bs[p]=0;r_stalls[p]=0;w_stalls[p]=0;
            ar_stalled[p]=0;aw_stalled[p]=0;w_stalled[p]=0;
            for(integer r=0;r<25;r++)begin read_hash[p][r]=64'hcbf29ce484222325;region_reads[p][r]=0;end
        end
        for(integer r=0;r<25;r++)begin starts[r]=r*128;sizes[r]=128;writable[r]=1;writes[r]=0;end
        writable[0]=0;memory[0]=512'habcdef1234567890;valid_frame[0]=1;
        golden[starts[2]]=512'h102030405060708090a0b0c0;golden[starts[6]]=512'hffeeddccbbaa99887766554433221100;
        repeat(3)step();rst_n=1;step();
        case(mode)
            0:begin
                read_one(0,0,0);write_one(0,2,1,0);read_one(0,2,0);
                write_one(0,6,0,0);read_one(1,6,0);
                `CHECK(writes[2]==1 && writes[6]==1 && rs[0]==2 && rs[1]==1,"Memory model accounting")
                $display("LAYER_MEMORY_PASS writes=2 reads=3 W_before_AW=1 cross_master=1");$finish;
            end
            1:read_one(0,2,0);
            2:write_one(0,0,0,0);
            3:write_one(0,2,0,1);
            4:begin write_one(0,2,0,0);write_one(0,2,0,0);end
            5:read_one(0,0,1);
            6:read_one(2,0,0);
            7:write_one(0,2,0,0);
        endcase
        $fatal(1,"MEMORY_TEST_FAILED_TO_REJECT mode=%0d",mode);
    end
    initial begin #10000;$fatal(1,"MEMORY_TEST_DEADLINE");end
endmodule
`undef CHECK
