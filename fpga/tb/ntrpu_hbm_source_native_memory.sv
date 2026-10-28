`timescale 1ns/1ps
// Native PC0/PC16 memory, initialized only through accepted AXI3 writes.
module ntrpu_hbm_source_native_memory (
    input wire clk,cold_reset_n,reader_stop,preload_start,
    output reg preload_done=0,
    output wire reader_started,reader_fault,
    input wire [1:0] cmd_valid,
    output wire [1:0] cmd_ready,
    input wire [127:0] cmd_addr,cmd_tag,
    input wire [15:0] cmd_len,
    output wire [1:0] r_data_valid,r_data_last,
    input wire [1:0] r_data_ready,
    output wire [1023:0] r_data,
    output wire [127:0] r_data_tag,
    output wire [1:0] r_cpl_valid,
    input wire [1:0] r_cpl_ready,
    output wire [127:0] r_cpl_tag,
    output wire [3:0] r_cpl_status,
    output wire [25:0] r_cpl_bytes
);
    reg mem_clk=0,ref_clk=0,apb_clk=0,apb_rst_n=0;
    always #1.111 mem_clk=~mem_clk;
    always #5 ref_clk=~ref_clk;
    initial begin #1.7;forever #5 apb_clk=~apb_clk;end
    initial begin #100;apb_rst_n=1;end
    wire reader_rst_n,mem_rst_n,reader_drained;
    wire apb_complete_0,apb_complete_1;
    wire [1:0] h_arvalid,h_arready,h_rvalid,h_rlast,h_rready;
    wire [67:0] h_araddr;
    wire [7:0] h_arlen;
    wire [11:0] h_arid,h_rid;
    wire [5:0] h_arsize;
    wire [3:0] h_arburst,h_rresp;
    wire [511:0] h_rdata;
    reg [67:0] h_awaddr=0;
    reg [7:0] h_awlen=0;
    reg [1:0] h_awvalid=0,h_wvalid=0,h_wlast=0;
    reg [511:0] h_wdata=0;
    wire [1:0] h_awready,h_wready,h_bvalid;
    wire [3:0] h_bresp;
    wire [11:0] h_bid;
    wire [1:0] h_bready=3;
    wire DRAM_0_STAT_CATTRIP,DRAM_1_STAT_CATTRIP;
    wire [6:0] DRAM_0_STAT_TEMP,DRAM_1_STAT_TEMP;
    ntrpu_hbm_read_return reader(
        .clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_reset_n),
        .init_complete_async({apb_complete_1,apb_complete_0}),.stop(reader_stop),.fault_in(1'b0),
        .rst_n(reader_rst_n),.mem_rst_n(mem_rst_n),.started(reader_started),.fault(reader_fault),.drained(reader_drained),
        .cmd_valid(cmd_valid),.cmd_ready(cmd_ready),.cmd_addr(cmd_addr),.cmd_tag(cmd_tag),.cmd_len(cmd_len),
        .data_valid(r_data_valid),.data_ready(r_data_ready),.data_last(r_data_last),.data(r_data),.data_tag(r_data_tag),
        .cpl_valid(r_cpl_valid),.cpl_ready(r_cpl_ready),.cpl_tag(r_cpl_tag),.cpl_status(r_cpl_status),.cpl_bytes(r_cpl_bytes),
        .h_arvalid(h_arvalid),.h_arready(h_arready),.h_araddr(h_araddr),.h_arlen(h_arlen),.h_arid(h_arid),.h_arsize(h_arsize),.h_arburst(h_arburst),
        .h_rvalid(h_rvalid),.h_rready(h_rready),.h_rdata(h_rdata),.h_rid(h_rid),.h_rresp(h_rresp),.h_rlast(h_rlast));
    ntrpu_hbm_native_rw memory(
        .axi_clk(mem_clk),.axi_cold_reset_n(mem_rst_n),
        .s_awaddr(h_awaddr),.s_awburst(4'b0101),.s_awid(12'b0),.s_awlen(h_awlen),.s_awsize(6'b101101),
        .s_awvalid(h_awvalid),.s_awready(h_awready),.s_wdata(h_wdata),.s_wstrb(64'hffffffffffffffff),
        .s_wlast(h_wlast),.s_wvalid(h_wvalid),.s_wready(h_wready),.s_bid(h_bid),.s_bresp(h_bresp),.s_bvalid(h_bvalid),.s_bready(h_bready),
        .s_araddr(h_araddr),.s_arburst(h_arburst),.s_arid(h_arid),.s_arlen(h_arlen),.s_arsize(h_arsize),
        .s_arvalid(h_arvalid),.s_arready(h_arready),.s_rdata(h_rdata),.s_rdata_parity(),.s_rid(h_rid),
        .s_rlast(h_rlast),.s_rresp(h_rresp),.s_rvalid(h_rvalid),.s_rready(h_rready),
        .HBM_REF_CLK_0(ref_clk),.HBM_REF_CLK_1(ref_clk),
        .APB_0_PWDATA(32'b0),.APB_0_PADDR(22'b0),.APB_0_PCLK(apb_clk),.APB_0_PENABLE(1'b0),.APB_0_PRESET_N(apb_rst_n),.APB_0_PSEL(1'b0),.APB_0_PWRITE(1'b0),
        .APB_1_PWDATA(32'b0),.APB_1_PADDR(22'b0),.APB_1_PCLK(apb_clk),.APB_1_PENABLE(1'b0),.APB_1_PRESET_N(apb_rst_n),.APB_1_PSEL(1'b0),.APB_1_PWRITE(1'b0),
        .APB_0_PRDATA(),.APB_0_PREADY(),.APB_0_PSLVERR(),.APB_1_PRDATA(),.APB_1_PREADY(),.APB_1_PSLVERR(),
        .apb_complete_0(apb_complete_0),.apb_complete_1(apb_complete_1),
        .DRAM_0_STAT_CATTRIP(DRAM_0_STAT_CATTRIP),.DRAM_0_STAT_TEMP(DRAM_0_STAT_TEMP),
        .DRAM_1_STAT_CATTRIP(DRAM_1_STAT_CATTRIP),.DRAM_1_STAT_TEMP(DRAM_1_STAT_TEMP));
    integer responses[0:1]='{0,0},write_bursts[0:1]='{0,0},read_bursts[0:1]='{0,0};
    function automatic integer length_for(input integer i);
        case(i)0:return 64;1:return 128;2:return 256;3:return 4096;endcase
    endfunction
    function automatic [7:0] pattern(input integer i,offset);
        return 8'((offset*37)^(offset>>7)^((i+1)*53));
    endfunction
    task automatic preload(input integer channel);
        integer index,offset,beats;
        for(integer slot=0;slot<2;slot++)begin
            index=channel+2*slot;
            for(offset=0;offset<length_for(index);offset+=512)begin
                beats=(length_for(index)-offset)/32;if(beats>16)beats=16;
                @(negedge mem_clk);
                h_awaddr[channel*34+:34]=34'h4000+34'h200000000*34'(channel)+34'(slot*4096+offset);
                h_awlen[channel*4+:4]=4'(beats-1);h_awvalid[channel]=1;
                do @(posedge mem_clk);while(!h_awready[channel]);
                @(negedge mem_clk);h_awvalid[channel]=0;
                for(integer beat=0;beat<beats;beat++)begin
                    for(integer b=0;b<32;b++)h_wdata[channel*256+b*8+:8]=pattern(index,offset+beat*32+b);
                    h_wlast[channel]=(beat==beats-1);h_wvalid[channel]=1;
                    do @(posedge mem_clk);while(!h_wready[channel]);
                    @(negedge mem_clk);h_wvalid[channel]=0;
                end
                wait(responses[channel]==write_bursts[channel]);
            end
        end
    endtask
    initial begin
        wait(preload_start && reader_started);
        fork preload(0);preload(1);join
        @(negedge clk);preload_done=1;
        $display("NH PRELOAD_DONE t=%0d bytes=4544 responses=%0d",longint'($realtime*1000),responses[0]+responses[1]);
    end
    always @(posedge mem_clk)if(mem_rst_n)begin
        for(integer c=0;c<2;c++)begin
            if(h_awvalid[c] && h_awready[c])begin
                write_bursts[c]++;$display("NH AW t=%0d c=%0d addr=%h len=%h",longint'($realtime*1000),c,h_awaddr[c*34+:34],h_awlen[c*4+:4]);
            end
            if(h_wvalid[c] && h_wready[c])$display("NH W t=%0d c=%0d data=%h last=%h",longint'($realtime*1000),c,h_wdata[c*256+:256],h_wlast[c]);
            if(h_bvalid[c] && h_bready[c])begin
                if(h_bresp[c*2+:2]!==0 || h_bid[c*6+:6]!==0)$fatal(1,"NH bad preload response");
                responses[c]++;$display("NH B t=%0d c=%0d resp=%h id=%h",longint'($realtime*1000),c,h_bresp[c*2+:2],h_bid[c*6+:6]);
            end
            if(h_arvalid[c] && h_arready[c])begin
                if(!preload_done)$fatal(1,"NH read before publication");
                read_bursts[c]++;$display("NH AR t=%0d c=%0d addr=%h len=%h",longint'($realtime*1000),c,h_araddr[c*34+:34],h_arlen[c*4+:4]);
            end
            if(h_rvalid[c] && h_rready[c])$display("NH R t=%0d c=%0d data=%h last=%h resp=%h id=%h",longint'($realtime*1000),c,h_rdata[c*256+:256],h_rlast[c],h_rresp[c*2+:2],h_rid[c*6+:6]);
        end
    end
endmodule
