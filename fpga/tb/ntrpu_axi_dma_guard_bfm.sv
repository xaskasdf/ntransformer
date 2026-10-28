// Simulation-only DMA producer/consumer with asynchronous reset cancellation.
`timescale 1ns/1ps
module ntrpu_axi_dma_guard_bfm (
    input wire clk,rst_n,
    output logic held=0,ready=0,
    output logic[63:0] s_host_axi_araddr, s_host_axi_awaddr,
    output logic[7:0] s_host_axi_arlen, s_host_axi_awlen,
    output logic[2:0] s_host_axi_arsize, s_host_axi_arprot, s_host_axi_awsize, s_host_axi_awprot,
    output logic[1:0] s_host_axi_arburst, s_host_axi_awburst,
    output logic[3:0] s_host_axi_arid, s_host_axi_arcache, s_host_axi_awid, s_host_axi_awcache,
    output logic s_host_axi_arlock, s_host_axi_arvalid, s_host_axi_awlock, s_host_axi_awvalid,
    input wire s_host_axi_arready, s_host_axi_awready,
    input wire[511:0] s_host_axi_rdata,
    input wire[1:0] s_host_axi_rresp,
    input wire[3:0] s_host_axi_rid,
    input wire s_host_axi_rlast, s_host_axi_rvalid,
    output logic s_host_axi_rready,
    output logic[511:0] s_host_axi_wdata,
    output logic[63:0] s_host_axi_wstrb,
    output logic s_host_axi_wlast, s_host_axi_wvalid,
    input wire s_host_axi_wready,
    input wire[3:0] s_host_axi_bid,
    input wire[1:0] s_host_axi_bresp,
    input wire s_host_axi_bvalid,
    output logic s_host_axi_bready
);
    bit busy=0;
    integer quiet_edges=0,up_edges=0,epoch=0,phase=0;
    integer aw_count=0,w_count=0,b_count=0,ar_count=0,r_count=0,aborts=0;
    task automatic need(input logic ok,input string why);
        if(ok!==1'b1) $fatal(1,"FAIL DMA epoch=%0d phase=%0d: %s",epoch,phase,why);
    endtask
    task automatic tick();@(posedge clk);#0.001;endtask
    task automatic clear_bus();
        s_host_axi_awvalid=0;s_host_axi_wvalid=0;s_host_axi_arvalid=0;
        s_host_axi_bready=0;s_host_axi_rready=0;
        s_host_axi_awaddr=0;s_host_axi_araddr=0;s_host_axi_awlen=0;s_host_axi_arlen=0;
        s_host_axi_awsize=6;s_host_axi_arsize=6;s_host_axi_awburst=1;s_host_axi_arburst=1;
        s_host_axi_awid=9;s_host_axi_arid=9;s_host_axi_awcache=2;s_host_axi_arcache=2;
        s_host_axi_awprot=0;s_host_axi_arprot=0;s_host_axi_awlock=0;s_host_axi_arlock=0;
        s_host_axi_wdata=0;s_host_axi_wstrb=0;s_host_axi_wlast=0;
    endtask
    initial clear_bus();
    always @(posedge clk) begin
        if(!rst_n && !busy && !s_host_axi_awvalid && !s_host_axi_wvalid &&
           !s_host_axi_arvalid && !s_host_axi_bready && !s_host_axi_rready) begin
            if(quiet_edges==7) held<=1;else quiet_edges<=quiet_edges+1;
        end else begin quiet_edges<=0;held<=0;end
        if(rst_n) begin
            if(up_edges==7) ready<=1;else up_edges<=up_edges+1;
        end else begin up_edges<=0;ready<=0;end
        if(rst_n && ready && !busy)
            need(!s_host_axi_bvalid && !s_host_axi_rvalid,"unsolicited or stale response while idle");
    end
    task automatic send_aw(input logic[63:0] addr,input integer beats,delay_edges);
        repeat(delay_edges) tick();
        @(negedge clk);s_host_axi_awaddr=addr;s_host_axi_awlen=8'(beats-1);s_host_axi_awvalid=1;
        do begin @(posedge clk);end while(!s_host_axi_awready);
        aw_count++;phase=1;
        $display("NTRPU_DMA_AW epoch=%0d ps=%0d addr=%016h beats=%0d count=%0d",epoch,longint'($realtime*1000),addr,beats,aw_count);
        @(negedge clk);s_host_axi_awvalid=0;
    endtask
    task automatic send_w(input integer beats,delay_edges,input logic[511:0] payload[0:3]);
        repeat(delay_edges) tick();
        for(integer beat=0;beat<beats;beat++) begin
            @(negedge clk);s_host_axi_wdata=payload[beat];s_host_axi_wstrb='1;
            s_host_axi_wlast=beat==beats-1;s_host_axi_wvalid=1;
            do begin @(posedge clk);end while(!s_host_axi_wready);
            w_count++;phase=2;
            $display("NTRPU_DMA_W epoch=%0d ps=%0d beat=%0d last=%0b strb=%016h data=%0128h count=%0d",epoch,longint'($realtime*1000),beat,s_host_axi_wlast,s_host_axi_wstrb,s_host_axi_wdata,w_count);
            @(negedge clk);s_host_axi_wvalid=0;
        end
    endtask
    task automatic write(input logic[63:0] addr,input integer beats,
                         input logic[511:0] payload[0:3],
                         input integer aw_delay,w_delay,b_stall,output bit aborted);
        need(rst_n && ready && !busy && beats>=1 && beats<=4 && addr[5:0]==0 &&
             int'(addr[11:0])+beats*64<=4096,"write precondition");
        busy=1;epoch++;phase=0;aborted=0;
        $display("NTRPU_DMA_BEGIN epoch=%0d ps=%0d kind=0 addr=%016h beats=%0d",epoch,longint'($realtime*1000),addr,beats);
        fork
            begin
                fork send_aw(addr,beats,aw_delay);send_w(beats,w_delay,payload);join
                while(!s_host_axi_bvalid) tick();
                phase=3;
                $display("NTRPU_DMA_B_HELD epoch=%0d ps=%0d",epoch,longint'($realtime*1000));
                repeat(b_stall) begin tick();need(s_host_axi_bvalid && s_host_axi_bid==9 && s_host_axi_bresp==0,"held B changed");end
                @(negedge clk);s_host_axi_bready=1;
                @(posedge clk);need(s_host_axi_bvalid && s_host_axi_bid==9 && s_host_axi_bresp==0,"B handshake");b_count++;
                $display("NTRPU_DMA_B epoch=%0d ps=%0d count=%0d",epoch,longint'($realtime*1000),b_count);
                @(negedge clk);s_host_axi_bready=0;
            end
            begin @(negedge rst_n);aborted=1;end
        join_any
        disable fork;
        if(aborted) aborts++;
        clear_bus();busy=0;
        $display("NTRPU_DMA_END epoch=%0d ps=%0d aborted=%0d phase=%0d",epoch,longint'($realtime*1000),aborted,phase);
        phase=0;
    endtask
    task automatic read(input logic[63:0] addr,input integer beats,r_stall,
                        output logic[511:0] payload[0:3],output bit aborted);
        logic[511:0] value;
        need(rst_n && ready && !busy && beats>=1 && beats<=4 && addr[5:0]==0 &&
             int'(addr[11:0])+beats*64<=4096,"read precondition");
        busy=1;epoch++;phase=0;aborted=0;
        $display("NTRPU_DMA_BEGIN epoch=%0d ps=%0d kind=1 addr=%016h beats=%0d",epoch,longint'($realtime*1000),addr,beats);
        fork
            begin
                @(negedge clk);s_host_axi_araddr=addr;s_host_axi_arlen=8'(beats-1);s_host_axi_arvalid=1;
                do begin @(posedge clk);end while(!s_host_axi_arready);
                ar_count++;phase=4;
                $display("NTRPU_DMA_AR epoch=%0d ps=%0d addr=%016h beats=%0d count=%0d",epoch,longint'($realtime*1000),addr,beats,ar_count);
                @(negedge clk);s_host_axi_arvalid=0;
                for(integer beat=0;beat<beats;beat++) begin
                    while(!s_host_axi_rvalid) tick();
                    value=s_host_axi_rdata;phase=5;
                    $display("NTRPU_DMA_R_HELD epoch=%0d ps=%0d beat=%0d",epoch,longint'($realtime*1000),beat);
                    repeat(r_stall) begin
                        tick();need(s_host_axi_rvalid && s_host_axi_rdata===value && s_host_axi_rid==9 &&
                            s_host_axi_rresp==0 && s_host_axi_rlast===(beat==beats-1),"held R changed");
                    end
                    @(negedge clk);s_host_axi_rready=1;
                    @(posedge clk);need(s_host_axi_rvalid && s_host_axi_rdata===value,"R handshake");
                    payload[beat]=value;r_count++;
                    $display("NTRPU_DMA_R epoch=%0d ps=%0d beat=%0d last=%0b data=%0128h count=%0d",epoch,longint'($realtime*1000),beat,s_host_axi_rlast,value,r_count);
                    @(negedge clk);s_host_axi_rready=0;
                end
            end
            begin @(negedge rst_n);aborted=1;end
        join_any
        disable fork;
        if(aborted) aborts++;
        clear_bus();busy=0;
        $display("NTRPU_DMA_END epoch=%0d ps=%0d aborted=%0d phase=%0d",epoch,longint'($realtime*1000),aborted,phase);
        phase=0;
    endtask

    // Probe methods do not alter the counters/epochs of the existing FFN bank.
    function automatic logic [511:0] probe_word(input integer tag,beat);
        logic [511:0] value;
        for(integer lane=0;lane<64;lane++) value[lane*8+:8]=8'(tag*31+beat*7+lane*13);
        return value;
    endfunction
    task automatic probe_aw(input logic[63:0] addr,input integer beats,delay_edges);
        repeat(delay_edges) tick();
        @(negedge clk);s_host_axi_awaddr=addr;s_host_axi_awlen=8'(beats-1);s_host_axi_awvalid=1;
        do begin @(posedge clk);end while(!s_host_axi_awready);
        @(negedge clk);s_host_axi_awvalid=0;
    endtask
    task automatic probe_w(input integer beats,tag,delay_edges);
        repeat(delay_edges) tick();
        for(integer beat=0;beat<beats;beat++) begin
            @(negedge clk);s_host_axi_wdata=probe_word(tag,beat);s_host_axi_wstrb='1;
            s_host_axi_wlast=beat==beats-1;s_host_axi_wvalid=1;
            do begin @(posedge clk);end while(!s_host_axi_wready);
            @(negedge clk);s_host_axi_wvalid=0;
        end
    endtask
    task automatic probe(input bit write_access,input logic[63:0] addr,
                         input integer beats,tag,response,order);
        logic[511:0] value;
        need(rst_n && ready && !busy && beats>=1 && beats<=4 && addr[5:0]==0 &&
             int'(addr[11:0])+beats*64<=4096,"probe legal AXI framing");
        busy=1;
        if(write_access) begin
            fork probe_aw(addr,beats,order==1 ? 7:0);probe_w(beats,tag,order==2 ? 7:0);join
            while(!s_host_axi_bvalid) tick();
            repeat(5) begin tick();need(s_host_axi_bvalid && s_host_axi_bid==9 && s_host_axi_bresp==response,"probe held B");end
            @(negedge clk);s_host_axi_bready=1;
            @(posedge clk);need(s_host_axi_bvalid && s_host_axi_bid==9 && s_host_axi_bresp==response,"probe B response");
            @(negedge clk);s_host_axi_bready=0;
        end else begin
            @(negedge clk);s_host_axi_araddr=addr;s_host_axi_arlen=8'(beats-1);s_host_axi_arvalid=1;
            do begin @(posedge clk);end while(!s_host_axi_arready);
            @(negedge clk);s_host_axi_arvalid=0;
            for(integer beat=0;beat<beats;beat++) begin
                while(!s_host_axi_rvalid) tick();
                value=response==3 ? 512'b0:probe_word(tag,beat);
                repeat(4) begin
                    tick();need(s_host_axi_rvalid && s_host_axi_rdata===value && s_host_axi_rid==9 &&
                        s_host_axi_rresp==response && s_host_axi_rlast===(beat==beats-1),"probe held R");
                end
                @(negedge clk);s_host_axi_rready=1;
                @(posedge clk);need(s_host_axi_rvalid && s_host_axi_rdata===value && s_host_axi_rid==9 &&
                    s_host_axi_rresp==response && s_host_axi_rlast===(beat==beats-1),"probe R response");
                @(negedge clk);s_host_axi_rready=0;
            end
        end
        clear_bus();busy=0;
    endtask
endmodule
