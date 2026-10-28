// Independent AXI write sink for the memory-to-memory GEMV bank.
`timescale 1ns/1ps
module ntrpu_result_memory_model (
    input logic clk,rst_n,start,
    input logic [5:0] rows,
    input logic [63:0] address,
    input int identity,scenario,
    input logic [2047:0] expected_output,
    input logic commit_seen,
    input logic [63:0] m_axi_awaddr,
    input logic [7:0] m_axi_awlen,
    input logic [2:0] m_axi_awsize,m_axi_awprot,
    input logic [1:0] m_axi_awburst,
    input logic [3:0] m_axi_awid,m_axi_awcache,
    input logic m_axi_awlock,m_axi_awvalid,
    output logic m_axi_awready=0,
    input logic [511:0] m_axi_wdata,
    input logic [63:0] m_axi_wstrb,
    input logic m_axi_wlast,m_axi_wvalid,
    output logic m_axi_wready=0,
    output logic [3:0] m_axi_bid=0,
    output logic [1:0] m_axi_bresp=0,
    output logic m_axi_bvalid=0,
    input logic m_axi_bready,
    output int aw_count=0,w_count=0,b_count=0,byte_count=0,aw_stalls=0,w_stalls=0
);
    logic [2047:0] expected;
    logic [63:0] next_address,strobe,wanted;
    logic [71:0] saved_aw;
    logic [576:0] saved_w;
    logic [7:0] memory [0:319];
    int n,mode,id,cycle,first_write,burst_beats,beat_count,total_beats,b_cursor,bytes_this;
    bit have_aw=0,committed=0,aw_held=0,w_held=0,clear_b=0;
    function automatic void need(input bit ok,input string reason);
        if(!ok) $fatal(1,"FAIL STORE WRITE case=%0d cycle=%0d: %s",id,cycle,reason);
    endfunction
    always @(negedge clk) begin
        if(!rst_n) begin m_axi_awready=0;m_axi_wready=0;m_axi_bvalid=0;end
        else begin
            if(clear_b) begin m_axi_bvalid=0;clear_b=0;end
            m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
            if(mode==43 && (first_write==0 || cycle<first_write+80)) m_axi_awready=0;
            if(mode==44 && (first_write==0 || cycle<first_write+80)) m_axi_wready=0;
            if(mode==54 || mode==55) m_axi_awready=1;
            if(mode==54 && w_count>=1) m_axi_wready=0;
            if(have_aw && beat_count==burst_beats && !m_axi_bvalid && mode!=55 && !(mode==45 && cycle<first_write+80)) begin
                m_axi_bvalid=1;m_axi_bid=0;m_axi_bresp=0;
                if((mode==40 && b_count==0) || (mode==41 && b_count==1)) m_axi_bresp=2;
                if(mode==42 && b_count==0) m_axi_bid=1;
            end
        end
    end
    always @(posedge clk) begin
        if(!rst_n) begin
            have_aw=0;committed=0;aw_held=0;w_held=0;clear_b=0;
            aw_count=0;w_count=0;b_count=0;byte_count=0;aw_stalls=0;w_stalls=0;
            first_write=0;cycle=0;beat_count=0;b_cursor=0;mode=0;
        end else begin
            cycle++;
            if(start) begin
                need(!m_axi_awvalid && !m_axi_wvalid && !m_axi_bvalid,"previous write remains active");
                expected=expected_output;next_address=address;n=int'(rows);mode=scenario;id=identity;
                total_beats=(n+7)/8;burst_beats=total_beats;
                if(burst_beats>(4096-int'(address[11:0]))/64) burst_beats=(4096-int'(address[11:0]))/64;
                have_aw=0;committed=0;aw_held=0;w_held=0;clear_b=0;
                aw_count=0;w_count=0;b_count=0;byte_count=0;aw_stalls=0;w_stalls=0;
                first_write=0;cycle=0;beat_count=0;b_cursor=0;
                foreach(memory[j]) memory[j]=8'ha5;
            end
            if(commit_seen) committed=1;
            if(aw_held) need(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen}===saved_aw,"AW stability");
            if(w_held) need(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"W stability");
            aw_held=m_axi_awvalid&&!m_axi_awready;w_held=m_axi_wvalid&&!m_axi_wready;
            saved_aw={m_axi_awaddr,m_axi_awlen};saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            aw_stalls+=int'(aw_held);w_stalls+=int'(w_held);
            if(m_axi_awvalid || m_axi_wvalid) begin
                need(committed,"write before compute completion");
                if(first_write==0) first_write=cycle;
            end
            if(m_axi_awvalid && m_axi_awready) begin
                need(!have_aw && aw_count==b_count,"more than one outstanding write");
                need(m_axi_awaddr==next_address && int'(m_axi_awlen)+1==burst_beats,"write address/page split");
                need(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"write attributes");
                have_aw=1;aw_count++;
            end
            if(m_axi_wvalid && m_axi_wready) begin
                need(beat_count<burst_beats && w_count<total_beats,"extra write beat");
                strobe=0;bytes_this=0;
                for(int lane=0;lane<8;lane++) begin
                    wanted=0;
                    if(w_count*8+lane<n) begin
                        wanted=expected[(w_count*8+lane)*64+:64];strobe[lane*8+:8]=8'hff;bytes_this+=8;
                        $display("NTRPU_STORE_VALUE case=%0d row=%0d bits=%016h",id,w_count*8+lane,m_axi_wdata[lane*64+:64]);
                    end
                    need(m_axi_wdata[lane*64+:64]===wanted,"result bytes mismatch independent reference");
                end
                need(m_axi_wstrb==strobe && m_axi_wlast==(beat_count==burst_beats-1),"write strobe/last");
                for(int j=0;j<64;j++) if(m_axi_wstrb[j]) memory[w_count*64+j]=m_axi_wdata[j*8+:8];
                w_count++;beat_count++;byte_count+=bytes_this;
                for(int j=0;j<320;j++)
                    if(j>=byte_count) need(memory[j]==8'ha5,"write modified tail guard");
            end
            if(m_axi_bvalid && m_axi_bready) begin
                need(have_aw && beat_count==burst_beats,"early write response");
                next_address+=64'(burst_beats*64);b_count++;have_aw=0;beat_count=0;clear_b=1;
                burst_beats=total_beats-w_count;
                if(burst_beats>(4096-int'(next_address[11:0]))/64) burst_beats=(4096-int'(next_address[11:0]))/64;
            end
        end
    end
endmodule
