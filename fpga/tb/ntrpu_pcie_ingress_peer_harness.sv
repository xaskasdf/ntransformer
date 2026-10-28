`timescale 1ps / 1ps
// Actual XDMA bypass -> ingress -> bridge -> peer receiver RTL.
// Only destination AXI responses and trusted isolation controls are BFMs.
// line/done aliases observe the production hierarchy; they never drive it.
module ntrpu_pcie_ingress_peer_harness;
    wire clk=board.EP.user_clk;
    wire rst_n=board.EP.user_resetn;
    reg stop=0,source_stopped=0,ingress_isolated=0;
    reg [63:0] quiesce_epoch=0;
    reg arm_valid=0;
    wire arm_ready;
    reg [63:0] arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;
    wire [63:0] epoch,valid_bytes;
    wire lease_held,failed,transport_idle;
    wire [3:0] error_code;
    wire awready,wready,bvalid;
    wire [3:0] bid;
    wire [1:0] bresp;

    integer cycles=0,current_job=-1,aw_count=0,w_count=0,b_count=0,line_count=0,result_count=0;
    integer begin_b=0,begin_line=0,begin_result=0,aw_stall=0,w_stall=0,b_stall=0;
    integer dest_aw=0,dest_w=0,dest_b=0,begin_dest=0;
    reg allow_destination=0;
    wire line_valid=dut.line_valid,line_ready=dut.line_ready;
    wire [63:0] line_epoch=dut.line_epoch,line_tag=dut.line_tag,line_address=dut.line_address;
    wire [511:0] line_data=dut.line_data;
    wire done_valid=dut.done_valid,done_ready=dut.done_ready,done_error=dut.ingress.done_error;
    wire [63:0] done_epoch=dut.saved_epoch,done_tag=dut.saved_tag;
    wire [63:0] peer_epoch;
    wire peer_quarantined,peer_axi_busy;
    wire [2:0] bridge_phase,destination_status;
    wire [63:0] m_axi_awaddr,m_axi_wstrb;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize,m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awcache,m_axi_awid;
    wire m_axi_awlock,m_axi_awvalid,m_axi_wlast,m_axi_wvalid,m_axi_bready;
    wire [511:0] m_axi_wdata;
    wire m_axi_awready=rst_n && allow_destination && cycles%5==0;
    wire m_axi_wready=rst_n && allow_destination && cycles%7==0;
    reg m_axi_bvalid=0;
    reg [3:0] m_axi_bid=0;
    reg [1:0] m_axi_bresp=0;
    wire result_valid;
    reg result_ready=0;
    wire [63:0] result_epoch,result_tag;
    wire [3:0] result_status;
    ntrpu_pcie_ingress_peer dut (
        .clk(clk),.rst_n(rst_n),.stop(stop),.source_stopped(source_stopped),.ingress_isolated(ingress_isolated),
        .quiesce_epoch(quiesce_epoch),.arm_valid(arm_valid),.arm_ready(arm_ready),
        .arm_epoch(arm_epoch),.arm_tag(arm_tag),.arm_window(arm_window),.arm_destination(arm_destination),
        .epoch(epoch),.lease_held(lease_held),.failed(failed),.error_code(error_code),
        .transport_idle(transport_idle),.valid_bytes(valid_bytes),
        .s_axi_awaddr(board.EP.m_axib_awaddr),.s_axi_awid(board.EP.m_axib_awid),
        .s_axi_awlen(board.EP.m_axib_awlen),.s_axi_awsize(board.EP.m_axib_awsize),
        .s_axi_awburst(board.EP.m_axib_awburst),.s_axi_awlock(board.EP.m_axib_awlock),
        .s_axi_awvalid(board.EP.m_axib_awvalid),.s_axi_awready(awready),
        .s_axi_wdata(board.EP.m_axib_wdata),.s_axi_wstrb(board.EP.m_axib_wstrb),
        .s_axi_wlast(board.EP.m_axib_wlast),.s_axi_wvalid(board.EP.m_axib_wvalid),.s_axi_wready(wready),
        .s_axi_bid(bid),.s_axi_bresp(bresp),.s_axi_bvalid(bvalid),.s_axi_bready(board.EP.m_axib_bready),

        .peer_epoch(peer_epoch),.peer_quarantined(peer_quarantined),.peer_axi_busy(peer_axi_busy),
        .bridge_phase(bridge_phase),.destination_status(destination_status),
        .m_axi_awaddr(m_axi_awaddr),.m_axi_awid(m_axi_awid),.m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),.m_axi_awburst(m_axi_awburst),.m_axi_awcache(m_axi_awcache),
        .m_axi_awprot(m_axi_awprot),.m_axi_awlock(m_axi_awlock),.m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),.m_axi_wlast(m_axi_wlast),.m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),.m_axi_bresp(m_axi_bresp),.m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready),
        .result_valid(result_valid),.result_ready(result_ready),.result_epoch(result_epoch),
        .result_tag(result_tag),.result_status(result_status)
    );
    assign board.EP.m_axib_awready=awready;
    assign board.EP.m_axib_wready=wready;
    assign board.EP.m_axib_bid=bid;
    assign board.EP.m_axib_bresp=bresp;
    assign board.EP.m_axib_bvalid=bvalid;
    assign board.EP.m_axib_arready=1'b0;
    assign board.EP.m_axib_rid=4'b0;
    assign board.EP.m_axib_rdata=512'b0;
    assign board.EP.m_axib_rresp=2'b11;
    assign board.EP.m_axib_rlast=1'b1;
    assign board.EP.m_axib_rvalid=1'b0;

    function automatic [31:0] offset(input integer job);
        case(job) 0:offset=0;1,2:offset='h80000;3:offset='hfffc0;default:offset='x;endcase
    endfunction
    function automatic [511:0] expected_line(input integer job);
        reg [511:0] value;
        for(integer lane=0;lane<64;lane++) value[lane*8+:8]=8'(job*53+lane*29+23);
        return value;
    endfunction
    task arm_job(input integer job);
        begin
            @(negedge clk);
            current_job=job;begin_b=b_count;begin_line=line_count;begin_result=result_count;begin_dest=dest_b;
            source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
            arm_epoch=job+1;arm_tag=64'h2000+job;
            arm_window=64'h100000000+offset(job);arm_destination=64'h1fffffffc0;
            arm_valid=1;
            do @(posedge clk); while(!arm_ready);
            @(negedge clk);arm_valid=0;source_stopped=0;ingress_isolated=0;
            if(epoch!==64'(job+1) || peer_epoch!==epoch || valid_bytes!==0 || failed) $fatal(1,"NTRPU_LINK_ARM_STATE");
        end
    endtask
    task finish_job(input integer job);
        integer count;
        begin
            case(job) 0:count=16;1:count=33;2:count=2;3:count=32;endcase
            while(b_count!=begin_b+count || !transport_idle) @(negedge clk);
            if(job==2) begin
                if(!failed || error_code!=4 || line_valid || line_count!=begin_line || valid_bytes!==64'hf)
                    $fatal(1,"NTRPU_LINK_CONFLICT_STATE");
            end else begin

                while(!peer_axi_busy) @(negedge clk);
                repeat(8) @(negedge clk);
                if(result_valid || done_ready!==1 || dest_aw!=begin_dest || dest_w!=begin_dest)
                    $fatal(1,"NTRPU_PEER_LINK_EARLY_DRAIN");
                allow_destination=1;
                while(dest_aw!=begin_dest+1 || dest_w!=begin_dest+1) @(negedge clk);
                allow_destination=0;
                repeat(7) @(negedge clk);
                m_axi_bid=0;m_axi_bresp=0;m_axi_bvalid=1;
                do @(posedge clk); while(!m_axi_bready);
                @(negedge clk);m_axi_bvalid=0;
                while(bridge_phase!=0 || peer_axi_busy) @(negedge clk);
            end
            repeat(4) @(negedge clk);
            if(result_valid) $fatal(1,"NTRPU_LINK_BARRIER_REQUIRED");
            source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch-1;
            repeat(4) @(negedge clk);
            if(result_valid) $fatal(1,"NTRPU_LINK_STALE_BARRIER");
            quiesce_epoch=epoch;
            while(!result_valid) @(negedge clk);
            if(result_status!==(job==2 ? 4'd4 : 4'd0) || result_epoch!==epoch || result_tag!==arm_tag)
                $fatal(1,"NTRPU_LINK_RESULT");
            repeat(5) @(negedge clk);
            result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;
            if(result_count!=begin_result+1 || !transport_idle) $fatal(1,"NTRPU_LINK_RETIRE");
        end
    endtask
    task check_complete;
        begin
            if(aw_count!=83 || w_count!=83 || b_count!=83 || line_count!=3 || result_count!=4 ||
               !transport_idle || line_valid || result_valid || done_ready || peer_axi_busy || dest_aw!=3 || dest_w!=3 || dest_b!=3) $fatal(1,"NTRPU_LINK_DRAIN");
            $display("NTRPU_LINK_COUNTS aw=%0d w=%0d b=%0d lines=%0d results=%0d aw_stall=%0d w_stall=%0d b_stall=%0d ps=%0d",
                     aw_count,w_count,b_count,line_count,result_count,aw_stall,w_stall,b_stall,$time);
        end
    endtask
    always @(posedge clk) if(rst_n) begin
        cycles<=cycles+1;
        if(board.EP.m_axi_awvalid || board.EP.m_axi_arvalid || board.EP.m_axi_wvalid || board.EP.m_axib_arvalid)
            $fatal(1,"NTRPU_LINK_UNEXPECTED_DMA_OR_READ");
        if(board.EP.m_axib_awvalid && !awready) aw_stall<=aw_stall+1;
        if(board.EP.m_axib_wvalid && !wready) w_stall<=w_stall+1;
        if(bvalid && !board.EP.m_axib_bready) b_stall<=b_stall+1;

        if(epoch!==peer_epoch || (result_valid && peer_axi_busy)) $fatal(1,"NTRPU_PEER_LINK_EPOCH_OR_DRAIN");
        if(m_axi_awvalid && (current_job==2 || m_axi_awaddr!==arm_destination || m_axi_awid!==0 ||
            m_axi_awlen!==0 || m_axi_awsize!==6 || m_axi_awburst!==1 || m_axi_awcache!==0 || m_axi_awprot!==0 || m_axi_awlock!==0))
            $fatal(1,"NTRPU_PEER_LINK_AW_FIELDS");
        if(m_axi_wvalid && (current_job==2 || m_axi_wdata!==expected_line(current_job) ||
            m_axi_wstrb!==64'hffffffffffffffff || m_axi_wlast!==1)) $fatal(1,"NTRPU_PEER_LINK_W_FIELDS");
        if(m_axi_awvalid && m_axi_awready) begin
            dest_aw<=dest_aw+1;
            $display("NTRPU_PEER_LINK_MAW addr=%h id=%h len=%0d size=%0d burst=%0d cache=%h prot=%h lock=%b ps=%0d",m_axi_awaddr,m_axi_awid,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awcache,m_axi_awprot,m_axi_awlock,$time);
        end
        if(m_axi_wvalid && m_axi_wready) begin
            dest_w<=dest_w+1;
            $display("NTRPU_PEER_LINK_MW data=%h strb=%h last=%b ps=%0d",m_axi_wdata,m_axi_wstrb,m_axi_wlast,$time);
        end
        if(m_axi_bvalid && m_axi_bready) begin
            if(dest_aw!=begin_dest+1 || dest_w!=begin_dest+1 || m_axi_bid!==0 || m_axi_bresp!==0)
                $fatal(1,"NTRPU_PEER_LINK_B_ORDER");
            dest_b<=dest_b+1;
            $display("NTRPU_PEER_LINK_MB id=%h resp=%0d ps=%0d",m_axi_bid,m_axi_bresp,$time);
        end
        if(arm_valid && arm_ready)
            $display("NTRPU_LINK_ARM job=%0d epoch=%h tag=%h window=%h dest=%h ps=%0d",current_job,arm_epoch,arm_tag,arm_window,arm_destination,$time);
        if(board.EP.m_axib_awvalid && awready) begin
            aw_count<=aw_count+1;
            if(board.EP.m_axib_awlock!==0) $fatal(1,"NTRPU_LINK_LOCK");
            $display("NTRPU_LINK_AW addr=%h id=%h len=%0d size=%0d burst=%0d ps=%0d",board.EP.m_axib_awaddr,board.EP.m_axib_awid,board.EP.m_axib_awlen,board.EP.m_axib_awsize,board.EP.m_axib_awburst,$time);
        end
        if(board.EP.m_axib_wvalid && wready) begin
            w_count<=w_count+1;
            $display("NTRPU_LINK_W data=%h strb=%h last=%b ps=%0d",board.EP.m_axib_wdata,board.EP.m_axib_wstrb,board.EP.m_axib_wlast,$time);
        end
        if(bvalid && board.EP.m_axib_bready) begin
            b_count<=b_count+1;
            if(bid!==0 || bresp!==((current_job==2 && b_count==begin_b+1) ? 2'd2 : 2'd0)) $fatal(1,"NTRPU_LINK_B_RESPONSE");
            $display("NTRPU_LINK_B id=%h resp=%0d ps=%0d",bid,bresp,$time);
        end
        if(line_valid && (current_job==2 || line_data!==expected_line(current_job) ||
           line_epoch!==epoch || line_tag!==arm_tag || line_address!==arm_destination || valid_bytes!==64'hffffffffffffffff))
            $fatal(1,"NTRPU_LINK_LINE_CONTENT");
        if(line_valid && line_ready) begin
            line_count<=line_count+1;
            $display("NTRPU_LINK_LINE job=%0d epoch=%h tag=%h addr=%h data=%h ps=%0d",current_job,line_epoch,line_tag,line_address,line_data,$time);
        end
        if(done_valid && done_ready) $display("NTRPU_LINK_DONE epoch=%h tag=%h error=%b ps=%0d",done_epoch,done_tag,done_error,$time);
        if(result_valid && result_ready) begin
            result_count<=result_count+1;
            $display("NTRPU_LINK_RESULT job=%0d epoch=%h tag=%h status=%0d ps=%0d",current_job,result_epoch,result_tag,result_status,$time);
        end
    end
    initial begin #200000000;$fatal(1,"NTRPU_LINK_TIMEOUT");end
endmodule
