// Bounded simulation sink at the unmodified XDMA bypass interface.
// Three distinct 64-bit addresses; not a DDR model or synthesizable receiver.
`timescale 1ps/1ps
module ntrpu_pcie_bypass_memory;
    wire clk = board.EP.user_clk;
    wire rst_n = board.EP.user_resetn;
    integer cycles=0, aw_count=0, w_count=0, b_count=0, ar_count=0, r_count=0;
    integer aw_stall=0, w_stall=0, ar_stall=0, b_delay=-1, r_delay=-1;
    integer writes=0, aw_slot=0, ar_slot=0;
    bit aw_pending=0, w_pending=0, bvalid=0, rvalid=0;
    logic [3:0] awid=0, bid=0, rid=0;
    logic [511:0] wdata=0, rdata=0, memory[0:2];
    logic [63:0] wstrb=0;
    wire awready = rst_n && !aw_pending && !bvalid && b_delay<0 && cycles%3==0;
    wire wready = rst_n && !w_pending && !bvalid && b_delay<0 && cycles%5==0;
    wire arready = rst_n && !rvalid && r_delay<0 && cycles%4==0;
    assign board.EP.m_axib_awready=awready;
    assign board.EP.m_axib_wready=wready;
    assign board.EP.m_axib_bid=bid;
    assign board.EP.m_axib_bresp=2'b00;
    assign board.EP.m_axib_bvalid=bvalid;
    assign board.EP.m_axib_arready=arready;
    assign board.EP.m_axib_rid=rid;
    assign board.EP.m_axib_rdata=rdata;
    assign board.EP.m_axib_rresp=2'b00;
    assign board.EP.m_axib_rlast=1'b1;
    assign board.EP.m_axib_rvalid=rvalid;

    function automatic integer slot(input logic [63:0] address);
        case(address)
            64'h0000000100000000: slot=0;
            64'h0000000100080000: slot=1;
            64'h00000001000fffc0: slot=2;
            default: begin
                $fatal(1,"NTRPU_BYPASS_ADDRESS address=%h",address);
                slot=-1;
            end
        endcase
    endfunction

    function automatic logic [31:0] expected(input integer index);
        case(index)
            0: expected=32'h10234567;
            1: expected=32'h89abcdef;
            2: expected=32'hfedcba98;
            default: expected='x;
        endcase
    endfunction

    initial begin
        for(integer i=0;i<3;i++) memory[i]={64{8'ha5}};
        #200000000;
        $fatal(1,"NTRPU_BYPASS_TIMEOUT");
    end

    always @(posedge clk) begin
        if(rst_n) begin
            cycles<=cycles+1;
            if(board.EP.m_axi_awvalid || board.EP.m_axi_arvalid || board.EP.m_axi_wvalid)
                $fatal(1,"NTRPU_BYPASS_UNEXPECTED_DMA_TRAFFIC");
            if(board.EP.m_axib_awvalid && !awready) aw_stall<=aw_stall+1;
            if(board.EP.m_axib_wvalid && !wready) w_stall<=w_stall+1;
            if(board.EP.m_axib_arvalid && !arready) ar_stall<=ar_stall+1;
            if(board.EP.m_axib_awvalid && awready) begin
                $display("NTRPU_BYPASS_AW addr=%h id=%h len=%0d size=%0d burst=%0d ps=%0d",
                    board.EP.m_axib_awaddr,board.EP.m_axib_awid,board.EP.m_axib_awlen,
                    board.EP.m_axib_awsize,board.EP.m_axib_awburst,$time);
                if(board.EP.m_axib_awlen!==0 || !(board.EP.m_axib_awsize inside {2,6}) ||
                   board.EP.m_axib_awburst!==1 || $isunknown(board.EP.m_axib_awid))
                    $fatal(1,"NTRPU_BYPASS_AW_FORMAT");
                aw_slot<=slot(board.EP.m_axib_awaddr); awid<=board.EP.m_axib_awid;
                aw_pending<=1; aw_count<=aw_count+1;
            end
            if(board.EP.m_axib_wvalid && wready) begin
                $display("NTRPU_BYPASS_W data=%h strb=%h last=%b ps=%0d",
                    board.EP.m_axib_wdata,board.EP.m_axib_wstrb,board.EP.m_axib_wlast,$time);
                if(board.EP.m_axib_wstrb!==64'hf || board.EP.m_axib_wlast!==1)
                    $fatal(1,"NTRPU_BYPASS_W_FORMAT");
                wdata<=board.EP.m_axib_wdata; wstrb<=board.EP.m_axib_wstrb;
                w_pending<=1; w_count<=w_count+1;
            end
            if(aw_pending && w_pending) begin
                if(wdata[31:0]!==expected(aw_slot) || aw_slot!=writes)
                    $fatal(1,"NTRPU_BYPASS_WRITE_CONTENT");
                for(integer lane=0;lane<64;lane++)
                    if(wstrb[lane]) memory[aw_slot][lane*8+:8]<=wdata[lane*8+:8];
                writes<=writes+1; bid<=awid; b_delay<=5;
                aw_pending<=0; w_pending<=0;
            end
            if(b_delay>0) b_delay<=b_delay-1;
            if(b_delay==0) begin bvalid<=1; b_delay<=-1; end
            if(bvalid && board.EP.m_axib_bready) begin
                $display("NTRPU_BYPASS_B id=%h resp=0 ps=%0d",bid,$time);
                bvalid<=0; b_count<=b_count+1;
            end
            if(board.EP.m_axib_arvalid && arready) begin
                $display("NTRPU_BYPASS_AR addr=%h id=%h len=%0d size=%0d burst=%0d ps=%0d",
                    board.EP.m_axib_araddr,board.EP.m_axib_arid,board.EP.m_axib_arlen,
                    board.EP.m_axib_arsize,board.EP.m_axib_arburst,$time);
                if(board.EP.m_axib_arlen!==0 || !(board.EP.m_axib_arsize inside {2,6}) ||
                   board.EP.m_axib_arburst!==1 || $isunknown(board.EP.m_axib_arid) || writes!=3)
                    $fatal(1,"NTRPU_BYPASS_AR_FORMAT");
                ar_slot=slot(board.EP.m_axib_araddr);
                if(ar_slot!=ar_count) $fatal(1,"NTRPU_BYPASS_READ_ORDER");
                rdata<=memory[ar_slot]; rid<=board.EP.m_axib_arid;
                ar_count<=ar_count+1; r_delay<=7;
            end
            if(r_delay>0) r_delay<=r_delay-1;
            if(r_delay==0) begin rvalid<=1; r_delay<=-1; end
            if(rvalid && board.EP.m_axib_rready) begin
                $display("NTRPU_BYPASS_R id=%h resp=0 last=1 data=%h ps=%0d",rid,rdata,$time);
                rvalid<=0; r_count<=r_count+1;
            end
        end
    end

    task check_complete;
        begin
            if(aw_count!=3 || w_count!=3 || b_count!=3 || ar_count!=3 || r_count!=3 ||
               aw_pending || w_pending || bvalid || rvalid || b_delay!=-1 || r_delay!=-1)
                $fatal(1,"NTRPU_BYPASS_INCOMPLETE");
            for(integer i=0;i<3;i++)
                if(memory[i]!=={{60{8'ha5}},expected(i)}) $fatal(1,"NTRPU_BYPASS_SENTINEL");
            $display("NTRPU_BYPASS_MEMORY_PASS aw=%0d w=%0d b=%0d ar=%0d r=%0d aw_stall=%0d w_stall=%0d ar_stall=%0d ps=%0d",
                aw_count,w_count,b_count,ar_count,r_count,aw_stall,w_stall,ar_stall,$time);
        end
    endtask
endmodule
