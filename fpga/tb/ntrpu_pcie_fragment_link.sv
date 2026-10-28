// Native XDMA format probe. Lease management, assembly and memory are BFMs.
`timescale 1ps/1ps
module ntrpu_pcie_fragment_link;
    wire clk=board.EP.user_clk,rst_n=board.EP.user_resetn;
    integer cycles=0,aw_count=0,w_count=0,b_count=0,ar_count=0,r_count=0;
    integer decoded=0,accepted=0,rejected=0,committed=0;
    integer aw_stall=0,w_stall=0,ar_stall=0,b_delay=-1,r_delay=-1;
    integer current_job=-1;
    reg aw_pending=0,w_pending=0,bvalid=0,rvalid=0;
    reg [3:0] awid=0,bid=0,rid=0;
    reg [1:0] bresp=0;
    reg [63:0] lease_window=0,lease_epoch=0,lease_tag=0,awaddr=0,wstrb=0;
    reg [7:0] awlen=0;
    reg [2:0] awsize=0;
    reg [1:0] awburst=0;
    reg awlock=0,wlast=0;
    reg [511:0] wdata=0,rdata=0,memory[0:2],staged=0;
    reg [63:0] staged_valid=0;
    reg [511:0] next_line;
    reg [63:0] next_bitmap;
    wire decode_accept;
    wire [6:0] decode_faults;
    wire [63:0] decoded_bitmap;
    wire [511:0] decoded_data;
    wire awready=rst_n && !aw_pending && !bvalid && b_delay<0 && cycles%3==0;
    wire wready=rst_n && !w_pending && !bvalid && b_delay<0 && cycles%5==0;
    wire arready=rst_n && !rvalid && r_delay<0 && cycles%4==0;
    ntrpu_pcie_fragment_decode dut (
        .lease_valid(1'b1),.lease_window(lease_window),.lease_epoch(lease_epoch),.lease_tag(lease_tag),
        .awaddr(awaddr),.awlen(awlen),.awsize(awsize),.awburst(awburst),.awlock(awlock),
        .wdata(wdata),.wstrb(wstrb),.wlast(wlast),.accept(decode_accept),.faults(decode_faults),
        .byte_valid(decoded_bitmap),.byte_data(decoded_data)
    );
    assign board.EP.m_axib_awready=awready;
    assign board.EP.m_axib_wready=wready;
    assign board.EP.m_axib_bid=bid;
    assign board.EP.m_axib_bresp=bresp;
    assign board.EP.m_axib_bvalid=bvalid;
    assign board.EP.m_axib_arready=arready;
    assign board.EP.m_axib_rid=rid;
    assign board.EP.m_axib_rdata=rdata;
    assign board.EP.m_axib_rresp=2'b00;
    assign board.EP.m_axib_rlast=1'b1;
    assign board.EP.m_axib_rvalid=rvalid;
    function automatic [31:0] base(input integer job);
        case(job) 0:base=0;1,3:base='h80000;2:base='hfffc0;default:base='x;endcase
    endfunction
    function automatic [511:0] expected(input integer job);
        for(integer i=0;i<64;i++) expected[i*8+:8]=8'(job*53+i*29+23);
    endfunction
    function automatic [6:0] expected_fault(input integer n);
        case(n) 6:expected_fault='h30;7:expected_fault=8;8:expected_fault='h20;
            9:expected_fault='h40;10:expected_fault=2;11:expected_fault='h10;
            15:expected_fault=4;default:expected_fault=0;endcase
    endfunction
    task prepare(input integer job);
        begin
            @(negedge clk);
            if(aw_pending || w_pending || bvalid || b_delay>=0 || aw_count!=b_count || w_count!=b_count)
                $fatal(1,"NTRPU_FRAME_CONTEXT_BUSY");
            if(job!=current_job) begin
                current_job=job;lease_window=64'h100000000+base(job);
                lease_epoch=64'h100000001+64'(job);lease_tag=64'h8000000100200300+64'(job);
                staged=0;staged_valid=0;
                $display("NTRPU_FRAME_LEASE job=%0d window=%h epoch=%h tag=%h ps=%0d",job,lease_window,lease_epoch,lease_tag,$time);
            end
        end
    endtask
    task wait_write(input integer n);
        while(b_count!=n+1) @(negedge clk);
    endtask
    task check_complete;
        begin
            if(aw_count!=16 || w_count!=16 || b_count!=16 || ar_count!=12 || r_count!=12 ||
               decoded!=16 || accepted!=9 || rejected!=7 || committed!=4 || aw_pending || w_pending || bvalid || rvalid || b_delay>=0 || r_delay>=0)
                $fatal(1,"NTRPU_FRAME_FINAL_COUNTS");
            if(memory[0]!==expected(0) || memory[1]!==expected(3) || memory[2]!==expected(2))
                $fatal(1,"NTRPU_FRAME_FINAL_MEMORY");
            $display("NTRPU_FRAME_COUNTS aw=%0d w=%0d b=%0d ar=%0d r=%0d decoded=%0d accepted=%0d rejected=%0d committed=%0d aw_stall=%0d w_stall=%0d ar_stall=%0d ps=%0d",
                aw_count,w_count,b_count,ar_count,r_count,decoded,accepted,rejected,committed,aw_stall,w_stall,ar_stall,$time);
        end
    endtask
    initial begin
        for(integer i=0;i<3;i++) memory[i]={64{8'ha5}};
        #200000000;$fatal(1,"NTRPU_FRAME_TIMEOUT");
    end
    always @(posedge clk) if(rst_n) begin
        cycles<=cycles+1;
        if(board.EP.m_axi_awvalid || board.EP.m_axi_arvalid || board.EP.m_axi_wvalid) $fatal(1,"NTRPU_FRAME_DMA_TRAFFIC");
        if(board.EP.m_axib_awvalid && !awready) aw_stall<=aw_stall+1;
        if(board.EP.m_axib_wvalid && !wready) w_stall<=w_stall+1;
        if(board.EP.m_axib_arvalid && !arready) ar_stall<=ar_stall+1;
        if(board.EP.m_axib_awvalid && awready) begin
            $display("NTRPU_FRAME_AW addr=%h id=%h len=%0d size=%0d burst=%0d lock=%b ps=%0d",board.EP.m_axib_awaddr,board.EP.m_axib_awid,board.EP.m_axib_awlen,board.EP.m_axib_awsize,board.EP.m_axib_awburst,board.EP.m_axib_awlock,$time);
            if(board.EP.m_axib_awlen!==0 || board.EP.m_axib_awsize!==6 || board.EP.m_axib_awburst!==1 || board.EP.m_axib_awlock!==0)
                $fatal(1,"NTRPU_FRAME_ACTUAL_AW_PROFILE");
            awaddr<=board.EP.m_axib_awaddr;awid<=board.EP.m_axib_awid;
            awlen<=board.EP.m_axib_awlen;awsize<=board.EP.m_axib_awsize;awburst<=board.EP.m_axib_awburst;awlock<=board.EP.m_axib_awlock;
            aw_pending<=1;aw_count<=aw_count+1;
        end
        if(board.EP.m_axib_wvalid && wready) begin
            $display("NTRPU_FRAME_W data=%h strb=%h last=%b ps=%0d",board.EP.m_axib_wdata,board.EP.m_axib_wstrb,board.EP.m_axib_wlast,$time);
            if(board.EP.m_axib_wlast!==1) $fatal(1,"NTRPU_FRAME_ACTUAL_W_PROFILE");
            wdata<=board.EP.m_axib_wdata;wstrb<=board.EP.m_axib_wstrb;wlast<=board.EP.m_axib_wlast;w_pending<=1;w_count<=w_count+1;
        end
        if(aw_pending && w_pending) begin
            $display("NTRPU_FRAME_DECODE n=%0d accept=%b faults=%h mask=%h data=%h ps=%0d",decoded,decode_accept,decode_faults,decoded_bitmap,decoded_data,$time);
            if(decoded>=16 || decode_faults!==expected_fault(decoded) || decode_accept!==(expected_fault(decoded)==0)) $fatal(1,"NTRPU_FRAME_DECISION");
            if(decode_accept) begin
                next_line=staged;next_bitmap=staged_valid | decoded_bitmap;
                for(integer i=0;i<64;i++) if(decoded_bitmap[i]) next_line[i*8+:8]=decoded_data[i*8+:8];
                staged<=next_line;staged_valid<=next_bitmap;accepted<=accepted+1;
                if(next_bitmap=='1 && staged_valid!='1) begin
                    if(next_line!==expected(current_job)) $fatal(1,"NTRPU_FRAME_ASSEMBLED_BYTES");
                    memory[current_job==3 ? 1 : current_job]<=next_line;committed<=committed+1;
                    $display("NTRPU_FRAME_COMMIT job=%0d data=%h ps=%0d",current_job,next_line,$time);
                end
            end else begin
                if(decoded_bitmap!==0) $fatal(1,"NTRPU_FRAME_REJECT_BITMAP");
                rejected<=rejected+1;
            end
            bid<=awid;bresp<=decode_accept ? 0 : 2;decoded<=decoded+1;b_delay<=5;aw_pending<=0;w_pending<=0;
        end
        if(b_delay>0) b_delay<=b_delay-1;
        if(b_delay==0) begin bvalid<=1;b_delay<=-1;end
        if(bvalid && board.EP.m_axib_bready) begin
            $display("NTRPU_FRAME_B id=%h resp=%h ps=%0d",bid,bresp,$time);bvalid<=0;b_count<=b_count+1;
        end
        if(board.EP.m_axib_arvalid && arready) begin
            $display("NTRPU_FRAME_AR addr=%h id=%h len=%0d size=%0d burst=%0d ps=%0d",board.EP.m_axib_araddr,board.EP.m_axib_arid,board.EP.m_axib_arlen,board.EP.m_axib_arsize,board.EP.m_axib_arburst,$time);
            if(board.EP.m_axib_arlen!==0 || board.EP.m_axib_arsize!==2 || board.EP.m_axib_arburst!==1 || b_count!=16 || ar_count>=12)
                $fatal(1,"NTRPU_FRAME_READ_PROFILE");
            case({board.EP.m_axib_araddr[63:6],6'b0})
                64'h100000000:rdata<=memory[0];
                64'h100080000:rdata<=memory[1];
                64'h1000fffc0:rdata<=memory[2];
                default:$fatal(1,"NTRPU_FRAME_READ_ADDRESS");
            endcase
            rid<=board.EP.m_axib_arid;ar_count<=ar_count+1;r_delay<=7;
        end
        if(r_delay>0) r_delay<=r_delay-1;
        if(r_delay==0) begin rvalid<=1;r_delay<=-1;end
        if(rvalid && board.EP.m_axib_rready) begin
            $display("NTRPU_FRAME_R id=%h data=%h resp=0 last=1 ps=%0d",rid,rdata,$time);rvalid<=0;r_count<=r_count+1;
        end
    end
endmodule
