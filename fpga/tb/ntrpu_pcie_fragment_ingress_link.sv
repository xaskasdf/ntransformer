// Actual native XDMA -> production fragment collector. Control/line sink/readback are BFMs.
`timescale 1ps/1ps
module ntrpu_pcie_fragment_ingress_link;
    wire clk=board.EP.user_clk,rst_n=board.EP.user_resetn;
    reg stop=0,source_stopped=0,ingress_isolated=0,arm_valid=0;
    reg [63:0] quiesce_epoch=0,arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;
    wire arm_ready,lease_held,failed,transport_idle;
    wire [63:0] epoch,valid_bytes;
    wire [3:0] error_code,last_reject_code;
    wire [6:0] last_reject_faults;
    wire [31:0] rejected_frames,rejected_identity;
    wire line_valid,done_ready,result_valid;
    reg line_ready=0,done_valid=0,result_ready=0;
    reg [63:0] done_epoch=0,done_tag=0;
    wire [63:0] line_epoch,line_tag,line_address,result_epoch,result_tag;
    wire [511:0] line_data;
    wire [3:0] result_status;
    integer cycles=0,aw_count=0,w_count=0,b_count=0,ar_count=0,r_count=0;
    integer decoded=0,accepted=0,rejected=0,committed=0,results=0,arms=0,dones=0;
    integer aw_stall=0,w_stall=0,ar_stall=0,line_stall=0,done_delay=-1,r_delay=-1;
    integer current_job=-1;
    reg rvalid=0;
    reg [3:0] rid=0;
    reg [511:0] rdata=0,memory[0:2];
    wire aw_pending=dut.aw_held,w_pending=dut.w_held;
    wire [63:0] lease_window=dut.window_line,lease_epoch=epoch,lease_tag=dut.tag;
    wire decode_accept=dut.decode_accept;
    wire [6:0] decode_faults=dut.decode_faults;
    wire [63:0] decoded_bitmap=dut.decoded_valid;
    wire [511:0] decoded_data=dut.decoded_data,staged=dut.payload;
    wire [63:0] staged_valid=valid_bytes;
    wire arready=rst_n && !rvalid && r_delay<0 && cycles%4==0;
    ntrpu_pcie_fragment_ingress dut (
        .clk(clk),.rst_n(rst_n),.stop(stop),.source_stopped(source_stopped),.ingress_isolated(ingress_isolated),.quiesce_epoch(quiesce_epoch),
        .arm_valid(arm_valid),.arm_ready(arm_ready),.arm_epoch(arm_epoch),.arm_tag(arm_tag),.arm_window(arm_window),.arm_destination(arm_destination),
        .epoch(epoch),.lease_held(lease_held),.failed(failed),.error_code(error_code),.transport_idle(transport_idle),.valid_bytes(valid_bytes),
        .rejected_frames(rejected_frames),.rejected_identity(rejected_identity),.last_reject_code(last_reject_code),.last_reject_faults(last_reject_faults),
        .s_axi_awaddr(board.EP.m_axib_awaddr),.s_axi_awid(board.EP.m_axib_awid),.s_axi_awlen(board.EP.m_axib_awlen),.s_axi_awsize(board.EP.m_axib_awsize),
        .s_axi_awburst(board.EP.m_axib_awburst),.s_axi_awlock(board.EP.m_axib_awlock),.s_axi_awvalid(board.EP.m_axib_awvalid),.s_axi_awready(board.EP.m_axib_awready),
        .s_axi_wdata(board.EP.m_axib_wdata),.s_axi_wstrb(board.EP.m_axib_wstrb),.s_axi_wlast(board.EP.m_axib_wlast),.s_axi_wvalid(board.EP.m_axib_wvalid),.s_axi_wready(board.EP.m_axib_wready),
        .s_axi_bid(board.EP.m_axib_bid),.s_axi_bresp(board.EP.m_axib_bresp),.s_axi_bvalid(board.EP.m_axib_bvalid),.s_axi_bready(board.EP.m_axib_bready),
        .line_valid(line_valid),.line_ready(line_ready),.line_epoch(line_epoch),.line_tag(line_tag),.line_address(line_address),.line_data(line_data),
        .done_valid(done_valid),.done_ready(done_ready),.done_epoch(done_epoch),.done_tag(done_tag),.done_error(1'b0),
        .result_valid(result_valid),.result_ready(result_ready),.result_epoch(result_epoch),.result_tag(result_tag),.result_status(result_status)
    );
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
    task finish_lease;
        begin
            if(current_job>=0 && results!=arms) begin
                while(!transport_idle || !done_ready && dones!=arms) @(negedge clk);
                source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
                while(!result_valid) @(negedge clk);
                if(result_status!==0 || result_epoch!==epoch || result_tag!==arm_tag || failed)
                    $fatal(1,"NTRPU_COLLECTOR_RESULT_IDENTITY");
                repeat(4) @(negedge clk);
                result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;
                if(results!=arms || !transport_idle) $fatal(1,"NTRPU_COLLECTOR_RESULT_DRAIN");
            end
        end
    endtask
    task prepare(input integer job);
        begin
            @(negedge clk);
            if(job!=current_job) begin
                finish_lease;
                source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
                arm_epoch=64'h100000001+64'(job);arm_tag=64'h8000000100200300+64'(job);
                arm_window=64'h100000000+base(job);arm_destination=64'h100000000+base(job);
                arm_valid=1;do @(posedge clk);while(!arm_ready);
                @(negedge clk);arm_valid=0;current_job=job;source_stopped=0;ingress_isolated=0;
                if(!lease_held || failed || valid_bytes!=0 || !transport_idle) $fatal(1,"NTRPU_COLLECTOR_ARM_STATE");
                $display("NTRPU_FRAME_LEASE job=%0d window=%h epoch=%h tag=%h ps=%0d",job,lease_window,lease_epoch,lease_tag,$time);
            end
        end
    endtask
    task wait_write(input integer n);
        while(b_count!=n+1) @(negedge clk);
    endtask
    task check_complete;
        begin
            finish_lease;
            if(aw_count!=16 || w_count!=16 || b_count!=16 || ar_count!=12 || r_count!=12 ||
               decoded!=16 || accepted!=9 || rejected!=7 || committed!=4 || results!=4 || arms!=4 || dones!=4 ||
               rejected_frames!=7 || rejected_identity!=3 || !transport_idle || line_valid || result_valid || done_valid || rvalid || r_delay>=0)
                $fatal(1,"NTRPU_COLLECTOR_FINAL_COUNTS");
            if(memory[0]!==expected(0) || memory[1]!==expected(3) || memory[2]!==expected(2))
                $fatal(1,"NTRPU_COLLECTOR_FINAL_MEMORY");
            $display("NTRPU_FRAME_COUNTS aw=%0d w=%0d b=%0d ar=%0d r=%0d decoded=%0d accepted=%0d rejected=%0d committed=%0d aw_stall=%0d w_stall=%0d ar_stall=%0d ps=%0d",
                aw_count,w_count,b_count,ar_count,r_count,decoded,accepted,rejected,committed,aw_stall,w_stall,ar_stall,$time);
            $display("NTRPU_COLLECTOR_PASS arms=%0d lines=%0d done=%0d results=%0d rejected=%0d identity=%0d line_stall=%0d ps=%0d",arms,committed,dones,results,rejected_frames,rejected_identity,line_stall,$time);
        end
    endtask
    initial begin
        for(integer i=0;i<3;i++) memory[i]={64{8'ha5}};
        #200000000;$fatal(1,"NTRPU_COLLECTOR_TIMEOUT");
    end
    always @(negedge clk) if(rst_n) line_ready=cycles%7==0 && done_delay<0 && !done_valid;
    always @(posedge clk) if(rst_n) begin
        cycles<=cycles+1;
        if(failed || error_code!=0 || board.EP.m_axi_awvalid || board.EP.m_axi_arvalid || board.EP.m_axi_wvalid) $fatal(1,"NTRPU_COLLECTOR_LEASE_OR_DMA");
        if(arm_valid && arm_ready) begin arms<=arms+1;$display("NTRPU_COLLECTOR_ARM epoch=%h tag=%h window=%h dest=%h ps=%0d",arm_epoch,arm_tag,arm_window,arm_destination,$time);end
        if(board.EP.m_axib_awvalid && !board.EP.m_axib_awready) aw_stall<=aw_stall+1;
        if(board.EP.m_axib_wvalid && !board.EP.m_axib_wready) w_stall<=w_stall+1;
        if(board.EP.m_axib_arvalid && !arready) ar_stall<=ar_stall+1;
        if(line_valid && !line_ready) line_stall<=line_stall+1;
        if(board.EP.m_axib_awvalid && board.EP.m_axib_awready) begin
            aw_count<=aw_count+1;$display("NTRPU_FRAME_AW addr=%h id=%h len=%0d size=%0d burst=%0d lock=%b ps=%0d",board.EP.m_axib_awaddr,board.EP.m_axib_awid,board.EP.m_axib_awlen,board.EP.m_axib_awsize,board.EP.m_axib_awburst,board.EP.m_axib_awlock,$time);
        end
        if(board.EP.m_axib_wvalid && board.EP.m_axib_wready) begin
            w_count<=w_count+1;$display("NTRPU_FRAME_W data=%h strb=%h last=%b ps=%0d",board.EP.m_axib_wdata,board.EP.m_axib_wstrb,board.EP.m_axib_wlast,$time);
        end
        if(aw_pending && w_pending) begin
            $display("NTRPU_FRAME_DECODE n=%0d accept=%b faults=%h mask=%h data=%h ps=%0d",decoded,decode_accept,decode_faults,decoded_bitmap,decoded_data,$time);
            if(decoded>=16 || decode_faults!==expected_fault(decoded) || decode_accept!==(expected_fault(decoded)==0)) $fatal(1,"NTRPU_COLLECTOR_DECISION");
            decoded<=decoded+1;if(decode_accept) accepted<=accepted+1;else rejected<=rejected+1;
        end
        if(board.EP.m_axib_bvalid && board.EP.m_axib_bready) begin
            $display("NTRPU_FRAME_B id=%h resp=%h ps=%0d",board.EP.m_axib_bid,board.EP.m_axib_bresp,$time);b_count<=b_count+1;
        end
        if(line_valid && line_ready) begin
            if(line_data!==expected(current_job) || line_epoch!==epoch || line_tag!==arm_tag || line_address!==arm_destination || valid_bytes!='1)
                $fatal(1,"NTRPU_COLLECTOR_LINE_CONTENT");
            memory[current_job==3 ? 1 : current_job]<=line_data;committed<=committed+1;
            done_epoch<=line_epoch;done_tag<=line_tag;done_delay<=9;
            $display("NTRPU_FRAME_COMMIT job=%0d data=%h ps=%0d",current_job,line_data,$time);
            $display("NTRPU_COLLECTOR_LINE epoch=%h tag=%h addr=%h data=%h ps=%0d",line_epoch,line_tag,line_address,line_data,$time);
        end
        if(done_delay>0) done_delay<=done_delay-1;
        if(done_delay==0) begin done_valid<=1;done_delay<=-1;end
        if(done_valid && done_ready) begin done_valid<=0;dones<=dones+1;$display("NTRPU_COLLECTOR_DONE epoch=%h tag=%h error=0 ps=%0d",done_epoch,done_tag,$time);end
        if(result_valid && result_ready) begin results<=results+1;$display("NTRPU_COLLECTOR_RESULT epoch=%h tag=%h status=%0d ps=%0d",result_epoch,result_tag,result_status,$time);end
        if(board.EP.m_axib_arvalid && arready) begin
            $display("NTRPU_FRAME_AR addr=%h id=%h len=%0d size=%0d burst=%0d ps=%0d",board.EP.m_axib_araddr,board.EP.m_axib_arid,board.EP.m_axib_arlen,board.EP.m_axib_arsize,board.EP.m_axib_arburst,$time);
            if(board.EP.m_axib_arlen!==0 || board.EP.m_axib_arsize!==2 || board.EP.m_axib_arburst!==1 || b_count!=16 || committed!=4 || ar_count>=12)
                $fatal(1,"NTRPU_COLLECTOR_READ_PROFILE");
            case({board.EP.m_axib_araddr[63:6],6'b0})
                64'h100000000:rdata<=memory[0];64'h100080000:rdata<=memory[1];64'h1000fffc0:rdata<=memory[2];
                default:$fatal(1,"NTRPU_COLLECTOR_READ_ADDRESS");
            endcase
            rid<=board.EP.m_axib_arid;ar_count<=ar_count+1;r_delay<=7;
        end
        if(r_delay>0) r_delay<=r_delay-1;
        if(r_delay==0) begin rvalid<=1;r_delay<=-1;end
        if(rvalid && board.EP.m_axib_rready) begin $display("NTRPU_FRAME_R id=%h data=%h resp=0 last=1 ps=%0d",rid,rdata,$time);rvalid<=0;r_count<=r_count+1;end
    end
endmodule
