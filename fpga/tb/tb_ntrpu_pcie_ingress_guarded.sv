`timescale 1ns / 1ps
module tb_ntrpu_pcie_ingress_guarded;
    reg clk=0; always #2.5 clk=~clk;
    reg rst_n=0,stop=0,source_stopped=0,ingress_isolated=0;
    reg [63:0] quiesce_epoch=0;
    reg arm_valid=0;
    wire arm_ready;
    reg [63:0] arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;
    wire [63:0] epoch,valid_bytes;
    wire lease_held,failed,transport_idle;
    wire [3:0] error_code;
    reg [63:0] s_axi_awaddr=0;
    reg [3:0] s_axi_awid=0;
    reg [7:0] s_axi_awlen=0;
    reg [2:0] s_axi_awsize=2;
    reg [1:0] s_axi_awburst=1;
    reg s_axi_awlock=0,s_axi_awvalid=0,s_axi_wlast=0,s_axi_wvalid=0,s_axi_bready=0;
    wire s_axi_awready,s_axi_wready,s_axi_bvalid;
    reg [511:0] s_axi_wdata=0;
    reg [63:0] s_axi_wstrb=0;
    wire [3:0] s_axi_bid;
    wire [1:0] s_axi_bresp;

    wire result_valid;
    reg result_ready=0;
    wire [63:0] result_epoch,result_tag,peer_epoch;
    wire [3:0] result_status;
    wire peer_quarantined,peer_axi_busy;
    wire [2:0] bridge_phase,destination_status;
    wire [36:0] m_axi_awaddr;
    wire [63:0] m_axi_wstrb;
    wire guard_idle;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize,m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awcache,m_axi_awid;
    wire m_axi_awlock,m_axi_awvalid,m_axi_wlast,m_axi_wvalid,m_axi_bready;
    wire [511:0] m_axi_wdata;
    reg m_axi_awready=0,m_axi_wready=0,m_axi_bvalid=0;
    reg [3:0] m_axi_bid=0;
    reg [1:0] m_axi_bresp=0;
    integer jobs=0,results=0,aw_count=0,w_count=0,b_count=0,cycle=0,seed=1;
    integer down_aw=0,down_w=0,down_b=0,total_aw=0,total_w=0,total_b=0;
    integer result_in_job=0,expected_status=0;
    integer aw_stalls=0,w_stalls=0,b_stalls=0,result_stalls=0,foreign_b=0,held_cycles=0;
    reg [511:0] expected_data;
    reg [63:0] expected_bitmap;
    reg aw_stalled=0,w_stalled=0,b_stalled=0,result_stalled=0;
    reg [88:0] held_aw;
    reg [576:0] held_w;
    reg [5:0] held_b;
    reg [131:0] held_result;
    reg [2:0] previous_phase=0;
    ntrpu_pcie_ingress_guarded dut (.*);
    function automatic [511:0] pattern(input integer job);
        reg [511:0] x;
        for(integer j=0;j<64;j=j+1) x[j*8+:8]=8'(job*37+j*19+seed*7);
        return x;
    endfunction
    function automatic integer lowest(input reg [3:0] mask);
        for(integer j=0;j<4;j=j+1) if(mask[j]) return j;
        return 0;
    endfunction
    task automatic idle(input integer count);
        repeat(count) @(negedge clk);
    endtask
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"GUARDED_CHECK job=%0d cycle=%0d %s",jobs,cycle,message);
    endtask
    task automatic aw(input reg [63:0] address,input reg [3:0] id,input reg [7:0] len,input reg [2:0] size);
        @(negedge clk);s_axi_awaddr=address;s_axi_awid=id;s_axi_awlen=len;s_axi_awsize=size;s_axi_awvalid=1;
        do @(posedge clk); while(!s_axi_awready);
        @(negedge clk);s_axi_awvalid=0;
    endtask
    task automatic w(input reg [511:0] data,input reg [63:0] mask,input bit last);
        @(negedge clk);s_axi_wdata=data;s_axi_wstrb=mask;s_axi_wlast=last;s_axi_wvalid=1;
        do @(posedge clk); while(!s_axi_wready);
        @(negedge clk);s_axi_wvalid=0;
    endtask
    task automatic response(input reg [3:0] id,input reg [1:0] resp);
        while(!s_axi_bvalid) @(negedge clk);
        need(s_axi_bid===id && s_axi_bresp===resp,"B_ID_OR_RESPONSE");
        idle(3+(seed%3));
        need(s_axi_bvalid && s_axi_bid===id && s_axi_bresp===resp,"B_STABILITY");
        s_axi_bready=1;@(posedge clk);@(negedge clk);s_axi_bready=0;
    endtask
    task automatic tx(input reg [63:0] address,input reg [63:0] mask,input reg [511:0] data,
                      input integer order,input reg [1:0] resp,input reg [7:0] len=0,input reg [2:0] size=2);
        reg [3:0] id;
        integer prior_b;
        id=4'((b_count+3)%16);prior_b=b_count;
        fork
            begin if(order==1) idle(4);aw(address,id,len,size);end
            begin if(order==0) idle(4);w(data,mask,1);end
        join
        response(id,resp);
        need(b_count==prior_b+1,"EXACTLY_ONE_B");
    endtask
    task automatic start_job(input string name);
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        arm_epoch=epoch+1;arm_tag=64'h1000+jobs+1;
        arm_window=(jobs%2==0) ? 64'h1000fffc0 : 64'h100000000;
        case(jobs+1)
            13: arm_destination=64'h2000000000; // first invalid 128GiB line
            14: arm_destination=64'h1000000000000000; // truncation alias to zero
            18,21: arm_destination=0;
            19: arm_destination=64'h100000040; // above 4GiB
            20: arm_destination=64'hffffffffffffffc0; // 65-bit end carry
            default: arm_destination=64'h1fffffffc0;
        endcase
        jobs=jobs+1;down_aw=0;down_w=0;down_b=0;result_in_job=0;expected_status=0;
        expected_data=pattern(jobs);expected_bitmap=0;
        $display("NTRPU_GUARDED_CASE job=%0d name=%s",jobs,name);
        @(negedge clk);arm_valid=1;
        do @(posedge clk); while(!arm_ready);
        @(negedge clk);arm_valid=0;source_stopped=0;ingress_isolated=0;
        need(epoch==arm_epoch && peer_epoch==arm_epoch && lease_held && !failed && valid_bytes==0,"ARM_STATE");
    endtask
    task automatic fragment(input integer word_index,input reg [3:0] mask,input integer order);
        reg [63:0] strobes;
        strobes={60'b0,mask}<<(word_index*4);
        tx(arm_window+word_index*4+lowest(mask),strobes,expected_data,order,0);
        // Byte reference is independent of the DUT's decoder and merge loop.
        for(integer j=0;j<4;j=j+1) if(mask[j]) expected_bitmap[word_index*4+j]=1;
        need(valid_bytes===expected_bitmap,"BYTE_BITMAP");
        if(expected_bitmap!=64'hffffffffffffffff) need(!m_axi_awvalid && !m_axi_wvalid && down_aw==0 && down_w==0,"PARTIAL_PUBLICATION");
    endtask
    task automatic fill_line;
        for(integer j=0;j<16;j=j+1) fragment(j,15,(j+seed)%3);
        need(valid_bytes==64'hffffffffffffffff,"FULL_BITMAP");
    endtask

    task automatic destination_channels;
        fork
            begin
                while(!m_axi_awvalid) @(negedge clk);
                idle(3+(seed%4));m_axi_awready=1;
                do @(posedge clk); while(!m_axi_awvalid);
                @(negedge clk);m_axi_awready=0;
            end
            begin
                while(!m_axi_wvalid) @(negedge clk);
                idle(7-(seed%4));m_axi_wready=1;
                do @(posedge clk); while(!m_axi_wvalid);
                @(negedge clk);m_axi_wready=0;
            end
        join
        need(down_aw==1 && down_w==1 && peer_axi_busy,"DESTINATION_CHANNEL_COUNTS");
    endtask
    task automatic destination_b(input reg [3:0] id,input reg [1:0] resp);
        @(negedge clk);m_axi_bid=id;m_axi_bresp=resp;m_axi_bvalid=1;
        do @(posedge clk); while(!m_axi_bready);
        @(negedge clk);m_axi_bvalid=0;
    endtask
    task automatic destination(input reg [1:0] resp);
        destination_channels();idle(5+seed%5);destination_b(0,resp);
    endtask
    task automatic stop_at(input integer phase);
        while(bridge_phase!=phase) @(negedge clk);
        $display("NTRPU_GUARDED_STOP phase=%0d cycle=%0d",phase,cycle);
        stop=1;
    endtask
    task automatic hold_outstanding(input integer count);
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        arm_epoch=epoch+1;arm_valid=1;
        repeat(count) begin
            @(negedge clk);
            need(peer_axi_busy && !guard_idle && !result_valid && !arm_ready && lease_held,"PREMATURE_RETIRE_OR_REARM");
            need(epoch==peer_epoch && epoch==jobs,"PARTIAL_REARM");
            held_cycles=held_cycles+1;
            $display("NTRPU_GUARDED_HOLD busy=%b idle=%b result=%b arm=%b epoch=%h peer=%h cycle=%0d",peer_axi_busy,guard_idle,result_valid,arm_ready,epoch,peer_epoch,cycle);
        end
        arm_valid=0;source_stopped=0;ingress_isolated=0;
    endtask
    task automatic finish_job(input integer status,input integer writes);
        expected_status=status;
        idle(8);need(!result_valid,"RESULT_WITHOUT_BARRIER");
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch-1;
        idle(5);need(!result_valid && !arm_ready,"STALE_BARRIER");
        quiesce_epoch=epoch;
        while(!result_valid) @(negedge clk);
        need(result_status==status && result_epoch==epoch && result_tag==arm_tag,"RESULT_ID_STATUS");
        idle(5);need(lease_held && !arm_ready && !peer_axi_busy && guard_idle,"LEASE_BEFORE_RETIRE");
        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;
        need(down_aw==writes && down_w==writes && down_b==writes && result_in_job==1 && transport_idle,"FINAL_COUNTS_DRAIN");
        // Actively offer invalid rearm requests; neither engine may advance.
        stop=0;arm_valid=1;arm_epoch=epoch;idle(4);need(!arm_ready && epoch==peer_epoch && epoch==jobs,"REUSED_EPOCH");
        arm_epoch=epoch+1;quiesce_epoch=epoch-1;idle(4);need(!arm_ready && epoch==jobs && peer_epoch==jobs,"STALE_REARM");
        quiesce_epoch=epoch;arm_window=arm_window+1;idle(4);need(!arm_ready && epoch==jobs && peer_epoch==jobs,"MISALIGNED_WINDOW");arm_valid=0;arm_window=arm_window-1;
        arm_destination=arm_destination+1;arm_valid=1;idle(4);need(!arm_ready && epoch==jobs && peer_epoch==jobs,"MISALIGNED_DESTINATION");arm_valid=0;arm_destination=arm_destination-1;
    endtask
    always @(posedge clk) if(rst_n) begin
        cycle=cycle+1;
        need(epoch===peer_epoch,"EPOCH_DIVERGENCE");
        need(!(result_valid && (peer_axi_busy || !guard_idle)),"RESULT_BEFORE_AXI_DRAIN");
        if(m_axi_awvalid || m_axi_wvalid) need(!(jobs==13 || jobs==14 || jobs==20),"REJECTED_ADDRESS_REACHED_MEMORY");
        if(aw_stalled) need(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awcache,m_axi_awid,m_axi_awprot,m_axi_awlock}===held_aw,"MONITOR_AW_STABLE");
        if(w_stalled) need(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===held_w,"MONITOR_W_STABLE");
        if(b_stalled) need(s_axi_bvalid && {s_axi_bid,s_axi_bresp}===held_b,"MONITOR_B_STABLE");
        if(result_stalled) need(result_valid && {result_epoch,result_tag,result_status}===held_result,"MONITOR_RESULT_STABLE");
        aw_stalled=m_axi_awvalid && !m_axi_awready;held_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awcache,m_axi_awid,m_axi_awprot,m_axi_awlock};
        w_stalled=m_axi_wvalid && !m_axi_wready;held_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
        b_stalled=s_axi_bvalid && !s_axi_bready;held_b={s_axi_bid,s_axi_bresp};
        result_stalled=result_valid && !result_ready;held_result={result_epoch,result_tag,result_status};
        if(aw_stalled) aw_stalls=aw_stalls+1;
        if(w_stalled) w_stalls=w_stalls+1;
        if(b_stalled) b_stalls=b_stalls+1;
        if(result_stalled) result_stalls=result_stalls+1;
        if(bridge_phase!=previous_phase) begin
            $display("NTRPU_GUARDED_PHASE phase=%0d busy=%b idle=%b status=%0d cycle=%0d",bridge_phase,peer_axi_busy,guard_idle,destination_status,cycle);previous_phase=bridge_phase;
        end
        if(arm_valid && arm_ready) $display("NTRPU_GUARDED_ARM epoch=%h tag=%h window=%h dest=%h cycle=%0d",arm_epoch,arm_tag,arm_window,arm_destination,cycle);
        if(s_axi_awvalid && s_axi_awready) begin
            aw_count=aw_count+1;
            $display("NTRPU_GUARDED_AW addr=%h id=%h len=%0d size=%0d cycle=%0d",s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,cycle);
        end
        if(s_axi_wvalid && s_axi_wready) begin
            w_count=w_count+1;
            $display("NTRPU_GUARDED_W data=%h strb=%h last=%b cycle=%0d",s_axi_wdata,s_axi_wstrb,s_axi_wlast,cycle);
        end
        if(s_axi_bvalid && s_axi_bready) begin
            b_count=b_count+1;$display("NTRPU_GUARDED_B id=%h resp=%0d cycle=%0d",s_axi_bid,s_axi_bresp,cycle);
        end
        if(m_axi_awvalid) need(m_axi_awaddr===arm_destination && m_axi_awid==0 && m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awcache==0 && m_axi_awprot==0 && !m_axi_awlock,"DESTINATION_AW_FIELDS");
        if(m_axi_wvalid) need(m_axi_wdata===expected_data && m_axi_wstrb==64'hffffffffffffffff && m_axi_wlast && valid_bytes==64'hffffffffffffffff,"DESTINATION_DATA");
        if(m_axi_awvalid && m_axi_awready) begin
            down_aw=down_aw+1;total_aw=total_aw+1;need(down_aw==1,"DUPLICATE_AW");
            $display("NTRPU_GUARDED_MAW addr=%h id=%h len=%0d size=%0d cycle=%0d",m_axi_awaddr,m_axi_awid,m_axi_awlen,m_axi_awsize,cycle);
        end
        if(m_axi_wvalid && m_axi_wready) begin
            down_w=down_w+1;total_w=total_w+1;need(down_w==1,"DUPLICATE_W");
            $display("NTRPU_GUARDED_MW data=%h strb=%h last=%b cycle=%0d",m_axi_wdata,m_axi_wstrb,m_axi_wlast,cycle);
        end
        if(m_axi_bvalid && m_axi_bready) begin
            need(down_aw==1 && down_w==1,"B_BEFORE_CHANNELS");
            if(m_axi_bid==0) begin down_b=down_b+1;total_b=total_b+1;need(down_b==1,"DUPLICATE_FINAL_B");end
            else foreign_b=foreign_b+1;
            $display("NTRPU_GUARDED_MB id=%h resp=%0d cycle=%0d",m_axi_bid,m_axi_bresp,cycle);
        end
        if(result_valid && result_ready) begin
            results=results+1;result_in_job=result_in_job+1;
            $display("NTRPU_GUARDED_RESULT epoch=%h tag=%h status=%0d cycle=%0d",result_epoch,result_tag,result_status,cycle);
        end
    end
    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        idle(40);rst_n=1;idle(3);
        start_job("complete");fill_line();destination(0);finish_job(0,1);
        start_job("sparse_and_identical_duplicate");
        for(integer j=0;j<16;j=j+1) begin
            fragment(j,5,j%3);if(j==0) fragment(j,5,1);fragment(j,10,(j+1)%3);
        end
        destination(0);finish_job(0,1);
        start_job("conflicting_duplicate");fragment(0,15,0);
        tx(arm_window,15,expected_data ^ 512'b1,1,2);finish_job(4,0);
        start_job("holes_stop");for(integer j=0;j<16;j=j+1) fragment(j,1,j%3);
        idle(5);stop=1;finish_job(1,0);
        start_job("upper_address_rejection");tx(arm_window ^ 64'h100000000,15,expected_data,2,3);finish_job(3,0);
        for(integer phase=1;phase<=5;phase=phase+1) begin
            start_job($sformatf("stop_phase_%0d",phase));
            fork fill_line();stop_at(phase);destination(0);join
            finish_job(1,1);
        end
        start_job("destination_slverr");fill_line();destination(2);finish_job(5,1);
        start_job("destination_decerr");fill_line();destination(3);finish_job(5,1);
        start_job("first_invalid_destination");fill_line();finish_job(5,0);
        start_job("high_bit_destination_alias");fill_line();finish_job(5,0);
        start_job("missing_final_b");fill_line();destination_channels();hold_outstanding(100);
        need(bridge_phase==3,"MISSING_B_PHASE");destination_b(0,0);finish_job(0,1);
        start_job("recovery_without_reset");fill_line();destination(0);finish_job(0,1);
        start_job("stop_partial_aw");aw(arm_window,5,0,2);stop=1;idle(5);
        need(!transport_idle && !arm_ready,"STOP_DROPPED_AW");w(expected_data,15,1);response(5,2);finish_job(1,0);
        start_job("zero_destination");fill_line();destination(0);finish_job(0,1);
        start_job("above_4gib_destination");fill_line();destination(0);finish_job(0,1);
        start_job("destination_end_overflow");fill_line();finish_job(5,0);
        start_job("zero_recovery_after_rejections");fill_line();destination(0);finish_job(0,1);
        idle(5);
        need(jobs==21 && results==21 && total_aw==14 && total_w==14 && total_b==14 && foreign_b==0 && aw_count==b_count && w_count==aw_count,"SUITE_COUNTS");
        need(aw_stalls>0 && w_stalls>0 && b_stalls>0 && result_stalls>0 && held_cycles==100,"STALL_COVERAGE");
        $display("NTRPU_GUARDED_PASS jobs=%0d results=%0d writes=%0d foreign=%0d aw=%0d w=%0d b=%0d cycles=%0d aw_stall=%0d w_stall=%0d b_stall=%0d result_stall=%0d held=%0d seed=%0d",
                 jobs,results,total_aw,foreign_b,aw_count,w_count,b_count,cycle,aw_stalls,w_stalls,b_stalls,result_stalls,held_cycles,seed);
        $finish;
    end
    initial begin #500000; $fatal(1,"GUARDED_TIMEOUT job=%0d phase=%0d busy=%b cycle=%0d",jobs,bridge_phase,peer_axi_busy,cycle); end
endmodule
