`timescale 1ns / 1ps
module tb_ntrpu_pcie_fragment_ingress;
    reg clk=0; always #2.5 clk=~clk;
    reg rst_n=0,stop=0,source_stopped=0,ingress_isolated=0;
    reg [63:0] quiesce_epoch=0;
    reg arm_valid=0;
    wire arm_ready;
    reg [63:0] arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;
    wire [63:0] epoch,valid_bytes;
    wire lease_held,failed,transport_idle;
    wire [3:0] error_code;
    wire [31:0] rejected_frames,rejected_identity;
    wire [3:0] last_reject_code;
    wire [6:0] last_reject_faults;
    integer expected_rejections=0;
    reg [63:0] s_axi_awaddr=0;
    reg [3:0] s_axi_awid=0;
    reg [7:0] s_axi_awlen=0;
    reg [2:0] s_axi_awsize=6;
    reg [1:0] s_axi_awburst=1;
    reg s_axi_awlock=0,s_axi_awvalid=0,s_axi_wlast=0,s_axi_wvalid=0,s_axi_bready=0;
    wire s_axi_awready,s_axi_wready,s_axi_bvalid;
    reg [511:0] s_axi_wdata=0;
    reg [63:0] s_axi_wstrb=0;
    wire [3:0] s_axi_bid;
    wire [1:0] s_axi_bresp;
    wire line_valid;
    reg line_ready=0;
    wire [63:0] line_epoch,line_tag,line_address;
    wire [511:0] line_data;
    reg done_valid=0,done_error=0;
    wire done_ready;
    reg [63:0] done_epoch=0,done_tag=0;
    wire result_valid;
    reg result_ready=0;
    wire [63:0] result_epoch,result_tag;
    wire [3:0] result_status;
    integer jobs=0,lines=0,results=0,aw_count=0,w_count=0,b_count=0,cycle=0,seed=1;
    integer line_in_job=0,result_in_job=0,expected_status=0;
    integer aw_stalls=0,w_stalls=0,b_stalls=0,line_stalls=0,result_stalls=0;
    reg [511:0] expected_data;
    reg [63:0] expected_bitmap;
    reg b_stalled=0,line_stalled=0,result_stalled=0;
    reg [5:0] held_b;
    reg [703:0] held_line;
    reg [131:0] held_result;
    ntrpu_pcie_fragment_ingress dut (.*);

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
        if(!condition) $fatal(1,"FRAGMENT_CHECK job=%0d cycle=%0d %s",jobs,cycle,message);
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
        if(resp!=0) expected_rejections=expected_rejections+1;
        need(rejected_frames==expected_rejections,"B_REJECT_COUNT");
        $display("NTRPU_FRAGMENT_STATS rejected=%0d identity=%0d code=%0d faults=%h bitmap=%h failed=%b error=%0d cycle=%0d",rejected_frames,rejected_identity,last_reject_code,last_reject_faults,valid_bytes,failed,error_code,cycle);
    endtask
    task automatic tx(input reg [63:0] address,input reg [63:0] mask,input reg [511:0] data,
                      input integer order,input reg [1:0] resp,input reg [7:0] len=0,input reg [2:0] size=6);
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
        arm_epoch=epoch==0 ? 64'h100000001 : epoch+1;arm_tag=64'h8000000100001000+64'(jobs)+1;
        arm_window=64'h1000fffc0;
        arm_destination=64'h1fffffffc0;
        jobs=jobs+1;line_in_job=0;result_in_job=0;expected_status=0;
        expected_data=pattern(jobs);expected_bitmap=0;
        $display("NTRPU_FRAGMENT_CASE job=%0d name=%s",jobs,name);
        @(negedge clk);arm_valid=1;
        do @(posedge clk); while(!arm_ready);
        @(negedge clk);arm_valid=0;source_stopped=0;ingress_isolated=0;
        need(epoch==arm_epoch && lease_held && !failed && valid_bytes==0,"ARM_STATE");
    endtask
    function automatic [511:0] frame(input reg [63:0] e,input reg [63:0] t,input integer half,input reg [511:0] data);
        reg [511:0] x;
        x=0;x[63:0]=64'h313046555052544e;x[127:64]=e;x[191:128]=t;
        x[255:192]=64'(half*32);x[511:256]=data[half*256+:256];return x;
    endfunction
    task automatic fragment(input integer half,input integer order);
        tx(arm_window,'1,frame(epoch,arm_tag,half,expected_data),order,0);
        expected_bitmap=expected_bitmap | (64'hffffffff << (half*32));
        need(valid_bytes===expected_bitmap,"BYTE_BITMAP");
        if(expected_bitmap!='1) need(!line_valid && line_in_job==0,"PARTIAL_PUBLICATION");
    endtask
    task automatic reject_frame(input reg [511:0] f,input reg [63:0] mask,input reg [63:0] address,
                                input reg [1:0] resp,input integer code,input integer order);
        reg [511:0] old_data;reg [63:0] old_bitmap;reg old_failed;
        old_data=line_data;old_bitmap=valid_bytes;old_failed=failed;
        tx(address,mask,f,order,resp);
        need(line_data===old_data && valid_bytes===old_bitmap && failed===old_failed,"REJECT_PRESERVE");
        need(last_reject_code==code,"REJECT_CODE");
    endtask
    task automatic fill_line;
        fragment(0,0);fragment(1,1);
        while(!line_valid) @(negedge clk);
        need(valid_bytes=='1,"FULL_BITMAP");
    endtask
    task automatic send_line;
        while(!line_valid) @(negedge clk);
        idle(4);line_ready=1;@(posedge clk);@(negedge clk);line_ready=0;
        need(line_in_job==1 && done_ready,"LINE_HANDOFF");
    endtask
    task automatic destination_response(input bit error,input bit foreign_tag=0);
        @(negedge clk);done_epoch=epoch;done_tag=arm_tag ^ (foreign_tag ? 64'h10000 : 0);done_error=error;done_valid=1;
        do @(posedge clk); while(!done_ready);
        @(negedge clk);done_valid=0;
    endtask
    task automatic finish_job(input integer status,input integer expected_lines);
        expected_status=status;
        idle(4);need(!result_valid,"RESULT_WITHOUT_BARRIER");
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch-1;
        idle(4);need(!result_valid && !arm_ready,"STALE_BARRIER");
        quiesce_epoch=epoch;
        while(!result_valid) @(negedge clk);
        need(result_status==status && result_epoch==epoch && result_tag==arm_tag,"RESULT_ID_STATUS");
        idle(4);need(lease_held && !arm_ready,"LEASE_BEFORE_RETIRE");
        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;
        need(line_in_job==expected_lines && result_in_job==1 && transport_idle,"FINAL_COUNTS_DRAIN");
        // Retained lease still needs a valid future epoch and aligned context.
        arm_epoch=epoch;idle(1);need(!arm_ready,"REUSED_EPOCH");
        arm_epoch=epoch+1;arm_window=arm_window+1;idle(1);need(!arm_ready,"MISALIGNED_WINDOW");arm_window=arm_window-1;
        arm_destination=arm_destination+1;idle(1);need(!arm_ready,"MISALIGNED_DESTINATION");arm_destination=arm_destination-1;
        stop=0;
    endtask

    always @(posedge clk) if(rst_n) begin
        cycle=cycle+1;
        if(b_stalled) need(s_axi_bvalid && {s_axi_bid,s_axi_bresp}===held_b,"MONITOR_B_STABLE");
        if(line_stalled) need(line_valid && {line_epoch,line_tag,line_address,line_data}===held_line,"MONITOR_LINE_STABLE");
        if(result_stalled) need(result_valid && {result_epoch,result_tag,result_status}===held_result,"MONITOR_RESULT_STABLE");
        b_stalled=s_axi_bvalid && !s_axi_bready;held_b={s_axi_bid,s_axi_bresp};
        line_stalled=line_valid && !line_ready;held_line={line_epoch,line_tag,line_address,line_data};
        result_stalled=result_valid && !result_ready;held_result={result_epoch,result_tag,result_status};
        if(s_axi_awvalid && !s_axi_awready) aw_stalls=aw_stalls+1;
        if(s_axi_wvalid && !s_axi_wready) w_stalls=w_stalls+1;
        if(b_stalled) b_stalls=b_stalls+1;
        if(line_stalled) line_stalls=line_stalls+1;
        if(result_stalled) result_stalls=result_stalls+1;
        if(arm_valid && arm_ready) $display("NTRPU_FRAGMENT_ARM epoch=%h tag=%h window=%h dest=%h cycle=%0d",arm_epoch,arm_tag,arm_window,arm_destination,cycle);
        if(s_axi_awvalid && s_axi_awready) begin
            aw_count=aw_count+1;
            $display("NTRPU_FRAGMENT_AW addr=%h id=%h len=%0d size=%0d burst=%0d lock=%b cycle=%0d",s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,cycle);
        end
        if(s_axi_wvalid && s_axi_wready) begin
            w_count=w_count+1;
            $display("NTRPU_FRAGMENT_W data=%h strb=%h last=%b cycle=%0d",s_axi_wdata,s_axi_wstrb,s_axi_wlast,cycle);
        end
        if(s_axi_bvalid && s_axi_bready) begin
            b_count=b_count+1;$display("NTRPU_FRAGMENT_B id=%h resp=%0d cycle=%0d",s_axi_bid,s_axi_bresp,cycle);
        end
        if(line_valid) need(line_epoch==epoch && line_tag==arm_tag && line_address==arm_destination && line_data===expected_data && valid_bytes==64'hffffffffffffffff,"LINE_CONTENT");
        if(line_valid && line_ready) begin
            lines=lines+1;line_in_job=line_in_job+1;need(line_in_job==1,"DUPLICATE_LINE");
            $display("NTRPU_FRAGMENT_LINE epoch=%h tag=%h addr=%h data=%h cycle=%0d",line_epoch,line_tag,line_address,line_data,cycle);
        end
        if(done_valid && done_ready) $display("NTRPU_FRAGMENT_DONE epoch=%h tag=%h error=%b cycle=%0d",done_epoch,done_tag,done_error,cycle);
        if(result_valid && result_ready) begin
            results=results+1;result_in_job=result_in_job+1;
            $display("NTRPU_FRAGMENT_RESULT epoch=%h tag=%h status=%0d cycle=%0d",result_epoch,result_tag,result_status,cycle);
        end
    end

    reg [511:0] altered;
    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        idle(40);rst_n=1;idle(3);
        tx(64'h1000fffc0,'1,frame(0,0,0,pattern(0)),1,3);
        need(!lease_held && !result_valid,"UNARMED_WRITE");
        start_job("complete");fill_line();
        idle(5);need(!result_valid && !done_ready,"STAGING_IS_NOT_DONE");
        send_line();idle(4);need(!result_valid,"DESTINATION_REQUIRED");destination_response(0);finish_job(0,1);

        start_job("old_identity_and_identical_duplicate");fragment(1,1);fragment(1,0);
        reject_frame(frame(epoch-1,arm_tag-1,0,pattern(1)),'1,arm_window,2,7,2);
        reject_frame(frame(epoch,arm_tag^(64'h1<<40),0,expected_data),'1,arm_window,2,7,0);
        fragment(0,2);send_line();destination_response(0);finish_job(0,1);

        start_job("conflicting_duplicate");fragment(0,0);
        tx(arm_window,'1,frame(epoch,arm_tag,0,expected_data)^(512'b1<<256),1,2);
        need(failed && error_code==4,"CONFLICT_NOT_REJECTED");
        tx(arm_window,'1,frame(epoch,arm_tag,1,expected_data),2,2);finish_job(4,0);

        start_job("frame_local_invalids");
        reject_frame(frame(epoch,arm_tag,0,expected_data),64'h7fffffffffffffff,arm_window,2,2,0);
        reject_frame(frame(epoch,arm_tag,0,expected_data)^512'b1,'1,arm_window,2,2,1);
        altered=frame(epoch,arm_tag,0,expected_data);altered[255:192]=64;
        reject_frame(altered,'1,arm_window,2,2,2);
        reject_frame(frame(epoch,arm_tag,0,expected_data),'1,arm_window^(64'h1<<32),3,3,0);
        reject_frame(frame(epoch^(64'h1<<40),arm_tag,0,expected_data),'1,arm_window,2,7,1);
        reject_frame(frame(epoch,arm_tag^(64'h1<<40),0,expected_data),'1,arm_window,2,7,2);
        reject_frame(frame(epoch,arm_tag,0,expected_data),64'hfffffffffffffffe,arm_window,2,2,0);
        reject_frame(frame(epoch,arm_tag,0,expected_data),0,arm_window,2,2,1);
        fill_line();send_line();destination_response(0);finish_job(0,1);

        start_job("unsupported_size");tx(arm_window,'1,frame(epoch,arm_tag,0,expected_data),2,2,0,2);
        need(!failed && valid_bytes==0,"FORMAT_POISON");fill_line();send_line();destination_response(0);finish_job(0,1);
        start_job("unsupported_burst");aw(arm_window,9,1,6);w(frame(epoch,arm_tag,0,expected_data),'1,0);
        idle(5);need(!s_axi_bvalid && !transport_idle,"BURST_PREMATURE_B");
        w(frame(epoch,arm_tag,1,expected_data),'1,1);response(9,2);
        need(!failed && valid_bytes==0 && last_reject_faults==2,"BURST_STAGING");
        fill_line();send_line();destination_response(0);finish_job(0,1);
        start_job("early_last");tx(arm_window,'1,frame(epoch,arm_tag,0,expected_data),0,2,1,6);
        need(!failed && valid_bytes==0,"EARLY_LAST_STAGING");fill_line();send_line();destination_response(0);finish_job(0,1);
        start_job("late_last");aw(arm_window,7,0,6);w(frame(epoch,arm_tag,0,expected_data),'1,0);
        idle(4);need(!s_axi_bvalid,"LAST_REQUIRED");w(frame(epoch,arm_tag,1,expected_data),'1,1);response(7,2);
        need(!failed && valid_bytes==0,"LATE_LAST_STAGING");fill_line();send_line();destination_response(0);finish_job(0,1);

        start_job("stop_after_aw");aw(arm_window,5,0,6);stop=1;idle(4);
        need(!transport_idle && !arm_ready,"STOP_DROPPED_AW");w(frame(epoch,arm_tag,0,expected_data),'1,1);response(5,2);finish_job(1,0);
        start_job("stop_after_w");w(frame(epoch,arm_tag,0,expected_data),'1,1);stop=1;idle(4);
        need(!transport_idle && !arm_ready,"STOP_DROPPED_W");aw(arm_window,6,0,6);response(6,2);finish_job(1,0);
        start_job("stop_exposed_line");fill_line();stop=1;idle(7);
        need(line_valid && !result_valid,"STOP_RETRACTED_LINE");send_line();destination_response(0);finish_job(1,1);
        start_job("stop_waiting_destination");fill_line();send_line();stop=1;idle(7);
        need(done_ready && !result_valid && !arm_ready,"STOP_DROPPED_DESTINATION");destination_response(0);finish_job(1,1);
        start_job("destination_error");fill_line();send_line();destination_response(1);finish_job(5,1);
        start_job("foreign_destination_response");fill_line();send_line();destination_response(0,1);idle(4);
        need(done_ready && !result_valid,"FOREIGN_RESPONSE_COMPLETED");destination_response(0);finish_job(5,1);
        start_job("write_after_publication");fill_line();tx(arm_window,'1,frame(epoch,arm_tag,0,expected_data),0,2);
        need(failed && error_code==6,"LATE_WRITE_ACCEPTED");send_line();destination_response(0);finish_job(6,1);
        start_job("missing_half_then_stop");fragment(1,1);
        idle(7);need(!line_valid,"HOLES_PUBLISHED");stop=1;finish_job(1,0);

        start_job("queue_pressure");aw(arm_window,1,0,6);w(frame(epoch,arm_tag,0,expected_data),'1,1);
        while(!s_axi_bvalid) @(negedge clk);
        s_axi_awaddr=arm_window;s_axi_awid=2;s_axi_awlen=0;s_axi_awsize=6;s_axi_awvalid=1;
        s_axi_wstrb='1;s_axi_wdata=frame(epoch,arm_tag,1,expected_data);s_axi_wlast=1;s_axi_wvalid=1;
        idle(5);need(!s_axi_awready && !s_axi_wready,"QUEUE_OVERWRITE");response(1,0);
        @(posedge clk);need(s_axi_awready && s_axi_wready,"QUEUE_DID_NOT_RESUME");
        @(negedge clk);s_axi_awvalid=0;s_axi_wvalid=0;response(2,0);
        need(valid_bytes=='1,"QUEUE_PAIRING");send_line();destination_response(0);finish_job(0,1);

        start_job("missing_last_long_drain");aw(arm_window,15,0,6);
        for(integer j=0;j<260;j=j+1) w(frame(epoch,arm_tag,j%2,expected_data),'1,0);
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;idle(6);
        need(!result_valid && !arm_ready && !s_axi_bvalid && valid_bytes==0,"MISSING_LAST_RETIRED");
        source_stopped=0;ingress_isolated=0;w(frame(epoch,arm_tag,1,expected_data),'1,1);response(15,2);
        need(!failed && last_reject_faults==2,"DRAIN_POISON");fill_line();send_line();destination_response(0);finish_job(0,1);

        start_job("late_after_retirement");fill_line();send_line();destination_response(0);finish_job(0,1);
        source_stopped=0;ingress_isolated=0;
        tx(arm_window,'1,frame(epoch,arm_tag,0,expected_data),2,2);
        need(!failed && !result_valid && result_status==0 && error_code==0,"RETIRED_OUTCOME_CHANGED");
        start_job("stale_after_retirement_rearm");
        reject_frame(frame(epoch-1,arm_tag-1,0,pattern(19)),'1,arm_window,2,7,1);
        fill_line();send_line();destination_response(0);finish_job(0,1);
        idle(5);
        need(jobs==20 && results==20 && lines==16 && aw_count==b_count,"SUITE_COUNTS");
        need(rejected_identity==5 && rejected_frames==expected_rejections,"REJECT_FINAL_COUNTS");
        need(aw_stalls>0 && w_stalls>0 && b_stalls>0 && line_stalls>0 && result_stalls>0,"STALL_COVERAGE");
        $display("NTRPU_FRAGMENT_PASS jobs=%0d results=%0d lines=%0d aw=%0d w=%0d b=%0d rejected=%0d identity=%0d cycles=%0d aw_stall=%0d w_stall=%0d b_stall=%0d line_stall=%0d result_stall=%0d seed=%0d",
            jobs,results,lines,aw_count,w_count,b_count,rejected_frames,rejected_identity,cycle,aw_stalls,w_stalls,b_stalls,line_stalls,result_stalls,seed);
        $finish;
    end
    initial begin #500000; $fatal(1,"FRAGMENT_TIMEOUT job=%0d cycle=%0d",jobs,cycle); end
endmodule

