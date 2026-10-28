`timescale 1ns/1ps
`ifdef XILINX_SIMULATOR
module short(in1,in1); inout in1; endmodule
`endif
module tb_ntrpu_multibuffer_mig;
    import arch_package::*;
    wire clk, rst_n, source_reset_n;
    reg source_clk=0, mgmt_clk=0;
    initial begin #0.7; forever #2.5 source_clk=~source_clk; end
    initial begin #1.1; forever #5 mgmt_clk=~mgmt_clk; end
    reg external_reset_async=1, rearm=0, stop=0;
    initial begin #40; external_reset_async=0; end
    wire admit,running,rearm_ready,restart_required,peers_reset_req,host_alive,device_alive;
    reg cmd_valid=0,data_valid=0,cpl_ready=0;
    wire cmd_ready,data_ready,cpl_valid,fault,drained,inactive_port_idle;
    reg [63:0] cmd_addr=0,cmd_tag=0;
    reg [7:0] cmd_len=0;
    reg [511:0] data=0;
    wire [63:0] cpl_tag;
    wire [1:0] cpl_status;
    wire [7:0] banks_used;
    wire [9:0] pending_counts;
    wire m_axi_awvalid,m_axi_awready,m_axi_wvalid,m_axi_wready,m_axi_wlast,m_axi_bvalid,m_axi_bready;
    wire [36:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [3:0] m_axi_awid,m_axi_awcache,m_axi_bid;
    wire [2:0] m_axi_awsize,m_axi_awprot;
    wire [1:0] m_axi_awburst,m_axi_bresp;
    wire m_axi_awlock;
    wire [511:0] m_axi_wdata;
    wire [63:0] m_axi_wstrb;
    reg [36:0] m_axi_araddr=0;
    wire [7:0] m_axi_arlen=0;
    wire [2:0] m_axi_arsize=6,m_axi_arprot=0;
    wire [1:0] m_axi_arburst=1;
    wire [3:0] m_axi_arid=0,m_axi_arcache=0;
    wire m_axi_arlock=0;
    wire m_axi_arready,m_axi_rvalid,m_axi_rlast;
    reg m_axi_arvalid=0,m_axi_rready=0;
    wire [511:0] m_axi_rdata;
    wire [1:0] m_axi_rresp;
    wire [3:0] m_axi_rid;
    reg hold_w=0,hold_b=0,hold_cpl=0;
    wire mig_wready,mig_bvalid;
    assign m_axi_wready=mig_wready && !hold_w;
    assign m_axi_bvalid=mig_bvalid && !hold_b;
    ntrpu_multibuffer_mig_cold dut (.*,
        .ui_clk(clk), .memory_reset_n(rst_n));
    `include "ntrpu_multibuffer_mig_model.vh"

    integer source_cycle=0,accepted=0,completed=0,writes=0,reads=0;
    integer aw_count=0,b_count=0,peak=0,blocked=0,fill=0,fill_beat=0;
    integer aw_cursor=0,w_cursor=0,b_cursor=0,w_beat=0,valid_jobs=0;
    integer memory_queue[0:63],lengths[0:63],statuses[0:63];
    reg [63:0] addresses[0:63];
    bit finished[0:63];
    longint aw_time[0:63],w_time[0:63],b_time[0:63];
    reg [63:0] golden_addr[0:255];
    reg [511:0] golden_data[0:255];
    integer golden_count=0;
    reg aw_held=0,w_held=0,cpl_held=0,r_held=0;
    reg [65:0] saved_cpl;
    reg [44:0] saved_aw;
    reg [576:0] saved_w;
    reg [518:0] saved_r;
    function automatic longint ps; return longint'($realtime/1ps); endfunction
    function automatic [511:0] pattern(input integer tag,beat);
        for(integer lane=0;lane<64;lane++) pattern[lane*8+:8]=8'(tag*37+beat*13+lane*19);
    endfunction
    task automatic need(input bit condition,input string message);
        if(!condition) $fatal(1,"MULTIBUFFER_MIG_CHECK %s",message);
    endtask
    task automatic wait_source(input integer n);
        repeat(n) @(negedge source_clk);
    endtask
    always @(negedge source_clk)
        cpl_ready=admit && !hold_cpl && source_cycle%7!=0;
    always @(posedge source_clk) begin
        source_cycle++;
        if(!admit) begin
            blocked++;
            need(!cmd_ready && !cpl_valid,"CLOSED_ADMISSION");
        end
        if(source_reset_n) begin
            need(inactive_port_idle,"RESERVED_PORT_ACTIVITY");
            if(cmd_valid && cmd_ready) begin
                need(admit && !stop && !fault && calibrated && !ui_rst,"COMMAND_ADMISSION");
                need(cmd_tag==accepted+1 && fill==0,"COMMAND_ORDER");
                accepted++;fill=accepted;fill_beat=0;
                addresses[accepted]=cmd_addr;lengths[accepted]=cmd_len+1;
                statuses[accepted]=({1'b0,cmd_addr}+65'((cmd_len+1)*64)>65'd137438953472) ? 3:0;
                if(statuses[accepted]==0) begin memory_queue[valid_jobs]=accepted;valid_jobs++;end
                $display("COLD CMD %0d %0d %h %0d %0d",ps(),accepted,cmd_addr,cmd_len+1,statuses[accepted]);
            end
            if(data_valid && data_ready) begin
                need(fill>0 && data===pattern(fill,fill_beat),"INPUT_FRAME");
                $display("COLD IN %0d %0d %0d %h",ps(),fill,fill_beat,data);
                fill_beat++;
                if(fill_beat==lengths[fill]) fill=0;
            end
            if(cpl_held) need(cpl_valid && {cpl_tag,cpl_status}===saved_cpl,"COMPLETION_STABILITY");
            cpl_held=cpl_valid && !cpl_ready;saved_cpl={cpl_tag,cpl_status};
            if(cpl_valid && cpl_ready) begin
                integer tag;
                tag=completed+1;
                need(tag<=accepted && cpl_tag==tag && cpl_status==statuses[tag],"COMPLETION_ORDER_STATUS");
                if(statuses[tag]==0) need(b_time[tag]>0 && b_time[tag]<ps(),"COMPLETION_BEFORE_MIG_B");
                completed++;finished[tag]=1;
                $display("COLD CPL %0d %0d %0d",ps(),tag,cpl_status);
            end
        end
    end
    always @(posedge clk) begin
        if(m_axi_awvalid || (m_axi_wvalid && !hold_w) || m_axi_arvalid)
            need(calibrated && !ui_rst && rst_n,"MIG_TRAFFIC_BEFORE_READY");
        if(rst_n) begin
            need(!interrupt_ecc,"ECC_INTERRUPT");
            if(aw_held) need(m_axi_awvalid && {m_axi_awaddr,m_axi_awlen}===saved_aw,"AW_STABILITY");
            if(w_held) need(m_axi_wvalid && {m_axi_wdata,m_axi_wstrb,m_axi_wlast}===saved_w,"W_STABILITY");
            aw_held=m_axi_awvalid && !m_axi_awready;saved_aw={m_axi_awaddr,m_axi_awlen};
            w_held=m_axi_wvalid && !m_axi_wready;saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
            if(m_axi_awvalid && m_axi_awready) begin
                integer tag;
                need(aw_cursor<valid_jobs,"UNAUTHORIZED_MIG_AW");tag=memory_queue[aw_cursor++];
                need(m_axi_awaddr==addresses[tag] && m_axi_awlen==lengths[tag]-1 &&
                     m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && !m_axi_awlock,"MIG_AW_FIELDS");
                aw_time[tag]=ps();aw_count++;peak=(aw_count-b_count>peak)?aw_count-b_count:peak;
                need(aw_count-b_count<=4,"CREDIT_OVERFLOW");
                $display("COLD AW %0d %0d %h %0d",ps(),tag,m_axi_awaddr,m_axi_awlen+1);
            end
            if(m_axi_wvalid && m_axi_wready) begin
                integer tag,slot;
                reg [63:0] address;
                need(w_cursor<valid_jobs,"UNAUTHORIZED_MIG_W");tag=memory_queue[w_cursor];
                need(m_axi_wdata===pattern(tag,w_beat) && m_axi_wstrb=='1 &&
                     m_axi_wlast==(w_beat==lengths[tag]-1),"MIG_W_BYTES");
                address=addresses[tag]+w_beat*64;slot=-1;
                for(integer k=0;k<golden_count;k++) if(golden_addr[k]==address) slot=k;
                if(slot<0) begin slot=golden_count++;golden_addr[slot]=address;end
                golden_data[slot]=m_axi_wdata;writes++;
                $display("COLD W %0d %0d %0d %b %h",ps(),tag,w_beat,m_axi_wlast,m_axi_wdata);
                w_beat++;
                if(m_axi_wlast) begin w_time[tag]=ps();w_cursor++;w_beat=0;end
            end
            if(m_axi_bvalid && m_axi_bready) begin
                integer tag;
                need(b_cursor<valid_jobs,"UNEXPECTED_MIG_B");tag=memory_queue[b_cursor++];
                need(m_axi_bid==0 && m_axi_bresp==0 && aw_time[tag]>0 && w_time[tag]>0 &&
                     aw_time[tag]<ps() && w_time[tag]<ps(),"MIG_B_ORDER");
                b_count++;b_time[tag]=ps();$display("COLD B %0d %0d %0d",ps(),tag,m_axi_bresp);
            end
            if(r_held) need(m_axi_rvalid && {m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata}===saved_r,"MIG_R_STABILITY");
            r_held=m_axi_rvalid && !m_axi_rready;saved_r={m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata};
        end
    end
    always @(calibrated or ui_rst or rst_n or admit or running or restart_required)
        $strobe("COLD STATE %0d calib=%b ui_rst=%b memory_reset_n=%b admit=%b running=%b restart=%b",ps(),calibrated,ui_rst,rst_n,admit,running,restart_required);

    task automatic send(input reg [63:0] address,input integer count,input integer stop_beat=-1);
        integer tag;
        tag=accepted+1;
        @(negedge source_clk);cmd_addr=address;cmd_len=count-1;cmd_tag=tag;cmd_valid=1;
        do @(posedge source_clk);while(!cmd_ready);
        @(negedge source_clk);cmd_valid=0;
        for(integer beat=0;beat<count;beat++) begin
            data=pattern(tag,beat);data_valid=1;
            do @(posedge source_clk);while(!data_ready);
            @(negedge source_clk);data_valid=0;
            if(beat==stop_beat) begin stop=1;$display("COLD STOP %0d fill",ps());end
        end
    endtask
    task automatic drain;
        integer age;
        age=0;
        while(!drained || completed!=accepted) begin wait_source(1);age++;need(age<10000,"DRAIN_TIMEOUT");end
        need(aw_cursor==valid_jobs && w_cursor==valid_jobs && b_cursor==valid_jobs && !fill,"INCOMPLETE_DRAIN");
        $display("COLD DRAIN %0d",ps());
    endtask
    task automatic readback;
        for(integer slot=0;slot<golden_count;slot++) begin
            @(negedge clk);m_axi_araddr=golden_addr[slot][36:0];m_axi_arvalid=1;
            do @(posedge clk);while(!m_axi_arready);
            $display("COLD AR %0d %h",ps(),m_axi_araddr);
            @(negedge clk);m_axi_arvalid=0;
            while(!m_axi_rvalid) @(negedge clk);
            repeat(3) @(negedge clk);
            need(m_axi_rid==0 && m_axi_rresp==0 && m_axi_rlast && m_axi_rdata===golden_data[slot],"READBACK_BYTES");
            m_axi_rready=1;@(posedge clk);
            $display("COLD R %0d %h %h",ps(),m_axi_araddr,m_axi_rdata);reads++;
            @(negedge clk);m_axi_rready=0;
        end
    endtask
    initial begin
        wait(rearm_ready);@(negedge mgmt_clk);rearm=1;
        repeat(4) @(negedge mgmt_clk);rearm=0;
    end
    initial begin
        for(integer k=0;k<64;k++) begin aw_time[k]=0;w_time[k]=0;b_time[k]=0;finished[k]=0;end
        #100;
        $display("COLD CASE cold_admission");send(0,1);drain;
        $display("COLD CASE sparse_ranks");
        send(64'h100000040,1);drain;send(64'h800000000,1);drain;
        send(64'h1000000000,1);drain;send(64'h1800000000,1);drain;
        send(64'h1fffffffc0,1);drain;readback;
        $display("COLD CASE burst_boundaries");
        send(64'h100001000,64);send(64'h1ffffffe00,8);drain;readback;
        $display("COLD CASE four_credits");
        @(negedge clk);hold_b=1;
        for(integer k=0;k<4;k++) send(64'h100002000+k*4096,1);
        wait_source(2000);need(aw_count-b_count==4,"FOUR_MIG_CREDITS");
        $display("COLD SATURATED %0d pending=%0d",ps(),aw_count-b_count);
        stop=1;$display("COLD STOP %0d b_held",ps());
        wait_source(20);need(!drained && !cmd_ready,"B_HELD_DRAIN");
        @(negedge clk);hold_b=0;drain;stop=0;
        $display("COLD CASE stop_fill");send(64'h100006000,8,1);drain;stop=0;
        $display("COLD CASE stop_after_aw");
        @(negedge clk);hold_w=1;send(64'h100007000,8);
        wait(aw_time[14]>0);@(negedge source_clk);stop=1;$display("COLD STOP %0d aw_exposed",ps());
        wait_source(20);need(!drained && !cmd_ready,"W_HELD_DRAIN");
        @(negedge clk);hold_w=0;drain;stop=0;
        $display("COLD CASE result_backpressure");
        hold_cpl=1;send(64'h100008000,1);wait(cpl_valid);wait_source(20);
        need(!drained,"RESULT_OWNERSHIP");hold_cpl=0;drain;readback;
        $display("COLD CASE range_reject");send(64'h2000000000,1);drain;
        need(fault && !cmd_ready,"REJECT_CLOSES_ADMISSION");readback;
        need(accepted==16 && completed==16 && valid_jobs==15 && peak==4 && blocked>20,"FINAL_COUNTS");
        for(integer rank=0;rank<4;rank++) need(rank_acts[rank]>0,"RANK_COVERAGE");
        verify_periodic_complete();
        $display("COLD PASS commands=%0d completions=%0d writes=%0d reads=%0d locations=%0d peak=%0d blocked=%0d ps=%0d",accepted,completed,writes,reads,golden_count,peak,blocked,ps());
        $finish;
    end
    initial begin #300000;$fatal(1,"MULTIBUFFER_MIG_TIMEOUT commands=%0d completions=%0d",accepted,completed);end
    `include "ntrpu_ddr_periodic_ingress.vh"
endmodule
