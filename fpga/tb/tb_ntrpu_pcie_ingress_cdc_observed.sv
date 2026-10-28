`timescale 1ns / 1ps
`ifdef XILINX_SIMULATOR
module short(in1,in1); inout in1; endmodule
`endif
module tb_ntrpu_pcie_ingress_cdc #(parameter integer HOST_PERIOD_PS=4000,HOST_PHASE_PS=700);
    import arch_package::*;
    wire clk;
    reg rst_n=0,stop=0,source_stopped=0,ingress_isolated=0;
    reg [63:0] quiesce_epoch=0;
    reg arm_valid=0;
    wire arm_ready;
    reg [63:0] arm_epoch=0,arm_tag=0,arm_window=0,arm_destination=0;
    wire [63:0] epoch,valid_bytes;
    wire lease_held,failed,transport_idle;
    wire [3:0] error_code;
    reg [63:0] h_axi_awaddr=0;
    reg [3:0] h_axi_awid=0;
    reg [7:0] h_axi_awlen=0;
    reg [2:0] h_axi_awsize=2;
    reg [1:0] h_axi_awburst=1;
    reg h_axi_awlock=0,h_axi_awvalid=0,h_axi_wlast=0,h_axi_wvalid=0,h_axi_bready=0;
    wire h_axi_awready,h_axi_wready,h_axi_bvalid;
    reg [511:0] h_axi_wdata=0;
    reg [63:0] h_axi_wstrb=0;
    wire [3:0] h_axi_bid;
    wire [1:0] h_axi_bresp;
    wire [63:0] s_axi_awaddr=dut.d_axi_awaddr;
    wire [3:0] s_axi_awid=dut.d_axi_awid;
    wire [7:0] s_axi_awlen=dut.d_axi_awlen;
    wire [2:0] s_axi_awsize=dut.d_axi_awsize;
    wire [1:0] s_axi_awburst=dut.d_axi_awburst;
    wire s_axi_awlock=dut.d_axi_awlock;
    wire s_axi_awvalid=dut.d_axi_awvalid;
    wire s_axi_awready=dut.d_axi_awready;
    wire [511:0] s_axi_wdata=dut.d_axi_wdata;
    wire [63:0] s_axi_wstrb=dut.d_axi_wstrb;
    wire s_axi_wlast=dut.d_axi_wlast;
    wire s_axi_wvalid=dut.d_axi_wvalid;
    wire s_axi_wready=dut.d_axi_wready;
    wire [3:0] s_axi_bid=dut.d_axi_bid;
    wire [1:0] s_axi_bresp=dut.d_axi_bresp;
    wire s_axi_bvalid=dut.d_axi_bvalid;
    wire s_axi_bready=dut.d_axi_bready;

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
    wire m_axi_awready,m_axi_wready,m_axi_bvalid;
    wire [3:0] m_axi_bid;
    wire [1:0] m_axi_bresp;
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
    reg [511:0] golden [0:5];
    reg [5:0] written=0;
    integer read_count=0,ar_count=0;
    reg r_stalled=0;
    reg [518:0] held_r;
    function automatic [63:0] location(input integer slot);
        case(slot)
            0:return 0;1:return 64'h100000040;2:return 64'h800000000;
            3:return 64'h1000000000;4:return 64'h1800000000;5:return 64'h1fffffffc0;
            default:return 64'hffffffffffffffff;
        endcase
    endfunction
    function automatic integer write_slot(input integer job);
        if(job<=6) return job-1;
        if(job==10) return 1;
        if(job==11) return 5;
        if(job==13) return 0;
        return -1;
    endfunction
    reg host_clk=0,host_rst_n=0;
    initial begin #(HOST_PHASE_PS/1000.0); forever #(HOST_PERIOD_PS/2000.0) host_clk=~host_clk; end
    initial begin wait(rst_n===1'b1); repeat(8) @(negedge host_clk);host_rst_n=1; end
    task automatic host_idle(input integer count); repeat(count) @(negedge host_clk); endtask
    ntrpu_pcie_ingress_async dut (
        .host_clk(host_clk),
        .host_rst_n(host_rst_n),
        .s_axi_awaddr(h_axi_awaddr),
        .s_axi_awid(h_axi_awid),
        .s_axi_awlen(h_axi_awlen),
        .s_axi_awsize(h_axi_awsize),
        .s_axi_awburst(h_axi_awburst),
        .s_axi_awlock(h_axi_awlock),
        .s_axi_awvalid(h_axi_awvalid),
        .s_axi_awready(h_axi_awready),
        .s_axi_wdata(h_axi_wdata),
        .s_axi_wstrb(h_axi_wstrb),
        .s_axi_wlast(h_axi_wlast),
        .s_axi_wvalid(h_axi_wvalid),
        .s_axi_wready(h_axi_wready),
        .s_axi_bid(h_axi_bid),
        .s_axi_bresp(h_axi_bresp),
        .s_axi_bvalid(h_axi_bvalid),
        .s_axi_bready(h_axi_bready),
        .*
    );
    logic sys_clk=0,sys_rst=1;
    always #2 sys_clk=~sys_clk;
    initial begin #20;sys_rst=0;end
    localparam longint unsigned CAPACITY=64'd137438953472;
    longint unsigned rank_acts[4]='{default:0};
    bit any_axi=0,calibration_seen=0,complete=0;
    longint last_ui_ps=0,ui_samples=0;
    wire ui_rst, calibrated, interrupt_ecc;
    wire act_n, reset_n, parity;
    wire [17:0] adr;
    wire [1:0] ba, bg, ck_t, ck_c;
    wire [3:0] cke, odt, cs_n;
    wire [71:0] dq;
    wire [17:0] dqs_t, dqs_c;
    ntrpu_ddr_geometry mig (
        .sys_rst(sys_rst), .c0_sys_clk_p(sys_clk), .c0_sys_clk_n(~sys_clk),
        .c0_ddr4_act_n(act_n), .c0_ddr4_adr(adr), .c0_ddr4_ba(ba), .c0_ddr4_bg(bg),
        .c0_ddr4_cke(cke), .c0_ddr4_odt(odt), .c0_ddr4_cs_n(cs_n),
        .c0_ddr4_ck_t(ck_t), .c0_ddr4_ck_c(ck_c), .c0_ddr4_reset_n(reset_n),
        .c0_ddr4_parity(parity), .c0_ddr4_dq(dq), .c0_ddr4_dqs_t(dqs_t), .c0_ddr4_dqs_c(dqs_c),
        .c0_init_calib_complete(calibrated), .c0_ddr4_ui_clk(clk),
        .c0_ddr4_ui_clk_sync_rst(ui_rst), .dbg_clk(), .dbg_bus(),
        .c0_ddr4_aresetn(rst_n),
        .c0_ddr4_s_axi_ctrl_awvalid(1'b0), .c0_ddr4_s_axi_ctrl_awaddr(32'b0),
        .c0_ddr4_s_axi_ctrl_awready(), .c0_ddr4_s_axi_ctrl_wvalid(1'b0),
        .c0_ddr4_s_axi_ctrl_wdata(32'b0), .c0_ddr4_s_axi_ctrl_wready(),
        .c0_ddr4_s_axi_ctrl_bvalid(), .c0_ddr4_s_axi_ctrl_bready(1'b0), .c0_ddr4_s_axi_ctrl_bresp(),
        .c0_ddr4_s_axi_ctrl_arvalid(1'b0), .c0_ddr4_s_axi_ctrl_araddr(32'b0),
        .c0_ddr4_s_axi_ctrl_arready(), .c0_ddr4_s_axi_ctrl_rvalid(),
        .c0_ddr4_s_axi_ctrl_rready(1'b0), .c0_ddr4_s_axi_ctrl_rdata(), .c0_ddr4_s_axi_ctrl_rresp(),
        .c0_ddr4_interrupt(interrupt_ecc),
        .c0_ddr4_s_axi_awid(m_axi_awid), .c0_ddr4_s_axi_awaddr(m_axi_awaddr[36:0]), .c0_ddr4_s_axi_awlen(m_axi_awlen),
        .c0_ddr4_s_axi_awsize(m_axi_awsize), .c0_ddr4_s_axi_awburst(m_axi_awburst),
        .c0_ddr4_s_axi_awlock(m_axi_awlock), .c0_ddr4_s_axi_awcache(m_axi_awcache),
        .c0_ddr4_s_axi_awprot(m_axi_awprot), .c0_ddr4_s_axi_awqos(4'b0),
        .c0_ddr4_s_axi_awvalid(m_axi_awvalid), .c0_ddr4_s_axi_awready(m_axi_awready),
        .c0_ddr4_s_axi_wdata(m_axi_wdata), .c0_ddr4_s_axi_wstrb(m_axi_wstrb), .c0_ddr4_s_axi_wlast(m_axi_wlast),
        .c0_ddr4_s_axi_wvalid(m_axi_wvalid), .c0_ddr4_s_axi_wready(m_axi_wready),
        .c0_ddr4_s_axi_bready(m_axi_bready), .c0_ddr4_s_axi_bid(m_axi_bid), .c0_ddr4_s_axi_bresp(m_axi_bresp),
        .c0_ddr4_s_axi_bvalid(m_axi_bvalid),
        .c0_ddr4_s_axi_arid(m_axi_arid), .c0_ddr4_s_axi_araddr(m_axi_araddr[36:0]), .c0_ddr4_s_axi_arlen(m_axi_arlen),
        .c0_ddr4_s_axi_arsize(m_axi_arsize), .c0_ddr4_s_axi_arburst(m_axi_arburst),
        .c0_ddr4_s_axi_arlock(m_axi_arlock), .c0_ddr4_s_axi_arcache(m_axi_arcache),
        .c0_ddr4_s_axi_arprot(m_axi_arprot), .c0_ddr4_s_axi_arqos(4'b0),
        .c0_ddr4_s_axi_arvalid(m_axi_arvalid), .c0_ddr4_s_axi_arready(m_axi_arready),
        .c0_ddr4_s_axi_rready(m_axi_rready), .c0_ddr4_s_axi_rid(m_axi_rid), .c0_ddr4_s_axi_rdata(m_axi_rdata),
        .c0_ddr4_s_axi_rresp(m_axi_rresp), .c0_ddr4_s_axi_rlast(m_axi_rlast), .c0_ddr4_s_axi_rvalid(m_axi_rvalid)
    );

    for (genvar slot = 0; slot < 2; slot++) begin: dimms
        time reset_low_at = 0, last_reset_width = 0;
        ddr4_rdimm_wrapper #(
            .MC_DQ_WIDTH(72), .MC_DQS_BITS(18), .MC_DM_WIDTH(18),
            .MC_CKE_NUM(2), .MC_ODT_WIDTH(2), .MC_ABITS(18), .MC_BANK_WIDTH(2), .MC_BANK_GROUP(2),
            .MC_CS_NUM(2), .MC_RANKS_NUM(2), .NUM_PHYSICAL_PARTS(18),
            .CALIB_EN("NO"), .tCK(1250), .MEM_PART_WIDTH("x4"), .MC_CA_MIRROR("ON"),
            .DDR_SIM_MODEL("MICRON"), .DM_DBI("NONE"), .MC_REG_CTRL("ON"),
            .DIMM_MODEL("RDIMM"), .RDIMM_SLOTS(2), .CONFIGURED_DENSITY(_16G)
        ) memory (
            .ddr4_act_n(act_n), .ddr4_addr(adr), .ddr4_ba(ba), .ddr4_bg(bg), .ddr4_par(parity),
            .ddr4_cke(cke[slot*2 +: 2]), .ddr4_odt(odt[slot*2 +: 2]), .ddr4_cs_n(cs_n[slot*2 +: 2]),
            .ddr4_ck_t(ck_t[slot]), .ddr4_ck_c(ck_c[slot]), .ddr4_reset_n(reset_n),
            .ddr4_dm_dbi_n(), .ddr4_dq(dq), .ddr4_dqs_t(dqs_t), .ddr4_dqs_c(dqs_c),
            .ddr4_alert_n(), .initDone(calibrated), .scl(1'b0), .sa0(1'b0), .sa1(1'b0),
            .sa2(1'b0), .sda(), .bfunc(1'b0), .vddspd(1'b1)
        );
        // Observe the public wrapper's RCD output; never force model state.
        always @(memory.reset_n) begin
            if (memory.reset_n === 0) reset_low_at = $time;
            if (memory.reset_n === 1 && reset_low_at != 0) last_reset_width = $time-reset_low_at;
            $display("NTRPU_DDR_RDIMM_RESET slot=%0d time_ps=%0d reset_n=%b last_low_ps=%0d",
                     slot, longint'($realtime*1000), memory.reset_n, last_reset_width*1000);
        end
        always @(posedge ck_t[slot]) begin
            if (calibrated && !act_n)
                for (int rank = 0; rank < 2; rank++)
                    if (cs_n[slot*2+rank] === 1'b0) begin
                        rank_acts[slot*2+rank]++;
                        $display("NTRPU_MIG_LINE_ACT ps=%0d rank=%0d count=%0d",longint'($realtime*1000),slot*2+rank,rank_acts[slot*2+rank]);
                    end
        end
    end



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
        if(!condition) $fatal(1,"MIG_LINE_CHECK job=%0d cycle=%0d %s",jobs,cycle,message);
    endtask
    task automatic aw(input reg [63:0] address,input reg [3:0] id,input reg [7:0] len,input reg [2:0] size);
        @(negedge host_clk);h_axi_awaddr=address;h_axi_awid=id;h_axi_awlen=len;h_axi_awsize=size;h_axi_awvalid=1;
        do @(posedge host_clk); while(!h_axi_awready);
        @(negedge host_clk);h_axi_awvalid=0;
    endtask
    task automatic w(input reg [511:0] data,input reg [63:0] mask,input bit last);
        @(negedge host_clk);h_axi_wdata=data;h_axi_wstrb=mask;h_axi_wlast=last;h_axi_wvalid=1;
        do @(posedge host_clk); while(!h_axi_wready);
        @(negedge host_clk);h_axi_wvalid=0;
    endtask
    task automatic response(input reg [3:0] id,input reg [1:0] resp);
        while(!h_axi_bvalid) @(negedge host_clk);
        need(h_axi_bid===id && h_axi_bresp===resp,"B_ID_OR_RESPONSE");
        host_idle(3+(seed%3));
        need(h_axi_bvalid && h_axi_bid===id && h_axi_bresp===resp,"B_STABILITY");
        h_axi_bready=1;@(posedge host_clk);@(negedge host_clk);h_axi_bready=0;
    endtask
    task automatic tx(input reg [63:0] address,input reg [63:0] mask,input reg [511:0] data,
                      input integer order,input reg [1:0] resp,input reg [7:0] len=0,input reg [2:0] size=2);
        reg [3:0] id;
        integer prior_b;
        id=4'((b_count+3)%16);prior_b=b_count;
        fork
            begin if(order==1) host_idle(4);aw(address,id,len,size);end
            begin if(order==0) host_idle(4);w(data,mask,1);end
        join
        response(id,resp);
        need(b_count==prior_b+1,"EXACTLY_ONE_B");
    endtask
    task automatic start_job(input string name);
        source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        arm_epoch=epoch+1;arm_tag=64'h1000+jobs+1;
        arm_window=(jobs%2==0) ? 64'h1000fffc0 : 64'h100000000;
        if(jobs+1<=6) arm_destination=location(jobs);
        else case(jobs+1)
            7:arm_destination=64'h2000000000;
            8:arm_destination=64'h1000000000000000;
            9:arm_destination=64'hffffffffffffffc0;
            10:arm_destination=location(1);
            11:arm_destination=location(5);
            default:arm_destination=0;
        endcase
        jobs=jobs+1;down_aw=0;down_w=0;down_b=0;result_in_job=0;expected_status=0;
        expected_data=pattern(jobs);expected_bitmap=0;
        $display("NTRPU_MIG_LINE_CASE job=%0d name=%s",jobs,name);
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

    task automatic wait_commit;
        while(down_b!=1 || bridge_phase!=0 || peer_axi_busy || !guard_idle) @(negedge clk);
    endtask
    task automatic stop_at(input integer phase);
        while(bridge_phase!=phase) @(negedge clk);
        $display("NTRPU_MIG_LINE_STOP phase=%0d cycle=%0d",phase,cycle);
        stop=1;
    endtask
    task automatic finish_job(input integer status,input integer writes);
        expected_status=status;
        need(host_aw_count==aw_count && host_w_count==w_count && host_b_count==b_count &&
             host_aw_count==host_b_count && host_w_count==host_b_count &&
             !h_axi_awvalid && !h_axi_wvalid && !h_axi_bvalid,"EXTERNAL_CDC_DRAIN_BARRIER");
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
        if(m_axi_awvalid || m_axi_wvalid) need(!(jobs==7 || jobs==8 || jobs==9 || jobs==12),"REJECTED_ADDRESS_REACHED_MEMORY");
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
            $display("NTRPU_MIG_LINE_PHASE phase=%0d busy=%b idle=%b status=%0d cycle=%0d",bridge_phase,peer_axi_busy,guard_idle,destination_status,cycle);previous_phase=bridge_phase;
        end
        if(arm_valid && arm_ready) $display("NTRPU_MIG_LINE_ARM epoch=%h tag=%h window=%h dest=%h cycle=%0d",arm_epoch,arm_tag,arm_window,arm_destination,cycle);
        if(s_axi_awvalid && s_axi_awready) begin
            aw_count=aw_count+1;
            $display("NTRPU_MIG_LINE_AW addr=%h id=%h len=%0d size=%0d cycle=%0d",s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,cycle);
        end
        if(s_axi_wvalid && s_axi_wready) begin
            w_count=w_count+1;
            $display("NTRPU_MIG_LINE_W data=%h strb=%h last=%b cycle=%0d",s_axi_wdata,s_axi_wstrb,s_axi_wlast,cycle);
        end
        if(s_axi_bvalid && s_axi_bready) begin
            b_count=b_count+1;$display("NTRPU_MIG_LINE_B id=%h resp=%0d cycle=%0d",s_axi_bid,s_axi_bresp,cycle);
        end
        if(m_axi_awvalid) need(m_axi_awaddr===arm_destination && m_axi_awid==0 && m_axi_awlen==0 && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awcache==0 && m_axi_awprot==0 && !m_axi_awlock,"DESTINATION_AW_FIELDS");
        if(m_axi_wvalid) need(m_axi_wdata===expected_data && m_axi_wstrb==64'hffffffffffffffff && m_axi_wlast && valid_bytes==64'hffffffffffffffff,"DESTINATION_DATA");
        if(m_axi_awvalid && m_axi_awready) begin
            down_aw=down_aw+1;total_aw=total_aw+1;need(down_aw==1,"DUPLICATE_AW");
            $display("NTRPU_MIG_LINE_MAW addr=%h id=%h len=%0d size=%0d cycle=%0d",m_axi_awaddr,m_axi_awid,m_axi_awlen,m_axi_awsize,cycle);
        end
        if(m_axi_wvalid && m_axi_wready) begin
            down_w=down_w+1;total_w=total_w+1;need(down_w==1,"DUPLICATE_W");
            $display("NTRPU_MIG_LINE_MW data=%h strb=%h last=%b cycle=%0d",m_axi_wdata,m_axi_wstrb,m_axi_wlast,cycle);
        end
        if(m_axi_bvalid && m_axi_bready) begin
            need(down_aw==1 && down_w==1,"B_BEFORE_CHANNELS");
            if(m_axi_bid==0) begin down_b=down_b+1;total_b=total_b+1;need(down_b==1,"DUPLICATE_FINAL_B");end
            else foreign_b=foreign_b+1;
            $display("NTRPU_MIG_LINE_MB id=%h resp=%0d cycle=%0d",m_axi_bid,m_axi_bresp,cycle);
        end
        if(result_valid && result_ready) begin
            results=results+1;result_in_job=result_in_job+1;
            $display("NTRPU_MIG_LINE_RESULT epoch=%h tag=%h status=%0d cycle=%0d",result_epoch,result_tag,result_status,cycle);
        end
    end

    task automatic readback;
        for(integer slot=0;slot<6;slot=slot+1) if(written[slot]) begin
            @(negedge clk);m_axi_araddr=37'(location(slot));m_axi_arvalid=1;
            do @(posedge clk); while(!m_axi_arready);
            @(negedge clk);m_axi_arvalid=0;
            while(!m_axi_rvalid) @(negedge clk);
            need(m_axi_rdata===golden[slot] && m_axi_rid==0 && m_axi_rresp==0 && m_axi_rlast,"MIG_READBACK_DATA");
            idle(3);m_axi_rready=1;@(posedge clk);@(negedge clk);m_axi_rready=0;
        end
    endtask
    always @(posedge clk) begin
        if(calibrated) begin
            if(calibration_seen) need(longint'($realtime*1000)-last_ui_ps==5000,"MIG_UI_PERIOD");
            last_ui_ps=longint'($realtime*1000);calibration_seen=1;ui_samples++;
        end
        if(m_axi_awvalid || m_axi_wvalid || m_axi_arvalid)
            need(calibrated && !ui_rst && rst_n,"TRAFFIC_BEFORE_CALIBRATION");
        if(rst_n) begin
            need(calibrated && !ui_rst && !interrupt_ecc,"MIG_STATE_OR_ECC");
            if(r_stalled) need(m_axi_rvalid && {m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata}===held_r,"MIG_R_STABILITY");
            r_stalled=m_axi_rvalid && !m_axi_rready;held_r={m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata};
            if(m_axi_arvalid && m_axi_arready) begin
                ar_count=ar_count+1;
                $display("NTRPU_MIG_LINE_AR job=%0d addr=%h cycle=%0d ps=%0d",jobs,m_axi_araddr,cycle,longint'($realtime*1000));
            end
            if(m_axi_rvalid && m_axi_rready) begin
                read_count=read_count+1;
                $display("NTRPU_MIG_LINE_R job=%0d addr=%h data=%h id=%h resp=%0d last=%b cycle=%0d ps=%0d",jobs,m_axi_araddr,m_axi_rdata,m_axi_rid,m_axi_rresp,m_axi_rlast,cycle,longint'($realtime*1000));
            end
        end
    end
    always @(calibrated or ui_rst)
        $strobe("NTRPU_MIG_LINE_STATE ps=%0d calib=%b ui_rst=%b rst=%b",longint'($realtime*1000),calibrated,ui_rst,rst_n);
    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        wait(calibrated===1'b1 && ui_rst===1'b0);idle(40);rst_n=1;wait(host_rst_n===1'b1);idle(20);
        $display("NTRPU_CDC_LINE_CONFIG host_period_ps=%0d host_phase_ps=%0d ui_period_ps=5000 ps=%0d",HOST_PERIOD_PS,HOST_PHASE_PS,longint'($realtime*1000));
        $display("NTRPU_MIG_LINE_START seed=%0d ps=%0d",seed,longint'($realtime*1000));
        for(integer job=1;job<=13;job=job+1) begin
            start_job($sformatf("memory_job_%0d",job));
            if(job==12) begin
                fragment(0,15,0);tx(arm_window,15,expected_data ^ 512'b1,1,2);finish_job(4,0);
            end else begin
                if(job==10) begin fork fill_line();stop_at(3);join end
                else fill_line();
                if(job==7 || job==8 || job==9) finish_job(5,0);
                else begin
                    wait_commit();golden[write_slot(job)]=expected_data;written[write_slot(job)]=1;
                    finish_job(job==10 ? 1 : 0,1);
                end
            end
            readback();
        end
        idle(20);
        need(jobs==13 && results==13 && aw_count==194 && w_count==194 && b_count==194 && total_aw==9 && total_w==9 && total_b==9,"SUITE_COUNTS");
        need(read_count==63 && ar_count==63 && guard_idle && !peer_axi_busy && transport_idle && !m_axi_rvalid,"FINAL_DRAIN");
        for(integer rank=0;rank<4;rank=rank+1) need(rank_acts[rank]>0,"RANK_ACTIVITY");
        need(host_aw_count==194 && host_w_count==194 && host_b_count==194,"HOST_CDC_DRAIN");
        verify_periodic_complete();
        $display("NTRPU_CDC_LINE_PASS aw=%0d w=%0d b=%0d ps=%0d",host_aw_count,host_w_count,host_b_count,longint'($realtime*1000));
        $display("NTRPU_MIG_LINE_PASS jobs=%0d results=%0d source_aw=%0d source_w=%0d source_b=%0d writes=%0d reads=%0d bytes=%0d cycles=%0d rank0=%0d rank1=%0d rank2=%0d rank3=%0d seed=%0d ps=%0d",jobs,results,aw_count,w_count,b_count,total_w,read_count,read_count*64,cycle,rank_acts[0],rank_acts[1],rank_acts[2],rank_acts[3],seed,longint'($realtime*1000));
        $finish;
    end

    integer host_aw_count=0,host_w_count=0,host_b_count=0;
    longint prior_host_ps=0;
    reg host_b_stalled=0;
    reg [5:0] host_b_held;
    always @(posedge host_clk) if(host_rst_n) begin
        if(prior_host_ps) need(longint'($realtime*1000)-prior_host_ps==HOST_PERIOD_PS,"HOST_CLOCK_PERIOD");
        prior_host_ps=longint'($realtime*1000);
        if(host_b_stalled) need(h_axi_bvalid && {h_axi_bid,h_axi_bresp}===host_b_held,"HOST_B_STABILITY");
        host_b_stalled=h_axi_bvalid && !h_axi_bready;host_b_held={h_axi_bid,h_axi_bresp};
        if(h_axi_awvalid && h_axi_awready) host_aw_count++;
        if(h_axi_wvalid && h_axi_wready) host_w_count++;
        if(h_axi_bvalid && h_axi_bready) host_b_count++;
    end
    always @(posedge host_clk) if(host_rst_n) begin
        if(h_axi_awvalid && h_axi_awready) $display("NTRPU_CDC_LINE_HOST_AW addr=%h id=%h len=%0d size=%0d burst=%0d lock=%b ps=%0d",h_axi_awaddr,h_axi_awid,h_axi_awlen,h_axi_awsize,h_axi_awburst,h_axi_awlock,longint'($realtime*1000));
        if(h_axi_wvalid && h_axi_wready) $display("NTRPU_CDC_LINE_HOST_W data=%h strb=%h last=%b ps=%0d",h_axi_wdata,h_axi_wstrb,h_axi_wlast,longint'($realtime*1000));
        if(h_axi_bvalid && h_axi_bready) $display("NTRPU_CDC_LINE_HOST_B id=%h resp=%0d ps=%0d",h_axi_bid,h_axi_bresp,longint'($realtime*1000));
    end
    always @(posedge clk) if(rst_n) begin
        if(s_axi_awvalid && s_axi_awready) $display("NTRPU_CDC_LINE_DEVICE_AW addr=%h id=%h len=%0d size=%0d burst=%0d lock=%b ps=%0d",s_axi_awaddr,s_axi_awid,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock,longint'($realtime*1000));
        if(s_axi_wvalid && s_axi_wready) $display("NTRPU_CDC_LINE_DEVICE_W data=%h strb=%h last=%b ps=%0d",s_axi_wdata,s_axi_wstrb,s_axi_wlast,longint'($realtime*1000));
        if(s_axi_bvalid && s_axi_bready) $display("NTRPU_CDC_LINE_DEVICE_B id=%h resp=%0d ps=%0d",s_axi_bid,s_axi_bresp,longint'($realtime*1000));
    end
    initial begin #500000; $fatal(1,"MIG_LINE_TIMEOUT job=%0d phase=%0d busy=%b cycle=%0d",jobs,bridge_phase,peer_axi_busy,cycle); end
    `include "ntrpu_ddr_periodic_ingress.vh"
endmodule
