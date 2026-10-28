`timescale 1ps / 1ps
// Actual XDMA bypass -> AMD asynchronous converter -> guarded ingress -> MIG.
// Trusted context/isolation controls and UI readback are test drivers only.
// Cold coordinated reset, one channel with generic DDR geometry; not a board.
// line/done aliases observe the production hierarchy; they never drive it.
`ifdef XILINX_SIMULATOR
module short(in1,in1); inout in1; endmodule
`endif
module ntrpu_pcie_fragment_native_host_harness;
    import arch_package::*;
    wire clk;
    wire host_clk=board.EP.user_clk;
    reg rst_n=0,host_rst_n=0;
    integer device_aw=0,device_w=0,device_b=0;
    integer read_count=0,ar_count=0;
    reg [511:0] golden [0:2];
    reg [2:0] written=0;
    reg r_stalled=0;
    reg [518:0] held_r;
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
    wire guard_idle;
    function automatic [63:0] location(input integer slot);
        case(slot) 0:return 0;1:return 64'h100000040;2:return 64'h1fffffffc0;default:return '1;endcase
    endfunction
    wire stop;
    reg source_stopped=0,ingress_isolated=0;
    reg [63:0] quiesce_epoch=0;
    wire arm_valid;
    wire arm_ready;
    wire [63:0] arm_epoch,arm_tag,arm_window,arm_destination;
    wire [63:0] epoch,valid_bytes;
    wire lease_held,failed,transport_idle;
    wire [3:0] error_code;
    wire [31:0] rejected_frames,rejected_identity;
    wire [3:0] last_reject_code;
    wire [6:0] last_reject_faults;
    wire awready,wready,bvalid;
    wire [3:0] bid;
    wire [1:0] bresp;

    integer cycles=0,current_job=-1,aw_count=0,w_count=0,b_count=0,line_count=0,result_count=0;
    integer begin_b=0,begin_line=0,begin_result=0,aw_stall=0,w_stall=0,b_stall=0;
    integer dest_aw=0,dest_w=0,dest_b=0,begin_dest=0;
    wire line_valid=dut.data_path.core.peer.line_valid,line_ready=dut.data_path.core.peer.line_ready;
    wire [63:0] line_epoch=dut.data_path.core.peer.line_epoch,line_tag=dut.data_path.core.peer.line_tag,line_address=dut.data_path.core.peer.line_address;
    wire [511:0] line_data=dut.data_path.core.peer.line_data;
    wire done_valid=dut.data_path.core.peer.done_valid,done_ready=dut.data_path.core.peer.done_ready,done_error=dut.data_path.core.peer.ingress.done_error;
    wire [63:0] done_epoch=dut.data_path.core.peer.saved_epoch,done_tag=dut.data_path.core.peer.saved_tag;
    wire [63:0] peer_epoch;
    wire peer_quarantined,peer_axi_busy;
    wire [2:0] bridge_phase,destination_status;
    wire [36:0] m_axi_awaddr;
    wire [63:0] m_axi_wstrb;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize,m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awcache,m_axi_awid;
    wire m_axi_awlock,m_axi_awvalid,m_axi_wlast,m_axi_wvalid,m_axi_bready;
    wire [511:0] m_axi_wdata;
    wire m_axi_awready;
    wire m_axi_wready;
    wire m_axi_bvalid;
    wire [3:0] m_axi_bid;
    wire [1:0] m_axi_bresp;
    wire result_valid;
    wire result_ready;
    wire irq,protocol_fault;
    wire [63:0] result_epoch,result_tag;
    wire [3:0] result_status;
    ntrpu_pcie_fragment_host dut (
        .s_axil_awaddr(board.EP.m_axil_awaddr),
        .s_axil_wdata(board.EP.m_axil_wdata),
        .s_axil_araddr(board.EP.m_axil_araddr),
        .s_axil_wstrb(board.EP.m_axil_wstrb),
        .s_axil_awvalid(board.EP.m_axil_awvalid),
        .s_axil_wvalid(board.EP.m_axil_wvalid),
        .s_axil_bready(board.EP.m_axil_bready),
        .s_axil_arvalid(board.EP.m_axil_arvalid),
        .s_axil_rready(board.EP.m_axil_rready),
        .s_axil_awready(board.EP.m_axil_awready),
        .s_axil_wready(board.EP.m_axil_wready),
        .s_axil_arready(board.EP.m_axil_arready),
        .s_axil_bvalid(board.EP.m_axil_bvalid),
        .s_axil_rvalid(board.EP.m_axil_rvalid),
        .s_axil_bresp(board.EP.m_axil_bresp),
        .s_axil_rresp(board.EP.m_axil_rresp),
        .s_axil_rdata(board.EP.m_axil_rdata),
        .irq(irq),.protocol_fault(protocol_fault),
        .host_clk(host_clk),.host_rst_n(host_rst_n),.guard_idle(guard_idle),
        .clk(clk),.rst_n(rst_n),.stop(stop),.source_stopped(source_stopped),.ingress_isolated(ingress_isolated),
        .quiesce_epoch(quiesce_epoch),.arm_valid(arm_valid),.arm_ready(arm_ready),
        .arm_epoch(arm_epoch),.arm_tag(arm_tag),.arm_window(arm_window),.arm_destination(arm_destination),
        .epoch(epoch),.lease_held(lease_held),.failed(failed),.error_code(error_code),
        .transport_idle(transport_idle),.valid_bytes(valid_bytes),
        .rejected_frames(rejected_frames),.rejected_identity(rejected_identity),
        .last_reject_code(last_reject_code),.last_reject_faults(last_reject_faults),
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
    logic sys_clk=0,sys_rst=1;
    always #2000 sys_clk=~sys_clk;
    initial begin #20000;sys_rst=0;end
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
                     slot, $time, memory.reset_n, last_reset_width);
        end
        always @(posedge ck_t[slot]) begin
            if (calibrated && !act_n)
                for (int rank = 0; rank < 2; rank++)
                    if (cs_n[slot*2+rank] === 1'b0) begin
                        rank_acts[slot*2+rank]++;
                        $display("NTRPU_MIG_LINE_ACT ps=%0d rank=%0d count=%0d",$time,slot*2+rank,rank_acts[slot*2+rank]);
                    end
        end
    end



    initial begin
        wait(calibrated===1'b1 && ui_rst===1'b0 && board.EP.user_resetn===1'b1);
        repeat(40) @(negedge clk);rst_n=1;
        repeat(8) @(negedge host_clk);host_rst_n=1;
        $display("NTRPU_NATIVE_MIG_START ps=%0d",$time);
    end
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
    // AMD enumeration probes BAR0 once before the 72 fixture reads.
    integer control_aw=0,control_w=0,control_b=0,control_ar=0,control_r=0;
    task wait_cold_ready;
        wait(host_rst_n && rst_n);
    endtask
    task prepare_job(input integer job);
        begin
            @(negedge clk);
            current_job=job;begin_b=b_count;begin_line=line_count;begin_result=result_count;begin_dest=dest_b;
            source_stopped=0;ingress_isolated=0;quiesce_epoch=epoch-1;
        end
    endtask
    task allow_arm(input integer job);
        begin
            if(!arm_valid || epoch!=(job==0 ? 64'd0 : 64'h100000000+job)) $fatal(1,"NTRPU_NATIVE_CONTROL_PENDING_ARM");
            @(negedge clk);source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch;
        end
    endtask
    task after_arm(input integer job);
        begin
            @(negedge clk);
            if(epoch!==64'h100000001+job || peer_epoch!==epoch || valid_bytes!==0 || failed || arm_valid)
                $fatal(1,"NTRPU_NATIVE_CONTROL_ARM_STATE");
            source_stopped=0;ingress_isolated=0;
        end
    endtask
    task drain_job(input integer job);
        integer count;
        begin
            case(job) 0:count=3;1:count=4;2:count=2;3:count=3;endcase
            while(b_count!=begin_b+count || !transport_idle) @(negedge clk);
            if(job==2) begin
                if(failed || error_code!=0 || line_valid || line_count!=begin_line || valid_bytes!==64'hffffffff)
                    $fatal(1,"NTRPU_LINK_PARTIAL_BEFORE_STOP");
            end else begin

                while(dest_b!=begin_dest+1 || bridge_phase!=0 || peer_axi_busy || !guard_idle) @(negedge clk);
                golden[job==0 ? 0 : job==3 ? 2 : 1]=expected_line(job);
                written[job==0 ? 0 : job==3 ? 2 : 1]=1;

            end
            if(device_aw!=aw_count || device_w!=w_count || device_b!=b_count ||
               aw_count!=b_count || w_count!=b_count) $fatal(1,"NTRPU_NATIVE_CDC_BARRIER_DRAIN");
        end
    endtask
    task publish_job(input integer job);
        begin
            repeat(4) @(negedge clk);
            if(result_valid) $fatal(1,"NTRPU_LINK_BARRIER_REQUIRED");
            source_stopped=1;ingress_isolated=1;quiesce_epoch=epoch-1;
            repeat(4) @(negedge clk);
            if(result_valid) $fatal(1,"NTRPU_LINK_STALE_BARRIER");
            quiesce_epoch=epoch;
            while(!result_valid) @(negedge clk);
            if(result_status!==(job==2 ? 4'd1 : 4'd0) || result_epoch!==epoch || result_tag!==arm_tag)
                $fatal(1,"NTRPU_LINK_RESULT");
            repeat(5) @(negedge clk);
        end
    endtask
    task after_retire(input integer job);
        begin
            if(result_count!=begin_result+1 || !transport_idle) $fatal(1,"NTRPU_LINK_RETIRE");
            readback();
        end
    endtask
    task check_complete;
        begin
            verify_periodic_complete();
            if(rejected_frames!=4 || rejected_identity!=2 || last_reject_code!=3 || last_reject_faults!=4)
                $fatal(1,"NTRPU_NATIVE_FRAGMENT_TELEMETRY");
            if(control_aw!=82 || control_w!=82 || control_b!=82 || control_ar!=73 || control_r!=73 || protocol_fault) $fatal(1,"NTRPU_NATIVE_CONTROL_DRAIN");
            if(aw_count!=12 || w_count!=12 || b_count!=12 || line_count!=3 || result_count!=4 ||
               !transport_idle || line_valid || result_valid || done_ready || peer_axi_busy || !guard_idle ||
               device_aw!=12 || device_w!=12 || device_b!=12 ||
               dest_aw!=3 || dest_w!=3 || dest_b!=3 || read_count!=8 || ar_count!=8 || m_axi_rvalid) $fatal(1,"NTRPU_LINK_DRAIN");
            $display("NTRPU_NATIVE_MIG_PASS jobs=4 writes=3 write_bytes=192 reads=8 read_bytes=512 source=12 ps=%0d",$time);
            $display("NTRPU_LINK_COUNTS aw=%0d w=%0d b=%0d lines=%0d results=%0d aw_stall=%0d w_stall=%0d b_stall=%0d ps=%0d",
                     aw_count,w_count,b_count,line_count,result_count,aw_stall,w_stall,b_stall,$time);
        end
    endtask
    always @(posedge clk) if(rst_n) begin
        cycles<=cycles+1;

        if(epoch!==peer_epoch || (result_valid && (peer_axi_busy || !guard_idle))) $fatal(1,"NTRPU_PEER_LINK_EPOCH_OR_DRAIN");
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

    always @(posedge host_clk) if(host_rst_n) begin
        if(board.EP.m_axi_awvalid || board.EP.m_axi_arvalid || board.EP.m_axi_wvalid || board.EP.m_axib_arvalid)
            $fatal(1,"NTRPU_LINK_UNEXPECTED_DMA_OR_READ");
        if(board.EP.m_axib_awvalid && !awready) aw_stall<=aw_stall+1;
        if(board.EP.m_axib_wvalid && !wready) w_stall<=w_stall+1;
        if(bvalid && !board.EP.m_axib_bready) b_stall<=b_stall+1;
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
            if(bid!==0 || bresp!==(b_count==begin_b ? (current_job==3 ? 2'd3 : 2'd2) : 2'd0)) $fatal(1,"NTRPU_LINK_B_RESPONSE");
            $display("NTRPU_LINK_B id=%h resp=%0d ps=%0d",bid,bresp,$time);
        end
    end
    task automatic readback;
        begin
            for(integer slot=0;slot<3;slot++) if(written[slot]) begin
                @(negedge clk);m_axi_araddr=37'(location(slot));m_axi_arvalid=1;
                do @(posedge clk); while(!m_axi_arready);
                @(negedge clk);m_axi_arvalid=0;
                while(!m_axi_rvalid) @(negedge clk);
                if(m_axi_rdata!==golden[slot] || m_axi_rid!==0 || m_axi_rresp!==0 || m_axi_rlast!==1)
                    $fatal(1,"NTRPU_NATIVE_MIG_READBACK");
                repeat(3) @(negedge clk);m_axi_rready=1;
                @(posedge clk);@(negedge clk);m_axi_rready=0;
            end
        end
    endtask
    always @(posedge clk) begin
        if(calibrated) begin
            if(calibration_seen && $time-last_ui_ps!=5000) $fatal(1,"NTRPU_NATIVE_MIG_UI_CLOCK");
            last_ui_ps=$time;calibration_seen=1;
        end
        if(m_axi_awvalid || m_axi_wvalid || m_axi_arvalid)
            if(!calibrated || ui_rst || !rst_n) $fatal(1,"NTRPU_NATIVE_MIG_EARLY_TRAFFIC");
        if(rst_n) begin
            if(!calibrated || ui_rst || interrupt_ecc || !board.EP.user_resetn) $fatal(1,"NTRPU_NATIVE_MIG_STATE");
            if(r_stalled && (!m_axi_rvalid || {m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata}!==held_r))
                $fatal(1,"NTRPU_NATIVE_MIG_R_STABILITY");
            r_stalled=m_axi_rvalid && !m_axi_rready;held_r={m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata};
            if(dut.data_path.d_axi_awvalid && dut.data_path.d_axi_awready) device_aw++;
            if(dut.data_path.d_axi_wvalid && dut.data_path.d_axi_wready) device_w++;
            if(dut.data_path.d_axi_bvalid && dut.data_path.d_axi_bready) device_b++;
            if(m_axi_arvalid && m_axi_arready) begin
                ar_count++;
                $display("NTRPU_NATIVE_MIG_AR job=%0d addr=%h ps=%0d",current_job,m_axi_araddr,$time);
            end
            if(m_axi_rvalid && m_axi_rready) begin
                read_count++;
                $display("NTRPU_NATIVE_MIG_R job=%0d addr=%h data=%h id=%h resp=%0d last=%b ps=%0d",current_job,m_axi_araddr,m_axi_rdata,m_axi_rid,m_axi_rresp,m_axi_rlast,$time);
            end
        end
    end
    always @(calibrated or ui_rst)
        $strobe("NTRPU_NATIVE_MIG_STATE ps=%0d calib=%b ui_rst=%b rst=%b",$time,calibrated,ui_rst,rst_n);
    always @(posedge clk) if(rst_n && dut.data_path.d_axi_awvalid && dut.data_path.d_axi_awready)
        $display("NTRPU_NATIVE_MIG_DAW addr=%h id=%h len=%0d size=%0d burst=%0d lock=%b ps=%0d",dut.data_path.d_axi_awaddr,dut.data_path.d_axi_awid,dut.data_path.d_axi_awlen,dut.data_path.d_axi_awsize,dut.data_path.d_axi_awburst,dut.data_path.d_axi_awlock,$time);
    always @(posedge clk) if(rst_n && dut.data_path.d_axi_wvalid && dut.data_path.d_axi_wready)
        $display("NTRPU_NATIVE_MIG_DW data=%h strb=%h last=%b ps=%0d",dut.data_path.d_axi_wdata,dut.data_path.d_axi_wstrb,dut.data_path.d_axi_wlast,$time);
    always @(posedge clk) if(rst_n && dut.data_path.d_axi_bvalid && dut.data_path.d_axi_bready)
        $display("NTRPU_NATIVE_MIG_DB id=%h resp=%0d ps=%0d",dut.data_path.d_axi_bid,dut.data_path.d_axi_bresp,$time);
    always @(posedge host_clk) if(host_rst_n) begin
        if(protocol_fault) $fatal(1,"NTRPU_NATIVE_CONTROL_FAULT");
        if(board.EP.m_axil_awvalid && board.EP.m_axil_awready) begin
            control_aw++;
            $display("NTRPU_NATIVE_CONTROL_HOST_AW awaddr=%h ps=%0d",board.EP.m_axil_awaddr,$time);
        end
        if(board.EP.m_axil_wvalid && board.EP.m_axil_wready) begin
            control_w++;
            $display("NTRPU_NATIVE_CONTROL_HOST_W wdata=%h wstrb=%h ps=%0d",board.EP.m_axil_wdata,board.EP.m_axil_wstrb,$time);
        end
        if(board.EP.m_axil_bvalid && board.EP.m_axil_bready) begin
            control_b++;
            $display("NTRPU_NATIVE_CONTROL_HOST_B bresp=%h ps=%0d",board.EP.m_axil_bresp,$time);
        end
        if(board.EP.m_axil_arvalid && board.EP.m_axil_arready) begin
            control_ar++;
            $display("NTRPU_NATIVE_CONTROL_HOST_AR araddr=%h ps=%0d",board.EP.m_axil_araddr,$time);
        end
        if(board.EP.m_axil_rvalid && board.EP.m_axil_rready) begin
            control_r++;
            $display("NTRPU_NATIVE_CONTROL_HOST_R rdata=%h rresp=%h ps=%0d",board.EP.m_axil_rdata,board.EP.m_axil_rresp,$time);
        end
    end
    always @(posedge clk) if(rst_n) begin
        if(dut.d_axil_awvalid && dut.d_axil_awready) begin
            $display("NTRPU_NATIVE_CONTROL_DEVICE_AW awaddr=%h ps=%0d",dut.d_axil_awaddr,$time);
        end
        if(dut.d_axil_wvalid && dut.d_axil_wready) begin
            $display("NTRPU_NATIVE_CONTROL_DEVICE_W wdata=%h wstrb=%h ps=%0d",dut.d_axil_wdata,dut.d_axil_wstrb,$time);
        end
        if(dut.d_axil_bvalid && dut.d_axil_bready) begin
            $display("NTRPU_NATIVE_CONTROL_DEVICE_B bresp=%h ps=%0d",dut.d_axil_bresp,$time);
        end
        if(dut.d_axil_arvalid && dut.d_axil_arready) begin
            $display("NTRPU_NATIVE_CONTROL_DEVICE_AR araddr=%h ps=%0d",dut.d_axil_araddr,$time);
        end
        if(dut.d_axil_rvalid && dut.d_axil_rready) begin
            $display("NTRPU_NATIVE_CONTROL_DEVICE_R rdata=%h rresp=%h ps=%0d",dut.d_axil_rdata,dut.d_axil_rresp,$time);
        end
    end
    initial begin #700000000;$fatal(1,"NTRPU_LINK_TIMEOUT");end
    `include "ntrpu_ddr_periodic_ingress.vh"
endmodule
