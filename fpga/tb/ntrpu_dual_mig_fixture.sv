`timescale 1ns/1ps
module ntrpu_dual_mig_fixture #(parameter integer PORT=0) (
    output wire clk, calibrated, ui_rst, interrupt_ecc,
    input wire rst_n, hold_w, hold_b,
    input wire m_axi_awvalid, m_axi_wvalid, m_axi_wlast, m_axi_bready,
    output wire m_axi_awready, m_axi_wready, m_axi_bvalid,
    input wire [36:0] m_axi_awaddr,
    input wire [7:0] m_axi_awlen,
    input wire [3:0] m_axi_awid, m_axi_awcache,
    input wire [2:0] m_axi_awsize,
    input wire [1:0] m_axi_awburst,
    input wire [511:0] m_axi_wdata,
    input wire [63:0] m_axi_wstrb,
    output wire [3:0] m_axi_bid,
    output wire [1:0] m_axi_bresp,
    input wire [36:0] m_axi_araddr,
    input wire m_axi_arvalid, m_axi_rready,
    output wire m_axi_arready, m_axi_rvalid, m_axi_rlast,
    output wire [511:0] m_axi_rdata,
    output wire [1:0] m_axi_rresp,
    output wire [3:0] m_axi_rid
);
    import arch_package::*;
    wire [2:0] m_axi_awprot=0,m_axi_arprot=0,m_axi_arsize=6;
    wire m_axi_awlock=0,m_axi_arlock=0;
    wire [3:0] m_axi_arid=0,m_axi_arcache=0;
    wire [7:0] m_axi_arlen=0;
    wire [1:0] m_axi_arburst=1;
    wire mig_wready,mig_bvalid;
    assign m_axi_wready=mig_wready && !hold_w;
    assign m_axi_bvalid=mig_bvalid && !hold_b;
// Cold generic MIG/model wiring copied from the qualified ingress fixture.
    logic sys_clk=0,sys_rst=1;
    initial begin #(PORT*0.731);forever #2 sys_clk=~sys_clk;end
    initial begin #(PORT==0 ? 20 : 6020);sys_rst=0;end
    localparam longint unsigned CAPACITY=64'd137438953472;
    longint unsigned rank_acts[4]='{default:0};
    bit any_axi=0,calibration_seen=0,complete=0;
    longint last_ui_ps=0,ui_samples=0;
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
        .c0_ddr4_s_axi_wvalid(m_axi_wvalid && !hold_w), .c0_ddr4_s_axi_wready(mig_wready),
        .c0_ddr4_s_axi_bready(m_axi_bready && !hold_b), .c0_ddr4_s_axi_bid(m_axi_bid), .c0_ddr4_s_axi_bresp(m_axi_bresp),
        .c0_ddr4_s_axi_bvalid(mig_bvalid),
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
            $display("DUAL_RDIMM_RESET port=%0d slot=%0d time_ps=%0d reset_n=%b last_low_ps=%0d",
                     PORT, slot, longint'($realtime*1000), memory.reset_n, last_reset_width*1000);
        end
        always @(posedge ck_t[slot]) begin
            if (calibrated && !act_n)
                for (int rank = 0; rank < 2; rank++)
                    if (cs_n[slot*2+rank] === 1'b0) begin
                        rank_acts[slot*2+rank]++;
                        $display("DUAL_ACT port=%0d ps=%0d rank=%0d count=%0d",PORT,longint'($realtime*1000),slot*2+rank,rank_acts[slot*2+rank]);
                    end
        end
    end




    `include "ntrpu_dual_mig_periodic.vh"
endmodule
