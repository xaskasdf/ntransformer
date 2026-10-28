// Two-clock integration of the qualified FFN fabric and AMD AXI converters.
// clk/rst_n cover fabric and memory; host_clk/host_rst_n cover its initiator.
// Both domains and all peers MUST share coordinated resets (see PG059).
// This wrapper does not detect clock loss or implement the reset supervisor.
module ntrpu_ffn_async_fabric #(parameter integer WATCHDOG_CYCLES=65536,parameter bit OUTPUT_FP32=1,EXTERNAL_RETIRE=0) (
`ifdef NTRPU_SUPERVISED_FFN
    output wire irq_source_reset_drained,irq_host_reset_drained,
    output wire irq_source_operational,irq_host_operational,
`endif
input logic clk,rst_n,
    input logic host_clk,host_rst_n,usr_irq_ack,
    input logic [31:0] s_axil_awaddr,s_axil_wdata,s_axil_araddr,
    input logic [2:0] s_axil_awprot,s_axil_arprot,
    input logic [3:0] s_axil_wstrb,
    input logic s_axil_awvalid,s_axil_wvalid,s_axil_bready,s_axil_arvalid,s_axil_rready,
    output logic s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid,
    output logic [1:0] s_axil_bresp,s_axil_rresp,
    output logic [31:0] s_axil_rdata,
    output logic irq,
    output logic activation_selected,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid, m_axi_arcache,
    output logic m_axi_arlock, m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid, m_axi_awcache,
    output logic m_axi_awlock, m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready,
    input logic[63:0] s_host_axi_araddr, s_host_axi_awaddr,
    input logic[7:0] s_host_axi_arlen, s_host_axi_awlen,
    input logic[2:0] s_host_axi_arsize, s_host_axi_arprot, s_host_axi_awsize, s_host_axi_awprot,
    input logic[1:0] s_host_axi_arburst, s_host_axi_awburst,
    input logic[3:0] s_host_axi_arid, s_host_axi_arcache, s_host_axi_awid, s_host_axi_awcache,
    input logic s_host_axi_arlock, s_host_axi_arvalid, s_host_axi_awlock, s_host_axi_awvalid,
    output logic s_host_axi_arready, s_host_axi_awready,
    output logic[511:0] s_host_axi_rdata,
    output logic[1:0] s_host_axi_rresp,
    output logic[3:0] s_host_axi_rid,
    output logic s_host_axi_rlast, s_host_axi_rvalid,
    input logic s_host_axi_rready,
    input logic[511:0] s_host_axi_wdata,
    input logic[63:0] s_host_axi_wstrb,
    input logic s_host_axi_wlast, s_host_axi_wvalid,
    output logic s_host_axi_wready,
    output logic[3:0] s_host_axi_bid,
    output logic[1:0] s_host_axi_bresp,
    output logic s_host_axi_bvalid,
    input logic s_host_axi_bready
);
    logic [31:0] d_axil_awaddr;
    logic [31:0] d_axil_wdata;
    logic [31:0] d_axil_araddr;
    logic [2:0] d_axil_awprot;
    logic [2:0] d_axil_arprot;
    logic [3:0] d_axil_wstrb;
    logic  d_axil_awvalid;
    logic  d_axil_wvalid;
    logic  d_axil_bready;
    logic  d_axil_arvalid;
    logic  d_axil_rready;
    logic  d_axil_awready;
    logic  d_axil_wready;
    logic  d_axil_bvalid;
    logic  d_axil_arready;
    logic  d_axil_rvalid;
    logic [1:0] d_axil_bresp;
    logic [1:0] d_axil_rresp;
    logic [31:0] d_axil_rdata;
    logic [63:0] d_host_axi_araddr;
    logic [63:0] d_host_axi_awaddr;
    logic [7:0] d_host_axi_arlen;
    logic [7:0] d_host_axi_awlen;
    logic [2:0] d_host_axi_arsize;
    logic [2:0] d_host_axi_arprot;
    logic [2:0] d_host_axi_awsize;
    logic [2:0] d_host_axi_awprot;
    logic [1:0] d_host_axi_arburst;
    logic [1:0] d_host_axi_awburst;
    logic [3:0] d_host_axi_arid;
    logic [3:0] d_host_axi_arcache;
    logic [3:0] d_host_axi_awid;
    logic [3:0] d_host_axi_awcache;
    logic  d_host_axi_arlock;
    logic  d_host_axi_arvalid;
    logic  d_host_axi_awlock;
    logic  d_host_axi_awvalid;
    logic  d_host_axi_arready;
    logic  d_host_axi_awready;
    logic [511:0] d_host_axi_rdata;
    logic [1:0] d_host_axi_rresp;
    logic [3:0] d_host_axi_rid;
    logic  d_host_axi_rlast;
    logic  d_host_axi_rvalid;
    logic  d_host_axi_rready;
    logic [511:0] d_host_axi_wdata;
    logic [63:0] d_host_axi_wstrb;
    logic  d_host_axi_wlast;
    logic  d_host_axi_wvalid;
    logic  d_host_axi_wready;
    logic [3:0] d_host_axi_bid;
    logic [1:0] d_host_axi_bresp;
    logic  d_host_axi_bvalid;
    logic  d_host_axi_bready;
    logic device_irq,irq_sample;
    generate if(EXTERNAL_RETIRE) begin : with_irq
        // Inner fabric already produces the retained request in host_clk.
        assign irq=device_irq;
    end else begin : legacy
    // CSR retains completion until its MMIO ACK; no pulse crosses domains.
    xpm_cdc_single #(.DEST_SYNC_FF(3),.INIT_SYNC_FF(0),.SIM_ASSERT_CHK(1),.SRC_INPUT_REG(1)) irq_cdc(
        .src_clk(clk),.src_in(device_irq),.dest_clk(host_clk),.dest_out(irq_sample));
    assign irq=host_rst_n && irq_sample;
    end endgenerate
    ntrpu_host_axi_cdc memory_cdc(
        .s_axi_aclk(host_clk),
        .s_axi_aresetn(host_rst_n),
        .m_axi_aclk(clk),
        .m_axi_aresetn(rst_n),
        .s_axi_araddr(s_host_axi_araddr),
        .m_axi_araddr(d_host_axi_araddr),
        .s_axi_awaddr(s_host_axi_awaddr),
        .m_axi_awaddr(d_host_axi_awaddr),
        .s_axi_arlen(s_host_axi_arlen),
        .m_axi_arlen(d_host_axi_arlen),
        .s_axi_awlen(s_host_axi_awlen),
        .m_axi_awlen(d_host_axi_awlen),
        .s_axi_arsize(s_host_axi_arsize),
        .m_axi_arsize(d_host_axi_arsize),
        .s_axi_arprot(s_host_axi_arprot),
        .m_axi_arprot(d_host_axi_arprot),
        .s_axi_awsize(s_host_axi_awsize),
        .m_axi_awsize(d_host_axi_awsize),
        .s_axi_awprot(s_host_axi_awprot),
        .m_axi_awprot(d_host_axi_awprot),
        .s_axi_arburst(s_host_axi_arburst),
        .m_axi_arburst(d_host_axi_arburst),
        .s_axi_awburst(s_host_axi_awburst),
        .m_axi_awburst(d_host_axi_awburst),
        .s_axi_arid(s_host_axi_arid),
        .m_axi_arid(d_host_axi_arid),
        .s_axi_arcache(s_host_axi_arcache),
        .m_axi_arcache(d_host_axi_arcache),
        .s_axi_awid(s_host_axi_awid),
        .m_axi_awid(d_host_axi_awid),
        .s_axi_awcache(s_host_axi_awcache),
        .m_axi_awcache(d_host_axi_awcache),
        .s_axi_arlock(s_host_axi_arlock),
        .m_axi_arlock(d_host_axi_arlock),
        .s_axi_arvalid(s_host_axi_arvalid),
        .m_axi_arvalid(d_host_axi_arvalid),
        .s_axi_awlock(s_host_axi_awlock),
        .m_axi_awlock(d_host_axi_awlock),
        .s_axi_awvalid(s_host_axi_awvalid),
        .m_axi_awvalid(d_host_axi_awvalid),
        .s_axi_arready(s_host_axi_arready),
        .m_axi_arready(d_host_axi_arready),
        .s_axi_awready(s_host_axi_awready),
        .m_axi_awready(d_host_axi_awready),
        .s_axi_rdata(s_host_axi_rdata),
        .m_axi_rdata(d_host_axi_rdata),
        .s_axi_rresp(s_host_axi_rresp),
        .m_axi_rresp(d_host_axi_rresp),
        .s_axi_rid(s_host_axi_rid),
        .m_axi_rid(d_host_axi_rid),
        .s_axi_rlast(s_host_axi_rlast),
        .m_axi_rlast(d_host_axi_rlast),
        .s_axi_rvalid(s_host_axi_rvalid),
        .m_axi_rvalid(d_host_axi_rvalid),
        .s_axi_rready(s_host_axi_rready),
        .m_axi_rready(d_host_axi_rready),
        .s_axi_wdata(s_host_axi_wdata),
        .m_axi_wdata(d_host_axi_wdata),
        .s_axi_wstrb(s_host_axi_wstrb),
        .m_axi_wstrb(d_host_axi_wstrb),
        .s_axi_wlast(s_host_axi_wlast),
        .m_axi_wlast(d_host_axi_wlast),
        .s_axi_wvalid(s_host_axi_wvalid),
        .m_axi_wvalid(d_host_axi_wvalid),
        .s_axi_wready(s_host_axi_wready),
        .m_axi_wready(d_host_axi_wready),
        .s_axi_bid(s_host_axi_bid),
        .m_axi_bid(d_host_axi_bid),
        .s_axi_bresp(s_host_axi_bresp),
        .m_axi_bresp(d_host_axi_bresp),
        .s_axi_bvalid(s_host_axi_bvalid),
        .m_axi_bvalid(d_host_axi_bvalid),
        .s_axi_bready(s_host_axi_bready),
        .m_axi_bready(d_host_axi_bready),
        .s_axi_awqos(4'b0),
        .m_axi_awqos(),
        .s_axi_awregion(4'b0),
        .m_axi_awregion(),
        .s_axi_arqos(4'b0),
        .m_axi_arqos(),
        .s_axi_arregion(4'b0),
        .m_axi_arregion());
    ntrpu_axil_cdc_recode control_cdc(
        .s_axi_aclk(host_clk),
        .s_axi_aresetn(host_rst_n),
        .m_axi_aclk(clk),
        .m_axi_aresetn(rst_n),
        .s_axi_awaddr(s_axil_awaddr),
        .m_axi_awaddr(d_axil_awaddr),
        .s_axi_wdata(s_axil_wdata),
        .m_axi_wdata(d_axil_wdata),
        .s_axi_araddr(s_axil_araddr),
        .m_axi_araddr(d_axil_araddr),
        .s_axi_awprot(s_axil_awprot),
        .m_axi_awprot(d_axil_awprot),
        .s_axi_arprot(s_axil_arprot),
        .m_axi_arprot(d_axil_arprot),
        .s_axi_wstrb(s_axil_wstrb),
        .m_axi_wstrb(d_axil_wstrb),
        .s_axi_awvalid(s_axil_awvalid),
        .m_axi_awvalid(d_axil_awvalid),
        .s_axi_wvalid(s_axil_wvalid),
        .m_axi_wvalid(d_axil_wvalid),
        .s_axi_bready(s_axil_bready),
        .m_axi_bready(d_axil_bready),
        .s_axi_arvalid(s_axil_arvalid),
        .m_axi_arvalid(d_axil_arvalid),
        .s_axi_rready(s_axil_rready),
        .m_axi_rready(d_axil_rready),
        .s_axi_awready(s_axil_awready),
        .m_axi_awready(d_axil_awready),
        .s_axi_wready(s_axil_wready),
        .m_axi_wready(d_axil_wready),
        .s_axi_bvalid(s_axil_bvalid),
        .m_axi_bvalid(d_axil_bvalid),
        .s_axi_arready(s_axil_arready),
        .m_axi_arready(d_axil_arready),
        .s_axi_rvalid(s_axil_rvalid),
        .m_axi_rvalid(d_axil_rvalid),
        .s_axi_bresp(s_axil_bresp),
        .m_axi_bresp(d_axil_bresp),
        .s_axi_rresp(s_axil_rresp),
        .m_axi_rresp(d_axil_rresp),
        .s_axi_rdata(s_axil_rdata),
        .m_axi_rdata(d_axil_rdata));
    ntrpu_ffn_fabric #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32),.EXTERNAL_RETIRE(EXTERNAL_RETIRE)) fabric(
`ifdef NTRPU_SUPERVISED_FFN
        .irq_source_reset_drained(irq_source_reset_drained),.irq_host_reset_drained(irq_host_reset_drained),
        .irq_source_operational(irq_source_operational),.irq_host_operational(irq_host_operational),
`endif
        .clk(clk),
        .rst_n(rst_n),
        .host_clk(host_clk),.host_rst_n(host_rst_n),.usr_irq_ack(usr_irq_ack),
        .s_axil_awaddr(d_axil_awaddr),
        .s_axil_wdata(d_axil_wdata),
        .s_axil_araddr(d_axil_araddr),
        .s_axil_awprot(d_axil_awprot),
        .s_axil_arprot(d_axil_arprot),
        .s_axil_wstrb(d_axil_wstrb),
        .s_axil_awvalid(d_axil_awvalid),
        .s_axil_wvalid(d_axil_wvalid),
        .s_axil_bready(d_axil_bready),
        .s_axil_arvalid(d_axil_arvalid),
        .s_axil_rready(d_axil_rready),
        .s_axil_awready(d_axil_awready),
        .s_axil_wready(d_axil_wready),
        .s_axil_bvalid(d_axil_bvalid),
        .s_axil_arready(d_axil_arready),
        .s_axil_rvalid(d_axil_rvalid),
        .s_axil_bresp(d_axil_bresp),
        .s_axil_rresp(d_axil_rresp),
        .s_axil_rdata(d_axil_rdata),
        .irq(device_irq),
        .activation_selected(activation_selected),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arid(m_axi_arid),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arvalid(m_axi_arvalid),
        .m_axi_arready(m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(m_axi_rvalid),
        .m_axi_rready(m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awid(m_axi_awid),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready),
        .s_host_axi_araddr(d_host_axi_araddr),
        .s_host_axi_awaddr(d_host_axi_awaddr),
        .s_host_axi_arlen(d_host_axi_arlen),
        .s_host_axi_awlen(d_host_axi_awlen),
        .s_host_axi_arsize(d_host_axi_arsize),
        .s_host_axi_arprot(d_host_axi_arprot),
        .s_host_axi_awsize(d_host_axi_awsize),
        .s_host_axi_awprot(d_host_axi_awprot),
        .s_host_axi_arburst(d_host_axi_arburst),
        .s_host_axi_awburst(d_host_axi_awburst),
        .s_host_axi_arid(d_host_axi_arid),
        .s_host_axi_arcache(d_host_axi_arcache),
        .s_host_axi_awid(d_host_axi_awid),
        .s_host_axi_awcache(d_host_axi_awcache),
        .s_host_axi_arlock(d_host_axi_arlock),
        .s_host_axi_arvalid(d_host_axi_arvalid),
        .s_host_axi_awlock(d_host_axi_awlock),
        .s_host_axi_awvalid(d_host_axi_awvalid),
        .s_host_axi_arready(d_host_axi_arready),
        .s_host_axi_awready(d_host_axi_awready),
        .s_host_axi_rdata(d_host_axi_rdata),
        .s_host_axi_rresp(d_host_axi_rresp),
        .s_host_axi_rid(d_host_axi_rid),
        .s_host_axi_rlast(d_host_axi_rlast),
        .s_host_axi_rvalid(d_host_axi_rvalid),
        .s_host_axi_rready(d_host_axi_rready),
        .s_host_axi_wdata(d_host_axi_wdata),
        .s_host_axi_wstrb(d_host_axi_wstrb),
        .s_host_axi_wlast(d_host_axi_wlast),
        .s_host_axi_wvalid(d_host_axi_wvalid),
        .s_host_axi_wready(d_host_axi_wready),
        .s_host_axi_bid(d_host_axi_bid),
        .s_host_axi_bresp(d_host_axi_bresp),
        .s_host_axi_bvalid(d_host_axi_bvalid),
        .s_host_axi_bready(d_host_axi_bready));
endmodule
