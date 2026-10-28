// Three-clock recovery island for ABI2 FFN, both AMD converters and IRQ.
// All attached AXI peers must follow these resets and acknowledge recovery.
// Management clock must be independent and continuous. See SUPERVISED_FFN.md.
module ntrpu_ffn_supervised_fabric #(parameter integer WATCHDOG_CYCLES=65536,parameter bit OUTPUT_FP32=1) (
    input wire mgmt_clk,external_reset_async,rearm,
    // Registered acknowledgements in host_clk and clk respectively. Held
    // means the peer has cleared its protocol state, not just seen a pin.
    // Ready means operational capability; it must stay high while busy.
    input wire host_peer_held,host_peer_ready,memory_peer_held,memory_peer_ready,
    output wire peers_reset_req,rearm_ready,running,host_admit,device_admit,
    output wire peers_held,peers_ready,host_alive,device_alive,
input logic clk,
    output wire rst_n,
    input logic host_clk,usr_irq_ack,
    output wire host_rst_n,
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
    wire irq_source_reset_drained,irq_host_reset_drained;
    wire irq_source_operational,irq_host_operational;
    wire [7:0] status_local,status_mgmt;
    assign status_local={memory_peer_ready,host_peer_ready,irq_host_operational,irq_source_operational,
                         memory_peer_held,host_peer_held,irq_host_reset_drained,irq_source_reset_drained};
    for (genvar i=0;i<8;i++) begin : status_cdc
        ntrpu_cdc_level_qualifier sync (
            .src_level(status_local[i]),.dest_clk(mgmt_clk),.dest_level(status_mgmt[i]));
    end
    assign peers_held=&status_mgmt[3:0];
    assign peers_ready=&status_mgmt[7:4];
    ntrpu_reset_supervisor supervisor (.device_clk(clk),.device_reset_n(rst_n),.host_reset_n(host_rst_n),.*);
    wire  raw_s_axil_awvalid;
    assign raw_s_axil_awvalid=host_admit && s_axil_awvalid;
    wire  raw_s_axil_wvalid;
    assign raw_s_axil_wvalid=host_admit && s_axil_wvalid;
    wire  raw_s_axil_bready;
    assign raw_s_axil_bready=host_admit && s_axil_bready;
    wire  raw_s_axil_arvalid;
    assign raw_s_axil_arvalid=host_admit && s_axil_arvalid;
    wire  raw_s_axil_rready;
    assign raw_s_axil_rready=host_admit && s_axil_rready;
    wire  raw_s_axil_awready;
    assign s_axil_awready=host_admit && raw_s_axil_awready;
    wire  raw_s_axil_wready;
    assign s_axil_wready=host_admit && raw_s_axil_wready;
    wire  raw_s_axil_bvalid;
    assign s_axil_bvalid=host_admit && raw_s_axil_bvalid;
    wire  raw_s_axil_arready;
    assign s_axil_arready=host_admit && raw_s_axil_arready;
    wire  raw_s_axil_rvalid;
    assign s_axil_rvalid=host_admit && raw_s_axil_rvalid;
    wire  raw_irq;
    assign irq=host_admit && raw_irq;
    wire  raw_m_axi_arvalid;
    assign m_axi_arvalid=device_admit && raw_m_axi_arvalid;
    wire  raw_m_axi_arready;
    assign raw_m_axi_arready=device_admit && m_axi_arready;
    wire  raw_m_axi_rvalid;
    assign raw_m_axi_rvalid=device_admit && m_axi_rvalid;
    wire  raw_m_axi_rready;
    assign m_axi_rready=device_admit && raw_m_axi_rready;
    wire  raw_m_axi_awvalid;
    assign m_axi_awvalid=device_admit && raw_m_axi_awvalid;
    wire  raw_m_axi_awready;
    assign raw_m_axi_awready=device_admit && m_axi_awready;
    wire  raw_m_axi_wvalid;
    assign m_axi_wvalid=device_admit && raw_m_axi_wvalid;
    wire  raw_m_axi_wready;
    assign raw_m_axi_wready=device_admit && m_axi_wready;
    wire  raw_m_axi_bvalid;
    assign raw_m_axi_bvalid=device_admit && m_axi_bvalid;
    wire  raw_m_axi_bready;
    assign m_axi_bready=device_admit && raw_m_axi_bready;
    wire  raw_s_host_axi_arvalid;
    assign raw_s_host_axi_arvalid=host_admit && s_host_axi_arvalid;
    wire  raw_s_host_axi_awvalid;
    assign raw_s_host_axi_awvalid=host_admit && s_host_axi_awvalid;
    wire  raw_s_host_axi_arready;
    assign s_host_axi_arready=host_admit && raw_s_host_axi_arready;
    wire  raw_s_host_axi_awready;
    assign s_host_axi_awready=host_admit && raw_s_host_axi_awready;
    wire  raw_s_host_axi_rvalid;
    assign s_host_axi_rvalid=host_admit && raw_s_host_axi_rvalid;
    wire  raw_s_host_axi_rready;
    assign raw_s_host_axi_rready=host_admit && s_host_axi_rready;
    wire  raw_s_host_axi_wvalid;
    assign raw_s_host_axi_wvalid=host_admit && s_host_axi_wvalid;
    wire  raw_s_host_axi_wready;
    assign s_host_axi_wready=host_admit && raw_s_host_axi_wready;
    wire  raw_s_host_axi_bvalid;
    assign s_host_axi_bvalid=host_admit && raw_s_host_axi_bvalid;
    wire  raw_s_host_axi_bready;
    assign raw_s_host_axi_bready=host_admit && s_host_axi_bready;
    ntrpu_ffn_async_fabric #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32),.EXTERNAL_RETIRE(1)) fabric(
        .clk(clk),
        .rst_n(rst_n),
        .host_clk(host_clk),
        .host_rst_n(host_rst_n),
        .usr_irq_ack(usr_irq_ack),
        .s_axil_awaddr(s_axil_awaddr),
        .s_axil_wdata(s_axil_wdata),
        .s_axil_araddr(s_axil_araddr),
        .s_axil_awprot(s_axil_awprot),
        .s_axil_arprot(s_axil_arprot),
        .s_axil_wstrb(s_axil_wstrb),
        .s_axil_awvalid(raw_s_axil_awvalid),
        .s_axil_wvalid(raw_s_axil_wvalid),
        .s_axil_bready(raw_s_axil_bready),
        .s_axil_arvalid(raw_s_axil_arvalid),
        .s_axil_rready(raw_s_axil_rready),
        .s_axil_awready(raw_s_axil_awready),
        .s_axil_wready(raw_s_axil_wready),
        .s_axil_bvalid(raw_s_axil_bvalid),
        .s_axil_arready(raw_s_axil_arready),
        .s_axil_rvalid(raw_s_axil_rvalid),
        .s_axil_bresp(s_axil_bresp),
        .s_axil_rresp(s_axil_rresp),
        .s_axil_rdata(s_axil_rdata),
        .irq(raw_irq),
        .activation_selected(activation_selected),
        .m_axi_araddr(m_axi_araddr),
        .m_axi_arlen(m_axi_arlen),
        .m_axi_arsize(m_axi_arsize),
        .m_axi_arprot(m_axi_arprot),
        .m_axi_arburst(m_axi_arburst),
        .m_axi_arid(m_axi_arid),
        .m_axi_arcache(m_axi_arcache),
        .m_axi_arlock(m_axi_arlock),
        .m_axi_arvalid(raw_m_axi_arvalid),
        .m_axi_arready(raw_m_axi_arready),
        .m_axi_rdata(m_axi_rdata),
        .m_axi_rresp(m_axi_rresp),
        .m_axi_rid(m_axi_rid),
        .m_axi_rlast(m_axi_rlast),
        .m_axi_rvalid(raw_m_axi_rvalid),
        .m_axi_rready(raw_m_axi_rready),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awid(m_axi_awid),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(raw_m_axi_awvalid),
        .m_axi_awready(raw_m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(raw_m_axi_wvalid),
        .m_axi_wready(raw_m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(raw_m_axi_bvalid),
        .m_axi_bready(raw_m_axi_bready),
        .s_host_axi_araddr(s_host_axi_araddr),
        .s_host_axi_awaddr(s_host_axi_awaddr),
        .s_host_axi_arlen(s_host_axi_arlen),
        .s_host_axi_awlen(s_host_axi_awlen),
        .s_host_axi_arsize(s_host_axi_arsize),
        .s_host_axi_arprot(s_host_axi_arprot),
        .s_host_axi_awsize(s_host_axi_awsize),
        .s_host_axi_awprot(s_host_axi_awprot),
        .s_host_axi_arburst(s_host_axi_arburst),
        .s_host_axi_awburst(s_host_axi_awburst),
        .s_host_axi_arid(s_host_axi_arid),
        .s_host_axi_arcache(s_host_axi_arcache),
        .s_host_axi_awid(s_host_axi_awid),
        .s_host_axi_awcache(s_host_axi_awcache),
        .s_host_axi_arlock(s_host_axi_arlock),
        .s_host_axi_arvalid(raw_s_host_axi_arvalid),
        .s_host_axi_awlock(s_host_axi_awlock),
        .s_host_axi_awvalid(raw_s_host_axi_awvalid),
        .s_host_axi_arready(raw_s_host_axi_arready),
        .s_host_axi_awready(raw_s_host_axi_awready),
        .s_host_axi_rdata(s_host_axi_rdata),
        .s_host_axi_rresp(s_host_axi_rresp),
        .s_host_axi_rid(s_host_axi_rid),
        .s_host_axi_rlast(s_host_axi_rlast),
        .s_host_axi_rvalid(raw_s_host_axi_rvalid),
        .s_host_axi_rready(raw_s_host_axi_rready),
        .s_host_axi_wdata(s_host_axi_wdata),
        .s_host_axi_wstrb(s_host_axi_wstrb),
        .s_host_axi_wlast(s_host_axi_wlast),
        .s_host_axi_wvalid(raw_s_host_axi_wvalid),
        .s_host_axi_wready(raw_s_host_axi_wready),
        .s_host_axi_bid(s_host_axi_bid),
        .s_host_axi_bresp(s_host_axi_bresp),
        .s_host_axi_bvalid(raw_s_host_axi_bvalid),
        .s_host_axi_bready(raw_s_host_axi_bready),
        .irq_source_reset_drained(irq_source_reset_drained),
        .irq_host_reset_drained(irq_host_reset_drained),
        .irq_source_operational(irq_source_operational),
        .irq_host_operational(irq_host_operational));
endmodule
