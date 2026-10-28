// Single-clock FFN plus a privileged host AXI memory port.
// No DDR, PCIe, address translation, CDC or coherency is implied.
module ntrpu_ffn_fabric #(parameter integer WATCHDOG_CYCLES=65536,parameter bit OUTPUT_FP32=1,EXTERNAL_RETIRE=0) (
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
    logic [1:0][63:0] s_axi_araddr, s_axi_awaddr;
    logic [1:0][7:0] s_axi_arlen, s_axi_awlen;
    logic [1:0][2:0] s_axi_arsize, s_axi_arprot, s_axi_awsize, s_axi_awprot;
    logic [1:0][1:0] s_axi_arburst, s_axi_awburst;
    logic [1:0][3:0] s_axi_arid, s_axi_arcache, s_axi_awid, s_axi_awcache;
    logic [1:0] s_axi_arlock, s_axi_arvalid, s_axi_awlock, s_axi_awvalid;
    logic [1:0] s_axi_arready, s_axi_awready;
    logic [1:0][511:0] s_axi_rdata;
    logic [1:0][1:0] s_axi_rresp;
    logic [1:0][3:0] s_axi_rid;
    logic [1:0] s_axi_rlast, s_axi_rvalid;
    logic [1:0] s_axi_rready;
    logic [1:0][511:0] s_axi_wdata;
    logic [1:0][63:0] s_axi_wstrb;
    logic [1:0] s_axi_wlast, s_axi_wvalid;
    logic [1:0] s_axi_wready;
    logic [1:0][3:0] s_axi_bid;
    logic [1:0][1:0] s_axi_bresp;
    logic [1:0] s_axi_bvalid;
    logic [1:0] s_axi_bready;
    assign s_axi_araddr[1]=s_host_axi_araddr;
    assign s_axi_awaddr[1]=s_host_axi_awaddr;
    assign s_axi_arlen[1]=s_host_axi_arlen;
    assign s_axi_awlen[1]=s_host_axi_awlen;
    assign s_axi_arsize[1]=s_host_axi_arsize;
    assign s_axi_arprot[1]=s_host_axi_arprot;
    assign s_axi_awsize[1]=s_host_axi_awsize;
    assign s_axi_awprot[1]=s_host_axi_awprot;
    assign s_axi_arburst[1]=s_host_axi_arburst;
    assign s_axi_awburst[1]=s_host_axi_awburst;
    assign s_axi_arid[1]=s_host_axi_arid;
    assign s_axi_arcache[1]=s_host_axi_arcache;
    assign s_axi_awid[1]=s_host_axi_awid;
    assign s_axi_awcache[1]=s_host_axi_awcache;
    assign s_axi_arlock[1]=s_host_axi_arlock;
    assign s_axi_arvalid[1]=s_host_axi_arvalid;
    assign s_axi_awlock[1]=s_host_axi_awlock;
    assign s_axi_awvalid[1]=s_host_axi_awvalid;
    assign s_host_axi_arready=s_axi_arready[1];
    assign s_host_axi_awready=s_axi_awready[1];
    assign s_host_axi_rdata=s_axi_rdata[1];
    assign s_host_axi_rresp=s_axi_rresp[1];
    assign s_host_axi_rid=s_axi_rid[1];
    assign s_host_axi_rlast=s_axi_rlast[1];
    assign s_host_axi_rvalid=s_axi_rvalid[1];
    assign s_axi_rready[1]=s_host_axi_rready;
    assign s_axi_wdata[1]=s_host_axi_wdata;
    assign s_axi_wstrb[1]=s_host_axi_wstrb;
    assign s_axi_wlast[1]=s_host_axi_wlast;
    assign s_axi_wvalid[1]=s_host_axi_wvalid;
    assign s_host_axi_wready=s_axi_wready[1];
    assign s_host_axi_bid=s_axi_bid[1];
    assign s_host_axi_bresp=s_axi_bresp[1];
    assign s_host_axi_bvalid=s_axi_bvalid[1];
    assign s_axi_bready[1]=s_host_axi_bready;
    ntrpu_ffn_device #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32),.EXTERNAL_RETIRE(EXTERNAL_RETIRE)) device(
        .m_axi_araddr(s_axi_araddr[0]),
        .m_axi_awaddr(s_axi_awaddr[0]),
        .m_axi_arlen(s_axi_arlen[0]),
        .m_axi_awlen(s_axi_awlen[0]),
        .m_axi_arsize(s_axi_arsize[0]),
        .m_axi_arprot(s_axi_arprot[0]),
        .m_axi_awsize(s_axi_awsize[0]),
        .m_axi_awprot(s_axi_awprot[0]),
        .m_axi_arburst(s_axi_arburst[0]),
        .m_axi_awburst(s_axi_awburst[0]),
        .m_axi_arid(s_axi_arid[0]),
        .m_axi_arcache(s_axi_arcache[0]),
        .m_axi_awid(s_axi_awid[0]),
        .m_axi_awcache(s_axi_awcache[0]),
        .m_axi_arlock(s_axi_arlock[0]),
        .m_axi_arvalid(s_axi_arvalid[0]),
        .m_axi_awlock(s_axi_awlock[0]),
        .m_axi_awvalid(s_axi_awvalid[0]),
        .m_axi_arready(s_axi_arready[0]),
        .m_axi_awready(s_axi_awready[0]),
        .m_axi_rdata(s_axi_rdata[0]),
        .m_axi_rresp(s_axi_rresp[0]),
        .m_axi_rid(s_axi_rid[0]),
        .m_axi_rlast(s_axi_rlast[0]),
        .m_axi_rvalid(s_axi_rvalid[0]),
        .m_axi_rready(s_axi_rready[0]),
        .m_axi_wdata(s_axi_wdata[0]),
        .m_axi_wstrb(s_axi_wstrb[0]),
        .m_axi_wlast(s_axi_wlast[0]),
        .m_axi_wvalid(s_axi_wvalid[0]),
        .m_axi_wready(s_axi_wready[0]),
        .m_axi_bid(s_axi_bid[0]),
        .m_axi_bresp(s_axi_bresp[0]),
        .m_axi_bvalid(s_axi_bvalid[0]),
        .m_axi_bready(s_axi_bready[0]),
        .*
    );
    ntrpu_axi_arbiter arbiter(.*);
endmodule

