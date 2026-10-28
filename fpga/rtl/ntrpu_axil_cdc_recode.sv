// AMD AXI-Lite CDC with reversible RRESP payload coding inside the bridge.
// The FFN CSR returns only 00/11 on reads; without coding, synthesis merges
// the two source response registers and the router ignores their skew group.
// Map {b1,b0} -> {b1,b0^b1} on both ends: 00 stays 00, 10/11 swap, 01 stays 01.
// Thus all external response codes, data and handshakes remain unchanged.
// The encoded values are private to the transparent AMD payload transport.
// Resets must be coordinated across both domains and all connected peers.
module ntrpu_axil_cdc_recode (
    input logic s_axi_aclk,s_axi_aresetn,m_axi_aclk,m_axi_aresetn,
    input logic [31:0] s_axi_awaddr,s_axi_wdata,s_axi_araddr,
    input logic [2:0] s_axi_awprot,s_axi_arprot,
    input logic [3:0] s_axi_wstrb,
    input logic s_axi_awvalid,s_axi_wvalid,s_axi_bready,s_axi_arvalid,s_axi_rready,
    output logic s_axi_awready,s_axi_wready,s_axi_bvalid,s_axi_arready,s_axi_rvalid,
    output logic [1:0] s_axi_bresp,s_axi_rresp,
    output logic [31:0] s_axi_rdata,
    output logic [31:0] m_axi_awaddr,m_axi_wdata,m_axi_araddr,
    output logic [2:0] m_axi_awprot,m_axi_arprot,
    output logic [3:0] m_axi_wstrb,
    output logic m_axi_awvalid,m_axi_wvalid,m_axi_bready,m_axi_arvalid,m_axi_rready,
    input logic m_axi_awready,m_axi_wready,m_axi_bvalid,m_axi_arready,m_axi_rvalid,
    input logic [1:0] m_axi_bresp,m_axi_rresp,
    input logic [31:0] m_axi_rdata
);
    logic [1:0] encoded_rresp,received_rresp;
    assign encoded_rresp={m_axi_rresp[1],m_axi_rresp[0]^m_axi_rresp[1]};
    assign s_axi_rresp={received_rresp[1],received_rresp[0]^received_rresp[1]};
    ntrpu_host_axil_cdc bridge(
        .m_axi_rresp(encoded_rresp),.s_axi_rresp(received_rresp),.*);
endmodule
