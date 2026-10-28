// Generated explicit port wiring. Response pause is a test-only backpressure
// shim held constant until reset; it never retracts an offered CDC VALID.
assign raw_request[0][576:82]=0;
assign core_request[0][576:82]=0;
assign raw_response[0]=0;
assign raw_response[1]=0;
assign raw_response[2][576:6]=0;
assign hreturn[2][576:6]=0;
assign raw_request[2]=0;
assign core_request[2]=0;
assign raw_request[3][576:32]=0;
assign core_request[3][576:32]=0;
assign raw_response[3]=0;
assign raw_request[4][576:36]=0;
assign core_request[4][576:36]=0;
assign raw_response[4]=0;
assign raw_response[5][576:2]=0;
assign hreturn[5][576:2]=0;
assign raw_request[5]=0;
assign core_request[5]=0;
assign raw_request[6][576:32]=0;
assign core_request[6][576:32]=0;
assign raw_response[6]=0;
assign raw_response[7][576:34]=0;
assign hreturn[7][576:34]=0;
assign raw_request[7]=0;
assign core_request[7]=0;
ntrpu_ingress_axi_cdc data_cdc (
    .s_axi_aclk(host_clk),
    .s_axi_aresetn(host_rst_n),
    .m_axi_aclk(clk),
    .m_axi_aresetn(boundary_rst_n),
    .s_axi_awvalid(hv[0]),
    .s_axi_awready(hready[0]),
    .m_axi_awvalid(in_valid[0]),
    .m_axi_awready(in_ready[0]),
    .s_axi_awaddr(hdata[0][81:18]),
    .m_axi_awaddr(raw_request[0][81:18]),
    .s_axi_awid(hdata[0][17:14]),
    .m_axi_awid(raw_request[0][17:14]),
    .s_axi_awlen(hdata[0][13:6]),
    .m_axi_awlen(raw_request[0][13:6]),
    .s_axi_awsize(hdata[0][5:3]),
    .m_axi_awsize(raw_request[0][5:3]),
    .s_axi_awburst(hdata[0][2:1]),
    .m_axi_awburst(raw_request[0][2:1]),
    .s_axi_awlock(hdata[0][0]),
    .m_axi_awlock(raw_request[0][0]),
    .s_axi_wvalid(hv[1]),
    .s_axi_wready(hready[1]),
    .m_axi_wvalid(in_valid[1]),
    .m_axi_wready(in_ready[1]),
    .s_axi_wdata(hdata[1][576:65]),
    .m_axi_wdata(raw_request[1][576:65]),
    .s_axi_wstrb(hdata[1][64:1]),
    .m_axi_wstrb(raw_request[1][64:1]),
    .s_axi_wlast(hdata[1][0]),
    .m_axi_wlast(raw_request[1][0]),
    .s_axi_bvalid(hvalid[2]),
    .s_axi_bready(hr[2]),
    .m_axi_bvalid(out_valid[2] && !response_pause[2]),
    .m_axi_bready(cdc_response_ready[2]),
    .s_axi_bid(hreturn[2][5:2]),
    .m_axi_bid(raw_response[2][5:2]),
    .s_axi_bresp(hreturn[2][1:0]),
    .m_axi_bresp(raw_response[2][1:0]),
    .s_axi_awcache(4'd0),
    .m_axi_awcache(),
    .s_axi_awprot(3'd0),
    .m_axi_awprot(),
    .s_axi_awqos(4'd0),
    .m_axi_awqos(),
    .s_axi_awregion(4'd0),
    .m_axi_awregion(),
    .s_axi_araddr(64'd0),
    .m_axi_araddr(),
    .s_axi_arid(4'd0),
    .m_axi_arid(),
    .s_axi_arlen(8'd0),
    .m_axi_arlen(),
    .s_axi_arsize(3'd6),
    .m_axi_arsize(),
    .s_axi_arburst(2'd1),
    .m_axi_arburst(),
    .s_axi_arlock(1'd0),
    .m_axi_arlock(),
    .s_axi_arcache(4'd0),
    .m_axi_arcache(),
    .s_axi_arprot(3'd0),
    .m_axi_arprot(),
    .s_axi_arqos(4'd0),
    .m_axi_arqos(),
    .s_axi_arregion(4'd0),
    .m_axi_arregion(),
    .s_axi_arvalid(1'd0),
    .m_axi_arvalid(),
    .s_axi_rready(1'd1),
    .m_axi_rready(),
    .s_axi_arready(),
    .m_axi_arready(1'd0),
    .s_axi_rdata(),
    .m_axi_rdata(512'd0),
    .s_axi_rid(),
    .m_axi_rid(4'd0),
    .s_axi_rresp(),
    .m_axi_rresp(2'd0),
    .s_axi_rlast(),
    .m_axi_rlast(1'd0),
    .s_axi_rvalid(),
    .m_axi_rvalid(1'd0)
);
ntrpu_host_axil_cdc control_cdc (
    .s_axi_aclk(host_clk),
    .s_axi_aresetn(host_rst_n),
    .m_axi_aclk(clk),
    .m_axi_aresetn(boundary_rst_n),
    .s_axi_awvalid(hv[3]),
    .s_axi_awready(hready[3]),
    .m_axi_awvalid(in_valid[3]),
    .m_axi_awready(in_ready[3]),
    .s_axi_awaddr(hdata[3][31:0]),
    .m_axi_awaddr(raw_request[3][31:0]),
    .s_axi_wvalid(hv[4]),
    .s_axi_wready(hready[4]),
    .m_axi_wvalid(in_valid[4]),
    .m_axi_wready(in_ready[4]),
    .s_axi_wdata(hdata[4][35:4]),
    .m_axi_wdata(raw_request[4][35:4]),
    .s_axi_wstrb(hdata[4][3:0]),
    .m_axi_wstrb(raw_request[4][3:0]),
    .s_axi_bvalid(hvalid[5]),
    .s_axi_bready(hr[5]),
    .m_axi_bvalid(out_valid[5] && !response_pause[5]),
    .m_axi_bready(cdc_response_ready[5]),
    .s_axi_bresp(hreturn[5][1:0]),
    .m_axi_bresp(raw_response[5][1:0]),
    .s_axi_arvalid(hv[6]),
    .s_axi_arready(hready[6]),
    .m_axi_arvalid(in_valid[6]),
    .m_axi_arready(in_ready[6]),
    .s_axi_araddr(hdata[6][31:0]),
    .m_axi_araddr(raw_request[6][31:0]),
    .s_axi_rvalid(hvalid[7]),
    .s_axi_rready(hr[7]),
    .m_axi_rvalid(out_valid[7] && !response_pause[7]),
    .m_axi_rready(cdc_response_ready[7]),
    .s_axi_rdata(hreturn[7][33:2]),
    .m_axi_rdata(raw_response[7][33:2]),
    .s_axi_rresp(hreturn[7][1:0]),
    .m_axi_rresp(raw_response[7][1:0]),
    .s_axi_awprot(3'd0),
    .m_axi_awprot(),
    .s_axi_arprot(3'd0),
    .m_axi_arprot()
);
