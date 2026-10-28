// Simulation-only cold PCIe/XDMA, supervised FFN, generated MIG and DDR model.
`timescale 1ns/1ps
`ifdef XILINX_SIMULATOR
module short(in1,in1); inout in1; endmodule
`endif
module ntrpu_pcie_ddr_ffn_app (
    input logic  host_clk,
    input logic  pcie_resetn,
    input logic  pcie_link_up,
    input logic  usr_irq_ack,
    input logic [31:0] s_axil_awaddr,
    input logic [31:0] s_axil_wdata,
    input logic [31:0] s_axil_araddr,
    input logic [2:0] s_axil_awprot,
    input logic [2:0] s_axil_arprot,
    input logic [3:0] s_axil_wstrb,
    input logic  s_axil_awvalid,
    input logic  s_axil_wvalid,
    input logic  s_axil_bready,
    input logic  s_axil_arvalid,
    input logic  s_axil_rready,
    output logic  s_axil_awready,
    output logic  s_axil_wready,
    output logic  s_axil_bvalid,
    output logic  s_axil_arready,
    output logic  s_axil_rvalid,
    output logic [1:0] s_axil_bresp,
    output logic [1:0] s_axil_rresp,
    output logic [31:0] s_axil_rdata,
    output logic  irq,
    input logic [63:0] s_host_axi_araddr,
    input logic [63:0] s_host_axi_awaddr,
    input logic [7:0] s_host_axi_arlen,
    input logic [7:0] s_host_axi_awlen,
    input logic [2:0] s_host_axi_arsize,
    input logic [2:0] s_host_axi_arprot,
    input logic [2:0] s_host_axi_awsize,
    input logic [2:0] s_host_axi_awprot,
    input logic [1:0] s_host_axi_arburst,
    input logic [1:0] s_host_axi_awburst,
    input logic [3:0] s_host_axi_arid,
    input logic [3:0] s_host_axi_arcache,
    input logic [3:0] s_host_axi_awid,
    input logic [3:0] s_host_axi_awcache,
    input logic  s_host_axi_arlock,
    input logic  s_host_axi_arvalid,
    input logic  s_host_axi_awlock,
    input logic  s_host_axi_awvalid,
    output logic  s_host_axi_arready,
    output logic  s_host_axi_awready,
    output logic [511:0] s_host_axi_rdata,
    output logic [1:0] s_host_axi_rresp,
    output logic [3:0] s_host_axi_rid,
    output logic  s_host_axi_rlast,
    output logic  s_host_axi_rvalid,
    input logic  s_host_axi_rready,
    input logic [511:0] s_host_axi_wdata,
    input logic [63:0] s_host_axi_wstrb,
    input logic  s_host_axi_wlast,
    input logic  s_host_axi_wvalid,
    output logic  s_host_axi_wready,
    output logic [3:0] s_host_axi_bid,
    output logic [1:0] s_host_axi_bresp,
    output logic  s_host_axi_bvalid,
    input logic  s_host_axi_bready
);
    import arch_package::*;
    wire clk,rst_n,host_rst_n;
    logic mgmt_clk=0;
    always #5 mgmt_clk=~mgmt_clk;
    wire external_reset_async=!pcie_resetn;
    wire rearm=1'b1;
    wire peers_reset_req,rearm_ready,running,host_admit,device_admit;
    wire peers_held,peers_ready,host_alive,device_alive,capacity_guard_idle;
    logic host_peer_held=0,host_peer_ready=0,memory_peer_held=0,memory_peer_ready=0;
    bit pcie_reset_seen=0,host_ever_active=0;
    integer host_quiet=0,host_up=0,quiet_edges=0,up_edges=0;
    longint dt=0;
    always @(posedge clk) dt++;
    // Procedural assertion task: XSIM debug-off can reorder a void function
    // relative to monitor assignments. Keep clock/data checks in sequence.
    task automatic need(input bit ok,input string why);
        if(!ok) $fatal(1,"PCIE_DDR_FFN_CHECK %s",why);
    endtask
    logic  activation_selected;
    logic [36:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize;
    logic [2:0] m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid;
    logic [3:0] m_axi_arcache;
    logic  m_axi_arlock;
    logic  m_axi_arvalid;
    logic  m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic  m_axi_rlast;
    logic  m_axi_rvalid;
    logic  m_axi_rready;
    logic [36:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize;
    logic [2:0] m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid;
    logic [3:0] m_axi_awcache;
    logic  m_axi_awlock;
    logic  m_axi_awvalid;
    logic  m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic  m_axi_wlast;
    logic  m_axi_wvalid;
    logic  m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic  m_axi_bvalid;
    logic  m_axi_bready;
    logic sys_clk=0,sys_rst=1;
    always #2 sys_clk=~sys_clk;
    initial begin #20;sys_rst=0;end
    localparam longint unsigned CAPACITY=64'd137438953472;
    longint unsigned rank_acts[4]='{default:0};
    bit dma_error_mode=0;
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
                        $display("NTRPU_DDR_FFN_ACT ps=%0d rank=%0d count=%0d",longint'($realtime*1000),slot*2+rank,rank_acts[slot*2+rank]);
                    end
        end
    end


    always @(posedge clk) begin
        if(calibrated) begin
            // Round to the declared 1ps precision before comparing periods.
            // Subtracting real-valued nanoseconds loses exact equality at
            // larger timestamps even for a stable simulator clock.
            if(calibration_seen)
                need(longint'($realtime*1000)-last_ui_ps==5000,
                     $sformatf("MIG UI period ps=%0d expected5000",longint'($realtime*1000)-last_ui_ps));
            last_ui_ps=longint'($realtime*1000);calibration_seen=1;ui_samples++;
            if(ui_samples<=2) $display("NTRPU_DDR_FFN_UI ps=%0d sample=%0d dt=%0d",last_ui_ps,ui_samples,dt);
        end
        if(m_axi_arvalid || m_axi_awvalid || m_axi_wvalid)
            need(calibrated && !ui_rst && rst_n && device_admit,"traffic before MIG calibration/admission");
        if(rst_n) need(!interrupt_ecc,"unexpected ECC interrupt");
        if((m_axi_arvalid&&m_axi_arready)||(m_axi_awvalid&&m_axi_awready)) any_axi=1;
        if(any_axi) need(rst_n && !ui_rst && calibrated,"cold bank cannot qualify a runtime reset");
    end
    always @(calibrated or ui_rst)
        $strobe("NTRPU_DDR_FFN_STATE ps=%0d calib=%b ui_rst=%b rst=%b held=%b ready=%b admit=%b running=%b",
                longint'($realtime*1000),calibrated,ui_rst,rst_n,memory_peer_held,memory_peer_ready,device_admit,running);

    byte unsigned memory [0:7][0:5119];
    bit written [0:7][0:5119];
    int cycle=0,read_left=0,write_left=0,b_wait=0;
    int input_bytes=0,core_bytes=0,result_read_bytes=0;
    bit reading=0,writing=0,b_pending=0,clear_r=0,clear_b=0;
    logic [63:0] read_address,write_address;
    logic [3:0] read_id,write_id;
    function automatic int locate(input logic [63:0] address);
        for(int i=0;i<8;i++)
            if(address>=64'h100000000+64'(i)*64'h20000000+4032 &&
               address<64'h100000000+64'(i)*64'h20000000+5120) return i;
        $fatal(1,"PCIE_FFN_RAM_UNMAPPED addr=%h",address);return -1;
    endfunction
    function automatic int offset(input logic [63:0] address,input int b);
        return int'(address-(64'h100000000+64'(b)*64'h20000000));
    endfunction
    function automatic int q(input int row);
        if(row%32==0) return 127;
        if(row%32==1) return 1;
        return (row%32)*7-113;
    endfunction
    function automatic byte unsigned expected(input int b,off);
        logic [63:0] bits;
        int r,k;
        k=off%64;r=off/(b==7 ? 4:8);
        if(b==4) bits=$realtobits(257.0);
        else if(b==5) bits=$realtobits(real'(q(r)));
        else if(b==6) begin
            bits=$realtobits(257.0);
            if(k<32) return 8'(q(k));
            if(k<40) return bits[(k-32)*8+:8];
            return 0;
        end else bits={32'b0,$shortrealtobits(shortreal'(257*q((r*7)%32)))};
        return bits[(off%(b==7 ? 4:8))*8+:8];
    endfunction

    // Scoreboard only: every DUT read/response comes from the generated MIG.
    // W may precede AW. Associate accepted beats after both are observed;
    // DDR_W logs keep the original handshake time, RAM_W is the association.
    logic [511:0] w_queue[0:255];
    logic [63:0] w_masks[0:255];
    bit w_lasts[0:255];
    int w_received=0,w_stored=0,write_beats=0;
    longint mig_aw=0,mig_w=0,mig_b=0,mig_ar=0,mig_r=0;
    initial for(int b=0;b<8;b++) for(int i=0;i<5120;i++) begin
        memory[b][i]='ha5;written[b][i]=0;
    end
    ntrpu_ffn_guarded_fabric #(.OUTPUT_FP32(1)) device(.*);

    // Cold-start adapter only. A real XDMA reset must have been observed,
    // no AXI command may have existed, and both interfaces must be quiet.
    // Once traffic starts this bank never claims that XDMA is reset/drained.
    always @(posedge host_clk) begin
        if(pcie_resetn===0) pcie_reset_seen<=1;
        if(s_host_axi_awvalid || s_host_axi_wvalid || s_host_axi_arvalid ||
           s_axil_awvalid || s_axil_wvalid || s_axil_arvalid) host_ever_active<=1;
        if(!host_rst_n && pcie_reset_seen && !host_ever_active &&
           {s_host_axi_awvalid,s_host_axi_wvalid,s_host_axi_arvalid,s_host_axi_bvalid,s_host_axi_rvalid,
            s_axil_awvalid,s_axil_wvalid,s_axil_arvalid,s_axil_bvalid,s_axil_rvalid}===10'b0) begin
            if(host_quiet==7) host_peer_held<=1;else host_quiet<=host_quiet+1;
        end else begin host_quiet<=0;host_peer_held<=0;end
        if(host_rst_n && pcie_resetn && pcie_link_up) begin
            if(host_up==7) host_peer_ready<=1;else host_up<=host_up+1;
        end else begin host_up<=0;host_peer_ready<=0;end
        if(host_ever_active) need(pcie_resetn && host_rst_n && host_admit,"cold bank lost host admission after traffic");
        if(s_host_axi_awvalid || s_host_axi_wvalid || s_host_axi_arvalid ||
           s_axil_awvalid || s_axil_wvalid || s_axil_arvalid)
            need(host_admit && device_admit && running,"host request before admission");
    end
    always @(posedge clk) begin
        if(!rst_n && !ui_rst && !any_axi && !reading && !writing && w_received==0 &&
           !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && m_axi_rvalid===0 && m_axi_bvalid===0) begin
            if(quiet_edges==7) memory_peer_held<=1;else quiet_edges<=quiet_edges+1;
        end else begin quiet_edges<=0;memory_peer_held<=0;end
        if(rst_n && !ui_rst && calibrated) begin
            if(up_edges==7) memory_peer_ready<=1;else up_edges<=up_edges+1;
        end else begin up_edges<=0;memory_peer_ready<=0;end
    end
    always @(rst_n or host_rst_n or host_admit or device_admit or running or pcie_resetn or pcie_link_up or peers_held or peers_ready)
        $strobe("NTRPU_PCIE_DDR_STATE ps=%0d pcie_resetn=%b link=%b rst=%b hrst=%b da=%b ha=%b run=%b ph=%b pr=%b calib=%b ui_rst=%b",
            longint'($realtime*1000),pcie_resetn,pcie_link_up,rst_n,host_rst_n,device_admit,host_admit,running,peers_held,peers_ready,calibrated,ui_rst);
    always @(posedge clk) if(rst_n && !dma_error_mode) begin
        cycle++;
        if(m_axi_rvalid && m_axi_rready) begin
            need(reading && read_left>0,"MIG R without AR");
            need(m_axi_rresp===0 && m_axi_rid===read_id && m_axi_rlast===(read_left==1),"MIG R identity/response/LAST");
            for(int lane=0;lane<64;lane++) begin
                need(written[locate(read_address)][offset(read_address,locate(read_address))+lane],"MIG read uninitialized");
                need(m_axi_rdata[lane*8+:8]===memory[locate(read_address)][offset(read_address,locate(read_address))+lane],"actual MIG data differs from accepted writes");
            end
            $display("NTRPU_PCIE_FFN_RAM_R addr=%016h id=%0d data=%0128h time_ps=%0d",read_address,m_axi_rid,m_axi_rdata,longint'($realtime*1000));
            $display("NTRPU_PCIE_DDR_R ps=%0d id=%0d resp=%0d last=%0d data=%0128h",longint'($realtime*1000),m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata);
            if(locate(read_address)==7) result_read_bytes+=64;
            read_address+=64;read_left--;if(read_left==0) reading=0;mig_r++;
        end
        if(m_axi_arvalid && m_axi_arready) begin
            need(!reading && m_axi_arsize==6 && m_axi_arburst==1 && !m_axi_arlock && m_axi_araddr[5:0]==0 &&
                int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"MIG AR contract");
            reading=1;read_left=int'(m_axi_arlen)+1;read_address=m_axi_araddr;read_id=m_axi_arid;mig_ar++;
            $display("NTRPU_PCIE_FFN_RAM_AR addr=%016h beats=%0d id=%0d time_ps=%0d",read_address,read_left,read_id,longint'($realtime*1000));
        end
        if(m_axi_awvalid && m_axi_awready) begin
            need(!writing && !b_pending && m_axi_awsize==6 && m_axi_awburst==1 && !m_axi_awlock && m_axi_awaddr[5:0]==0 &&
                int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64<=4096,"MIG AW contract");
            writing=1;write_beats=int'(m_axi_awlen)+1;write_address=m_axi_awaddr;write_id=m_axi_awid;mig_aw++;
            $display("NTRPU_PCIE_FFN_RAM_AW addr=%016h beats=%0d id=%0d time_ps=%0d",write_address,write_beats,write_id,longint'($realtime*1000));
        end
        if(m_axi_wvalid && m_axi_wready) begin
            need(w_received<256 && !$isunknown(m_axi_wdata),"MIG W queue/payload");
            w_queue[w_received]=m_axi_wdata;w_masks[w_received]=m_axi_wstrb;w_lasts[w_received]=m_axi_wlast;w_received++;mig_w++;
            $display("NTRPU_PCIE_DDR_W ps=%0d data=%0128h strb=%016h last=%0d",longint'($realtime*1000),m_axi_wdata,m_axi_wstrb,m_axi_wlast);
        end
        if(writing) while(w_stored<w_received) begin
            need(w_stored<write_beats && w_masks[w_stored]==='1 && w_lasts[w_stored]===(w_stored+1==write_beats),"MIG W strobe/LAST");
            for(int lane=0;lane<64;lane++) begin
                need(!written[locate(write_address)][offset(write_address,locate(write_address))+lane],"duplicate MIG byte write");
                if(locate(write_address)>=4)
                    need(w_queue[w_stored][lane*8+:8]===expected(locate(write_address),offset(write_address,locate(write_address))+lane-4032),"MIG FFN numerical result");
                memory[locate(write_address)][offset(write_address,locate(write_address))+lane]=w_queue[w_stored][lane*8+:8];
                written[locate(write_address)][offset(write_address,locate(write_address))+lane]=1;
            end
            if(locate(write_address)<4) input_bytes+=64;else core_bytes+=64;
            $display("NTRPU_PCIE_FFN_RAM_W addr=%016h id=%0d data=%0128h time_ps=%0d",write_address,write_id,w_queue[w_stored],longint'($realtime*1000));
            write_address+=64;w_stored++;if(w_stored==write_beats) b_pending=1;
        end
        if(m_axi_bvalid && m_axi_bready) begin
            need(writing && b_pending && w_received==write_beats && w_stored==w_received,"MIG B after complete write");
            need(m_axi_bresp===0 && m_axi_bid===write_id,"MIG B identity/response");
            $display("NTRPU_PCIE_DDR_B ps=%0d id=%0d resp=%0d",longint'($realtime*1000),m_axi_bid,m_axi_bresp);
            writing=0;b_pending=0;w_received=0;w_stored=0;mig_b++;
        end
            if(device.compute.fabric.fabric.device.job_valid && device.compute.fabric.fabric.device.job_ready)
                $display("NTRPU_PCIE_FFN_JOB sequence=%08h epoch=%08h time_ps=%0d",device.compute.fabric.fabric.device.job_sequence,device.compute.fabric.fabric.device.job_epoch,longint'($realtime*1000));
            if(device.compute.fabric.fabric.device.done_valid && device.compute.fabric.fabric.device.done_ready)
                $display("NTRPU_PCIE_FFN_DONE sequence=%08h epoch=%08h status=%0d stages=%0d bytes=%0d stage_bytes=%0d time_ps=%0d",device.compute.fabric.fabric.device.done_sequence,device.compute.fabric.fabric.device.done_epoch,device.compute.fabric.fabric.device.done_status,device.compute.fabric.fabric.device.done_stages,device.compute.fabric.fabric.device.done_bytes,device.compute.fabric.fabric.device.done_stage_bytes,longint'($realtime*1000));
            if(device.compute.fabric.fabric.device.with_irq.control.notify_retire_valid && device.compute.fabric.fabric.device.with_irq.control.notify_retire_ready)
                $display("NTRPU_PCIE_FFN_RETIRE sequence=%08h epoch=%08h time_ps=%0d",device.compute.fabric.fabric.device.with_irq.control.notify_retire_sequence,device.compute.fabric.fabric.device.with_irq.control.notify_retire_epoch,longint'($realtime*1000));
            if(device.compute.fabric.fabric.device.with_irq.control.notify_service_valid && device.compute.fabric.fabric.device.with_irq.control.notify_service_ready)
                $display("NTRPU_PCIE_FFN_SERVICE_ACCEPT sequence=%08h epoch=%08h time_ps=%0d",device.compute.fabric.fabric.device.with_irq.control.notify_sequence,device.compute.fabric.fabric.device.with_irq.control.notify_epoch,longint'($realtime*1000));

    end
    task verify_ddr_complete;
        verify_periodic_complete();
        need(calibrated && !ui_rst && running && host_admit && device_admit && capacity_guard_idle,"DDR final admission/drain");
        need(!reading && !writing && !b_pending && w_received==0 && mig_aw==mig_b,"DDR all responses complete");
        need(mig_w==63 && mig_r==64 && input_bytes==3328 && core_bytes==704 && result_read_bytes==128,"DDR numerical byte coverage");
        $display("NTRPU_PCIE_DDR_PASS aw=%0d w=%0d b=%0d ar=%0d r=%0d rank0=%0d rank1=%0d rank2=%0d rank3=%0d",mig_aw,mig_w,mig_b,mig_ar,mig_r,rank_acts[0],rank_acts[1],rank_acts[2],rank_acts[3]);
    endtask
    logic [31:0] held_aw,held_w,held_ar;
    int acknowledgements=0,retires=0;
    bit last_irq=0;
    always @(posedge host_clk) if(host_rst_n) begin
        if(s_axil_awvalid && s_axil_awready) held_aw=s_axil_awaddr;
        if(s_axil_wvalid && s_axil_wready) held_w=s_axil_wdata;
        if(s_axil_arvalid && s_axil_arready) held_ar=s_axil_araddr;
        if(s_axil_bvalid && s_axil_bready)
            $display("NTRPU_PCIE_FFN_MMIO_W addr=%08h data=%08h resp=%0d time_ps=%0d",held_aw,held_w,s_axil_bresp,longint'($realtime*1000));
        if(s_axil_rvalid && s_axil_rready)
            $display("NTRPU_PCIE_FFN_MMIO_R addr=%08h data=%08h resp=%0d time_ps=%0d",held_ar,s_axil_rdata,s_axil_rresp,longint'($realtime*1000));
        if(irq!=last_irq) $display("NTRPU_PCIE_FFN_IRQ level=%0d time_ps=%0d",irq,longint'($realtime*1000));
        if(irq && usr_irq_ack) begin
            acknowledgements++;
            $display("NTRPU_PCIE_FFN_ACK time_ps=%0d",longint'($realtime*1000));
        end
        last_irq=irq;
    end

    // Read-only diagnostics of generated, unmodified MIG RTL. No force,
    // memory preload, periodic-read disable or model-warning suppression.
`define NTRPU_PER_MC mig.inst.u_ddr4_mem_intfc.u_ddr_mc
`define NTRPU_PER_CAL mig.inst.u_ddr4_mem_intfc.u_ddr_cal_top
    longint per_requests=0,per_accepts=0,per_returns=0,per_cas=0,user_cas=0;
    bit periodic_config_logged=0;
    always @(posedge clk) if(calibrated) begin
        if(!periodic_config_logged) begin
            periodic_config_logged=1;
            $display("NTRPU_DDR_PERIODIC_CONFIG ps=%0d interval=%0d config=%08h abits=%0d colbits=%0d",
                longint'($realtime*1000),`NTRPU_PER_MC.periodic_interval_config,
                `NTRPU_PER_MC.periodic_config,`NTRPU_PER_MC.ABITS,`NTRPU_PER_MC.COLBITS);
        end
        if(`NTRPU_PER_MC.per_rd_req) per_requests++;
        if(`NTRPU_PER_MC.per_rd_accept[0]) per_accepts++;
        if(`NTRPU_PER_MC.per_rd_done) per_returns++;
        if(`NTRPU_PER_MC.per_rd_req || |`NTRPU_PER_MC.per_rd_accept || `NTRPU_PER_MC.per_rd_done)
            $display("NTRPU_DDR_PERIODIC ps=%0d req=%b accept=%h done=%b rst=%b axi_rst=%b aw=%b ar=%b rvalid=%b rready=%b",
                longint'($realtime*1000),`NTRPU_PER_MC.per_rd_req,`NTRPU_PER_MC.per_rd_accept,
                `NTRPU_PER_MC.per_rd_done,`NTRPU_PER_MC.rst_r1,rst_n,
                m_axi_awvalid&&m_axi_awready,m_axi_arvalid&&m_axi_arready,m_axi_rvalid,m_axi_rready);
        if(`NTRPU_PER_MC.rdCAS) begin
            if(`NTRPU_PER_MC.winInjTxn) per_cas++;else user_cas++;
            $display("NTRPU_DDR_READ_CAS ps=%0d injected=%b rmw=%b rank=%h bg=%h bank=%h col=%h rows=%h",
                longint'($realtime*1000),`NTRPU_PER_MC.winInjTxn,`NTRPU_PER_MC.winRmw,
                `NTRPU_PER_MC.winRankC,`NTRPU_PER_MC.winGroupC,`NTRPU_PER_MC.winBankC,
                `NTRPU_PER_MC.winCol,`NTRPU_PER_MC.cmd_row_cas);
        end
        if(|`NTRPU_PER_CAL.rdDataEn_pi) begin
            $display("NTRPU_DDR_READ_RETURN ps=%0d phy=%h injected=%h rmw=%h native=%h periodic_done=%h",
                longint'($realtime*1000),`NTRPU_PER_CAL.rdDataEn_pi,`NTRPU_PER_CAL.rdInj,
                `NTRPU_PER_CAL.rdRmw,`NTRPU_PER_CAL.rdDataEn,`NTRPU_PER_CAL.per_rd_done);
            need((`NTRPU_PER_CAL.rdDataEn & (`NTRPU_PER_CAL.rdInj | `NTRPU_PER_CAL.rdRmw))==0,
                "periodic/RMW read leaked to native data-valid");
        end
    end
    for(genvar watch_slot=0;watch_slot<2;watch_slot++) begin: periodic_pin_observer
        always @(posedge ck_t[watch_slot]) if(calibrated)
            for(int rank=0;rank<2;rank++)
                if(cs_n[watch_slot*2+rank]===0 && (!act_n || adr[17:15]!=3'b111))
                    $display("NTRPU_DDR_PIN_CMD ps=%0d slot=%0d rank=%0d act_n=%b cmd=%h adr=%h bg=%h bank=%h cke=%b",
                        longint'($realtime*1000),watch_slot,watch_slot*2+rank,act_n,adr[17:15],adr,bg,ba,cke[watch_slot*2+rank]);
    end
    task verify_periodic_complete;
        int waited;
        longint begin_ps;
        waited=0;begin_ps=longint'($realtime*1000);
        // Observe a falling UI edge to avoid races with rising-edge counters.
        // Do not disable, reset, force or otherwise influence periodic traffic.
        do begin
            @(negedge clk);waited++;
            need(waited<200,"periodic read drain timeout");
        end while(per_requests!=per_returns || per_accepts!=per_returns || per_cas!=per_returns);
        $display("NTRPU_DDR_PERIODIC_DRAIN begin_ps=%0d end_ps=%0d falling_edges=%0d requests=%0d returns=%0d",
                 begin_ps,longint'($realtime*1000),waited,per_requests,per_returns);
        need(periodic_config_logged && per_requests>0 && per_accepts>0 && per_returns>0 && per_cas>0 && user_cas>0,
            "periodic and user read witness coverage");
        $display("NTRPU_DDR_PERIODIC_PASS requests=%0d accepts=%0d returns=%0d periodic_cas=%0d user_cas=%0d",
            per_requests,per_accepts,per_returns,per_cas,user_cas);
    endtask
`undef NTRPU_PER_MC
`undef NTRPU_PER_CAL
// Passive observations of XDMA, the guard input and the real MIG interface.
bit dma_error_direction=0,dma_error_bad=0;
int dma_error_case=0;
int dma_counts[3][5]; // HOST, PRE, MIG; AW,W,B,AR,R.
logic [3:0] dma_awid[3],dma_arid[3];
localparam logic [63:0] DMA_VALID_ADDRESS=64'h1e0000fc0;
task dma_error_begin(input int case_id,input bit c2h,bad);
    @(negedge clk);
    need(capacity_guard_idle && !reading && !writing && !b_pending,"DMA error phase starts drained");
    dma_error_mode=1;dma_error_case=case_id;dma_error_direction=c2h;dma_error_bad=bad;
    for(int s=0;s<3;s++) for(int c=0;c<5;c++) dma_counts[s][c]=0;
    $display("NTRPU_DMA_ERROR_AXI_BEGIN case=%0d direction=%0d bad=%0d ps=%0d",case_id,c2h,bad,longint'($realtime*1000));
endtask
task dma_error_verify;
    @(negedge clk);
    need(capacity_guard_idle && running && host_admit && device_admit && calibrated && !ui_rst,"DMA error case leaves peers admitted and guard drained");
    for(int s=0;s<3;s++) begin
        for(int c=0;c<5;c++)
            need(dma_counts[s][c]==((s==2 && dma_error_bad) ? 0 : (dma_error_direction ? (c>=3):(c<3))),"DMA exact accepted beat/address/response counts");
        $display("NTRPU_DMA_ERROR_AXI_PASS case=%0d scope=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d ps=%0d",
                 dma_error_case,s,dma_counts[s][0],dma_counts[s][1],dma_counts[s][2],dma_counts[s][3],dma_counts[s][4],longint'($realtime*1000));
    end
endtask
    always @(posedge host_clk) if(host_rst_n && dma_error_mode) begin
        if(s_host_axi_awvalid && s_host_axi_awready) begin
            need(dma_counts[0][0]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[0][0]++;
            need(s_host_axi_awaddr===DMA_VALID_ADDRESS+((dma_error_bad && 0!=2) ? (64'h1<<37):0),"DMA preserves exact full address");
            need(s_host_axi_awlen===0 && s_host_axi_awsize===6 && s_host_axi_awburst===1 && !$isunknown(s_host_axi_awid),"DMA one aligned 64-byte INCR beat");
            dma_awid[0]=s_host_axi_awid;
            $display("NTRPU_DMA_ERROR_AXI_AW case=%0d scope=0 addr=%016h len=%0d size=%0d burst=%0d id=%0d ps=%0d",dma_error_case,s_host_axi_awaddr,s_host_axi_awlen,s_host_axi_awsize,s_host_axi_awburst,s_host_axi_awid,longint'($realtime*1000));
        end
        if(s_host_axi_wvalid && s_host_axi_wready) begin
            need(dma_counts[0][1]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[0][1]++;
            need(s_host_axi_wstrb==='1 && s_host_axi_wlast===1,"DMA complete W beat");
            for(int lane=0;lane<64;lane++) need(s_host_axi_wdata[lane*8+:8]===(dma_error_bad ? (8'hc3^8'(lane)):expected(7,lane)),"DMA exact W payload");
            $display("NTRPU_DMA_ERROR_AXI_W case=%0d scope=0 data=%0128h strb=%016h last=%0d ps=%0d",dma_error_case,s_host_axi_wdata,s_host_axi_wstrb,s_host_axi_wlast,longint'($realtime*1000));
        end
        if(s_host_axi_bvalid && s_host_axi_bready) begin
            need(dma_counts[0][2]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[0][2]++;
            need(dma_counts[0][0]==1 && dma_counts[0][1]==1 && s_host_axi_bid===dma_awid[0],"DMA B follows accepted AW/W with same ID");
            need(s_host_axi_bresp===(dma_error_bad ? 2'b11:2'b00),"DMA actual B DECERR/OKAY");
            $display("NTRPU_DMA_ERROR_AXI_B case=%0d scope=0 id=%0d resp=%0d ps=%0d",dma_error_case,s_host_axi_bid,s_host_axi_bresp,longint'($realtime*1000));
        end
        if(s_host_axi_arvalid && s_host_axi_arready) begin
            need(dma_counts[0][3]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[0][3]++;
            need(s_host_axi_araddr===DMA_VALID_ADDRESS+((dma_error_bad && 0!=2) ? (64'h1<<37):0),"DMA preserves exact full address");
            need(s_host_axi_arlen===0 && s_host_axi_arsize===6 && s_host_axi_arburst===1 && !$isunknown(s_host_axi_arid),"DMA one aligned 64-byte INCR beat");
            dma_arid[0]=s_host_axi_arid;
            $display("NTRPU_DMA_ERROR_AXI_AR case=%0d scope=0 addr=%016h len=%0d size=%0d burst=%0d id=%0d ps=%0d",dma_error_case,s_host_axi_araddr,s_host_axi_arlen,s_host_axi_arsize,s_host_axi_arburst,s_host_axi_arid,longint'($realtime*1000));
        end
        if(s_host_axi_rvalid && s_host_axi_rready) begin
            need(dma_counts[0][4]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[0][4]++;
            need(dma_counts[0][3]==1 && s_host_axi_rid===dma_arid[0] && s_host_axi_rlast===1,"DMA R after AR with matching ID/LAST");
            need(s_host_axi_rresp===(dma_error_bad ? 2'b11:2'b00),"DMA actual R DECERR/OKAY");
            for(int lane=0;lane<64;lane++) need(s_host_axi_rdata[lane*8+:8]===(dma_error_bad ? 8'b0:expected(7,lane)),"DMA zero rejected read or exact sentinel");
            $display("NTRPU_DMA_ERROR_AXI_R case=%0d scope=0 id=%0d resp=%0d last=%0d data=%0128h ps=%0d",dma_error_case,s_host_axi_rid,s_host_axi_rresp,s_host_axi_rlast,s_host_axi_rdata,longint'($realtime*1000));
        end
    end
    always @(posedge clk) if(rst_n && dma_error_mode) begin
        if(device.pre_axi_awvalid && device.pre_axi_awready) begin
            need(dma_counts[1][0]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[1][0]++;
            need(device.pre_axi_awaddr===DMA_VALID_ADDRESS+((dma_error_bad && 1!=2) ? (64'h1<<37):0),"DMA preserves exact full address");
            need(device.pre_axi_awlen===0 && device.pre_axi_awsize===6 && device.pre_axi_awburst===1 && !$isunknown(device.pre_axi_awid),"DMA one aligned 64-byte INCR beat");
            dma_awid[1]=device.pre_axi_awid;
            $display("NTRPU_DMA_ERROR_AXI_AW case=%0d scope=1 addr=%016h len=%0d size=%0d burst=%0d id=%0d ps=%0d",dma_error_case,device.pre_axi_awaddr,device.pre_axi_awlen,device.pre_axi_awsize,device.pre_axi_awburst,device.pre_axi_awid,longint'($realtime*1000));
        end
        if(device.pre_axi_wvalid && device.pre_axi_wready) begin
            need(dma_counts[1][1]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[1][1]++;
            need(device.pre_axi_wstrb==='1 && device.pre_axi_wlast===1,"DMA complete W beat");
            for(int lane=0;lane<64;lane++) need(device.pre_axi_wdata[lane*8+:8]===(dma_error_bad ? (8'hc3^8'(lane)):expected(7,lane)),"DMA exact W payload");
            $display("NTRPU_DMA_ERROR_AXI_W case=%0d scope=1 data=%0128h strb=%016h last=%0d ps=%0d",dma_error_case,device.pre_axi_wdata,device.pre_axi_wstrb,device.pre_axi_wlast,longint'($realtime*1000));
        end
        if(device.pre_axi_bvalid && device.pre_axi_bready) begin
            need(dma_counts[1][2]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[1][2]++;
            need(dma_counts[1][0]==1 && dma_counts[1][1]==1 && device.pre_axi_bid===dma_awid[1],"DMA B follows accepted AW/W with same ID");
            need(device.pre_axi_bresp===(dma_error_bad ? 2'b11:2'b00),"DMA actual B DECERR/OKAY");
            $display("NTRPU_DMA_ERROR_AXI_B case=%0d scope=1 id=%0d resp=%0d ps=%0d",dma_error_case,device.pre_axi_bid,device.pre_axi_bresp,longint'($realtime*1000));
        end
        if(device.pre_axi_arvalid && device.pre_axi_arready) begin
            need(dma_counts[1][3]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[1][3]++;
            need(device.pre_axi_araddr===DMA_VALID_ADDRESS+((dma_error_bad && 1!=2) ? (64'h1<<37):0),"DMA preserves exact full address");
            need(device.pre_axi_arlen===0 && device.pre_axi_arsize===6 && device.pre_axi_arburst===1 && !$isunknown(device.pre_axi_arid),"DMA one aligned 64-byte INCR beat");
            dma_arid[1]=device.pre_axi_arid;
            $display("NTRPU_DMA_ERROR_AXI_AR case=%0d scope=1 addr=%016h len=%0d size=%0d burst=%0d id=%0d ps=%0d",dma_error_case,device.pre_axi_araddr,device.pre_axi_arlen,device.pre_axi_arsize,device.pre_axi_arburst,device.pre_axi_arid,longint'($realtime*1000));
        end
        if(device.pre_axi_rvalid && device.pre_axi_rready) begin
            need(dma_counts[1][4]==0 && 1'b1,"DMA no duplicate or invalid MIG transfer");
            dma_counts[1][4]++;
            need(dma_counts[1][3]==1 && device.pre_axi_rid===dma_arid[1] && device.pre_axi_rlast===1,"DMA R after AR with matching ID/LAST");
            need(device.pre_axi_rresp===(dma_error_bad ? 2'b11:2'b00),"DMA actual R DECERR/OKAY");
            for(int lane=0;lane<64;lane++) need(device.pre_axi_rdata[lane*8+:8]===(dma_error_bad ? 8'b0:expected(7,lane)),"DMA zero rejected read or exact sentinel");
            $display("NTRPU_DMA_ERROR_AXI_R case=%0d scope=1 id=%0d resp=%0d last=%0d data=%0128h ps=%0d",dma_error_case,device.pre_axi_rid,device.pre_axi_rresp,device.pre_axi_rlast,device.pre_axi_rdata,longint'($realtime*1000));
        end
    end
    always @(posedge clk) if(rst_n && dma_error_mode) begin
        if(m_axi_awvalid && m_axi_awready) begin
            need(dma_counts[2][0]==0 && !dma_error_bad,"DMA no duplicate or invalid MIG transfer");
            dma_counts[2][0]++;
            need(m_axi_awaddr===DMA_VALID_ADDRESS+((dma_error_bad && 2!=2) ? (64'h1<<37):0),"DMA preserves exact full address");
            need(m_axi_awlen===0 && m_axi_awsize===6 && m_axi_awburst===1 && !$isunknown(m_axi_awid),"DMA one aligned 64-byte INCR beat");
            dma_awid[2]=m_axi_awid;
            $display("NTRPU_DMA_ERROR_AXI_AW case=%0d scope=2 addr=%016h len=%0d size=%0d burst=%0d id=%0d ps=%0d",dma_error_case,m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,longint'($realtime*1000));
        end
        if(m_axi_wvalid && m_axi_wready) begin
            need(dma_counts[2][1]==0 && !dma_error_bad,"DMA no duplicate or invalid MIG transfer");
            dma_counts[2][1]++;
            need(m_axi_wstrb==='1 && m_axi_wlast===1,"DMA complete W beat");
            for(int lane=0;lane<64;lane++) need(m_axi_wdata[lane*8+:8]===(dma_error_bad ? (8'hc3^8'(lane)):expected(7,lane)),"DMA exact W payload");
            $display("NTRPU_DMA_ERROR_AXI_W case=%0d scope=2 data=%0128h strb=%016h last=%0d ps=%0d",dma_error_case,m_axi_wdata,m_axi_wstrb,m_axi_wlast,longint'($realtime*1000));
        end
        if(m_axi_bvalid && m_axi_bready) begin
            need(dma_counts[2][2]==0 && !dma_error_bad,"DMA no duplicate or invalid MIG transfer");
            dma_counts[2][2]++;
            need(dma_counts[2][0]==1 && dma_counts[2][1]==1 && m_axi_bid===dma_awid[2],"DMA B follows accepted AW/W with same ID");
            need(m_axi_bresp===(dma_error_bad ? 2'b11:2'b00),"DMA actual B DECERR/OKAY");
            $display("NTRPU_DMA_ERROR_AXI_B case=%0d scope=2 id=%0d resp=%0d ps=%0d",dma_error_case,m_axi_bid,m_axi_bresp,longint'($realtime*1000));
        end
        if(m_axi_arvalid && m_axi_arready) begin
            need(dma_counts[2][3]==0 && !dma_error_bad,"DMA no duplicate or invalid MIG transfer");
            dma_counts[2][3]++;
            need(m_axi_araddr===DMA_VALID_ADDRESS+((dma_error_bad && 2!=2) ? (64'h1<<37):0),"DMA preserves exact full address");
            need(m_axi_arlen===0 && m_axi_arsize===6 && m_axi_arburst===1 && !$isunknown(m_axi_arid),"DMA one aligned 64-byte INCR beat");
            dma_arid[2]=m_axi_arid;
            $display("NTRPU_DMA_ERROR_AXI_AR case=%0d scope=2 addr=%016h len=%0d size=%0d burst=%0d id=%0d ps=%0d",dma_error_case,m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,longint'($realtime*1000));
        end
        if(m_axi_rvalid && m_axi_rready) begin
            need(dma_counts[2][4]==0 && !dma_error_bad,"DMA no duplicate or invalid MIG transfer");
            dma_counts[2][4]++;
            need(dma_counts[2][3]==1 && m_axi_rid===dma_arid[2] && m_axi_rlast===1,"DMA R after AR with matching ID/LAST");
            need(m_axi_rresp===(dma_error_bad ? 2'b11:2'b00),"DMA actual R DECERR/OKAY");
            for(int lane=0;lane<64;lane++) need(m_axi_rdata[lane*8+:8]===(dma_error_bad ? 8'b0:expected(7,lane)),"DMA zero rejected read or exact sentinel");
            $display("NTRPU_DMA_ERROR_AXI_R case=%0d scope=2 id=%0d resp=%0d last=%0d data=%0128h ps=%0d",dma_error_case,m_axi_rid,m_axi_rresp,m_axi_rlast,m_axi_rdata,longint'($realtime*1000));
        end
    end
    task verify_periodic_final;
        int waited;
        longint begin_ps;
        waited=0;begin_ps=longint'($realtime*1000);
        // Observe a falling UI edge to avoid races with rising-edge counters.
        // Do not disable, reset, force or otherwise influence periodic traffic.
        do begin
            @(negedge clk);waited++;
            need(waited<200,"periodic read drain timeout");
        end while(per_requests!=per_returns || per_accepts!=per_returns || per_cas!=per_returns);
        $display("NTRPU_DDR_PERIODIC_FINAL_DRAIN begin_ps=%0d end_ps=%0d falling_edges=%0d requests=%0d returns=%0d",
                 begin_ps,longint'($realtime*1000),waited,per_requests,per_returns);
        need(periodic_config_logged && per_requests>0 && per_accepts>0 && per_returns>0 && per_cas>0 && user_cas>0,
            "periodic and user read witness coverage");
        $display("NTRPU_DDR_PERIODIC_FINAL_PASS requests=%0d accepts=%0d returns=%0d periodic_cas=%0d user_cas=%0d",
            per_requests,per_accepts,per_returns,per_cas,user_cas);
    endtask
endmodule
