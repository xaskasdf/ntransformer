`timescale 1ns/1ps
// Cold-start real MIG + generic two-slot128GiB model + supervised FFN.
// A real synthesized-RTL capacity guard sits between the arbiter and MIG.
// This bank does not qualify warm reset or the exact Micron DIMM.
`ifdef XILINX_SIMULATOR
module short(in1, in1);
    inout in1;
endmodule
`endif
module tb_ntrpu_ffn_guarded_ddr;
    import arch_package::*;
    logic host_osc=0,mgmt_clk=0,external_reset_async=1,rearm=1;
    bit device_enabled=1,host_enabled=1,host_clock_ready=0;
    wire clk;
    wire host_clk=host_osc && host_enabled;
    wire rst_n,host_rst_n,device_admit,host_admit,running;
    wire peers_reset_req,rearm_ready,peers_held,peers_ready,host_dma_held,host_dma_ready;
    logic memory_peer_held=0,memory_peer_ready=0;
    integer host_half_ps=2000,quiet_edges=0,up_edges=0,fault=0;
    longint dt=0,ht=0,mt=0;

    always #5 mgmt_clk=~mgmt_clk;
    initial begin wait(host_clock_ready);#0.7;forever #(host_half_ps/1000.0) host_osc=~host_osc;end
    always @(posedge clk) dt++;
    always @(posedge host_clk) ht++;
    always @(posedge mgmt_clk) mt++;
    logic  job_valid;
    logic  job_ready;
    logic [1:0] job_layout;
    logic [13:0] job_dim;
    logic [13:0] job_hidden;
    logic [5:0] job_tile_k [0:2];
    logic [63:0] job_addr [0:7];
    logic [63:0] job_base [0:7];
    logic [63:0] job_bytes [0:7];
    logic [31:0] job_buffer_epoch [0:7];
    logic [31:0] job_sequence;
    logic [31:0] job_epoch;
    logic [31:0] expected_epoch;
    logic  cancel_valid;
    logic  cancel_ready;
    logic  done_valid;
    logic  done_ready;
    logic [3:0] done_status;
    logic [3:0] done_child_status;
    logic [2:0] done_stages;
    logic [1:0] done_phase;
    logic [16:0] done_bytes;
    logic [17:0] done_stage_bytes;
    logic [31:0] done_sequence;
    logic [31:0] done_epoch;
    logic [31:0] done_hls_status;
    logic  activation_selected;
    logic [63:0] m_axi_araddr;
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
    logic [63:0] m_axi_awaddr;
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

    logic[63:0] s_host_axi_araddr, s_host_axi_awaddr;
    logic[7:0] s_host_axi_arlen, s_host_axi_awlen;
    logic[2:0] s_host_axi_arsize, s_host_axi_arprot, s_host_axi_awsize, s_host_axi_awprot;
    logic[1:0] s_host_axi_arburst, s_host_axi_awburst;
    logic[3:0] s_host_axi_arid, s_host_axi_arcache, s_host_axi_awid, s_host_axi_awcache;
    logic s_host_axi_arlock, s_host_axi_arvalid, s_host_axi_awlock, s_host_axi_awvalid;
    logic s_host_axi_arready, s_host_axi_awready;
    logic[511:0] s_host_axi_rdata;
    logic[1:0] s_host_axi_rresp;
    logic[3:0] s_host_axi_rid;
    logic s_host_axi_rlast, s_host_axi_rvalid;
    logic s_host_axi_rready;
    logic[511:0] s_host_axi_wdata;
    logic[63:0] s_host_axi_wstrb;
    logic s_host_axi_wlast, s_host_axi_wvalid;
    logic s_host_axi_wready;
    logic[3:0] s_host_axi_bid;
    logic[1:0] s_host_axi_bresp;
    logic s_host_axi_bvalid;
    logic s_host_axi_bready;

    wire capacity_guard_idle;
    ntrpu_ffn_guarded_host_bfm #(.OUTPUT_FP32(1)) dut(.*);
    // Passive observation of the capacity-guard input ports inside the synthesized top.
    wire [63:0] pre_axi_araddr=dut.device.capacity_guard.s_axi_araddr;
    wire [7:0] pre_axi_arlen=dut.device.capacity_guard.s_axi_arlen;
    wire [2:0] pre_axi_arsize=dut.device.capacity_guard.s_axi_arsize;
    wire [2:0] pre_axi_arprot=dut.device.capacity_guard.s_axi_arprot;
    wire [1:0] pre_axi_arburst=dut.device.capacity_guard.s_axi_arburst;
    wire [3:0] pre_axi_arid=dut.device.capacity_guard.s_axi_arid;
    wire [3:0] pre_axi_arcache=dut.device.capacity_guard.s_axi_arcache;
    wire pre_axi_arlock=dut.device.capacity_guard.s_axi_arlock;
    wire pre_axi_arvalid=dut.device.capacity_guard.s_axi_arvalid;
    wire pre_axi_arready=dut.device.capacity_guard.s_axi_arready;
    wire [511:0] pre_axi_rdata=dut.device.capacity_guard.s_axi_rdata;
    wire [1:0] pre_axi_rresp=dut.device.capacity_guard.s_axi_rresp;
    wire [3:0] pre_axi_rid=dut.device.capacity_guard.s_axi_rid;
    wire pre_axi_rlast=dut.device.capacity_guard.s_axi_rlast;
    wire pre_axi_rvalid=dut.device.capacity_guard.s_axi_rvalid;
    wire pre_axi_rready=dut.device.capacity_guard.s_axi_rready;
    wire [63:0] pre_axi_awaddr=dut.device.capacity_guard.s_axi_awaddr;
    wire [7:0] pre_axi_awlen=dut.device.capacity_guard.s_axi_awlen;
    wire [2:0] pre_axi_awsize=dut.device.capacity_guard.s_axi_awsize;
    wire [2:0] pre_axi_awprot=dut.device.capacity_guard.s_axi_awprot;
    wire [1:0] pre_axi_awburst=dut.device.capacity_guard.s_axi_awburst;
    wire [3:0] pre_axi_awid=dut.device.capacity_guard.s_axi_awid;
    wire [3:0] pre_axi_awcache=dut.device.capacity_guard.s_axi_awcache;
    wire pre_axi_awlock=dut.device.capacity_guard.s_axi_awlock;
    wire pre_axi_awvalid=dut.device.capacity_guard.s_axi_awvalid;
    wire pre_axi_awready=dut.device.capacity_guard.s_axi_awready;
    wire [511:0] pre_axi_wdata=dut.device.capacity_guard.s_axi_wdata;
    wire [63:0] pre_axi_wstrb=dut.device.capacity_guard.s_axi_wstrb;
    wire pre_axi_wlast=dut.device.capacity_guard.s_axi_wlast;
    wire pre_axi_wvalid=dut.device.capacity_guard.s_axi_wvalid;
    wire pre_axi_wready=dut.device.capacity_guard.s_axi_wready;
    wire [3:0] pre_axi_bid=dut.device.capacity_guard.s_axi_bid;
    wire [1:0] pre_axi_bresp=dut.device.capacity_guard.s_axi_bresp;
    wire pre_axi_bvalid=dut.device.capacity_guard.s_axi_bvalid;
    wire pre_axi_bready=dut.device.capacity_guard.s_axi_bready;
    ntrpu_axi_dma_guard_bfm dma(.clk(host_clk),.rst_n(host_rst_n),.held(host_dma_held),.ready(host_dma_ready),.*);
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

    localparam int LIMIT=16384;
    logic [7:0] memory[0:7][0:LIMIT-1],host_image[0:3][0:LIMIT-1],fixture_bytes[0:LIMIT-1];
    logic [7:0] golden[0:3][0:65535];
    bit written[0:7][0:LIMIT-1];
    logic [63:0] bases[0:7];
    int spans[0:7];
    integer d=32,h=32;
    localparam bit OUTPUT_FP32=1;
    localparam integer DOWN_BYTES=4;
    bit pause_host_reads=0;
    int ident=0,seed=1,rng,host_rng,clock_count=0;
    int core_ar,core_r,core_aw,core_w,core_b,core_bytes,down_bytes;
    int host_aw,host_w,host_b,host_ar,host_r,host_load_bytes,host_result_bytes;
    int phase_cycles[0:3],core_cycles,total_cycles=0,total_bytes=0;
    int ar_stalls=0,aw_stalls=0,w_stalls=0,done_stalls=0,contention=0;
    bit computing=0,server_idle=1;
    wire [1:0] phase=dut.device.compute.fabric.fabric.device.core.controller.phase;
    function automatic void need(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL FFN TRANSFER id=%0d clock=%0d: %s",ident,clock_count,why);
    endfunction
    function automatic int buffer_id(input logic [63:0] addr);
        for(int b=0;b<8;b++) if(addr>=bases[b]+4032 && addr<bases[b]+4032+64'(spans[b])) return b;
        need(0,"memory address outside registered spans");return -1;
    endfunction
    function automatic logic [511:0] word_at(input logic [63:0] addr);
        logic [511:0] value;
        int b,off;
        b=buffer_id(addr);off=int'(addr-bases[b]);
        need(off+64<=4032+spans[b],"read bounds");
        for(int lane=0;lane<64;lane++) begin
            need(written[b][off+lane],"read before host/core produced data");
            value[lane*8+:8]=memory[b][off+lane];
        end
        return value;
    endfunction
    task automatic tick;@(posedge clk);#1;endtask
    task automatic host_tick;@(posedge host_clk);#0.1;endtask
    // Passive scoreboard only: all R/READY/B signals come from MIG.
    // Payload is loaded solely through accepted hostAXI W beats.
    initial begin : memory_observer
        bit reading,aw_seen,read_host,write_host,arhs,rhs,awhs,whs,bhs;
        logic [63:0] read_addr,write_addr;
        logic [3:0] read_id,write_id;
        logic [511:0] queue[0:3];
        logic [63:0] masks[0:3];bit lasts[0:3];
        int read_left,burst_len,received,stored,b,off;
        reading=0;aw_seen=0;received=0;stored=0;
        forever begin
            @(posedge clk);
            if(!rst_n) begin
                need(!any_axi,"runtime reset outside cold-bank scope");
                reading=0;aw_seen=0;received=0;stored=0;server_idle=1;
            end else if(!probing) begin
                arhs=m_axi_arvalid&&m_axi_arready;rhs=m_axi_rvalid&&m_axi_rready;
                awhs=m_axi_awvalid&&m_axi_awready;whs=m_axi_wvalid&&m_axi_wready;bhs=m_axi_bvalid&&m_axi_bready;
                if(arhs) begin
                    need(!reading && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arcache==2 &&
                         m_axi_arprot==0 && !m_axi_arlock && (m_axi_arid==0 || m_axi_arid==9),"shared AR contract");
                    need(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"shared AR page");
                    need(m_axi_araddr<CAPACITY && m_axi_araddr+(64'(m_axi_arlen)+1)*64<=CAPACITY,"AR128GiB range before narrowing");
                    read_host=m_axi_arid==9;read_id=m_axi_arid;read_addr=m_axi_araddr;
                    read_left=int'(m_axi_arlen)+1;reading=1;
                    b=buffer_id(read_addr);
                    if(!read_host) begin
                        need(computing && (phase==0 ? b==0 || b==1:phase==1 ? b==0 || b==2:phase==2 ? b==4 || b==5:b==6 || b==3),"core read phase");
                        core_ar++;
                    end else host_ar++;
                    $display("NTRPU_TRANSFER_AR id=%0d host=%0d addr=%016h beats=%0d",ident,read_host,read_addr,read_left);
                end
                if(rhs) begin
                    need(reading && read_left>0,"unexpected memory R");
                    need(m_axi_rresp==0 && m_axi_rid==read_id && m_axi_rlast==(read_left==1),"MIG R identity/response/last");
                    need(m_axi_rdata===word_at(read_addr),"MIG R data differs from accepted writes");
                    $display("NTRPU_DDR_FFN_R id=%0d ps=%0d host=%0d addr=%016h rid=%0d last=%0d data=%0128h",ident,longint'($realtime*1000),read_host,read_addr,m_axi_rid,m_axi_rlast,m_axi_rdata);
                    if(read_host) host_r++;else core_r++;
                    read_left--;read_addr+=64;if(read_left==0) reading=0;
                end
                if(awhs) begin
                    need(!aw_seen && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awcache==2 &&
                         m_axi_awprot==0 && !m_axi_awlock && (m_axi_awid==0 || m_axi_awid==9),"shared AW contract");
                    need(m_axi_awaddr<CAPACITY && m_axi_awaddr+(64'(m_axi_awlen)+1)*64<=CAPACITY,"AW128GiB range before narrowing");
                    write_host=m_axi_awid==9;write_id=m_axi_awid;write_addr=m_axi_awaddr;
                    burst_len=int'(m_axi_awlen)+1;need(burst_len<=4 && int'(write_addr[11:0])+burst_len*64<=4096,"AW burst/page");
                    aw_seen=1;b=buffer_id(write_addr);
                    if(write_host) begin need(b<=3 && !computing,"host writes immutable inputs before start");host_aw++;end
                    else begin need(computing && b==int'(phase)+4,"core write phase");core_aw++;end
                    $display("NTRPU_TRANSFER_AW id=%0d host=%0d addr=%016h beats=%0d",ident,write_host,write_addr,burst_len);
                end
                if(whs) begin
                    need(received<4,"shared W queue overflow");
                    queue[received]=m_axi_wdata;masks[received]=m_axi_wstrb;lasts[received]=m_axi_wlast;received++;
                end
                if(aw_seen) while(stored<received) begin
                    b=buffer_id(write_addr);off=int'(write_addr-bases[b])+stored*64;
                    need(off+64<=4032+spans[b] && lasts[stored]==(stored==burst_len-1) && masks[stored]=='1,"shared W bounds/last/strobe");
                    for(int lane=0;lane<64;lane++) begin
                        need(write_host || !written[b][off+lane],"duplicate core output write");
                        need(queue[stored][lane*8+:8]===(write_host ? host_image[b][off+lane]:golden[b-4][off+lane-4032]),"physical memory byte");
                        memory[b][off+lane]=queue[stored][lane*8+:8];written[b][off+lane]=1;
                    end
                    if(write_host) begin
                        host_w++;host_load_bytes+=64;
                        $display("NTRPU_TRANSFER_LOAD id=%0d buffer=%0d offset=%0d data=%0128h",ident,b,off-4032,queue[stored]);
                    end else begin
                        core_w++;core_bytes+=64;if(b==7) down_bytes+=64;
                        $display("NTRPU_FFN_MEMORY_W id=%0d buffer=%0d offset=%0d strb=%016h data=%0128h",ident,b,off-4032,masks[stored],queue[stored]);
                    end
                    stored++;
                end
                if(bhs) begin
                    need(aw_seen && received==burst_len && stored==received,"B drain");
                    need(m_axi_bresp==0 && m_axi_bid==write_id,"MIG B identity/response");
                    $display("NTRPU_DDR_FFN_B id=%0d ps=%0d host=%0d bid=%0d resp=%0d",ident,longint'($realtime*1000),write_host,m_axi_bid,m_axi_bresp);
                    if(write_host) host_b++;else core_b++;
                    aw_seen=0;received=0;stored=0;
                end
                server_idle=!reading && !aw_seen && received==0 && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid;
            end
        end
    end
    always @(posedge clk) begin
        clock_count++;
        need(clock_count<1500000,"integration time budget");
        if(computing) begin
            core_cycles++;phase_cycles[int'(phase)]++;
            ar_stalls+=int'(dut.device.compute.fabric.fabric.s_axi_arvalid[0]&&!dut.device.compute.fabric.fabric.s_axi_arready[0]);
            aw_stalls+=int'(dut.device.compute.fabric.fabric.s_axi_awvalid[0]&&!dut.device.compute.fabric.fabric.s_axi_awready[0]);
            w_stalls+=int'(dut.device.compute.fabric.fabric.s_axi_wvalid[0]&&!dut.device.compute.fabric.fabric.s_axi_wready[0]);
            if(dut.device.compute.fabric.fabric.s_axi_arvalid==2'b11) contention++;
        end
        if(done_valid && !done_ready) done_stalls++;
    end
    function automatic int q(input int index);
        if(index%32==0) return 127;
        if(index%32==1) return 1;
        return (index%32)*7-113;
    endfunction
    function automatic logic [7:0] expected_byte(input int b,input int off);
        int row,lane,k,source;
        logic [63:0] bits;
        if(b==6) begin
            k=off/64;lane=off%64;
            if(lane<32) return k*32+lane<h ? 8'(q(k*32+lane)):8'd0;
            bits=$realtobits(real'(257+2*k));
            if(lane<40) return bits[(lane-32)*8+:8];
            return 0;
        end
        row=off/((b==7) ? DOWN_BYTES:8);lane=off%((b==7) ? DOWN_BYTES:8);
        if(b==4) bits=$realtobits(real'(257+2*(row/32)));
        else if(b==5) bits=$realtobits(real'(q(row)));
        else begin
            source=(row*7)%h;
            if(OUTPUT_FP32) bits={32'b0,$shortrealtobits(shortreal'((257+2*(source/32))*q(source)))};
            else bits=$realtobits(real'((257+2*(source/32))*q(source)));
        end
        return bits[lane*8+:8];
    endfunction
    always @(posedge clk) begin
        if(!rst_n && !ui_rst && !any_axi && server_idle && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid &&
           m_axi_rvalid===0 && m_axi_bvalid===0) begin
            if(quiet_edges==7) memory_peer_held<=1;else quiet_edges<=quiet_edges+1;
        end else begin quiet_edges<=0;memory_peer_held<=0;end
        if(rst_n && !ui_rst && calibrated) begin
            if(up_edges==7) memory_peer_ready<=1;else up_edges<=up_edges+1;
        end else begin up_edges<=0;memory_peer_ready<=0;end
    end
    task automatic trace(input string kind);
        $display("NTRPU_DMA_REC_%s id=%0d fault=%0d ps=%0d dt=%0d ht=%0d mt=%0d den=%0d hen=%0d ext=%0d rst=%0d hrst=%0d da=%0d ha=%0d run=%0d dh=%0d dr=%0d mh=%0d mr=%0d",
            kind,ident,fault,longint'($realtime*1000),dt,ht,mt,device_enabled,host_enabled,external_reset_async,
            rst_n,host_rst_n,device_admit,host_admit,running,host_dma_held,host_dma_ready,memory_peer_held,memory_peer_ready);
    endtask
    always @(rst_n or host_rst_n or host_admit or device_admit or running)
        begin #0.001;trace("OUT");end
    always @(posedge mgmt_clk) begin #0.001;trace("M");end
    task automatic await_run();
        integer watchdog;
        watchdog=0;
        while(!(running && device_admit && host_admit && job_ready)) begin
            @(negedge mgmt_clk);watchdog++;
            need(watchdog<100000,"MIG cold-start admission timeout");
        end
        repeat(8) @(negedge host_clk);
        need(!dma.busy && !s_host_axi_bvalid && !s_host_axi_rvalid && !dut.raw_irq && !done_valid &&
             !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && server_idle,"stale traffic after recovery");
    endtask

    // Separate pre-FFN probe phase. All memory responses still originate in MIG.
    bit probing=0;
    integer probe_tx=0,probe_response=0;
    bit [14:0] gp_held='0;
    logic[1023:0] gp_previous[0:14];
    integer gp_hold_count[0:14]='{default:0};
    task automatic gp_watch(input integer ch,input logic valid,ready,input logic[1023:0] sample);
        if(gp_held[ch]) need(valid && sample===gp_previous[ch],"probe VALID/payload changed while stalled");
        gp_held[ch]=valid&&!ready;gp_previous[ch]=sample;
        if(valid&&!ready) begin
            gp_hold_count[ch]++;
            $display("NTRPU_GP_HOLD tx=%0d ch=%0d ps=%0d sample=%0256h",probe_tx,ch,longint'($realtime*1000),sample);
        end
    endtask
    always @(posedge host_clk) if(probing) begin
        need(rst_n && host_rst_n && running && device_admit && host_admit && calibrated && !ui_rst,"probe admission lost");
        gp_watch(0,s_host_axi_arvalid,s_host_axi_arready,{s_host_axi_araddr,s_host_axi_arlen,s_host_axi_arsize,s_host_axi_arburst,s_host_axi_arid,s_host_axi_arcache,s_host_axi_arprot,s_host_axi_arlock});
        if(s_host_axi_arvalid&&s_host_axi_arready) $display("NTRPU_GP_ADDR tx=%0d side=0 ps=%0d write=0 addr=%016h beats=%0d size=%0d burst=%0d aid=%0d cache=%0d prot=%0d lock=%0d",probe_tx,longint'($realtime*1000),s_host_axi_araddr,int'(s_host_axi_arlen)+1,s_host_axi_arsize,s_host_axi_arburst,s_host_axi_arid,s_host_axi_arcache,s_host_axi_arprot,s_host_axi_arlock);
        gp_watch(1,s_host_axi_awvalid,s_host_axi_awready,{s_host_axi_awaddr,s_host_axi_awlen,s_host_axi_awsize,s_host_axi_awburst,s_host_axi_awid,s_host_axi_awcache,s_host_axi_awprot,s_host_axi_awlock});
        if(s_host_axi_awvalid&&s_host_axi_awready) $display("NTRPU_GP_ADDR tx=%0d side=0 ps=%0d write=1 addr=%016h beats=%0d size=%0d burst=%0d aid=%0d cache=%0d prot=%0d lock=%0d",probe_tx,longint'($realtime*1000),s_host_axi_awaddr,int'(s_host_axi_awlen)+1,s_host_axi_awsize,s_host_axi_awburst,s_host_axi_awid,s_host_axi_awcache,s_host_axi_awprot,s_host_axi_awlock);
        gp_watch(2,s_host_axi_wvalid,s_host_axi_wready,{s_host_axi_wdata,s_host_axi_wstrb,s_host_axi_wlast});
        if(s_host_axi_wvalid&&s_host_axi_wready) $display("NTRPU_GP_W tx=%0d side=0 ps=%0d data=%0128h strb=%016h last=%0d",probe_tx,longint'($realtime*1000),s_host_axi_wdata,s_host_axi_wstrb,s_host_axi_wlast);
        gp_watch(3,s_host_axi_rvalid,s_host_axi_rready,{s_host_axi_rdata,s_host_axi_rresp,s_host_axi_rid,s_host_axi_rlast});
        if(s_host_axi_rvalid&&s_host_axi_rready) $display("NTRPU_GP_R tx=%0d side=0 ps=%0d data=%0128h resp=%0d aid=%0d last=%0d",probe_tx,longint'($realtime*1000),s_host_axi_rdata,s_host_axi_rresp,s_host_axi_rid,s_host_axi_rlast);
        gp_watch(4,s_host_axi_bvalid,s_host_axi_bready,{s_host_axi_bresp,s_host_axi_bid});
        if(s_host_axi_bvalid&&s_host_axi_bready) $display("NTRPU_GP_B tx=%0d side=0 ps=%0d resp=%0d aid=%0d",probe_tx,longint'($realtime*1000),s_host_axi_bresp,s_host_axi_bid);
    end
    always @(posedge clk) if(probing) begin
        need(rst_n && host_rst_n && running && device_admit && host_admit && calibrated && !ui_rst,"probe admission lost");
        gp_watch(5,pre_axi_arvalid,pre_axi_arready,{pre_axi_araddr,pre_axi_arlen,pre_axi_arsize,pre_axi_arburst,pre_axi_arid,pre_axi_arcache,pre_axi_arprot,pre_axi_arlock});
        if(pre_axi_arvalid&&pre_axi_arready) $display("NTRPU_GP_ADDR tx=%0d side=1 ps=%0d write=0 addr=%016h beats=%0d size=%0d burst=%0d aid=%0d cache=%0d prot=%0d lock=%0d",probe_tx,longint'($realtime*1000),pre_axi_araddr,int'(pre_axi_arlen)+1,pre_axi_arsize,pre_axi_arburst,pre_axi_arid,pre_axi_arcache,pre_axi_arprot,pre_axi_arlock);
        gp_watch(6,pre_axi_awvalid,pre_axi_awready,{pre_axi_awaddr,pre_axi_awlen,pre_axi_awsize,pre_axi_awburst,pre_axi_awid,pre_axi_awcache,pre_axi_awprot,pre_axi_awlock});
        if(pre_axi_awvalid&&pre_axi_awready) $display("NTRPU_GP_ADDR tx=%0d side=1 ps=%0d write=1 addr=%016h beats=%0d size=%0d burst=%0d aid=%0d cache=%0d prot=%0d lock=%0d",probe_tx,longint'($realtime*1000),pre_axi_awaddr,int'(pre_axi_awlen)+1,pre_axi_awsize,pre_axi_awburst,pre_axi_awid,pre_axi_awcache,pre_axi_awprot,pre_axi_awlock);
        gp_watch(7,pre_axi_wvalid,pre_axi_wready,{pre_axi_wdata,pre_axi_wstrb,pre_axi_wlast});
        if(pre_axi_wvalid&&pre_axi_wready) $display("NTRPU_GP_W tx=%0d side=1 ps=%0d data=%0128h strb=%016h last=%0d",probe_tx,longint'($realtime*1000),pre_axi_wdata,pre_axi_wstrb,pre_axi_wlast);
        gp_watch(8,pre_axi_rvalid,pre_axi_rready,{pre_axi_rdata,pre_axi_rresp,pre_axi_rid,pre_axi_rlast});
        if(pre_axi_rvalid&&pre_axi_rready) $display("NTRPU_GP_R tx=%0d side=1 ps=%0d data=%0128h resp=%0d aid=%0d last=%0d",probe_tx,longint'($realtime*1000),pre_axi_rdata,pre_axi_rresp,pre_axi_rid,pre_axi_rlast);
        gp_watch(9,pre_axi_bvalid,pre_axi_bready,{pre_axi_bresp,pre_axi_bid});
        if(pre_axi_bvalid&&pre_axi_bready) $display("NTRPU_GP_B tx=%0d side=1 ps=%0d resp=%0d aid=%0d",probe_tx,longint'($realtime*1000),pre_axi_bresp,pre_axi_bid);
    end
    always @(posedge clk) if(probing) begin
        need(rst_n && host_rst_n && running && device_admit && host_admit && calibrated && !ui_rst,"probe admission lost");
        if(probe_response==3) need(!m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && !m_axi_rvalid && !m_axi_bvalid,"invalid probe exposed to MIG");
        gp_watch(10,m_axi_arvalid,m_axi_arready,{m_axi_araddr,m_axi_arlen,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock});
        if(m_axi_arvalid&&m_axi_arready) $display("NTRPU_GP_ADDR tx=%0d side=2 ps=%0d write=0 addr=%016h beats=%0d size=%0d burst=%0d aid=%0d cache=%0d prot=%0d lock=%0d",probe_tx,longint'($realtime*1000),m_axi_araddr,int'(m_axi_arlen)+1,m_axi_arsize,m_axi_arburst,m_axi_arid,m_axi_arcache,m_axi_arprot,m_axi_arlock);
        gp_watch(11,m_axi_awvalid,m_axi_awready,{m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock});
        if(m_axi_awvalid&&m_axi_awready) $display("NTRPU_GP_ADDR tx=%0d side=2 ps=%0d write=1 addr=%016h beats=%0d size=%0d burst=%0d aid=%0d cache=%0d prot=%0d lock=%0d",probe_tx,longint'($realtime*1000),m_axi_awaddr,int'(m_axi_awlen)+1,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock);
        gp_watch(12,m_axi_wvalid,m_axi_wready,{m_axi_wdata,m_axi_wstrb,m_axi_wlast});
        if(m_axi_wvalid&&m_axi_wready) $display("NTRPU_GP_W tx=%0d side=2 ps=%0d data=%0128h strb=%016h last=%0d",probe_tx,longint'($realtime*1000),m_axi_wdata,m_axi_wstrb,m_axi_wlast);
        gp_watch(13,m_axi_rvalid,m_axi_rready,{m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast});
        if(m_axi_rvalid&&m_axi_rready) $display("NTRPU_GP_R tx=%0d side=2 ps=%0d data=%0128h resp=%0d aid=%0d last=%0d",probe_tx,longint'($realtime*1000),m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast);
        gp_watch(14,m_axi_bvalid,m_axi_bready,{m_axi_bresp,m_axi_bid});
        if(m_axi_bvalid&&m_axi_bready) $display("NTRPU_GP_B tx=%0d side=2 ps=%0d resp=%0d aid=%0d",probe_tx,longint'($realtime*1000),m_axi_bresp,m_axi_bid);
    end

    task automatic probe_op(input bit write_access,input logic[63:0] addr,input integer beats,tag,response);
        @(negedge host_clk);probe_tx++;probe_response=response;
        $display("NTRPU_GP_BEGIN tx=%0d ps=%0d write=%0d addr=%016h beats=%0d tag=%0d resp=%0d",probe_tx,longint'($realtime*1000),write_access,addr,beats,tag,response);
        dma.probe(write_access,addr,beats,tag,response,probe_tx%3);
        repeat(4) @(negedge clk);
        need(capacity_guard_idle && !s_host_axi_bvalid && !s_host_axi_rvalid && !pre_axi_arvalid &&
             !pre_axi_awvalid && !pre_axi_wvalid && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid &&
             !m_axi_rvalid && !m_axi_bvalid && gp_held==0,"probe completely drained");
        $display("NTRPU_GP_END tx=%0d ps=%0d idle=%0d",probe_tx,longint'($realtime*1000),capacity_guard_idle);
    endtask
    task automatic probe_bank();
        logic[63:0] sentinel[0:2],invalid_addr[0:5];integer lengths[0:5];
        sentinel[0]=0;sentinel[1]=64'h10000;sentinel[2]=CAPACITY-256;
        invalid_addr[0]=CAPACITY;lengths[0]=1;
        invalid_addr[1]=CAPACITY+65536;lengths[1]=4;
        invalid_addr[2]=64'h1000000010000;lengths[2]=4;
        invalid_addr[3]=64'h8000001fffffff00;lengths[3]=4;
        invalid_addr[4]=64'hffffffffffffff00;lengths[4]=4;
        invalid_addr[5]=8*CAPACITY+65536;lengths[5]=3;
        @(negedge clk);probing=1;
        $display("NTRPU_GP_START ps=%0d capacity=137438953472",longint'($realtime*1000));
        for(integer region=0;region<3;region++) probe_op(1,sentinel[region],4,region+1,0);
        for(integer region=0;region<3;region++) probe_op(0,sentinel[region],4,region+1,0);
        for(integer attempt=0;attempt<6;attempt++) begin
            probe_op(1,invalid_addr[attempt],lengths[attempt],100+attempt,3);
            probe_op(0,invalid_addr[attempt],lengths[attempt],100+attempt,3);
            for(integer region=0;region<3;region++) probe_op(0,sentinel[region],4,region+1,0);
        end
        for(integer ch=0;ch<15;ch++) $display("NTRPU_GP_STALL ch=%0d count=%0d",ch,gp_hold_count[ch]);
        need(probe_tx==36 && gp_hold_count[3]>0 && gp_hold_count[4]>0,"probe coverage");
        $display("NTRPU_GP_PASS tx=36 invalid=12 sentinel_bytes=768 host_w=32 host_r=104 mig_w=12 mig_r=84 ps=%0d",longint'($realtime*1000));
        @(negedge clk);probing=0;
    endtask

    task automatic clear_metrics();
        need(server_idle && !dma.busy && job_ready,"clear metrics only when drained");
        core_ar=0;core_r=0;core_aw=0;core_w=0;core_b=0;core_bytes=0;down_bytes=0;core_cycles=0;
        host_ar=0;host_r=0;host_aw=0;host_w=0;host_b=0;host_load_bytes=0;host_result_bytes=0;
        for(integer p=0;p<4;p++) phase_cycles[p]=0;
        // Payload RAM is retained across reset. Only the coverage of writes
        // for the next FFN is cleared, so old intermediates cannot satisfy it.
        for(integer b=4;b<8;b++) for(integer i=0;i<LIMIT;i++) written[b][i]=0;
    endtask
    task automatic images();
        logic[63:0] one;
        integer off,source;
        one=$realtobits(1.0);
        for(integer lane=0;lane<64;lane++) host_image[0][4032+lane]=0;
        for(integer lane=0;lane<32;lane++) host_image[0][4032+lane]=8'(q(lane));
        for(integer lane=0;lane<8;lane++) host_image[0][4064+lane]=one[lane*8+:8];
        for(integer b=1;b<=3;b++) begin
            off=4032;
            for(integer row=0;row<32;row++) begin
                host_image[b][off]=b==1 ? 8'h04:8'h00;
                host_image[b][off+1]=b==1 ? 8'h5c:8'h3c;
                source=b==1 ? 1:b==2 ? row:(row*7)%32;
                for(integer lane=0;lane<32;lane++) host_image[b][off+2+lane]=lane==source ? 1:0;
                off+=34;
            end
            need(off==4032+spans[b],"synthetic packed weight length");
        end
        for(integer b=4;b<8;b++) for(integer off=0;off<spans[b];off++) golden[b-4][off]=expected_byte(b,off);
    endtask
    task automatic load_inputs();
        logic[511:0] words[0:3];
        integer beats,off;
        bit aborted;
        for(integer b=0;b<4;b++) begin
            off=0;
            while(off<spans[b]) begin
                beats=(spans[b]-off)/64;if(beats>4) beats=4;
                if(beats>(4096-((4032+off)%4096))/64) beats=(4096-((4032+off)%4096))/64;
                for(integer beat=0;beat<beats;beat++) for(integer lane=0;lane<64;lane++)
                    words[beat][lane*8+:8]=host_image[b][4032+off+beat*64+lane];
                dma.write(bases[b]+4032+64'(off),beats,words,(off/64)%3==2 ? 4:0,(off/64)%3==1 ? 4:0,3,aborted);
                need(!aborted,"normal load aborted");off+=beats*64;
            end
        end
        need(server_idle && host_aw==host_b,"input loading did not drain");
    endtask
    task automatic run_ffn();
        logic[511:0] words[0:3];
        bit aborted;
        integer watchdog;
        need(host_load_bytes==3328,"all synthetic inputs must arrive through host AXI");
        $display("NTRPU_DMA_REC_FFN_BEGIN id=%0d ps=%0d",ident,longint'($realtime*1000));
        @(negedge clk);job_valid=1;job_layout=2;job_dim=32;job_hidden=32;
        job_sequence='hfffffffe;job_epoch=32'(ident+600);expected_epoch=job_epoch;
        for(integer p=0;p<3;p++) job_tile_k[p]=2;
        for(integer b=0;b<8;b++) begin
            job_addr[b]=bases[b]+4032;job_base[b]=bases[b];job_bytes[b]=64'(4032+spans[b]);job_buffer_epoch[b]=job_epoch;
        end
        do begin @(posedge clk);end while(!job_ready);
        #1;job_valid=0;computing=1;job_dim=0;job_hidden=0;job_layout=0;job_sequence=0;job_epoch=0;expected_epoch=0;
        for(integer b=0;b<8;b++) begin job_addr[b]=0;job_base[b]=0;job_bytes[b]=0;job_buffer_epoch[b]=0;end
        for(integer p=0;p<3;p++) job_tile_k[p]=0;
        watchdog=0;
        while(!done_valid) begin tick();watchdog++;need(watchdog<30000,"FFN timeout");end
        computing=0;
        need(done_status==0 && done_child_status==0 && done_stages==4 && done_phase==3 &&
             done_bytes==128 && done_stage_bytes==704 && done_sequence=='hfffffffe &&
             done_epoch==32'(ident+600) && done_hls_status==0,"FFN completion identity/accounting");
        need(core_aw==core_b && core_bytes==704 && down_bytes==128 && server_idle,"FFN memory drain");
        for(integer off=0;off<128;off+=64) begin
            dma.read(bases[7]+4032+64'(off),1,2,words,aborted);need(!aborted,"result read aborted");
            for(integer lane=0;lane<64;lane++) need(words[0][lane*8+:8]===golden[3][off+lane],"result byte");
            host_result_bytes+=64;
        end
        @(negedge clk);done_ready=1;tick();@(negedge clk);done_ready=0;
        watchdog=0;while(!job_ready) begin tick();watchdog++;need(watchdog<5000,"FFN software retirement");end
        for(integer b=0;b<8;b++) for(integer i=0;i<LIMIT;i++)
            need(written[b][i]===(i>=4032 && i<4032+spans[b]),"observed AXI write coverage");
        $display("NTRPU_FFN_MEMORY_CASE id=%0d dim=32 hidden=32 mode=0 target=0 reset=0 cycles=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d bytes=%0d down_bytes=%0d p0=%0d p1=%0d p2=%0d p3=%0d",
            ident,core_cycles,core_ar,core_r,core_aw,core_w,core_b,core_bytes,down_bytes,phase_cycles[0],phase_cycles[1],phase_cycles[2],phase_cycles[3]);
        $display("NTRPU_DMA_REC_FFN_END id=%0d ps=%0d load_bytes=%0d result_bytes=%0d",ident,longint'($realtime*1000),host_load_bytes,host_result_bytes);
        total_cycles+=core_cycles;total_bytes+=core_bytes;
    endtask
    initial begin : scenarios
        logic[511:0] words[0:3],readback[0:3];
        logic[63:0] retained_hash;
        bit aborted,stop_host;
        integer kind,watchdog,aw_before,w_before,b_before,ar_before,r_before;
        job_valid=0;cancel_valid=0;done_ready=0;job_layout=0;job_dim=0;job_hidden=0;
        job_sequence=0;job_epoch=0;expected_epoch=0;
        for(integer p=0;p<3;p++) job_tile_k[p]=0;
        for(integer b=0;b<8;b++) begin
            bases[b]=64'h100000000+64'(b)*64'h20000000;
            job_addr[b]=0;job_base[b]=0;job_bytes[b]=0;job_buffer_epoch[b]=0;
            for(integer i=0;i<LIMIT;i++) written[b][i]=0;
        end
        spans[0]=64;spans[1]=1088;spans[2]=1088;spans[3]=1088;
        spans[4]=256;spans[5]=256;spans[6]=64;spans[7]=128;
        if($value$plusargs("SEED=%d",seed)) begin end
        if($value$plusargs("HOST_HALF_PS=%d",host_half_ps)) begin end
        need(host_half_ps==2000 || host_half_ps==3100,"host period");
        rng=seed;host_rng=seed+1;host_clock_ready=1;
        $display("NTRPU_FFN_MEMORY_SEED seed=%0d",seed);
        $display("NTRPU_FFN_MEMORY_FORMAT fp32=1");
        $display("NTRPU_DMA_REC_CONFIG host_period_ps=%0d device_period_ps=5000 mgmt_period_ps=10000 dim=32 hidden=32",host_half_ps*2);
        wait(ui_rst===0);
        repeat(64) @(negedge mgmt_clk);external_reset_async=0;await_run();
        $display("NTRPU_DDR_FFN_ADMITTED ps=%0d calib=%b ui_rst=%b",longint'($realtime*1000),calibrated,ui_rst);
        probe_bank();
        for(integer iteration=1;iteration<=3;iteration++) begin
            ident=iteration;
            for(integer b=0;b<8;b++) begin
                bases[b]=64'h100000000+64'((b+iteration-1)%4)*64'h800000000+64'(b/4)*64'h100000;
                for(integer i=0;i<LIMIT;i++) written[b][i]=0;
            end
            if(iteration==3) begin bases[0]=64'h100000000-4096;bases[7]=CAPACITY-4160;end
            for(integer b=0;b<8;b++)
                $display("NTRPU_DDR_FFN_REGION id=%0d buffer=%0d base=%016h addr=%016h bytes=%0d",ident,b,bases[b],bases[b]+4032,spans[b]);
            images();clear_metrics();load_inputs();run_ffn();
        end
        for(integer rank=0;rank<4;rank++) need(rank_acts[rank]>0,"all four physical ranks must activate");
        $display("NTRPU_FFN_MEMORY_PASS cases=3 cycles=%0d bytes=%0d resets=0 cancels=0 ar_stalls=%0d aw_stalls=%0d w_stalls=%0d done_stalls=%0d",total_cycles,total_bytes,ar_stalls,aw_stalls,w_stalls,done_stalls);
        $display("NTRPU_DDR_FFN_PASS ffns=3 fp32=96 load_bytes=9984 result_bytes=384 rank0=%0d rank1=%0d rank2=%0d rank3=%0d",rank_acts[0],rank_acts[1],rank_acts[2],rank_acts[3]);
        need(capacity_guard_idle,"capacity guard must drain before success");
        $display("NTRPU_DDR_GUARD_PASS idle=%0d address_width=37 ffns=3",capacity_guard_idle);
        $display("NTRPU_GUARDED_TOP_PASS ffns=3 probes=36 memory_addr_width=37 idle=%0d",capacity_guard_idle);
        complete=1;$finish;
    end
    initial begin #2000000;$fatal(1,"FAIL DDR FFN global watchdog");end
    final if(!complete) $error("NTRPU_DDR_FFN_INCOMPLETE");
endmodule
