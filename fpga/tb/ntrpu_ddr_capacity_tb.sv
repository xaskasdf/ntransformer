// Generic two-slot 128 GiB DDR4 channel. Not an exact Micron F1 qualification.
`timescale 1ps/1ps
`ifdef XILINX_SIMULATOR
module short(in1, in1);
    inout in1;
endmodule
`endif
module ntrpu_ddr_capacity_tb #(parameter bit ECC_TEST = 0, parameter bit RESET_TEST = 0,
                              parameter integer RESET_HOLD_CYCLES = 250);
    import arch_package::*;
    localparam longint unsigned CAPACITY = 64'd137438953472;
    localparam integer LOCATIONS = 36;
    logic sys_clk = 0, sys_rst = 1, aresetn = 0;
    always #2000 sys_clk = ~sys_clk; // Actual IP reference period: 4000 ps.
    initial begin #20000; sys_rst = 0; end
    wire ui_clk, ui_rst, calibrated, interrupt_ecc;
    wire act_n, reset_n, parity;
    wire [17:0] adr;
    wire [1:0] ba, bg, ck_t, ck_c;
    wire [3:0] cke, odt, cs_n;
    wire [71:0] dq;
    wire [17:0] dqs_t, dqs_c;
    logic [36:0] awaddr = 0, araddr = 0;
    logic [7:0] awlen = 0, arlen = 0;
    logic awvalid = 0, wvalid = 0, wlast = 0, bready = 0, arvalid = 0, rready = 0;
    logic [511:0] wdata = 0;
    logic [63:0] wstrb = 0;
    wire awready, wready, bvalid, arready, rvalid, rlast;
    wire [511:0] rdata;
    wire [1:0] bresp, rresp;
    wire [3:0] bid, rid;
    logic [31:0] ctrl_awaddr = 0, ctrl_wdata = 0, ctrl_araddr = 0;
    logic ctrl_awvalid = 0, ctrl_wvalid = 0, ctrl_bready = 0;
    logic ctrl_arvalid = 0, ctrl_rready = 0;
    wire ctrl_awready, ctrl_wready, ctrl_bvalid, ctrl_arready, ctrl_rvalid;
    wire [1:0] ctrl_bresp, ctrl_rresp;
    wire [31:0] ctrl_rdata;
    bit expected_ecc_window = 0;
    longint unsigned cycles = 0, writes = 0, reads = 0, checked_bytes = 0;
    longint unsigned w_stall = 0, r_stall = 0, rank_acts[4] = '{default:0};
    time calibrated_ps = 0, first_axi_ps = 0, done_ps = 0;
    bit complete = 0, holding_r = 0;
    logic [518:0] held_r;
    longint unsigned addresses[LOCATIONS];
    logic [511:0] expected_data[LOCATIONS];

    ntrpu_ddr_geometry dut (
        .sys_rst(sys_rst), .c0_sys_clk_p(sys_clk), .c0_sys_clk_n(~sys_clk),
        .c0_ddr4_act_n(act_n), .c0_ddr4_adr(adr), .c0_ddr4_ba(ba), .c0_ddr4_bg(bg),
        .c0_ddr4_cke(cke), .c0_ddr4_odt(odt), .c0_ddr4_cs_n(cs_n),
        .c0_ddr4_ck_t(ck_t), .c0_ddr4_ck_c(ck_c), .c0_ddr4_reset_n(reset_n),
        .c0_ddr4_parity(parity), .c0_ddr4_dq(dq), .c0_ddr4_dqs_t(dqs_t), .c0_ddr4_dqs_c(dqs_c),
        .c0_init_calib_complete(calibrated), .c0_ddr4_ui_clk(ui_clk),
        .c0_ddr4_ui_clk_sync_rst(ui_rst), .dbg_clk(), .dbg_bus(),
        .c0_ddr4_aresetn(aresetn),
        .c0_ddr4_s_axi_ctrl_awvalid(ctrl_awvalid), .c0_ddr4_s_axi_ctrl_awaddr(ctrl_awaddr),
        .c0_ddr4_s_axi_ctrl_awready(ctrl_awready), .c0_ddr4_s_axi_ctrl_wvalid(ctrl_wvalid),
        .c0_ddr4_s_axi_ctrl_wdata(ctrl_wdata), .c0_ddr4_s_axi_ctrl_wready(ctrl_wready),
        .c0_ddr4_s_axi_ctrl_bvalid(ctrl_bvalid), .c0_ddr4_s_axi_ctrl_bready(ctrl_bready), .c0_ddr4_s_axi_ctrl_bresp(ctrl_bresp),
        .c0_ddr4_s_axi_ctrl_arvalid(ctrl_arvalid), .c0_ddr4_s_axi_ctrl_araddr(ctrl_araddr),
        .c0_ddr4_s_axi_ctrl_arready(ctrl_arready), .c0_ddr4_s_axi_ctrl_rvalid(ctrl_rvalid),
        .c0_ddr4_s_axi_ctrl_rready(ctrl_rready), .c0_ddr4_s_axi_ctrl_rdata(ctrl_rdata), .c0_ddr4_s_axi_ctrl_rresp(ctrl_rresp),
        .c0_ddr4_interrupt(interrupt_ecc),
        .c0_ddr4_s_axi_awid(4'b0), .c0_ddr4_s_axi_awaddr(awaddr), .c0_ddr4_s_axi_awlen(awlen),
        .c0_ddr4_s_axi_awsize(3'd6), .c0_ddr4_s_axi_awburst(2'b01),
        .c0_ddr4_s_axi_awlock(1'b0), .c0_ddr4_s_axi_awcache(4'b0),
        .c0_ddr4_s_axi_awprot(3'b0), .c0_ddr4_s_axi_awqos(4'b0),
        .c0_ddr4_s_axi_awvalid(awvalid), .c0_ddr4_s_axi_awready(awready),
        .c0_ddr4_s_axi_wdata(wdata), .c0_ddr4_s_axi_wstrb(wstrb), .c0_ddr4_s_axi_wlast(wlast),
        .c0_ddr4_s_axi_wvalid(wvalid), .c0_ddr4_s_axi_wready(wready),
        .c0_ddr4_s_axi_bready(bready), .c0_ddr4_s_axi_bid(bid), .c0_ddr4_s_axi_bresp(bresp),
        .c0_ddr4_s_axi_bvalid(bvalid),
        .c0_ddr4_s_axi_arid(4'b0), .c0_ddr4_s_axi_araddr(araddr), .c0_ddr4_s_axi_arlen(arlen),
        .c0_ddr4_s_axi_arsize(3'd6), .c0_ddr4_s_axi_arburst(2'b01),
        .c0_ddr4_s_axi_arlock(1'b0), .c0_ddr4_s_axi_arcache(4'b0),
        .c0_ddr4_s_axi_arprot(3'b0), .c0_ddr4_s_axi_arqos(4'b0),
        .c0_ddr4_s_axi_arvalid(arvalid), .c0_ddr4_s_axi_arready(arready),
        .c0_ddr4_s_axi_rready(rready), .c0_ddr4_s_axi_rid(rid), .c0_ddr4_s_axi_rdata(rdata),
        .c0_ddr4_s_axi_rresp(rresp), .c0_ddr4_s_axi_rlast(rlast), .c0_ddr4_s_axi_rvalid(rvalid)
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
                    if (cs_n[slot*2+rank] === 1'b0) rank_acts[slot*2+rank]++;
        end
    end

    always @(posedge ui_clk) begin
        if (ui_rst) aresetn <= 0;
        else aresetn <= 1;
        if (!aresetn || ui_rst || !calibrated) holding_r = 0;
        if (aresetn && calibrated) begin
            cycles++;
            if (wvalid && !wready) w_stall++;
            if (rvalid && !rready) r_stall++;
            if (holding_r && (!rvalid || {rid,rresp,rlast,rdata} !== held_r))
                $fatal(1, "NTRPU_DDR AXI read changed under backpressure");
            holding_r = rvalid && !rready;
            held_r = {rid,rresp,rlast,rdata};
            if (!expected_ecc_window && interrupt_ecc === 1'b1)
                $fatal(1, "NTRPU_DDR unexpected ECC interrupt");
        end
    end

    function automatic logic [511:0] pattern(input longint unsigned addr, input int seed);
        logic [511:0] value;
        for (int lane = 0; lane < 8; lane++)
            value[lane*64 +: 64] = (addr * 64'h9e3779b97f4a7c15) ^ (64'h0102040810204080 << lane) ^ (64'(seed) * 64'hd6e8feb86659fd93);
        return value;
    endfunction

    task automatic write_words(input longint unsigned addr, input int beats, input int seed, input logic [63:0] mask);
        if (addr + beats*64 > CAPACITY || (addr & 63) || ((addr & 4095) + beats*64 > 4096))
            $fatal(1, "NTRPU_DDR illegal test write");
        @(negedge ui_clk); awaddr = 37'(addr); awlen = 8'(beats-1); awvalid = 1;
        do @(posedge ui_clk); while (awready !== 1'b1);
        @(negedge ui_clk); awvalid = 0;
        for (int beat = 0; beat < beats; beat++) begin
            wdata = pattern(addr + beat*64, seed); wstrb = mask; wlast = (beat == beats-1); wvalid = 1;
            do @(posedge ui_clk); while (wready !== 1'b1);
            writes++;
            @(negedge ui_clk); wvalid = 0;
        end
        repeat (3) @(negedge ui_clk);
        bready = 1;
        do @(posedge ui_clk); while (bvalid !== 1'b1);
        if (bresp !== 0 || bid !== 0) $fatal(1, "NTRPU_DDR bad write response");
        @(negedge ui_clk); bready = 0;
    endtask

    task automatic read_words(input longint unsigned addr, input int beats, input int seed, input logic [511:0] first_expected);
        logic [511:0] wanted;
        @(negedge ui_clk); araddr = 37'(addr); arlen = 8'(beats-1); arvalid = 1;
        do @(posedge ui_clk); while (arready !== 1'b1);
        @(negedge ui_clk); arvalid = 0;
        for (int beat = 0; beat < beats; beat++) begin
            repeat (2) @(negedge ui_clk); rready = 1;
            do @(posedge ui_clk); while (rvalid !== 1'b1);
            wanted = beats == 1 ? first_expected : pattern(addr + beat*64, seed);
            if (rdata !== wanted || rresp !== 0 || rid !== 0 || rlast !== (beat == beats-1))
                $fatal(1, "NTRPU_DDR read mismatch addr=%h beat=%0d actual=%h expected=%h resp=%h last=%b", addr, beat, rdata, wanted, rresp, rlast);
            reads++; checked_bytes += 64;
            @(negedge ui_clk); rready = 0;
        end
    endtask

    task automatic ctrl_write(input logic [31:0] addr, input logic [31:0] value);
        // AW and W handshake independently; do not assume same-cycle readiness.
        fork
            begin
                @(negedge ui_clk); ctrl_awaddr = addr; ctrl_awvalid = 1;
                do @(posedge ui_clk); while (ctrl_awready !== 1'b1);
                @(negedge ui_clk); ctrl_awvalid = 0;
            end
            begin
                @(negedge ui_clk); ctrl_wdata = value; ctrl_wvalid = 1;
                do @(posedge ui_clk); while (ctrl_wready !== 1'b1);
                @(negedge ui_clk); ctrl_wvalid = 0;
            end
        join
        @(negedge ui_clk); ctrl_bready = 1;
        do @(posedge ui_clk); while (ctrl_bvalid !== 1'b1);
        if (ctrl_bresp !== 0) $fatal(1, "NTRPU_DDR control write response addr=%h", addr);
        @(negedge ui_clk); ctrl_bready = 0;
    endtask

    task automatic ctrl_read(input logic [31:0] addr, output logic [31:0] value);
        @(negedge ui_clk); ctrl_araddr = addr; ctrl_arvalid = 1;
        do @(posedge ui_clk); while (ctrl_arready !== 1'b1);
        @(negedge ui_clk); ctrl_arvalid = 0; ctrl_rready = 1;
        do @(posedge ui_clk); while (ctrl_rvalid !== 1'b1);
        if (ctrl_rresp !== 0) $fatal(1, "NTRPU_DDR control read response addr=%h", addr);
        value = ctrl_rdata;
        @(negedge ui_clk); ctrl_rready = 0;
    endtask

    task automatic require_ecc_status(input logic [31:0] expected_status, input bit irq);
        logic [31:0] value;
        ctrl_read(32'h0000, value);
        if (value !== expected_status || interrupt_ecc !== irq)
            $fatal(1, "NTRPU_DDR ECC status=%h expected=%h irq=%b expected_irq=%b", value, expected_status, interrupt_ecc, irq);
    endtask

    task automatic ecc_fault_test;
        logic [31:0] value;
        // Public register map in AMD ddr4_v2_2_axi_ctrl_top/reg_bank.sv:
        // status 0x000 W1C, IRQ enable 0x004, enable 0x008, CE count 0x00c,
        // one-shot XOR for data bits 31:0 of the next write at 0x300.
        $display("NTRPU_DDR_ECC_REQUESTED");
        require_ecc_status(0, 0);
        ctrl_read(32'h000c, value);
        if (value !== 0) $fatal(1, "NTRPU_DDR baseline CE count nonzero");
        ctrl_write(32'h0004, 3);
        expected_ecc_window = 1;
        ctrl_write(32'h0300, 1);
        write_words(64'h1_0000_1000, 1, 1001, '1);
        read_words(64'h1_0000_1000, 1, 0, pattern(64'h1_0000_1000, 1001));
        require_ecc_status(2, 1);
        ctrl_read(32'h000c, value);
        if (value !== 1) $fatal(1, "NTRPU_DDR expected one corrected event, got %h", value);
        $display("NTRPU_DDR_ECC_CORRECTION_PASS bits=1 status=2 ce_count=1 irq=1");
        ctrl_write(32'h0000, 3);
        require_ecc_status(0, 0);
        expected_ecc_window = 0;
        write_words(64'h1_0000_1000, 1, 1002, '1);
        read_words(64'h1_0000_1000, 1, 0, pattern(64'h1_0000_1000, 1002));
        require_ecc_status(0, 0);

        expected_ecc_window = 1;
        ctrl_write(32'h0300, 3);
        write_words(CAPACITY-1024, 1, 1003, '1);
        @(negedge ui_clk); araddr = 37'(CAPACITY-1024); arlen = 0; arvalid = 1;
        do @(posedge ui_clk); while (arready !== 1'b1);
        @(negedge ui_clk); arvalid = 0; rready = 1;
        do @(posedge ui_clk); while (rvalid !== 1'b1);
        // AMD AXI read channel converts multiple ECC errors to SLVERR (2).
        if (rresp !== 2 || rid !== 0 || rlast !== 1)
            $fatal(1, "NTRPU_DDR uncorrectable read missing SLVERR resp=%h", rresp);
        reads++; // Deliberately corrupted payload is not counted as compared bytes.
        @(negedge ui_clk); rready = 0;
        require_ecc_status(1, 1);
        ctrl_read(32'h000c, value);
        if (value !== 1) $fatal(1, "NTRPU_DDR UE changed CE counter");
        $display("NTRPU_DDR_ECC_DETECTION_PASS bits=2 status=1 rresp=2 irq=1");
        ctrl_write(32'h0000, 3);
        require_ecc_status(0, 0);
        expected_ecc_window = 0;
        write_words(CAPACITY-1024, 1, 1004, '1);
        read_words(CAPACITY-1024, 1, 0, pattern(CAPACITY-1024, 1004));
        require_ecc_status(0, 0);
        ctrl_read(32'h000c, value);
        if (value !== 1) $fatal(1, "NTRPU_DDR clean rewrite changed CE counter");
        ctrl_write(32'h000c, 0);
        ctrl_read(32'h000c, value);
        if (value !== 0) $fatal(1, "NTRPU_DDR CE counter did not clear");
        $display("NTRPU_DDR_ECC_PASS corrected=1 detected=1 clean_rewrites=2 irq_events=2");
    endtask

    task automatic reset_recovery_test;
        logic [31:0] value;
        time reset_at, recovered_at;
        longint unsigned addr;
        addr = CAPACITY-8192;
        $display("NTRPU_DDR_RESET_REQUESTED");
        write_words(addr, 8, 2001, '1);
        @(negedge ui_clk); araddr = 37'(addr); arlen = 7; arvalid = 1;
        do @(posedge ui_clk); while (arready !== 1'b1);
        @(negedge ui_clk); arvalid = 0; rready = 1;
        for (int beat = 0; beat < 2; beat++) begin
            do @(posedge ui_clk); while (rvalid !== 1'b1);
            if (rdata !== pattern(addr+beat*64, 2001) || rresp !== 0 || rid !== 0 || rlast !== 0)
                $fatal(1, "NTRPU_DDR pre-reset read mismatch beat=%0d", beat);
            reads++; checked_bytes += 64;
            @(negedge ui_clk);
        end
        rready = 0;
        do @(posedge ui_clk); while (rvalid !== 1'b1);
        repeat (3) @(negedge ui_clk); // Leave the next beat pending under backpressure.
        if (rvalid !== 1 || rlast !== 0) $fatal(1, "NTRPU_DDR no pending read at reset");
        @(negedge sys_clk); sys_rst = 1; reset_at = $time;
        $display("NTRPU_DDR_RESET_ASSERT time_ps=%0d aborted_read_beats=6", reset_at);
        // Reset duration for this accelerated model; not a board reset specification.
        repeat (RESET_HOLD_CYCLES) @(negedge sys_clk);
        if (calibrated !== 0) $fatal(1, "NTRPU_DDR calibration did not drop during reset");
        sys_rst = 0;
        wait (calibrated === 1'b1 && aresetn === 1'b1);
        recovered_at = $time;
        $display("NTRPU_DDR_RESET_RECALIBRATED time_ps=%0d elapsed_ps=%0d", recovered_at, recovered_at-reset_at);
        if (dimms[0].reset_low_at < reset_at || dimms[1].reset_low_at < reset_at ||
            dimms[0].last_reset_width < 100000 || dimms[1].last_reset_width < 100000)
            $fatal(1, "NTRPU_DDR RDIMM reset did not reach model minimum of 100 ns");
        // No pre-reset completion may leak into the new transaction epoch.
        repeat (32) begin
            @(posedge ui_clk);
            if (rvalid !== 0 || bvalid !== 0 || ctrl_rvalid !== 0 || ctrl_bvalid !== 0)
                $fatal(1, "NTRPU_DDR stale response after reset");
        end
        ctrl_read(32'h0008, value);
        if (value !== 1) $fatal(1, "NTRPU_DDR ECC checking not restored after reset");
        require_ecc_status(0, 0);
        ctrl_read(32'h000c, value);
        if (value !== 0) $fatal(1, "NTRPU_DDR CE counter not reset");
        // Rewrite before reading: warm reset does not promise retained application data.
        write_words(addr, 8, 2002, '1);
        read_words(addr, 8, 2002, '0);
        require_ecc_status(0, 0);
        $display("NTRPU_DDR_RESET_PASS interrupted_reads=1 aborted_beats=6 stale_responses=0 checked_bytes=640");
    endtask

    initial begin: test
        logic [511:0] patch_value;
        logic [31:0] ecc_enabled;
        $display("NTRPU_DDR_SCOPE generic_128gib_two_slot_x4_ecc time_unit=ps");
        wait (calibrated === 1'b1 && aresetn === 1'b1);
        calibrated_ps = $time;
        $display("NTRPU_DDR_CALIBRATED time_ps=%0d", calibrated_ps);
        ctrl_read(32'h0008, ecc_enabled);
        if (ecc_enabled !== 1) $fatal(1, "NTRPU_DDR ECC checking is not enabled");
        addresses[0] = 0;
        for (int bitno = 6; bitno <= 36; bitno++) addresses[bitno-5] = 64'd1 << bitno;
        addresses[32] = (64'd1 << 32) - 64;
        addresses[33] = (64'd1 << 35) - 64;
        addresses[34] = (64'd1 << 36) - 64;
        addresses[35] = CAPACITY - 64;
        first_axi_ps = $time;
        for (int i = 0; i < LOCATIONS; i++) begin
            expected_data[i] = pattern(addresses[i], i+1);
            write_words(addresses[i], 1, i+1, '1);
        end
        for (int i = LOCATIONS-1; i >= 0; i--) read_words(addresses[i], 1, 0, expected_data[i]);
        $display("NTRPU_DDR_SPARSE_COMPARE_PASS locations=%0d address_bits=6..36", LOCATIONS);
        for (int i = 0; i < LOCATIONS; i += 3) begin
            patch_value = pattern(addresses[i], 100+i);
            write_words(addresses[i], 1, 100+i, 64'h5555555555555555);
            for (int byte_no = 0; byte_no < 64; byte_no += 2)
                expected_data[i][byte_no*8 +: 8] = patch_value[byte_no*8 +: 8];
        end
        for (int i = LOCATIONS-1; i >= 0; i--) read_words(addresses[i], 1, 0, expected_data[i]);
        $display("NTRPU_DDR_MASKED_COMPARE_PASS locations=%0d", LOCATIONS);
        // Full bursts ending at 4 GiB and 128 GiB boundaries.
        write_words((64'd1 << 32)-512, 8, 901, '1);
        write_words(CAPACITY-512, 8, 902, '1);
        read_words((64'd1 << 32)-512, 8, 901, '0);
        read_words(CAPACITY-512, 8, 902, '0);
        for (int rank = 0; rank < 4; rank++)
            if (!rank_acts[rank]) $fatal(1, "NTRPU_DDR rank %0d never activated after calibration", rank);
        $display("NTRPU_DDR_CAPACITY_PASS locations=%0d checked_bytes=%0d", LOCATIONS, checked_bytes);
        require_ecc_status(0, 0);
        if (ECC_TEST) ecc_fault_test();
        if (RESET_TEST) begin
            if (!ECC_TEST) $fatal(1, "NTRPU_DDR reset mode requires ECC test");
            reset_recovery_test();
        end
        complete = 1; done_ps = $time;
        $finish;
    end
    initial begin
        #2000000000;
        $fatal(1, "NTRPU_DDR exceeded 2 ms simulation budget");
    end
    initial forever begin
        #10000000;
        $display("NTRPU_DDR_PROGRESS time_ps=%0d calib=%b ui_rst=%b writes=%0d reads=%0d", $time, calibrated, ui_rst, writes, reads);
    end
    final begin
        $display("NTRPU_DDR_METRICS complete=%0d calibrated_ps=%0d first_axi_ps=%0d done_ps=%0d cycles=%0d writes=%0d reads=%0d checked_bytes=%0d w_stall=%0d r_stall=%0d rank0=%0d rank1=%0d rank2=%0d rank3=%0d",
            complete, calibrated_ps, first_axi_ps, done_ps, cycles, writes, reads, checked_bytes,
            w_stall, r_stall, rank_acts[0], rank_acts[1], rank_acts[2], rank_acts[3]);
        if (!complete) $error("NTRPU_DDR_INCOMPLETE");
    end
endmodule
