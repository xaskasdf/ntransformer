// Isolate warm-reset semantics in one installed generic DDR4 model.
// Mode register values are from the reviewed full-channel command trace.
`timescale 1ps/1ps
module ntrpu_ddr_model_reset_tb;
    import arch_package::*;
    logic ck = 0;
    int open_row = 0;
    time reset_low_ps = 2600000;
    int full_delays = 0;
    time cold_reset_ps, cold_cke_delay_ps, warm_cke_delay_ps;
    bit complete = 0;
    always #625 ck = ~ck;
    DDR4_if #(.CONFIGURED_DQ_BITS(4)) memory();
    assign memory.CK = {ck, ~ck};
    assign memory.DM_n = 1'b1;
    assign memory.DQ = 4'bzzzz;
    assign memory.DQS_t = 1'bz;
    assign memory.DQS_c = 1'bz;
    ddr4_model #(.CONFIGURED_DQ_BITS(4), .CONFIGURED_DENSITY(_16G)) dut (
        .model_enable(), .iDDR4(memory)
    );
    task automatic command(input logic [3:0] opcode, input int bg, ba, addr);
        @(negedge ck);
        memory.CS_n = 0;
        {memory.ACT_n, memory.RAS_n_A16, memory.CAS_n_A15, memory.WE_n_A14} = opcode;
        memory.BG = 2'(bg); memory.BA = 2'(ba); memory.ADDR = 14'(addr);
        @(negedge ck); memory.CS_n = 1;
        repeat (15) @(negedge ck); // At least the captured 20 ns MRS spacing.
    endtask
    task automatic initialize_modes;
        command(4'b1000, 0, 3, 'h000);
        command(4'b1000, 1, 2, 'h424);
        command(4'b1000, 1, 1, 'h040);
        command(4'b1000, 1, 0, 'h000);
        command(4'b1000, 0, 2, 'h000);
        command(4'b1000, 0, 1, 'h401);
        command(4'b1000, 0, 0, 'h314);
        #200000;
        command(4'b1110, 0, 0, 'h714); // ZQCL, ADDR[10]=1.
        #2000000;
    endtask
    initial begin
        if (!$value$plusargs("OPEN_ROW=%d", open_row)) open_row = 0;
        if (!$value$plusargs("RESET_LOW_PS=%d", reset_low_ps)) reset_low_ps = 2600000;
        if (!$value$plusargs("FULL_DELAYS=%d", full_delays)) full_delays = 0;
        if (full_delays != 0 && full_delays != 1) $fatal(1, "NTRPU_MODEL bad timing mode");
        cold_reset_ps = full_delays ? 200000000 : 200000;
        cold_cke_delay_ps = full_delays ? 500000000 : 600000;
        warm_cke_delay_ps = full_delays ? 500000000 : 800000;
        $display("NTRPU_MODEL_TIMING full_delays=%0d cold_reset_ps=%0d cold_cke_ps=%0d warm_cke_ps=%0d",
                 full_delays, cold_reset_ps, cold_cke_delay_ps, warm_cke_delay_ps);
        if (open_row != 0 && open_row != 1) $fatal(1, "NTRPU_MODEL bad mode");
        if (reset_low_ps < 100000 || reset_low_ps > 1000000000) $fatal(1, "NTRPU_MODEL bad reset width");
        memory.RESET_n = 0; memory.CKE = 0; memory.CS_n = 1;
        memory.ACT_n = 1; memory.RAS_n_A16 = 1; memory.CAS_n_A15 = 1; memory.WE_n_A14 = 1;
        memory.BA = 0; memory.BG = 0; memory.ADDR = 0; memory.ADDR_17 = 0;
        memory.ODT = 0; memory.PARITY = 0; memory.TEN = 0; memory.C = 0;
        memory.ZQ = 1; memory.PWR = 1; memory.VREF_CA = 1; memory.VREF_DQ = 1;
        #(cold_reset_ps) memory.RESET_n = 1;
        #(cold_cke_delay_ps) memory.CKE = 1;
        #600000;
        initialize_modes();
        $display("NTRPU_MODEL_COLD_DONE time_ps=%0d", $time);
        if (open_row) command(4'b0000, 0, 0, 'h123);
        #100000;
        @(negedge ck); memory.RESET_n = 0; memory.CKE = 0; memory.CS_n = 1;
        $display("NTRPU_MODEL_WARM_RESET open_row=%0d low_ps=%0d time_ps=%0d", open_row, reset_low_ps, $time);
        #(reset_low_ps) memory.RESET_n = 1;
        #(warm_cke_delay_ps) memory.CKE = 1;
        #600000;
        initialize_modes();
        complete = 1;
        $display("NTRPU_MODEL_WINDOW_COMPLETE open_row=%0d time_ps=%0d", open_row, $time);
        $finish;
    end
    initial begin #2000000000; $fatal(1, "NTRPU_MODEL timeout"); end
    final if (!complete) $error("NTRPU_MODEL_INCOMPLETE");
endmodule
