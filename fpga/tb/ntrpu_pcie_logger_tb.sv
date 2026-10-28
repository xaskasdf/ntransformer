`timescale 1ns/1ps
`define RX_LOG 1'b0
`define TX_LOG 1'b1
`define SYNC_CC_RDY 1'b1
`define SYNC_RQ_RDY 1'b0
module ntrpu_logger_com;
    reg [7:0] frame_store_tx [0:5119];
    reg [7:0] frame_store_rx [0:5119];
    integer frame_store_tx_idx = 0;
    integer frame_store_rx_idx = 0;
    integer parsed = 0, parsed_bytes = 0;
    `include "logger_tasks.svh"
    // Call-count observer only. Full DMA keeps the unmodified vendor parser.
    task TSK_PARSE_FRAME(input bit txrx);
        begin
            if (!txrx) $fatal(1, "LOGGER_SYNC_DIRECTION");
            parsed++;
            parsed_bytes += frame_store_tx_idx;
            frame_store_tx_idx = 0;
        end
    endtask
endmodule
module ntrpu_logger_tx;
    reg user_clk = 0;
    reg user_lnk_up_n = 0;
    reg s_axis_cc_tready = 1, s_axis_rq_tready = 1;
    reg [511:0] pcie_tlp_data = 0;
    reg [3:0] pcie_tlp_rem = 13;
    always #2 user_clk = ~user_clk;
    `include "logger_sync_tasks.svh"
endmodule
module ntrpu_logger_rp;
    ntrpu_logger_com com_usrapp();
    ntrpu_logger_tx tx_usrapp();
endmodule
module board;
    ntrpu_logger_rp RP();
    reg [511:0] data;
    reg [63:0] cq;
    reg [1:0] fmt, attr;
    reg [4:0] kind;
    reg [2:0] tc;
    reg td, ep;
    reg [9:0] length;
    integer mode, i, rem_value, direction, checks = 0;

    task decode(input bit tx, input [1:0] expected_fmt,
                input [4:0] expected_type, input [9:0] expected_length);
        begin
            RP.com_usrapp.TSK_DECIPHER_FRAME(fmt, kind, tc, td, ep, attr, length, tx);
            if ({fmt, kind, length} !== {expected_fmt, expected_type, expected_length})
                $fatal(1, "LOGGER_HEADER fmt=%h kind=%h length=%h", fmt, kind, length);
            checks++;
        end
    endtask

    initial begin
        if (!$value$plusargs("MODE=%d", mode)) mode = 0;
        data = 0;
        data[511:480] = 32'h4a000008; // Completion with eight data DWORDs.
        data[255:248] = 3; // Destination-high byte that broke the original logger.
        case (mode)
            1: RP.com_usrapp.TSK_NTRPU_LOG_TX(0, 1, 0, 0, 0);
            2: RP.com_usrapp.TSK_NTRPU_LOG_TX(1, 1, 0, data, 0);
            3: begin
                data = 0; data[255:224] = 32'h40000006;
                RP.com_usrapp.TSK_NTRPU_LOG_TX(1, 1, 0, data, 0);
            end
            4: RP.com_usrapp.TSK_READ_DATA_512(1, 1, 1, data, 4'bx001);
            5: begin
                data = 0; data[255] = 1'bx;
                RP.com_usrapp.TSK_NTRPU_LOG_TX(1, 1, 0, data, 0);
            end
            default: begin
                RP.com_usrapp.TSK_NTRPU_LOG_TX(1, 1, 1, data, 5);
                if (RP.com_usrapp.frame_store_tx_idx != 44) $fatal(1, "LOGGER_CC_SIZE");
                RP.com_usrapp.TSK_ORIGINAL_DECIPHER(fmt, kind, tc, td, ep, attr, length, 1);
                if (kind !== 3) $fatal(1, "LOGGER_REPRO_NOT_OBSERVED");
                $display("NTRPU_LOGGER_ORIGINAL_DEFECT_REPRODUCED type=%0d", kind);
                decode(1, 2, 10, 8);
                // All 16 remainder encodings in both directions, including bytes.
                for (i = 0; i < 64; i++) data[511-i*8 -: 8] = i;
                for (direction = 0; direction < 2; direction++)
                    for (rem_value = 0; rem_value < 16; rem_value++) begin
                        RP.com_usrapp.frame_store_tx_idx = 0;
                        RP.com_usrapp.frame_store_rx_idx = 0;
                        RP.com_usrapp.TSK_READ_DATA_512(1, 1, direction, data, rem_value);
                        if ((direction ? RP.com_usrapp.frame_store_tx_idx : RP.com_usrapp.frame_store_rx_idx)
                            != 64-4*rem_value) $fatal(1, "LOGGER_REMAINDER_SIZE");
                        for (i = 0; i < 64-4*rem_value; i++)
                            if ((direction ? RP.com_usrapp.frame_store_tx[i] : RP.com_usrapp.frame_store_rx[i]) !== 8'(i))
                                $fatal(1, "LOGGER_BYTE_ORDER");
                        checks++;
                    end
                // Real generated control layout: zero prefix then a 256-bit image.
                for (i = 0; i < 4; i++) begin
                    data = 0;
                    case (i)
                        0: data[255:224] = 32'h04000001; // Config read: 12 bytes.
                        1: data[255:224] = 32'h44000001; // Config write: 16 bytes.
                        2: data[255:224] = 32'h40000001; // BAR write: 16 bytes.
                        3: data[255:224] = 32'h20000001; // Memory read64: 16 bytes.
                    endcase
                    data[159:128] = 32'h12345678;
                    RP.com_usrapp.frame_store_tx_idx = 0;
                    RP.com_usrapp.TSK_NTRPU_LOG_TX(1, 1, 0, data, 0);
                    if (RP.com_usrapp.frame_store_tx_idx != (i == 0 ? 12 : 16))
                        $fatal(1, "LOGGER_RQ_COUNT");
                    decode(1, i == 0 ? 0 : (i == 3 ? 1 : 2), i < 2 ? 4 : 0, 1);
                    if (i != 0 && RP.com_usrapp.frame_store_tx[12] !== 8'h12)
                        $fatal(1, "LOGGER_RQ_BODY");
                end
                // Multi-beat completion keeps header and appends every payload byte.
                RP.com_usrapp.frame_store_tx_idx = 0;
                data = 0; data[511:480] = 32'h4a000010;
                RP.com_usrapp.TSK_NTRPU_LOG_TX(1, 0, 1, data, 0);
                data = 0; data[511:480] = 32'h11223344;
                RP.com_usrapp.TSK_NTRPU_LOG_TX(0, 1, 1, data, 13);
                if (RP.com_usrapp.frame_store_tx_idx != 76 || RP.com_usrapp.frame_store_tx[64] !== 8'h11)
                    $fatal(1, "LOGGER_CC_APPEND");
                decode(1, 2, 10, 16);
                for (i = 0; i < 4; i++) begin
                    cq = 0; cq[14:11] = i % 2; cq[9:0] = 16;
                    RP.com_usrapp.frame_store_rx_idx = 0;
                    RP.com_usrapp.TSK_BUILD_CQ_TO_PCIE_PKT(i < 2 ? 64'h300000fc0 : 64'hfc0, 8'hff, cq);
                    decode(0, (i < 2 ? 1 : 0) + (i % 2 ? 2 : 0), 0, 16);
                end
                // The actual vendor synchronization task is entered twice by
                // the completion's fork: AXIS wait-only and log final beat.
                RP.com_usrapp.frame_store_tx_idx = 64;
                fork
                    RP.tx_usrapp.TSK_ORIGINAL_TX_SYNCHRONIZE(0, 0, 0, `SYNC_CC_RDY);
                    RP.tx_usrapp.TSK_ORIGINAL_TX_SYNCHRONIZE(0, 1, 1, `SYNC_CC_RDY);
                join
                if (RP.com_usrapp.parsed == 1) $fatal(1, "LOGGER_STATIC_RACE_NOT_REPRODUCED");
                $display("NTRPU_LOGGER_STATIC_RACE_REPRODUCED parses=%0d", RP.com_usrapp.parsed);
                RP.com_usrapp.parsed = 0;
                RP.com_usrapp.parsed_bytes = 0;
                RP.com_usrapp.frame_store_tx_idx = 64;
                fork
                    RP.tx_usrapp.TSK_TX_SYNCHRONIZE(0, 0, 0, `SYNC_CC_RDY);
                    RP.tx_usrapp.TSK_TX_SYNCHRONIZE(0, 1, 1, `SYNC_CC_RDY);
                join
                if (RP.com_usrapp.parsed != 1 || RP.com_usrapp.parsed_bytes != 76 ||
                    RP.com_usrapp.frame_store_tx_idx != 0) $fatal(1, "LOGGER_SYNC_FINAL_COUNT");
                $display("NTRPU_LOGGER_CONCURRENT_SYNC_PASS parses=1 bytes=76");
                $display("NTRPU_LOGGER_UNIT_PASS checks=%0d remainder_cases=32 original_defect=1", checks);
                $finish;
            end
        endcase
        $fatal(1, "LOGGER_NEGATIVE_ESCAPED mode=%0d", mode);
    end
endmodule
