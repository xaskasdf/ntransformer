`timescale 1ps/1ps
module ntrpu_pcie_cq_check_tb #(parameter bit HOST64 = 0);
    reg clk = 0, reset = 1, valid = 0, ready = 1, last = 0;
    reg [511:0] data = 0;
    reg [15:0] keep = 0;
    reg [182:0] user = 0;
    integer mode = 0, offset, lane, sent, count, first_lane, packet;
    integer packet_bytes, packet_offset = 0, region, request_offset;
    reg [63:0] base;
    ntrpu_pcie_cq_check #(.PAYLOAD_BYTES(HOST64 ? 1536 : 768), .HOST64(HOST64)) check_cq(.*);
    always #2000 clk = ~clk;

    task drive_beat(input bit stall);
        begin
            valid = 1;
            ready = !stall;
            if (stall) begin
                @(negedge clk);
                if (mode == 6) data[200] = ~data[200];
                @(negedge clk);
                ready = 1;
            end
            @(negedge clk);
            valid = 0;
            // Two deliberate VALID gaps between all accepted beats.
            repeat (2) @(negedge clk);
        end
    endtask

    task read_request(input logic [63:0] addr, input integer bytes_count);
        begin
            data = 0; user = 0; keep = 'hf; last = 1;
            data[63:0] = addr; data[74:64] = bytes_count/4;
            user[80] = 1; user[3:0] = 'hf; user[11:8] = 'hf;
            drive_beat(0);
        end
    endtask

    initial begin
        if (!$value$plusargs("MODE=%d", mode)) mode = 0;
        repeat (3) @(negedge clk);
        reset = 0;
        if (HOST64) begin
            read_request(64'h100000100, 32);
            read_request(64'h200000100, 32);
            for (region = 1; region <= 2; region++) begin
                base = (64'(region) << 32) + 'hfc0;
                if (mode == 8) read_request(base, 128);
                read_request(base, 64);
                read_request(base+64, 256);
                read_request(base+320, 256);
                read_request(base+576, 192);
            end
            read_request(64'h300000300, 32);
            read_request(64'h400000300, 32);
        end else begin
            read_request('h100, 32);
            read_request('h400, 512);
            read_request('h600, 256);
            read_request('h300, 32);
        end
        // Three 256-byte writes: five beats each, each final beat has 16 bytes.
        for (packet = 0; packet < (HOST64 ? 8 : 3); packet++) begin
            packet_bytes = !HOST64 ? 256 : (packet%4 == 0 ? 64 : (packet%4 == 3 ? 192 : 256));
            sent = 0;
            while (sent < packet_bytes) begin
                data = 0; user = 0; keep = 0;
                first_lane = sent == 0 ? 16 : 0;
                count = (packet_bytes-sent < 64-first_lane) ? packet_bytes-sent : 64-first_lane;
                if (sent == 0) begin
                    data[63:0] = HOST64 ? ((64'(3+packet/4) << 32) + 'hfc0 + packet_offset%768) : 'h800 + packet_offset;
                    if (mode == 2 && packet == 1) data[63:0] = 'h800;
                    if (HOST64 && mode == 9 && packet == 4) data[63:32] = 3;
                    data[74:64] = packet_bytes/4; data[78:75] = 1;
                    user[80] = 1; user[3:0] = 'hf; user[11:8] = 'hf;
                    keep[3:0] = 'hf;
                end
                for (lane = first_lane; lane < first_lane+count; lane++) begin
                    offset = packet_offset + sent + lane-first_lane;
                    // Separate stimulus formula: quotient/remainder instead of checker shifts.
                    data[lane*8 +: 8] = (offset % 256) ^ (offset / 256) ^ 8'ha5;
                    user[16+lane] = 1;
                    keep[lane/4] = 1;
                end
                last = sent+count == packet_bytes;
                if (packet == 0 && sent == 0) begin
                    case (mode)
                        1: data[128] = ~data[128];
                        3: last = 1;
                        4: keep[8] = 0;
                        5: user[35] = 1'bx;
                    endcase
                end
                drive_beat(sent == 0);
                sent += count;
                if (mode == 7 && packet == 0) begin
                    check_cq.verify_complete();
                    $fatal(1, "Negative incomplete test was not rejected");
                end
            end
            packet_offset += packet_bytes;
        end
        check_cq.verify_complete();
        $display("NTRPU_CQ_UNIT_PASS mode=%0d host64=%0d", mode, HOST64);
        $finish;
    end
    initial begin #10000000; $fatal(1, "CQ_UNIT_TIMEOUT"); end
endmodule

module ntrpu_pcie_cq_host64_tb;
    ntrpu_pcie_cq_check_tb #(.HOST64(1)) test();
endmodule
