// Test-only scoreboard for PG213 CQ, 512 bits, DWORD alignment, no straddle.
// Bounded aligned transfers: original fixture or two 64-bit descriptor regions.
`timescale 1ps/1ps
module ntrpu_pcie_cq_check #(
    parameter integer PAYLOAD_BYTES = 768,
    parameter bit HOST64 = 0
) (
    input wire clk, reset, valid, ready, last,
    input wire [511:0] data,
    input wire [15:0] keep,
    input wire [182:0] user
);
    integer c2h_bytes = 0, c2h_packets = 0, h2c_bytes = 0, h2c_requests = 0;
    integer descriptors = 0, packet_left = 0, accepted = 0, stalls = 0;
    integer lane, first_lane, beat_bytes, packet_bytes;
    bit in_packet = 0, held = 0;
    reg [63:0] address;
    reg [711:0] held_bus;
    reg [15:0] expected_keep;
    reg [63:0] expected_be;
    time first_c2h = 0, last_c2h = 0;
    reg [3:0] descriptor_seen = 0;
    integer descriptor_index, scan;
    reg [63:0] expected_address, descriptor_address;

    function automatic [63:0] host_address(input bit c2h, input integer offset);
        if (HOST64)
            host_address = (64'((c2h ? 3 : 1) + offset/768) << 32) + 'hfc0 + offset%768;
        else host_address = (c2h ? 64'h800 : 64'h400) + offset;
    endfunction

    function automatic [7:0] pattern(input integer offset);
        pattern = 8'(offset ^ (offset >> 8) ^ 'ha5);
    endfunction

    always @(posedge clk) begin
        if (!reset) begin
            if (held && (valid !== 1'b1 || {data, keep, user, last} !== held_bus))
                $fatal(1, "CQ_STALLED_CHANGED");
            held = valid === 1'b1 && ready !== 1'b1;
            held_bus = {data, keep, user, last};
            if (held) stalls++;
            if (valid === 1'b1 && ready === 1'b1) begin
                accepted++;
                if ($isunknown({keep, last, user[85:80], user[79:16]}))
                    $fatal(1, "CQ_UNKNOWN_CONTROL");
                first_lane = 0;
                if (!in_packet) begin
                    if (user[81:80] !== 2'b01 || user[83:82] !== 2'b00 ||
                        keep[3:0] !== 4'hf || $isunknown(data[127:0]))
                        $fatal(1, "CQ_BAD_HEADER");
                    if (data[79] || data[1:0] != 0 || user[3:0] !== 4'hf || user[11:8] !== 4'hf)
                        $fatal(1, "CQ_UNSUPPORTED_REQUEST");
                    address = {data[63:2], 2'b00};
                    packet_bytes = int'(data[74:64]) * 4;
                    if (packet_bytes <= 0 || packet_bytes > PAYLOAD_BYTES)
                        $fatal(1, "CQ_BAD_LENGTH");
                    if (HOST64 && (packet_bytes > 256 || int'(address[11:0])+packet_bytes > 4096))
                        $fatal(1, "CQ_PAGE_OR_SIZE");
                    $display("NTRPU_CQ_REQUEST time_ps=%0d type=%0d address=%h bytes=%0d",
                        $time, data[78:75], address, packet_bytes);
                    if (data[78:75] == 0) begin
                        descriptor_index = -1;
                        if (HOST64) begin
                            for (scan = 0; scan < 4; scan++) begin
                                descriptor_address = (64'(scan+1) << 32) + (scan < 2 ? 'h100 : 'h300);
                                if (address == descriptor_address) descriptor_index = scan;
                            end
                        end
                        if (HOST64 && descriptor_index >= 0) begin
                            if (packet_bytes != 32 || descriptor_seen[descriptor_index])
                                $fatal(1, "CQ_BAD_DESCRIPTOR");
                            descriptor_seen[descriptor_index] = 1;
                            descriptors++;
                        end else if (!HOST64 && (address == 'h100 || address == 'h300)) begin
                            if (packet_bytes != 32 || descriptors >= 2 ||
                                address != (descriptors == 0 ? 'h100 : 'h300))
                                $fatal(1, "CQ_BAD_DESCRIPTOR");
                            descriptors++;
                        end else begin
                            expected_address = host_address(0, h2c_bytes);
                            if (address != expected_address || h2c_bytes + packet_bytes > PAYLOAD_BYTES ||
                                (HOST64 && h2c_bytes%768+packet_bytes > 768))
                                $fatal(1, "CQ_H2C_RANGE");
                            h2c_bytes += packet_bytes;
                            h2c_requests++;
                        end
                        if (keep !== 16'h000f || last !== 1'b1 || user[79:16] !== 64'b0)
                            $fatal(1, "CQ_READ_HAS_PAYLOAD");
                    end else if (data[78:75] == 1) begin
                        expected_address = host_address(1, c2h_bytes);
                        if (address != expected_address || c2h_bytes + packet_bytes > PAYLOAD_BYTES ||
                            (HOST64 && c2h_bytes%768+packet_bytes > 768))
                            $fatal(1, "CQ_C2H_RANGE");
                        c2h_packets++;
                        packet_left = packet_bytes;
                        first_lane = 16;
                        in_packet = 1;
                    end else $fatal(1, "CQ_UNEXPECTED_TYPE");
                end else if (user[81:80] !== 2'b00) $fatal(1, "CQ_NESTED_PACKET");

                if (in_packet) begin
                    beat_bytes = (packet_left < 64-first_lane) ? packet_left : 64-first_lane;
                    expected_keep = 16'hffff >> (16 - (first_lane + beat_bytes)/4);
                    expected_be = (64'hffffffffffffffff >> (64-beat_bytes)) << first_lane;
                    if (keep !== expected_keep || user[79:16] !== expected_be)
                        $fatal(1, "CQ_BAD_BYTE_ENABLE");
                    for (lane = first_lane; lane < first_lane+beat_bytes; lane++) begin
                        if (data[lane*8 +: 8] !== pattern(c2h_bytes))
                            $fatal(1, "CQ_DATA_MISMATCH offset=%0d", c2h_bytes);
`ifdef NTRPU_PCIE_HOST64
                        if (HOST64) board.ntrpu_host.write_byte(host_address(1, c2h_bytes), data[lane*8 +: 8]);
`endif
                        if (c2h_bytes == 0) first_c2h = $time;
                        c2h_bytes++;
                    end
                    last_c2h = $time;
                    packet_left -= beat_bytes;
                    if (last !== (packet_left == 0)) $fatal(1, "CQ_BAD_LAST");
                    if (last) in_packet = 0;
                end
            end
        end
    end

    task verify_complete;
        begin
            if (in_packet || held || c2h_bytes != PAYLOAD_BYTES ||
                h2c_bytes != PAYLOAD_BYTES || descriptors != (HOST64 ? 4 : 2) ||
                (HOST64 && descriptor_seen != 4'hf) ||
                c2h_packets < 2 || h2c_requests < 2)
                $fatal(1, "CQ_INCOMPLETE");
            $display("NTRPU_CQ_COMPLETE c2h_bytes=%0d c2h_packets=%0d h2c_bytes=%0d h2c_requests=%0d descriptors=%0d beats=%0d stalls=%0d first_c2h_ps=%0d last_c2h_ps=%0d",
                c2h_bytes, c2h_packets, h2c_bytes, h2c_requests, descriptors,
                accepted, stalls, first_c2h, last_c2h);
        end
    endtask
endmodule
