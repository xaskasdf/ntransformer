`timescale 1ps/1ps
module ntrpu_host_memory_tb;
    ntrpu_host_memory memory();
    integer mode = 0, region, i, count, offset, global_offset;
    reg [63:0] base, next_pointer;
    reg [7:0] value, expected;
    initial begin
        if (!$value$plusargs("MODE=%d", mode)) mode = 0;
        memory.initialize();
        case (mode)
            1: value = memory.read_completion(64'h0000000000000fc0, 64, 0);
            2: value = memory.read_completion(64'h0000000100000fc0, 128, 0);
            3: memory.write_byte(64'h0000000100000fc0, 0);
            4: begin
                memory.write_byte(64'h0000000300000fc0, 0);
                memory.write_byte(64'h0000000300000fc0, 0);
            end
            5: begin
                memory.initialized_byte[4096] = 0;
                value = memory.read_completion(64'h0000000100000fc0, 64, 0);
            end
            6: value = memory.read_completion(64'h00000001000012c0, 64, 0);
            7: value = memory.read_completion(64'hxxxxxxxx00000fc0, 64, 0);
        endcase
        if (mode != 0) $fatal(1, "HOST_UNIT_NEGATIVE_MISSED");
        for (region = 0; region < 4; region++) begin
            base = (64'(region+1) << 32) + (region < 2 ? 'h100 : 'h300);
            next_pointer = 0;
            for (i = 0; i < 32; i++) begin
                value = memory.read_completion(base, 32, i);
                if (i >= 24) next_pointer[(i-24)*8 +: 8] = value;
            end
            if (next_pointer !== (region % 2 == 0 ? base+64'h100000000 : 64'd0))
                $fatal(1, "HOST_UNIT_NEXT_POINTER");
            if (memory.read_completion(base, 32, 50) !== 0) $fatal(1, "HOST_UNIT_CC_PADDING");
        end
        for (region = 0; region < 2; region++) begin
            base = (64'(region+1) << 32) + 'hfc0;
            offset = 0;
            while (offset < 768) begin
                count = 256;
                if (count > 4096-((int'(base)+offset)%4096)) count = 4096-((int'(base)+offset)%4096);
                if (count > 768-offset) count = 768-offset;
                for (i = 0; i < count; i++) begin
                    global_offset = region*768+offset+i;
                    expected = 8'((global_offset%256) ^ (global_offset/256) ^ 'ha5);
                    value = memory.read_completion(base+offset, count, i);
                    if (value !== expected) $fatal(1, "HOST_UNIT_ALIAS_OR_DATA");
                    memory.write_byte(base+64'h200000000+offset+i, value);
                end
                offset += count;
            end
        end
        memory.verify_complete();
        $display("NTRPU_HOST_MEMORY_UNIT_PASS payload=1536 descriptors=4 high_regions=4");
        $finish;
    end
endmodule
