// Bounded simulation host memory. Decode ALL 64 address bits before indexing.
// Four 32-byte descriptors, two 768-byte sources and two 768-byte destinations.
`timescale 1ps/1ps
module ntrpu_host_memory;
    reg [7:0] bytes_store [0:8191];
    bit initialized_byte [0:8191], read_seen [0:8191];
    bit initialized = 0;
    integer unique_read_bytes = 0, written_bytes = 0;

    function automatic [63:0] region_base(input integer region);
        case (region)
            0: region_base = 64'h0000000100000100;
            1: region_base = 64'h0000000200000100;
            2: region_base = 64'h0000000300000300;
            3: region_base = 64'h0000000400000300;
            4: region_base = 64'h0000000100000fc0;
            5: region_base = 64'h0000000200000fc0;
            6: region_base = 64'h0000000300000fc0;
            7: region_base = 64'h0000000400000fc0;
            default: $fatal(1, "HOST_BAD_REGION");
        endcase
    endfunction

    function automatic integer locate(input logic [63:0] address);
        integer region, length;
        begin
            if ($isunknown(address)) $fatal(1, "HOST_UNKNOWN_ADDRESS");
            locate = -1;
            for (region = 0; region < 8; region++) begin
                length = region < 4 ? 32 : 768;
                if (address >= region_base(region) && address-region_base(region) < length)
                    locate = region*1024 + int'(address-region_base(region));
            end
            if (locate < 0) $fatal(1, "HOST_UNMAPPED address=%h", address);
        end
    endfunction

    task initialize;
        integer i, region, index;
        reg [255:0] descriptor;
        reg [63:0] source, destination, next_descriptor;
        begin
            if (initialized) $fatal(1, "HOST_DOUBLE_INITIALIZATION");
            for (i = 0; i < 8192; i++) begin
                initialized_byte[i] = 0; read_seen[i] = 0;
            end
            for (region = 0; region < 4; region++) begin
                source = region < 2 ? region_base(4+region) : 64'((region-2)*768);
                destination = region < 2 ? 64'(region*768) : region_base(4+region);
                next_descriptor = region % 2 == 0 ? region_base(region+1) : 0;
                descriptor = {next_descriptor, destination, source, 32'd768,
                              (region % 2 == 0 ? 32'had4b0010 : 32'had4b0013)};
                for (i = 0; i < 32; i++) begin
                    index = region*1024+i;
                    bytes_store[index] = descriptor[i*8 +: 8]; initialized_byte[index] = 1;
                end
            end
            for (i = 0; i < 1536; i++) begin
                index = (4+i/768)*1024 + i%768;
                bytes_store[index] = 8'(i ^ (i >> 8) ^ 'ha5); initialized_byte[index] = 1;
            end
            initialized = 1;
            $display("NTRPU_HOST_MEMORY_INITIALIZED regions=8 descriptors=4 payload=1536 address_bits=64");
        end
    endtask

    function automatic [7:0] read_completion(input logic [63:0] address,
                                            input integer count, offset);
        integer first_index, last_index, index;
        begin
            if (!initialized) $fatal(1, "HOST_NOT_INITIALIZED");
            if (count <= 0 || count > 256 || offset < 0 || address[1:0] != 0 ||
                int'(address[11:0])+count > 4096 || address+count-1 < address)
                $fatal(1, "HOST_BAD_READ_REQUEST address=%h bytes=%0d", address, count);
            first_index = locate(address);
            last_index = locate(address+count-1);
            if (first_index/1024 >= 6 || first_index/1024 != last_index/1024)
                $fatal(1, "HOST_READ_PERMISSION");
            // The AMD CC packer evaluates unused final-beat lanes too.
            read_completion = 0;
            if (offset < count) begin
                index = locate(address+offset);
                if (!initialized_byte[index]) $fatal(1, "HOST_UNINITIALIZED");
                read_completion = bytes_store[index];
                if (!read_seen[index]) begin read_seen[index] = 1; unique_read_bytes++; end
            end
        end
    endfunction

    task write_byte(input logic [63:0] address, input logic [7:0] value);
        integer index;
        begin
            if (!initialized) $fatal(1, "HOST_NOT_INITIALIZED");
            index = locate(address);
            if (index/1024 < 6) $fatal(1, "HOST_WRITE_PERMISSION");
            if ($isunknown(value)) $fatal(1, "HOST_UNKNOWN_DATA");
            if (initialized_byte[index]) $fatal(1, "HOST_DUPLICATE_WRITE");
            bytes_store[index] = value;
            initialized_byte[index] = 1; written_bytes++;
        end
    endtask

    task verify_complete;
        integer i, index;
        reg [7:0] expected;
        begin
            if (unique_read_bytes != 1664 || written_bytes != 1536)
                $fatal(1, "HOST_INCOMPLETE reads=%0d writes=%0d", unique_read_bytes, written_bytes);
            for (i = 0; i < 1536; i++) begin
                index = (6+i/768)*1024 + i%768;
                expected = 8'((i % 256) ^ (i / 256) ^ 'ha5);
                if (!initialized_byte[index] || bytes_store[index] !== expected)
                    $fatal(1, "HOST_FINAL_DATA offset=%0d", i);
            end
            $display("NTRPU_HOST_MEMORY_COMPLETE read_bytes=%0d written_bytes=%0d regions=8 descriptors=4",
                     unique_read_bytes, written_bytes);
        end
    endtask
endmodule
