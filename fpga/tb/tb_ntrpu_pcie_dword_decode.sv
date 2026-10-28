`timescale 1ns / 1ps

// Independent scalar reference: compare absolute byte addresses, not the
// RTL's lane-vector expression. Run unchanged against RTL and synth netlist.
module tb_ntrpu_pcie_dword_decode;
    reg lease_valid = 1, awlock = 0, wlast = 1;
    reg [63:0] lease_line = 0, awaddr = 0, wstrb = 0;
    reg [7:0] awlen = 0;
    reg [2:0] awsize = 2;
    reg [1:0] awburst = 1;
    reg [511:0] wdata = 0;
    wire accept;
    wire [3:0] faults;
    wire [63:0] line_address, byte_valid;
    wire [511:0] byte_data;
    integer cases = 0, allowed = 0, rejected = 0;
    integer lease_fail = 0, format_fail = 0, line_fail = 0, mask_fail = 0;
    reg [63:0] bases [0:4];
    ntrpu_pcie_dword_decode dut (.*);

    task automatic check;
        reg [3:0] expected_faults;
        reg [64:0] line_start, lease_start, first_byte, word_end, byte_address;
        reg expected_accept;
        begin
            // 65-bit arithmetic makes the top-of-address-space test exact.
            line_start = ({1'b0, awaddr} / 65'd64) * 65'd64;
            lease_start = {1'b0, lease_line};
            first_byte = {1'b0, awaddr};
            word_end = (first_byte / 65'd4) * 65'd4 + 65'd4;
            expected_faults = 0;
            if (!lease_valid || lease_line % 64 != 0) expected_faults[0] = 1;
            if (awlen != 0 || awsize != 2 || awburst != 1 || awlock || !wlast)
                expected_faults[1] = 1;
            if (line_start != lease_start) expected_faults[2] = 1;
            if (wstrb == 0) expected_faults[3] = 1;
            for (integer j = 0; j < 64; j = j + 1) begin
                byte_address = line_start + j;
                if (wstrb[j] && (byte_address < first_byte || byte_address >= word_end))
                    expected_faults[3] = 1;
            end
            expected_accept = expected_faults == 0;
            #10;
            if (faults !== expected_faults || accept !== expected_accept)
                $fatal(1,"DECODE_DECISION case=%0d addr=%h strb=%h faults=%h expected=%h", cases, awaddr, wstrb, faults, expected_faults);
            if (line_address !== line_start[63:0] || byte_data !== wdata)
                $fatal(1,"DECODE_PAYLOAD case=%0d", cases);
            if (byte_valid !== (expected_accept ? wstrb : 64'b0))
                $fatal(1,"DECODE_BITMAP case=%0d", cases);
            cases = cases + 1;
            if (expected_accept) allowed = allowed + 1; else rejected = rejected + 1;
            if (expected_faults[0]) lease_fail = lease_fail + 1;
            if (expected_faults[1]) format_fail = format_fail + 1;
            if (expected_faults[2]) line_fail = line_fail + 1;
            if (expected_faults[3]) mask_fail = mask_fail + 1;
        end
    endtask

    initial begin
        bases[0]=0; bases[1]=64'h100000000; bases[2]=64'h1000fffc0;
        bases[3]=64'h7fffffffc0; bases[4]=64'hffffffffffffffc0;
        // Every byte address and every DWORD mask, including empty masks
        // and masks below an unaligned address. Vary all data byte lanes.
        for (integer bank=0; bank<5; bank=bank+1) begin
            lease_line=bases[bank];
            for (integer offset=0; offset<64; offset=offset+1) begin
                awaddr=lease_line+offset;
                for (integer mask=0; mask<16; mask=mask+1) begin
                    wstrb={60'b0,4'(mask)} << ((offset/4)*4);
                    for (integer j=0; j<64; j=j+1)
                        wdata[j*8+:8]=8'(bank*41+offset*7+mask*13+j*3);
                    check();
                end
            end
        end
        // Every incorrect single lane for every address; catches accidental
        // broadening of the last unaligned beat past its DWORD boundary.
        lease_line=64'h100000000;
        for (integer offset=0; offset<64; offset=offset+1) begin
            awaddr=lease_line+offset;
            for (integer lane=0; lane<64; lane=lane+1) begin
                wstrb=64'b1 << lane;
                check();
            end
        end
        awaddr=lease_line; wstrb=15;
        for (integer n=0;n<256;n=n+1) begin awlen=8'(n);check();end
        awlen=0;
        for (integer n=0;n<8;n=n+1) begin awsize=3'(n);check();end
        awsize=2;
        for (integer n=0;n<4;n=n+1) begin awburst=2'(n);check();end
        awburst=1;awlock=1;check();awlock=0;wlast=0;check();wlast=1;
        lease_valid=0;check();lease_valid=1;
        // All high address bits participate, not only the low 32 bits.
        for (integer bit_index=6;bit_index<64;bit_index=bit_index+1) begin
            awaddr=lease_line ^ (64'b1 << bit_index);check();
        end
        awaddr=lease_line;
        for (integer n=1;n<64;n=n+1) begin
            lease_line=64'h100000000+n;check();
        end
        lease_line=64'h100000000;wstrb=64'hffffffffffffffff;check();
        if (cases!=9609 || allowed!=2243 || rejected!=7366 ||
            lease_fail!=64 || format_fail!=267 || line_fail!=121 || mask_fail!=6977)
            $fatal(1,"DECODE_COVERAGE cases=%0d allowed=%0d rejected=%0d lease=%0d format=%0d line=%0d mask=%0d",
                   cases,allowed,rejected,lease_fail,format_fail,line_fail,mask_fail);
        $display("NTRPU_DWORD_DECODE_PASS cases=%0d allowed=%0d rejected=%0d lease=%0d format=%0d line=%0d mask=%0d",
                 cases,allowed,rejected,lease_fail,format_fail,line_fail,mask_fail);
        $finish;
    end
    initial begin #200000; $fatal(1,"DECODE_TIMEOUT"); end
endmodule
