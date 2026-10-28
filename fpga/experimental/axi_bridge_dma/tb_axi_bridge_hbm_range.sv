`timescale 1ns/1ps
`default_nettype none
module tb_axi_bridge_hbm_range;
    reg [63:0] address=0;
    reg [12:0] bytes=0;
    integer known_count=0,unknown_count=0;
    reg [31:0] rng=32'h31415926;
    // Combinational admission predicate, no state transitions in this bench.
    ntrpu_axi_bridge_hbm_source candidate(.clk(1'b0),.cold_rst_n(1'b0),
        .src_req_address(address),.src_req_bytes(bytes));
    reference_hbm_source reference_dut(.clk(1'b0),.cold_rst_n(1'b0),
        .src_req_address(address),.src_req_bytes(bytes));
    task check(input bit unknowns);
        #1;
        if(candidate.bad_request !== reference_dut.bad_request)
            $fatal(1,"HR MISMATCH address=%h bytes=%h old=%b new=%b",address,bytes,reference_dut.bad_request,candidate.bad_request);
        if(unknowns)unknown_count++;else known_count++;
    endtask
    function automatic [31:0] random_word;
        rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);return rng;
    endfunction
    reg [63:0] bases[0:11];
    reg [12:0] lengths[0:9];
    reg [63:0] saved_address;
    reg [12:0] saved_bytes;
    integer p,n,b,z,k;
    initial begin
        bases[0]=0;bases[1]=64'h1000;bases[2]=64'h100000000;
        bases[3]=64'h200000000;bases[4]=64'h3fffff000;bases[5]=64'h3ffffffc0;
        bases[6]=64'h400000000;bases[7]=64'h400000040;
        bases[8]=64'h8000000000000000;bases[9]=64'hfffffffffffff000;
        bases[10]=64'hffffffffffffffc0;bases[11]=64'hffffffffffffffff;
        lengths[0]=0;lengths[1]=1;lengths[2]=63;lengths[3]=64;lengths[4]=128;
        lengths[5]=4032;lengths[6]=4096;lengths[7]=4097;lengths[8]=8128;lengths[9]=8191;
        for(p=0;p<12;p++)for(n=0;n<8192;n++)begin address=bases[p];bytes=13'(n);check(0);end
        for(p=0;p<4096;p++)for(n=1;n<=64;n++)begin address=64'h3fffff000+64'(p);bytes=13'(n*64);check(0);end
        for(k=0;k<20000;k++)begin address={random_word(),random_word()};bytes=13'(random_word());check(0);end
        for(p=0;p<12;p++)for(n=0;n<10;n++)for(b=0;b<77;b++)for(z=0;z<2;z++)begin
            address=bases[p];bytes=lengths[n];
            if(b<64)address[b]=z?1'bz:1'bx;else bytes[b-64]=z?1'bz:1'bx;
            check(1);
        end
        for(k=0;k<10000;k++)begin
            address={random_word(),random_word()};bytes=13'(random_word());
            for(b=0;b<6;b++)begin
                p=integer'(random_word()%77);
                if(p<64)address[p]=(b&1)?1'bz:1'bx;else bytes[p-64]=(b&1)?1'bz:1'bx;
            end
            check(1);
        end
        $display("HR PASS known=%0d unknown=%0d",known_count,unknown_count);$finish;
    end
endmodule
`default_nettype wire
