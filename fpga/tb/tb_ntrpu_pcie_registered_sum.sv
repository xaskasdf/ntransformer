`timescale 1ns / 1ps
module tb_ntrpu_pcie_registered_sum;
    reg clk=0,cold_rst_n=0,enable=1;
    reg signed [15:0] added_span=0;
    reg [14:0] removed_sum=0;
    wire signed [22:0] candidate;
    ntrpu_pcie_inventory_sum_accumulator dut (.*);
    reg signed [22:0] expected=0;
    reg signed [31:0] reference_wide;
    reg [30:0] bits;
    reg [31:0] random_state=32'h5e17a93b;
    integer i,checks=0,injected_unknown=0,negative_results=0,above_protocol=0,above_low_range=0;

    function automatic [31:0] step(input [31:0] x);
        reg [31:0] v;
        begin v=x^(x<<13);v=v^(v>>17);step=v^(v<<5);end
    endfunction

    task sample;
        begin
            #2;
            if (candidate!==expected) $fatal(1,"REGISTERED_SUM_EARLY_CHANGE check=%0d actual=%h expected=%h",checks,candidate,expected);
            if (!cold_rst_n) expected=0;
            else if (enable) begin
                reference_wide=$signed({11'b0,expected[20:0]})+$signed(added_span)-$signed({17'b0,removed_sum});
                expected=reference_wide[22:0];
            end
            clk=1;
            #1;
            if (candidate!==expected) $fatal(1,"REGISTERED_SUM_MISMATCH check=%0d actual=%h expected=%h",checks,candidate,expected);
            if (candidate<0) negative_results=negative_results+1;
            if (candidate>1052672) above_protocol=above_protocol+1;
            if (candidate>2097151) above_low_range=above_low_range+1;
            checks=checks+1;
            #2;clk=0;
        end
    endtask

    initial begin
        $display("REGISTERED_SUM_MODE mutant=0");
        sample;
        cold_rst_n=1;added_span=32;sample;
        added_span=0;removed_sum=7;sample;
        enable=0;added_span='x;removed_sum='x;sample;
        enable=1;added_span=0;removed_sum=0;sample;
        for (i=0;i<70;i=i+1) begin added_span=32767;removed_sum=0;sample;end
        cold_rst_n=0;sample;
        cold_rst_n=1;added_span=-32768;removed_sum=32767;sample;
        added_span=0;removed_sum=0;sample;
        for (i=0;i<20000;i=i+1) begin
            random_state=step(random_state);
            {added_span,removed_sum}=random_state[30:0];
            cold_rst_n=(i%257)!=0;enable=(i%17)!=0;
            sample;
        end
        enable=1;
        for (i=0;i<31;i=i+1) begin
            cold_rst_n=0;sample;
            cold_rst_n=1;bits=31'h12345678;bits[i]=1'bx;
            {added_span,removed_sum}=bits;sample;injected_unknown=injected_unknown+1;
            added_span=0;removed_sum=0;sample;
            cold_rst_n=0;sample;
            cold_rst_n=1;bits=31'h12345678;bits[i]=1'bz;
            {added_span,removed_sum}=bits;sample;injected_unknown=injected_unknown+1;
            added_span=0;removed_sum=0;sample;
        end
        cold_rst_n=0;sample;
        if (checks!=20265 || injected_unknown!=62 || negative_results==0 || above_protocol==0 || above_low_range==0)
            $fatal(1,"REGISTERED_SUM_COVERAGE checks=%0d unknown=%0d negative=%0d protocol=%0d overflow=%0d",checks,injected_unknown,negative_results,above_protocol,above_low_range);
        $display("REGISTERED_SUM_PASS checks=%0d injected_unknown=%0d",checks,injected_unknown);
        $display("REGISTERED_SUM_RANGE negative=%0d protocol=%0d overflow=%0d",negative_results,above_protocol,above_low_range);
        $finish;
    end
endmodule
