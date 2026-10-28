`timescale 1ns / 1ps
module tb_ntrpu_pcie_byte_counter_retime;
    reg clk=0,cold_rst_n=0,enable=1;
    reg signed [15:0] added_span=0;
    reg [51:0] removed_spans=0;
    wire signed [22:0] candidate;
    ntrpu_pcie_inventory_byte_accumulator dut (.*);
    reg signed [22:0] expected=0;
    reg [14:0] reference_removed;
    reg [67:0] bits;
    reg [31:0] random_state=32'h5e17a93b;
    integer i,j,checks=0,injected_unknown=0,negative_results=0,above_low_range=0;

    function automatic [31:0] step(input [31:0] x);
        reg [31:0] v;
        begin
            v=x^(x<<13); v=v^(v>>17); step=v^(v<<5);
        end
    endfunction

    task sample;
        begin
            #2;
            reference_removed=({2'b0,removed_spans[12:0]}+{2'b0,removed_spans[25:13]})+
                              ({2'b0,removed_spans[38:26]}+{2'b0,removed_spans[51:39]});
            if (!cold_rst_n) expected=0;
            else if (enable) expected=$signed({2'b0,expected[20:0]})+$signed(added_span)-$signed({1'b0,reference_removed});
            clk=1;
            #1;
            if (candidate!==expected) $fatal(1,"BYTE_RETIME_MISMATCH check=%0d actual=%h expected=%h",checks,candidate,expected);
            if (candidate<0) negative_results=negative_results+1;
            if (candidate>2097151) above_low_range=above_low_range+1;
            checks=checks+1;
            #2;clk=0;
        end
    endtask

    initial begin
        sample;
        for (i=0;i<20000;i=i+1) begin
            for (j=0;j<3;j=j+1) begin
                random_state=step(random_state); bits={bits[35:0],random_state};
            end
            {added_span,removed_spans}=bits;
            cold_rst_n=(i%257)!=0; enable=(i%17)!=0;
            sample;
        end
        cold_rst_n=0;enable=1;sample;
        cold_rst_n=1;added_span=19;removed_spans=0;sample;
        enable=0;added_span='x;removed_spans='x;sample;
        enable=1;added_span=0;removed_spans=0;sample;
        for (i=0;i<68;i=i+1) begin
            cold_rst_n=0;sample;
            cold_rst_n=1;bits=68'h123456789abcdef01;bits[i]=1'bx;
            {added_span,removed_spans}=bits;sample;injected_unknown=injected_unknown+1;
            added_span=0;removed_spans=0;sample;
            cold_rst_n=0;sample;
            cold_rst_n=1;bits=68'h123456789abcdef01;bits[i]=1'bz;
            {added_span,removed_spans}=bits;sample;injected_unknown=injected_unknown+1;
            added_span=0;removed_spans=0;sample;
        end
        cold_rst_n=0;sample;
        if (checks!=20414 || injected_unknown!=136 || negative_results==0 || above_low_range==0)
            $fatal(1,"BYTE_RETIME_COVERAGE checks=%0d unknown=%0d negative=%0d overflow=%0d",checks,injected_unknown,negative_results,above_low_range);
        $display("BYTE_RETIME_PASS checks=%0d injected_unknown=%0d",checks,injected_unknown);
        $display("BYTE_RETIME_RANGE negative=%0d overflow=%0d",negative_results,above_low_range);
        $finish;
    end
endmodule
