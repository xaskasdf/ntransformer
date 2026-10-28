`timescale 1ns / 1ps
module tb_ntrpu_pcie_tag_metadata;
    reg clk=0, cold_rst_n=0;
    always #2 clk=~clk;
    reg [3:0] lookup_valid=0;
    reg [31:0] lookup_tags=0;
    reg [11:0] lookup_cutoffs=0;
    reg [5:0] write_valid=0, write_full=0;
    reg [47:0] write_tags=0;
    reg [179:0] write_data=0;
    wire [3:0] result_valid, result_error;
    wire [119:0] result_data;
    ntrpu_pcie_tag_metadata dut(.*);
    integer input_file, rc, row=0, phase;
    reg rst_in;
    reg [3:0] valid_in;
    reg [31:0] tags_in;
    reg [11:0] cutoffs_in;
    reg [5:0] write_valid_in, write_full_in;
    reg [47:0] write_tags_in;
    reg [179:0] write_data_in;
    reg [127:0] previous_result=0;
    string vectors;
    initial begin
        if (!$value$plusargs("VECTORS=%s",vectors)) $fatal(1,"Missing vectors");
        input_file=$fopen(vectors,"r");
        if (!input_file) $fatal(1,"Cannot open vectors");
        $display("META_SCOPE tags=256 width=30 lookups=4 updates=6 latency=1 native=0");
        // Also usable with a separately reviewed UNISIM netlist (GSR startup).
        #200;
        while (!$feof(input_file)) begin
            rc=$fscanf(input_file,"%d %h %h %h %h %h %h %h %h\n",
                phase,rst_in,valid_in,tags_in,cutoffs_in,
                write_valid_in,write_full_in,write_tags_in,write_data_in);
            if (rc!=9) $fatal(1,"Malformed vector row %0d rc=%0d",row,rc);
            @(negedge clk);
            cold_rst_n=rst_in; lookup_valid=valid_in; lookup_tags=tags_in;
            lookup_cutoffs=cutoffs_in; write_valid=write_valid_in;
            write_full=write_full_in; write_tags=write_tags_in; write_data=write_data_in;
            #1;
            if ({result_valid,result_error,result_data}!==previous_result)
                $fatal(1,"Result changed before capture row %0d",row);
            @(posedge clk); #1;
            $display("META_ROW row=%0d phase=%0d valid=%h error=%h data=%h",
                row,phase,result_valid,result_error,result_data);
            previous_result={result_valid,result_error,result_data};
            row=row+1;
        end
        $fclose(input_file);
        $display("META_SUITE_PASS rows=%0d",row);
        $finish;
    end
    initial begin #2000000; $fatal(1,"Metadata bank watchdog"); end
endmodule
