`timescale 1ns / 1ps
module tb_ntrpu_pcie_metadata_effect_binary;
    reg clk=0,cold_rst_n=0,valid_in=0;
    reg [56:0] earlier=0,later=0;
    reg [29:0] seed=0;
    wire valid_out;
    wire [56:0] effect_out;
    wire [29:0] metadata_out;
    ntrpu_pcie_metadata_effect_stage dut (.*);
    reg expected_valid;
    reg [56:0] expected_effect;
    reg [29:0] expected_metadata;
    reg [87:0] previous=0;
    string vectors;
    integer fd,fields,rows=0;
    initial begin
        if (!$value$plusargs("VECTORS=%s",vectors)) $fatal(1,"EFFECT_BINARY_NO_VECTORS");
        fd=$fopen(vectors,"r");
        if (!fd) $fatal(1,"EFFECT_BINARY_OPEN");
        // Allow the mapped netlist's glbl startup to finish before reset/checks.
        #200;clk=1;#1;clk=0;#1;
        if ({valid_out,effect_out,metadata_out}!==88'b0) $fatal(1,"EFFECT_BINARY_RESET");
        while (!$feof(fd)) begin
            fields=$fscanf(fd,"%h %h %h %h %h %h %h %h\n",cold_rst_n,valid_in,earlier,later,seed,expected_valid,expected_effect,expected_metadata);
            if (fields!=8) $fatal(1,"EFFECT_BINARY_FORMAT row=%0d",rows);
            #2;
            if ({valid_out,effect_out,metadata_out}!==previous) $fatal(1,"EFFECT_BINARY_PRE_EDGE row=%0d",rows);
            clk=1;#1;
            if ({valid_out,effect_out,metadata_out}!=={expected_valid,expected_effect,expected_metadata})
                $fatal(1,"EFFECT_BINARY_MISMATCH row=%0d actual=%h expected=%h",rows,{valid_out,effect_out,metadata_out},{expected_valid,expected_effect,expected_metadata});
            $display("EFFECT_BINARY_ROW index=%0d value=%h",rows,{valid_out,effect_out,metadata_out});
            previous={valid_out,effect_out,metadata_out};clk=0;#1;rows=rows+1;
        end
        $fclose(fd);
        if (rows!=12805) $fatal(1,"EFFECT_BINARY_COUNT rows=%0d",rows);
        $display("EFFECT_BINARY_PASS rows=12805");
        $finish;
    end
endmodule
