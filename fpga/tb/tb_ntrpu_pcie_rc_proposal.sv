`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_proposal;
    reg clk=0, cold_rst_n=0; always #2 clk=~clk;
    reg [3:0] header_valid=0;
    reg [511:0] header_data=0;
    wire [3:0] header_seen, proposal_valid, length_error, terminal_error;
    wire [383:0] captured_headers;
    wire [51:0] delivered_bytes, proposed_remaining;
    wire [47:0] proposed_address;
    ntrpu_pcie_rc_proposal dut(.*);
    integer fd,rc,row=0,phase;
    reg rst_in;reg [3:0] valid_in;reg [511:0] data_in;
    reg [551:0] previous=0;
    wire [551:0] combined={header_seen,proposal_valid,captured_headers,delivered_bytes,proposed_remaining,proposed_address,length_error,terminal_error};
    string vectors;
    initial begin
        if (!$value$plusargs("VECTORS=%s",vectors)) $fatal(1,"Missing vectors");
        fd=$fopen(vectors,"r");if (!fd) $fatal(1,"Cannot open vectors");
        $display("RC_PROPOSAL_SCOPE lanes=4 latency=1 native=0");
        #200;
        while (!$feof(fd)) begin
            rc=$fscanf(fd,"%d %h %h %h\n",phase,rst_in,valid_in,data_in);
            if (rc!=4) $fatal(1,"Malformed vector row %0d",row);
            @(negedge clk);cold_rst_n=rst_in;header_valid=valid_in;header_data=data_in;
            #1;if (combined!==previous) $fatal(1,"Output changed before capture row %0d",row);
            @(posedge clk);#1;
            $display("RC_PROPOSAL_ROW row=%0d phase=%0d seen=%h valid=%h headers=%h delivered=%h remaining=%h address=%h length=%h terminal=%h",row,phase,header_seen,proposal_valid,captured_headers,delivered_bytes,proposed_remaining,proposed_address,length_error,terminal_error);
            previous=combined;row=row+1;
        end
        $fclose(fd);$display("RC_PROPOSAL_SUITE_PASS rows=%0d",row);$finish;
    end
    initial begin #2000000;$fatal(1,"Proposal watchdog");end
endmodule
