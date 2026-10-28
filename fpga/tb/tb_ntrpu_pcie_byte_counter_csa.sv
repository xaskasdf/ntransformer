`timescale 1ns / 1ps
// Supplement to the complete protocol bank. Call the actual candidate function.
module tb_ntrpu_pcie_byte_counter_csa;
    ntrpu_pcie_inventory_scheduler dut (
        .clk(1'b0), .cold_rst_n(1'b0), .sequence_in(32'b0),
        .profile_valid(1'b0), .user_reset(1'b0), .link_up(1'b0),
        .requester_id(16'b0), .attribute_mask(2'b0),
        .rq_valid(1'b0), .rc_valid(1'b0), .rc_ready(1'b0),
        .rq_ready(4'b0), .rq_data(512'b0), .rc_data(512'b0),
        .rq_user(137'b0), .rc_user(161'b0)
    );
    reg [20:0] current_bytes;
    reg signed [15:0] rq_sum;
    reg [12:0] removed0, removed1, removed2, removed3;
    reg [14:0] reference_rc_sum;
    reg signed [22:0] expected, actual;
    reg [88:0] bits;
    reg [31:0] random_state=32'h1a2b3c4d;
    integer i,j,checks=0,unknown_checks=0;

    function automatic [31:0] step(input [31:0] x);
        reg [31:0] v;
        begin
            v=x^(x<<13); v=v^(v>>17); step=v^(v<<5);
        end
    endfunction

    task check;
        begin
            {current_bytes,rq_sum,removed0,removed1,removed2,removed3}=bits;
            reference_rc_sum=({2'b0,removed0}+{2'b0,removed1})+
                             ({2'b0,removed2}+{2'b0,removed3});
            expected=$signed({2'b0,current_bytes})+$signed(rq_sum)-$signed({1'b0,reference_rc_sum});
            actual=dut.byte_counter_csa(current_bytes,rq_sum,removed0,removed1,removed2,removed3);
            if (actual!==expected) $fatal(1,"BYTE_CSA_MISMATCH check=%0d bits=%h actual=%h expected=%h",checks,bits,actual,expected);
            checks=checks+1;
        end
    endtask

    initial begin
        #1;
        // Full stored counter/operand ranges, not just legal live traffic.
        for (i=0;i<20000;i=i+1) begin
            for (j=0;j<3;j=j+1) begin
                random_state=step(random_state);
                bits={bits[56:0],random_state};
            end
            check;
        end
        for (i=0;i<64;i=i+1) begin
            bits={ {21{i[0]}}, {16{i[1]}}, {13{i[2]}},
                   {13{i[3]}}, {13{i[4]}}, {13{i[5]}} };
            check;
        end
        // Every operand bit exercises X and Z against the original arithmetic.
        for (i=0;i<89;i=i+1) begin
            bits=89'h123456789abcdef01234567; bits[i]=1'bx;
            check; unknown_checks=unknown_checks+1;
            bits=89'h123456789abcdef01234567; bits[i]=1'bz;
            check; unknown_checks=unknown_checks+1;
        end
        $display("BYTE_CSA_PASS checks=%0d unknown_checks=%0d",checks,unknown_checks);
        $finish;
    end
endmodule
