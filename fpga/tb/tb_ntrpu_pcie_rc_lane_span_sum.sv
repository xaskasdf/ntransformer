`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_lane_span_sum;
    ntrpu_pcie_inventory_scheduler dut (
        .clk(1'b0),.cold_rst_n(1'b0),.sequence_in(32'b0),
        .profile_valid(1'b0),.user_reset(1'b0),.link_up(1'b0),
        .requester_id(16'b0),.attribute_mask(2'b0),
        .rq_valid(1'b0),.rc_valid(1'b0),.rc_ready(1'b0),
        .rq_ready(4'b0),.rq_data(512'b0),.rc_data(512'b0),
        .rq_user(137'b0),.rc_user(161'b0)
    );
    reg [207:0] values;
    reg [14:0] actual,expected;
    reg [31:0] random_state=32'h7931a5c7;
    integer i,lane,bit_index,polarity,context_id,mask,checks=0,unknown_checks=0;

    function automatic [31:0] step(input [31:0] x);
        reg [31:0] v;
        begin v=x^(x<<13);v=v^(v>>17);step=v^(v<<5);end
    endfunction

    // Independent sequential accumulation at wider precision, truncate once.
    function automatic [14:0] reference_sum(input [207:0] terms);
        reg [17:0] total;
        integer n;
        begin
            total=0;
            for(n=0;n<16;n=n+1) total=total+{5'b0,terms[n*13+:13]};
            reference_sum=total[14:0];
        end
    endfunction

    task check;
        begin
            actual=dut.rc_lane_span_sum(values);
            expected=reference_sum(values);
            if(actual!==expected)
                $fatal(1,"RC_LANE_SPAN_SUM_MISMATCH check=%0d values=%h actual=%h expected=%h",checks,values,actual,expected);
            checks=checks+1;
        end
    endtask

    initial begin
        #1;
        // Every lane subset, at maxima: covers all carries and modular sums.
        // More than four active terms are helper-only tests, not legal RC beats.
        for(mask=0;mask<65536;mask=mask+1) begin
            values=0;
            for(lane=0;lane<16;lane=lane+1)
                if((mask>>lane)&1) values[lane*13+:13]=13'h1fff;
            check;
        end
        for(i=0;i<4096;i=i+1) begin
            for(lane=0;lane<16;lane=lane+1) begin
                random_state=step(random_state);
                values[lane*13+:13]=random_state[12:0];
            end
            check;
        end
        // Each input bit independently X/Z, against zero and maximal contexts.
        for(context_id=0;context_id<2;context_id=context_id+1)
            for(bit_index=0;bit_index<208;bit_index=bit_index+1)
                for(polarity=0;polarity<2;polarity=polarity+1) begin
                    values=context_id ? {208{1'b1}} : 208'b0;
                    values[bit_index]=polarity ? 1'bz : 1'bx;
                    check;unknown_checks=unknown_checks+1;
                end
        if(checks!=70464 || unknown_checks!=832) $fatal(1,"Incomplete RC lane sum bank");
        $display("RC_LANE_SPAN_SUM_PASS checks=%0d unknown_checks=%0d",checks,unknown_checks);
        $finish;
    end
endmodule
