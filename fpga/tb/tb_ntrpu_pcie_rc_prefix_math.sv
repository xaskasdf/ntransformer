`timescale 1ns / 1ps
module tb_ntrpu_pcie_rc_prefix_math;
    ntrpu_pcie_inventory_scheduler dut (
        .clk(1'b0),.cold_rst_n(1'b0),.sequence_in(32'b0),
        .profile_valid(1'b0),.user_reset(1'b0),.link_up(1'b0),
        .requester_id(16'b0),.attribute_mask(2'b0),
        .rq_valid(1'b0),.rc_valid(1'b0),.rc_ready(1'b0),
        .rq_ready(4'b0),.rq_data(512'b0),.rc_data(512'b0),
        .rq_user(137'b0),.rc_user(161'b0)
    );
    reg [68:0] bits;
    reg [12:0] remaining;
    reg [11:0] address;
    reg [43:0] counts;
    reg [3:0] mask;
    reg [24:0] actual,expected;
    reg [31:0] random_state=32'h5a73b129;
    integer i,j,k,m,r,a,context_id,checks=0,unknown_checks=0;
    reg [10:0] word_values [0:5];
    reg [12:0] remaining_values [0:7];
    reg [11:0] address_values [0:3];

    function automatic [31:0] step(input [31:0] x);
        reg [31:0] v;
        begin v=x^(x<<13);v=v^(v>>17);step=v^(v<<5);end
    endfunction

    function automatic [24:0] recurrence(input [12:0] initial_remaining,
        input [11:0] initial_address,input [3:0] enabled,input [43:0] words);
        reg [12:0] stored_remaining;
        reg [12:0] captured_delivered;
        reg [11:0] stored_address;
        reg [10:0] count;
        reg signed [14:0] available,delivered;
        integer n;
        begin
            stored_remaining=initial_remaining;stored_address=initial_address;
            for (n=0;n<4;n=n+1) begin
                if (enabled[n]) begin
                    count=words[n*11+:11];
                    available=count*4-stored_address[1:0];
                    delivered=(available<stored_remaining) ? available : stored_remaining;
                    captured_delivered=delivered;
                    stored_remaining=stored_remaining-captured_delivered;
                    stored_address=stored_address+captured_delivered;
                end
            end
            recurrence={stored_remaining,stored_address};
        end
    endfunction

    task check;
        begin
            {remaining,address,counts}=bits;
            expected=recurrence(remaining,address,mask,counts);
            actual=dut.rc_prefix_state(remaining,address,mask,counts);
            if (actual!==expected) $fatal(1,"RC_PREFIX_MATH_MISMATCH check=%0d mask=%h bits=%h actual=%h expected=%h",checks,mask,bits,actual,expected);
            checks=checks+1;
        end
    endtask

    initial begin
        #1;
        word_values[0]=0;word_values[1]=1;word_values[2]=2;
        word_values[3]=1023;word_values[4]=1024;word_values[5]=2047;
        remaining_values[0]=0;remaining_values[1]=1;remaining_values[2]=2;remaining_values[3]=3;
        remaining_values[4]=4;remaining_values[5]=4095;remaining_values[6]=4096;remaining_values[7]=8191;
        address_values[0]=0;address_values[1]=1;address_values[2]=2;address_values[3]=4095;
        for (r=0;r<8;r=r+1) for (a=0;a<4;a=a+1)
            for (i=0;i<6;i=i+1) for (j=0;j<6;j=j+1) for (m=0;m<16;m=m+1) begin
                bits={remaining_values[r],address_values[a],word_values[j],word_values[i],word_values[j],word_values[i]};
                mask=m;check;
            end
        for (i=0;i<20000;i=i+1) begin
            for (j=0;j<3;j=j+1) begin random_state=step(random_state);bits={bits[36:0],random_state};end
            mask=i;check;
        end
        // All69 seed/count bits, each X and Z, under every known ancestry mask.
        for (context_id=0;context_id<3;context_id=context_id+1)
            for (i=0;i<69;i=i+1) for (m=0;m<16;m=m+1) for (k=0;k<2;k=k+1) begin
                case (context_id)
                    0:bits={13'd0,12'd1,11'd2047,11'd1024,11'd1,11'd0};
                    1:bits={13'd100,12'd4,11'd0,11'd1,11'd1024,11'd2047};
                    2:bits={13'd8191,12'd4095,11'd1,11'd0,11'd2047,11'd1024};
                endcase
                bits[i]=k ? 1'bz : 1'bx;mask=m;check;unknown_checks=unknown_checks+1;
            end
        if (checks!=45056 || unknown_checks!=6624) $fatal(1,"Incomplete prefix arithmetic bank");
        $display("RC_PREFIX_MATH_PASS checks=%0d unknown_checks=%0d",checks,unknown_checks);
        $finish;
    end
endmodule
