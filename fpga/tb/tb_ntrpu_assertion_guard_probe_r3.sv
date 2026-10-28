`timescale 1ns/1ps
// Simulator diagnostic only. No hardware model or arithmetic qualification.
module tb_ntrpu_assertion_guard_probe_r3;
    logic clk=0,rst_n=0,ce=0,we=0;
    logic [10:0] address=0;
    logic [31:0] data='x,reference_data[0:8191];
    int cycles=0,mode=0,sample_now=0,calls=0;
    always begin #2.5; if(!clk)cycles++;clk=~clk;end
`define GUARD_REQUIRE(condition, assertion_reason) begin \
        calls++; \
        $display("GUARD_CALL cycle=%0d mode=%0d ce=%b we=%b ok=%b why=%s",cycles,mode,ce,we,bit'(condition),assertion_reason); \
        if(!bit'(condition))$fatal(1,"GUARD_NEGATIVE %s",assertion_reason); \
end
    task automatic require_task(input bit ok,input string why);
        calls++;
        $display("GUARD_CALL cycle=%0d mode=%0d ce=%b we=%b ok=%b why=%s",cycles,mode,ce,we,ok,why);
        if(!ok)$fatal(1,"GUARD_NEGATIVE %s",why);
    endtask
    function automatic int index_value(input int index);
        `GUARD_REQUIRE(index>=0 && index<8192,"Index bounds");return index;
    endfunction
    always @(posedge clk)if(rst_n)begin
        if(mode<3)begin
            if(ce && we)
                `GUARD_REQUIRE(data===reference_data[sample_now*2048+int'(address)],"Actual post residual arithmetic");
        end else begin
            if(ce && we)
                require_task(data===reference_data[sample_now*2048+int'(address)],"Actual post residual arithmetic");
        end
    end
    initial begin
        if($value$plusargs("MODE=%d",mode))begin end
        for(int k=0;k<8192;k++)reference_data[k]=32'h3f800000;
        address=11'(index_value(0));calls=0;
        repeat(5)@(posedge clk);@(negedge clk);rst_n=1;
        repeat(3)@(negedge clk);
        if(calls!=0)$fatal(1,"GUARD_UNEXPECTED_CALL idle calls=%0d",calls);
        if(mode%3!=0)begin
            ce=1;we=1;data=mode%3==1 ? 32'h3f800000:32'h40000000;
            @(negedge clk);ce=0;we=0;
        end
        repeat(2)@(negedge clk);
        if(calls!=(mode%3==1 ? 1:0))$fatal(1,"GUARD_CALL_COUNT calls=%0d",calls);
        $display("GUARD_PASS mode=%0d calls=%0d",mode,calls);$finish;
    end
    initial begin #1000;$fatal(1,"GUARD_TIMEOUT");end
`undef GUARD_REQUIRE
endmodule
