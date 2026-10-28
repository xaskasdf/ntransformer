`timescale 1ns / 1ps
// Diagnostic branches with real handshakes, never forced internal state.
module tb_ntrpu_retained_channel_faults;
    reg clk=0;
    always #2.5 clk=~clk;
    reg cold=0, boundary=0, loss=0, sv=0, mr=0;
    reg [16:0] sd=0;
    wire [1:0] sr,mv,invalid,fenced;
    wire [16:0] md[2];
    wire [1:0] pending[2],accepted[2],delivered[2],abandoned[2],fault[2];
    integer scenario=-1, direction=0;
    for(genvar d=0;d<2;d=d+1) begin:channel
        ntrpu_retained_channel #(.WIDTH(17),.COUNT_WIDTH(2),.TO_BOUNDARY(d)) dut(
            .clk(clk),.cold_rst_n(cold),.boundary_rst_n(boundary),.loss(loss),
            .s_valid(sv),.s_ready(sr[d]),.s_data(sd),.m_valid(mv[d]),
            .m_ready(mr),.m_data(md[d]),.invalid(invalid[d]),.fenced(fenced[d]),
            .pending(pending[d]),.accepted(accepted[d]),.delivered(delivered[d]),
            .abandoned(abandoned[d]),.fault(fault[d]));
    end
    always @(posedge clk) if(scenario>=0) begin
        $display("FAULT_PRE case=%0d dir=%0d t=%0.0f cold=%b bound=%b loss=%b sv=%b sr=%b sd=%h mv=%b mr=%b md=%h invalid=%b fence=%b pending=%0d accepted=%0d delivered=%0d abandoned=%0d fault=%0d",
            scenario,direction,$realtime*1000,cold,boundary,loss,sv,sr[direction],sd,mv[direction],mr,md[direction],invalid[direction],fenced[direction],pending[direction],accepted[direction],delivered[direction],abandoned[direction],fault[direction]);
        #0.001;
        $display("FAULT_POST case=%0d dir=%0d t=%0.0f sr=%b mv=%b md=%h invalid=%b fence=%b pending=%0d accepted=%0d delivered=%0d abandoned=%0d fault=%0d",
            scenario,direction,$realtime*1000,sr[direction],mv[direction],md[direction],invalid[direction],fenced[direction],pending[direction],accepted[direction],delivered[direction],abandoned[direction],fault[direction]);
    end
    task automatic step(input bit c,b,l,v,r,input reg[16:0] data=0);
        @(negedge clk);cold=c;boundary=b;loss=l;sv=v;mr=r;sd=data;
        @(posedge clk);#0.01;
    endtask
    task automatic start_case(input integer n,d);
        @(negedge clk);scenario=n;direction=d;cold=0;boundary=0;loss=0;sv=0;mr=0;sd=0;
        repeat(3) step(0,0,0,0,0);
        repeat(3) step(1,0,0,0,0);
        if(fault[d]!=0 || invalid[d]) $fatal(1,"FAULT_INITIAL_RESET case=%0d",n);
        step(1,1,0,0,0);
    endtask
    task automatic finish_case(input integer a,d,b,f);
        repeat(3) step(1,1,0,0,0);
        if(accepted[direction]!=a || delivered[direction]!=d || abandoned[direction]!=b ||
            fault[direction]!=f || pending[direction]!=0 || !invalid[direction])
            $fatal(1,"FAULT_FINAL case=%0d",scenario);
        $display("FAULT_CASE_PASS case=%0d accepted=%0d delivered=%0d abandoned=%0d fault=%0d",scenario,a,d,b,f);
        @(negedge clk);scenario=-1;cold=0;boundary=0;sv=0;mr=0;loss=0;
    endtask
    initial begin
        #200;
        // Illegal runtime reset must diagnose without erasing retained requests.
        start_case(0,0);
        step(1,1,0,1,0,17'h12345);step(1,1,0,1,0,17'h0abcd);
        repeat(3) step(1,0,0,0,0);
        repeat(3) step(1,1,0,1,0,17'h1ffff); // blocked new admission
        repeat(3) step(1,1,0,0,1);
        finish_case(2,2,0,1);
        // Same contract violation diagnoses then retires the facing responses.
        start_case(1,1);
        step(1,1,0,1,0,17'h12345);step(1,1,0,1,0,17'h0abcd);
        repeat(3) step(1,0,0,0,0);
        finish_case(2,0,2,1);
        // Four actual events overflow 2-bit counters; fourth still drains.
        for(integer d=0;d<2;d=d+1) begin
            start_case(2+d,d);
            for(integer n=0;n<4;n=n+1) begin
                step(1,1,0,1,1,17'h10010+n);
                step(1,1,0,0,1);
            end
            if(d==0) repeat(3) step(1,1,0,1,1,17'h1ffff);
            repeat(3) step(1,0,0,0,0);
            finish_case(3,3,0,2);
        end
        // Late completions after reset: four consumed, both counters saturate.
        start_case(4,1);
        step(1,1,1,0,0);step(1,0,0,0,0);
        for(integer n=0;n<4;n=n+1) step(1,1,0,1,0,17'h10100+n);
        finish_case(3,0,3,2);
        // One-cycle loss and repeated reset/reopen never restore admission.
        start_case(5,1);
        step(1,1,0,1,0,17'h12345);step(1,1,0,1,0,17'h0abcd);
        step(1,1,1,0,0);repeat(3) step(1,1,0,0,0);
        repeat(2) step(1,0,0,0,0);
        step(1,1,0,1,0,17'h055aa);
        repeat(2) step(1,0,0,0,0);
        finish_case(3,0,3,0);
        // An offered response remains valid/data-stable throughout live loss.
        start_case(6,1);
        step(1,1,0,1,0,17'h12345);
        repeat(8) step(1,1,1,0,0);
        step(1,1,0,0,1); // legitimate consumer receipt even after invalidation
        repeat(2) step(1,0,0,0,0);
        finish_case(1,1,0,0);
        $display("FAULT_SUITE_PASS cases=7");$finish;
    end
    initial begin #20000;$fatal(1,"FAULT_WATCHDOG");end
endmodule
