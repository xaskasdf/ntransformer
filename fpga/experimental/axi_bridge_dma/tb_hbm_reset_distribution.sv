`timescale 1ns/1ps
`default_nettype none
module reset_distribution_case #(
    parameter integer ID=0,
    parameter realtime HALF=2.5,MEM_HALF=1.111,PHASE=0.13
)(output reg done=0);
    reg clk=0,mem_clk=0,source_on=1,memory_on=1,cold_n=1;
    reg [1:0] calibration=0,adapter_fault=0;
    wire src_n,mem_n,started,fault,ref_src_n,ref_mem_n,ref_started,ref_fault;
    integer checks=0,groups=0,edges=0,popped=0;
    reg s_valid=0,m_ready=0;
    reg [518:0] s_data=0;
    wire [1:0] s_ready,m_valid,empty;
    wire [264:0] response_data;
    wire [518:0] return_data;
    ntrpu_hbm_startup_distributed dut(.clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_n),
        .init_complete_async(calibration),.adapter_fault_mem(adapter_fault),
        .src_rst_n(src_n),.mem_rst_n(mem_n),.started(started),.fault(fault));
    ntrpu_hbm_startup_reference reference(.clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_n),
        .init_complete_async(calibration),.adapter_fault_mem(adapter_fault),
        .src_rst_n(ref_src_n),.mem_rst_n(ref_mem_n),.started(ref_started),.fault(ref_fault));
    // Actual frozen consumer queues, not a behavioral replacement for them.
    ntrpu_hbm_read_response_queue response_q(.clk(mem_clk),.rst_n(mem_n),
        .s_valid(s_valid),.s_ready(s_ready[0]),.s_data(s_data[264:0]),
        .m_valid(m_valid[0]),.m_ready(m_ready),.m_data(response_data),.empty(empty[0]));
    ntrpu_hbm_read_return_cut return_q(.clk(mem_clk),.rst_n(mem_n),
        .s_valid(s_valid),.s_ready(s_ready[1]),.s_data(s_data),
        .m_valid(m_valid[1]),.m_ready(m_ready),.m_data(return_data),.empty(empty[1]));
    always #(HALF) if(source_on)clk=~clk;else clk=0;
    always #(MEM_HALF) if(memory_on)mem_clk=~mem_clk;else mem_clk=0;
    task check(input bit value,input string message);
        checks++;if(!value)$fatal(1,"RD FAIL case=%0d %s",ID,message);
    endtask
    function automatic [518:0] payload(input integer tag);
        payload={7'h53,{16{32'h6ac9e731 ^ (tag*32'h01010101) ^ ID}}};
    endfunction
    task ticks(input integer count);
        repeat(count)begin @(posedge clk);#0.02;end
    endtask
    task mticks(input integer count);
        repeat(count)begin @(posedge mem_clk);#0.02;end
    endtask
    always @(posedge mem_clk or negedge cold_n)begin
        if(!cold_n)edges=0;else edges++;
        #0.003;
        if(cold_n)begin
            check(ref_mem_n===(edges>=3),"reference release at third edge");
            check(mem_n===(edges>=4),"distribution release at fourth edge");
        end
    end
    always @(negedge cold_n)begin
        #0.004;
        check(src_n===0 && mem_n===0,"asynchronous assertion without clocks");
        check(s_ready===0 && m_valid===0,"cold reset masks queue handshakes");
    end
    always @(negedge mem_n)begin
        if(cold_n && edges>=4)begin
            #0.001;check(mem_n===1,"fault must preserve reset epoch");
        end
    end
    always @(posedge clk)begin
        #0.006;check(src_n===ref_src_n,"source release unchanged");
        if(started)check(mem_n && ref_started,"startup waits for distributed release");
    end
    task cold;
        @(negedge clk);#(PHASE);cold_n=0;
        ticks(4);mticks(4);
        @(negedge mem_clk);#0.11;cold_n=1;groups++;
    endtask
    task acquire;
        integer cycles;
        // With the source clock stopped, the synchronous status register still
        // holds the previous epoch until its first new edge clears it.
        cycles=0;
        while((started!==1 || src_n!==1 || mem_n!==1) && cycles<60)begin ticks(1);cycles++;end
        check(started && !fault && src_n && mem_n,"qualified startup");groups++;
    endtask
    task push(input integer tag);
        @(negedge mem_clk);s_data=payload(tag);s_valid=1;
        #0.01;check(s_ready===3,"both queues accept");
        mticks(1);@(negedge mem_clk);s_valid=0;
    endtask
    task stalled(input integer tag,input integer count);
        reg [518:0] expected;
        expected=payload(tag);
        repeat(count)begin
            mticks(1);
            check(mem_n && src_n,"fault must preserve reset epoch");
            check(m_valid===3 && return_data===expected && response_data===expected[264:0],
                  "stalled payload retained across fault");
        end
        groups++;
    endtask
    task drain(input integer first_tag);
        reg [518:0] expected;
        @(negedge mem_clk);m_ready=1;
        for(integer tag=first_tag;tag<first_tag+2;tag++)begin
            expected=payload(tag);
            @(posedge mem_clk);
            check(m_valid===3 && return_data===expected && response_data===expected[264:0],
                  "ordered response after backpressure");popped++;
            #0.02;
        end
        @(negedge mem_clk);m_ready=0;
        check(empty===3 && m_valid===0,"queues drained without loss or duplication");groups++;
    endtask
    initial begin
        #0.19;cold_n=0;ticks(4);mticks(4);
        @(negedge mem_clk);#(PHASE);cold_n=1;
        ticks(25);check(!started && !fault && mem_n,"calibration required");groups++;
        calibration=1;ticks(20);check(!started,"both stacks required");groups++;
        calibration=3;acquire();
        push(0);push(1);stalled(0,8);
        calibration=2;ticks(12);check(fault,"calibration loss latches fault");
        stalled(0,12);calibration=3;ticks(12);check(fault,"fault sticky");
        drain(0);
        cold();acquire();push(2);push(3);
        adapter_fault=1;ticks(12);check(fault,"adapter fault latches fault");
        stalled(2,12);adapter_fault=0;drain(2);
        cold();acquire();push(4);push(5);
        @(negedge mem_clk);memory_on=0;ticks(8);
        check(mem_n && !empty[0] && !empty[1],"clock pause preserves outstanding payload");
        @(negedge clk);source_on=0;#30;cold_n=0;#0.02;
        check(!mem_n && !src_n,"stopped-clock cold assertion");
        cold_n=1;#30;check(!mem_n && !src_n,"stopped-clock release blocked");groups++;
        memory_on=1;mticks(6);check(mem_n && !src_n,"independent memory release");
        check(empty===3 && m_valid===0,"cold epoch discards old payload");groups++;
        source_on=1;acquire();push(6);push(7);drain(6);
        // Pause just before the new fourth edge: no level-based early release.
        cold();@(posedge mem_clk);@(posedge mem_clk);@(posedge mem_clk);
        @(negedge mem_clk);memory_on=0;#20;
        check(ref_mem_n && !mem_n,"fourth edge required with paused memory clock");groups++;
        memory_on=1;mticks(1);check(mem_n,"fourth edge releases distribution");
        acquire();check(popped==6,"six pairs of queue transfers completed");
        $display("RD CASE PASS case=%0d checks=%0d groups=%0d popped=%0d",ID,checks,groups,popped);done=1;
    end
endmodule
module tb_hbm_reset_distribution;
    wire [3:0] done;
    reset_distribution_case #(.ID(0),.HALF(2.5),.MEM_HALF(1.111),.PHASE(0.13)) a(done[0]);
    reset_distribution_case #(.ID(1),.HALF(2.5),.MEM_HALF(1.111),.PHASE(0.79)) b(done[1]);
    reset_distribution_case #(.ID(2),.HALF(2.5),.MEM_HALF(3.5),.PHASE(0.31)) c(done[2]);
    reset_distribution_case #(.ID(3),.HALF(3.7),.MEM_HALF(1.111),.PHASE(0.59)) d(done[3]);
    initial begin wait(&done);$display("RD PASS cases=4");$finish;end
    initial begin #30000;$fatal(1,"RD TIMEOUT");end
endmodule
`default_nettype wire
