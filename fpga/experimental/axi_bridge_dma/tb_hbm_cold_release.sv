`timescale 1ns/1ps
`default_nettype none
module cold_release_case #(
    parameter integer ID=0,
    parameter realtime HALF=2.5, MEM_HALF=1.111, PHASE=0.4
)(output reg done=0);
    reg clk=0, mem_clk=0, clocks_on=1, cold_n=1;
    reg [1:0] ready=0, adapter_fault=0;
    wire src_n, mem_n, started, fault;
    integer checks=0, groups=0, cycles;
    reg sampled_release;
    ntrpu_hbm_startup dut(.clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_n),
        .init_complete_async(ready),.adapter_fault_mem(adapter_fault),
        .src_rst_n(src_n),.mem_rst_n(mem_n),.started(started),.fault(fault));
    always #(HALF) if(clocks_on)clk=~clk;else clk=0;
    always #(MEM_HALF) if(clocks_on)mem_clk=~mem_clk;else mem_clk=0;
    task check(input bit condition,input string message);
        checks++;if(!condition)$fatal(1,"RC FAIL case=%0d %s",ID,message);
    endtask
    task tick(input integer n);
        repeat(n)begin @(posedge clk);#0.02;end
    endtask
    // Check the pre-edge reset state after NBA updates. The old raw reset
    // starts sampling before the canonical reader release and must fail here.
    always @(posedge clk)begin
        sampled_release=src_n;
        #0.01;
        if(sampled_release===0)
            check({dut.ready_meta,dut.ready_sync,dut.ready_final,
                   dut.fault_meta,dut.fault_sync,dut.fault_final}===15'b0,
                  "status held until canonical reader release");
    end
    task cold;
        @(negedge clk);#0.1;cold_n=0;#0.02;
        check(src_n===0 && mem_n===0,"asynchronous domain assertion");
        tick(3);check(!started && !fault,"clocked cold state clears status");
        @(negedge clk);#(PHASE);cold_n=1;groups++;
    endtask
    task acquire;
        cycles=0;
        while(started!==1 && cycles<50)begin tick(1);cycles++;end
        check(started===1 && fault===0 && src_n===1 && mem_n===1,"qualified startup");
        groups++;
    endtask
    initial begin
        // High async status exposes sampling before the synchronized release.
        #0.2;cold_n=0;ready=3;adapter_fault=3;tick(3);
        @(negedge clk);#(PHASE);cold_n=1;
        tick(18);check(fault===1 && started===0,"pre-start adapter fault");groups++;
        ready=0;adapter_fault=0;cold();tick(20);
        check(!started && !fault && src_n && mem_n,"missing calibration blocks startup");groups++;
        ready=1;tick(20);check(!started,"both stacks required");groups++;
        ready=3;acquire();
        ready=2;tick(8);check(fault && src_n && mem_n,"calibration loss preserves reset epoch");groups++;
        ready=3;tick(20);check(fault && src_n && mem_n,"fault remains sticky");groups++;
        cold();acquire();
        adapter_fault=1;tick(8);check(fault && src_n && mem_n,"running adapter fault preserves reset epoch");groups++;
        adapter_fault=0;
        @(negedge clk);clocks_on=0;#20;cold_n=0;#0.02;
        check(src_n===0 && mem_n===0,"cold assertion while clocks stopped");
        check({dut.ready_meta,dut.ready_sync,dut.ready_final,
               dut.fault_meta,dut.fault_sync,dut.fault_final}===15'b0,"async status clear while stopped");
        cold_n=1;#30;check(src_n===0 && mem_n===0,"no cold release without clocks");groups++;
        clocks_on=1;tick(1);acquire();
        $display("RC CASE PASS case=%0d checks=%0d groups=%0d",ID,checks,groups);done=1;
    end
endmodule
module tb_hbm_cold_release;
    wire [3:0] done;
    cold_release_case #(.ID(0),.HALF(2.5),.MEM_HALF(1.111),.PHASE(0.1)) a(done[0]);
    cold_release_case #(.ID(1),.HALF(2.5),.MEM_HALF(1.111),.PHASE(1.3)) b(done[1]);
    cold_release_case #(.ID(2),.HALF(2.5),.MEM_HALF(3.5),.PHASE(2.4)) c(done[2]);
    cold_release_case #(.ID(3),.HALF(3.7),.MEM_HALF(1.111),.PHASE(3.1)) d(done[3]);
    initial begin wait(&done);$display("RC PASS cases=4");$finish;end
    initial begin #20000;$fatal(1,"RC TIMEOUT");end
endmodule
`default_nettype wire
