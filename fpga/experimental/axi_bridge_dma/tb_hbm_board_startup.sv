`timescale 1ns/1ps
`default_nettype none
module startup_case #(parameter integer HOLD=16, STABLE=32)(output reg done=0);
    reg clk=0, enable=1, supervisor_n=0, locked=0;
    wire clock_reset, configuration_n, fault;
    integer checks=0, groups=0, wait_cycles;
    ntrpu_hbm_board_startup #(.RESET_HOLD_CYCLES(HOLD),.LOCK_STABLE_CYCLES(STABLE)) dut(
        .mgmt_clk(clk),.supervisor_rst_n(supervisor_n),.fabric_locked_async(locked),
        .fabric_clock_reset(clock_reset),.configuration_rst_n(configuration_n),.lock_fault(fault));
    always #5 if(enable)clk=~clk;else clk=0;
    task check(input bit condition,input string message);
        checks++;if(!condition)$fatal(1,"BS FAIL hold=%0d stable=%0d %s",HOLD,STABLE,message);
    endtask
    task tick(input integer n);
        repeat(n)begin @(posedge clk);#0.1;end
    endtask
    task cold;
        @(negedge clk);#1;supervisor_n=0;locked=0;#0.1;
        check(clock_reset===1 && configuration_n===0 && fault===0,"asynchronous cold assertion");
        tick(2);@(negedge clk);#1;supervisor_n=1;
        // Three release synchronizer samples, then HOLD management cycles.
        for(integer i=0;i<3+HOLD;i++)begin
            tick(1);check(configuration_n===0,"cold state held during oscillator reset");
            check(clock_reset===(i<2+HOLD),"reset hold duration");
        end
        groups++;
    endtask
    task acquire;
        @(negedge clk);#1;locked=1;wait_cycles=0;
        while(configuration_n!==1 && wait_cycles<STABLE+8)begin
            tick(1);wait_cycles++;
            if(wait_cycles<STABLE+3)check(configuration_n===0,"no early lock release");
        end
        check(wait_cycles==STABLE+3 && configuration_n===1,"qualified lock releases configuration");
        check(clock_reset===0 && fault===0,"clean startup");groups++;
    endtask
    initial begin
        tick(2);cold();
        tick(STABLE+8);check(configuration_n===0 && clock_reset===0,"missing lock stalls startup");groups++;
        if(STABLE>1)begin
            @(negedge clk);locked=1;tick(1);@(negedge clk);locked=0;
            tick(STABLE+6);check(configuration_n===0,"brief lock rejected");groups++;
        end
        acquire();
        // A sampled running lock loss may raise fault, but cannot reset an epoch.
        @(negedge clk);locked=0;
        repeat(8)begin tick(1);check(configuration_n===1 && clock_reset===0,"lock loss preserves epoch");end
        check(fault===1,"lock loss latched");groups++;
        locked=1;tick(STABLE+8);check(fault===1 && configuration_n===1,"relock cannot clear fault");groups++;
        // Supervisor still asserts both resets without any management clock edges.
        @(negedge clk);enable=0;#20;supervisor_n=0;#0.1;
        check(clock_reset===1 && configuration_n===0 && fault===0,"cold assertion with halted clock");
        supervisor_n=1;#40;check(clock_reset===1 && configuration_n===0,"halted clock cannot release reset");groups++;
        enable=1;cold();acquire();
        // A stopped management clock cannot qualify LOCKED or create progress.
        cold();@(negedge clk);enable=0;locked=1;#80;
        check(configuration_n===0,"stopped management clock cannot qualify lock");groups++;
        enable=1;locked=0;tick(6);acquire();
        $display("BS CASE PASS hold=%0d stable=%0d checks=%0d groups=%0d",HOLD,STABLE,checks,groups);
        done=1;
    end
endmodule
module tb_hbm_board_startup;
    wire [3:0] done;
    startup_case #(.HOLD(1),.STABLE(1)) a(done[0]);
    startup_case #(.HOLD(4),.STABLE(3)) b(done[1]);
    startup_case #(.HOLD(16),.STABLE(32)) c(done[2]);
    startup_case #(.HOLD(17),.STABLE(9)) d(done[3]);
    initial begin wait(&done);$display("BS PASS cases=4");$finish;end
    initial begin #20000;$fatal(1,"BS TIMEOUT");end
endmodule
`default_nettype wire
