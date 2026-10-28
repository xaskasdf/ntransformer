`timescale 1ps/1ps
// Diagnostic only: reset after the first management sample of a hold-done
// level. Exercise real source registers; never force DUT internal state.
module tb_ntrpu_reset_hold_retry;
    parameter integer PROFILE=0;
    localparam integer HP=PROFILE==0 ? 4000 : 31000;
    localparam integer DP=PROFILE==0 ? 5000 : 23000;
    logic mgmt_clk=0,host_clk=0,device_clk=0;
    logic external_reset_async=1,peers_held=1,peers_ready=0,rearm=0;
    wire peers_reset_req,host_reset_n,device_reset_n,host_admit,device_admit;
    wire rearm_ready,host_alive,device_alive,running;
    integer attempt=0;
    ntrpu_reset_supervisor dut(.*);
    always #5000 mgmt_clk=~mgmt_clk;
    initial begin #700;forever #(HP/2) host_clk=~host_clk;end
    initial begin #1100;forever #(DP/2) device_clk=~device_clk;end
    always @(posedge mgmt_clk) begin
        #1;
        $display("NTRPU_HOLD_M profile=%0d attempt=%0d ps=%0t ext=%0b hd=%0b dd=%0b hm=%0b dm=%0b req=%0b hrst=%0b drst=%0b ha=%0b da=%0b run=%0b",
            PROFILE,attempt,$time,external_reset_async,dut.host_done,dut.device_done,dut.host_done_m,dut.device_done_m,
            peers_reset_req,host_reset_n,device_reset_n,host_admit,device_admit,running);
        if(host_reset_n || device_reset_n || host_admit || device_admit || running)
            $fatal(1,"FAIL HOLD retry admitted without rearm");
    end
    always @(dut.host_done or dut.device_done) begin
        #1;
        $display("NTRPU_HOLD_LEVEL profile=%0d attempt=%0d ps=%0t hd=%0b dd=%0b",PROFILE,attempt,$time,dut.host_done,dut.device_done);
    end
    initial begin
        $display("NTRPU_HOLD_CONFIG profile=%0d host_ps=%0d device_ps=%0d mgmt_ps=10000",PROFILE,HP,DP);
        for(attempt=1;attempt<=8;attempt=attempt+1) begin
            external_reset_async=1;
            #80000;
            if(dut.host_done || dut.device_done) $fatal(1,"FAIL HOLD counter not cleared");
            external_reset_async=0;
            if(attempt%2) @(posedge dut.host_done);else @(posedge dut.device_done);
            @(posedge mgmt_clk);
            #(101+137*attempt);
            $display("NTRPU_HOLD_RETRY profile=%0d attempt=%0d ps=%0t target=%0d hd=%0b dd=%0b",PROFILE,attempt,$time,attempt%2,dut.host_done,dut.device_done);
            external_reset_async=1;
            #78700;
        end
        #100000;
        $display("NTRPU_HOLD_RETRY_RETURNED profile=%0d attempts=8",PROFILE);
        $finish;
    end
    initial begin #100000000;$fatal(1,"FAIL HOLD watchdog");end
endmodule
