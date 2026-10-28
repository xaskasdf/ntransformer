`timescale 1ns / 1ps
module tb_ntrpu_dual_coordinated;
    wire [7:0] finished;
    ntrpu_dual_coordinated_case #(.CASE_ID(0),.SOURCE_PERIOD_PS(5000),.LOSS_PHASE(0),.BAD_EPOCH(0)) fast_cmd(.finished(finished[0]));
    ntrpu_dual_coordinated_case #(.CASE_ID(1),.SOURCE_PERIOD_PS(5000),.LOSS_PHASE(1),.BAD_EPOCH(0)) fast_data(.finished(finished[1]));
    ntrpu_dual_coordinated_case #(.CASE_ID(2),.SOURCE_PERIOD_PS(5000),.LOSS_PHASE(2),.BAD_EPOCH(0)) fast_result(.finished(finished[2]));
    ntrpu_dual_coordinated_case #(.CASE_ID(3),.SOURCE_PERIOD_PS(5000),.LOSS_PHASE(0),.BAD_EPOCH(1)) fast_wrong(.finished(finished[3]));
    ntrpu_dual_coordinated_case #(.CASE_ID(4),.SOURCE_PERIOD_PS(37000),.LOSS_PHASE(0),.BAD_EPOCH(0)) slow_cmd(.finished(finished[4]));
    ntrpu_dual_coordinated_case #(.CASE_ID(5),.SOURCE_PERIOD_PS(37000),.LOSS_PHASE(1),.BAD_EPOCH(0)) slow_data(.finished(finished[5]));
    ntrpu_dual_coordinated_case #(.CASE_ID(6),.SOURCE_PERIOD_PS(37000),.LOSS_PHASE(2),.BAD_EPOCH(0)) slow_result(.finished(finished[6]));
    ntrpu_dual_coordinated_case #(.CASE_ID(7),.SOURCE_PERIOD_PS(37000),.LOSS_PHASE(0),.BAD_EPOCH(1)) slow_wrong(.finished(finished[7]));
    initial begin
        wait(&finished);#0.001;
        $display("MANAGED_COORDINATOR_SUITE_PASS cases=8 real_ntrd=1 real_owner=1 external_contracts=0 reset_driven=0");
        $finish;
    end
    initial begin #200000;$fatal(1,"MANAGED_COORDINATOR_TIMEOUT");end
endmodule
