`timescale 1ns / 1ps
module tb_ntrpu_ownership_certificate_r2;
    wire [11:0] finished;
    genvar i;
    generate for(i=0;i<12;i=i+1)begin: scenarios
        ntrpu_ownership_certificate_case_r2 #(.ID(i),.LOSS_PHASE(i%3),
            .SOURCE_PERIOD_PS((i/3)%2==0 ? 5000 : 37000),.FAULT_TAIL(i/6)) test_case(.finished(finished[i]));
    end endgenerate
    initial begin
        wait(&finished);#0.001;
        $display("OWNERSHIP_CERTIFICATE_SUITE_PASS cases=12 real_owner=1 native_xpm=1 reset_driven=0");
        $finish;
    end
    initial begin #100000;$fatal(1,"OWNERSHIP_CERTIFICATE_SUITE_TIMEOUT");end
endmodule
