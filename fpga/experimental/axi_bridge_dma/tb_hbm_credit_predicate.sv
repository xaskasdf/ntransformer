`timescale 1ns/1ps
module tb_hbm_credit_predicate;
    reg [7:0] remaining;
    reg ar_fire,capture_fire;
    wire next_available;
    ntrpu_hbm_capture_credit_predictor dut(.*);
    integer n,len,a,c,next_count,checks=0,excluded=0;
    initial begin
        for(n=0;n<=128;n=n+1)
            for(len=0;len<16;len=len+1)
                for(a=0;a<2;a=a+1)
                    for(c=0;c<2;c=c+1)begin
                        if(n==0&&c==1)excluded=excluded+1;
                        else begin
                            remaining=n;ar_fire=a;capture_fire=c;#1;
                            next_count=n+(a?(len+1):0)-c;
                            if(next_available!==(next_count!=0))
                                $fatal(1,"CREDIT_PREDICATE bounded credit availability n=%0d len=%0d ar=%0d capture=%0d",n,len,a,c);
                            checks=checks+1;
                        end
                    end
        if(checks!=8224||excluded!=32)$fatal(1,"CREDIT_PREDICATE inventory");
        $display("CREDIT_PREDICATE PASS checks=%0d excluded=%0d",checks,excluded);
        $display("CREDIT_PREDICATE ALL PASS");$finish;
    end
    initial begin #20000;$fatal(1,"CREDIT_PREDICATE watchdog");end
endmodule
