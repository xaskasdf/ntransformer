`timescale 1ns/1ps
module tb_ntrpu_mig_cold_start;
    reg mgmt_clk=0;always #5 mgmt_clk=~mgmt_clk;
    wire [7:0] passed;
    for(genvar g=0;g<8;g++) begin: cases
        reg source_clk=0,ui_clk=0,run_source=1,run_ui=1;
        always #2.5 if(run_source) source_clk=~source_clk;
        initial begin #0.713;forever #3.65 if(run_ui) ui_clk=~ui_clk;end
        reg external_reset_async=1,rearm=1,calibrated=0,ui_rst=1;
        reg source_clean=(g!=7),memory_bvalid=(g==6),memory_rvalid=0;
        wire source_reset_n,memory_reset_n,admit,peers_reset_req,rearm_ready,running,restart_required;
        wire host_alive,device_alive;
        reg done=0,seen=0;integer entries=0;
        assign passed[g]=done;
        ntrpu_mig_cold_start dut(.*);
        always @(posedge source_clk) begin
            if(entries==0 && admit && (!calibrated || ui_rst)) $fatal(1,"COLD_START_EARLY_ADMISSION case=%0d",g);
            if(admit && !seen) begin entries++;seen=1;end
            if(!admit) seen=0;
        end
        initial begin
            #50;external_reset_async=0;
            #700;
            if(admit || running) $fatal(1,"COLD_START_UNCALIBRATED case=%0d",g);
            calibrated=1;ui_rst=0;
            if(g>=6) begin
                #1200;if(rearm_ready || admit || running) $fatal(1,"COLD_START_DIRTY_PEER case=%0d",g);
                memory_bvalid=0;source_clean=1;
            end
            wait(admit);#200;
            if(g==1) calibrated=0;
            if(g==2) ui_rst=1;
            if(g==3) run_source=0;
            if(g==4) run_ui=0;
            if(g==5) external_reset_async=1;
            #1500;
            if(g>=1 && g<=5) begin
                if(admit || running || source_reset_n || memory_reset_n || !restart_required)
                    $fatal(1,"COLD_START_FAULT_NOT_CLOSED case=%0d",g);
                calibrated=1;ui_rst=0;run_source=1;run_ui=1;external_reset_async=0;
            end
            #1800;
            if(entries!=1 || ((g>=1 && g<=5) && (admit || running || rearm_ready)))
                $fatal(1,"COLD_START_REOPENED case=%0d",g);
            if((g==0 || g>=6) && (!admit || !running || restart_required))
                $fatal(1,"COLD_START_HEALTHY_STOPPED case=%0d",g);
            $display("COLD_START_CASE_PASS case=%0d entries=%0d locked=%b",g,entries,restart_required);
            done=1;
        end
    end
    initial begin wait(&passed);$display("COLD_START_SUITE_PASS cases=8");$finish;end
    initial begin #15000;$fatal(1,"COLD_START_SUITE_TIMEOUT");end
endmodule
