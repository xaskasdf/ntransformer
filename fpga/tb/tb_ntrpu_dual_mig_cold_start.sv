`timescale 1ns/1ps
module tb_ntrpu_dual_mig_cold_start;
    reg mgmt_clk=0;always #5 mgmt_clk=~mgmt_clk;
    wire [12:0] passed;
    for(genvar g=0;g<13;g++) begin: cases
        reg source_clk=0,run_source=1;
        reg [1:0] ui_clk=0,run_ui=3,calibrated=0,ui_rst=3;
        always #(g==12 ? 0.5 : 2.5) if(run_source) source_clk=~source_clk;
        always #2.5 if(run_ui[0]) ui_clk[0]=~ui_clk[0];
        initial begin #0.731;forever #3.65 if(run_ui[1]) ui_clk[1]=~ui_clk[1];end
        reg external_reset_async=1,rearm=1,source_clean=(g!=9);
        reg [1:0] memory_bvalid=(g==10 ? 1:0),memory_rvalid=(g==11 ? 2:0);
        wire source_reset_n,admit,rearm_ready,peers_reset_req,source_alive,restart_required,running;
        wire [1:0] memory_reset_n,memory_alive;
        reg done=0,seen=0;integer entries=0;
        assign passed[g]=done;
        ntrpu_dual_mig_cold_start dut(.*);
        always @(posedge source_clk) begin
            if(entries==0 && admit && (calibrated!=3 || ui_rst!=0)) $fatal(1,"DUAL_START_EARLY case=%0d",g);
            if(admit && !seen) begin entries++;seen=1;end
            if(!admit) seen=0;
        end
        initial begin
            #40;external_reset_async=0;
            #1100;calibrated[0]=1;ui_rst[0]=0;
            #1100;if(admit || running) $fatal(1,"DUAL_START_ONE_READY case=%0d",g);
            calibrated[1]=1;ui_rst[1]=0;
            if(g>=9 && g<=11) begin
                #1100;if(admit || running || rearm_ready) $fatal(1,"DUAL_START_DIRTY case=%0d",g);
                source_clean=1;memory_bvalid=0;memory_rvalid=0;
            end
            wait(admit);
            if(g==12)begin @(posedge source_clk);#0.1;external_reset_async=1;end
            else #300;
            case(g)
                1:calibrated[0]=0;
                2:calibrated[1]=0;
                3:ui_rst[0]=1;
                4:ui_rst[1]=1;
                5:run_source=0;
                6:run_ui[0]=0;
                7:run_ui[1]=0;
                8:external_reset_async=1;
            endcase
            #1500;
            if((g>=1 && g<=8) || g==12) begin
                if(admit || running || source_reset_n || memory_reset_n!=0 || !restart_required)
                    $fatal(1,"DUAL_START_FAULT_NOT_CLOSED case=%0d",g);
                calibrated=3;ui_rst=0;run_source=1;run_ui=3;external_reset_async=0;
            end
            #1800;
            if(entries!=1 || (((g>=1 && g<=8) || g==12) && (admit || running || rearm_ready)))
                $fatal(1,"DUAL_START_REOPENED case=%0d",g);
            if((g==0 || (g>=9 && g<=11)) && (!admit || !running || restart_required)) $fatal(1,"DUAL_START_HEALTHY case=%0d",g);
            $display("DUAL_START_CASE_PASS case=%0d entries=%0d locked=%b",g,entries,restart_required);done=1;
        end
    end
    initial begin wait(&passed);$display("DUAL_START_SUITE_PASS cases=13");$finish;end
    initial begin #15000;$fatal(1,"DUAL_START_TIMEOUT");end
endmodule
