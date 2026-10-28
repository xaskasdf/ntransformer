`timescale 1ns/1ps
`ifndef NTRPU_LEDGER_BITS
`define NTRPU_LEDGER_BITS 32
`endif
module tb_ntrpu_xdma_descriptor_ledger;
    localparam integer BITS=`NTRPU_LEDGER_BITS;
    reg clk=0;
    always #2 clk=~clk;
    reg cold_rst_n=0,core_ready=0,arm_valid=0,publish_valid=0,publish_channel=0;
    reg close_request=0,host_closed=0,release_valid=0;
    reg [63:0] arm_generation=0,arm_tag=0,publish_generation=0,publish_tag=0;
    reg [63:0] host_generation=0,host_tag=0;
    reg [31:0] publish_index=0;
    reg [7:0] h2c_status=0,c2h_status=0;
    wire arm_ready,active,closed,accounted;
    wire [63:0] generation,tag,published,completed,outstanding;
    wire [1:0] started;
    wire [8:0] faults;
    integer phase=0,cases=0;
    ntrpu_xdma_descriptor_ledger #(.COUNTER_BITS(BITS)) dut(.*);
    task require(input bit ok,input string why);
        if(!ok) $fatal(1,"LEDGER_ASSERT phase=%0d %s",phase,why);
    endtask
    task cycles(input integer n);repeat(n) @(negedge clk);endtask
    task configure;
        begin
            cold_rst_n=0;core_ready=0;arm_valid=0;publish_valid=0;
            h2c_status=0;c2h_status=0;close_request=0;host_closed=0;release_valid=0;
            cycles(3);cold_rst_n=1;core_ready=1;cycles(1);
            require(faults==0 && !active && !accounted,"new cold configuration");
        end
    endtask
    task arm(input [63:0] g,input [63:0] t);
        begin
            arm_generation=g;arm_tag=t;cycles(1);require(arm_ready,"fresh arm ready");
            arm_valid=1;cycles(1);arm_valid=0;
            require(active && generation==g && tag==t && published==0 && completed==0,"new session ledger");
        end
    endtask
    task publish(input bit channel,input [31:0] index);
        begin
            publish_channel=channel;publish_index=index;publish_generation=generation;publish_tag=tag;
            publish_valid=1;cycles(1);publish_valid=0;
        end
    endtask
    task host_close;
        begin
            close_request=1;host_generation=generation;host_tag=tag;host_closed=1;cycles(1);
            require(closed,"current host closure accepted");
        end
    endtask
    task finish_case(input [8:0] expected_faults);
        begin
            cycles(2);require(faults===expected_faults,"exact retained fault set");
            if(expected_faults!=0) require(!accounted,"fault never authorizes accounting");
            cases=cases+1;
            $display("LEDGER_CASE t=%0t phase=%0d faults=%h published=%h completed=%h outstanding=%h active=%b closed=%b accounted=%b",$time,phase,faults,published,completed,outstanding,active,closed,accounted);
        end
    endtask
    always @(posedge clk) if(cold_rst_n) begin
        if(arm_valid&&arm_ready) $display("LEDGER_ARM t=%0t phase=%0d generation=%h tag=%h",$time,phase,arm_generation,arm_tag);
        if(publish_valid) $display("LEDGER_PUBLISH t=%0t phase=%0d channel=%b index=%h generation=%h tag=%h",$time,phase,publish_channel,publish_index,publish_generation,publish_tag);
        if(h2c_status[3]||c2h_status[3]) $display("LEDGER_DONE t=%0t phase=%0d mask=%b",$time,phase,{c2h_status[3],h2c_status[3]});
    end
    initial begin
        cycles(1);phase=1;configure;arm(11,101);
        publish(0,0);h2c_status='h41;cycles(1);
        h2c_status[3]=1;publish(0,1);h2c_status[3]=0;
        publish(1,0);c2h_status='h41;cycles(1);
        close_request=1;cycles(3);require(!closed && !accounted,"wait for host closure");
        c2h_status='h08;cycles(1);c2h_status=0;host_close;
        h2c_status=1;cycles(2);require(outstanding==1 && !accounted,"Run zero retains descriptor");
        h2c_status=0;cycles(2);require(!accounted,"Busy zero alone is insufficient");
        h2c_status=8;cycles(1);h2c_status=0;cycles(1);
        require(accounted && published==64'h0000000100000002 && published==completed,"all descriptors retired");
        finish_case(0);
        release_valid=1;cycles(1);release_valid=0;require(!active,"local release after external retirement");
        phase=2;arm(11,102);close_request=1;cycles(3);
        require(!closed && !accounted && faults==0,"old high certificate cannot arm new round");
        host_closed=0;cycles(1);host_close;require(accounted,"fresh zero-work closure");finish_case(0);
        release_valid=1;cycles(1);release_valid=0;

        phase=3;configure;arm(13,103);h2c_status=8;cycles(1);h2c_status=0;finish_case(9'h008);
        phase=4;configure;arm(14,104);publish(0,0);publish(0,0);finish_case(9'h002);
        require(published==1 && completed==0,"duplicate does not create a second legitimate descriptor");
        phase=5;configure;arm(15,105);publish(0,0);h2c_status='h41;cycles(1);
        h2c_status=0;cycles(1);h2c_status='h41;cycles(1);finish_case(9'h010);
        require(published==1 && completed==0,"Run restart never clears obligations");
        phase=6;configure;arm(16,106);close_request=1;host_closed=1;
        host_generation=generation;host_tag=tag+1;cycles(1);finish_case(9'h002);
        phase=7;configure;arm(17,107);host_close;require(accounted,"empty closed ledger");
        host_closed=0;#1;require(!accounted,"immediate host revocation");cycles(1);
        host_closed=1;finish_case(9'h100);
        phase=8;configure;arm(18,108);publish(0,0);h2c_status='h41;cycles(1);host_close;
        h2c_status=0;publish(0,1);h2c_status=8;cycles(2);h2c_status=0;
        require(published==2 && completed==2,"late publication and real completions remain recorded");finish_case(9'h020);
        phase=9;configure;arm(19,109);host_close;require(accounted,"proof before readiness loss");
        core_ready=0;#1;require(!accounted,"readiness gates proof immediately");cycles(1);
        core_ready=1;finish_case(9'h001);
        phase=10;configure;arm(20,110);publish(0,0);release_valid=1;cycles(1);
        release_valid=0;require(active && outstanding==1,"premature release preserves obligations");finish_case(9'h080);
        phase=11;configure;arm(21,111);publish_channel=0;publish_index=0;
        publish_generation=generation+1;publish_tag=tag;publish_valid=1;cycles(1);publish_valid=0;
        require(published==0,"foreign descriptor is rejected and faults retained");finish_case(9'h002);
        phase=12;configure;arm(22,112);host_close;release_valid=1;cycles(1);release_valid=0;
        arm_valid=1;cycles(1);arm_valid=0;require(!active && !arm_ready,"identity cannot be reused");finish_case(9'h002);
        if(BITS==4) begin
            phase=13;configure;arm(23,113);publish(0,0);h2c_status='h41;cycles(1);
            h2c_status='h49;cycles(1);h2c_status='h41;
            for(integer i=1;i<15;i=i+1) begin h2c_status='h49;publish(0,i);h2c_status='h41;end
            require(published==15 && completed==15,"counter reaches maximum without wrap");
            publish(0,15);require(published==15 && completed==15,"saturate and retain overflow fault");finish_case(9'h040);
        end
        phase=14;configure;arm(24,114);h2c_status='h41;cycles(1);finish_case(9'h010);
        phase=15;configure;publish_channel=0;publish_index=0;publish_generation=1;publish_tag=1;
        publish_valid=1;cycles(1);publish_valid=0;finish_case(9'h004);
        phase=16;configure;arm(26,116);host_close;close_request=0;#1;
        require(!accounted,"withdrawn request revokes immediately");cycles(1);close_request=1;finish_case(9'h100);
        require(cases==(BITS==4?16:15),"complete case matrix");
        $display("LEDGER_PASS cases=%0d bits=%0d native_xdma=0 global_drain=0 reset_permission=0",cases,BITS);$finish;
    end
    initial begin #20000;$fatal(1,"LEDGER_WATCHDOG phase=%0d",phase);end
endmodule
