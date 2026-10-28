`timescale 1ns / 1ps
module tb_ntrpu_local_reset_actuator;
    reg m=0,h=0,d=0,hrun=1,drun=1,cold=0;
    always #5 m=~m;
    initial begin #0.7;forever begin #3.5;if(hrun)h=~h;end end
    initial begin #1.1;forever begin #18.5;if(drun)d=~d;end end
    reg req=0,hold_req=0,ph=1,pr=1;
    reg [2:0] kind=0;
    reg [63:0] gen=64'h100000001,tag=64'h200000000;
    wire ack,hn,dn,held_req,hh,dh,hu,du;
    wire [2:0] ak,state;
    wire [63:0] ag,at;
    integer scenario=-1,passes=0,hedges=0,dedges=0;
    ntrpu_local_reset_actuator dut(.mgmt_clk(m),.host_clk(h),.device_clk(d),
        .cold_rst_n(cold),.req_valid(req),.hold_request(hold_req),.peers_held(ph),.peers_ready(pr),
        .req_kind(kind),.req_generation(gen),.req_tag(tag),.ack_valid(ack),.ack_kind(ak),
        .ack_generation(ag),.ack_tag(at),.host_reset_n(hn),.device_reset_n(dn),
        .peers_reset_req(held_req),.host_held(hh),.device_held(dh),.host_up(hu),.device_up(du),.state_debug(state));
    always @(posedge h)if(!hn)hedges=hedges+1;
    always @(posedge d)if(!dn)dedges=dedges+1;
    task ck(input bit ok,input string s);if(!ok)$fatal(1,"LOCAL_ASSERT case=%0d %s",scenario,s);endtask
    task tick(input integer n);repeat(n)@(negedge m);endtask
    task mark(input string s);
        $display("LOCAL_EVENT case=%0d name=%s ps=%0.0f state=%0d ack=%b kind=%0d gen=%h tag=%h hn=%b dn=%b held=%b hh=%b dh=%b hu=%b du=%b he=%0d de=%0d",scenario,s,$realtime*1000,state,ack,kind,gen,tag,hn,dn,held_req,hh,dh,hu,du,hedges,dedges);
    endtask
    task cmd(input [2:0] k,input bit hold_value);
        begin @(negedge m);kind=k;hold_req=hold_value;tag=tag+1;req=1;#0.001;ck(!ack,"old identity must be masked");end
    endtask
    task start_hold;
        begin hedges=0;dedges=0;cmd(2,1);tick(8);ck(held_req && !hn && !dn,"both resets asserted");end
    endtask
    task wait_ack;
        integer n;begin n=0;while(!ack)begin tick(1);n=n+1;ck(n<2000,"bounded ACK");end
        ck({ak,ag,at}=={kind,gen,tag},"exact response identity");mark("ACK");end
    endtask
    task pass(input string s);passes=passes+1;$display("LOCAL_CASE_PASS case=%0d name=%s ps=%0.0f",scenario,s,$realtime*1000);endtask
    always @(posedge m)begin #0.001;if(cold && ack)begin
        ck(req && {ak,ag,at}=={kind,gen,tag},"continuous response identity");
        if(ak==2)ck(!hn && !dn && held_req && hedges>=64 && dedges>=64 && ph,"real held edges before ACK");
        else ck(ak==5 && hn && dn && !held_req && pr,"ready after both releases and peers");
    end end
    // Deassertion must be after real local positive clock edges.
    realtime hlast=0,dlast=0;
    always @(posedge h)hlast=$realtime;
    always @(posedge d)dlast=$realtime;
    always @(posedge hn)ck($realtime==hlast,"host synchronous release");
    always @(posedge dn)ck($realtime==dlast,"device synchronous release");
    initial begin
        tick(20);cold=1;tick(10);
        scenario=0;cmd(5,0);tick(50);ck(!ack && hn && dn && !held_req,"READY without a held phase has no ACK");
        cmd(7,1);tick(50);ck(!ack && !held_req,"unknown phase cannot assert reset");mark("NO_HOLD");pass("unrequested_reset_and_unknown_phase");

        scenario=1;@(negedge h);hrun=0;start_hold();tick(300);
        ck(!ack && !hn && !dn && !hh && dh && hedges==0 && dedges>=64,"host stop cannot count elapsed time");
        mark("HOST_STOP");pass("stopped_host_blocks_held");

        scenario=2;cmd(5,0);tick(100);ck(!ack && held_req && !hn && !dn,"early ready cannot release before counters");mark("EARLY_READY");
        ph=0;hrun=1;tick(400);ck(!ack && held_req && hh && dh,"missing held peer blocks release");mark("NO_PEER_HELD");pass("early_release_and_missing_held_peer");

        scenario=3;@(negedge d);drun=0;ph=1;pr=0;tick(80);
        ck(!ack && !held_req && hn && !dn,"stopped device blocks release completion");mark("DEVICE_STOP_RELEASE");
        drun=1;tick(100);ck(hn && dn && !ack,"missing peer readiness blocks ACK");mark("NO_PEER_READY");
        pr=1;wait_ack();pass("both_local_releases_and_peer_readiness");

        scenario=4;@(negedge d);drun=0;start_hold();tick(300);
        ck(!ack && hh && !dh && dedges==0 && held_req,"old held bits cannot certify a fresh hold");mark("DEVICE_STOP_NEW_HOLD");
        drun=1;wait_ack();pass("fresh_hold_clears_previous_counts");

        scenario=5;@(negedge m);req=0;#0.001;ck(!ack,"withdrawal masks response");tick(100);
        ck(held_req && !hn && !dn,"withdrawal never releases asserted reset");mark("WITHDRAWN");
        cmd(5,1);tick(100);ck(!ack && held_req,"ready command cannot override hold request");mark("HOLD_DOMINATES_READY");
        hold_req=0;wait_ack();pass("withdrawal_and_hold_dominance");

        scenario=6;start_hold();wait_ack();@(negedge h);hrun=0;
        @(negedge m);hedges=0;dedges=0;gen=gen+1;#0.001;ck(!ack,"generation replacement masks old ACK");
        tick(300);ck(!ack && held_req && !hh && dh && hedges==0,"fresh generation recounts stopped domain");mark("NEW_GENERATION");
        hrun=1;wait_ack();pass("generation_is_part_of_command_identity");

        scenario=7;cmd(5,0);wait_ack();pr=0;#0.001;ck(!ack,"lost peer readiness masks ACK");tick(20);
        ck(!held_req && hn && dn,"readiness loss alone cannot reset undrained transport");mark("READY_LOST");
        pr=1;wait_ack();req=0;tick(10);ck(!ack,"terminal withdrawal");pass("readiness_loss_requires_parent_recovery");
        ck(passes==8,"case count");$display("LOCAL_SUITE_PASS cases=8 hold_cycles=64 native_endpoint=0 complete_drain=0");$finish;
    end
    initial begin #200000;$fatal(1,"LOCAL_TIMEOUT");end
endmodule
