`timescale 1ns / 1ps
module tb_ntrpu_fragment_pool_bridge_carry;
    reg clk=0;always #2.5 clk=~clk;
    reg rst_n=0,stop=0,pool_admit=0,pool_restart_required=0,pool_fault=0,pool_drained=1;
    reg arm_valid=0,arm_channel=0,retire_valid=0,line_valid=0,done_ready=0;
    reg [63:0] arm_epoch=0,arm_tag=0,arm_destination=0,retire_epoch=0,retire_tag=0;
    reg [63:0] line_epoch=0,line_tag=0,line_address=0;
    reg [511:0] line_data=0;
    reg cmd_ready=0,data_ready=0;
    reg [1:0] cpl_valid=0;
    reg [127:0] cpl_tag=0;
    reg [3:0] cpl_status=0;
    wire arm_ready,retire_ready,line_ready,done_valid,cmd_valid,cmd_channel,data_valid;
    wire protocol_fault,session_lost,session_invalid,exhausted,idle,lease_held;
    wire [63:0] epoch,done_epoch,done_tag,cmd_addr,cmd_tag,next_cookie;
    wire [7:0] cmd_len;
    wire [511:0] data;
    wire [1:0] cpl_ready,done_pool_status;
    wire [3:0] done_status;
    wire [2:0] phase;
    wire [31:0] rejected_completions;
`ifdef NTRPU_CARRY_TEST
    localparam [63:0] FIRST=64'h00000000ffffffff;
`else
    localparam [63:0] FIRST=64'd1;
`endif
`ifdef NTRPU_NETLIST
    ntrpu_fragment_pool_bridge dut (.*);
`else
    ntrpu_fragment_pool_bridge #(.FIRST_COOKIE(FIRST)) dut (.*);
`endif
    integer seed=1,case_id=0,cycles=0,cases=0;
    integer narm=0,nline=0,ncmd=0,ndata=0,ncpl=0,ndone=0,nretire=0;
    reg [63:0] wanted_epoch,wanted_tag,wanted_address,wanted_cookie;
    reg wanted_channel;
    reg [511:0] wanted_data;

    function automatic [511:0] pattern(input integer job);
        reg [511:0] value;
        for(integer lane=0;lane<64;lane++) value[lane*8+:8]=8'(job*37+lane*23+seed*11);
        return value;
    endfunction
    task tick(input integer n);
        repeat(n) @(negedge clk);
    endtask
    task cold(input integer id);
        begin
            @(negedge clk);case_id=id;cases++;rst_n=0;stop=0;pool_admit=0;
            pool_restart_required=0;pool_fault=0;pool_drained=1;
            arm_valid=0;retire_valid=0;line_valid=0;done_ready=0;cmd_ready=0;data_ready=0;cpl_valid=0;
            $display("POOL_BRIDGE_CASE id=%0d seed=%0d ps=%0.0f",id,seed,$realtime/1ps);
            tick(3);rst_n=1;tick(3);
            if(session_invalid || !idle || next_cookie!==FIRST || exhausted) $fatal(1,"COLD_STATE");
            pool_admit=1;tick(3);
        end
    endtask
    task arm_job(input integer job,input bit channel,input [63:0] cookie);
        begin
            wanted_epoch=64'h100000001+job;wanted_tag=64'h800000010000abcd;
            wanted_address=channel ? 64'h1fffffffc0 : 64'h100000040;
            wanted_channel=channel;wanted_cookie=cookie;wanted_data=pattern(job);
            arm_epoch=wanted_epoch;arm_tag=wanted_tag;arm_destination=wanted_address;arm_channel=channel;
            arm_valid=1;#1;
            if(!arm_ready) $fatal(1,"ARM_NOT_READY");
            tick(1);arm_valid=0;tick(1);
            if(!lease_held || epoch!==wanted_epoch || phase!=1) $fatal(1,"ARM_IDENTITY");
        end
    endtask
    task offer_line;
        begin
            line_epoch=wanted_epoch;line_tag=wanted_tag;line_address=wanted_address;
            line_data=wanted_data;line_valid=1;#1;
            if(!line_ready) $fatal(1,"LINE_NOT_READY");
            tick(1);line_valid=0;
            // All upstream inputs may change after acceptance.
            line_epoch=0;line_tag=0;line_address=0;line_data='1;
            arm_epoch=0;arm_tag=0;arm_destination=0;arm_channel=~wanted_channel;
            tick(2);
        end
    endtask
    task accept_command;
        begin
            if(!cmd_valid || cmd_tag!==wanted_cookie || cmd_addr!==wanted_address ||
               cmd_channel!==wanted_channel || cmd_len!==0) $fatal(1,"COMMAND_CONTEXT");
            tick(3+seed%3);
            cmd_ready=1;tick(1);cmd_ready=0;pool_drained=0;tick(2);
            if(!data_valid || data!==wanted_data) $fatal(1,"DATA_CONTEXT");
        end
    endtask
    task accept_data;
        begin
            tick(2+seed%4);
            if(!data_valid || data!==wanted_data) $fatal(1,"DATA_STABILITY");
            data_ready=1;tick(1);data_ready=0;tick(2);
            if(phase!=4 || done_valid) $fatal(1,"DATA_IS_NOT_COMPLETION");
        end
    endtask
    task completion(input bit channel,input [63:0] cookie,input [1:0] status);
        begin
            cpl_tag[channel*64+:64]=cookie;cpl_status[channel*2+:2]=status;cpl_valid=2'b01<<channel;
            #1;if(!cpl_ready[channel]) $fatal(1,"CPL_NOT_READY");
            tick(1);cpl_valid=0;tick(2);
        end
    endtask
    task release_drain(input [3:0] status,input [1:0] raw_status);
        begin
            tick(4);
            if(done_valid || phase!=5) $fatal(1,"EARLY_DONE_BEFORE_DRAIN");
            pool_drained=1;tick(2);
            if(!done_valid || done_epoch!==wanted_epoch || done_tag!==wanted_tag ||
               done_status!==status || done_pool_status!==raw_status) $fatal(1,"DONE_CONTEXT");
            tick(5+seed%3);
            if(!done_valid || done_status!==status) $fatal(1,"DONE_RETENTION");
        end
    endtask
    task retire_job;
        begin
            done_ready=1;tick(1);done_ready=0;tick(1);
            if(done_valid || !lease_held || phase!=7) $fatal(1,"DONE_IS_NOT_RETIREMENT");
            retire_valid=1;retire_epoch=wanted_epoch;retire_tag=wanted_tag^(64'd1<<40);
            tick(3);if(retire_ready || !lease_held) $fatal(1,"FORGED_RETIRE_TAG");
            retire_tag=wanted_tag;retire_epoch=wanted_epoch^(64'd1<<32);
            tick(3);if(retire_ready || !lease_held) $fatal(1,"FORGED_RETIRE_EPOCH");
            retire_epoch=wanted_epoch;#1;if(!retire_ready) $fatal(1,"RETIRE_NOT_READY");
            tick(1);retire_valid=0;tick(1);
            if(!idle || lease_held) $fatal(1,"RETIRE_DID_NOT_RELEASE");
        end
    endtask
    task normal_job(input integer job,input bit channel,input [63:0] cookie);
        begin
            arm_job(job,channel,cookie);offer_line();accept_command();accept_data();
            completion(channel,cookie,0);release_drain(0,0);retire_job();
        end
    endtask
    task no_new_arm;
        begin
            arm_valid=1;arm_epoch=epoch+1;arm_tag=wanted_tag;arm_destination=0;arm_channel=0;
            tick(3);if(arm_ready || !idle) $fatal(1,"FAULTED_NEW_ARM");arm_valid=0;
        end
    endtask

    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        $display("POOL_BRIDGE_BEGIN seed=%0d first=%h",seed,FIRST);
        #200; // Allow the unisim global startup interval in synthesized replay.
`ifdef NTRPU_CARRY_TEST
        cold(102);normal_job(0,0,FIRST);
        if(exhausted || next_cookie!==64'h0000000100000000) $fatal(1,"COOKIE_CARRY");
        normal_job(1,1,64'h0000000100000000);
        normal_job(2,0,64'h0000000100000001);
        if(exhausted || next_cookie!==64'h0000000100000002) $fatal(1,"COOKIE_CARRY_FINAL");
`else
        cold(1);
        arm_valid=1;arm_tag=64'h800000010000abcd;arm_destination=0;arm_epoch=0;
        tick(2);if(arm_ready) $fatal(1,"ZERO_EPOCH");
        arm_epoch=64'h100000001;arm_destination=64'h2000000000;
        tick(2);if(arm_ready) $fatal(1,"HIGH_RANGE_ALIAS");
        arm_destination=1;tick(2);if(arm_ready) $fatal(1,"MISALIGNED_ARM");
        arm_destination=0;pool_drained=0;tick(2);if(arm_ready) $fatal(1,"UNDRAINED_ARM");
        pool_drained=1;stop=1;tick(2);if(arm_ready) $fatal(1,"STOP_ARM");arm_valid=0;stop=0;
        normal_job(0,0,FIRST);normal_job(1,1,FIRST+1);
        // Same external tag in a later epoch must have a different cookie.
        arm_job(2,0,FIRST+2);offer_line();accept_command();accept_data();
        completion(0,FIRST,0);if(phase!=4 || done_valid || !protocol_fault) $fatal(1,"STALE_CPL_RELEASE");
        completion(0,FIRST+2,0);release_drain(5,0);retire_job();no_new_arm();

        cold(2);arm_job(0,0,FIRST);stop=1;tick(2);stop=0;
        retire_epoch=wanted_epoch;retire_tag=wanted_tag;retire_valid=1;pool_drained=0;
        tick(3);if(retire_ready || !lease_held || cmd_valid || done_valid) $fatal(1,"PARTIAL_STOP_DRAIN");
        pool_drained=1;#1;if(!retire_ready) $fatal(1,"STICKY_PARTIAL_STOP");
        tick(1);retire_valid=0;tick(2);
        if(!idle || next_cookie!==FIRST) $fatal(1,"PARTIAL_STOP_COOKIE");
        normal_job(1,1,FIRST);

        // STOP at exposed line, command, data, completion and drain boundaries.
        for(integer step=0;step<5;step++) begin
            cold(10+step);arm_job(0,step%2,FIRST);
            if(step==0) stop=1;
            offer_line();if(step==1) stop=1;
            accept_command();if(step==2) stop=1;
            accept_data();if(step==3) stop=1;
            completion(wanted_channel,FIRST,0);if(step==4) stop=1;
            release_drain(1,0);retire_job();stop=0;tick(2);normal_job(1,~wanted_channel,FIRST+1);
        end

        cold(20);arm_job(0,0,FIRST);offer_line();accept_command();accept_data();
        completion(0,FIRST,0);release_drain(0,0);stop=1;tick(4);
        if(!done_valid || done_status!==0) $fatal(1,"EXPOSED_DONE_MUTATED");retire_job();

        for(integer bad=0;bad<3;bad++) begin
            cold(30+bad);arm_job(0,0,FIRST);
            line_epoch=wanted_epoch;line_tag=wanted_tag;line_address=wanted_address;line_data=wanted_data;
            case(bad) 0:line_epoch=line_epoch^(64'd1<<32);1:line_tag=line_tag^(64'd1<<40);2:line_address=line_address^(64'd1<<37);endcase
            line_valid=1;tick(1);line_valid=0;tick(2);
            if(!done_valid || done_status!=2 || cmd_valid || next_cookie!==FIRST || !protocol_fault) $fatal(1,"BAD_LINE_ACCEPTED");
            retire_job();no_new_arm();
        end

        cold(40);completion(1,64'hdeadbeef,0);
        if(!protocol_fault || rejected_completions!=1) $fatal(1,"ORPHAN_CPL");no_new_arm();
        for(integer bad=0;bad<4;bad++) begin
            cold(41+bad);arm_job(0,0,FIRST);offer_line();
            if(bad==2) begin completion(0,FIRST,0);if(phase!=2 || !cmd_valid) $fatal(1,"EARLY_COMMAND_CPL");end
            accept_command();
            if(bad==3) begin completion(0,FIRST,0);if(phase!=3 || !data_valid) $fatal(1,"EARLY_DATA_CPL");end
            accept_data();
            if(bad==0) completion(1,FIRST,0);
            if(bad==1) completion(0,FIRST^(64'd1<<40),0);
            if(done_valid || phase!=4 || !protocol_fault) $fatal(1,"FOREIGN_CPL_RELEASE");
            completion(0,FIRST,0);release_drain(5,0);retire_job();no_new_arm();
        end

        for(integer response=1;response<4;response++) begin
            cold(50+response);arm_job(0,0,FIRST);offer_line();accept_command();accept_data();
            pool_fault=1;completion(0,FIRST,2'(response));pool_fault=0;
            release_drain(4,2'(response));retire_job();no_new_arm();
        end
        cold(55);arm_job(0,0,FIRST);offer_line();accept_command();pool_fault=1;tick(2);pool_fault=0;
        accept_data();completion(0,FIRST,0);release_drain(5,0);retire_job();no_new_arm();

        for(integer both=0;both<2;both++) begin
            cold(60+both);arm_job(0,0,FIRST);offer_line();accept_command();accept_data();
            cpl_valid=3;cpl_tag={FIRST,FIRST+64'(both)};cpl_status=0;tick(1);cpl_valid=0;tick(2);
            if(rejected_completions!=(both ? 2 : 1) || !protocol_fault) $fatal(1,"DUAL_CPL_ACCOUNTING");
            if(both) begin if(phase!=4) $fatal(1,"DUAL_FOREIGN_RELEASE");completion(0,FIRST,0);end
            release_drain(5,0);retire_job();no_new_arm();
        end

        for(integer lost=0;lost<7;lost++) begin
            cold(70+lost);arm_job(0,0,FIRST);
            if(lost>0) offer_line();
            if(lost>1) accept_command();
            if(lost>2) accept_data();
            if(lost>3) completion(0,FIRST,0);
            if(lost>4) release_drain(0,0);
            if(lost==6) begin done_ready=1;tick(1);done_ready=0;tick(1);end
            if(lost%2) pool_restart_required=1;else pool_admit=0;
            #1;if(!session_invalid || retire_ready || line_ready || cpl_ready!=0) $fatal(1,"LOSS_NOT_IMMEDIATE");
            tick(3);
            if(lost==1 && !cmd_valid) $fatal(1,"LOST_COMMAND_RETRACTED");
            if(lost==2 && !data_valid) $fatal(1,"LOST_DATA_RETRACTED");
            pool_admit=1;pool_restart_required=0;pool_drained=1;cmd_ready=1;data_ready=1;
            retire_valid=1;retire_epoch=wanted_epoch;retire_tag=wanted_tag;tick(5);
            if(!session_lost || !session_invalid || !lease_held || retire_ready || arm_ready) $fatal(1,"SESSION_REOPENED");
            if((lost==1 || lost==2) && (phase!=4 || cmd_valid || data_valid)) $fatal(1,"LOST_HANDSHAKE_REPEATED");
            if(lost==5 && (!done_valid || done_status!=0)) $fatal(1,"LOST_DONE_MUTATED");
        end
        cold(90);normal_job(0,0,FIRST); // Both protocol peers start a fresh cold session.
`endif
        $display("POOL_BRIDGE_PASS seed=%0d first=%h cases=%0d arms=%0d lines=%0d commands=%0d data=%0d completions=%0d done=%0d retires=%0d cycles=%0d",seed,FIRST,cases,narm,nline,ncmd,ndata,ncpl,ndone,nretire,cycles);
        $finish;
    end
    always @(posedge clk) begin
        cycles++;
        if(rst_n) begin
            if(arm_valid && arm_ready) narm++;
            if(line_valid && line_ready) nline++;
            if(cmd_valid && cmd_ready) ncmd++;
            if(data_valid && data_ready) ndata++;
            for(integer ch=0;ch<2;ch++) if(cpl_valid[ch] && cpl_ready[ch]) ncpl++;
            if(done_valid && done_ready) ndone++;
            if(retire_valid && retire_ready) nretire++;
        end
        $display("POOL_BRIDGE_TRACE cycle=%0d case=%0d rst=%b stop=%b admit=%b restart=%b pfault=%b drained=%b av=%b ar=%b ae=%h at=%h aa=%h ac=%b rv=%b rr=%b re=%h rt=%h lv=%b lr=%b le=%h lt=%h la=%h ld=%h cv=%b cr=%b cc=%b ca=%h ct=%h cl=%h dv=%b dr=%b data=%h pv=%h pr=%h pt=%h ps=%h ov=%b ore=%b oe=%h ot=%h os=%h ops=%h epoch=%h held=%b fault=%b lost=%b invalid=%b exhausted=%b idle=%b phase=%h next=%h rejected=%h",cycles,case_id,rst_n,stop,pool_admit,pool_restart_required,pool_fault,pool_drained,arm_valid,arm_ready,arm_epoch,arm_tag,arm_destination,arm_channel,retire_valid,retire_ready,retire_epoch,retire_tag,line_valid,line_ready,line_epoch,line_tag,line_address,line_data,cmd_valid,cmd_ready,cmd_channel,cmd_addr,cmd_tag,cmd_len,data_valid,data_ready,data,cpl_valid,cpl_ready,cpl_tag,cpl_status,done_valid,done_ready,done_epoch,done_tag,done_status,done_pool_status,epoch,lease_held,protocol_fault,session_lost,session_invalid,exhausted,idle,phase,next_cookie,rejected_completions);
    end
    initial begin #100000;$fatal(1,"POOL_BRIDGE_TIMEOUT");end
endmodule
