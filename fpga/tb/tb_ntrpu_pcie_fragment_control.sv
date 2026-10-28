`timescale 1ns / 1ps
// Control-unit contract test. The guarded ingress and transport are NOT modeled
// as physical PCIe/MIG here. Readiness is deliberately withheld by this peer.
module tb_ntrpu_pcie_fragment_control;
    reg clk=0;always #2.5 clk=~clk;
    reg rst_n=0;
    reg [31:0] s_axil_awaddr=0,s_axil_wdata=0,s_axil_araddr=0;
    reg [3:0] s_axil_wstrb=0;
    reg s_axil_awvalid=0,s_axil_wvalid=0,s_axil_bready=0,s_axil_arvalid=0,s_axil_rready=0;
    wire s_axil_awready,s_axil_wready,s_axil_arready,s_axil_bvalid,s_axil_rvalid;
    wire [1:0] s_axil_bresp,s_axil_rresp;
    wire [31:0] s_axil_rdata;
    wire arm_valid,stop,result_ready,irq,protocol_fault;
    reg arm_ready=0;
    wire [63:0] arm_epoch,arm_tag,arm_window,arm_destination;
    reg [63:0] epoch=0,valid_bytes=0,quiesce_epoch=0;
    reg lease_held=0,failed=0,transport_idle=1,guard_idle=1,peer_axi_busy=0;
    reg [3:0] error_code=0;
    reg source_stopped=0,ingress_isolated=0,result_valid=0;
    reg [63:0] result_epoch=0,result_tag=0;
    reg [3:0] result_status=0;
    reg [31:0] rejected_frames=0,rejected_identity=0;
    reg [3:0] last_reject_code=0;
    reg [6:0] last_reject_faults=0;
    integer telemetry=0;
    integer cycle=0,checks=0,writes=0,reads=0,arms=0,retires=0,seed=1;
    integer cases=0,b_hold=0,r_hold=0,arm_hold=0;
    reg prev_b=0,prev_r=0,prev_arm=0;
    reg [1:0] saved_b;
    reg [33:0] saved_r;
    reg [255:0] saved_arm;
    ntrpu_pcie_fragment_control dut(.*);
    task automatic need(input bit yes,input string message);
        checks=checks+1;
        if(!yes) $fatal(1,"CONTROL_CHECK case=%0d cycle=%0d %s",cases,cycle,message);
    endtask
    task automatic idle(input integer n);repeat(n) @(negedge clk);endtask
    task automatic scenario(input string name);
        cases=cases+1;$display("NTRPU_CONTROL_CASE id=%0d name=%s",cases,name);
    endtask
    task automatic aw(input [31:0] address);
        @(negedge clk);s_axil_awaddr=address;s_axil_awvalid=1;
        do @(posedge clk);while(!s_axil_awready);
        @(negedge clk);s_axil_awvalid=0;s_axil_awaddr=32'hdeadbeef;
    endtask
    task automatic wd(input [31:0] data,input [3:0] strb);
        @(negedge clk);s_axil_wdata=data;s_axil_wstrb=strb;s_axil_wvalid=1;
        do @(posedge clk);while(!s_axil_wready);
        @(negedge clk);s_axil_wvalid=0;s_axil_wdata=32'hffffffff;s_axil_wstrb=0;
    endtask
    task automatic wr(input [31:0] address,input [31:0] data,input [3:0] strb=15,input [1:0] expected=0);
        integer order;
        order=(writes+seed)%3;
        fork
            begin if(order==1) idle(3);aw(address);end
            begin if(order==0) idle(4);wd(data,strb);end
        join
        while(!s_axil_bvalid) @(negedge clk);
        need(s_axil_bresp===expected,"WRITE_RESPONSE");
        idle(2+seed%3);s_axil_bready=1;
        @(posedge clk);@(negedge clk);s_axil_bready=0;
        writes=writes+1;
        $display("NTRPU_CONTROL_WRITE n=%0d addr=%h data=%h strb=%h resp=%0d cycle=%0d",writes,address,data,strb,expected,cycle);
    endtask
    task automatic rd(input [31:0] address,input [31:0] expected,input [1:0] response=0,input integer hold_cycles=0);
        @(negedge clk);s_axil_araddr=address;s_axil_arvalid=1;
        do @(posedge clk);while(!s_axil_arready);
        @(negedge clk);s_axil_arvalid=0;s_axil_araddr=32'hdeadbeef;
        while(!s_axil_rvalid) @(negedge clk);
        need(s_axil_rresp===response && s_axil_rdata===expected,"READ_RESPONSE");
        idle(hold_cycles>0 ? hold_cycles : 3+seed%4);s_axil_rready=1;
        @(posedge clk);@(negedge clk);s_axil_rready=0;
        reads=reads+1;
        $display("NTRPU_CONTROL_READ n=%0d addr=%h data=%h resp=%0d cycle=%0d",reads,address,expected,response,cycle);
    endtask
    task automatic live(input [31:0] frames,input [31:0] identity,input [3:0] code,input [6:0] faults);
        rejected_frames=frames;rejected_identity=identity;last_reject_code=code;last_reject_faults=faults;
        telemetry=telemetry+1;
        $display("NTRPU_CONTROL_TELEMETRY n=%0d frames=%h identity=%h code=%h faults=%h cycle=%0d",telemetry,frames,identity,code,faults,cycle);
    endtask
    task automatic desc(input [63:0] e,input [63:0] t,input [63:0] w,input [63:0] d);
        wr('h20,e[31:0]);wr('h24,e[63:32]);wr('h28,t[31:0]);wr('h2c,t[63:32]);
        wr('h30,w[31:0]);wr('h34,w[63:32]);wr('h38,d[31:0]);wr('h3c,d[63:32]);
    endtask
    task automatic ack(input [63:0] e,input [63:0] t);
        wr('h40,e[31:0]);wr('h44,e[63:32]);wr('h48,t[31:0]);wr('h4c,t[63:32]);
    endtask
    task automatic accept_arm;
        need(arm_valid,"ARM_NOT_QUEUED");arm_ready=1;
        @(posedge clk);@(negedge clk);arm_ready=0;
        need(!arm_valid,"DUPLICATE_ARM");
    endtask
    task automatic completion(input [3:0] status);
        @(negedge clk);result_epoch=epoch;result_tag=arm_tag;result_status=status;result_valid=1;
        idle(3);
    endtask
    always @(posedge clk) if(rst_n) begin
        cycle=cycle+1;
        if(prev_b) need(s_axil_bvalid && s_axil_bresp===saved_b,"B_STABLE");
        if(prev_r) need(s_axil_rvalid && {s_axil_rresp,s_axil_rdata}===saved_r,"R_STABLE");
        if(prev_arm) need(arm_valid && {arm_epoch,arm_tag,arm_window,arm_destination}===saved_arm,"ARM_STABLE");
        prev_b=s_axil_bvalid && !s_axil_bready;saved_b=s_axil_bresp;
        prev_r=s_axil_rvalid && !s_axil_rready;saved_r={s_axil_rresp,s_axil_rdata};
        prev_arm=arm_valid && !arm_ready;saved_arm={arm_epoch,arm_tag,arm_window,arm_destination};
        if(prev_b) b_hold=b_hold+1;if(prev_r) r_hold=r_hold+1;if(prev_arm) arm_hold=arm_hold+1;
        if(arm_valid && arm_ready) begin
            arms=arms+1;epoch<=arm_epoch;lease_held<=1;
            $display("NTRPU_CONTROL_ARM n=%0d epoch=%h tag=%h window=%h dest=%h cycle=%0d",arms,arm_epoch,arm_tag,arm_window,arm_destination,cycle);
        end
        if(result_ready) begin
            need(result_valid,"ACK_WITHOUT_RESULT");retires=retires+1;result_valid<=0;lease_held<=0;
            $display("NTRPU_CONTROL_RETIRE n=%0d epoch=%h tag=%h status=%0d cycle=%0d",retires,result_epoch,result_tag,result_status,cycle);
        end
    end
    initial begin
        if(!$value$plusargs("SEED=%d",seed)) seed=1;
        idle(40);rst_n=1;idle(3);
        scenario("register_identity_and_decode");
        rd(0,32'h4652544e);rd(4,'h10000);rd(8,'h1f);rd('hb0,0,3);rd('h10000000,0,3);rd(1,0,3);
        wr(0,1,15,2);wr('hb0,1,15,3);wr('h10000020,1,15,3);wr('h21,1,15,3);
        scenario("fragment_profile_and_readonly_telemetry");
        rd('h90,0);rd('h94,0);rd('h98,0);rd('h9c,'h5052544e);rd('ha0,'h31304655);
        rd('ha4,64);rd('ha8,32);rd('hac,0);
        for(integer a='h90;a<='hac;a=a+4) wr(a,'hffffffff,15,2);
        rd('h94,0);rd('hac,0);
        scenario("atomic_snapshot_and_response_stability");
        live(32'h80000001,32'h40000002,7,7'h30);
        wr('h10,8,0);rd('hac,0);rd('h90,0);
        wr('h10,8);rd('hac,1);rd('h90,'h80000001);rd('h94,'h40000002);rd('h98,'h307);
        fork
            rd('h90,'h80000001,0,40);
            begin
                do @(posedge clk);while(!(s_axil_arvalid && s_axil_arready));
                @(negedge clk);live(32'hffffffff,32'h80000004,2,7'h42);wr('h10,8);
            end
        join
        rd('h90,'hffffffff);rd('h94,'h80000004);rd('h98,'h422);
        live(9,5,4,0);rd('h90,'hffffffff);rd('h94,'h80000004);rd('h98,'h422);
        need(!arm_valid && !result_ready && arms==0 && retires==0,"DIAGNOSTIC_SIDE_EFFECT");
        scenario("strobe_and_command_validation");
        wr('h20,32'h11223344);wr('h20,32'haabbccdd,5);rd('h20,32'h11bb33dd);
        wr('h20,0,0);rd('h20,32'h11bb33dd);wr('h10,32'hffffffff,0);wr('h10,3,15,2);
        wr('h14,2,15,2);wr('h14,1,1);wr('h14,0,2);rd('h14,1);
        scenario("reject_invalid_arm");
        desc(0,1,'h100000000,'h100000040);wr('h10,1,15,2);
        wr('h20,1);wr('h30,1);wr('h10,1,15,2);wr('h30,0);wr('h38,'h41);wr('h10,1,15,2);
        need(!arm_valid && arms==0,"INVALID_ARM");
        scenario("immutable_pending_arm");
        desc(64'h100000001,64'h8877665544332211,64'h1000fffc0,64'h1fffffffc0);
        wr('h10,1);need(arm_valid && !stop,"ARM_SUBMIT");
        wr('h20,7,15,2);wr('h2c,0,15,2);wr('h10,1,15,2);wr('h10,2,15,2);wr('h10,4,15,2);
        need(arm_epoch==64'h100000001 && arm_tag==64'h8877665544332211 &&
             arm_window==64'h1000fffc0 && arm_destination==64'h1fffffffc0,"SNAPSHOT_64");
        scenario("barrier_is_read_only_and_not_synthesized");
        wr('h78,3,15,2);wr('h7c,1,15,2);wr('h80,1,15,2);
        source_stopped=1;quiesce_epoch=64'hffffffff00000001;
        rd('h78,1);rd('h7c,1);rd('h80,'hffffffff);
        idle(100);need(arm_valid && arms==0 && !result_ready,"READINESS_WITHHELD");
        ingress_isolated=1;rd('h78,3);need(arms==0,"DIAGNOSTICS_ARE_NOT_READY");
        accept_arm();rd('h18,1);rd('h84,1);rd('h88,1);
        scenario("stop_keeps_outstanding_lease");
        peer_axi_busy=1;guard_idle=0;transport_idle=0;
        wr('h10,2);need(stop,"STOP_NOT_LATCHED");wr('h10,2,15,2);
        wr('h10,4,15,2);wr('h10,1,15,2);wr('h38,0,15,2);
        idle(100);need(stop && lease_held && retires==0 && arms==1,"STOP_PREMATURE_RETIRE");
        peer_axi_busy=0;guard_idle=1;transport_idle=1;
        scenario("identity_checked_retirement");
        completion(1);need(irq,"RESULT_IRQ");rd('h60,1);rd('h64,1);rd('h68,'h44332211);rd('h6c,'h88776655);rd('h70,1);
        wr('h10,4,15,2);ack(1,arm_tag);wr('h10,4,15,2);
        ack(epoch,64'h0000000044332211);wr('h10,4,15,2);need(retires==0 && result_valid,"STALE_ACK");
        ack(epoch,arm_tag);wr('h10,4);need(retires==1 && !result_valid && !irq,"ACK_RESULT");
        wr('h10,4,15,2);rd('h1c,1);
        scenario("recovery_without_reset");
        desc(epoch,2,'h100000000,0);wr('h10,1,15,2);
        desc(epoch+1,64'hffffffff00000002,'h100000000,0);wr('h10,1);need(!stop,"STOP_NOT_CLEARED_ON_NEW_ARM");
        accept_arm();completion(0);ack(epoch,arm_tag);wr('h10,4);rd('h18,2);rd('h1c,2);
        scenario("epoch_exhaustion_no_wrap");
        desc(64'hffffffffffffffff,3,'h100000000,0);wr('h10,1);accept_arm();completion(0);ack(epoch,arm_tag);wr('h10,4);
        desc(0,4,'h100000000,0);wr('h10,1,15,2);need(!arm_valid,"EPOCH_WRAP");
        scenario("unexpected_result_fail_closed");
        result_epoch=0;result_tag=0;result_valid=1;idle(4);
        need(protocol_fault && irq,"UNEXPECTED_RESULT_NOT_FAULTED");rd('h8c,1);
        wr('h10,1,15,2);wr('h10,4,15,2);wr('h8c,0,15,2);wr('h20,1,15,2);
        result_valid=0;idle(4);need(protocol_fault,"FAULT_NOT_STICKY");
        wr('h10,8);rd('h90,9);rd('h94,5);rd('h98,4);rd('hac,1);
        need(protocol_fault && telemetry==3,"DIAGNOSTIC_CLEARED_FAULT");
        need(arms==3 && retires==3 && b_hold>0 && r_hold>0 && arm_hold>=100,"COUNTS");
        $display("NTRPU_CONTROL_PASS cases=%0d writes=%0d reads=%0d arms=%0d retires=%0d checks=%0d cycles=%0d b_hold=%0d r_hold=%0d arm_hold=%0d seed=%0d",cases,writes,reads,arms,retires,checks,cycle,b_hold,r_hold,arm_hold,seed);
        $finish;
    end
    initial begin #100000; $fatal(1,"CONTROL_TIMEOUT case=%0d",cases);end
endmodule
