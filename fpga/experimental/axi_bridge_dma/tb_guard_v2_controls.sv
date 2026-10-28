`timescale 1ns/1ps
`default_nettype none
module tb_guard_v2_controls;
    reg mgmt_clk=0,axi_clk=0,clock_enable=1,cold_rst_n=0;
    reg [3:0] native_status=0;
    wire client_cold_rst_n,client_link_up,client_stop,clock_alive,armed,fault;
    wire [1:0] fault_code;
    integer case_id=0,input_id=0,stop_kind=0;
    reg loss_window=0;
    ntrpu_axi_bridge_guard_v2 #(.CLOCK_TIMEOUT(32)) guard(
        .mgmt_clk(mgmt_clk),.axi_clk(axi_clk),.cold_rst_n(cold_rst_n),
        .native_link_up(native_status[0]),.native_axi_reset_n(native_status[1]),
        .native_ctl_reset_n(native_status[2]),.perst_n(native_status[3]),
        .client_cold_rst_n(client_cold_rst_n),.client_link_up(client_link_up),
        .client_stop(client_stop),.clock_alive(clock_alive),.armed(armed),
        .fault(fault),.fault_code(fault_code));
    always #7 mgmt_clk=~mgmt_clk;
    initial begin #1;forever #2 if(clock_enable)axi_clk=~axi_clk;end
    task require(input bit ok,input string message);
        if(ok!==1'b1)$fatal(1,"V2_CONTROL_CHECK case=%0d %s",case_id,message);
    endtask
    // An asynchronously closed permit must never reopen between a brief
    // input recovery and the management-domain sticky fault.
    always @(client_link_up)if(loss_window && client_link_up)
        $fatal(1,"V2_CONTROL_REOPEN case=%0d",case_id);
    initial begin
        if($value$plusargs("case=%d",case_id))begin end
        require(case_id>=0 && case_id<16,"valid case");
        input_id=case_id<12 ? case_id/3 : case_id-12;
        stop_kind=case_id<12 ? case_id%3 : 0;
        repeat(8)@(negedge mgmt_clk);cold_rst_n=1;native_status=4'b1111;
        wait(client_link_up);repeat(4)@(negedge axi_clk);
        require(armed && !fault && !client_stop && client_cold_rst_n,"qualified startup");
        if(stop_kind==1)begin @(negedge axi_clk);#0.001;clock_enable=0;end
        if(stop_kind==2)begin @(posedge axi_clk);#0.001;clock_enable=0;end
        @(negedge mgmt_clk);#0.001;
        native_status[input_id]=0;
        $display("V2 CONTROL_DROP case=%0d input=%0d stop=%0d t=%0t",case_id,input_id,stop_kind,$realtime);
        #0.010;
        require(!client_link_up && client_cold_rst_n,"closure before any next clock edge");
        loss_window=1;
        #0.040;
        if(case_id>=12)native_status[(input_id+1)%4]=0;
        #0.050;native_status[input_id]=1;
        if(case_id>=12)begin #0.050;native_status[(input_id+1)%4]=1;end
        repeat(48)@(negedge mgmt_clk);
        require(fault && client_stop && !client_link_up && client_cold_rst_n,"retained native fault");
        require(fault_code==(stop_kind==0 ? 1 : 3),"attributed fault cause");
        $display("V2 CONTROL_HELD case=%0d t=%0t",case_id,$realtime);
        clock_enable=1;repeat(64)@(negedge mgmt_clk);
        require(clock_alive && fault && client_stop && !client_link_up,"no rearm after clock recovery");
        $display("V2_CONTROL_PASS case=%0d input=%0d stop=%0d code=%0d",case_id,input_id,stop_kind,fault_code);
        $finish;
    end
    initial begin #100000;$fatal(1,"V2_CONTROL_TIMEOUT");end
endmodule
`default_nettype wire
