`timescale 1ns/1ps
`default_nettype none
module tb_guard_v3_topology;
    reg mgmt_clk=0,axi_clk=0,cold_rst_n=0,axi_n=0,ctl_n=0,link=0,perst=0;
    integer bad=0,axi_half=2,mgmt_half=5,phase=1;
    wire shared_ctl=bad ? ctl_n : axi_n;
    wire dc,dl,ds,da,dr,df,sc,sl,ss,sa,sr,sf;
    wire rc,rl,rs,ra,rr,rf;
    wire [1:0] dcode,scode,rcode;
    ntrpu_axi_bridge_guard_v3 distinct_guard(
        .mgmt_clk(mgmt_clk),.axi_clk(axi_clk),.cold_rst_n(cold_rst_n),
        .native_link_up(link),.native_axi_reset_n(axi_n),.native_ctl_reset_n(ctl_n),.perst_n(perst),
        .client_cold_rst_n(dc),.client_link_up(dl),.client_stop(ds),
        .clock_alive(da),.armed(dr),.fault(df),.fault_code(dcode));
    ntrpu_axi_bridge_guard_v2 reference_guard(
        .mgmt_clk(mgmt_clk),.axi_clk(axi_clk),.cold_rst_n(cold_rst_n),
        .native_link_up(link),.native_axi_reset_n(axi_n),.native_ctl_reset_n(ctl_n),.perst_n(perst),
        .client_cold_rst_n(rc),.client_link_up(rl),.client_stop(rs),
        .clock_alive(ra),.armed(rr),.fault(rf),.fault_code(rcode));
    ntrpu_axi_bridge_guard_v3 #(.SHARED_NATIVE_RESET(1)) shared_guard(
        .mgmt_clk(mgmt_clk),.axi_clk(axi_clk),.cold_rst_n(cold_rst_n),
        .native_link_up(link),.native_axi_reset_n(axi_n),.native_ctl_reset_n(shared_ctl),.perst_n(perst),
        .client_cold_rst_n(sc),.client_link_up(sl),.client_stop(ss),
        .clock_alive(sa),.armed(sr),.fault(sf),.fault_code(scode));
    initial begin
        if($value$plusargs("bad=%d",bad))begin end
        if($value$plusargs("axi_half=%d",axi_half))begin end
        if($value$plusargs("mgmt_half=%d",mgmt_half))begin end
        if($value$plusargs("phase=%d",phase))begin end
        forever #(mgmt_half)mgmt_clk=~mgmt_clk;
    end
    initial begin #0.001;#(phase);forever #(axi_half)axi_clk=~axi_clk;end
    always @(mgmt_clk or axi_clk)begin
        #0.001;
        if({dc,dl,ds,da,dr,df,dcode} !== {rc,rl,rs,ra,rr,rf,rcode})
            $fatal(1,"V3_DISTINCT_REFERENCE_MISMATCH");
    end
    task require(input bit ok,input string message);
        if(ok!==1'b1)$fatal(1,"V3_TOPOLOGY %s",message);
    endtask
    initial begin
        repeat(8)@(negedge mgmt_clk);cold_rst_n=1;link=1;perst=1;axi_n=1;
        repeat(96)@(negedge mgmt_clk);
        require(!bad,"invalid shared wiring was not rejected");
        require(sl && !sf && !dl && !dr && !df,"distinct control reset blocks startup independently");
        $display("V3 STAGGERED distinct_link=%b shared_link=%b",dl,sl);
        ctl_n=1;wait(dl);repeat(8)@(negedge mgmt_clk);
        #0.001;ctl_n=0;#0.010;
        require(!dl && sl && dc && sc,"control-only pulse affects distinct profile only");
        #0.040;ctl_n=1;repeat(8)@(negedge mgmt_clk);
        require(df && ds && !sf && !ss,"distinct fault is retained");
        #0.001;axi_n=0;#0.010;
        require(!sl && sc,"shared source closes asynchronously without cold reset");
        #0.040;axi_n=1;repeat(96)@(negedge mgmt_clk);
        require(sf && ss && !sl && df && ds && !dl,"both faults remain sticky after recovery");
        $display("V3_TOPOLOGY_PASS axi_half=%0d mgmt_half=%0d phase=%0d",axi_half,mgmt_half,phase);
        $finish;
    end
    initial begin #100000;$fatal(1,"V3_TOPOLOGY_TIMEOUT");end
endmodule
`default_nettype wire
