`timescale 1ns/1ps
module tb_ntrpu_xdma_local_master_replay #(parameter integer LIMIT=32);
    reg clk=0;
    always #2 clk=~clk;
    reg cold_rst_n=0,admit=0,req_valid=0,req_write=0,rsp_ready=0;
    reg [31:0] req_address=0,req_data=0;
    reg [3:0] req_strobe=0;
    reg [127:0] req_identity=0;
    wire req_ready,rsp_valid,rsp_timeout,timeout_latched,busy;
    wire [31:0] rsp_data,m_axil_awaddr,m_axil_wdata,m_axil_araddr;
    wire [1:0] rsp_code;
    wire [127:0] rsp_identity;
    wire [2:0] m_axil_awprot,m_axil_arprot;
    wire [3:0] m_axil_wstrb;
    wire m_axil_awvalid,m_axil_wvalid,m_axil_bready,m_axil_arvalid,m_axil_rready;
    reg m_axil_awready=0,m_axil_wready=0,m_axil_bvalid=0,m_axil_arready=0,m_axil_rvalid=0;
    reg [1:0] m_axil_bresp=0,m_axil_rresp=0;
    reg [31:0] m_axil_rdata=0;
    integer phase=0,cases=0;
`ifdef NTRPU_GATE_NETLIST
    localparam integer NETLIST=1;
    ntrpu_xdma_local_master dut(.*);
`else
    localparam integer NETLIST=0;
    ntrpu_xdma_local_master #(.WAIT_LIMIT(LIMIT)) dut(.*);
`endif

    task require(input bit condition,input string message);
        if(!condition) $fatal(1,"LOCAL_MASTER_ASSERT phase=%0d %s",phase,message);
    endtask
    task cycles(input integer count);
        repeat(count) @(negedge clk);
    endtask
    task configure;
        begin
            @(negedge clk);cold_rst_n=0;admit=0;req_valid=0;rsp_ready=0;
            m_axil_awready=0;m_axil_wready=0;m_axil_bvalid=0;m_axil_arready=0;m_axil_rvalid=0;
            cycles(3);cold_rst_n=1;admit=1;cycles(1);
            require(req_ready && !busy && !timeout_latched,"fresh configuration life");
        end
    endtask
    task submit(input bit write_request,input [31:0] address,input [31:0] data,input [3:0] strobe);
        begin
            @(negedge clk);req_valid=1;req_write=write_request;
            req_address=address;req_data=data;req_strobe=strobe;
            req_identity={64'h1020304050607080,64'(phase)};
            @(posedge clk);require(req_ready,"request accepted once");
            @(negedge clk);req_valid=0;
            // Poison the upstream payload after acceptance: the master must retain its copy.
            req_address=32'hffffffff;req_data=32'hbad0bad0;req_strobe=0;req_identity=0;
        end
    endtask
    task write_aw(input integer delay_cycles);
        begin
            cycles(delay_cycles);m_axil_awready=1;
            do @(posedge clk);while(!m_axil_awvalid);
            @(negedge clk);m_axil_awready=0;
        end
    endtask
    task write_w(input integer delay_cycles);
        begin
            cycles(delay_cycles);m_axil_wready=1;
            do @(posedge clk);while(!m_axil_wvalid);
            @(negedge clk);m_axil_wready=0;
        end
    endtask
    task write_response(input integer delay_cycles,input [1:0] response);
        begin
            cycles(delay_cycles);m_axil_bresp=response;m_axil_bvalid=1;
            do @(posedge clk);while(!m_axil_bready);
            @(negedge clk);m_axil_bvalid=0;
        end
    endtask
    task read_address(input integer delay_cycles);
        begin
            cycles(delay_cycles);m_axil_arready=1;
            do @(posedge clk);while(!m_axil_arvalid);
            @(negedge clk);m_axil_arready=0;
        end
    endtask
    task read_response(input integer delay_cycles,input [31:0] data,input [1:0] response);
        begin
            cycles(delay_cycles);m_axil_rdata=data;m_axil_rresp=response;m_axil_rvalid=1;
            do @(posedge clk);while(!m_axil_rready);
            @(negedge clk);m_axil_rvalid=0;
        end
    endtask
    task receive(input [31:0] data,input [1:0] code,input bit expired,input integer hold_cycles);
        integer j;
        begin
            require(rsp_valid,"actual response published");
            for(j=0;j<hold_cycles+1;j=j+1) begin
                require(rsp_valid && rsp_data===data && rsp_code===code && rsp_timeout===expired &&
                        rsp_identity==={64'h1020304050607080,64'(phase)},"retained response fields");
                require(!req_ready && busy && timeout_latched===expired,"one outstanding request and sticky timeout");
                cycles(1);
            end
            rsp_ready=1;@(posedge clk);require(rsp_valid,"response handshake");
            @(negedge clk);rsp_ready=0;cycles(1);
            require(!rsp_valid && !busy && req_ready===(admit&&!expired),"response consumed, admission condition");
            cases=cases+1;$display("LOCAL_MASTER_CASE t=%0t phase=%0d timeout=%0d code=%h data=%h",$time,phase,expired,code,data);
        end
    endtask
    task expect_timeout;
        begin
            cycles(LIMIT+4);require(timeout_latched && busy && !rsp_valid && !req_ready,"timeout must not invent completion");
            admit=0;cycles(4);require(busy && !rsp_valid,"admission close cannot cancel pending work");
        end
    endtask

    always @(posedge clk) if(cold_rst_n) begin
        if(req_valid&&req_ready) $display("LOCAL_MASTER_REQ t=%0t phase=%0d write=%b addr=%h data=%h strb=%h identity=%h",$time,phase,req_write,req_address,req_data,req_strobe,req_identity);
        if(m_axil_awvalid&&m_axil_awready) $display("LOCAL_MASTER_AW t=%0t phase=%0d addr=%h prot=%h",$time,phase,m_axil_awaddr,m_axil_awprot);
        if(m_axil_wvalid&&m_axil_wready) $display("LOCAL_MASTER_W t=%0t phase=%0d data=%h strb=%h",$time,phase,m_axil_wdata,m_axil_wstrb);
        if(m_axil_bvalid&&m_axil_bready) $display("LOCAL_MASTER_B t=%0t phase=%0d resp=%h",$time,phase,m_axil_bresp);
        if(m_axil_arvalid&&m_axil_arready) $display("LOCAL_MASTER_AR t=%0t phase=%0d addr=%h prot=%h",$time,phase,m_axil_araddr,m_axil_arprot);
        if(m_axil_rvalid&&m_axil_rready) $display("LOCAL_MASTER_R t=%0t phase=%0d data=%h resp=%h",$time,phase,m_axil_rdata,m_axil_rresp);
        if(rsp_valid&&rsp_ready) $display("LOCAL_MASTER_RSP t=%0t phase=%0d data=%h resp=%h identity=%h timeout=%b",$time,phase,rsp_data,rsp_code,rsp_identity,rsp_timeout);
    end
    initial begin
        // Run both variants after the simulator's configuration GSR interval.
        #121;
        $display("LOCAL_MASTER_REPLAY limit=%0d netlist=%0d",LIMIT,NETLIST);
        configure;
        phase=1;submit(0,'h0040,0,0);read_address(4);read_response(3,'h47,0);receive('h47,0,0,3);
        phase=2;submit(1,'h000c,1,4'hf);
        fork write_aw(0);write_w(5);join
        write_response(3,0);receive(0,0,0,2);
        phase=3;submit(1,'h100c,1,4'h1);
        fork write_aw(5);write_w(0);join
        write_response(2,2);receive(0,2,0,2);
        phase=4;submit(1,'h4080,'h12345678,4'ha);
        fork write_aw(0);write_w(0);join
        write_response(1,3);receive(0,3,0,2);
        phase=5;submit(0,'h1048,0,0);admit=0;cycles(3);require(m_axil_arvalid,"AR retained after admission closes");
        read_address(2);read_response(1,32'h89abcdef,2);receive(32'h89abcdef,2,0,LIMIT+8);admit=1;
        configure;phase=6;submit(0,'h0048,0,0);read_address(0);expect_timeout;
        require(m_axil_rready,"R still accepted after timeout");read_response(0,'h21,0);receive('h21,0,1,3);
        configure;phase=7;submit(1,'h000c,1,4'hf);write_aw(0);expect_timeout;
        require(!m_axil_awvalid && m_axil_wvalid,"only unaccepted W remains pending");write_w(0);write_response(0,0);receive(0,0,1,2);
        configure;phase=8;submit(1,'h100c,1,4'hf);write_w(0);expect_timeout;
        require(m_axil_awvalid && !m_axil_wvalid,"only unaccepted AW remains pending");write_aw(0);write_response(0,0);receive(0,0,1,2);
        configure;phase=9;submit(1,'h000c,1,4'hf);
        fork write_aw(0);write_w(0);join
        expect_timeout;require(m_axil_bready,"B still accepted after timeout");write_response(0,2);receive(0,2,1,2);
        configure;phase=10;submit(0,'h1004,0,0);expect_timeout;
        require(m_axil_arvalid,"AR remains offered after timeout");read_address(0);read_response(0,'h00fffe7e,0);receive('h00fffe7e,0,1,2);
        require(cases==10,"all scenarios complete");
        $display("LOCAL_MASTER_PASS cases=10 native_xdma=0 reset_permission=0");$finish;
    end
    initial begin #((8*LIMIT+1000)*4+120);$fatal(1,"LOCAL_MASTER_GLOBAL_TIMEOUT phase=%0d",phase);end
endmodule
