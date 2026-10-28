`timescale 1ns/1ps
`default_nettype none
// Local guard + unchanged real writer. Mapping, DDR/HBM and ACK are fixtures.
module tb_ntrpu_axi_bridge_guard;
    reg mgmt_clk=0,axi_clk=0,clock_enable=1,cold_rst_n=0;
    integer test_id=0,axi_half=2,mgmt_half=5,phase_ns=1;
    reg native_link_up=0,native_axi_reset_n=0,native_ctl_reset_n=0,perst_n=0;
    wire client_cold_rst_n,client_link_up,client_stop,clock_alive,armed,guard_fault;
    wire [1:0] guard_code;
    ntrpu_axi_bridge_guard #(.CLOCK_TIMEOUT(32),.QUALIFY_CYCLES(4)) guard(
        .mgmt_clk(mgmt_clk),.axi_clk(axi_clk),.cold_rst_n(cold_rst_n),
        .native_link_up(native_link_up),.native_axi_reset_n(native_axi_reset_n),
        .native_ctl_reset_n(native_ctl_reset_n),.perst_n(perst_n),
        .client_cold_rst_n(client_cold_rst_n),.client_link_up(client_link_up),
        .client_stop(client_stop),.clock_alive(clock_alive),.armed(armed),
        .fault(guard_fault),.fault_code(guard_code));
    initial begin
        if($value$plusargs("axi_half=%d",axi_half))begin end
        if($value$plusargs("mgmt_half=%d",mgmt_half))begin end
        if($value$plusargs("phase=%d",phase_ns))begin end
        if($value$plusargs("case=%d",test_id))begin end
        if(test_id==12)clock_enable=0;
        #1;
        forever #(mgmt_half)mgmt_clk=~mgmt_clk;
    end
    initial begin #0.001;#(phase_ns);forever #(axi_half)if(clock_enable)axi_clk=~axi_clk;end

    reg cfg_valid=0,cfg_rsp_ready=0,cmd_valid=0,src_req_ready=0;
    wire cfg_ready,cfg_rsp_valid,cfg_rsp_error,cmd_ready,map_valid;
    reg map_rsp_valid=0;
    wire map_rsp_ready;
    wire [63:0] map_session,map_generation,map_axi_base,map_pcie_base;
    wire src_req_valid,src_ready,src_cpl_ready;
    wire [127:0] src_req_tag;
    wire [63:0] src_req_address;
    wire [12:0] src_req_bytes;
    reg src_valid=0,src_last=0,src_cpl_valid=0;
    reg [511:0] src_data=0;
    wire [47:0] m_awaddr;
    wire [7:0] m_awlen;
    wire [3:0] m_awid;
    wire [2:0] m_awsize;
    wire [1:0] m_awburst;
    wire m_awvalid,m_wvalid,m_wlast,m_bready,ack_ready,result_valid,owned,fault,window_held;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    wire [7:0] fault_code;
    wire [127:0] result_tag;
    wire [12:0] result_bytes;
    reg m_awready=0,m_wready=0,m_bvalid=0,ack_valid=0,result_ready=0;
    localparam [127:0] TAG={64'd9,64'd1};
    ntrpu_axi_bridge_writer writer(
        .clk(axi_clk),.cold_rst_n(client_cold_rst_n),.link_up(client_link_up),.stop(client_stop),
        .cfg_valid(cfg_valid),.cfg_ready(cfg_ready),.cfg_session(64'd9),.cfg_generation(64'd1),
        .cfg_axi_base(64'h10000000),.cfg_pcie_base(64'h1234567800001000),
        .cfg_rsp_valid(cfg_rsp_valid),.cfg_rsp_ready(cfg_rsp_ready),.cfg_rsp_error(cfg_rsp_error),
        .map_valid(map_valid),.map_ready(1'b1),.map_session(map_session),.map_generation(map_generation),
        .map_axi_base(map_axi_base),.map_pcie_base(map_pcie_base),
        .map_rsp_valid(map_rsp_valid),.map_rsp_ok(1'b1),.map_rsp_ready(map_rsp_ready),
        .cmd_valid(cmd_valid),.cmd_ready(cmd_ready),.cmd_tag(TAG),.cmd_generation(64'd1),
        .cmd_source(64'h1000),.cmd_offset(64'd0),.cmd_bytes(13'd128),
        .reject_valid(),.reject_ready(1'b0),.reject_code(),.reject_tag(),
        .src_req_valid(src_req_valid),.src_req_ready(src_req_ready),.src_req_tag(src_req_tag),
        .src_req_address(src_req_address),.src_req_bytes(src_req_bytes),
        .src_valid(src_valid),.src_last(src_last),.src_ready(src_ready),.src_data(src_data),.src_tag(TAG),
        .src_cpl_valid(src_cpl_valid),.src_cpl_ready(src_cpl_ready),.src_cpl_tag(TAG),
        .src_cpl_bytes(13'd128),.src_cpl_status(2'd0),
        .m_awaddr(m_awaddr),.m_awid(m_awid),.m_awlen(m_awlen),.m_awsize(m_awsize),.m_awburst(m_awburst),
        .m_awvalid(m_awvalid),.m_awready(m_awready),.m_wdata(m_wdata),.m_wstrb(m_wstrb),
        .m_wlast(m_wlast),.m_wvalid(m_wvalid),.m_wready(m_wready),
        .m_bvalid(m_bvalid),.m_bready(m_bready),.m_bresp(2'd0),.m_bid(4'd0),
        .ack_valid(ack_valid),.ack_ready(ack_ready),.ack_tag(TAG),.ack_generation(64'd1),.ack_bytes(13'd128),
        .result_valid(result_valid),.result_ready(result_ready),.result_tag(result_tag),.result_bytes(result_bytes),
        .owned(owned),.fault(fault),.fault_code(fault_code),.window_held(window_held));
    task require(input bit ok,input string message);
        if(ok!==1'b1)$fatal(1,"BG CHECK case=%0d %s",test_id,message);
    endtask
    task atick(input integer n);repeat(n)@(negedge axi_clk);endtask
    task mtick(input integer n);repeat(n)@(negedge mgmt_clk);endtask
    integer aw_count=0,w_count=0,b_count=0,ack_count=0,result_count=0,mgmt_cycles=0;
    integer fault_at=-1,event_at=-1;
    reg aw_stalled=0,w_stalled=0;
    reg [47:0] held_address=0;
    reg [511:0] held_data=0;
    reg held_last=0;
    always @(posedge mgmt_clk)begin
        mgmt_cycles++;
        if(guard_fault && fault_at<0)begin
            fault_at=mgmt_cycles;
            $display("BG FAULT cycle=%0d code=%0d clock_enable=%b owned=%b cold=%b link=%b stop=%b",mgmt_cycles,guard_code,clock_enable,owned,client_cold_rst_n,client_link_up,client_stop);
        end
    end
    // Public offers must survive stalls, clock loss and operational resets.
    always @(posedge axi_clk)begin
        if(!client_cold_rst_n)begin aw_stalled<=0;w_stalled<=0;end
        else begin
            if(aw_stalled)require(m_awvalid && m_awaddr===held_address,"AW offer retained");
            if(w_stalled)require(m_wvalid && m_wdata===held_data && m_wlast===held_last,"W offer retained");
            aw_stalled<=m_awvalid && !m_awready;held_address<=m_awaddr;
            w_stalled<=m_wvalid && !m_wready;held_data<=m_wdata;held_last<=m_wlast;
            if(m_awvalid && m_awready)begin
                require(m_awaddr==48'h10000000 && m_awlen==1 && m_awsize==6 && m_awburst==1 && m_awid==0,"AW identity");
                aw_count++;$display("BG AW address=%h",m_awaddr);
            end
            if(m_wvalid && m_wready)begin
                require(m_wstrb==64'hffffffffffffffff && m_wlast==(w_count==1),"W framing");
                for(integer b=0;b<64;b++)require(m_wdata[b*8+:8]==8'(w_count*64+b),"W payload");
                w_count++;$display("BG W beat=%0d last=%b",w_count,m_wlast);
            end
            if(m_bvalid && m_bready)begin b_count++;$display("BG B count=%0d",b_count);end
            if(ack_valid && ack_ready)begin ack_count++;$display("BG ACK count=%0d",ack_count);end
            if(result_valid && result_ready)begin
                require(result_tag==TAG && result_bytes==128,"result identity");
                result_count++;$display("BG RESULT bytes=%0d",result_bytes);
            end
        end
    end
    task configure;
        atick(2);cfg_valid=1;do @(posedge axi_clk);while(!cfg_ready);
        @(negedge axi_clk);cfg_valid=0;wait(map_valid);atick(2);
        require(map_session==9 && map_generation==1 && map_axi_base==64'h10000000 &&
                map_pcie_base==64'h1234567800001000,"map fixture identity");
        map_rsp_valid=1;do @(posedge axi_clk);while(!map_rsp_ready);
        @(negedge axi_clk);map_rsp_valid=0;wait(cfg_rsp_valid);atick(2);
        require(!cfg_rsp_error && !fault,"startup must not fault writer");
        cfg_rsp_ready=1;@(posedge axi_clk);@(negedge axi_clk);cfg_rsp_ready=0;atick(2);
    endtask
    task command;
        cmd_valid=1;do @(posedge axi_clk);while(!cmd_ready);
        @(negedge axi_clk);cmd_valid=0;atick(2);
        require(owned && src_req_valid && src_req_tag==TAG && src_req_bytes==128 && src_req_address==64'h1000,"owned source context");
    endtask
    task load_source;
        src_req_ready=1;@(posedge axi_clk);@(negedge axi_clk);src_req_ready=0;
        for(integer beat=0;beat<2;beat++)begin
            for(integer b=0;b<64;b++)src_data[b*8+:8]=8'(beat*64+b);
            src_last=beat==1;src_valid=1;
            do @(posedge axi_clk);while(!src_ready);
            @(negedge axi_clk);src_valid=0;atick(1);
        end
        src_cpl_valid=1;do @(posedge axi_clk);while(!src_cpl_ready);
        @(negedge axi_clk);src_cpl_valid=0;wait(m_awvalid && m_wvalid);atick(2);
    endtask
    task complete_data;
        m_awready=1;m_wready=1;wait(aw_count==1 && w_count==2);atick(2);
        m_awready=0;m_wready=0;m_bvalid=1;do @(posedge axi_clk);while(!m_bready);
        @(negedge axi_clk);m_bvalid=0;atick(2);
        ack_valid=1;do @(posedge axi_clk);while(!ack_ready);
        @(negedge axi_clk);ack_valid=0;atick(4);
    endtask
    task finish_case;
        $display("BG CASE_PASS case=%0d fault=%b code=%0d owned=%b aw=%0d w=%0d b=%0d ack=%0d results=%0d latency=%0d",test_id,guard_fault,guard_code,owned,aw_count,w_count,b_count,ack_count,result_count,fault_at-event_at);
        $display("BRIDGE_GUARD_LOCAL_PASS case=%0d axi_half=%0d mgmt_half=%0d phase=%0d",test_id,axi_half,mgmt_half,phase_ns);
        $finish;
    endtask
    initial begin
        $dumpfile("guard.vcd");$dumpvars(1,tb_ntrpu_axi_bridge_guard);
        #0.002;mtick(8);
        require(!armed && !guard_fault && !client_link_up && !client_cold_rst_n,"cold startup closed");
        cold_rst_n=1;
        if(test_id==1)begin
            mtick(64);require(!armed && !guard_fault && !client_link_up,"native absent before first arm");
        end
        native_link_up=1;native_axi_reset_n=1;native_ctl_reset_n=1;perst_n=1;
        if(test_id==12)begin
            mtick(100);require(!armed && !clock_alive && !client_link_up && !client_cold_rst_n,"absent clock cannot release startup");
            finish_case;
        end
        wait(client_link_up);atick(4);require(!client_stop && !guard_fault,"qualified startup");
        configure;
        if(test_id==0 || test_id==1 || test_id==2 || test_id==11)begin
            if(test_id==2)mtick(256);
            if(test_id==11)begin
                @(negedge axi_clk);clock_enable=0;mtick(3);clock_enable=1;mtick(32);
                require(client_link_up && !guard_fault && !fault,"short clock pause stays below watchdog");
            end
            command;load_source;complete_data;wait(result_valid);atick(4);
            require(owned && !guard_fault && !fault,"healthy result retains ownership under backpressure");
            result_ready=1;@(posedge axi_clk);@(negedge axi_clk);result_ready=0;atick(2);
            require(!owned && result_count==1,"healthy completion");finish_case;
        end
        if(test_id!=16)command;
        if(test_id==9 || test_id==10 || test_id==15)load_source;
        if(test_id==15)begin
            m_awready=1;m_wready=1;@(posedge axi_clk);@(negedge axi_clk);m_awready=0;m_wready=0;
            require(aw_count==1 && w_count==1 && !m_awvalid && m_wvalid && m_wlast,"one data beat remains");
        end
        if(test_id==4)@(posedge axi_clk);else @(negedge axi_clk);
        #0.001;
        event_at=mgmt_cycles;$display("BG EVENT cycle=%0d case=%0d t=%0t",event_at,test_id,$time);
        case(test_id)
            3,4,9,13,14,15,16:clock_enable=0;
            5,10:perst_n=0;
            6:native_axi_reset_n=0;
            7:native_ctl_reset_n=0;
            8:native_link_up=0;
            default:$fatal(1,"Unknown case");
        endcase
        if(test_id==14)perst_n=0;
        // Short pulses deliberately end before the next management edge.
        #0.1;
        native_link_up=1;native_axi_reset_n=1;native_ctl_reset_n=1;perst_n=1;
        mtick(48);
        require(guard_fault && client_stop && !client_link_up,"operational fault closes admission");
        require(client_cold_rst_n,"operational event never asserts cold reset");
        require(fault_at>=event_at && fault_at-event_at<=44,"bounded independent detection");
        require(owned==(test_id!=16) && !cfg_ready && !cmd_ready && !result_valid,"retained context while unavailable");
        if(test_id!=16)require(src_req_tag==TAG && src_req_address==64'h1000 && src_req_bytes==128,"retained source identity");
        if(test_id==9 || test_id==10)require(m_awvalid && m_wvalid,"stalled AXI offers not canceled");
        if(test_id==15)require(!m_awvalid && m_wvalid && m_wlast,"partial burst offer retained");
        $display("BG CLOSED clock_enable=%b clock_level=%b cold=%b owned=%b awvalid=%b wvalid=%b t=%0t",clock_enable,axi_clk,client_cold_rst_n,owned,m_awvalid,m_wvalid,$time);
        clock_enable=1;mtick(64);atick(8);
        require(clock_alive && guard_fault && client_stop && !client_link_up,"fault sticky after clock and link recovery");
        require(fault && !cmd_ready && !cfg_ready,"writer sees retained fault on clock return");
        if(test_id==9 || test_id==10 || test_id==15)begin
            complete_data;atick(8);
            require(owned && !result_valid && result_count==0 && b_count==1 && ack_count==1,"late real B and ACK do not release failed ownership");
        end
        if(test_id==13)begin
            @(negedge axi_clk);clock_enable=0;cold_rst_n=0;mtick(8);
            require(!client_cold_rst_n && !guard_fault && !client_link_up,"configuration reset held without AXI clock");
            cold_rst_n=1;mtick(8);require(!client_cold_rst_n,"cold release waits for AXI clock");
            clock_enable=1;wait(client_link_up);atick(8);
            require(!owned && !fault && !guard_fault && !client_stop,"new configuration clears retained context");
        end
        finish_case;
    end
    initial begin #250000;$fatal(1,"BG TIMEOUT case=%0d",test_id);end
endmodule
`default_nettype wire
