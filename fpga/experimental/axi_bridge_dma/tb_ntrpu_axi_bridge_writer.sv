`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_axi_bridge_writer;
    reg clk=0,cold_rst_n=0,link_up=1,stop=0;
    always #5 clk=~clk;
    reg cfg_valid=0,cfg_rsp_ready=0,map_ready=0,map_rsp_valid=0,map_rsp_ok=1;
    reg [63:0] cfg_session=9,cfg_generation=1,cfg_axi_base=64'h10000000,cfg_pcie_base=64'h100000000;
    wire cfg_ready,cfg_rsp_valid,cfg_rsp_error,map_valid,map_rsp_ready;
    wire [63:0] map_session,map_generation,map_axi_base,map_pcie_base;
    reg cmd_valid=0,reject_ready=0;
    reg [127:0] cmd_tag={64'd9,64'd1};
    reg [63:0] cmd_generation=1,cmd_source=64'h4000,cmd_offset=0;
    reg [12:0] cmd_bytes=64;
    wire cmd_ready,reject_valid;
    wire [3:0] reject_code;
    wire [127:0] reject_tag;
    reg src_req_ready=0,src_valid=0,src_last=0,src_cpl_valid=0;
    reg [511:0] src_data=0;
    reg [127:0] src_tag={64'd9,64'd1},src_cpl_tag={64'd9,64'd1};
    reg [12:0] src_cpl_bytes=64;
    reg [1:0] src_cpl_status=0;
    wire src_req_valid,src_ready,src_cpl_ready;
    wire [127:0] src_req_tag;
    wire [63:0] src_req_address;
    wire [12:0] src_req_bytes;
    wire [47:0] m_awaddr;
    wire [3:0] m_awid;
    wire [7:0] m_awlen;
    wire [2:0] m_awsize;
    wire [1:0] m_awburst;
    wire m_awvalid,m_wvalid,m_wlast,m_bready;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    reg m_awready=0,m_wready=0,m_bvalid=0;
    reg [1:0] m_bresp=0;
    reg [3:0] m_bid=0;
    reg ack_valid=0,result_ready=0;
    reg [127:0] ack_tag={64'd9,64'd1};
    reg [63:0] ack_generation=1;
    reg [12:0] ack_bytes=64;
    wire ack_ready,result_valid,owned,fault,window_held;
    wire [127:0] result_tag;
    wire [12:0] result_bytes;
    wire [7:0] fault_code;
    ntrpu_axi_bridge_writer dut(.*);
    integer seed=0,case_id=0,passes=0,expected_bytes=64,aw_count=0,w_bytes=0,b_count=0,result_count=0;
    integer aw_stalls=0,w_stalls=0,result_stalls=0,map_stalls=0,source_stalls=0;
    reg aw_stalled=0,w_stalled=0,result_stalled=0,map_stalled=0,source_stalled=0;
    reg [47:0] last_address=0;
    reg [7:0] last_length=0;
    reg [511:0] last_data=0;
    reg last_wlast=0;
    reg [140:0] last_result=0;
    reg [255:0] last_map=0;
    reg [204:0] last_source=0;
    task require(input bit condition,input string reason);
        if(!condition)$fatal(1,"CHECK case=%0d seed=%0d %s",case_id,seed,reason);
    endtask
    task tick(input integer n);repeat(n)@(negedge clk);endtask
    function automatic [7:0] pattern(input integer offset);
        pattern=8'((case_id*13+offset*37)^(offset>>3));
    endfunction
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            aw_stalled<=0;w_stalled<=0;result_stalled<=0;map_stalled<=0;source_stalled<=0;
            aw_count=0;w_bytes=0;b_count=0;result_count=0;
        end else begin
            if(aw_stalled)require(m_awvalid && m_awaddr===last_address && m_awlen===last_length,"AW stable while stalled");
            if(w_stalled)require(m_wvalid && m_wdata===last_data && m_wlast===last_wlast && &m_wstrb,"W stable while stalled");
            if(result_stalled)require(result_valid && {result_tag,result_bytes}===last_result,"result stable while stalled");
            if(map_stalled)require(map_valid && {map_session,map_generation,map_axi_base,map_pcie_base}===last_map,"mapping offer stable");
            if(source_stalled)require(src_req_valid && {src_req_tag,src_req_address,src_req_bytes}===last_source,"source request stable");
            aw_stalled<=m_awvalid&&!m_awready;w_stalled<=m_wvalid&&!m_wready;
            result_stalled<=result_valid&&!result_ready;map_stalled<=map_valid&&!map_ready;
            source_stalled<=src_req_valid&&!src_req_ready;
            last_address<=m_awaddr;last_length<=m_awlen;last_data<=m_wdata;last_wlast<=m_wlast;
            last_result<={result_tag,result_bytes};last_map<={map_session,map_generation,map_axi_base,map_pcie_base};
            last_source<={src_req_tag,src_req_address,src_req_bytes};
            if(m_awvalid&&!m_awready)aw_stalls++;
            if(m_wvalid&&!m_wready)w_stalls++;
            if(result_valid&&!result_ready)result_stalls++;
            if(map_valid&&!map_ready)map_stalls++;
            if(src_req_valid&&!src_req_ready)source_stalls++;
            if(map_valid&&map_ready)$display("EV MAP case=%0d session=%h generation=%h axi=%h pcie=%h",case_id,map_session,map_generation,map_axi_base,map_pcie_base);
            if(src_req_valid&&src_req_ready)$display("EV SRC case=%0d tag=%h address=%h bytes=%0d",case_id,src_req_tag,src_req_address,src_req_bytes);
            if(m_awvalid&&m_awready)begin
                require(aw_count==0 && m_awaddr==48'h10000000 && m_awid==0 && m_awsize==6 && m_awburst==1 && (m_awlen+1)*64==expected_bytes,"AXI address and burst");
                aw_count++;
                $display("EV AW case=%0d address=%h len=%0d",case_id,m_awaddr,m_awlen);
            end
            if(m_wvalid&&m_wready)begin
                require(w_bytes<expected_bytes && (&m_wstrb) && m_wlast==(w_bytes+64==expected_bytes),"W bounds, strobes and last");
                for(integer j=0;j<64;j++)require(m_wdata[j*8+:8]===pattern(w_bytes+j),"payload byte");
                $display("EV W case=%0d offset=%0d last=%b data=%h",case_id,w_bytes,m_wlast,m_wdata);
                w_bytes+=64;
            end
            if(m_bvalid&&m_bready)begin b_count++;$display("EV B case=%0d response=%0d id=%0d",case_id,m_bresp,m_bid);end
            if(ack_valid&&ack_ready)$display("EV ACK case=%0d tag=%h generation=%h bytes=%0d",case_id,ack_tag,ack_generation,ack_bytes);
            if(result_valid&&result_ready)begin
                require(result_tag=={64'd9,64'd1} && result_bytes==expected_bytes,"result identity");
                result_count++;$display("EV RESULT case=%0d bytes=%0d",case_id,result_bytes);
            end
        end
    end
    task reset_case(input integer id,input integer length);
        @(negedge clk);cold_rst_n=0;case_id=id;expected_bytes=length;
        cfg_valid=0;cfg_rsp_ready=0;map_ready=0;map_rsp_valid=0;map_rsp_ok=1;
        cfg_session=9;cfg_generation=1;cfg_axi_base=64'h10000000;cfg_pcie_base=64'h100000000;
        cmd_valid=0;cmd_tag={64'd9,64'd1};cmd_generation=1;cmd_source=64'h4000;cmd_offset=0;cmd_bytes=length;
        reject_ready=0;src_req_ready=0;src_valid=0;src_cpl_valid=0;src_cpl_status=0;
        src_tag=cmd_tag;src_cpl_tag=cmd_tag;src_cpl_bytes=length;
        m_awready=0;m_wready=0;m_bvalid=0;m_bresp=0;m_bid=0;
        ack_valid=0;ack_tag=cmd_tag;ack_generation=1;ack_bytes=length;result_ready=0;link_up=1;stop=0;
        tick(3);cold_rst_n=1;tick(2);$display("CASE START id=%0d bytes=%0d",id,length);
    endtask
    task configure;
        cfg_valid=1;do @(posedge clk);while(!cfg_ready);
        @(negedge clk);cfg_valid=0;
        wait(map_valid);tick(3);require(!cmd_ready,"mapping must finish before command");
        map_ready=1;@(posedge clk);@(negedge clk);map_ready=0;
        tick(2);map_rsp_valid=1;do @(posedge clk);while(!map_rsp_ready);
        @(negedge clk);map_rsp_valid=0;wait(cfg_rsp_valid);tick(2);
        require(cfg_rsp_error==!map_rsp_ok,"mapping response");
        cfg_rsp_ready=1;@(posedge clk);@(negedge clk);cfg_rsp_ready=0;
    endtask
    task submit;
        cmd_valid=1;do @(posedge clk);while(!cmd_ready);
        @(negedge clk);cmd_valid=0;
    endtask
    task source_frame(input bit bad_last,input bit bad_cpl);
        wait(src_req_valid);tick(3);src_req_ready=1;
        @(posedge clk);@(negedge clk);src_req_ready=0;
        for(integer offset=0;offset<expected_bytes;offset+=64)begin
            tick((seed+offset/64)%3);
            for(integer j=0;j<64;j++)src_data[j*8+:8]=pattern(offset+j);
            src_last=(offset+64==expected_bytes)^bad_last;src_valid=1;
            do @(posedge clk);while(!src_ready);
            @(negedge clk);src_valid=0;
        end
        if(!bad_last)begin
            tick(2);src_cpl_status=bad_cpl?2:0;src_cpl_valid=1;
            do @(posedge clk);while(!src_cpl_ready);
            @(negedge clk);src_cpl_valid=0;
        end
    endtask
    task drain_bus(input integer order_mode);
        integer cycles;
        cycles=0;
        while(aw_count!=1 || w_bytes!=expected_bytes)begin
            m_awready=(order_mode==1 ? cycles>5 : cycles>0);
            m_wready=(order_mode==0 ? cycles>5 : cycles>0) && ((cycles+seed)%3!=0);
            tick(1);cycles++;
            require(cycles<500,"bounded AXI drain");
        end
        m_awready=0;m_wready=0;
    endtask
    task response(input integer code,input integer id);
        tick(2);m_bresp=code;m_bid=id;m_bvalid=1;
        do @(posedge clk);while(!m_bready);
        @(negedge clk);m_bvalid=0;
    endtask
    task acknowledge(input bit wrong_generation);
        tick(2);ack_generation=wrong_generation?2:1;ack_valid=1;
        do @(posedge clk);while(!ack_ready);
        @(negedge clk);ack_valid=0;
    endtask
    task finish_success;
        wait(result_valid);tick(5);require(owned && window_held && !cfg_ready,"ownership through result backpressure");
        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;tick(2);
        require(!owned && !fault && result_count==1 && !result_valid,"successful retirement exactly once");
    endtask
    task check_held;
        tick(5);require(fault && owned && window_held && !result_valid && !cmd_ready && !cfg_ready,"fault preserves context");
        link_up=1;tick(5);require(fault && owned && result_count==0,"link recovery does not retire");
    endtask
    task pass_case;
        passes++;$display("CASE PASS id=%0d fault=%h owned=%b aw=%0d bytes=%0d b=%0d results=%0d",case_id,fault_code,owned,aw_count,w_bytes,b_count,result_count);
    endtask
    initial begin
        if(!$value$plusargs("seed=%d",seed))seed=0;
        $dumpfile("writer.vcd");$dumpvars(1,tb_ntrpu_axi_bridge_writer);
        for(integer i=0;i<3;i++)begin
            reset_case(i,i==0?64:(i==1?128:4096));configure;
            if(i==2)cmd_source=64'h8000000000-4096;
            submit;require(owned,"accepted command");source_frame(0,0);drain_bus(i%2);
            if(i==1)begin acknowledge(0);tick(3);require(!result_valid && owned,"ACK cannot replace B");response(0,0);end
            else begin response(0,0);tick(3);require(!result_valid && owned,"B cannot replace ACK");acknowledge(0);end
            finish_success;
            if(i==0)begin
                submit;require(reject_valid && reject_code==2 && !owned && !src_req_valid,"retired sequence cannot replay");
                cfg_valid=1;do @(posedge clk);while(!cfg_ready);
                @(negedge clk);cfg_valid=0;tick(2);
                require(cfg_rsp_valid && cfg_rsp_error && !map_valid && !fault,"old mapping generation cannot replay");
            end
            pass_case;
        end
        reset_case(3,128);configure;submit;
        cmd_tag=0;cmd_source=0;cmd_offset=2048;cmd_generation=99;cmd_bytes=4096;
        cfg_generation=2;cfg_axi_base=64'h20000000;cfg_pcie_base=64'h200000000;cfg_valid=1;
        source_frame(0,0);tick(4);
        require(!cfg_ready && !map_valid && map_generation==1 && map_axi_base==64'h10000000 && map_pcie_base==64'h100000000,"active mapping immutable");
        require(src_req_tag=={64'd9,64'd1} && src_req_bytes==128 && src_req_address==64'h4000,"descriptor immutable");
        drain_bus(1);response(0,0);acknowledge(0);cfg_valid=0;finish_success;pass_case;
        reset_case(4,64);configure;cmd_generation=0;submit;
        require(reject_valid && reject_code==1 && !owned && !src_req_valid && !m_awvalid,"stale generation rejected");pass_case;
        reset_case(5,64);configure;cmd_source=64'h8000000000;submit;
        require(reject_valid && reject_code==3 && !owned && !m_awvalid,"39-bit source limit");pass_case;
        reset_case(6,64);configure;submit;source_frame(0,0);wait(m_awvalid);tick(2);link_up=0;
        tick(4);require(m_awvalid && m_wvalid && fault && owned,"fault preserves outstanding AXI offers and ownership");link_up=1;
        drain_bus(1);response(0,0);acknowledge(0);check_held;pass_case;
        reset_case(7,64);configure;submit;source_frame(0,0);drain_bus(0);response(2,0);acknowledge(0);check_held;pass_case;
        reset_case(8,64);configure;submit;source_frame(0,0);drain_bus(0);response(0,0);acknowledge(1);check_held;pass_case;
        reset_case(9,64);configure;submit;source_frame(1,0);check_held;
        require(!m_awvalid && aw_count==0 && w_bytes==0,"bad source last never reaches AXI");pass_case;
        reset_case(10,64);map_rsp_ok=0;configure;tick(3);
        require(fault && !cmd_ready && !owned && !m_awvalid,"failed map never authorizes DMA");pass_case;
        reset_case(11,64);configure;submit;source_frame(0,1);check_held;
        require(!m_awvalid && aw_count==0,"bad source completion never reaches AXI");pass_case;
        reset_case(12,64);configure;submit;source_frame(0,0);drain_bus(1);response(0,1);acknowledge(0);check_held;pass_case;
        reset_case(13,128);configure;cmd_offset=4032;submit;
        require(reject_valid && reject_code==3 && !owned && !m_awvalid,"destination 4-KiB boundary");pass_case;
        reset_case(14,64);configure;submit;acknowledge(0);check_held;
        require(src_req_valid && !m_awvalid,"early ACK fails and preserves source offer");pass_case;
        reset_case(15,64);configure;submit;source_frame(0,0);drain_bus(0);response(0,0);acknowledge(0);
        wait(result_valid);tick(3);link_up=0;tick(3);
        require(result_valid && fault,"previous result offer survives fault");
        result_ready=1;@(posedge clk);@(negedge clk);result_ready=0;link_up=1;tick(6);
        require(owned && fault && !result_valid && result_count==1,"late fault prevents reuse without duplicate result");pass_case;
        require(aw_stalls>0 && w_stalls>0 && result_stalls>0 && map_stalls>0 && source_stalls>0,"real stalls covered");
        $display("COVER aw=%0d w=%0d result=%0d map=%0d source=%0d",aw_stalls,w_stalls,result_stalls,map_stalls,source_stalls);
        require(passes==16,"all scenarios executed");$display("WRITER_LOCAL_PASS seed=%0d cases=%0d",seed,passes);$finish;
    end
    initial begin #1000000;$fatal(1,"TIMEOUT case=%0d",case_id);end
endmodule
`default_nettype wire
