`timescale 1ns/1ps
module tb_ntrpu_peer_read;
    import ntrpu_peer_read_pkg::*;
    parameter integer FULL=0,PAIR=0;
    localparam integer BW=FULL?256:512, BYTES=BW/8;
    reg clk=0,mem_clk=0,cold_reset_n=0;
    always #2.5 clk=~clk;
    always #1.111 mem_clk=~mem_clk;
    wire rst_n,mem_rst_n,started,reader_fault;
    reg [1:0] stop=0,link_up=3,revoke=0,cfg_valid=0,request_valid=0,reject_ready=3;
    reg [1:0] dma_ready=0,dma_done_valid=0,ack_valid=0,result_ready=0,command_open=3;
    wire [1:0] cfg_ready,cfg_error,request_ready,reject_valid,cmd_valid,cmd_ready;
    wire [1:0] data_valid,data_ready,data_last,cpl_valid,cpl_ready,dma_valid,dma_last;
    wire [1:0] result_valid,halted,quarantined;
    wire [63:0] cmd_addr[0:1],cmd_tag[0:1],data_tag[0:1],cpl_tag[0:1],dma_address[0:1],result_bytes[0:1];
    wire [7:0] cmd_len[0:1];
    wire [511:0] data[0:1],dma_data[0:1];
    wire [1:0] cpl_status[0:1],contexts_held[0:1];
    wire [12:0] cpl_bytes[0:1];
    wire [3:0] reject_code[0:1],result_status[0:1];
    lease_t cfg[0:1];
    request_t request[0:1];
    wire [$bits(request_t)-1:0] dma_context[0:1],result_context[0:1];
    wire [$bits(identity_t)-1:0] reject_identity[0:1];
    event_t dma_done[0:1],ack[0:1];
    wire [1:0] arvalid,arready,rvalid,rready,rlast;
    wire [33:0] araddr[0:1];
    wire [7:0] arlen[0:1];
    wire [BW-1:0] rdata[0:1];
    wire [1:0] rresp[0:1];
    reg [1:0] ar_open=3,r_open=3;
    reg inject_error=0;
    integer epoch=0,scenario=0,commands[0:1],local_completions[0:1],returns[0:1];
    integer nr[0:1],nw[0:1],rb[0:1];
    reg [33:0] qa[0:1][0:1023];
    integer ql[0:1][0:1023];
    integer ticks=0;
    wire bfm_clk=FULL?mem_clk:clk;
    lease_t expected_lease[0:1];
    function automatic [7:0] value_at(input [63:0] address,input integer lane);
        value_at=(address ^ (address>>8) ^ (address>>16) ^ (lane*91)) & 255;
    endfunction
    function automatic [BW-1:0] payload(input [33:0] address,input integer lane);
        for(integer b=0;b<BYTES;b++)payload[b*8+:8]=value_at(address+b,lane);
    endfunction
    for(genvar l=0;l<2;l++)begin: lanes
        if(!PAIR)begin: separate
        ntrpu_peer_read_lane #(.SOURCE_NODE(l)) peer(
            .clk(clk),.rst_n(rst_n),.stop(|stop),.link_up(link_up[l]),.reader_fault(reader_fault),.revoke(revoke[l]),
            .cfg_valid(cfg_valid[l]),.cfg_ready(cfg_ready[l]),.cfg(cfg[l]),.cfg_error(cfg_error[l]),
            .request_valid(request_valid[l]),.request_ready(request_ready[l]),.request(request[l]),
            .reject_valid(reject_valid[l]),.reject_ready(reject_ready[l]),.reject_code(reject_code[l]),.reject_identity(reject_identity[l]),
            .cmd_valid(cmd_valid[l]),.cmd_ready(cmd_ready[l]),.cmd_addr(cmd_addr[l]),.cmd_tag(cmd_tag[l]),.cmd_len(cmd_len[l]),
            .data_valid(data_valid[l]),.data_ready(data_ready[l]),.data_last(data_last[l]),.data(data[l]),.data_tag(data_tag[l]),
            .cpl_valid(cpl_valid[l]),.cpl_ready(cpl_ready[l]),.cpl_tag(cpl_tag[l]),.cpl_status(cpl_status[l]),.cpl_bytes(cpl_bytes[l]),
            .dma_valid(dma_valid[l]),.dma_ready(dma_ready[l]),.dma_last(dma_last[l]),.dma_data(dma_data[l]),
            .dma_context(dma_context[l]),.dma_address(dma_address[l]),
            .dma_done_valid(dma_done_valid[l]),.dma_done(dma_done[l]),.ack_valid(ack_valid[l]),.ack(ack[l]),
            .result_valid(result_valid[l]),.result_ready(result_ready[l]),.result_context(result_context[l]),
            .result_status(result_status[l]),.result_bytes(result_bytes[l]),.contexts_held(contexts_held[l]),
            .halted(halted[l]),.quarantined(quarantined[l]));
        end
        assign arready[l]=mem_rst_n && ar_open[l] && nw[l]<1024;
        assign rvalid[l]=mem_rst_n && r_open[l] && nr[l]<nw[l];
        assign rdata[l]=payload(qa[l][nr[l]]+rb[l]*BYTES,l);
        assign rlast[l]=rb[l]+1==ql[l][nr[l]];
        assign rresp[l]=(inject_error && l==0 && qa[l][nr[l]]+rb[l]*BYTES==4096-BYTES)?2'b10:2'b00;
        always @(posedge bfm_clk)begin
            if(!mem_rst_n)begin nr[l]<=0;nw[l]<=0;rb[l]<=0;end
            else begin
                if(arvalid[l] && arready[l])begin
                    if(nw[l]>=1024)$fatal(1,"PR_BFM_OVERFLOW");
                    qa[l][nw[l]]<=araddr[l];ql[l][nw[l]]<=int'(arlen[l])+1;nw[l]<=nw[l]+1;
                    $display("PR AR %0d %0d %0d %0d %0d",scenario,epoch,l,araddr[l],int'(arlen[l])+1);
                end
                if(rvalid[l] && rready[l])begin
                    if(rlast[l])begin nr[l]<=nr[l]+1;rb[l]<=0;end else rb[l]<=rb[l]+1;
                end
            end
        end
    end
    if(PAIR)begin: pair
        wire [7:0] h_arlen;
        wire [67:0] h_araddr;
        ntrpu_peer_read_pair dut(
            .clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_reset_n),.init_complete_async(2'b11),
            .stop(stop),.link_up(link_up),.revoke(revoke),.cfg_valid(cfg_valid),.request_valid(request_valid),.reject_ready(reject_ready),
            .dma_ready(dma_ready),.dma_done_valid(dma_done_valid),.ack_valid(ack_valid),.result_ready(result_ready),
            .cfg({cfg[1],cfg[0]}),.request({request[1],request[0]}),.dma_done({dma_done[1],dma_done[0]}),.ack({ack[1],ack[0]}),
            .rst_n(rst_n),.mem_rst_n(mem_rst_n),.started(started),.reader_fault(reader_fault),
            .cfg_ready(cfg_ready),.cfg_error(cfg_error),.request_ready(request_ready),.reject_valid(reject_valid),
            .reject_code({reject_code[1],reject_code[0]}),.reject_identity({reject_identity[1],reject_identity[0]}),
            .dma_valid(dma_valid),.dma_last(dma_last),.result_valid(result_valid),.halted(halted),.quarantined(quarantined),
            .dma_data({dma_data[1],dma_data[0]}),.dma_context({dma_context[1],dma_context[0]}),.result_context({result_context[1],result_context[0]}),
            .dma_address({dma_address[1],dma_address[0]}),.result_bytes({result_bytes[1],result_bytes[0]}),
            .result_status({result_status[1],result_status[0]}),.contexts_held({contexts_held[1],contexts_held[0]}),
            .h_arvalid(arvalid),.h_arready(arready),.h_araddr(h_araddr),.h_arlen(h_arlen),
            .h_rvalid(rvalid),.h_rready(rready),.h_rdata({rdata[1],rdata[0]}),.h_rid(12'b0),.h_rresp({rresp[1],rresp[0]}),.h_rlast(rlast));
        assign cmd_valid=dut.cmd_valid;assign cmd_ready=dut.cmd_ready;
        assign cpl_valid=dut.cpl_valid;assign cpl_ready=dut.cpl_ready;
        for(genvar l=0;l<2;l++)begin
            assign araddr[l]=h_araddr[l*34+:34];assign arlen[l]={4'b0,h_arlen[l*4+:4]};
            assign cmd_addr[l]=dut.cmd_addr[l*64+:64];assign cmd_tag[l]=dut.cmd_tag[l*64+:64];assign cmd_len[l]=dut.cmd_len[l*8+:8];
            assign cpl_tag[l]=dut.cpl_tag[l*64+:64];assign cpl_status[l]=dut.cpl_status[l*2+:2];assign cpl_bytes[l]=dut.cpl_bytes[l*13+:13];
        end
    end else if(FULL)begin: full
        wire [1:0] underlying_ready;
        wire [7:0] h_arlen;
        wire [67:0] h_araddr;
        ntrpu_hbm_read_return reader(
            .clk(clk),.mem_clk(mem_clk),.cold_reset_n(cold_reset_n),.init_complete_async(2'b11),
            .stop(|stop),.fault_in(1'b0),.rst_n(rst_n),.mem_rst_n(mem_rst_n),.started(started),.fault(reader_fault),
            .cmd_valid(cmd_valid&command_open),.cmd_ready(underlying_ready),.cmd_addr({cmd_addr[1],cmd_addr[0]}),
            .cmd_tag({cmd_tag[1],cmd_tag[0]}),.cmd_len({cmd_len[1],cmd_len[0]}),
            .data_valid(data_valid),.data_ready(data_ready),.data_last(data_last),.data({data[1],data[0]}),.data_tag({data_tag[1],data_tag[0]}),
            .cpl_valid(cpl_valid),.cpl_ready(cpl_ready),.cpl_tag({cpl_tag[1],cpl_tag[0]}),
            .cpl_status({cpl_status[1],cpl_status[0]}),.cpl_bytes({cpl_bytes[1],cpl_bytes[0]}),
            .h_arvalid(arvalid),.h_arready(arready),.h_araddr(h_araddr),.h_arlen(h_arlen),
            .h_rvalid(rvalid),.h_rready(rready),.h_rdata({rdata[1],rdata[0]}),.h_rid(12'b0),
            .h_rresp({rresp[1],rresp[0]}),.h_rlast(rlast));
        assign cmd_ready=underlying_ready&command_open;
        for(genvar l=0;l<2;l++)begin
            assign araddr[l]=h_araddr[l*34+:34];assign arlen[l]={4'b0,h_arlen[l*4+:4]};
        end
    end else begin: frames
        // Common clock for this focused ledger test; FULL uses real CDC/IP.
        assign rst_n=cold_reset_n;assign mem_rst_n=cold_reset_n;assign started=cold_reset_n;
        wire [1:0] faults;
        assign reader_fault=|faults;
        for(genvar l=0;l<2;l++)begin
            wire ready;
            ntrpu_hbm_read_frames #(.FRAMES(2)) reader(
                .clk(clk),.rst_n(rst_n),.admit(1'b1),.cmd_valid(cmd_valid[l]&command_open[l]),.cmd_ready(ready),
                .cmd_addr(cmd_addr[l]),.cmd_tag(cmd_tag[l]),.cmd_len(cmd_len[l]),
                .data_valid(data_valid[l]),.data_ready(data_ready[l]),.data(data[l]),.data_last(data_last[l]),.data_tag(data_tag[l]),
                .cpl_valid(cpl_valid[l]),.cpl_ready(cpl_ready[l]),.cpl_tag(cpl_tag[l]),.cpl_status(cpl_status[l]),.cpl_bytes(cpl_bytes[l]),
                .fault(faults[l]),.arvalid(arvalid[l]),.arready(arready[l]),.araddr(araddr[l]),.arlen(arlen[l]),
                .rvalid(rvalid[l]),.rready(rready[l]),.rdata(rdata[l]),.rid(4'b0),.rresp(rresp[l]),.rlast(rlast[l]));
            assign cmd_ready[l]=ready&command_open[l];
        end
    end
    // FULL=0's BFM must share the reader clock. Disable its separate oscillator
    // by selecting the clock at the BFM, rather than using asynchronous payloads.
    task automatic check(input bit condition,input string why);
        if(!condition)$fatal(1,"PR_FAIL case=%0d epoch=%0d %s",scenario,epoch,why);
    endtask
    task automatic tick(input integer n=1);repeat(n)@(negedge clk);endtask
    function automatic lease_t lease_for(input integer lane);
        lease_t v;
        v='0;v.source.session=epoch;v.source.node=lane;v.source.slot=20+lane;v.source.generation=1;
        v.target.session=epoch;v.target.node=lane^1;v.target.slot=30+lane;v.target.generation=1;
        v.source_base=lane*65536;v.source_size=65536;v.published_bytes=65536;
        v.target_base=64'h10000000000+lane*65536;v.target_size=65536;v.window_generation=1;v.window_bytes=65536;
        v.model_epoch=1;v.source_read=1;v.target_write=1;v.source_open=1;v.target_open=1;
        return v;
    endfunction
    function automatic request_t request_for(input integer lane,input integer seq,input integer offset=0,input integer length=4096);
        request_t v;
        v='0;v.identity.session=epoch;v.identity.source=lane;v.identity.sequence_id=seq;
        v.source=cfg[lane].source;v.target=cfg[lane].target;v.window_generation=cfg[lane].window_generation;
        v.model_epoch=cfg[lane].model_epoch;v.source_offset=offset;v.target_offset=offset;v.bytes=length;return v;
    endfunction
    function automatic event_t event_for(input integer lane,input integer seq,input integer bytes=4096);
        event_t e;
        e='0;e.identity.session=epoch;e.identity.source=lane;e.identity.sequence_id=seq;
        e.target=expected_lease[lane].target;e.window_generation=1;e.bytes=bytes;return e;
    endfunction
    task automatic configure;
        cfg_valid=3;#0.001;while(cfg_ready!=3)tick();tick();cfg_valid=0;check(cfg_error==0,"CONFIG");
    endtask
    task automatic boot;
        cold_reset_n=0;stop=0;link_up=3;revoke=0;cfg_valid=0;request_valid=0;dma_ready=0;
        dma_done_valid=0;ack_valid=0;result_ready=0;command_open=3;ar_open=3;r_open=3;inject_error=0;reject_ready=3;
        epoch=epoch+1;tick(10);cold_reset_n=1;while(!started)tick();tick(5);
        for(integer l=0;l<2;l++)begin cfg[l]=lease_for(l);expected_lease[l]=cfg[l];end
        configure();
        $display("PR BOOT %0d %0d",scenario,epoch);
    endtask
    task automatic submit(input integer lane,input request_t v);
        request[lane]=v;request_valid[lane]=1;#0.001;while(!request_ready[lane])tick();tick();request_valid[lane]=0;
    endtask
    task automatic reject(input request_t v,input integer code);
        integer before_commands;
        before_commands=commands[0];reject_ready[0]=0;submit(0,v);tick(3);
        check(reject_valid[0] && reject_code[0]==code && reject_identity[0]==v.identity,"REJECT_STATUS");
        check(commands[0]==before_commands && contexts_held[0]==0,"REJECT_ISSUED");
        $display("PR REJECT %0d %0d %0d %0d",scenario,epoch,code,v.identity.sequence_id);
        reject_ready[0]=1;tick();
    endtask
    task automatic notify_dma(input integer lane,input event_t e);
        dma_done[lane]=e;dma_done_valid[lane]=1;tick();dma_done_valid[lane]=0;
    endtask
    task automatic notify_ack(input integer lane,input event_t e);
        ack[lane]=e;ack_valid[lane]=1;tick();ack_valid[lane]=0;
    endtask
    task automatic wait_local(input integer count);
        while(local_completions[0]<count || local_completions[1]<count)tick();tick(2);
    endtask
    task automatic failed_retains;
        while(!result_valid[0])tick();check(result_status[0]!=0 && result_bytes[0]==0,"FALSE_SUCCESS");
        tick(10);result_ready[0]=1;tick(3);result_ready[0]=0;
        check(halted[0] && quarantined[0] && contexts_held[0]!=0 && !cfg_ready[0] && !request_ready[0],"FAILED_LEASE_RELEASED");
        $display("PR QUARANTINE %0d %0d %0d",scenario,epoch,contexts_held[0]);
    endtask
    reg [1:0] held_cmd=0,held_dma=0,held_result=0;
    reg [135:0] saved_cmd[0:1];
    reg [1600:0] saved_dma[0:1];
    reg [1091:0] saved_result[0:1];
    integer received[0:1][0:32];
    request_t observed;
    always @(posedge clk)begin
        ticks=ticks+1;
        for(integer l=0;l<2;l++)begin
            if(!rst_n)begin
                commands[l]=0;local_completions[l]=0;returns[l]=0;held_cmd[l]=0;held_dma[l]=0;held_result[l]=0;
                for(integer s=0;s<33;s++)received[l][s]=0;
            end else begin
                if(held_cmd[l])check(cmd_valid[l] && saved_cmd[l]=={cmd_addr[l],cmd_tag[l],cmd_len[l]},"COMMAND_STABILITY");
                if(held_dma[l])check(dma_valid[l] && saved_dma[l]=={dma_context[l],dma_address[l],dma_data[l],dma_last[l]},"DMA_STABILITY");
                if(held_result[l])check(result_valid[l] && saved_result[l]=={result_context[l],result_status[l],result_bytes[l]},"RESULT_STABILITY");
                held_cmd[l]=cmd_valid[l] && !cmd_ready[l];saved_cmd[l]={cmd_addr[l],cmd_tag[l],cmd_len[l]};
                held_dma[l]=dma_valid[l] && !dma_ready[l];saved_dma[l]={dma_context[l],dma_address[l],dma_data[l],dma_last[l]};
                held_result[l]=result_valid[l] && !result_ready[l];saved_result[l]={result_context[l],result_status[l],result_bytes[l]};
                if(cmd_valid[l] && cmd_ready[l])begin
                    commands[l]++;$display("PR CMD %0d %0d %0d %0d %0d %0d",scenario,epoch,l,cmd_tag[l],cmd_addr[l],int'(cmd_len[l])+1);
                end
                if(dma_valid[l] && dma_ready[l])begin
                    observed=dma_context[l];check(observed.identity.sequence_id<33,"TEST_SEQUENCE");
                    check(observed.identity.session==epoch && observed.identity.source==l && observed.source==expected_lease[l].source && observed.target==expected_lease[l].target,"CONTEXT_MIX");
                    check(dma_address[l]==expected_lease[l].target_base+observed.target_offset,"DESTINATION");
                    for(integer b=0;b<64;b++)check(dma_data[l][b*8+:8]==value_at(l*65536+observed.source_offset+received[l][observed.identity.sequence_id]*64+b,l),"PAYLOAD");
                    check(dma_last[l]==(received[l][observed.identity.sequence_id]+1==observed.bytes/64),"LAST");
                    received[l][observed.identity.sequence_id]++;
                    $display("PR DATA %0d %0d %0d %0d %0d %0d %h",scenario,epoch,l,observed.identity.sequence_id,received[l][observed.identity.sequence_id],dma_last[l],dma_data[l]);
                end
                if(cpl_valid[l] && cpl_ready[l])begin local_completions[l]++;$display("PR LOCAL %0d %0d %0d %0d %0d %0d",scenario,epoch,l,cpl_tag[l],cpl_status[l],cpl_bytes[l]);end
                if(dma_done_valid[l])$display("PR DMA %0d %0d %0d %0d %0d %0d",scenario,epoch,l,dma_done[l].identity.sequence_id,dma_done[l].status,dma_done[l].bytes);
                if(ack_valid[l])$display("PR ACK %0d %0d %0d %0d %0d %0d",scenario,epoch,l,ack[l].identity.sequence_id,ack[l].status,ack[l].bytes);
                if(result_valid[l] && result_ready[l])begin
                    returns[l]++;observed=result_context[l];
                    $display("PR RESULT %0d %0d %0d %0d %0d %0d",scenario,epoch,l,observed.identity.sequence_id,result_status[l],result_bytes[l]);
                end
            end
        end
    end
    request_t v;
    event_t e;
    initial begin
        scenario=1;boot();
        // Simultaneous config/descriptor VALID has explicit config priority.
        cfg[0].source.generation=2;cfg[0].target.generation=2;cfg[0].window_generation=2;
        cfg[1].source.generation=2;cfg[1].target.generation=2;cfg[1].window_generation=2;
        request[0]=request_for(0,1);request[0].identity.session=0;
        cfg_valid=3;request_valid[0]=1;#0.001;
        check(cfg_ready==3 && !request_ready[0],"CONFIG_PRIORITY_DEADLOCK");
        tick();cfg_valid=0;tick();request_valid=0;tick(2);
        check(commands[0]==0 && contexts_held[0]==0,"CONCURRENT_BAD_DESCRIPTOR_ISSUED");
        $display("PR CONFIG_PRIORITY_PASS");boot();
        cfg[0].source.generation=2;cfg[0].target.generation=2;cfg[0].window_generation=2;
        cfg[1].source.generation=2;cfg[1].target.generation=2;cfg[1].window_generation=2;
        revoke[0]=1;configure();revoke[0]=0;
        reject_ready[0]=0;submit(0,request_for(0,1));tick(2);
        check(reject_valid[0] && reject_code[0]==PERMISSION && commands[0]==0,"CONFIG_REVOKE_PRIORITY");
        $display("PR CONFIG_REVOKE_PASS");boot();
        // Public descriptor rejection matrix, all without a local command/AR.
        for(integer k=0;k<18;k++)begin
            v=request_for(0,1);
            case(k)
                0:v.identity.session=0;
                1:v.source.node=1;
                2:v.source.generation=2;
                3:v.target.generation=2;
                4:v.window_generation=2;
                5:v.model_epoch=2;
                6:v.bytes=0;
                7:v.source_offset=65536;
                8:v.target_offset=65536;
                9:v.source_offset=64'hffffffffffffffc0;
                10:v.bytes=8192;
                11:v.bytes=65;
                12:v.source_offset=1;
                13:begin v.source_offset=64;v.bytes=4096;end
                14:v.identity.sequence_id=0;
                15:v.target.session=0;
                16:v.identity.source=1;
                17:v.target_offset=1;
            endcase
            reject(v,(k<6 || k==15 || k==16)?HANDLE:(k<10)?RANGE:(k==14)?REPLAY:SPAN);
        end
        for(integer k=0;k<9;k++)begin
            boot();
            cfg[0].source.generation=2;cfg[0].target.generation=2;cfg[0].window_generation=2;
            cfg[1].source.generation=2;cfg[1].target.generation=2;cfg[1].window_generation=2;
            case(k)
                0:cfg[0].source_read=0;
                1:cfg[0].target_write=0;
                2:cfg[0].source_open=0;
                3:cfg[0].target_open=0;
                4:cfg[0].published_bytes=64;
                5:cfg[0].source_base=64'hffffffffffffffc0;
                6:cfg[0].source_base=64'h400000000;
                7:cfg[0].target_base=64'hffffffffffffffc0;
                8:cfg[0].window_bytes=64;
            endcase
            configure();reject(request_for(0,1),k<4?PERMISSION:k==4?UNPUBLISHED:k==6?SPAN:RANGE);
        end
        boot();revoke[0]=1;tick();revoke[0]=0;reject(request_for(0,1),PERMISSION);
        check(nw[0]==0,"REJECT_AR");$display("PR CASE 1 PASS");
        scenario=2;boot();command_open=0;
        for(integer l=0;l<2;l++)begin submit(l,request_for(l,1));submit(l,request_for(l,2,4096));end
        check(contexts_held[0]==2 && contexts_held[1]==2,"FOUR_CONTEXTS");
        cfg_valid=3;cfg[0].source_base=64'hffff;tick(12);check(cfg_ready==0,"LEASE_CHANGED_WHILE_HELD");cfg_valid=0;
        command_open=3;while(dma_valid!=3)tick();tick(12);dma_ready=3;wait_local(2);
        check(result_valid==0 && contexts_held[0]==2 && contexts_held[1]==2,"LOCAL_COMPLETION_RELEASED");
        for(integer l=0;l<2;l++)begin
            notify_ack(l,event_for(l,2));notify_dma(l,event_for(l,2));
            notify_dma(l,event_for(l,1));tick(4);check(!result_valid[l],"DMA_IS_NOT_ACK");notify_ack(l,event_for(l,1));
        end
        tick(12);check(result_valid==3 && result_status[0]==0 && result_status[1]==0,"SUCCESS");
        result_ready=3;while(returns[0]<2 || returns[1]<2)tick();tick(2);
        check(contexts_held[0]==0 && contexts_held[1]==0,"SUCCESS_NOT_RELEASED");
        cfg[0]=lease_for(0);reject(request_for(0,1),REPLAY);
        for(integer seq=3;seq<=10;seq++)begin
            integer size_bytes;
            size_bytes=(seq%4==0)?4096:(seq%4==1)?576:(seq%4==2)?512:64;
            for(integer l=0;l<2;l++)submit(l,request_for(l,seq,seq*4096,size_bytes));
            wait_local(seq);
            for(integer l=0;l<2;l++)begin notify_dma(l,event_for(l,seq,size_bytes));notify_ack(l,event_for(l,seq,size_bytes));end
            while(returns[0]<seq || returns[1]<seq)tick();tick(2);
            check(contexts_held[0]==0 && contexts_held[1]==0,"REUSED_CONTEXT_NOT_RELEASED");
        end
        $display("PR REUSE_PASS transfers=16");
        $display("PR CASE 2 PASS");
        scenario=3;boot();command_open=0;submit(0,request_for(0,1));
        reject_ready[0]=0;submit(0,request_for(0,1,4096));tick(2);check(reject_code[0]==REPLAY,"ACTIVE_DUPLICATE");
        reject_ready[0]=1;tick();reject_ready[0]=0;submit(0,request_for(0,2));tick(2);check(reject_code[0]==OVERLAP,"DESTINATION_OVERLAP");
        check(contexts_held[0]==1 && (PAIR || commands[0]==0),"DUPLICATE_RESERVATION");$display("PR CASE 3 PASS");
        scenario=4;boot();inject_error=1;dma_ready=3;submit(0,request_for(0,1));failed_retains();
        check(received[0][1]==0,"LATE_ERROR_PUBLISHED");$display("PR CASE 4 PASS");
        for(integer kind=0;kind<6;kind++)begin
            scenario=5+kind;boot();dma_ready=3;submit(0,request_for(0,1));
            if(kind!=4)while(local_completions[0]<1)tick();
            e=event_for(0,1);
            case(kind)
                0:e.status=1;
                1:e.identity.session=epoch+1;
                2:e.target.generation=2;
                3:e.bytes=64;
                4:begin end // premature ACK before payload
                5:begin notify_ack(0,e);end // duplicate ACK
            endcase
            if(kind==0)notify_dma(0,e);else notify_ack(0,e);
            failed_retains();$display("PR CASE %0d PASS",scenario);
        end
        // Stop and link loss at every exposed boundary. No failed context reuse.
        for(integer loss=0;loss<2;loss++)for(integer boundary=0;boundary<5;boundary++)begin
            scenario=11+loss*5+boundary;boot();
            if(boundary==0)command_open=0;
            if(boundary==1)ar_open=0;
            if(boundary==2)r_open=0;
            submit(0,request_for(0,1));
            if(boundary==1)while(!arvalid[0])tick();
            if(boundary==2)while(nw[0]==0)tick();
            if(boundary==3)while(!dma_valid[0])tick();
            if(boundary==4)begin dma_ready=3;while(local_completions[0]<1)tick();end
            if(loss)link_up[0]=0;else stop[0]=1;
            tick(5);failed_retains();command_open=3;ar_open=3;r_open=3;dma_ready=3;tick(400);
            check(contexts_held[0]==1 && quarantined[0],"DRAIN_FREED_QUARANTINE");
            $display("PR CASE %0d PASS",scenario);
        end
        scenario=21;boot();ar_open=0;
        for(integer l=0;l<2;l++)submit(l,request_for(l,1));
        check(contexts_held[0]==1 && contexts_held[1]==1,"GLOBAL_STOP_SETUP");
        stop[0]=1;tick(5);
        check(halted==3 && quarantined==3 && request_ready==0 && cfg_ready==0,"GLOBAL_STOP_ADMISSION");
        check(result_valid==3 && result_status[0]!=0 && result_status[1]!=0,"GLOBAL_STOP_RESULTS");
        result_ready=3;tick(4);
        check(contexts_held[0]==1 && contexts_held[1]==1,"GLOBAL_STOP_RELEASED");
        $display("PR QUARANTINE 21 %0d 2",epoch);
        $display("PR GLOBAL_STOP_PASS");$display("PR CASE 21 PASS");
        $display("PR ALL_PASS cases=21 full=%0d",FULL);$finish;
    end
    initial begin #2000000;$fatal(1,"PR_WATCHDOG case=%0d",scenario);end
endmodule
