`timescale 1ns/1ps
module tb_ntrpu_peer_shared;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,mem_clk=0,cold_reset_n=0;
    always #2.5 clk=~clk;
    always #1.111 mem_clk=~mem_clk;
    reg [1:0] stop=0,link_up=3,request_valid=0,reject_ready=3;
    reg [1:0] dma_ready=0,dma_done_valid=0,ack_valid=0,result_ready=0;
    reg cfg_valid=0,revoke=0;
    lease_t cfg;
    request_t req[0:1],expected[0:63];
    event_t dma_done[0:1],ack[0:1];
    wire rst_n,mem_rst_n,started,reader_fault,reader_drained,cfg_ready,cfg_error,admission_halted;
    wire [63:0] admission_watermark;
    wire [1:0] admission_pending,request_ready,reject_valid,dma_valid,dma_last,result_valid,halted,quarantined;
    wire [7:0] reject_code,result_status;
    wire [383:0] reject_identity;
    wire [1023:0] dma_data;
    wire [2047:0] dma_context,result_context;
    wire [127:0] dma_address,result_bytes;
    wire [3:0] contexts_held;
    wire [1:0] h_arvalid,h_arready,h_rvalid,h_rready,h_rlast;
    wire [67:0] h_araddr;
    wire [7:0] h_arlen;
    wire [11:0] h_arid;
    wire [11:0] h_rid=0;
    wire [5:0] h_arsize;
    wire [3:0] h_arburst,h_rresp;
    wire [511:0] h_rdata;
    reg [1:0] ar_open=3,r_open=3;
    reg inject_error=0;
    ntrpu_peer_read_shared dut(.*,.init_complete_async(2'b11),.request({req[1],req[0]}),
        .dma_done({dma_done[1],dma_done[0]}),.ack({ack[1],ack[0]}));
    integer epoch=0,scenario=0,nw[0:1],nr[0:1],rb[0:1];
    reg [33:0] qa[0:1][0:1023];
    integer ql[0:1][0:1023];
    function automatic [7:0] value_at(input logic [63:0] address);
        return (address^(address>>8)^(address>>16)^8'h5b)&255;
    endfunction
    function automatic [255:0] payload(input logic [33:0] address);
        for(integer b=0;b<32;b++)payload[b*8+:8]=value_at(address+b);
    endfunction
    for(genvar l=0;l<2;l++)begin: responders
        assign h_arready[l]=mem_rst_n && ar_open[l] && nw[l]<1024;
        assign h_rvalid[l]=mem_rst_n && r_open[l] && nr[l]<nw[l];
        assign h_rdata[l*256+:256]=payload(qa[l][nr[l]]+rb[l]*32);
        assign h_rlast[l]=rb[l]+1==ql[l][nr[l]];
        assign h_rresp[l*2+:2]=(inject_error && l==0 && qa[l][nr[l]]+rb[l]*32==4064)?2'b10:0;
        always @(posedge mem_clk)begin
            if(!mem_rst_n)begin nw[l]<=0;nr[l]<=0;rb[l]<=0;end
            else begin
                if(h_arvalid[l] && h_arready[l])begin
                    check(h_arid[l*6+:6]==0 && h_arsize[l*3+:3]==5 && h_arburst[l*2+:2]==1,"NATIVE_ATTRIBUTES");
                    qa[l][nw[l]]<=h_araddr[l*34+:34];ql[l][nw[l]]<=int'(h_arlen[l*4+:4])+1;nw[l]<=nw[l]+1;
                    $display("SF AR %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,l,h_araddr[l*34+:34],int'(h_arlen[l*4+:4])+1);
                end
                if(h_rvalid[l] && h_rready[l])begin
                    if(h_rlast[l])begin nr[l]<=nr[l]+1;rb[l]<=0;end else rb[l]<=rb[l]+1;
                end
            end
        end
    end
    task automatic check(input bit ok,input string why);if(!ok)$fatal(1,"SG_FAIL case=%0d epoch=%0d %s",scenario,epoch,why);endtask
    task automatic tick(input integer n=1);repeat(n)begin @(negedge clk);#0.001;end endtask
    function automatic lease_t initial_lease;
        lease_t v;v='0;v.source.session=epoch;v.source.node=0;v.source.slot=20;v.source.generation=1;
        v.target.session=epoch;v.target.node=1;v.target.slot=30;v.target.generation=1;
        v.source_base=0;v.source_size=65536;v.published_bytes=65536;
        v.target_base=64'h10000000000;v.target_size=65536;v.window_bytes=65536;v.window_generation=1;
        v.model_epoch=1;v.source_read=1;v.target_write=1;v.source_open=1;v.target_open=1;return v;
    endfunction
    function automatic request_t make_request(input integer seq,offset=0,length=4096);
        request_t v;v='0;v.identity.session=epoch;v.identity.source=0;v.identity.sequence_id=seq;
        v.source=cfg.source;v.target=cfg.target;v.window_generation=cfg.window_generation;v.model_epoch=cfg.model_epoch;
        v.source_offset=offset;v.target_offset=offset;v.bytes=length;return v;
    endfunction
    function automatic event_t notification(input request_t v);
        event_t e;e='0;e.identity=v.identity;e.target=v.target;e.window_generation=v.window_generation;e.bytes=v.bytes;return e;
    endfunction
    task automatic configure;
        cfg_valid=1;#0.001;check(cfg_ready,"CFG_READY");tick();cfg_valid=0;check(!cfg_error,"CFG_ERROR");tick();
    endtask
    task automatic boot(input integer number);
        tick();cold_reset_n=0;stop=0;link_up=3;revoke=0;cfg_valid=0;request_valid=0;reject_ready=3;
        dma_ready=0;dma_done_valid=0;ack_valid=0;result_ready=0;ar_open=3;r_open=3;inject_error=0;
        epoch++;scenario=number;tick(10);cold_reset_n=1;while(!started)tick();tick(4);cfg=initial_lease();configure();
    endtask
    task automatic submit(input integer l,input request_t v);
        req[l]=v;request_valid[l]=1;do @(posedge clk);while(!request_ready[l]);tick();request_valid[l]=0;
    endtask
    task automatic rejected_request(input integer l,input request_t v,input integer code);
        reject_ready[l]=0;submit(l,v);tick(3);
        check(reject_valid[l] && reject_code[l*4+:4]==code && reject_identity[l*192+:192]==v.identity,"REJECTION");
        reject_ready[l]=1;tick();
    endtask
    task automatic notify_dma(input integer l,input request_t v);
        dma_done[l]=notification(v);dma_done_valid[l]=1;tick();dma_done_valid[l]=0;
    endtask
    task automatic notify_ack(input integer l,input request_t v);
        ack[l]=notification(v);ack_valid[l]=1;tick();ack_valid[l]=0;
    endtask
    integer received[0:63],returned[0:63],results[0:1];
    reg [1:0] held_dma=0,held_result=0;
    reg [1600:0] saved_dma[0:1];
    reg [1091:0] saved_result[0:1];
    request_t observed;
    always @(posedge clk)begin
        if(!rst_n)begin
            held_dma=0;held_result=0;results[0]=0;results[1]=0;
            for(integer i=0;i<64;i++)begin received[i]=0;returned[i]=0;end
        end else for(integer l=0;l<2;l++)begin
            if(held_dma[l])check(dma_valid[l] && saved_dma[l]=={dma_context[l*1024+:1024],dma_address[l*64+:64],dma_data[l*512+:512],dma_last[l]},"DMA_STABILITY");
            if(held_result[l])check(result_valid[l] && saved_result[l]=={result_context[l*1024+:1024],result_status[l*4+:4],result_bytes[l*64+:64]},"RESULT_STABILITY");
            held_dma[l]=dma_valid[l]&&!dma_ready[l];saved_dma[l]={dma_context[l*1024+:1024],dma_address[l*64+:64],dma_data[l*512+:512],dma_last[l]};
            held_result[l]=result_valid[l]&&!result_ready[l];saved_result[l]={result_context[l*1024+:1024],result_status[l*4+:4],result_bytes[l*64+:64]};
            if(dut.cmd_valid[l] && dut.cmd_ready[l])
                $display("SF CMD %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,l,dut.cmd_tag[l*64+:64],dut.cmd_addr[l*64+:64],int'(dut.cmd_len[l*8+:8])+1);
            if(dma_valid[l] && dma_ready[l])begin
                observed=dma_context[l*1024+:1024];check(observed.identity.sequence_id<64,"SEQUENCE");
                check(observed===expected[observed.identity.sequence_id] && observed.identity.source==0,"SOURCE_CONTEXT_MIX");
                check(dma_address[l*64+:64]==cfg.target_base+observed.target_offset,"DESTINATION");
                for(integer b=0;b<64;b++)check(dma_data[l*512+b*8+:8]==value_at(cfg.source_base+observed.source_offset+received[observed.identity.sequence_id]*64+b),"PAYLOAD");
                check(dma_last[l]==(received[observed.identity.sequence_id]+1==observed.bytes/64),"LAST");
                received[observed.identity.sequence_id]++;
                $display("SF DATA %0d %0d %0d %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,observed.identity.sequence_id,received[observed.identity.sequence_id],dma_last[l],dma_data[l*512+:512]);
            end
            if(dut.cpl_valid[l] && dut.cpl_ready[l])
                $display("SF LOCAL %0d %0d %0d %0d %0d %0d",$rtoi($realtime*1000),epoch,l,dut.cpl_tag[l*64+:64],dut.cpl_status[l*2+:2],dut.cpl_bytes[l*13+:13]);
            if(dma_done_valid[l])$display("SF DMA %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,dma_done[l]);
            if(ack_valid[l])$display("SF ACK %0d %0d %0d %h",$rtoi($realtime*1000),epoch,l,ack[l]);
            if(result_valid[l] && result_ready[l])begin
                observed=result_context[l*1024+:1024];results[l]++;returned[observed.identity.sequence_id]++;
            end
        end
    end
    `define GUARD dut.guard
    `include "ntrpu_peer_shared_monitor.svh"
    `undef GUARD
    task automatic wait_data(input request_t v);
        while(received[v.identity.sequence_id]<v.bytes/64)tick();tick(4);
    endtask
    task automatic complete(input integer l,input request_t v);
        wait_data(v);notify_dma(l,v);tick(3);check(!result_valid[l],"DMA_IS_NOT_ACK");notify_ack(l,v);
        while(!returned[v.identity.sequence_id])tick();tick(3);
    endtask
    task automatic passed;tick(5);$display("SG PASS %0d",scenario);endtask
    request_t a,b,c,d,v;
    initial begin
        boot(1);a=make_request(1);expected[1]=a;r_open=0;
        fork submit(0,a);submit(1,a);join
        tick(5);check(contexts_held==1 && rejections[1]==1,"SHARED_DUPLICATE");
        rejected_request(1,make_request(2),OVERLAP);
        r_open=3;dma_ready=3;result_ready=3;complete(0,a);
        rejected_request(1,make_request(1,4096),REPLAY);check(contexts_held==0,"CASE1_RESERVATIONS");passed();

        boot(2);a=make_request(1);b=make_request(2,4096);c=make_request(3,8192);d=make_request(4,12288);
        expected[1]=a;expected[2]=b;expected[3]=c;expected[4]=d;
        submit(0,a);submit(1,b);submit(0,c);submit(1,d);while(dma_valid!=3)tick();tick(20);
        check(contexts_held==10,"SHARED_FOUR_CONTEXTS");cfg_valid=1;tick(8);check(!cfg_ready,"CFG_CHANGED_WHILE_HELD");cfg_valid=0;
        dma_ready=3;wait_data(a);wait_data(b);wait_data(c);wait_data(d);
        notify_ack(0,c);notify_dma(0,c);notify_ack(1,d);notify_dma(1,d);
        notify_dma(0,a);notify_dma(1,b);tick(6);check(result_valid==0 && contexts_held==10,"EARLY_SHARED_RELEASE");
        notify_ack(0,a);notify_ack(1,b);while(result_valid!=3)tick();tick(10);
        check(contexts_held==10,"RESULT_OFFER_RELEASED");result_ready=1;while(results[0]<2)tick();tick(4);
        check(contexts_held==8,"ASYMMETRIC_RETIRE");rejected_request(0,make_request(5,4096),OVERLAP);
        result_ready=3;while(results[1]<2)tick();tick(3);check(contexts_held==0,"CASE2_RESERVATIONS");passed();

        boot(3);dma_ready=3;result_ready=3;
        for(integer seq=1;seq<=20;seq++)begin
            integer bytes;
            case(seq%4)0:bytes=4096;1:bytes=64;2:bytes=512;3:bytes=576;endcase
            v=make_request(seq,(seq%8)*4096,bytes);expected[seq]=v;submit((seq-1)%2,v);complete((seq-1)%2,v);
            check(contexts_held==0 && admission_watermark==seq,"REUSE_OR_WATERMARK");
        end
        passed();

        boot(4);inject_error=1;dma_ready=3;result_ready=3;
        a=make_request(1);b=make_request(2,4096);expected[1]=a;expected[2]=b;submit(0,a);submit(1,b);
        while(!admission_halted)tick();tick(400);
        check(received[1]==0 && contexts_held==5 && quarantined==3 && !cfg_ready,"NATIVE_ERROR_RELEASED");passed();

        boot(5);a=make_request(1,0,64);expected[1]=a;submit(0,a);revoke=1;tick();revoke=0;
        rejected_request(1,make_request(2,64,64),PERMISSION);dma_ready=3;result_ready=3;complete(0,a);
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;configure();
        b=make_request(2,64,64);expected[2]=b;submit(1,b);complete(1,b);check(contexts_held==0,"REVOKE_ACCEPTED_WORK");passed();

        boot(6);ar_open=0;a=make_request(1);b=make_request(2,4096);expected[1]=a;expected[2]=b;submit(0,a);submit(1,b);
        while(h_arvalid!=3)tick();stop[0]=1;tick(10);stop=0;result_ready=3;tick(40);
        check(admission_halted && halted==3 && quarantined==3 && contexts_held==5 && request_ready==0 && !cfg_ready,"SHARED_GLOBAL_STOP");
        check(received[1]==0 && received[2]==0,"STOP_GENERATED_DATA");passed();
        $display("SG ALL_PASS mode=full cases=6");$finish;
    end
    initial begin #200000;$fatal(1,"SG_WATCHDOG case=%0d",scenario);end
endmodule
