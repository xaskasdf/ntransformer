`timescale 1ns/1ps
// Public-output differential test; malformed events must retain both reservations.
module tb_ntrpu_peer_parallel;
    import ntrpu_peer_read_pkg::*;
    reg clk=0,rst_n=0,stop=0,link_up=1,reader_fault=0,revoke=0;
    always #2.5 clk=~clk;
    reg cfg_valid=0,request_valid=0,reject_ready=1,cmd_ready=0;
    reg data_valid=0,data_last=1,cpl_valid=0,dma_ready=0;
    reg dma_done_valid=0,ack_valid=0,result_ready=0;
    reg [511:0] data=0;
    reg [63:0] data_tag=0,cpl_tag=0;
    reg [1:0] cpl_status=0;
    reg [12:0] cpl_bytes=64;
    lease_t cfg;
    request_t request;
    event_t dma_done,ack;
    wire [1:0] cfg_ready,cfg_error,request_ready,reject_valid,cmd_valid;
    wire [1:0] data_ready,cpl_ready,dma_valid,dma_last,result_valid,halted,quarantined;
    wire [63:0] cmd_addr[0:1],cmd_tag[0:1],dma_address[0:1],result_bytes[0:1];
    wire [7:0] cmd_len[0:1];
    wire [3:0] reject_code[0:1],result_status[0:1];
    wire [1:0] contexts_held[0:1];
    wire [191:0] reject_identity[0:1];
    wire [1023:0] dma_context[0:1],result_context[0:1];
    wire [511:0] dma_data[0:1];
    wire [3338:0] observed[0:1];
    integer scenario=0,epoch=0,compared=0,commands=0,retired=0,paired_cycles=0;
    for(genvar g=0;g<2;g++)begin: dut
        if(g==0)begin: baseline
            ntrpu_peer_read_lane_reference peer(
                .clk(clk),.rst_n(rst_n),.stop(stop),.link_up(link_up),.reader_fault(reader_fault),.revoke(revoke),
                .cfg_valid(cfg_valid),.cfg(cfg),.cfg_ready(cfg_ready[g]),.cfg_error(cfg_error[g]),
                .request_valid(request_valid),.request(request),.request_ready(request_ready[g]),
                .reject_valid(reject_valid[g]),.reject_ready(reject_ready),.reject_code(reject_code[g]),.reject_identity(reject_identity[g]),
                .cmd_valid(cmd_valid[g]),.cmd_ready(cmd_ready),.cmd_addr(cmd_addr[g]),.cmd_tag(cmd_tag[g]),.cmd_len(cmd_len[g]),
                .data_valid(data_valid),.data_ready(data_ready[g]),.data_last(data_last),.data(data),.data_tag(data_tag),
                .cpl_valid(cpl_valid),.cpl_ready(cpl_ready[g]),.cpl_tag(cpl_tag),.cpl_status(cpl_status),.cpl_bytes(cpl_bytes),
                .dma_valid(dma_valid[g]),.dma_ready(dma_ready),.dma_last(dma_last[g]),.dma_data(dma_data[g]),
                .dma_context(dma_context[g]),.dma_address(dma_address[g]),
                .dma_done_valid(dma_done_valid),.dma_done(dma_done),.ack_valid(ack_valid),.ack(ack),
                .result_valid(result_valid[g]),.result_ready(result_ready),.result_context(result_context[g]),
                .result_status(result_status[g]),.result_bytes(result_bytes[g]),.contexts_held(contexts_held[g]),
                .halted(halted[g]),.quarantined(quarantined[g]));
        end else begin: candidate
            ntrpu_peer_read_lane peer(
                .clk(clk),.rst_n(rst_n),.stop(stop),.link_up(link_up),.reader_fault(reader_fault),.revoke(revoke),
                .cfg_valid(cfg_valid),.cfg(cfg),.cfg_ready(cfg_ready[g]),.cfg_error(cfg_error[g]),
                .request_valid(request_valid),.request(request),.request_ready(request_ready[g]),
                .reject_valid(reject_valid[g]),.reject_ready(reject_ready),.reject_code(reject_code[g]),.reject_identity(reject_identity[g]),
                .cmd_valid(cmd_valid[g]),.cmd_ready(cmd_ready),.cmd_addr(cmd_addr[g]),.cmd_tag(cmd_tag[g]),.cmd_len(cmd_len[g]),
                .data_valid(data_valid),.data_ready(data_ready[g]),.data_last(data_last),.data(data),.data_tag(data_tag),
                .cpl_valid(cpl_valid),.cpl_ready(cpl_ready[g]),.cpl_tag(cpl_tag),.cpl_status(cpl_status),.cpl_bytes(cpl_bytes),
                .dma_valid(dma_valid[g]),.dma_ready(dma_ready),.dma_last(dma_last[g]),.dma_data(dma_data[g]),
                .dma_context(dma_context[g]),.dma_address(dma_address[g]),
                .dma_done_valid(dma_done_valid),.dma_done(dma_done),.ack_valid(ack_valid),.ack(ack),
                .result_valid(result_valid[g]),.result_ready(result_ready),.result_context(result_context[g]),
                .result_status(result_status[g]),.result_bytes(result_bytes[g]),.contexts_held(contexts_held[g]),
                .halted(halted[g]),.quarantined(quarantined[g]));
        end
        assign observed[g]={cfg_ready[g],cfg_error[g],request_ready[g],reject_valid[g],reject_code[g],reject_identity[g],
            cmd_valid[g],cmd_addr[g],cmd_tag[g],cmd_len[g],data_ready[g],cpl_ready[g],dma_valid[g],dma_last[g],dma_data[g],
            dma_context[g],dma_address[g],result_valid[g],result_context[g],result_status[g],result_bytes[g],
            contexts_held[g],halted[g],quarantined[g]};
    end
    task automatic check(input bit ok,input string why);
        if(!ok)$fatal(1,"PAR_FAIL case=%0d %s",scenario,why);
    endtask
    task automatic tick(input integer n=1); repeat(n)@(negedge clk); endtask
    always @(posedge clk)begin
        if(rst_n)begin
            if(cmd_valid[0] && cmd_ready)begin
                commands++;
                check(cmd_tag[0]==commands && cmd_len[0]==0,"COMMAND_ORDER");
            end
            if(result_valid[0] && result_ready)begin
                retired++;
                $display("PAR RETIRE %0d %0d %0d %0d",scenario,retired,result_status[0],contexts_held[0]);
            end
            if(dma_done_valid && ack_valid)begin
                paired_cycles++;$display("PAR PAIRED %0d",scenario);
            end
            #0.001;
            check(observed[0]===observed[1],"PUBLIC_OUTPUT_DIFFERENCE");
            compared++;
        end
    end
    task automatic boot;
        rst_n=0;stop=0;link_up=1;reader_fault=0;revoke=0;cfg_valid=0;request_valid=0;
        cmd_ready=0;data_valid=0;cpl_valid=0;dma_ready=0;dma_done_valid=0;ack_valid=0;result_ready=0;
        data=0;data_tag=0;cpl_tag=0;cfg='0;request='0;dma_done='0;ack='0;
        commands=0;retired=0;epoch++;tick(3);rst_n=1;tick();
        cfg.source.session=epoch;cfg.source.node=0;cfg.source.slot=20;cfg.source.generation=1;
        cfg.target.session=epoch;cfg.target.node=1;cfg.target.slot=30;cfg.target.generation=1;
        cfg.source_size=65536;cfg.published_bytes=65536;cfg.target_size=65536;
        cfg.target_base=64'h10000000000;cfg.window_generation=1;cfg.window_bytes=65536;
        cfg.model_epoch=1;cfg.source_read=1;cfg.target_write=1;cfg.source_open=1;cfg.target_open=1;
        cfg_valid=1;check(cfg_ready==3,"CFG_READY");tick();cfg_valid=0;check(cfg_error==0,"CFG_ERROR");
    endtask
    task automatic submit(input integer seq);
        request='0;request.identity.session=epoch;request.identity.source=0;request.identity.sequence_id=seq;
        request.source=cfg.source;request.target=cfg.target;request.window_generation=1;request.model_epoch=1;
        request.bytes=64;request.source_offset=(seq-1)*64;request.target_offset=(seq-1)*128;
        request_valid=1;#0.001;check(request_ready==3,"REQUEST_READY");tick();request_valid=0;
        check(reject_valid==0,"UNEXPECTED_REJECT");
    endtask
    task automatic payload(input integer seq);
        data_tag=seq;data={8{64'h0123456789abcdef^seq}};data_valid=1;
        // Both variants must hold the offered payload through downstream stalls.
        dma_ready=0;tick(3);check(data_ready==0 && dma_valid==3,"PAYLOAD_STALL");
        dma_ready=1;tick();data_valid=0;
        cpl_tag=seq;cpl_valid=1;#0.001;check(cpl_ready==3,"CPL_READY");tick();cpl_valid=0;
    endtask
    function automatic event_t event_for(input integer seq);
        event_t e;
        e='0;e.identity.session=epoch;e.identity.source=0;e.identity.sequence_id=seq;
        e.target=cfg.target;e.window_generation=1;e.bytes=64;return e;
    endfunction
    task automatic notify(input integer kind,input event_t e);
        if(kind==0)begin dma_done=e;dma_done_valid=1;end
        else begin ack=e;ack_valid=1;end
        tick();dma_done_valid=0;ack_valid=0;
    endtask
    event_t e;
    initial begin
        for(integer event_kind=0;event_kind<2;event_kind++)
        for(integer seq=1;seq<=2;seq++)
        for(integer fault=0;fault<12;fault++)begin
            scenario++;boot();submit(1);submit(2);check(contexts_held[0]==2,"TWO_RESERVATIONS");
            if(fault!=9)begin cmd_ready=1;tick(2);cmd_ready=0;check(commands==2,"ISSUED_BOTH");end
            if(fault!=9 && fault!=10)begin payload(1);payload(2);end
            e=event_for(seq);
            case(fault)
                1:e.identity.session=epoch+1;
                2:e.identity.source=1;
                3:e.target.generation=2;
                4:e.window_generation=2;
                5:e.target.slot=31;
                6:e.bytes=128;
                7:e.status=1;
                8:e.identity.sequence_id=3;
                11:notify(event_kind,e);
            endcase
            notify(event_kind,e);
            if(fault==0)begin
                notify(1-event_kind,e);
                e=event_for(3-seq);notify(event_kind,e);notify(1-event_kind,e);
                tick(8);check(result_valid==3 && result_status[0]==0 && contexts_held[0]==2,"RESULT_STALL");
                result_ready=1;tick(6);check(retired==2 && contexts_held[0]==0 && halted==0,"SUCCESS_RELEASE");
            end else begin
                tick(4);check(halted==3 && quarantined==3 && contexts_held[0]==2,"FAULT_RETAINS");
                check(result_valid==3 && result_status[0]!=0,"FAULT_REPORT");
                result_ready=1;tick(8);
                check(retired==2 && contexts_held[0]==2 && request_ready==0 && cfg_ready==0,"FAILURE_RELEASED");
            end
            $display("PAR CASE %0d PASS event=%0d seq=%0d fault=%0d held=%0d halted=%0d",scenario,event_kind,seq,fault,contexts_held[0],halted[0]);
        end
        // Mixed simultaneous completions: acceptance must remain per context,
        // while any malformed event still has global failure priority.
        for(integer mode=0;mode<8;mode++)begin
            scenario++;boot();submit(1);submit(2);cmd_ready=1;tick(2);cmd_ready=0;
            check(commands==2,"PAIRED_ISSUED");payload(1);payload(2);
            if(mode==7)notify(0,event_for(1));
            dma_done=event_for(1);ack=event_for(2);
            case(mode)
                0:ack=event_for(1);
                1:dma_done=event_for(2);
                3:ack.identity.session=epoch+1;
                4:dma_done.identity.session=epoch+1;
                5:begin dma_done.status=1;ack.status=1;end
                6:begin dma_done=event_for(2);ack.identity.session=epoch+1;end
            endcase
            dma_done_valid=1;ack_valid=1;tick();dma_done_valid=0;ack_valid=0;
            if(mode<3)begin
                if(mode<2)begin e=event_for(mode==0?2:1);notify(0,e);notify(1,e);end
                else begin
                    dma_done=event_for(2);ack=event_for(1);dma_done_valid=1;ack_valid=1;
                    tick();dma_done_valid=0;ack_valid=0;
                end
                tick(8);check(result_valid==3 && result_status[0]==0 && contexts_held[0]==2,"PAIRED_RESULT_STALL");
                result_ready=1;tick(6);check(retired==2 && contexts_held[0]==0 && halted==0,"PAIRED_SUCCESS");
            end else begin
                tick(4);check(halted==3 && quarantined==3 && contexts_held[0]==2,"PAIRED_FAULT_RETAINS");
                check(result_valid==3 && result_status[0]!=0,"PAIRED_FAULT_REPORT");
                result_ready=1;tick(8);
                check(retired==2 && contexts_held[0]==2 && request_ready==0 && cfg_ready==0,"PAIRED_FAILURE_RELEASED");
            end
            $display("PAR CASE %0d PASS event=2 seq=0 fault=%0d held=%0d halted=%0d",scenario,mode,contexts_held[0],halted[0]);
        end
        check(paired_cycles==9,"MISSING_PAIRED_EVENTS");
        $display("PAR COMPARED cycles=%0d paired=%0d",compared,paired_cycles);
        $display("PAR ALL_PASS cases=56");$finish;
    end
    initial begin #1000000;$fatal(1,"PAR_WATCHDOG case=%0d",scenario);end
endmodule
