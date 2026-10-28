`timescale 1ns/1ps
// Compare ordered public transactions across a one-cycle context-commit stage.
module tb_ntrpu_peer_commit;
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
    integer scenario=0,epoch=0,compared=0,paired_cycles=0;
    integer commands[0:1],retired[0:1];
    for(genvar g=0;g<2;g++)begin: dut
        if(g==0)begin: baseline
            ntrpu_peer_commit_reference peer(
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
        if(!ok)$fatal(1,"CMT_FAIL case=%0d %s",scenario,why);
    endtask
    task automatic tick(input integer n=1); repeat(n)@(negedge clk); endtask
    reg [1:0] held_cmd=0,held_dma=0,held_result=0;
    reg [135:0] saved_cmd[0:1];
    reg [1600:0] saved_dma[0:1];
    reg [1091:0] saved_result[0:1];
    always @(posedge clk)begin
        if(!rst_n)begin
            held_cmd=0;held_dma=0;held_result=0;
        end else begin
            for(integer g=0;g<2;g++)begin
                if(held_cmd[g])check(cmd_valid[g] && saved_cmd[g]=={cmd_tag[g],cmd_addr[g],cmd_len[g]},"CMD_STABILITY");
                if(held_dma[g])check(dma_valid[g] && saved_dma[g]=={dma_context[g],dma_address[g],dma_data[g],dma_last[g]},"DMA_STABILITY");
                if(held_result[g])check(result_valid[g] && saved_result[g]=={result_context[g],result_status[g],result_bytes[g]},"RESULT_STABILITY");
                held_cmd[g]=cmd_valid[g]&&!cmd_ready;saved_cmd[g]={cmd_tag[g],cmd_addr[g],cmd_len[g]};
                held_dma[g]=dma_valid[g]&&!dma_ready;saved_dma[g]={dma_context[g],dma_address[g],dma_data[g],dma_last[g]};
                held_result[g]=result_valid[g]&&!result_ready;saved_result[g]={result_context[g],result_status[g],result_bytes[g]};
                if(request_valid && request_ready[g])
                    $display("CMT TX %0d %0d REQ %h",scenario,g,request);
                if(cmd_valid[g] && cmd_ready)begin
                    commands[g]++;
                    check(cmd_tag[g]==commands[g] && cmd_len[g]==0 && cmd_addr[g]==(cmd_tag[g]-1)*64,"COMMAND_ORDER_ADDRESS");
                    $display("CMT TX %0d %0d CMD %h",scenario,g,{cmd_tag[g],cmd_addr[g],cmd_len[g]});
                end
                if(dma_valid[g] && dma_ready)
                    $display("CMT TX %0d %0d DATA %h",scenario,g,{dma_context[g],dma_address[g],dma_data[g],dma_last[g]});
                if(reject_valid[g] && reject_ready)
                    $display("CMT TX %0d %0d REJECT %h",scenario,g,{reject_identity[g],reject_code[g]});
                if(result_valid[g] && result_ready)begin
                    retired[g]++;
                    $display("CMT TX %0d %0d RESULT %h",scenario,g,{result_context[g],result_status[g],result_bytes[g]});
                end
            end
            if(dma_done_valid && ack_valid)begin paired_cycles++;$display("CMT PAIRED %0d",scenario);end
            #0.001;compared++;
        end
    end
    task automatic same_state;
        tick(4);
        check(commands[0]==commands[1] && retired[0]==retired[1],"TRANSACTION_COUNTS");
        check(contexts_held[0]==contexts_held[1] && halted[0]==halted[1] && quarantined[0]==quarantined[1] && cfg_ready[0]==cfg_ready[1],"QUIESCENT_STATE");
    endtask
    task automatic boot;
        rst_n=0;stop=0;link_up=1;reader_fault=0;revoke=0;cfg_valid=0;request_valid=0;
        cmd_ready=0;data_valid=0;cpl_valid=0;dma_ready=0;dma_done_valid=0;ack_valid=0;result_ready=0;
        data=0;data_tag=0;cpl_tag=0;cfg='0;request='0;dma_done='0;ack='0;
        commands[0]=0;commands[1]=0;retired[0]=0;retired[1]=0;epoch++;tick(3);rst_n=1;tick();
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
            if(fault!=9)begin cmd_ready=1;tick(2);cmd_ready=0;check(commands[0]==2 && commands[1]==2,"ISSUED_BOTH");end
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
                result_ready=1;tick(6);check(retired[0]==2 && retired[1]==2 && contexts_held[0]==0 && halted==0,"SUCCESS_RELEASE");
            end else begin
                tick(4);check(halted==3 && quarantined==3 && contexts_held[0]==2,"FAULT_RETAINS");
                check(result_valid==3 && result_status[0]!=0,"FAULT_REPORT");
                result_ready=1;tick(8);
                check(retired[0]==2 && retired[1]==2 && contexts_held[0]==2 && request_ready==0 && cfg_ready==0,"FAILURE_RELEASED");
            end
            same_state();
            $display("CMT CASE %0d PASS event=%0d seq=%0d fault=%0d held=%0d halted=%0d",scenario,event_kind,seq,fault,contexts_held[0],halted[0]);
        end
        // Mixed simultaneous completions: acceptance must remain per context,
        // while any malformed event still has global failure priority.
        for(integer mode=0;mode<8;mode++)begin
            scenario++;boot();submit(1);submit(2);cmd_ready=1;tick(2);cmd_ready=0;
            check(commands[0]==2 && commands[1]==2,"PAIRED_ISSUED");payload(1);payload(2);
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
                result_ready=1;tick(6);check(retired[0]==2 && retired[1]==2 && contexts_held[0]==0 && halted==0,"PAIRED_SUCCESS");
            end else begin
                tick(4);check(halted==3 && quarantined==3 && contexts_held[0]==2,"PAIRED_FAULT_RETAINS");
                check(result_valid==3 && result_status[0]!=0,"PAIRED_FAULT_REPORT");
                result_ready=1;tick(8);
                check(retired[0]==2 && retired[1]==2 && contexts_held[0]==2 && request_ready==0 && cfg_ready==0,"PAIRED_FAILURE_RELEASED");
            end
            same_state();
            $display("CMT CASE %0d PASS event=2 seq=0 fault=%0d held=%0d halted=%0d",scenario,mode,contexts_held[0],halted[0]);
        end
        check(paired_cycles==9,"MISSING_PAIRED_EVENTS");
        // Faults while each of the two slots is in the commit stage.
        for(integer slot=0;slot<2;slot++)for(integer fault=0;fault<8;fault++)begin
            scenario++;boot();
            if(slot==1)begin submit(1);tick(2);end
            if(fault==7)begin ack=event_for(99);ack_valid=1;end
            submit(slot+1);
            check(contexts_held[1]==slot+1 && !cfg_ready[1],"STAGE_RESERVATION");
            case(fault)
                0:stop=1;
                1:link_up=0;
                2:reader_fault=1;
                3:begin dma_done=event_for(slot+1);dma_done_valid=1;end
                4:begin ack=event_for(slot+1);ack_valid=1;end
                5:begin cpl_tag=slot+1;cpl_valid=1;end
                6:begin data_tag=slot+1;data_valid=1;end
            endcase
            tick();stop=0;link_up=1;reader_fault=0;dma_done_valid=0;ack_valid=0;cpl_valid=0;data_valid=0;
            tick(5);check(halted==3 && quarantined==3 && result_valid==3,"STAGED_FAULT_REPORT");
            check(contexts_held[1]==slot+1 && request_ready==0 && cfg_ready==0,"STAGED_FAULT_OWNERSHIP");
            result_ready=1;tick(9);
            check(retired[0]==slot+1 && retired[1]==slot+1 && contexts_held[1]==slot+1,"STAGED_FAILURE_RELEASED");
            same_state();$display("CMT CASE %0d PASS stage_fault=%0d slot=%0d",scenario,fault,slot);
        end
        // The very next edge must see the reservation and sequence watermark.
        for(integer kind=0;kind<2;kind++)begin
            scenario++;boot();submit(1);
            if(kind==0)request.identity.sequence_id=2;
            request_valid=1;reject_ready=0;tick();request_valid=0;
            check(reject_valid==3 && reject_code[0]==(kind==0?OVERLAP:REPLAY) && reject_code[1]==reject_code[0],"PENDING_CONFLICT");
            check(contexts_held[1]==1,"PENDING_CONFLICT_RESERVED");
            tick(3);reject_ready=1;tick();same_state();
            $display("CMT CASE %0d PASS pending_conflict=%0d",scenario,kind);
        end
        // Accepted work survives a subsequent revoke; new work does not.
        scenario++;boot();submit(1);revoke=1;tick();revoke=0;
        request.identity.sequence_id=2;request.source_offset=64;request.target_offset=128;
        reject_ready=0;request_valid=1;tick();request_valid=0;
        check(reject_valid==3 && reject_code[0]==PERMISSION && reject_code[1]==PERMISSION,"REVOKE_NEW_WORK");
        reject_ready=1;cmd_ready=1;tick(3);cmd_ready=0;payload(1);notify(0,event_for(1));notify(1,event_for(1));
        result_ready=1;tick(6);check(retired[0]==1 && retired[1]==1 && contexts_held[1]==0 && halted==0,"REVOKKE_ACCEPTED_WORK");
        same_state();$display("CMT CASE %0d PASS revoke_after_capture=1",scenario);
        // No gap is needed between captures, or between commit and replacement.
        scenario++;boot();cmd_ready=1;submit(1);
        check(cmd_valid[0] && !cmd_valid[1] && contexts_held[1]==1,"ONE_CYCLE_STAGE");
        submit(2);request='1;tick(4);cmd_ready=0;
        check(commands[0]==2 && commands[1]==2 && contexts_held[1]==2,"CONSECUTIVE_CAPTURE");
        payload(1);payload(2);notify(0,event_for(1));notify(1,event_for(1));notify(0,event_for(2));notify(1,event_for(2));
        result_ready=1;tick(8);check(retired[0]==2 && retired[1]==2 && contexts_held[1]==0,"CONSECUTIVE_RETIRE");
        same_state();$display("CMT CASE %0d PASS consecutive_capture=1",scenario);
        // Config cannot alter addresses between reservation and commit.
        scenario++;boot();submit(1);cfg_valid=1;cfg.source_base=4096;cfg.target_base=64'h20000000000;
        cfg.source.generation=2;cfg.target.generation=2;cfg.window_generation=2;
        tick(3);check(cfg_ready==0 && cfg_error==0,"STAGE_CONFIG_BLOCK");cfg_valid=0;
        cfg.source_base=0;cfg.target_base=64'h10000000000;cfg.source.generation=1;cfg.target.generation=1;cfg.window_generation=1;
        cmd_ready=1;tick(2);cmd_ready=0;payload(1);notify(0,event_for(1));notify(1,event_for(1));result_ready=1;tick(6);
        same_state();$display("CMT CASE %0d PASS config_during_stage=1",scenario);
        // Revoke in the handshake cycle still rejects, without a reservation.
        scenario++;boot();revoke=1;
        request='0;request.identity.session=epoch;request.identity.sequence_id=1;
        request.source=cfg.source;request.target=cfg.target;request.window_generation=1;request.model_epoch=1;request.bytes=64;
        request_valid=1;reject_ready=0;tick();request_valid=0;revoke=0;
        check(reject_valid==3 && reject_code[0]==PERMISSION && reject_code[1]==PERMISSION && contexts_held[1]==0,"REVOKE_AT_CAPTURE");
        reject_ready=1;tick();same_state();$display("CMT CASE %0d PASS revoke_at_capture=1",scenario);
        // Full slots hold an offered request through backpressure, without recapture.
        scenario++;boot();submit(1);submit(2);request.identity.sequence_id=3;request.source_offset=128;request.target_offset=256;request_valid=1;
        tick(5);check(request_ready==0 && contexts_held[1]==2,"CAPACITY_BACKPRESSURE");request_valid=0;
        cmd_ready=1;tick(3);cmd_ready=0;payload(1);payload(2);notify(0,event_for(1));notify(1,event_for(1));notify(0,event_for(2));notify(1,event_for(2));
        result_ready=1;tick(8);same_state();$display("CMT CASE %0d PASS capacity_backpressure=1",scenario);
        // Cold reset is the only operation that discards the occupied stage.
        scenario++;boot();submit(1);rst_n=0;tick(3);rst_n=1;tick(4);
        check(contexts_held[0]==0 && contexts_held[1]==0 && cmd_valid==0 && result_valid==0 && halted==0,"STAGE_COLD_RESET");
        same_state();$display("CMT CASE %0d PASS cold_reset=1",scenario);
        check(scenario==80,"CASE_COUNT");
        $display("CMT COMPARED cycles=%0d paired=%0d",compared,paired_cycles);
        $display("CMT ALL_PASS cases=80");$finish;
    end
    initial begin #1000000;$fatal(1,"CMT_WATCHDOG case=%0d",scenario);end
endmodule
