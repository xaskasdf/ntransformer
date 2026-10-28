`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_cache_query_r1;
    logic clk=0;always #2.5 clk=~clk;
    logic rst_n=0;
    logic bind_valid=0,bind_ready,engine_bind_valid,engine_bind_ready=1;
    logic [63:0] bind_sequence=101,bind_epoch=1;
    logic append_valid=0,append_ready,engine_append_valid,engine_append_ready=1;
    logic [31:0] append_layer=0,append_position=0;
    logic [63:0] append_sequence=101,append_epoch=1;
    logic command_valid=0,command_ready,sequencer_command_valid,sequencer_command_ready=1;
    logic sequencer_cache_lock=0;
    logic [31:0] command_layer=0;
    logic engine_lease_active=0;
    logic engine_done_valid=0,engine_done_ready=0,engine_done_bind=0;
    logic [3:0] engine_done_status=0;
    logic [12:0] engine_done_bytes=0;
    logic [31:0] engine_done_committed_tokens=0;
    logic [63:0] engine_done_sequence=0,engine_done_epoch=0;
    logic engine_cache_valid=0;
    logic [31:0] engine_cache_capacity=2;
    logic [63:0] engine_cache_sequence=101,engine_cache_epoch=1,expected_epoch=1;
    logic trusted_cache_valid,maintenance_busy,metadata_fault;
    logic [31:0] trusted_committed_tokens;
    logic [31:0] query_layer=0;
    wire query_valid;
    wire [31:0] query_committed_tokens;
    integer query_checks=0;
    wire ref_bind_ready;
    wire ref_engine_bind_valid;
    wire ref_append_ready;
    wire ref_engine_append_valid;
    wire ref_command_ready;
    wire ref_sequencer_command_valid;
    wire ref_trusted_cache_valid;
    wire [31:0] ref_trusted_committed_tokens;
    wire ref_maintenance_busy;
    wire ref_metadata_fault;
    ntrpu_attention_root_guard_query_r1 dut(.*);
    ntrpu_attention_root_guard_r1 reference(
        .clk(clk),
        .rst_n(rst_n),
        .bind_valid(bind_valid),
        .bind_ready(ref_bind_ready),
        .engine_bind_valid(ref_engine_bind_valid),
        .engine_bind_ready(engine_bind_ready),
        .bind_sequence(bind_sequence),
        .bind_epoch(bind_epoch),
        .append_valid(append_valid),
        .append_ready(ref_append_ready),
        .engine_append_valid(ref_engine_append_valid),
        .engine_append_ready(engine_append_ready),
        .append_layer(append_layer),
        .append_position(append_position),
        .append_sequence(append_sequence),
        .append_epoch(append_epoch),
        .command_valid(command_valid),
        .command_ready(ref_command_ready),
        .sequencer_command_valid(ref_sequencer_command_valid),
        .sequencer_command_ready(sequencer_command_ready),
        .sequencer_cache_lock(sequencer_cache_lock),
        .command_layer(command_layer),
        .engine_lease_active(engine_lease_active),
        .engine_done_valid(engine_done_valid),
        .engine_done_ready(engine_done_ready),
        .engine_done_bind(engine_done_bind),
        .engine_done_status(engine_done_status),
        .engine_done_bytes(engine_done_bytes),
        .engine_done_committed_tokens(engine_done_committed_tokens),
        .engine_done_sequence(engine_done_sequence),
        .engine_done_epoch(engine_done_epoch),
        .engine_cache_valid(engine_cache_valid),
        .engine_cache_capacity(engine_cache_capacity),
        .engine_cache_sequence(engine_cache_sequence),
        .engine_cache_epoch(engine_cache_epoch),
        .expected_epoch(expected_epoch),
        .trusted_cache_valid(ref_trusted_cache_valid),
        .trusted_committed_tokens(ref_trusted_committed_tokens),
        .maintenance_busy(ref_maintenance_busy),
        .metadata_fault(ref_metadata_fault)
    );
    integer seed=1,steps=0,cycles=0,resets=0;
    integer model[0:15];
    logic expected_valid=0,expected_fault=0;
    always @(posedge clk)begin
        cycles=cycles+1;
        if(cycles>20000)$fatal(1,"ROOTGUARD_FAIL global timeout");
        if(rst_n)begin
            if((engine_bind_valid && engine_bind_ready)+
               (engine_append_valid && engine_append_ready)+
               (sequencer_command_valid && sequencer_command_ready)>1)
                $fatal(1,"ROOTGUARD_FAIL double admission");
            if((maintenance_busy || sequencer_cache_lock || engine_lease_active || engine_done_valid) &&
               (bind_ready || append_ready || command_ready || engine_bind_valid ||
                engine_append_valid || sequencer_command_valid))
                $fatal(1,"ROOTGUARD_FAIL admitted while owned");
        end
    end
    task automatic tick;
        @(posedge clk);#1;@(negedge clk);
    endtask
    task automatic pause;
        repeat(1+((steps+seed)%4))tick();
    endtask
    function automatic [31:0] query_index(input integer index);
        case(index)
            16:query_index=32'd16;
            17:query_index=32'h10000000;
            18:query_index=32'h8000000f;
            19:query_index=32'hffffffff;
            default:query_index=32'(index);
        endcase
    endfunction
    task automatic snapshot(input integer tag);
        for(integer layer=0;layer<17;layer=layer+1)begin
            command_layer=layer;#1;
            if(trusted_cache_valid!==expected_valid || metadata_fault!==expected_fault ||
               trusted_committed_tokens!==(expected_valid && layer<16 ? model[layer]:0))
                $fatal(1,"ROOTGUARD_FAIL tag=%0d layer=%0d valid=%0d fault=%0d prefix=%0d expected=%0d",tag,layer,
                    trusted_cache_valid,metadata_fault,trusted_committed_tokens,layer<16?model[layer]:0);
            $display("ROOTGUARD_CHECK step=%0d tag=%0d layer=%0d valid=%0d fault=%0d prefix=%0d",steps,tag,layer,
                trusted_cache_valid,metadata_fault,trusted_committed_tokens);
            // The observation layer varies independently of the operation layer.
            for(integer qi=0;qi<20;qi++)begin
                query_layer=query_index(qi);#0.001;
                if(query_valid!==(expected_valid && qi<16) ||
                   query_committed_tokens!==(expected_valid && qi<16 ? model[qi]:0))
                    $fatal(1,"CACHEQUERY_FAIL independent query step=%0d command=%0d layer=%h valid=%0d prefix=%0d",steps,layer,query_layer,query_valid,query_committed_tokens);
                if({bind_ready,engine_bind_valid,append_ready,engine_append_valid,command_ready,sequencer_command_valid,trusted_cache_valid,trusted_committed_tokens,maintenance_busy,metadata_fault} !== {ref_bind_ready,ref_engine_bind_valid,ref_append_ready,ref_engine_append_valid,ref_command_ready,ref_sequencer_command_valid,ref_trusted_cache_valid,ref_trusted_committed_tokens,ref_maintenance_busy,ref_metadata_fault})
                    $fatal(1,"CACHEQUERY_FAIL legacy interface changed");
                $display("CACHEQUERY_CHECK step=%0d tag=%0d command=%0d layer=%08h valid=%0d prefix=%0d legacy_valid=%0d legacy_prefix=%0d reference_valid=%0d reference_prefix=%0d",steps,tag,layer,query_layer,query_valid,query_committed_tokens,trusted_cache_valid,trusted_committed_tokens,ref_trusted_cache_valid,ref_trusted_committed_tokens);
                query_checks++;
            end
        end
        steps=steps+1;command_layer=0;
    endtask
    task automatic clear_inputs;
        bind_valid=0;append_valid=0;command_valid=0;engine_done_valid=0;engine_done_ready=0;
        sequencer_cache_lock=0;engine_lease_active=0;engine_bind_ready=1;engine_append_ready=1;
    endtask
    task automatic reset;
        clear_inputs();rst_n=0;repeat(3)tick();rst_n=1;tick();
        expected_valid=0;expected_fault=0;for(integer i=0;i<16;i=i+1)model[i]=0;
        resets=resets+1;$display("ROOTGUARD_RESET count=%0d",resets);
    endtask
    task automatic start_bind;
        bind_valid=1;append_valid=1;command_valid=1;engine_bind_ready=0;#1;
        if(bind_ready || append_ready || command_ready || !engine_bind_valid || engine_append_valid || sequencer_command_valid)
            $fatal(1,"ROOTGUARD_FAIL bind priority under stall");
        pause();engine_bind_ready=1;#1;
        if(!bind_ready)$fatal(1,"ROOTGUARD_FAIL bind ready");
        tick();clear_inputs();engine_cache_valid=0;
        expected_valid=0;for(integer i=0;i<16;i=i+1)model[i]=0;
        if(!maintenance_busy)$fatal(1,"ROOTGUARD_FAIL bind ownership");
    endtask
    task automatic start_append(input integer layer,input integer position);
        append_layer=layer;append_position=position;append_sequence=bind_sequence;append_epoch=bind_epoch;
        append_valid=1;command_valid=1;engine_append_ready=0;#1;
        if(append_ready || command_ready || !engine_append_valid || sequencer_command_valid)
            $fatal(1,"ROOTGUARD_FAIL append priority under stall");
        pause();engine_append_ready=1;#1;
        if(!append_ready)$fatal(1,"ROOTGUARD_FAIL append ready");
        tick();clear_inputs();
        if(!maintenance_busy)$fatal(1,"ROOTGUARD_FAIL append ownership");
    endtask
    task automatic completion(input logic kind,input integer status,input integer bytes,input integer tokens);
        engine_done_bind=kind;engine_done_status=status;engine_done_bytes=bytes;
        engine_done_committed_tokens=tokens;engine_done_sequence=bind_sequence;engine_done_epoch=bind_epoch;
        engine_done_valid=1;engine_done_ready=0;
        bind_valid=1;append_valid=1;command_valid=1;
        pause();
        if(!maintenance_busy)$fatal(1,"ROOTGUARD_FAIL early release of held response");
        bind_valid=0;append_valid=0;command_valid=0;
    endtask
    task automatic consume;
        engine_done_ready=1;tick();engine_done_valid=0;engine_done_ready=0;
        if(maintenance_busy)$fatal(1,"ROOTGUARD_FAIL completion release");
    endtask
    task automatic good_bind;
        start_bind();engine_cache_valid=1;engine_cache_sequence=bind_sequence;engine_cache_epoch=bind_epoch;
        completion(1,0,0,0);consume();expected_valid=1;expected_fault=0;
    endtask
    task automatic new_identity;
        bind_epoch=bind_epoch+1;bind_sequence=bind_sequence+1;expected_epoch=bind_epoch;tick();
    endtask
    initial begin
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        $display("ROOTGUARD_CONFIG seed=%0d arithmetic_peer=0",seed);
        reset();snapshot(0);good_bind();snapshot(1);
        for(integer layer=0;layer<16;layer=layer+1)begin
            for(integer pos=0;pos<2;pos=pos+1)begin
                start_append(layer,pos);completion(0,0,4096,pos+1);
                snapshot(10+layer*2+pos);consume();model[layer]=pos+1;snapshot(50+layer*2+pos);
            end
        end
        // Rejected append must preserve every committed prefix.
        start_append(3,2);completion(0,1,0,2);consume();snapshot(100);
        start_append(16,0);completion(0,1,0,0);consume();snapshot(101);
        // Root lock protects all 32 heads, copies and held final result.
        command_valid=1;#1;
        if(!command_ready || !sequencer_command_valid)$fatal(1,"ROOTGUARD_FAIL root admission");
        tick();sequencer_cache_lock=1;bind_valid=1;append_valid=1;pause();snapshot(102);
        clear_inputs();engine_lease_active=1;bind_valid=1;append_valid=1;command_valid=1;pause();
        clear_inputs();snapshot(103);
        // Revocation is immediate, and an epoch rollback cannot restore access.
        expected_epoch=bind_epoch+1;#1;expected_valid=0;snapshot(104);tick();
        expected_epoch=bind_epoch;tick();snapshot(105);
        new_identity();good_bind();snapshot(106);
        start_append(5,0);completion(0,0,4096,1);consume();model[5]=1;snapshot(107);
        // Failure after partial AXI writes cannot publish another token.
        start_append(5,1);completion(0,3,64,1);consume();snapshot(108);
        // Mismatched done identity poisons metadata until a new successful bind.
        start_append(5,1);completion(0,0,4096,2);engine_done_sequence=999;consume();
        expected_valid=0;expected_fault=1;snapshot(109);
        new_identity();good_bind();snapshot(110);
        start_append(0,0);completion(0,0,4096,2);consume();expected_valid=0;expected_fault=1;snapshot(111);
        new_identity();good_bind();snapshot(112);
        start_append(0,0);completion(0,0,4032,1);consume();expected_valid=0;expected_fault=1;snapshot(113);
        new_identity();good_bind();snapshot(114);
        start_append(0,0);completion(1,0,4096,1);consume();expected_valid=0;expected_fault=1;snapshot(115);
        new_identity();good_bind();snapshot(116);
        // An epoch round-trip while maintenance is active remains revoked.
        start_append(0,0);expected_epoch=bind_epoch+1;tick();expected_epoch=bind_epoch;tick();
        completion(0,0,4096,1);consume();expected_valid=0;expected_fault=1;snapshot(117);
        new_identity();good_bind();snapshot(118);
        start_bind();completion(1,1,0,0);consume();snapshot(119);
        new_identity();start_bind();engine_cache_valid=1;engine_cache_sequence=bind_sequence;
        engine_cache_epoch=bind_epoch;completion(1,0,64,0);consume();expected_fault=1;snapshot(120);
        new_identity();good_bind();snapshot(121);
        engine_done_valid=1;engine_done_ready=1;tick();engine_done_valid=0;engine_done_ready=0;
        expected_valid=0;expected_fault=1;snapshot(122);
        new_identity();good_bind();snapshot(123);
        start_append(0,0);reset();snapshot(124);
        new_identity();good_bind();snapshot(125);
        start_bind();reset();snapshot(126);
        new_identity();good_bind();snapshot(127);
        sequencer_cache_lock=1;reset();snapshot(128);
        new_identity();good_bind();snapshot(129);
        $display("ROOTGUARD_PASS steps=%0d resets=%0d cycles=%0d",steps,resets,cycles);$display("CACHEQUERY_PASS seed=%0d checks=%0d steps=%0d",seed,query_checks,steps);$finish;
    end
endmodule
`default_nettype wire
