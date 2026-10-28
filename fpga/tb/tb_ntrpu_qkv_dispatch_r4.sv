// Real KV append, metadata guard and 32-head sequencer; synthetic snapshot/head arithmetic.
`timescale 1ns/1ps
module tb_ntrpu_qkv_dispatch_r4;
    logic  clk;
    logic  rst_n;
    logic  command_valid;
    wire  command_ready;
    logic [63:0] command_sequence;
    logic [63:0] command_epoch;
    logic [63:0] command_lease_base;
    logic [63:0] expected_epoch;
    logic [31:0] command_layer;
    logic [31:0] command_position;
    logic  cancel_valid;
    wire  cancel_ready;
    wire  busy;
    wire  cache_exclusive;
    wire  cache_idle;
    wire  cache_valid;
    wire  cache_metadata_fault;
    wire [63:0] cache_sequence;
    wire [63:0] cache_epoch;
    wire [31:0] cache_capacity;
    wire [31:0] cache_committed_tokens;
    logic  source_completion_valid;
    wire  source_completion_ready;
    logic [3:0] source_status;
    logic [63:0] source_sequence;
    logic [63:0] source_epoch;
    logic [31:0] source_layer;
    logic [31:0] source_position;
    logic [11:0] source_qk_words;
    logic [5:0] source_v_beats;
    logic [7:0] source_frames;
    logic  source_result_valid;
    wire  source_ce;
    wire [7:0] source_address;
    logic [511:0] source_q;
    logic  source_q_valid;
    wire  append_valid;
    wire  append_ready;
    wire  append_prepared;
    wire [31:0] append_layer;
    wire [31:0] append_position;
    wire [63:0] append_sequence;
    wire [63:0] append_epoch;
    wire  data_valid;
    wire  data_ready;
    wire [511:0] data_bits;
    wire [5:0] data_index;
    wire  data_last;
    wire [63:0] data_sequence;
    wire [63:0] data_epoch;
    wire  commit_valid;
    wire  commit_ready;
    wire [3:0] commit_status;
    wire [63:0] commit_sequence;
    wire [63:0] commit_epoch;
    wire  append_cancel_valid;
    wire  append_cancel_ready;
    wire  append_done_valid;
    wire  append_done_ready;
    wire  append_done_bind;
    wire [3:0] append_done_status;
    wire [12:0] append_done_bytes;
    wire [31:0] append_done_committed_tokens;
    wire [63:0] append_done_sequence;
    wire [63:0] append_done_epoch;
    wire  root_command_valid;
    wire  root_command_ready;
    wire [31:0] root_command_layer;
    wire [31:0] root_command_position;
    wire [63:0] root_command_sequence;
    wire [63:0] root_command_epoch;
    wire [63:0] root_command_lease_base;
    wire  q_valid;
    wire  q_ready;
    wire [511:0] q_data;
    wire [7:0] q_index;
    wire  q_last;
    wire [63:0] q_sequence;
    wire [63:0] q_epoch;
    wire  root_cancel_valid;
    wire  root_cancel_ready;
    wire  root_completion_valid;
    wire  root_completion_ready;
    wire [3:0] root_status;
    wire [63:0] root_sequence;
    wire [63:0] root_epoch;
    wire  root_result_valid;
    wire  root_result_ce;
    wire [10:0] root_result_address;
    wire [31:0] root_result_q;
    wire  root_result_q_valid;
    wire  completion_valid;
    logic  completion_ready;
    wire [3:0] completion_status;
    wire [3:0] completion_append_status;
    wire [3:0] completion_root_status;
    wire [63:0] completion_sequence;
    wire [63:0] completion_epoch;
    wire [31:0] completion_layer;
    wire [31:0] completion_position;
    wire [6:0] completion_kv_frames;
    wire [7:0] completion_q_frames;
    wire [12:0] completion_kv_bytes;
    wire [31:0] completion_kv_tokens;
    wire  completion_kv_committed;
    wire  result_valid;
    wire  completion_revoked;
    logic  result_ce;
    logic [10:0] result_address;
    wire [31:0] result_q;
    wire  result_q_valid;
    wire  maintenance_allowed;
    wire  engine_bind_valid;
    wire  engine_bind_ready;
    logic [63:0] bind_cache_base;
    logic [63:0] bind_region_base;
    logic [63:0] bind_region_bytes;
    logic [31:0] bind_capacity;
    logic [63:0] bind_sequence;
    logic [63:0] bind_epoch;
    wire  engine_cache_valid;
    wire [63:0] engine_cache_base;
    wire [63:0] engine_cache_region_base;
    wire [63:0] engine_cache_region_bytes;
    wire [3:0] inspect_layer;
    wire [31:0] engine_committed_tokens;
    wire  engine_append_valid;
    wire  engine_append_ready;
    wire  engine_done_ready;
    wire [63:0] m_axi_awaddr;
    wire [7:0] m_axi_awlen;
    wire [2:0] m_axi_awsize;
    wire [2:0] m_axi_awprot;
    wire [1:0] m_axi_awburst;
    wire [3:0] m_axi_awid;
    wire [3:0] m_axi_awcache;
    wire  m_axi_awlock;
    wire  m_axi_awvalid;
    logic  m_axi_awready;
    wire [511:0] m_axi_wdata;
    wire [63:0] m_axi_wstrb;
    wire  m_axi_wlast;
    wire  m_axi_wvalid;
    logic  m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic  m_axi_bvalid;
    wire  m_axi_bready;
    logic  bind_valid;
    wire  bind_ready;
    wire  sequencer_command_valid;
    wire  sequencer_command_ready;
    wire  seq_cache_lock;
    wire  engine_lease_active;
    wire  guard_maintenance_busy;
    wire  guard_metadata_fault;
    wire  head_command_valid;
    logic  head_command_ready;
    wire [2047:0] head_command_query;
    wire [31:0] head_command_layer;
    wire [31:0] head_command_position;
    wire [31:0] head_command_head;
    wire [63:0] head_command_sequence;
    wire [63:0] head_command_epoch;
    wire [63:0] head_command_lease;
    wire  head_cancel_valid;
    logic  head_cancel_ready;
    logic  head_completion_valid;
    wire  head_completion_ready;
    logic [3:0] head_completion_status;
    logic  head_result_valid;
    wire  head_result_ce;
    wire [5:0] head_result_address;
    logic [31:0] head_result_q;
    logic  head_result_q_valid;
    ntrpu_qkv_dispatch_r1 #(.WATCHDOG_CYCLES(64)) dut(.*);
    ntrpu_kv_append_r1 #(.WATCHDOG_CYCLES(512)) kv(
        .clk(clk),
        .rst_n(rst_n),
        .maintenance_allowed(maintenance_allowed),
        .bind_valid(engine_bind_valid),
        .bind_ready(engine_bind_ready),
        .bind_cache_base(bind_cache_base),
        .bind_region_base(bind_region_base),
        .bind_region_bytes(bind_region_bytes),
        .bind_capacity(bind_capacity),
        .bind_sequence(bind_sequence),
        .bind_epoch(bind_epoch),
        .expected_epoch(expected_epoch),
        .cache_valid(engine_cache_valid),
        .cache_base(engine_cache_base),
        .cache_region_base(engine_cache_region_base),
        .cache_region_bytes(engine_cache_region_bytes),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .cache_capacity(cache_capacity),
        .inspect_layer(inspect_layer),
        .committed_tokens(engine_committed_tokens),
        .cmd_valid(engine_append_valid),
        .cmd_ready(engine_append_ready),
        .prepared(append_prepared),
        .cmd_layer(append_layer),
        .cmd_position(append_position),
        .cmd_sequence(append_sequence),
        .cmd_epoch(append_epoch),
        .data_valid(data_valid),
        .data_ready(data_ready),
        .data_bits(data_bits),
        .data_index(data_index),
        .data_last(data_last),
        .data_sequence(data_sequence),
        .data_epoch(data_epoch),
        .commit_valid(commit_valid),
        .commit_ready(commit_ready),
        .commit_status(commit_status),
        .commit_sequence(commit_sequence),
        .commit_epoch(commit_epoch),
        .cancel_valid(append_cancel_valid),
        .cancel_ready(append_cancel_ready),
        .done_valid(append_done_valid),
        .done_ready(engine_done_ready),
        .done_bind(append_done_bind),
        .done_status(append_done_status),
        .done_bytes(append_done_bytes),
        .done_committed_tokens(append_done_committed_tokens),
        .done_sequence(append_done_sequence),
        .done_epoch(append_done_epoch),
        .m_axi_awaddr(m_axi_awaddr),
        .m_axi_awlen(m_axi_awlen),
        .m_axi_awsize(m_axi_awsize),
        .m_axi_awprot(m_axi_awprot),
        .m_axi_awburst(m_axi_awburst),
        .m_axi_awid(m_axi_awid),
        .m_axi_awcache(m_axi_awcache),
        .m_axi_awlock(m_axi_awlock),
        .m_axi_awvalid(m_axi_awvalid),
        .m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),
        .m_axi_wstrb(m_axi_wstrb),
        .m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),
        .m_axi_wready(m_axi_wready),
        .m_axi_bid(m_axi_bid),
        .m_axi_bresp(m_axi_bresp),
        .m_axi_bvalid(m_axi_bvalid),
        .m_axi_bready(m_axi_bready)
    );
    ntrpu_attention_root_guard_r1 guard(
        .clk(clk),
        .rst_n(rst_n),
        .bind_valid(bind_valid),
        .bind_ready(bind_ready),
        .engine_bind_valid(engine_bind_valid),
        .engine_bind_ready(engine_bind_ready),
        .bind_sequence(bind_sequence),
        .bind_epoch(bind_epoch),
        .append_valid(append_valid),
        .append_ready(append_ready),
        .engine_append_valid(engine_append_valid),
        .engine_append_ready(engine_append_ready),
        .append_layer(append_layer),
        .append_position(append_position),
        .append_sequence(append_sequence),
        .append_epoch(append_epoch),
        .command_valid(root_command_valid),
        .command_ready(root_command_ready),
        .sequencer_command_valid(sequencer_command_valid),
        .sequencer_command_ready(sequencer_command_ready),
        .sequencer_cache_lock(seq_cache_lock),
        .command_layer(root_command_layer),
        .engine_lease_active(engine_lease_active),
        .engine_done_valid(append_done_valid),
        .engine_done_ready(engine_done_ready),
        .engine_done_bind(append_done_bind),
        .engine_done_status(append_done_status),
        .engine_done_bytes(append_done_bytes),
        .engine_done_committed_tokens(append_done_committed_tokens),
        .engine_done_sequence(append_done_sequence),
        .engine_done_epoch(append_done_epoch),
        .engine_cache_valid(engine_cache_valid),
        .engine_cache_capacity(cache_capacity),
        .engine_cache_sequence(cache_sequence),
        .engine_cache_epoch(cache_epoch),
        .expected_epoch(expected_epoch),
        .trusted_cache_valid(cache_valid),
        .trusted_committed_tokens(cache_committed_tokens),
        .maintenance_busy(guard_maintenance_busy),
        .metadata_fault(guard_metadata_fault)
    );
    ntrpu_attention_head_sequencer_r1 #(.WATCHDOG_CYCLES(512)) seq(
        .clk(clk),
        .rst_n(rst_n),
        .command_valid(sequencer_command_valid),
        .command_ready(sequencer_command_ready),
        .command_layer(root_command_layer),
        .command_position(root_command_position),
        .command_sequence(root_command_sequence),
        .command_epoch(root_command_epoch),
        .command_lease_base(root_command_lease_base),
        .cache_idle(cache_idle),
        .cache_valid(cache_valid),
        .cache_sequence(cache_sequence),
        .cache_epoch(cache_epoch),
        .expected_epoch(expected_epoch),
        .cache_capacity(cache_capacity),
        .cache_committed_tokens(cache_committed_tokens),
        .cache_lock(seq_cache_lock),
        .q_valid(q_valid),
        .q_ready(q_ready),
        .q_data(q_data),
        .q_index(q_index),
        .q_last(q_last),
        .q_sequence(q_sequence),
        .q_epoch(q_epoch),
        .cancel_valid(root_cancel_valid || root_revoke),
        .cancel_ready(root_cancel_ready),
        .completion_valid(root_completion_valid),
        .completion_ready(root_completion_ready),
        .completion_status(root_status),
        .completion_sequence(root_sequence),
        .completion_epoch(root_epoch),
        .result_valid(root_result_valid),
        .result_ce(root_result_ce),
        .result_address(root_result_address),
        .result_q(root_result_q),
        .result_q_valid(root_result_q_valid),
        .head_command_valid(head_command_valid),
        .head_command_ready(head_command_ready),
        .head_command_query(head_command_query),
        .head_command_layer(head_command_layer),
        .head_command_position(head_command_position),
        .head_command_head(head_command_head),
        .head_command_sequence(head_command_sequence),
        .head_command_epoch(head_command_epoch),
        .head_command_lease(head_command_lease),
        .head_cancel_valid(head_cancel_valid),
        .head_cancel_ready(head_cancel_ready),
        .head_completion_valid(head_completion_valid),
        .head_completion_ready(head_completion_ready),
        .head_completion_status(head_completion_status),
        .head_result_valid(head_result_valid),
        .head_result_ce(head_result_ce),
        .head_result_address(head_result_address),
        .head_result_q(head_result_q),
        .head_result_q_valid(head_result_q_valid)
    );

    always #2.5 clk=~clk;
    logic bind_phase=0,bind_response_ready=0;
    logic head_active=0,head_owned=0,public_ack=0,admitted=0;
    logic injected=0,reset_case=0,root_revoke=0;
    integer cancel_events=0,last_append_status=0,last_append_bytes=0,last_append_tokens=0;
    integer total_resets=0,total_revocations=0,total_errors=0;
    integer wanted_status=0,wanted_source_acks=1;
    integer head_delay=0,peer_head=0;
    logic [2047:0] peer_query;
    assign maintenance_allowed=!busy && !guard_maintenance_busy && !seq_cache_lock;
    assign engine_done_ready=bind_phase ? bind_response_ready:append_done_ready;
    assign engine_lease_active=head_active;
    assign cache_idle=!guard_maintenance_busy && !seq_cache_lock && !engine_lease_active && !append_done_valid;
    assign cache_metadata_fault=guard_metadata_fault;
    assign inspect_layer=root_command_layer[3:0];
    integer seed=1,random_state,job=0,mode=0,cycle=0,job_cycles=0;
    integer input_frames=0,q_frames=0,source_reads=0,heads=0,head_acks=0,public_words=0,source_acks=0;
    integer aws=0,ws=0,bs=0,planned=0,aw_beats=0,w_burst=0;
    integer append_cycle=0,commit_cycle=0,append_ack_cycle=0,root_cycle=0,first_aw_cycle=0;
    integer totals_jobs=0,total_frames=0,total_heads=0,total_words=0;
    logic aw_pending=0,aw_hold=0,w_hold=0;
    logic [88:0] saved_aw;
    logic [576:0] saved_w;
    logic [511:0] committed_memory[0:63];
    logic [511:0] pending_words[0:3];
    logic [63:0] seq_model,epoch_model,lease_model,base_model;
    logic [31:0] layer_model,position_model,capacity_model;
    function automatic logic [31:0] word(input integer index);
        logic [22:0] mantissa;
        mantissa=23'(index*131+job*977);
        case(index%8)
            0:return 32'h00000000;
            1:return 32'h80000000;
            2:return {9'd0,mantissa}|32'd1;
            3:return 32'h007fffff;
            4:return 32'h3f000000|{9'd0,mantissa};
            5:return 32'hbf000000|{9'd0,mantissa};
            6:return 32'h7f7fffff;
            default:return 32'hff7fffff;
        endcase
    endfunction
    function automatic logic [511:0] source_frame(input integer index);
        logic [511:0] data;
        for(integer lane=0;lane<16;lane++)data[lane*32+:32]=word(index*16+lane);
        if((mode==30 && index==128) || (mode==31 && index==127))data[31:0]=32'h7fc00001;
        return data;
    endfunction
    function automatic logic [63:0] location(input integer beat);
        return base_model+64'(((int'(layer_model)*16+beat/4)*int'(capacity_model)+int'(position_model))*256+(beat%4)*64);
    endfunction
    task automatic require(input bit good,input string why);
        if(!good)$fatal(1,"DISPATCH_FAIL job=%0d mode=%0d cycle=%0d %s",job,mode,cycle,why);
    endtask
    always @(posedge clk)begin
        if(!rst_n)begin source_q_valid<=0;head_result_q_valid<=0;end
        else begin
            source_q_valid<=source_ce && !((mode==32 && source_address==128) || (mode==33 && source_address==0));
            if(source_ce)source_q<=source_frame(int'(source_address));
            head_result_q_valid<=head_result_ce && head_result_valid;
            if(head_result_ce)head_result_q<=(mode==37 && peer_head==0 && head_result_address==17) ? 32'h7fc00001:(peer_query[head_result_address*32+:32]^32'h00000100);
        end
    end
    task automatic step;
        logic af,wf,bf,hf,hack,srcack,bindhs,hcancel;
        integer want_burst;
        @(negedge clk);
        if(admitted && !injected)begin
            if((mode==40 && job_cycles==0) || (mode==41 && input_frames>=17) ||
                (mode==42 && aw_pending) || (mode==43 && append_ack_cycle>0 && root_cycle==0) ||
                (mode==44 && q_frames>=17) || (mode==45 && head_active && heads==4))begin
                cancel_valid=1;injected=1;
                $display("DISPATCH_INJECT cycle=%0d job=%0d kind=cancel kv=%0d q=%0d heads=%0d",cycle,job,input_frames,q_frames,heads);
            end else if(mode==46 && input_frames>=17)begin
                expected_epoch=epoch_model^64'h0100000000000000;injected=1;
                $display("DISPATCH_INJECT cycle=%0d job=%0d kind=epoch kv=%0d q=%0d heads=%0d",cycle,job,input_frames,q_frames,heads);
            end else if(mode==47 && q_frames>=17)begin
                source_result_valid=0;injected=1;
                $display("DISPATCH_INJECT cycle=%0d job=%0d kind=source_valid kv=%0d q=%0d heads=%0d",cycle,job,input_frames,q_frames,heads);
            end else if(mode==48 && q_frames>=17)begin
                source_sequence=seq_model^64'h0100000000000000;injected=1;
                $display("DISPATCH_INJECT cycle=%0d job=%0d kind=source_identity kv=%0d q=%0d heads=%0d",cycle,job,input_frames,q_frames,heads);
            end
        end
        m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
        head_command_ready=!head_active && $urandom_range(0,3)!=0;
        head_cancel_ready=head_active && !head_completion_valid;
        if(!m_axi_bvalid && aw_pending && w_burst==aw_beats && $urandom_range(0,3)!=0)begin m_axi_bvalid=1;m_axi_bresp=((mode==34 && bs==0) || (mode==35 && planned==64)) ? 2:0;m_axi_bid=0;end
        if(head_active && !head_completion_valid)begin
            if(head_delay>0)head_delay--;
            else begin head_completion_valid=1;head_result_valid=mode!=36;head_completion_status=mode==36 ? 5:0;end
        end
        @(posedge clk);cycle++;if(admitted)job_cycles++;
        af=rst_n&&m_axi_awvalid&&m_axi_awready;wf=rst_n&&m_axi_wvalid&&m_axi_wready;bf=rst_n&&m_axi_bvalid&&m_axi_bready;
        hf=rst_n&&head_command_valid&&head_command_ready;hack=rst_n&&head_completion_valid&&head_completion_ready;
        srcack=rst_n&&source_completion_valid&&source_completion_ready;bindhs=rst_n&&bind_valid&&bind_ready;
        hcancel=rst_n&&head_cancel_valid&&head_cancel_ready;
        if(rst_n && cancel_valid && cancel_ready)cancel_events++;
        if(rst_n)begin
            if(aw_hold)require(m_axi_awvalid && saved_aw=={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock},"Stable offered AW");
            if(w_hold)require(m_axi_wvalid && saved_w=={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"Stable offered W");
            if(command_valid && command_ready)begin
                require(!admitted,"Single admission");admitted=1;command_valid<=0;
                $display("DISPATCH_ADMIT cycle=%0d job=%0d sequence=%016h epoch=%016h layer=%0d position=%0d lease=%016h",cycle,job,command_sequence,command_epoch,command_layer,command_position,command_lease_base);
            end
            if(append_valid && append_ready)begin
                require(admitted && append_cycle==0 && input_frames==0 && root_cycle==0,"Append admission");
                require(append_sequence==seq_model && append_epoch==epoch_model && append_layer==layer_model && append_position==position_model,"Append descriptor");
                append_cycle=cycle;$display("DISPATCH_APPEND cycle=%0d job=%0d",cycle,job);
            end
            if(source_ce)begin
                require(source_completion_valid && source_result_valid && source_address<192 && !public_ack,"Owned snapshot read");
                require(source_address==(root_cycle>0 ? q_frames:128+input_frames),"K/V then Q RAM addressing");
                $display("DISPATCH_RAM cycle=%0d job=%0d address=%0d data=%0128h",cycle,job,source_address,source_frame(int'(source_address)));source_reads++;
            end
            if(data_valid && data_ready)begin
                for(integer lane=0;lane<16;lane++)require(data_bits[lane*32+23+:8]!=8'hff,"Dispatcher rejects nonfinite frame before forwarding");
                require(input_frames<64 && data_index==input_frames && data_last==(input_frames==63),"KV frame framing");
                require(data_sequence==seq_model && data_epoch==epoch_model && data_bits===source_frame(128+input_frames),"KV exact bytes");
                $display("DISPATCH_KV cycle=%0d job=%0d index=%0d data=%0128h",cycle,job,data_index,data_bits);input_frames++;
            end
            if(commit_valid && commit_ready)begin
                require((commit_status!=0 || input_frames==64) && commit_sequence==seq_model && commit_epoch==epoch_model,"Complete KV commit or explicit producer error");
                require(commit_cycle==0,"Single commit");commit_cycle=cycle;$display("DISPATCH_COMMIT cycle=%0d job=%0d status=%0d frames=%0d",cycle,job,commit_status,input_frames);
            end
            if(af)begin
                require(commit_cycle>0 && !aw_pending,"No write before complete producer commit");
                want_burst=4-planned%4;if(want_burst>64-int'((location(planned)>>6)&63))want_burst=64-int'((location(planned)>>6)&63);
                require(m_axi_awaddr==location(planned) && int'(m_axi_awlen)+1==want_burst,"Physical KV layout and page bounds");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && !m_axi_awlock && m_axi_awprot==0,"AW attributes");
                if(first_aw_cycle==0)first_aw_cycle=cycle;
                $display("DISPATCH_AW cycle=%0d job=%0d address=%016h beats=%0d",cycle,job,m_axi_awaddr,want_burst);
                aw_pending=1;aw_beats=want_burst;planned+=want_burst;aws++;
            end
            if(wf)begin
                require(commit_cycle>0 && ws<64 && w_burst<4,"W after commit");
                require(m_axi_wdata===source_frame(128+ws) && m_axi_wstrb==64'hffffffffffffffff,"Exact staged KV write");
                require(m_axi_wlast==((ws%4)==3 || ((location(ws)>>6)&63)==63),"WLAST framing");
                pending_words[w_burst]=m_axi_wdata;w_burst++;ws++;
                $display("DISPATCH_W cycle=%0d job=%0d index=%0d data=%0128h",cycle,job,ws-1,m_axi_wdata);
            end
            if(bf)begin
                require(aw_pending && w_burst==aw_beats,"B owns complete AW/W burst");
                if(m_axi_bresp==0)for(integer i=0;i<aw_beats;i++)committed_memory[planned-aw_beats+i]=pending_words[i];
                $display("DISPATCH_B cycle=%0d job=%0d beats=%0d response=%0d",cycle,job,aw_beats,m_axi_bresp);bs++;
            end
            if(append_done_valid && append_done_ready)begin
                require(!append_done_bind && append_done_sequence==seq_model && append_done_epoch==epoch_model && bs==aws && !aw_pending && w_burst==0,"KV done identity and drain");
                if(append_done_status==0)require(input_frames==64 && ws==64 && append_done_bytes==4096 && append_done_committed_tokens==position_model+1,"Successful KV complete before root");
                else require(append_done_committed_tokens==position_model,"Failed append leaves prefix unchanged");
                last_append_status=append_done_status;last_append_bytes=append_done_bytes;last_append_tokens=append_done_committed_tokens;
                append_ack_cycle=cycle;$display("DISPATCH_KV_ACK cycle=%0d job=%0d status=%0d bytes=%0d tokens=%0d",cycle,job,append_done_status,append_done_bytes,append_done_committed_tokens);
            end
            if(root_command_valid && root_command_ready)begin
                require(append_ack_cycle>0 && cycle>append_ack_cycle && cache_committed_tokens==position_model+1 && root_cycle==0,"Root only after acknowledged trusted prefix");
                require(root_command_sequence==seq_model && root_command_epoch==epoch_model && root_command_layer==layer_model && root_command_position==position_model && root_command_lease_base==lease_model,"Root descriptor");
                for(integer i=0;i<64;i++)require(committed_memory[i]===source_frame(128+i),"All KV bytes B-committed");
                root_cycle=cycle;$display("DISPATCH_ROOT cycle=%0d job=%0d prefix=%0d",cycle,job,cache_committed_tokens);
            end
            if(q_valid && q_ready)begin
                for(integer lane=0;lane<16;lane++)require(q_data[lane*32+23+:8]!=8'hff,"Dispatcher rejects nonfinite frame before forwarding");
                require(root_cycle>0 && q_frames<128 && q_index==q_frames && q_last==(q_frames==127),"Q after root framing");
                require(q_sequence==seq_model && q_epoch==epoch_model && q_data===source_frame(q_frames),"Exact Q stream");
                $display("DISPATCH_Q cycle=%0d job=%0d index=%0d data=%0128h",cycle,job,q_index,q_data);q_frames++;
            end
            if(hf)begin
                require(q_frames==128 && heads<32 && head_command_head==heads,"Ordered complete head queries");
                require(head_command_sequence==seq_model && head_command_epoch==epoch_model && head_command_layer==layer_model && head_command_position==position_model && head_command_lease==lease_model+64'(heads),"Head lease identity");
                for(integer i=0;i<64;i++)require(head_command_query[i*32+:32]===word(heads*64+i),"Head query bytes");
                $display("DISPATCH_HEAD cycle=%0d job=%0d head=%0d query=%0512h",cycle,job,heads,head_command_query);
            end
            if(hack)begin head_acks++;$display("DISPATCH_HEAD_ACK cycle=%0d job=%0d head=%0d",cycle,job,peer_head);end
            if(completion_valid && completion_ready)begin
                require(!public_ack && source_acks==0 && (completion_status!=0 || public_words==2048),"Public release after success readback or checked error");public_ack=1;
                $display("DISPATCH_ACK cycle=%0d job=%0d",cycle,job);
            end
            if(srcack)begin
                require(public_ack && !root_completion_valid && !seq_cache_lock && !aw_pending,"Source release after root release");
                source_acks++;$display("DISPATCH_SOURCE_ACK cycle=%0d job=%0d",cycle,job);
            end
            if(!completion_valid)require(!result_valid && !result_q_valid && result_q==0,"Private result invisible");
        end
        aw_hold=rst_n&&m_axi_awvalid&&!m_axi_awready;w_hold=rst_n&&m_axi_wvalid&&!m_axi_wready;
        saved_aw={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock};
        saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};
        #1;
        if(rst_n && admitted && !public_ack)require(cache_exclusive,"Cache ownership retained");
        if(bindhs)bind_valid=0;
        if(bf)begin aw_pending=0;w_burst=0;m_axi_bvalid=0;end
        if(hf)begin head_active=1;head_owned=1;peer_head=heads;peer_query=head_command_query;heads++;head_delay=mode==45 ? 16:3;end
        if(hcancel)begin head_completion_valid=1;head_completion_status=7;head_result_valid=0;end
        if(cancel_events>0)cancel_valid=0;
        if(hack)begin head_active=0;head_owned=0;head_completion_valid=0;head_result_valid=0;end
        if(srcack)begin source_completion_valid=0;source_result_valid=0;end
        require(job_cycles<15000,"Bounded integration timeout");
    endtask
    task automatic reset_all;
        @(negedge clk);
        $display("DISPATCH_RESET cycle=%0d job=%0d kv=%0d q=%0d heads=%0d aw=%0d w=%0d b=%0d",cycle,job,input_frames,q_frames,heads,aws,ws,bs);
        rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;root_revoke=0;
        source_completion_valid=0;source_result_valid=0;head_active=0;head_owned=0;
        head_completion_valid=0;head_result_valid=0;head_command_ready=0;
        aw_pending=0;w_burst=0;m_axi_bvalid=0;aw_hold=0;w_hold=0;admitted=0;
        repeat(3)@(negedge clk);rst_n=1;step();
        require(!busy && !completion_valid && !root_completion_valid && !append_done_valid && !cache_valid && !result_valid && !result_q_valid,"Coordinated reset releases all owners");
        total_resets++;reset_case=1;
    endtask
    task automatic run_case(input integer which);
        integer limit;
        job++;mode=which;admitted=0;public_ack=0;job_cycles=0;injected=0;reset_case=0;root_revoke=0;cancel_events=0;
        input_frames=0;q_frames=0;source_reads=0;heads=0;head_acks=0;public_words=0;source_acks=0;
        aws=0;ws=0;bs=0;planned=0;aw_beats=0;w_burst=0;append_cycle=0;commit_cycle=0;append_ack_cycle=0;root_cycle=0;first_aw_cycle=0;
        last_append_status=0;last_append_bytes=0;last_append_tokens=0;
        require(command_ready && !busy,"Idle before case");
        source_completion_valid=0;source_result_valid=0;cancel_valid=0;
        if(which!=3)begin
            seq_model=64'h1234567800000000+64'(job);epoch_model=64'hfedcba9800000000+64'(job);
            base_model=64'h900000000fc0;capacity_model=mode==1 ? 3:2;
            layer_model=mode==1 ? 15:0;position_model=0;
            expected_epoch=epoch_model;bind_phase=1;bind_response_ready=0;bind_valid=1;
            bind_cache_base=base_model;bind_region_base=base_model-4032;bind_region_bytes=4096+64'(capacity_model)*65536;
            bind_capacity=capacity_model;bind_sequence=seq_model;bind_epoch=epoch_model;
            limit=0;while(!append_done_valid)begin step();limit++;require(limit<40,"Bind progress");end
            require(append_done_bind && append_done_status==0,"Real cache bind");
            bind_response_ready=1;step();bind_response_ready=0;bind_phase=0;step();
        end else position_model=1;
        if(mode==12)layer_model=16;
        if(mode==13)position_model=capacity_model;
        if(mode==14)position_model=1;
        lease_model=mode==2 ? 64'hffffffffffffffe0:64'h2000000000001000+64'(job)*64;
        command_sequence=seq_model;command_epoch=epoch_model;command_layer=layer_model;command_position=position_model;command_lease_base=lease_model;
        source_completion_valid=1;source_status=0;source_sequence=seq_model;source_epoch=epoch_model;source_layer=layer_model;source_position=position_model;
        source_qk_words=2560;source_v_beats=32;source_frames=192;source_result_valid=1;
        command_valid=1;completion_ready=0;cancel_valid=0;result_ce=0;
        wanted_status=0;wanted_source_acks=1;
        case(mode)
            10:begin command_lease_base=0;wanted_status=1;end
            11:begin command_lease_base=64'hffffffffffffffe1;wanted_status=1;end
            12,13,14:wanted_status=1;
            15:begin expected_epoch=epoch_model^64'h0100000000000000;wanted_status=2;end
            16:begin source_sequence=seq_model^64'h0100000000000000;wanted_status=2;wanted_source_acks=0;end
            17:begin source_epoch=epoch_model^64'h0100000000000000;wanted_status=2;wanted_source_acks=0;end
            18:begin source_layer=layer_model+1;wanted_status=2;wanted_source_acks=0;end
            19:begin source_position=position_model+1;wanted_status=2;wanted_source_acks=0;end
            20:begin source_completion_valid=0;wanted_status=11;wanted_source_acks=0;end
            21:begin source_status=5;wanted_status=5;end
            22:begin source_result_valid=0;wanted_status=11;end
            23:begin source_qk_words=2559;wanted_status=11;end
            24:begin source_v_beats=31;wanted_status=11;end
            25:begin source_frames=191;wanted_status=11;end
            26:begin cancel_valid=1;wanted_status=7;end
            27:begin source_status=5;cancel_valid=1;wanted_status=5;end
            30,31,37:wanted_status=4;
            32,33:wanted_status=8;
            34,35:wanted_status=3;
            36:wanted_status=5;
            40,41,42,43,44,45:wanted_status=7;
            46:wanted_status=2;
            47,48:wanted_status=11;
            default:begin end
        endcase
        $display("DISPATCH_BEGIN job=%0d mode=%0d sequence=%016h epoch=%016h layer=%0d position=%0d base=%016h capacity=%0d expected=%0d",job,mode,seq_model,epoch_model,layer_model,position_model,base_model,capacity_model,wanted_status);
        while(!completion_valid && !reset_case)begin
            step();
            if(admitted && job_cycles==2)begin command_sequence=0;command_epoch=0;command_position=32'hffffffff;command_layer=32'hffffffff;command_lease_base=0;end
            if((mode==60 && input_frames>=17) || (mode==61 && aw_pending) || (mode==62 && q_frames>=17))reset_all();
        end
        if(mode==63 && !reset_case)reset_all();
        if(!reset_case)begin
            require(completion_status==wanted_status,"Exact completion status");
            require(completion_sequence==seq_model && completion_epoch==epoch_model && completion_layer==layer_model && completion_position==position_model,"Completion identity");
            require(completion_kv_frames==input_frames && completion_q_frames==q_frames,"Transferred frame extents");
            require(completion_append_status==last_append_status && completion_kv_bytes==last_append_bytes && completion_kv_tokens==last_append_tokens,"Observed append reply retained exactly");
            require(source_acks==0 && !aw_pending && w_burst==0 && !head_active,"No early source release or outstanding child work");
            if(mode>=10 && mode<=27)require(append_cycle==0 && root_cycle==0 && source_reads==0 && aws==0 && ws==0 && bs==0,"Admission error has no side effects");
            if(mode>=40 && mode<=48)require(injected,"Requested live fault was actually exercised");
            if(completion_kv_committed)require(input_frames==64 && last_append_status==0 && last_append_bytes==4096 && last_append_tokens==position_model+1,"Committed KV certificate");
            if(root_cycle>0 || mode==43)require(completion_kv_committed,"KV commit survives downstream failure");
            $display("DISPATCH_COMPLETE cycle=%0d job=%0d status=%0d append=%0d root=%0d kv=%0d q=%0d bytes=%0d tokens=%0d committed=%0d",cycle,job,completion_status,completion_append_status,completion_root_status,completion_kv_frames,completion_q_frames,completion_kv_bytes,completion_kv_tokens,completion_kv_committed);
            if(wanted_status==0)begin
                require(completion_append_status==0 && completion_root_status==0 && result_valid,"Integrated result success");
                require(completion_kv_committed && input_frames==64 && q_frames==128 && heads==32 && head_acks==32 && source_reads==192,"Whole pipeline success extents");
                for(integer i=0;i<2048;i++)begin
                    result_ce=1;result_address=11'((i*17+31)%2048);step();
                    require(result_q_valid && result_q===(word(int'(result_address))^32'h00000100),"Public head result bytes");
                    $display("DISPATCH_RESULT cycle=%0d job=%0d index=%0d data=%08h",cycle,job,result_address,result_q);public_words++;
                end
                result_ce=0;step();require(!result_q_valid && result_q==0,"Public read CE bubble");
                repeat(7)begin step();require(completion_valid && result_valid && !command_ready && source_acks==0 && root_completion_valid,"Held atomic response");end
                if(mode>=50 && mode<=54)begin
                    case(mode)
                        50:cancel_valid=1;
                        51:source_result_valid=0;
                        52:expected_epoch=epoch_model^64'h0100000000000000;
                        53:root_revoke=1;
                        54:source_sequence=seq_model^64'h0100000000000000;
                    endcase
                    result_ce=1;step();require(!result_valid && !result_q_valid && result_q==0 && completion_revoked && completion_status==0,"Immediate held-result revocation");
                    cancel_valid=0;root_revoke=0;source_result_valid=1;expected_epoch=epoch_model;
                    if(mode!=54)source_sequence=seq_model;
                    repeat(5)begin step();require(completion_valid && !result_valid && !result_q_valid && result_q==0 && completion_kv_committed,"Sticky revocation preserves KV commit");end
                    result_ce=0;total_revocations++;
                    $display("DISPATCH_REVOKED cycle=%0d job=%0d mode=%0d",cycle,job,mode);
                end
            end else begin
                total_errors++;result_ce=1;result_address=0;
                repeat(5)begin step();require(completion_valid && !result_valid && !result_q_valid && result_q==0 && source_acks==0,"Retained error hides payload and source ACK");end
                result_ce=0;expected_epoch=epoch_model;source_result_valid=1;
            end
            completion_ready=1;step();completion_ready=0;
            if(mode==48 || mode==54)begin
                repeat(5)begin step();require(busy && source_acks==0 && !source_completion_ready,"Foreign source response cannot be acknowledged");end
                source_sequence=seq_model;
            end
            limit=0;while(!command_ready)begin step();limit++;require(limit<30,"Final release progress");end
            require(source_acks==wanted_source_acks && !busy && !result_valid && !root_completion_valid,"Final ownership release");
            if(wanted_source_acks==0 && mode!=20)require(source_completion_valid,"Unowned source response retained");
            source_completion_valid=0;source_result_valid=0;
        end
        totals_jobs++;total_frames+=input_frames+q_frames;total_heads+=heads;total_words+=public_words;
        $display("DISPATCH_RELEASE cycle=%0d job=%0d aw=%0d w=%0d b=%0d source_acks=%0d reset=%0d",cycle,job,aws,ws,bs,source_acks,reset_case);
    endtask
    initial begin
        clk=0;
        rst_n=0;
        command_valid=0;
        command_sequence=0;
        command_epoch=0;
        command_lease_base=0;
        expected_epoch=0;
        command_layer=0;
        command_position=0;
        cancel_valid=0;
        source_completion_valid=0;
        source_status=0;
        source_sequence=0;
        source_epoch=0;
        source_layer=0;
        source_position=0;
        source_qk_words=0;
        source_v_beats=0;
        source_frames=0;
        source_result_valid=0;
        completion_ready=0;
        result_ce=0;
        result_address=0;
        bind_cache_base=0;
        bind_capacity=0;
        bind_epoch=0;
        bind_region_base=0;
        bind_region_bytes=0;
        bind_sequence=0;
        bind_valid=0;
        head_cancel_ready=0;
        head_command_ready=0;
        head_completion_status=0;
        head_completion_valid=0;
        head_result_valid=0;
        m_axi_awready=0;
        m_axi_bid=0;
        m_axi_bresp=0;
        m_axi_bvalid=0;
        m_axi_wready=0;
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        repeat(3)@(negedge clk);rst_n=1;step();
        run_case(0);run_case(3);run_case(1);run_case(2);
        for(integer mode_id=10;mode_id<=27;mode_id++)run_case(mode_id);
        for(integer mode_id=30;mode_id<=37;mode_id++)run_case(mode_id);
        for(integer mode_id=40;mode_id<=48;mode_id++)run_case(mode_id);
        for(integer mode_id=50;mode_id<=54;mode_id++)run_case(mode_id);
        for(integer mode_id=60;mode_id<=63;mode_id++)run_case(mode_id);
        require(totals_jobs==48 && total_resets==4 && total_errors==35 && total_revocations==5,"Scenario coverage");
        $display("DISPATCH_FAULT_PASS seed=%0d jobs=%0d frames=%0d heads=%0d words=%0d cycles=%0d errors=%0d revoked=%0d resets=%0d",seed,totals_jobs,total_frames,total_heads,total_words,cycle,total_errors,total_revocations,total_resets);
        $finish;
    end
endmodule
