// Real capture/dispatch/append/guard/sequencer; synthetic upstream RoPE RAM, AXI memory and arithmetic head.
`timescale 1ns/1ps
module tb_ntrpu_qkv_layer_reuse_r1;
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
    wire capture_completion_valid;
    wire  capture_completion_ready;
    wire [3:0] capture_status;
    wire [63:0] capture_sequence;
    wire [63:0] capture_epoch;
    wire [31:0] capture_layer;
    wire [31:0] capture_position;
    wire [11:0] capture_qk_words;
    wire [5:0] capture_v_beats;
    wire [7:0] capture_frames;
    wire capture_result_valid;
    wire  capture_ce;
    wire [7:0] capture_address;
    wire [511:0] capture_q;
    wire capture_q_valid;
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
    logic [63:0] command_v_address;
    logic [63:0] command_v_base;
    logic [63:0] command_v_bytes;
    logic command_v_source_valid;
    logic [63:0] command_v_source_sequence;
    logic [63:0] command_v_source_epoch;
    logic [31:0] command_v_source_position;
    logic source_completion_valid;
    wire source_completion_ready;
    logic [3:0] source_status;
    logic [63:0] source_sequence;
    logic [63:0] source_epoch;
    logic [31:0] source_position;
    logic [31:0] source_coefficient_position;
    logic [31:0] source_hls_status;
    logic [9:0] source_beats;
    logic source_result_valid;
    wire source_ce;
    wire [11:0] source_address;
    logic [31:0] source_q;
    logic source_q_valid;
    wire [63:0] m_axi_araddr;
    wire [7:0] m_axi_arlen;
    wire [2:0] m_axi_arsize;
    wire [2:0] m_axi_arprot;
    wire [1:0] m_axi_arburst;
    wire [3:0] m_axi_arid;
    wire [3:0] m_axi_arcache;
    wire m_axi_arlock;
    wire m_axi_arvalid;
    logic m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic m_axi_rlast;
    logic m_axi_rvalid;
    wire m_axi_rready;
    wire [3:0] completion_capture_status;
    wire [11:0] completion_capture_qk_words;
    wire [5:0] completion_capture_v_beats;
    ntrpu_qkv_capture_dispatch_r1 #(.WATCHDOG_CYCLES(64)) dut(.*);
    assign capture_completion_valid=dut.capture_done;
    assign capture_completion_ready=dut.capture_ack;
    assign capture_status=dut.capture_status;
    assign capture_sequence=dut.capture_sequence;
    assign capture_epoch=dut.capture_epoch;
    assign capture_layer=dut.capture_layer;
    assign capture_position=dut.capture_position;
    assign capture_qk_words=dut.capture_qk_words;
    assign capture_v_beats=dut.capture_v_beats;
    assign capture_frames=dut.capture_frames;
    assign capture_result_valid=dut.capture_result_valid;
    assign capture_ce=dut.capture_ce;
    assign capture_address=dut.capture_address;
    assign capture_q=dut.capture_q;
    assign capture_q_valid=dut.capture_q_valid;
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
        .cancel_valid(root_cancel_valid),
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

    integer prefix_model[0:15],last_writer[0:15][0:2],bind_count=0,history_checks=0;
    logic [511:0] physical_memory[0:3071];
    logic [63:0] pending_address;
    logic [287:0] observed_guard_prefixes,observed_engine_prefixes;
    always #2.5 clk=~clk;
    logic bind_phase=0,bind_response_ready=0;
    logic head_active=0,head_owned=0,public_ack=0,admitted=0;
    integer head_delay=0,peer_head=0;
    logic injected=0,was_reset=0;
    integer wanted_status=0,cancel_events=0,last_append_status=0,last_append_bytes=0,last_append_tokens=0;
    integer total_errors=0,total_resets=0,total_revocations=0;
    integer rope_reads=0,rope_acks=0,v_ar=0,v_r=0,v_beat=0,v_beats=0,capture_done_cycle=0;
    integer total_rope_reads=0,total_v_beats=0,total_rope_acks=0;
    logic v_active=0,v_ar_hold=0,v_r_hold=0;
    logic [63:0] v_address_model,v_burst_address;
    logic [71:0] v_saved_ar;
    logic [518:0] v_saved_r;
    logic [2047:0] peer_query;
    assign maintenance_allowed=!busy && !guard_maintenance_busy && !seq_cache_lock;
    assign engine_done_ready=bind_phase ? bind_response_ready:append_done_ready;
    assign engine_lease_active=head_active;
    assign cache_idle=!guard_maintenance_busy && !seq_cache_lock && !engine_lease_active && !append_done_valid;
    assign cache_metadata_fault=guard_metadata_fault;
    assign inspect_layer=root_command_layer[3:0];
    integer seed=1,random_state,job=0,mode=0,cycle=0,job_cycles=0;
    integer input_frames=0,q_frames=0,capture_reads=0,heads=0,head_acks=0,public_words=0,capture_acks=0;
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
    function automatic logic [31:0] historical_word(input integer index,input integer owner_job);
        logic [22:0] mantissa;
        mantissa=23'(index*131+owner_job*977);
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
    function automatic logic [511:0] capture_frame(input integer index);
        logic [511:0] data;
        for(integer lane=0;lane<16;lane++)data[lane*32+:32]=word(index*16+lane);
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
            source_q_valid<=source_ce && !(mode==23 && source_address==2559);
            if(source_ce)source_q<=(mode==22 && source_address==2048) ? 32'h7fc00001:word(int'(source_address));
            head_result_q_valid<=head_result_ce && head_result_valid;
            if(head_result_ce)head_result_q<=peer_query[head_result_address*32+:32]^32'h00000100;
        end
    end
    task automatic step;
        logic af,wf,bf,hf,hack,srcack,bindhs,arf,rf,rope_ack,hcancel;
        integer want_burst;
        @(negedge clk);
        if(admitted && !injected)begin
            if((mode==24 && v_active && v_r>=8) || (mode==25 && rope_reads>=2048) ||
                (mode==28 && capture_completion_valid && capture_done_cycle==0) ||
                (mode==30 && input_frames>=17) || (mode==31 && aw_pending) ||
                (mode==32 && q_frames>=17) || (mode==33 && append_ack_cycle>0 && root_cycle==0))begin
                cancel_valid=1;injected=1;
            end else if(mode==26 && rope_reads>=2048)begin expected_epoch=epoch_model^64'h0100000000000000;injected=1;end
            else if(mode==27 && rope_reads>=2048)begin source_sequence=seq_model^64'h0100000000000000;injected=1;end
            if(injected)$display("CAPTURE_INJECT cycle=%0d job=%0d mode=%0d qk=%0d v=%0d kv=%0d q=%0d",cycle,job,mode,rope_reads,v_r,input_frames,q_frames);
        end
        m_axi_arready=!v_active && $urandom_range(0,3)!=0;
        if(!m_axi_rvalid && v_active && $urandom_range(0,3)!=0)begin
            for(integer lane=0;lane<16;lane++)m_axi_rdata[lane*32+:32]=word(2560+int'((v_burst_address-v_address_model)/4)+v_beat*16+lane);
            if(mode==21 && v_r==31)m_axi_rdata[31:0]=32'h7fc00001;
            m_axi_rresp=mode==20 && v_r==31 ? 2:0;m_axi_rid=0;m_axi_rlast=v_beat==v_beats-1;m_axi_rvalid=1;
        end
        m_axi_awready=$urandom_range(0,3)!=0;m_axi_wready=$urandom_range(0,3)!=0;
        head_command_ready=!head_active && $urandom_range(0,3)!=0;
        head_cancel_ready=head_active && !head_completion_valid;
        if(!m_axi_bvalid && aw_pending && w_burst==aw_beats && $urandom_range(0,3)!=0)begin m_axi_bvalid=1;m_axi_bresp=mode==34 && planned==64 ? 2:0;m_axi_bid=0;end
        if(head_active && !head_completion_valid)begin
            if(head_delay>0)head_delay--;
            else begin head_completion_valid=1;head_result_valid=mode!=35;head_completion_status=mode==35 ? 5:0;end
        end
        @(posedge clk);cycle++;if(admitted)job_cycles++;
        af=rst_n&&m_axi_awvalid&&m_axi_awready;wf=rst_n&&m_axi_wvalid&&m_axi_wready;bf=rst_n&&m_axi_bvalid&&m_axi_bready;
        hf=rst_n&&head_command_valid&&head_command_ready;hack=rst_n&&head_completion_valid&&head_completion_ready;
        srcack=rst_n&&capture_completion_valid&&capture_completion_ready;bindhs=rst_n&&bind_valid&&bind_ready;
        arf=rst_n&&m_axi_arvalid&&m_axi_arready;rf=rst_n&&m_axi_rvalid&&m_axi_rready;
        rope_ack=rst_n&&source_completion_valid&&source_completion_ready;
        hcancel=rst_n&&head_cancel_valid&&head_cancel_ready;
        if(rst_n && cancel_valid && cancel_ready)cancel_events++;
        if(rst_n)begin
            if(v_ar_hold)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===v_saved_ar,"Stable V read address");
            if(v_r_hold)require(m_axi_rvalid && {m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast}===v_saved_r,"Stable V memory response");
            if(arf)begin
                require(!v_active && m_axi_araddr==v_address_model+64'(v_r*64),"Ordered V allocation read");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"V AXI attributes");
                require(int'(m_axi_arlen)+1==((32-v_r)<(64-int'(m_axi_araddr[11:6])) ? 32-v_r:64-int'(m_axi_araddr[11:6])),"V page extent");
                $display("CAPTURE_AR cycle=%0d job=%0d address=%016h beats=%0d",cycle,job,m_axi_araddr,int'(m_axi_arlen)+1);
            end
            if(rf)begin
                require(v_active,"Owned V response");
                $display("CAPTURE_R cycle=%0d job=%0d address=%016h response=%0d rid=%0d last=%0d data=%0128h",cycle,job,v_burst_address+64'(v_beat*64),m_axi_rresp,m_axi_rid,m_axi_rlast,m_axi_rdata);
            end
            if(source_ce)begin
                require(source_completion_valid && source_result_valid && source_address==rope_reads && source_address<2560 && !public_ack,"Owned ordered RoPE RAM read");
                require(v_r==32 && !v_active && !m_axi_rvalid,"Capture Q/K after V response drain");
                $display("CAPTURE_QK cycle=%0d job=%0d index=%0d data=%08h",cycle,job,source_address,word(int'(source_address)));rope_reads++;
            end
            if(source_q_valid)$display("CAPTURE_QK_RETURN cycle=%0d job=%0d index=%0d data=%08h",cycle,job,source_address,source_q);
            if(capture_completion_valid && capture_done_cycle==0)begin
                capture_done_cycle=cycle;
                require(capture_qk_words<=2560 && capture_v_beats<=32,"Bounded capture extents");
                if(capture_status==0 && !dut.capture_revoked)require(capture_qk_words==2560 && capture_v_beats==32 && capture_frames==192 && capture_result_valid,"Complete real capture");
                $display("CAPTURE_COMPLETE cycle=%0d job=%0d status=%0d qk=%0d v=%0d frames=%0d",cycle,job,capture_status,capture_qk_words,capture_v_beats,capture_frames);
            end
            if(rope_ack)begin
                require(public_ack && capture_acks==1 && !root_completion_valid && !seq_cache_lock && !aw_pending && !v_active,"RoPE release follows both downstream owners");
                require(source_sequence==seq_model && source_epoch==epoch_model && source_position==position_model && cache_exclusive && !command_ready,"Full source identity and exclusion through final drain");
                rope_acks++;$display("CAPTURE_SOURCE_ACK cycle=%0d job=%0d sequence=%016h epoch=%016h position=%0d",cycle,job,source_sequence,source_epoch,source_position);
            end
        end
        v_ar_hold=rst_n&&m_axi_arvalid&&!m_axi_arready;v_r_hold=rst_n&&m_axi_rvalid&&!m_axi_rready;
        v_saved_ar={m_axi_araddr,m_axi_arlen};v_saved_r={m_axi_rdata,m_axi_rresp,m_axi_rid,m_axi_rlast};
        if(rst_n)begin
            if(aw_hold)require(m_axi_awvalid && saved_aw=={m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awid,m_axi_awcache,m_axi_awprot,m_axi_awlock},"Stable offered AW");
            if(w_hold)require(m_axi_wvalid && saved_w=={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"Stable offered W");
            if(command_valid && command_ready)begin
                require(!admitted,"Single admission");admitted=1;command_valid<=0;
                $display("DISPATCH_ADMIT cycle=%0d job=%0d sequence=%016h epoch=%016h layer=%0d position=%0d lease=%016h",cycle,job,command_sequence,command_epoch,command_layer,command_position,command_lease_base);
            end
            if(append_valid && append_ready)begin
                require(admitted && capture_done_cycle>0 && cycle>capture_done_cycle && append_cycle==0 && input_frames==0 && root_cycle==0,"Append admission follows real capture");
                require(append_sequence==seq_model && append_epoch==epoch_model && append_layer==layer_model && append_position==position_model,"Append descriptor");
                append_cycle=cycle;$display("DISPATCH_APPEND cycle=%0d job=%0d",cycle,job);
            end
            if(capture_ce)begin
                require(capture_completion_valid && capture_result_valid && capture_address<192 && !public_ack,"Owned snapshot read");
                require(capture_address==(root_cycle>0 ? q_frames:128+input_frames),"K/V then Q RAM addressing");
                $display("DISPATCH_RAM cycle=%0d job=%0d address=%0d data=%0128h",cycle,job,capture_address,capture_frame(int'(capture_address)));capture_reads++;
            end
            if(data_valid && data_ready)begin
                require(input_frames<64 && data_index==input_frames && data_last==(input_frames==63),"KV frame framing");
                require(data_sequence==seq_model && data_epoch==epoch_model && data_bits===capture_frame(128+input_frames),"KV exact bytes");
                $display("DISPATCH_KV cycle=%0d job=%0d index=%0d data=%0128h",cycle,job,data_index,data_bits);input_frames++;
            end
            if(commit_valid && commit_ready)begin
                require((commit_status!=0 || input_frames==64) && commit_sequence==seq_model && commit_epoch==epoch_model,"Complete KV commit or producer fault");
                require(commit_cycle==0,"Single commit");commit_cycle=cycle;$display("DISPATCH_COMMIT cycle=%0d job=%0d status=%0d frames=%0d",cycle,job,commit_status,input_frames);
            end
            if(af)begin
                require(commit_cycle>0 && !aw_pending,"No write before complete producer commit");
                want_burst=4-planned%4;if(want_burst>64-int'((location(planned)>>6)&63))want_burst=64-int'((location(planned)>>6)&63);
                require(m_axi_awaddr==location(planned) && int'(m_axi_awlen)+1==want_burst,"Physical KV layout and page bounds");
                require(m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 && m_axi_awcache==2 && !m_axi_awlock && m_axi_awprot==0,"AW attributes");
                if(first_aw_cycle==0)first_aw_cycle=cycle;
                $display("DISPATCH_AW cycle=%0d job=%0d address=%016h beats=%0d",cycle,job,m_axi_awaddr,want_burst);
                pending_address=m_axi_awaddr;aw_pending=1;aw_beats=want_burst;planned+=want_burst;aws++;
            end
            if(wf)begin
                require(commit_cycle>0 && ws<64 && w_burst<4,"W after commit");
                require(m_axi_wdata===capture_frame(128+ws) && m_axi_wstrb==64'hffffffffffffffff,"Exact staged KV write");
                require(m_axi_wlast==((ws%4)==3 || ((location(ws)>>6)&63)==63),"WLAST framing");
                pending_words[w_burst]=m_axi_wdata;w_burst++;ws++;
                $display("DISPATCH_W cycle=%0d job=%0d index=%0d data=%0128h",cycle,job,ws-1,m_axi_wdata);
            end
            if(bf)begin
                require(aw_pending && w_burst==aw_beats,"B owns complete AW/W burst");
                if(m_axi_bresp==0)for(integer i=0;i<aw_beats;i++)begin
                    committed_memory[planned-aw_beats+i]=pending_words[i];
                    require(pending_address>=base_model && pending_address+64'(i*64)<base_model+196608,"Persistent allocation bounds");
                    physical_memory[int'((pending_address-base_model)/64)+i]=pending_words[i];
                end
                $display("DISPATCH_B cycle=%0d job=%0d beats=%0d response=%0d",cycle,job,aw_beats,m_axi_bresp);bs++;
            end
            if(append_done_valid && append_done_ready)begin
                require(!append_done_bind && append_done_sequence==seq_model && append_done_epoch==epoch_model && bs==aws && !aw_pending && w_burst==0,"KV reply identity and drain");
                require(append_done_bytes==ws*64,"Append bytes count accepted W");
                if(append_done_status==0)require(input_frames==64 && ws==64 && append_done_bytes==4096 && append_done_committed_tokens==position_model+1,"Full successful KV commit");
                else require(append_done_committed_tokens==position_model,"Failed KV leaves prefix unpublished");
                last_append_status=append_done_status;last_append_bytes=append_done_bytes;last_append_tokens=append_done_committed_tokens;
                append_ack_cycle=cycle;$display("DISPATCH_KV_ACK cycle=%0d job=%0d status=%0d bytes=%0d tokens=%0d",cycle,job,append_done_status,append_done_bytes,append_done_committed_tokens);
            end
            if(root_command_valid && root_command_ready)begin
                require(append_ack_cycle>0 && cycle>append_ack_cycle && cache_committed_tokens==position_model+1 && root_cycle==0,"Root only after acknowledged trusted prefix");
                require(root_command_sequence==seq_model && root_command_epoch==epoch_model && root_command_layer==layer_model && root_command_position==position_model && root_command_lease_base==lease_model,"Root descriptor");
                for(integer i=0;i<64;i++)require(committed_memory[i]===capture_frame(128+i),"All KV bytes B-committed");
                root_cycle=cycle;$display("DISPATCH_ROOT cycle=%0d job=%0d prefix=%0d",cycle,job,cache_committed_tokens);
            end
            if(q_valid && q_ready)begin
                require(root_cycle>0 && q_frames<128 && q_index==q_frames && q_last==(q_frames==127),"Q after root framing");
                require(q_sequence==seq_model && q_epoch==epoch_model && q_data===capture_frame(q_frames),"Exact Q stream");
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
                require(!public_ack && capture_acks==0 && (completion_status!=0 || public_words==2048),"Public release after checked response");public_ack=1;
                $display("DISPATCH_ACK cycle=%0d job=%0d",cycle,job);
            end
            if(srcack)begin
                require(public_ack && !root_completion_valid && !seq_cache_lock && !aw_pending,"Source release after root release");
                capture_acks++;$display("DISPATCH_CAPTURE_ACK cycle=%0d job=%0d",cycle,job);
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
        if(hf)begin head_active=1;head_owned=1;peer_head=heads;peer_query=head_command_query;heads++;head_delay=3;end
        if(cancel_events>0)cancel_valid=0;
        if(hcancel)begin head_completion_valid=1;head_completion_status=7;head_result_valid=0;end
        if(hack)begin head_active=0;head_owned=0;head_completion_valid=0;head_result_valid=0;end
        if(arf)begin v_active=1;v_burst_address=v_saved_ar[71:8];v_beats=int'(v_saved_ar[7:0])+1;v_beat=0;v_ar++;end
        if(rf)begin v_beat++;v_r++;m_axi_rvalid=0;if(m_axi_rlast)v_active=0;end
        if(rope_ack)begin source_completion_valid=0;source_result_valid=0;end
        require(job_cycles<20000,"Bounded integration timeout");
    endtask
    task automatic reset_all;
        @(negedge clk);
        $display("CAPTURE_RESET cycle=%0d job=%0d qk=%0d v=%0d kv=%0d q=%0d heads=%0d",cycle,job,rope_reads,v_r,input_frames,q_frames,heads);
        rst_n=0;command_valid=0;completion_ready=0;cancel_valid=0;source_completion_valid=0;source_result_valid=0;
        v_active=0;m_axi_rvalid=0;v_ar_hold=0;v_r_hold=0;aw_pending=0;w_burst=0;m_axi_bvalid=0;aw_hold=0;w_hold=0;
        head_active=0;head_owned=0;head_completion_valid=0;head_result_valid=0;admitted=0;
        repeat(3)@(negedge clk);rst_n=1;step();
        require(!busy && !completion_valid && !root_completion_valid && !append_done_valid && !cache_valid && !dut.capture_busy && !dut.dispatch_busy,"Coordinated reset releases full path");
        was_reset=1;total_resets++;
    endtask
    task automatic run_case(input integer which,which_layer,which_position,do_bind,group);
        integer limit;
        job++;mode=which;admitted=0;public_ack=0;job_cycles=0;injected=0;was_reset=0;cancel_events=0;
        last_append_status=0;last_append_bytes=0;last_append_tokens=0;
        input_frames=0;q_frames=0;capture_reads=0;heads=0;head_acks=0;public_words=0;capture_acks=0;
        rope_reads=0;rope_acks=0;v_ar=0;v_r=0;capture_done_cycle=0;
        aws=0;ws=0;bs=0;planned=0;aw_beats=0;w_burst=0;append_cycle=0;commit_cycle=0;append_ack_cycle=0;root_cycle=0;first_aw_cycle=0;
        require(command_ready && !busy,"Idle before case");
        if(do_bind!=0)begin
            seq_model=64'h1234567800000001^(group==2 ? 64'h0000010000000000:64'd0);epoch_model=64'hfedcba9800000001^(group==2 ? 64'h0000010000000000:64'd0);
            base_model=64'h900000000fc0;capacity_model=3;
            layer_model=32'(which_layer);position_model=32'(which_position);
            expected_epoch=epoch_model;bind_phase=1;bind_response_ready=0;bind_valid=1;
            bind_cache_base=base_model;bind_region_base=base_model-4032;bind_region_bytes=4096+64'(capacity_model)*65536;
            bind_capacity=capacity_model;bind_sequence=seq_model;bind_epoch=epoch_model;
            limit=0;while(!append_done_valid)begin step();limit++;require(limit<40,"Bind progress");end
            require(append_done_bind && append_done_status==0,"Real cache bind");
            bind_response_ready=1;step();bind_response_ready=0;bind_phase=0;step();
            for(integer layer=0;layer<16;layer++)begin
                prefix_model[layer]=0;
                for(integer token=0;token<3;token++)last_writer[layer][token]=0;
            end
            bind_count++;
            $display("LAYERS_BIND cycle=%0d job=%0d group=%0d sequence=%016h epoch=%016h capacity=%0d",cycle,job,group,seq_model,epoch_model,capacity_model);
        end
        layer_model=32'(which_layer);position_model=32'(which_position);
        lease_model=mode==2 ? 64'hffffffffffffffe0:64'h2000000000001000+64'(job)*64;
        command_sequence=seq_model;command_epoch=epoch_model;command_layer=layer_model;command_position=position_model;command_lease_base=lease_model;
        source_completion_valid=1;source_status=0;source_sequence=seq_model;source_epoch=epoch_model;source_position=position_model;
        source_coefficient_position=position_model;source_hls_status=0;source_beats=164;source_result_valid=1;
        command_v_base=64'hc000000000;command_v_address=command_v_base+64'(mode==1 ? 0:mode==2 ? 128:4032);command_v_bytes=16384;
        command_v_source_valid=1;command_v_source_sequence=seq_model;command_v_source_epoch=epoch_model;command_v_source_position=position_model;
        v_address_model=command_v_address;
        command_valid=1;completion_ready=0;cancel_valid=0;result_ce=0;wanted_status=0;
        case(mode)
            10:begin command_v_bytes=64'(command_v_address-command_v_base)+2047;wanted_status=1;end
            11:begin command_v_source_epoch=epoch_model^64'h0100000000000000;wanted_status=2;end
            12:begin source_status=5;wanted_status=5;end
            13:begin source_beats=163;wanted_status=11;end
            20,34:wanted_status=3;
            21,22:wanted_status=4;
            23:wanted_status=8;
            24,25,28,30,31,32,33:wanted_status=7;
            26:wanted_status=2;
            27:wanted_status=11;
            35:wanted_status=5;
            90,91,92:wanted_status=1;
            default:begin end
        endcase
        $display("CAPTURE_BEGIN job=%0d sequence=%016h epoch=%016h position=%0d va=%016h",job,seq_model,epoch_model,position_model,v_address_model);
        $display("DISPATCH_BEGIN job=%0d mode=%0d sequence=%016h epoch=%016h layer=%0d position=%0d base=%016h capacity=%0d",job,mode,seq_model,epoch_model,layer_model,position_model,base_model,capacity_model);
        while(!completion_valid && !was_reset)begin
            step();
            if((mode==50 && v_active && v_r>=8) || (mode==51 && rope_reads>=2048) || (mode==52 && q_frames>=17))reset_all();
            if(admitted && job_cycles==2)begin command_sequence=0;command_epoch=0;command_position=32'hffffffff;command_layer=32'hffffffff;command_lease_base=0;end
        end
        if(mode==53 && !was_reset)reset_all();
        if(!was_reset)begin
            require(completion_status==wanted_status,"Exact composed status");
            require(completion_sequence==seq_model && completion_epoch==epoch_model && completion_layer==layer_model && completion_position==position_model,"Composed identity");
            require(completion_kv_frames==input_frames && completion_q_frames==q_frames && completion_kv_bytes==last_append_bytes && completion_kv_tokens==last_append_tokens && completion_append_status==last_append_status,"Accepted dispatch extents");
            require(capture_acks==0 && rope_acks==0 && !v_active && !aw_pending && w_burst==0 && !head_active,"No premature release or outstanding memory");
            if((mode>=10 && mode<=28) || mode>=90)require(append_cycle==0 && root_cycle==0 && input_frames==0 && q_frames==0,"Capture failure prevents downstream effects");
            if((mode>=24 && mode<=28) || (mode>=30 && mode<=33))require(injected,"Requested fault exercised");
            if(root_cycle>0 || mode==33)require(completion_kv_committed,"KV commit survives later cancellation/failure");
            if(completion_kv_committed)require(input_frames==64 && last_append_status==0 && last_append_bytes==4096 && last_append_tokens==position_model+1,"Committed prefix certificate");
            $display("CAPDISP_COMPLETE cycle=%0d job=%0d status=%0d capture=%0d append=%0d root=%0d qk=%0d v=%0d kv=%0d q=%0d bytes=%0d tokens=%0d committed=%0d",cycle,job,completion_status,completion_capture_status,completion_append_status,completion_root_status,completion_capture_qk_words,completion_capture_v_beats,completion_kv_frames,completion_q_frames,completion_kv_bytes,completion_kv_tokens,completion_kv_committed);
            if(wanted_status==0)begin
                require(completion_capture_status==0 && completion_append_status==0 && completion_root_status==0 && result_valid && completion_kv_committed,"Whole path success");
                require(input_frames==64 && q_frames==128 && heads==32 && head_acks==32 && capture_reads==192 && rope_reads==2560 && v_r==32,"Whole path success counts");
                for(integer i=0;i<2048;i++)begin
                    result_ce=1;result_address=11'((i*17+31)%2048);step();
                    require(result_q_valid && result_q===(word(int'(result_address))^32'h00000100),"Public head result bytes");
                    $display("DISPATCH_RESULT cycle=%0d job=%0d index=%0d data=%08h",cycle,job,result_address,result_q);public_words++;
                end
                result_ce=0;step();require(!result_q_valid && result_q==0,"Public CE bubble");
                repeat(7)begin step();require(completion_valid && result_valid && capture_result_valid && !command_ready && capture_acks==0 && rope_acks==0,"Whole response/source held");end
                if(mode>=40 && mode<=43)begin
                    case(mode)
                        40:cancel_valid=1;
                        41:source_result_valid=0;
                        42:source_sequence=seq_model^64'h0100000000000000;
                        43:expected_epoch=epoch_model^64'h0100000000000000;
                    endcase
                    result_ce=1;step();require(!result_valid && !result_q_valid && result_q==0 && completion_revoked && completion_status==0,"Immediate composed revocation");
                    cancel_valid=0;source_result_valid=1;expected_epoch=epoch_model;
                    repeat(5)begin step();require(completion_valid && !result_valid && !result_q_valid && result_q==0 && completion_kv_committed && capture_acks==0 && rope_acks==0,"Sticky composed revocation");end
                    result_ce=0;total_revocations++;
                    $display("CAPTURE_REVOKED cycle=%0d job=%0d mode=%0d",cycle,job,mode);
                end
            end else begin
                total_errors++;result_ce=1;result_address=0;
                repeat(5)begin step();require(completion_valid && !result_valid && !result_q_valid && result_q==0 && capture_acks==0 && rope_acks==0,"Held composed error");end
                result_ce=0;expected_epoch=epoch_model;source_result_valid=1;
            end
            completion_ready=1;step();completion_ready=0;
            if(mode==27 || mode==42)begin
                repeat(7)begin step();require(busy && cache_exclusive && !command_ready && !source_completion_ready && rope_acks==0,"Foreign upstream owner retains global cache exclusion");end
                source_sequence=seq_model;
                $display("CAPTURE_SOURCE_RESTORED cycle=%0d job=%0d",cycle,job);
            end
            limit=0;while(!command_ready)begin step();limit++;require(limit<30,"Whole path final drain");end
            require(capture_acks==1 && rope_acks==1 && !busy && !dut.capture_busy && !dut.dispatch_busy && !result_valid && !root_completion_valid,"Whole path final ownership release");
        end
        if(completion_kv_committed)begin
            require(layer_model<16 && position_model==32'(prefix_model[layer_model]),"Only own next prefix may publish");
            prefix_model[layer_model]++;last_writer[layer_model][position_model]=job;
        end
        observed_guard_prefixes=0;observed_engine_prefixes=0;
        for(integer layer=0;layer<16;layer++)begin
            require(guard.prefixes[layer]==18'(prefix_model[layer]) && kv.prefixes[layer]==18'(prefix_model[layer]),"All layer prefixes agree with acknowledged commits");
            observed_guard_prefixes[layer*18+:18]=guard.prefixes[layer];
            observed_engine_prefixes[layer*18+:18]=kv.prefixes[layer];
            for(integer token=0;token<prefix_model[layer];token++)begin
                require(last_writer[layer][token]>0,"Published token has a successful writer");
                for(integer beat_index=0;beat_index<64;beat_index++)begin
                    for(integer lane=0;lane<16;lane++)require(
                        physical_memory[((layer*16+beat_index/4)*3+token)*4+beat_index%4][lane*32+:32]===historical_word(2048+beat_index*16+lane,last_writer[layer][token]),
                        "Every published layer/token retains its physical KV bytes");
                    history_checks++;
                end
            end
        end
        $display("LAYERS_STATE cycle=%0d job=%0d guard=%072h engine=%072h physical_checks=%0d binds=%0d",cycle,job,observed_guard_prefixes,observed_engine_prefixes,history_checks,bind_count);
        totals_jobs++;total_frames+=input_frames+q_frames;total_heads+=heads;total_words+=public_words;
        total_rope_reads+=rope_reads;total_v_beats+=v_r;total_rope_acks+=rope_acks;
        $display("CAPDISP_RELEASE cycle=%0d job=%0d aw=%0d w=%0d b=%0d capture_acks=%0d rope_acks=%0d reset=%0d",cycle,job,aws,ws,bs,capture_acks,rope_acks,was_reset);
    endtask
    initial begin
        command_v_address=0;
        command_v_base=0;
        command_v_bytes=0;
        command_v_source_valid=0;
        command_v_source_sequence=0;
        command_v_source_epoch=0;
        command_v_source_position=0;
        source_completion_valid=0;
        source_status=0;
        source_sequence=0;
        source_epoch=0;
        source_position=0;
        source_coefficient_position=0;
        source_hls_status=0;
        source_beats=0;
        source_result_valid=0;
        source_q=0;
        source_q_valid=0;
        m_axi_arready=0;
        m_axi_rdata=0;
        m_axi_rresp=0;
        m_axi_rid=0;
        m_axi_rlast=0;
        m_axi_rvalid=0;
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
        run_case(0,0,0,1,1);
        run_case(0,1,0,0,1);
        run_case(0,2,0,0,1);
        run_case(0,3,0,0,1);
        run_case(0,4,0,0,1);
        run_case(0,5,0,0,1);
        run_case(0,6,0,0,1);
        run_case(0,7,0,0,1);
        run_case(0,8,0,0,1);
        run_case(0,9,0,0,1);
        run_case(0,10,0,0,1);
        run_case(0,11,0,0,1);
        run_case(0,12,0,0,1);
        run_case(0,13,0,0,1);
        run_case(0,14,0,0,1);
        run_case(0,15,0,0,1);
        run_case(90,0,0,0,1);
        run_case(90,8,2,0,1);
        run_case(34,7,1,0,1);
        run_case(0,12,1,0,1);
        run_case(0,7,1,0,1);
        run_case(0,1,1,0,1);
        run_case(0,6,1,0,1);
        run_case(0,11,1,0,1);
        run_case(0,0,1,0,1);
        run_case(0,5,1,0,1);
        run_case(0,10,1,0,1);
        run_case(0,15,1,0,1);
        run_case(0,4,1,0,1);
        run_case(0,9,1,0,1);
        run_case(0,14,1,0,1);
        run_case(0,3,1,0,1);
        run_case(0,8,1,0,1);
        run_case(0,13,1,0,1);
        run_case(0,2,1,0,1);
        run_case(0,15,2,0,1);
        run_case(0,14,2,0,1);
        run_case(0,13,2,0,1);
        run_case(0,12,2,0,1);
        run_case(0,11,2,0,1);
        run_case(0,10,2,0,1);
        run_case(0,9,2,0,1);
        run_case(0,8,2,0,1);
        run_case(0,7,2,0,1);
        run_case(0,6,2,0,1);
        run_case(40,5,2,0,1);
        run_case(0,4,2,0,1);
        run_case(0,3,2,0,1);
        run_case(0,2,2,0,1);
        run_case(0,1,2,0,1);
        run_case(0,0,2,0,1);
        run_case(90,15,3,0,1);
        run_case(91,16,0,0,1);
        run_case(92,268435456,0,0,1);
        run_case(0,15,0,1,2);
        run_case(0,0,0,0,2);
        run_case(11,1,0,0,2);
        run_case(0,1,0,0,2);
        require(totals_jobs==58 && total_errors==7 && total_resets==0 && total_revocations==1 && bind_count==2,"Shared cache coverage");
        require(total_frames==9856 && total_heads==1632 && total_words==104448,"Complete layer/token transport coverage");
        $display("LAYERS_PASS seed=%0d jobs=%0d frames=%0d heads=%0d words=%0d cycles=%0d qk=%0d v=%0d acks=%0d errors=%0d resets=%0d revoked=%0d",seed,totals_jobs,total_frames,total_heads,total_words,cycle,total_rope_reads,total_v_beats,total_rope_acks,total_errors,total_resets,total_revocations);
        $finish;
    end
endmodule
