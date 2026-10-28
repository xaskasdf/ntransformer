`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_memory_r1 #(parameter integer WATCHDOG_CYCLES=32);
    logic clk=0;
    logic rst_n=0;
    logic bind_valid=0;
    logic bind_ready;
    logic [63:0] bind_cache_base=0;
    logic [63:0] bind_region_base=0;
    logic [63:0] bind_region_bytes=0;
    logic [31:0] bind_capacity=0;
    logic [63:0] bind_sequence=0;
    logic [63:0] bind_epoch=0;
    logic [63:0] expected_epoch=0;
    logic cache_valid;
    logic [63:0] cache_base;
    logic [63:0] cache_region_base;
    logic [63:0] cache_region_bytes;
    logic [63:0] cache_sequence;
    logic [63:0] cache_epoch;
    logic [31:0] cache_capacity;
    logic append_valid=0;
    logic append_ready;
    logic append_prepared;
    logic [31:0] append_layer=0;
    logic [31:0] append_position=0;
    logic [63:0] append_sequence=0;
    logic [63:0] append_epoch=0;
    logic data_valid=0;
    logic data_ready;
    logic [511:0] data_bits=0;
    logic [5:0] data_index=0;
    logic data_last=0;
    logic [63:0] data_sequence=0;
    logic [63:0] data_epoch=0;
    logic commit_valid=0;
    logic commit_ready;
    logic [3:0] commit_status=0;
    logic [63:0] commit_sequence=0;
    logic [63:0] commit_epoch=0;
    logic append_cancel_valid=0;
    logic append_cancel_ready;
    logic append_done_valid;
    logic append_done_ready=0;
    logic append_done_bind;
    logic [3:0] append_done_status;
    logic [12:0] append_done_bytes;
    logic [31:0] append_done_committed_tokens;
    logic [63:0] append_done_sequence;
    logic [63:0] append_done_epoch;
    logic lease_active;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize;
    logic [2:0] m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid;
    logic [3:0] m_axi_awcache;
    logic m_axi_awlock;
    logic m_axi_awvalid;
    logic m_axi_awready=0;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast;
    logic m_axi_wvalid;
    logic m_axi_wready=0;
    logic [3:0] m_axi_bid=0;
    logic [1:0] m_axi_bresp=0;
    logic m_axi_bvalid=0;
    logic m_axi_bready;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize;
    logic [2:0] m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid;
    logic [3:0] m_axi_arcache;
    logic m_axi_arlock;
    logic m_axi_arvalid;
    logic m_axi_arready=0;
    logic [511:0] m_axi_rdata=0;
    logic [1:0] m_axi_rresp=0;
    logic [3:0] m_axi_rid=0;
    logic m_axi_rlast=0;
    logic m_axi_rvalid=0;
    logic m_axi_rready;
    logic command_valid=0;
    logic command_ready;
    logic [2047:0] command_query=0;
    logic [31:0] command_position=0;
    logic [31:0] command_layer=0;
    logic [31:0] command_head=0;
    logic [63:0] command_sequence=0;
    logic [63:0] command_epoch=0;
    logic [63:0] command_lease=0;
    logic cancel_valid=0;
    logic cancel_ready;
    logic completion_valid;
    logic completion_ready=0;
    logic [3:0] completion_status;
    logic result_valid;
    logic result_ce=0;
    logic [5:0] result_address=0;
    logic [31:0] result_q;
    logic result_q_valid;
    logic hls_start;
    logic hls_ready=0;
    logic hls_done=0;
    logic [31:0] hls_return=0;
    logic [31:0] hls_tile_start;
    logic [31:0] hls_tile_count;
    logic [31:0] hls_query_position;
    logic [31:0] hls_layer;
    logic [31:0] hls_query_head;
    logic [31:0] hls_kv_head;
    logic [63:0] hls_sequence;
    logic [63:0] hls_epoch;
    logic hls_query_ce=0;
    logic [5:0] hls_query_address=0;
    logic [31:0] hls_query_q;
    logic hls_state_in_ce=0;
    logic [6:0] hls_state_in_address=0;
    logic [63:0] hls_state_in_q;
    logic hls_state_out_ce=0;
    logic hls_state_out_we=0;
    logic [6:0] hls_state_out_address=0;
    logic [63:0] hls_state_out_d=0;
    logic hls_output_ce=0;
    logic hls_output_we=0;
    logic [5:0] hls_output_address=0;
    logic [31:0] hls_output_d=0;
    logic hls_keys_ce=0;
    logic hls_values_ce=0;
    logic [11:0] hls_keys_address=0;
    logic [11:0] hls_values_address=0;
    logic [31:0] hls_keys_q;
    logic [31:0] hls_values_q;
    logic hls_keys_q_valid;
    logic hls_values_q_valid;
    always #2.5 clk=~clk;
    ntrpu_attention_memory_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
    wire  tile_valid=dut.kv_path.tile_valid;
    wire  tile_ready=dut.kv_path.tile_ready;
    wire [511:0] tile_data=dut.kv_path.tile_data;
    wire  tile_plane=dut.kv_path.tile_plane;
    wire  tile_last=dut.kv_path.tile_last;
    wire [7:0] tile_index=dut.kv_path.tile_index;
    wire [63:0] tile_sequence=dut.kv_path.tile_sequence;
    wire [63:0] tile_epoch=dut.kv_path.tile_epoch;
    wire [63:0] tile_lease_id=dut.kv_path.tile_lease_id;
    localparam logic [63:0] BASE=64'ha000000000;
    logic [511:0] memory[0:262143],pending_w[0:3];
    bit initialized[0:262143];
    int seed=1,random_state,op=0,writer_op=0,reader_op=0,cycles=0;
    int bind_count=0,append_count=0,lease_count=0,read_count=0,reset_count=0;
    int prefix[0:15],generation=0,cap=0,write_error_mode=0,write_b_count=0,read_error_at=-1,read_r_count=0;
    int aw_count=0,w_count=0,b_count=0,ar_count=0,r_count=0,tile_count=0;
    int aw_waits=0,w_waits=0,ar_waits=0,tile_waits=0,simultaneous_rw=0;
    bit aw_pending=0,w_last_seen=0,write_applied=0,ar_pending=0;
    int aw_beats=0,w_pending=0,ar_beats=0,ar_index=0;
    logic [63:0] aw_address,ar_address,base_model=0,seq_model=0,epoch_model=0;
    logic [63:0] next_lease=64'h1122334400000000,lease_id_model=0;
    int lease_layer_model=0,lease_head_model=0,lease_position_model=0,lease_prefix_model=0,lease_generation=0,next_start_model=0;
    bit freeze_aw=0,freeze_w=0,freeze_ar=0,freeze_r=0,freeze_b=0;
    bit aw_held=0,w_held=0,ar_held=0;
    logic [71:0] saved_aw,saved_ar;
    logic [576:0] saved_w;
    bit tile_held=0;
    logic [713:0] saved_tile;
    function automatic logic [511:0] pattern(input int image,layer,pos,index);
        for(int lane=0;lane<16;lane++)
            pattern[lane*32+:32]=32'h3f000000 | (32'(image*100003+layer*4099+pos*97+index*23+lane*3)&32'h007fffff);
    endfunction
    function automatic int slot(input logic [63:0] address);
        if(!(address>=BASE && address<BASE+64'd16777216 && address[5:0]==0))
            $fatal(1,"ATM_FAIL op=%0d: Physical memory range",op);
        return int'((address-BASE)>>6);
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"ATM_FAIL op=%0d: %s",op,why);
    endtask
    always @(posedge clk) begin
        cycles++;
        if(!rst_n)begin
            aw_pending=0;w_last_seen=0;write_applied=0;ar_pending=0;aw_beats=0;w_pending=0;ar_beats=0;ar_index=0;
            m_axi_awready<=0;m_axi_wready<=0;m_axi_bvalid<=0;m_axi_bresp<=0;m_axi_bid<=0;
            m_axi_arready<=0;m_axi_rvalid<=0;m_axi_rdata<=0;m_axi_rresp<=0;m_axi_rid<=0;m_axi_rlast<=0;
            aw_held=0;w_held=0;ar_held=0;tile_held=0;
        end else begin
            m_axi_awready<=!freeze_aw && $urandom_range(0,3)!=0;
            m_axi_wready<=!freeze_w && $urandom_range(0,3)!=0;
            m_axi_arready<=!freeze_ar && $urandom_range(0,3)!=0;
            if(aw_held)require(m_axi_awvalid && saved_aw=={m_axi_awaddr,m_axi_awlen},"Held AW changed");
            if(w_held)require(m_axi_wvalid && saved_w=={m_axi_wdata,m_axi_wstrb,m_axi_wlast},"Held W changed");
            if(ar_held)require(m_axi_arvalid && saved_ar=={m_axi_araddr,m_axi_arlen},"Held AR changed");
            if(tile_held)require(tile_valid && saved_tile=={tile_data,tile_plane,tile_last,tile_index,tile_sequence,tile_epoch,tile_lease_id},"Held tile changed");
            tile_held=tile_valid && !tile_ready;
            saved_tile={tile_data,tile_plane,tile_last,tile_index,tile_sequence,tile_epoch,tile_lease_id};
            aw_held=m_axi_awvalid && !m_axi_awready;w_held=m_axi_wvalid && !m_axi_wready;ar_held=m_axi_arvalid && !m_axi_arready;
            saved_aw={m_axi_awaddr,m_axi_awlen};saved_w={m_axi_wdata,m_axi_wstrb,m_axi_wlast};saved_ar={m_axi_araddr,m_axi_arlen};
            aw_waits+=int'(aw_held);w_waits+=int'(w_held);ar_waits+=int'(ar_held);tile_waits+=int'(tile_valid && !tile_ready);
            if(m_axi_bvalid && m_axi_bready)begin
                $display("ATM_B cycle=%0d id=%0d ordinal=%0d resp=%0d",cycles,writer_op,write_b_count,m_axi_bresp);
                b_count++;write_b_count++;m_axi_bvalid<=0;aw_pending=0;w_last_seen=0;write_applied=0;w_pending=0;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 &&
                    m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW protocol");
                aw_address=m_axi_awaddr;aw_beats=int'(m_axi_awlen)+1;aw_pending=1;aw_count++;
                require(aw_beats<=4 && int'(aw_address[11:0])+aw_beats*64<=4096,"AW page");
                $display("ATM_AW cycle=%0d id=%0d address=%016h beats=%0d",cycles,writer_op,aw_address,aw_beats);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(w_pending<4 && !w_last_seen && m_axi_wstrb==64'hffffffffffffffff,"W protocol");
                pending_w[w_pending]=m_axi_wdata;w_pending++;w_count++;
                if(m_axi_wlast)w_last_seen=1;
                $display("ATM_W cycle=%0d id=%0d last=%0d data=%0128h",cycles,writer_op,m_axi_wlast,m_axi_wdata);
            end
            if(aw_pending && w_last_seen && !write_applied)begin
                require(w_pending==aw_beats,"AW/W lengths");
                for(int i=0;i<aw_beats;i++)begin memory[slot(aw_address+64'(i*64))]=pending_w[i];initialized[slot(aw_address+64'(i*64))]=1;end
                write_applied=1;
            end
            if(!m_axi_bvalid && write_applied && !freeze_b && $urandom_range(0,3)!=0)begin
                m_axi_bvalid<=1;m_axi_bresp<=(write_error_mode==1 || (write_error_mode==2 && write_b_count==15)) ? 2:0;m_axi_bid<=0;
            end
            if(m_axi_rvalid && m_axi_rready)begin
                $display("ATM_R cycle=%0d id=%0d address=%016h resp=%0d last=%0d data=%0128h",cycles,reader_op,ar_address+64'(ar_index*64),m_axi_rresp,m_axi_rlast,m_axi_rdata);
                r_count++;read_r_count++;ar_index++;m_axi_rvalid<=0;
                if(ar_index==ar_beats)ar_pending=0;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!ar_pending && !m_axi_rvalid && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 &&
                    m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR protocol");
                ar_address=m_axi_araddr;ar_beats=int'(m_axi_arlen)+1;ar_index=0;ar_pending=1;ar_count++;
                require(int'(ar_address[11:0])+ar_beats*64<=4096,"AR page");
                $display("ATM_AR cycle=%0d id=%0d address=%016h beats=%0d",cycles,reader_op,ar_address,ar_beats);
            end
            if(!m_axi_rvalid && ar_pending && !freeze_r && $urandom_range(0,3)!=0)begin
                require(initialized[slot(ar_address+64'(ar_index*64))],"Read uninitialized physical bytes");
                m_axi_rdata<=memory[slot(ar_address+64'(ar_index*64))];m_axi_rvalid<=1;m_axi_rlast<=ar_index+1==ar_beats;
                m_axi_rresp<=read_r_count==read_error_at ? 2:0;m_axi_rid<=0;
            end
            if((aw_pending || m_axi_wvalid) && (ar_pending || m_axi_rvalid))simultaneous_rw++;
        end
    end
    task automatic bind_cache(input int capacity,offset);
        int id,ticks;
        op++;id=op;bind_count++;generation++;ticks=0;
        @(negedge clk);require(bind_ready,"Bind blocked after releasing lease");
        base_model=BASE+64'(offset);cap=capacity;seq_model=64'h1234567800000000+64'(generation);epoch_model=64'hfedcba9800000000+64'(generation);
        bind_cache_base=base_model;bind_region_base=BASE;bind_region_bytes=64'(capacity)*65536+8192;bind_capacity=32'(capacity);
        bind_sequence=seq_model;bind_epoch=epoch_model;expected_epoch=epoch_model;bind_valid=1;
        $display("ATM_BIND cycle=%0d id=%0d generation=%0d capacity=%0d base=%016h sequence=%016h epoch=%016h",cycles,id,generation,cap,base_model,seq_model,epoch_model);
        do @(posedge clk);while(!bind_ready);
        @(negedge clk);bind_valid=0;bind_cache_base=0;bind_capacity=0;bind_sequence=0;bind_epoch=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Bind timeout");end
        require(append_done_bind && append_done_status==0 && cache_valid && cache_base==base_model && cache_capacity==32'(cap),"Bind result");
        require(append_done_sequence==seq_model && append_done_epoch==epoch_model,"Bind identity");
        append_done_ready=1;@(negedge clk);append_done_ready=0;
        for(int i=0;i<16;i++)prefix[i]=0;
        $display("ATM_BIND_DONE cycle=%0d id=%0d status=0",cycles,id);
    endtask
    task automatic append_token(input int layer,pos,mode,want_status);
        int id,index,ticks,old_count;
        op++;id=op;writer_op=id;append_count++;index=0;ticks=0;old_count=prefix[layer];write_error_mode=mode;write_b_count=0;
        @(negedge clk);append_layer=32'(layer);append_position=32'(pos);append_sequence=seq_model;append_epoch=epoch_model;append_valid=1;
        $display("ATM_APPEND cycle=%0d id=%0d generation=%0d layer=%0d position=%0d mode=%0d prefix=%0d",cycles,id,generation,layer,pos,mode,old_count);
        do @(posedge clk);while(!append_ready);
        @(negedge clk);append_valid=0;append_layer=0;append_position=0;append_sequence=0;append_epoch=0;
        while(!append_prepared && !append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Append admission timeout");end
        while(index<64 && append_prepared)begin
            data_valid=$urandom_range(0,3)!=0;data_bits=pattern(generation,layer,pos,index);data_index=6'(index);data_last=index==63;
            data_sequence=seq_model;data_epoch=epoch_model;commit_valid=data_valid && index==63;commit_status=0;commit_sequence=seq_model;commit_epoch=epoch_model;
            @(posedge clk);if(data_valid && data_ready)index++;
            @(negedge clk);
        end
        data_valid=0;commit_valid=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<4000,"Append completion timeout");end
        require(!append_done_bind && append_done_status==4'(want_status),"Append completion status");
        require(append_done_sequence==seq_model && append_done_epoch==epoch_model,"Append completion identity");
        if(want_status==0)begin require(append_done_bytes==4096,"Complete physical token");prefix[layer]=pos+1;end
        require(append_done_committed_tokens==32'(prefix[layer]),"Owned append counter");
        $display("ATM_APPEND_DONE cycle=%0d id=%0d status=%0d bytes=%0d committed=%0d",cycles,id,append_done_status,append_done_bytes,append_done_committed_tokens);
        repeat(4)@(negedge clk);append_done_ready=1;@(negedge clk);append_done_ready=0;write_error_mode=0;
    endtask
    int queries=0,peer_jobs=0,completions=0,result_words=0,kv_words=0,q_words=0,state_words=0;
    int q_layer=0,q_head=0,q_position=0,q_mode=0,q_start_cycle=0,q_tiles=0;
    logic [63:0] q_seq,q_epoch,q_lease;
    logic [31:0] qbits[0:63],acc[0:63];
    bit peer_busy=0;
    function automatic logic [31:0] rotate(input logic [31:0] word,input int count);
        rotate=(word<<count)|(word>>(32-count));
    endfunction
    function automatic logic [31:0] query_word(input int image,layer,head,lane);
        query_word=32'h3e000000 | (32'(image*131071+layer*4099+head*131+lane)&32'h007fffff);
    endfunction
    function automatic logic [31:0] kv_word(input int pos,lane,plane);
        logic [511:0] bits;
        bits=pattern(generation,q_layer,pos,plane*32+(q_head/4)*4+lane/16);
        return bits[(lane%16)*32+:32];
    endfunction
    function automatic logic [31:0] reference_lane(input int lane,position);
        logic [31:0] value;
        value=query_word(generation,q_layer,q_head,lane);
        for(int pos=0;pos<=position;pos++)
            value=rotate(value,3)^kv_word(pos,lane,0)^rotate(kv_word(pos,lane,1),7)^query_word(generation,q_layer,q_head,lane);
        return value;
    endfunction
    function automatic logic [63:0] state_tag(input int ending,index);
        return (64'(generation)<<48)|(64'(q_layer)<<40)|(64'(q_head)<<32)|(64'(ending)<<8)|64'(index);
    endfunction
    task automatic tick;@(posedge clk);#1;endtask
    task automatic quiet_peer;
        hls_ready=0;hls_done=0;hls_return=0;hls_query_ce=0;hls_state_in_ce=0;
        hls_state_out_ce=0;hls_state_out_we=0;hls_output_ce=0;hls_output_we=0;
        hls_keys_ce=0;hls_values_ce=0;
    endtask
    always @(posedge clk)if(rst_n)begin
        if(hls_start || peer_busy)require(lease_active,"Lease released while arithmetic peer owns tile");
        if(result_valid)require(completion_valid && !lease_active && !peer_busy,"Result before global drain");
        if(!result_valid)require(!result_q_valid && result_q===0,"Private result exposed");
        if(dut.compute_valid && dut.compute_ready)
            $display("ATM_COMPUTE cycle=%0d query=%0d start=%0d tokens=%0d layer=%0d head=%0d kv=%0d lease=%016h",cycles,queries,dut.compute_start,dut.compute_tokens,dut.compute_layer,dut.compute_query_head,dut.compute_kv_head,dut.compute_lease);
        if(dut.finish_valid && dut.finish_ready)
            $display("ATM_FINISH cycle=%0d query=%0d start=%0d status=%0d abort=%0d",cycles,queries,dut.compute_start,dut.finish_status,dut.compute_abort);
        if(dut.query_done_valid && dut.query_done_ready)
            $display("ATM_CERTIFICATE cycle=%0d query=%0d status=%0d tokens=%0d tiles=%0d lease=%016h",cycles,queries,dut.query_done_status,dut.query_done_tokens,dut.query_done_tiles,dut.query_done_lease);
        if(dut.kv_path.lease_cmd_valid && dut.kv_path.lease_cmd_ready)
            $display("ATM_LEASE cycle=%0d query=%0d release=%0d lease=%016h",cycles,queries,dut.kv_path.lease_cmd_release,dut.kv_path.lease_cmd_id);
        if(dut.kv_path.lease_done_valid && dut.kv_path.lease_done_ready)
            $display("ATM_LEASE_DONE cycle=%0d query=%0d release=%0d status=%0d active=%0d",cycles,queries,dut.kv_path.lease_done_release,dut.kv_path.lease_done_status,lease_active);
    end
    task automatic reset_all;
        reset_count++;$display("ATM_RESET cycle=%0d count=%0d query=%0d",cycles,reset_count,queries);
        @(negedge clk);rst_n=0;bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;append_done_ready=0;
        command_valid=0;cancel_valid=0;completion_ready=0;result_ce=0;quiet_peer();peer_busy=0;
        freeze_aw=0;freeze_w=0;freeze_ar=0;freeze_r=0;freeze_b=0;read_error_at=-1;write_error_mode=0;
        repeat(4)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(command_ready && !completion_valid && !lease_active && !result_valid && !hls_start,"Reset ownership");
        for(int i=0;i<16;i++)prefix[i]=0;
    endtask
    task automatic start_query(input int layer,head,position,mode);
        queries++;reader_op=queries;q_layer=layer;q_head=head;q_position=position;q_mode=mode;q_tiles=0;
        q_seq=seq_model;q_epoch=epoch_model;q_lease=next_lease;next_lease++;
        read_r_count=0;read_error_at=mode==1 ? 3:-1;q_start_cycle=cycles;
        freeze_ar=mode==2;freeze_r=mode==3;
        @(negedge clk);require(command_ready,"Query admission blocked");command_valid=1;
        command_position=32'(position);command_layer=32'(layer);command_head=32'(head);
        command_sequence=q_seq;command_epoch=q_epoch;command_lease=q_lease;
        for(int lane=0;lane<64;lane++)command_query[lane*32+:32]=query_word(generation,layer,head,lane);
        if(mode==15)command_query[31:0]=32'h7f800000;
        if(mode==16)command_sequence=q_seq^64'h100000000;
        $display("ATM_QUERY cycle=%0d query=%0d generation=%0d layer=%0d head=%0d position=%0d mode=%0d seq=%016h epoch=%016h lease=%016h",cycles,queries,generation,layer,head,position,mode,command_sequence,command_epoch,command_lease);
        tick();@(negedge clk);command_valid=0;command_query='1;command_sequence=0;command_lease=0;
    endtask
    task automatic cancel_query(input bit revoke);
        @(negedge clk);
        if(revoke)expected_epoch=epoch_model+1;
        else begin require(cancel_ready,"Cancel not accepted");cancel_valid=1;end
        tick();@(negedge clk);cancel_valid=0;
    endtask
    // A deterministic checksum peer exercises every Q/K/V word and state bank.
    // This is NOT the Vitis core and does not claim attention arithmetic.
    task automatic run_peer;
        int first,count,local_mode;
        logic final_tile,aborted;
        logic [31:0] k,v;
        first=int'(hls_tile_start);count=int'(hls_tile_count);final_tile=first+count==q_position+1;
        local_mode=q_mode==14 && first==0 ? 0:q_mode;aborted=0;
        peer_jobs++;peer_busy=1;q_tiles++;
        require(hls_start && lease_active && hls_query_position==32'(q_position) && hls_layer==32'(q_layer) &&
            hls_query_head==32'(q_head) && hls_kv_head==32'(q_head/4) && hls_sequence==q_seq && hls_epoch==q_epoch,"Peer metadata");
        $display("ATM_PEER cycle=%0d query=%0d start=%0d count=%0d mode=%0d",cycles,queries,first,count,local_mode);
        repeat(7)begin tick();require(hls_start && lease_active && !bind_ready,"Start/lease not retained");end
        if(local_mode==4)begin cancel_query(0);aborted=1;end
        @(negedge clk);hls_ready=1;tick();@(negedge clk);hls_ready=0;
        for(int lane=0;lane<64;lane++)begin
            hls_query_ce=1;hls_query_address=6'(lane);tick();
            require(hls_query_q===query_word(generation,q_layer,q_head,lane),"Private Q changed");
            qbits[lane]=hls_query_q;acc[lane]=hls_query_q;q_words++;
            $display("ATM_Q cycle=%0d query=%0d start=%0d lane=%0d data=%08h",cycles,queries,first,lane,hls_query_q);
            @(negedge clk);
        end
        hls_query_ce=0;
        if(first!=0)begin
            for(int index=0;index<104;index++)begin
                hls_state_in_ce=1;hls_state_in_address=7'(index);tick();state_words++;
                if(index<40)require(hls_state_in_q===state_tag(first,index),"State tag mismatch");
                else begin
                    require(hls_state_in_q==={32'd0,reference_lane(index-40,first-1)},"Continuation state mismatch");
                    acc[index-40]=hls_state_in_q[31:0];
                end
                $display("ATM_STATE_R cycle=%0d query=%0d start=%0d index=%0d data=%016h",cycles,queries,first,index,hls_state_in_q);
                @(negedge clk);
            end
        end
        hls_state_in_ce=0;
        for(int index=0;index<count*64;index++)begin
            hls_keys_ce=1;hls_values_ce=1;hls_keys_address=12'(index);hls_values_address=12'(index);tick();
            if(aborted)require(!hls_keys_q_valid && !hls_values_q_valid && hls_keys_q===0 && hls_values_q===0,"Revoked K/V leaked");
            else begin
                require(hls_keys_q_valid && hls_values_q_valid && hls_keys_q===kv_word(first+index/64,index%64,0) &&
                    hls_values_q===kv_word(first+index/64,index%64,1),"K/V data mismatch");
            end
            k=hls_keys_q;v=hls_values_q;acc[index%64]=rotate(acc[index%64],3)^k^rotate(v,7)^qbits[index%64];kv_words+=2;
            $display("ATM_KV cycle=%0d query=%0d start=%0d index=%0d valid=%0d key=%08h value=%08h",cycles,queries,first,index,hls_keys_q_valid && hls_values_q_valid,k,v);
            @(negedge clk);
            if(index==2 && (local_mode==5 || local_mode==6))begin
                hls_keys_ce=0;hls_values_ce=0;cancel_query(local_mode==6);aborted=1;
            end
        end
        hls_keys_ce=0;hls_values_ce=0;
        if(local_mode==9)begin
            hls_keys_ce=1;hls_keys_address=12'(count*64);tick();@(negedge clk);hls_keys_ce=0;aborted=1;
        end
        for(int index=0;index<104;index++)begin
            hls_state_out_ce=1;hls_state_out_we=1;hls_state_out_address=7'(index);
            hls_state_out_d=index<40 ? state_tag(first+count,index):{32'd0,acc[index-40]};
            if(local_mode==8 && index==103)hls_state_out_we=0;
            if(!final_tile && index==103)begin hls_done=1;hls_return=local_mode==7 || local_mode==14 ? 5:0;end
            tick();$display("ATM_STATE_W cycle=%0d query=%0d start=%0d index=%0d we=%0d data=%016h",cycles,queries,first,index,hls_state_out_we,hls_state_out_d);
            @(negedge clk);
        end
        hls_state_out_ce=0;hls_state_out_we=0;
        if(final_tile)begin
            for(int lane=0;lane<64;lane++)begin
                hls_output_ce=1;hls_output_we=1;hls_output_address=6'(lane);hls_output_d=32'h3f000000|(acc[lane]&32'h007fffff);
                if(local_mode==10 && lane==9)hls_output_d=32'h7f800001;
                if(local_mode==11 && lane==63)hls_output_we=0;
                if(lane==63)begin hls_done=1;hls_return=local_mode==7 || local_mode==14 ? 5:0;end
                tick();$display("ATM_OUTPUT_W cycle=%0d query=%0d lane=%0d we=%0d data=%08h",cycles,queries,lane,hls_output_we,hls_output_d);
                @(negedge clk);
            end
        end
        quiet_peer();peer_busy=0;#1;
        $display("ATM_PEER_DONE cycle=%0d query=%0d start=%0d aborted=%0d",cycles,queries,first,aborted);
    endtask
    task automatic serve;
        int ticks;
        ticks=0;
        while(!completion_valid)begin
            if(hls_start)run_peer();else @(negedge clk);
            ticks++;require(ticks<30000+WATCHDOG_CYCLES,"Query/lease drain timeout");
        end
    endtask
    task automatic collect(input int want_status,input bit revoke);
        int ticks;logic [3:0] saved;
        ticks=0;
        while(!completion_valid)begin @(negedge clk);ticks++;require(ticks<60000+WATCHDOG_CYCLES,"Completion timeout");end
        saved=completion_status;require(saved==4'(want_status) && result_valid==(want_status==0),"Completion status/authorization");
        require(!lease_active && !peer_busy && !m_axi_rvalid && !ar_pending,"Completion before drain");
        repeat(7)begin tick();require(completion_valid && completion_status==saved && !command_ready,"Completion not held");end
        if(want_status==0)begin
            for(int lane=0;lane<64;lane++)begin
                @(negedge clk);result_ce=1;result_address=6'(lane);tick();
                require(result_q_valid && result_q===(32'h3f000000|(reference_lane(lane,q_position)&32'h007fffff)),"Final checksum mismatch");
                result_words++;$display("ATM_RESULT cycle=%0d query=%0d lane=%0d data=%08h",cycles,queries,lane,result_q);
            end
        end
        if(revoke)begin
            @(negedge clk);expected_epoch=epoch_model+1;result_ce=1;#1;
            require(!result_valid && !result_q_valid && result_q===0,"Late epoch leak");
            tick();@(negedge clk);expected_epoch=epoch_model;tick();
            require(!result_valid && !result_q_valid && completion_status==saved,"Late revocation not sticky");
        end
        $display("ATM_COMPLETE cycle=%0d query=%0d status=%0d peers=%0d revoke=%0d elapsed=%0d",cycles,queries,saved,q_tiles,revoke,cycles-q_start_cycle);
        completions++;@(negedge clk);completion_ready=1;result_ce=0;tick();@(negedge clk);completion_ready=0;
        expected_epoch=epoch_model;read_error_at=-1;freeze_ar=0;freeze_r=0;
    endtask
    task automatic inject;
        if(q_mode==2)begin wait(m_axi_arvalid);cancel_query(0);repeat(4)tick();@(negedge clk);freeze_ar=0;end
        if(q_mode==3)begin wait(ar_pending);repeat(WATCHDOG_CYCLES+8)tick();@(negedge clk);freeze_r=0;end
        if(q_mode==13)begin wait(dut.query_done_tokens==64);cancel_query(0);end
    endtask
    task automatic run_query(input int layer,head,position,mode,want_status);
        start_query(layer,head,position,mode);
        fork serve();collect(want_status,mode==12);inject();join
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        reset_all();bind_cache(4,4032);
        for(int pos=0;pos<3;pos++)append_token(0,pos,0,0);
        append_token(15,0,0,0);
        for(int head=0;head<32;head++)run_query(0,head,2,0,0);
        run_query(15,31,0,0,0);
        for(int mode=1;mode<=12;mode++)
            run_query(0,0,0,mode,mode==1 ? 3:mode==2 || mode==4 || mode==5 || mode==9 ? 7:mode==3 ? 8:mode==6 ? 2:mode==7 ? 10:mode==8 || mode==10 || mode==11 ? 4:0);
        run_query(0,0,0,15,4);run_query(0,0,0,16,2);run_query(16,0,0,0,1);run_query(0,32,0,0,1);
        append_token(0,3,1,3);run_query(0,0,3,0,1);append_token(0,3,0,0);run_query(0,0,3,0,0);
        bind_cache(130,4032);
        for(int pos=0;pos<129;pos++)append_token(15,pos,0,0);
        run_query(15,31,63,0,0);run_query(15,31,64,0,0);run_query(15,31,128,0,0);
        fork run_query(15,31,128,0,0);begin wait(ar_pending);append_token(15,129,0,0);end join
        require(simultaneous_rw>0,"No physical AXI overlap");
        run_query(15,31,64,13,7);run_query(15,31,64,14,10);
        // Reset peers together with AR pending, HLS start held and result held.
        for(int phase=0;phase<3;phase++)begin
            start_query(15,31,0,0);
            if(phase==0)begin freeze_r=1;wait(ar_pending);end
            else if(phase==1)wait(hls_start);
            else serve();
            $display("ATM_QUERY_RESET cycle=%0d query=%0d phase=%0d",cycles,queries,phase);reset_all();
            bind_cache(1,64);append_token(15,0,0,0);
        end
        run_query(15,31,0,0,0);
        $display("ATM_PASS seed=%0d watchdog=%0d binds=%0d appends=%0d queries=%0d resets=%0d peers=%0d q_words=%0d kv_words=%0d state_words=%0d result_words=%0d completions=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d overlap=%0d cycles=%0d",seed,WATCHDOG_CYCLES,bind_count,append_count,queries,reset_count,peer_jobs,q_words,kv_words,state_words,result_words,completions,aw_count,w_count,b_count,ar_count,r_count,simultaneous_rw,cycles);
        $finish;
    end
    initial begin #1000000000;$fatal(1,"ATM_FAIL global timeout");end
endmodule
`default_nettype wire
