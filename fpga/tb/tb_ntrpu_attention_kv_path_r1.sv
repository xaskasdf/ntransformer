`timescale 1ns/1ps
module tb_ntrpu_attention_kv_path_r1 #(parameter integer WATCHDOG_CYCLES=32);
    logic clk=0,rst_n=0;
    logic bind_valid=0;
    logic bind_ready;
    logic [63:0] bind_cache_base=0,bind_region_base=0,bind_region_bytes=0;
    logic [31:0] bind_capacity=0;
    logic [63:0] bind_sequence=0,bind_epoch=0,expected_epoch=0;
    logic cache_valid;
    logic [63:0] cache_base,cache_region_base,cache_region_bytes,cache_sequence,cache_epoch;
    logic [31:0] cache_capacity;
    logic append_valid=0;
    logic append_ready,append_prepared;
    logic [31:0] append_layer=0,append_position=0;
    logic [63:0] append_sequence=0,append_epoch=0;
    logic data_valid=0;
    logic data_ready;
    logic [511:0] data_bits=0;
    logic [5:0] data_index=0;
    logic data_last=0;
    logic [63:0] data_sequence=0,data_epoch=0;
    logic commit_valid=0;
    logic commit_ready;
    logic [3:0] commit_status=0;
    logic [63:0] commit_sequence=0,commit_epoch=0;
    logic append_cancel_valid=0;
    logic append_cancel_ready;
    logic append_done_valid;
    logic append_done_ready=0;
    logic append_done_bind;
    logic [3:0] append_done_status;
    logic [12:0] append_done_bytes;
    logic [31:0] append_done_committed_tokens;
    logic [63:0] append_done_sequence,append_done_epoch;
    logic query_valid=0;
    logic query_ready,lease_active;
    logic [31:0] query_position=0,query_layer=0,query_head=0;
    logic [63:0] query_sequence=0,query_epoch=0,query_lease=0;
    logic query_cancel_valid=0;
    logic query_cancel_ready;
    logic query_done_valid;
    logic query_done_ready=0;
    logic [3:0] query_done_status;
    logic [31:0] query_done_tokens,query_done_tiles;
    logic [63:0] query_done_sequence,query_done_epoch,query_done_lease;
    logic compute_valid;
    logic compute_ready=0;
    logic compute_abort;
    logic [31:0] compute_start,compute_tokens,compute_position,compute_layer,compute_query_head,compute_kv_head;
    logic [63:0] compute_sequence,compute_epoch,compute_lease;
    logic keys_ce=0,values_ce=0;
    logic [11:0] keys_address=0,values_address=0;
    logic [31:0] keys_q,values_q;
    logic keys_q_valid,values_q_valid;
    logic finish_valid=0;
    logic finish_ready;
    logic [3:0] finish_status=0;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid,m_axi_awcache;
    logic m_axi_awlock,m_axi_awvalid;
    logic m_axi_awready=0;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast,m_axi_wvalid;
    logic m_axi_wready=0;
    logic [3:0] m_axi_bid=0;
    logic [1:0] m_axi_bresp=0;
    logic m_axi_bvalid=0;
    logic m_axi_bready;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize,m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid,m_axi_arcache;
    logic m_axi_arlock,m_axi_arvalid;
    logic m_axi_arready=0;
    logic [511:0] m_axi_rdata=0;
    logic [1:0] m_axi_rresp=0;
    logic [3:0] m_axi_rid=0;
    logic m_axi_rlast=0,m_axi_rvalid=0;
    logic m_axi_rready;
    always #2.5 clk=~clk;
    ntrpu_attention_kv_path_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
    wire  tile_valid=dut.tile_valid;
    wire  tile_ready=dut.tile_ready;
    wire [511:0] tile_data=dut.tile_data;
    wire  tile_plane=dut.tile_plane;
    wire  tile_last=dut.tile_last;
    wire [7:0] tile_index=dut.tile_index;
    wire [63:0] tile_sequence=dut.tile_sequence;
    wire [63:0] tile_epoch=dut.tile_epoch;
    wire [63:0] tile_lease_id=dut.tile_lease_id;
    localparam logic [63:0] BASE=64'ha000000000;
    logic [511:0] memory[0:131071],pending_w[0:3];
    bit initialized[0:131071];
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
        if(!(address>=BASE && address<BASE+64'd8388608 && address[5:0]==0))
            $fatal(1,"KVP_FAIL op=%0d: Physical memory range",op);
        return int'((address-BASE)>>6);
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"KVP_FAIL op=%0d: %s",op,why);
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
                $display("KVP_B cycle=%0d id=%0d ordinal=%0d resp=%0d",cycles,writer_op,write_b_count,m_axi_bresp);
                b_count++;write_b_count++;m_axi_bvalid<=0;aw_pending=0;w_last_seen=0;write_applied=0;w_pending=0;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 &&
                    m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW protocol");
                aw_address=m_axi_awaddr;aw_beats=int'(m_axi_awlen)+1;aw_pending=1;aw_count++;
                require(aw_beats<=4 && int'(aw_address[11:0])+aw_beats*64<=4096,"AW page");
                $display("KVP_AW cycle=%0d id=%0d address=%016h beats=%0d",cycles,writer_op,aw_address,aw_beats);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(w_pending<4 && !w_last_seen && m_axi_wstrb==64'hffffffffffffffff,"W protocol");
                pending_w[w_pending]=m_axi_wdata;w_pending++;w_count++;
                if(m_axi_wlast)w_last_seen=1;
                $display("KVP_W cycle=%0d id=%0d last=%0d data=%0128h",cycles,writer_op,m_axi_wlast,m_axi_wdata);
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
                $display("KVP_R cycle=%0d id=%0d address=%016h resp=%0d last=%0d data=%0128h",cycles,reader_op,ar_address+64'(ar_index*64),m_axi_rresp,m_axi_rlast,m_axi_rdata);
                r_count++;read_r_count++;ar_index++;m_axi_rvalid<=0;
                if(ar_index==ar_beats)ar_pending=0;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!ar_pending && !m_axi_rvalid && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 &&
                    m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR protocol");
                ar_address=m_axi_araddr;ar_beats=int'(m_axi_arlen)+1;ar_index=0;ar_pending=1;ar_count++;
                require(int'(ar_address[11:0])+ar_beats*64<=4096,"AR page");
                $display("KVP_AR cycle=%0d id=%0d address=%016h beats=%0d",cycles,reader_op,ar_address,ar_beats);
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
        $display("KVP_BIND cycle=%0d id=%0d generation=%0d capacity=%0d base=%016h sequence=%016h epoch=%016h",cycles,id,generation,cap,base_model,seq_model,epoch_model);
        do @(posedge clk);while(!bind_ready);
        @(negedge clk);bind_valid=0;bind_cache_base=0;bind_capacity=0;bind_sequence=0;bind_epoch=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Bind timeout");end
        require(append_done_bind && append_done_status==0 && cache_valid && cache_base==base_model && cache_capacity==32'(cap),"Bind result");
        require(append_done_sequence==seq_model && append_done_epoch==epoch_model,"Bind identity");
        append_done_ready=1;@(negedge clk);append_done_ready=0;
        for(int i=0;i<16;i++)prefix[i]=0;
        $display("KVP_BIND_DONE cycle=%0d id=%0d status=0",cycles,id);
    endtask
    task automatic append_token(input int layer,pos,mode,want_status);
        int id,index,ticks,old_count;
        op++;id=op;writer_op=id;append_count++;index=0;ticks=0;old_count=prefix[layer];write_error_mode=mode;write_b_count=0;
        @(negedge clk);append_layer=32'(layer);append_position=32'(pos);append_sequence=seq_model;append_epoch=epoch_model;append_valid=1;
        $display("KVP_APPEND cycle=%0d id=%0d generation=%0d layer=%0d position=%0d mode=%0d prefix=%0d",cycles,id,generation,layer,pos,mode,old_count);
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
        $display("KVP_APPEND_DONE cycle=%0d id=%0d status=%0d bytes=%0d committed=%0d",cycles,id,append_done_status,append_done_bytes,append_done_committed_tokens);
        repeat(4)@(negedge clk);append_done_ready=1;@(negedge clk);append_done_ready=0;write_error_mode=0;
    endtask
    int queries=0,consumer_jobs=0,ram_reads=0,query_completions=0;
    int q_layer=0,q_head=0,q_position=0,q_mode=0,q_start_cycle=0;
    int q_seen_tiles=0,q_seen_tokens=0;
    logic [63:0] q_seq,q_epoch,q_lease;
    task automatic tick; @(posedge clk);#1;endtask
    task automatic hidden;
        require(!keys_q_valid && !values_q_valid && keys_q===0 && values_q===0,"Private RAM leaked");
    endtask
    always @(posedge clk)if(rst_n)begin
        if(compute_valid && compute_ready)begin
            consumer_jobs++;
            $display("KVP_COMPUTE cycle=%0d query=%0d start=%0d tokens=%0d layer=%0d head=%0d kv=%0d lease=%016h",cycles,queries,compute_start,compute_tokens,compute_layer,compute_query_head,compute_kv_head,compute_lease);
        end
        if(finish_valid && finish_ready)
            $display("KVP_FINISH cycle=%0d query=%0d start=%0d status=%0d abort=%0d",cycles,queries,compute_start,finish_status,compute_abort);
        if(dut.lease_cmd_valid && dut.lease_cmd_ready)
            $display("KVP_LEASE cycle=%0d query=%0d release=%0d layer=%0d head=%0d position=%0d seq=%016h epoch=%016h lease=%016h",cycles,queries,dut.lease_cmd_release,dut.lease_cmd_layer,dut.lease_cmd_query_head,dut.lease_cmd_position,dut.lease_cmd_sequence,dut.lease_cmd_epoch,dut.lease_cmd_id);
        if(dut.lease_done_valid && dut.lease_done_ready)
            $display("KVP_LEASE_DONE cycle=%0d query=%0d release=%0d status=%0d prefix=%0d active=%0d",cycles,queries,dut.lease_done_release,dut.lease_done_status,dut.lease_done_prefix,lease_active);
        if(tile_valid && tile_ready)
            $display("KVP_TILE cycle=%0d query=%0d plane=%0d index=%0d last=%0d seq=%016h epoch=%016h lease=%016h data=%0128h",cycles,queries,tile_plane,tile_index,tile_last,tile_sequence,tile_epoch,tile_lease_id,tile_data);
        if(dut.read_done_valid && dut.read_done_ready)
            $display("KVP_READ_DONE cycle=%0d query=%0d status=%0d beats=%0d bytes=%0d",cycles,queries,dut.read_done_status,dut.read_done_beats,dut.read_done_bytes);
    end
    task automatic reset_all;
        reset_count++;$display("KVP_RESET cycle=%0d count=%0d query=%0d",cycles,reset_count,queries);
        @(negedge clk);rst_n=0;bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;append_done_ready=0;
        query_valid=0;query_cancel_valid=0;query_done_ready=0;compute_ready=0;keys_ce=0;values_ce=0;finish_valid=0;
        freeze_aw=0;freeze_w=0;freeze_ar=0;freeze_r=0;freeze_b=0;read_error_at=-1;write_error_mode=0;
        repeat(3)@(negedge clk);
        require(!cache_valid && !lease_active && !query_done_valid && !compute_valid && !append_done_valid,"Reset visibility");
        require(!m_axi_awvalid && !m_axi_wvalid && !m_axi_arvalid,"Reset AXI");hidden();
        rst_n=1;for(int i=0;i<16;i++)prefix[i]=0;
    endtask
    task automatic start_query(input int layer,head,position,mode);
        int ticks;
        @(negedge clk);require(query_ready,"Previous query retained");
        queries++;reader_op=queries;q_layer=layer;q_head=head;q_position=position;q_mode=mode;
        q_seq=seq_model;q_epoch=epoch_model;q_lease=next_lease;next_lease++;
        if(mode==12)q_seq^=64'h100000000;
        if(mode==13)q_lease=lease_id_model;
        query_layer=32'(layer);query_head=32'(head);query_position=32'(position);
        query_sequence=q_seq;query_epoch=q_epoch;query_lease=q_lease;query_valid=1;
        read_r_count=0;read_error_at=mode==1 ? 3:-1;q_seen_tiles=0;q_seen_tokens=0;q_start_cycle=cycles;
        $display("KVP_QUERY cycle=%0d query=%0d generation=%0d layer=%0d head=%0d position=%0d mode=%0d seq=%016h epoch=%016h lease=%016h",cycles,queries,generation,layer,head,position,mode,q_seq,q_epoch,q_lease);
        tick();hidden();@(negedge clk);query_valid=0;
        query_layer=32'hffffffff;query_head=32'hffffffff;query_position=32'hffffffff;
        query_sequence=0;query_epoch=0;query_lease=0;
    endtask
    task automatic wait_available;
        int ticks;
        ticks=0;
        while(!compute_valid && !query_done_valid)begin
            tick();ticks++;require(ticks<WATCHDOG_CYCLES+30000,"Query progress timeout");hidden();
        end
    endtask
    task automatic cancel_query(input bit revoke);
        @(negedge clk);
        if(revoke)expected_epoch=epoch_model+1;
        else begin require(query_cancel_ready,"Query cancellation rejected");query_cancel_valid=1;end
        $display("KVP_CANCEL cycle=%0d query=%0d revoke=%0d",cycles,queries,revoke);
        tick();hidden();@(negedge clk);query_cancel_valid=0;
    endtask
    function automatic logic [31:0] expected_word(input int plane,address);
        logic [511:0] bits;
        bits=pattern(generation,q_layer,int'(compute_start)+address/64,plane*32+(q_head/4)*4+(address%64)/16);
        return bits[(address%16)*32+:32];
    endfunction
    task automatic consume_tile(input int mode);
        int words,observed;
        require(compute_valid && lease_active && !query_ready && !bind_ready,"Consumer started without lease");
        require(compute_start==32'(q_seen_tokens) && compute_tokens==(q_position-q_seen_tokens>=63 ? 64:q_position-q_seen_tokens+1),"Tile partition");
        require(compute_position==32'(q_position) && compute_layer==32'(q_layer) && compute_query_head==32'(q_head) && compute_kv_head==32'(q_head/4),"Owned query metadata");
        require(compute_sequence==q_seq && compute_epoch==q_epoch && compute_lease==q_lease,"Owned query identity");
        @(negedge clk);bind_valid=1;query_valid=1;keys_ce=1;values_ce=1;
        repeat(7)begin tick();hidden();require(compute_valid && lease_active && !bind_ready && !query_ready,"Available lease released early");end
        @(negedge clk);bind_valid=0;query_valid=0;keys_ce=0;values_ce=0;compute_ready=1;
        words=int'(compute_tokens)*64;
        tick();require(finish_ready && lease_active,"Consumer not retained");
        @(negedge clk);compute_ready=0;
        observed=mode>=5 && mode<=8 ? 3:words;
        for(int a=0;a<observed;a++)begin
            @(negedge clk);keys_ce=1;values_ce=1;keys_address=12'(a);values_address=12'(words-1-a);
            tick();require(keys_q_valid && values_q_valid,"RAM response missing");
            require(keys_q===expected_word(0,a) && values_q===expected_word(1,words-1-a),"AXI-to-RAM value mismatch");
            require(lease_active && !query_ready && !bind_ready && !query_done_valid,"Lease lost while reading");
            ram_reads++;
            $display("KVP_RAM cycle=%0d query=%0d start=%0d ka=%0d va=%0d k=%08h v=%08h",cycles,queries,compute_start,a,words-1-a,keys_q,values_q);
        end
        @(negedge clk);keys_ce=0;values_ce=0;
        if(mode==5 || mode==6)begin
            cancel_query(mode==6);
            repeat(9)begin tick();hidden();require(compute_abort && lease_active && !query_done_valid,"Consumer drain lease missing");end
        end else if(mode==8)begin
            @(negedge clk);keys_ce=1;keys_address=12'(words);tick();hidden();
            require(compute_abort && lease_active,"Bad RAM read did not retain lease");
            @(negedge clk);keys_ce=0;
        end
        @(negedge clk);finish_valid=1;finish_status=mode==7 ? 5:0;
        tick();hidden();@(negedge clk);finish_valid=0;
        if(mode==0)begin q_seen_tiles++;q_seen_tokens+=words/64;end
    endtask
    task automatic finish_query(input int want_status);
        int ticks;
        ticks=0;
        while(!query_done_valid)begin tick();ticks++;require(ticks<WATCHDOG_CYCLES+30000,"Query completion timeout");hidden();end
        require(query_done_status==4'(want_status),"Query completion status");
        require(query_done_sequence==q_seq && query_done_epoch==q_epoch && query_done_lease==q_lease,"Query completion identity");
        require(query_done_tokens==32'(q_seen_tokens) && query_done_tiles==32'(q_seen_tiles),"Query publication counters");
        require(!lease_active && !m_axi_arvalid && !ar_pending && !m_axi_rvalid,"Query returned before lease/AXI drain");
        @(negedge clk);query_valid=1;
        repeat(5)begin tick();hidden();require(query_done_valid && !query_ready && query_done_status==4'(want_status),"Held query result changed");end
        @(negedge clk);query_valid=0;query_done_ready=1;
        query_completions++;
        $display("KVP_QUERY_DONE cycle=%0d query=%0d status=%0d tokens=%0d tiles=%0d elapsed=%0d",cycles,queries,query_done_status,query_done_tokens,query_done_tiles,cycles-q_start_cycle);
        tick();require(query_ready,"Query did not retire");
        @(negedge clk);query_done_ready=0;expected_epoch=epoch_model;read_error_at=-1;
        lease_id_model=q_lease;
    endtask
    task automatic run_query(input int layer,head,position,mode,want_status);
        start_query(layer,head,position,mode);
        if(mode==2)begin
            wait(m_axi_arvalid);cancel_query(0);repeat(9)begin tick();require(lease_active && !query_done_valid,"Cancelled AR was not drained");end
            @(negedge clk);freeze_ar=0;
        end else if(mode==9)begin
            wait(ar_pending);repeat(WATCHDOG_CYCLES+4)tick();
            require(lease_active && !query_done_valid,"Timed-out R was not drained");@(negedge clk);freeze_r=0;
        end
        wait_available();
        if(mode==3 || mode==4)cancel_query(mode==4);
        else if(want_status==0)begin
            while(compute_valid)begin consume_tile(0);wait_available();end
        end else if(mode>=5 && mode<=8)consume_tile(mode);
        else require(!compute_valid,"Failed query reached consumer");
        finish_query(want_status);
    endtask
    task automatic long_fault(input int mode,want_status);
        start_query(15,31,64,mode);wait_available();consume_tile(0);
        if(mode==16 || mode==17)begin
            wait(query_done_tokens==64);cancel_query(mode==17);
        end else begin
            if(mode==15)read_error_at=read_r_count;
            wait_available();
            if(mode==14)consume_tile(7);
            else if(mode==18)consume_tile(5);
            else require(!compute_valid,"Failed second tile reached consumer");
        end
        finish_query(want_status);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        $display("KVP_START seed=%0d watchdog=%0d",seed,WATCHDOG_CYCLES);
        reset_all();bind_cache(4,4032);
        run_query(0,0,0,10,1);append_token(0,0,1,3);run_query(0,0,0,10,1);
        for(int pos=0;pos<3;pos++)append_token(0,pos,0,0);
        append_token(15,0,0,0);
        run_query(0,32,0,11,1);run_query(0,0,0,12,2);
        for(int head=0;head<32;head++)run_query(head[0] ? 15:0,head,head[0] ? 0:2,0,0);
        run_query(0,0,0,13,2);
        run_query(0,0,2,1,3);run_query(0,0,2,0,0);
        freeze_ar=1;run_query(0,0,2,2,7);
        run_query(0,0,2,3,7);run_query(0,0,2,4,2);
        run_query(0,0,2,5,7);run_query(0,0,2,6,2);
        run_query(0,0,2,7,5);run_query(0,0,2,8,9);
        freeze_r=1;run_query(0,0,2,9,8);
        // Appending a later token cannot grow the active query snapshot.
        fork
            run_query(0,31,2,0,0);
            begin wait(lease_active);append_token(0,3,0,0);end
        join
        require(prefix[0]==4,"Concurrent append missing");
        // Acquisition waits for all write responses that publish its first token.
        freeze_b=1;
        fork
            append_token(2,0,0,0);
            begin wait(m_axi_bready);run_query(2,0,0,0,0);end
            begin wait(m_axi_bready);repeat(8)begin tick();require(!lease_active && !compute_valid,"Acquired unpublished B");end @(negedge clk);freeze_b=0;end
        join
        // Cancellation before acquisition must leave the ongoing append intact.
        freeze_b=1;
        fork
            append_token(3,0,0,0);
            begin
                wait(m_axi_bready);start_query(3,0,0,24);
                wait(dut.lease_cmd_valid && !dut.lease_cmd_ready);cancel_query(0);finish_query(7);
                @(negedge clk);freeze_b=0;
            end
        join
        run_query(3,0,0,0,0);
        bind_cache(66,64);for(int pos=0;pos<65;pos++)append_token(15,pos,0,0);
        fork
            run_query(15,31,64,0,0);
            begin wait(lease_active);append_token(15,65,0,0);end
        join
        require(simultaneous_rw>0 && prefix[15]==66,"Concurrent AXI coverage missing");
        long_fault(14,5);long_fault(15,3);long_fault(16,7);long_fault(17,2);long_fault(18,7);
        run_query(15,31,64,0,0);
        // Reset load pending, available, executing, and held query response.
        for(int phase=0;phase<4;phase++)begin
            if(phase==0)freeze_r=1;
            start_query(15,31,0,20+phase);
            if(phase==0)wait(ar_pending);
            else begin
                wait_available();
                if(phase>=2)begin
                    @(negedge clk);compute_ready=1;tick();@(negedge clk);compute_ready=0;
                end
                if(phase==3)begin
                    @(negedge clk);finish_valid=1;finish_status=0;tick();@(negedge clk);finish_valid=0;
                    wait(query_done_valid);
                end
            end
            $display("KVP_QUERY_RESET cycle=%0d query=%0d phase=%0d",cycles,queries,phase);reset_all();
            bind_cache(1,4032);run_query(15,31,0,10,1);append_token(15,0,0,0);run_query(15,31,0,0,0);
        end
        require(!aw_pending && !ar_pending && !m_axi_bvalid && !m_axi_rvalid && !lease_active,"Undrained bus/lease");
        $display("KVP_PASS seed=%0d watchdog=%0d binds=%0d appends=%0d queries=%0d resets=%0d consumers=%0d reads=%0d completions=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d overlap=%0d cycles=%0d",seed,WATCHDOG_CYCLES,bind_count,append_count,queries,reset_count,consumer_jobs,ram_reads,query_completions,aw_count,w_count,b_count,ar_count,r_count,simultaneous_rw,cycles);
        $finish;
    end
    initial begin #1000000000;$fatal(1,"KVP_FAIL global timeout");end
endmodule
