`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_engine_r1 #(parameter integer WATCHDOG_CYCLES=65536);
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
    wire hls_start=dut.hls_start;
    wire hls_ready=dut.hls_ready;
    wire hls_done=dut.hls_done;
    wire [31:0] hls_return=dut.hls_return;
    wire [31:0] hls_tile_start=dut.hls_tile_start;
    wire [31:0] hls_tile_count=dut.hls_tile_count;
    wire [31:0] hls_query_position=dut.hls_query_position;
    wire [31:0] hls_layer=dut.hls_layer;
    wire [31:0] hls_query_head=dut.hls_query_head;
    wire [31:0] hls_kv_head=dut.hls_kv_head;
    wire [63:0] hls_sequence=dut.hls_sequence;
    wire [63:0] hls_epoch=dut.hls_epoch;
    wire hls_query_ce=dut.hls_query_ce;
    wire [5:0] hls_query_address=dut.hls_query_address;
    wire [31:0] hls_query_q=dut.hls_query_q;
    wire hls_state_in_ce=dut.hls_state_in_ce;
    wire [6:0] hls_state_in_address=dut.hls_state_in_address;
    wire [63:0] hls_state_in_q=dut.hls_state_in_q;
    wire hls_state_out_ce=dut.hls_state_out_ce;
    wire hls_state_out_we=dut.hls_state_out_we;
    wire [6:0] hls_state_out_address=dut.hls_state_out_address;
    wire [63:0] hls_state_out_d=dut.hls_state_out_d;
    wire hls_output_ce=dut.hls_output_ce;
    wire hls_output_we=dut.hls_output_we;
    wire [5:0] hls_output_address=dut.hls_output_address;
    wire [31:0] hls_output_d=dut.hls_output_d;
    wire hls_keys_ce=dut.hls_keys_ce;
    wire hls_values_ce=dut.hls_values_ce;
    wire [11:0] hls_keys_address=dut.hls_keys_address;
    wire [11:0] hls_values_address=dut.hls_values_address;
    wire [31:0] hls_keys_q=dut.hls_keys_q;
    wire [31:0] hls_values_q=dut.hls_values_q;
    wire hls_keys_q_valid=dut.hls_keys_q_valid;
    wire hls_values_q_valid=dut.hls_values_q_valid;
    always #2.5 clk=~clk;
    ntrpu_attention_engine_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
    wire  tile_valid=dut.memory_path.kv_path.tile_valid;
    wire  tile_ready=dut.memory_path.kv_path.tile_ready;
    wire [511:0] tile_data=dut.memory_path.kv_path.tile_data;
    wire  tile_plane=dut.memory_path.kv_path.tile_plane;
    wire  tile_last=dut.memory_path.kv_path.tile_last;
    wire [7:0] tile_index=dut.memory_path.kv_path.tile_index;
    wire [63:0] tile_sequence=dut.memory_path.kv_path.tile_sequence;
    wire [63:0] tile_epoch=dut.memory_path.kv_path.tile_epoch;
    wire [63:0] tile_lease_id=dut.memory_path.kv_path.tile_lease_id;
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
    logic [511:0] fixture_kv[0:47807];
    logic [31:0] fixture_q[0:9151],fixture_expected[0:9151];
    logic [159:0] fixture_command[0:142];
    logic [127:0] fixture_image[0:15];
    int current_image=0;
    string fixture_dir;
    function automatic logic [511:0] pattern(input int image,layer,pos,index);
        if(layer!=0 || current_image<1 || current_image>16 || pos<0 || pos>=cap || index<0 || index>=64)
            $fatal(1,"AHE_FAIL invalid fixture append index");
        return fixture_kv[int'(fixture_image[current_image-1][95:64])+pos*64+index];
    endfunction
    function automatic int slot(input logic [63:0] address);
        if(!(address>=BASE && address<BASE+64'd16777216 && address[5:0]==0))
            $fatal(1,"AHE_FAIL op=%0d: Physical memory range",op);
        return int'((address-BASE)>>6);
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"AHE_FAIL op=%0d: %s",op,why);
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
                $display("AHE_B cycle=%0d id=%0d ordinal=%0d resp=%0d",cycles,writer_op,write_b_count,m_axi_bresp);
                b_count++;write_b_count++;m_axi_bvalid<=0;aw_pending=0;w_last_seen=0;write_applied=0;w_pending=0;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 &&
                    m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW protocol");
                aw_address=m_axi_awaddr;aw_beats=int'(m_axi_awlen)+1;aw_pending=1;aw_count++;
                require(aw_beats<=4 && int'(aw_address[11:0])+aw_beats*64<=4096,"AW page");
                $display("AHE_AW cycle=%0d id=%0d address=%016h beats=%0d",cycles,writer_op,aw_address,aw_beats);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(w_pending<4 && !w_last_seen && m_axi_wstrb==64'hffffffffffffffff,"W protocol");
                pending_w[w_pending]=m_axi_wdata;w_pending++;w_count++;
                if(m_axi_wlast)w_last_seen=1;
                $display("AHE_W cycle=%0d id=%0d last=%0d data=%0128h",cycles,writer_op,m_axi_wlast,m_axi_wdata);
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
                $display("AHE_R cycle=%0d id=%0d address=%016h resp=%0d last=%0d data=%0128h",cycles,reader_op,ar_address+64'(ar_index*64),m_axi_rresp,m_axi_rlast,m_axi_rdata);
                r_count++;read_r_count++;ar_index++;m_axi_rvalid<=0;
                if(ar_index==ar_beats)ar_pending=0;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!ar_pending && !m_axi_rvalid && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 &&
                    m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR protocol");
                ar_address=m_axi_araddr;ar_beats=int'(m_axi_arlen)+1;ar_index=0;ar_pending=1;ar_count++;
                require(int'(ar_address[11:0])+ar_beats*64<=4096,"AR page");
                $display("AHE_AR cycle=%0d id=%0d address=%016h beats=%0d",cycles,reader_op,ar_address,ar_beats);
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
        $display("AHE_BIND cycle=%0d id=%0d generation=%0d capacity=%0d base=%016h sequence=%016h epoch=%016h",cycles,id,generation,cap,base_model,seq_model,epoch_model);
        do @(posedge clk);while(!bind_ready);
        @(negedge clk);bind_valid=0;bind_cache_base=0;bind_capacity=0;bind_sequence=0;bind_epoch=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Bind timeout");end
        require(append_done_bind && append_done_status==0 && cache_valid && cache_base==base_model && cache_capacity==32'(cap),"Bind result");
        require(append_done_sequence==seq_model && append_done_epoch==epoch_model,"Bind identity");
        append_done_ready=1;@(negedge clk);append_done_ready=0;
        for(int i=0;i<16;i++)prefix[i]=0;
        $display("AHE_BIND_DONE cycle=%0d id=%0d status=0",cycles,id);
    endtask
    task automatic append_token(input int layer,pos,mode,want_status);
        int id,index,ticks,old_count;
        op++;id=op;writer_op=id;append_count++;index=0;ticks=0;old_count=prefix[layer];write_error_mode=mode;write_b_count=0;
        @(negedge clk);append_layer=32'(layer);append_position=32'(pos);append_sequence=seq_model;append_epoch=epoch_model;append_valid=1;
        $display("AHE_APPEND cycle=%0d id=%0d generation=%0d layer=%0d position=%0d mode=%0d prefix=%0d",cycles,id,generation,layer,pos,mode,old_count);
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
        $display("AHE_APPEND_DONE cycle=%0d id=%0d status=%0d bytes=%0d committed=%0d",cycles,id,append_done_status,append_done_bytes,append_done_committed_tokens);
        repeat(4)@(negedge clk);append_done_ready=1;@(negedge clk);append_done_ready=0;write_error_mode=0;
    endtask
    int queries=0,completions=0,result_words=0,starts=0,finishes=0,state_writes=0,output_writes=0;
    int key_reads=0,value_reads=0,query_reads=0,state_reads=0;
    int fixture_case=0,mode=0,query_start=0,query_tiles=0;
    bit aborting=0,tile_running=0;
    logic [103:0] seen_state=0;
    logic [63:0] seen_output=0;
    function automatic logic [31:0] expected_kv(input int plane,address);
        int position,head,lane,word;
        logic [511:0] beat;
        position=int'(hls_tile_start)+address/64;
        head=int'(hls_query_head)/4;lane=address%64;word=lane/16;
        beat=fixture_kv[int'(fixture_image[current_image-1][95:64])+position*64+plane*32+head*4+word];
        return beat[(lane%16)*32+:32];
    endfunction
    function automatic logic [31:0] ordered(input logic [31:0] bits);
        return bits[31] ? ~bits:(bits|32'h80000000);
    endfunction
    function automatic logic [32:0] ulp_distance(input logic [31:0] a,b);
        logic [31:0] x,y;
        if((a&32'h7fffffff)==0 && (b&32'h7fffffff)==0)return 0;
        x=ordered(a);y=ordered(b);return x>y ? {1'b0,x}-{1'b0,y}:{1'b0,y}-{1'b0,x};
    endfunction
    task automatic tick;@(posedge clk);#1;endtask
    // Observe exactly the ports presented to the real HLS module. Read checks
    // sample the synchronous RAM response after the acceptance edge.
    always @(posedge clk)begin: arithmetic_trace
        bit qce,kce,vce,sce;
        int qa,ka,va,sa;
        if(!rst_n)begin tile_running=0;seen_state=0;seen_output=0;end
        else begin
            if(hls_start && !tile_running)begin
                require(lease_active && hls_tile_count>0 && hls_tile_count<=64,"HLS start without complete KV lease");
                tile_running=1;starts++;query_tiles++;seen_state=0;seen_output=0;
                $display("AHE_START cycle=%0d query=%0d fixture=%0d start=%0d count=%0d position=%0d layer=%0d head=%0d kv=%0d sequence=%016h epoch=%016h",cycles,queries,fixture_case,hls_tile_start,hls_tile_count,hls_query_position,hls_layer,hls_query_head,hls_kv_head,hls_sequence,hls_epoch);
            end
            if(tile_running)require(lease_active,"Lease released before real arithmetic drained");
            if(result_valid)require(completion_valid && !lease_active && !tile_running,"Result exposed before drain");
            if(!result_valid)require(!result_q_valid && result_q===0,"Private result exposed");
            if(hls_state_out_ce && hls_state_out_we)begin
                require(tile_running && hls_state_out_address<104,"Invalid real state write");
                require(!seen_state[hls_state_out_address],"Repeated real state word");
                seen_state[hls_state_out_address]=1;state_writes++;
                $display("AHE_STATE cycle=%0d query=%0d start=%0d index=%0d data=%016h",cycles,queries,hls_tile_start,hls_state_out_address,hls_state_out_d);
            end
            if(hls_output_ce && hls_output_we)begin
                require(tile_running && hls_tile_start+hls_tile_count==hls_query_position+1,"Early real output write");
                require(!seen_output[hls_output_address],"Repeated real output word");
                seen_output[hls_output_address]=1;output_writes++;
                $display("AHE_OUTPUT cycle=%0d query=%0d lane=%0d data=%08h",cycles,queries,hls_output_address,hls_output_d);
            end
            if(hls_done && tile_running)begin
                if(!aborting)begin
                    require(hls_return==0 && (&seen_state),"Real core result/state coverage");
                    if(hls_tile_start+hls_tile_count==hls_query_position+1)require(&seen_output,"Real core final coverage");
                    else require(seen_output==0,"Real core partial output");
                end
                require(hls_ready,"Non-pipelined real core completion handshake");
                $display("AHE_DONE cycle=%0d query=%0d start=%0d status=%0d aborting=%0d",cycles,queries,hls_tile_start,hls_return,aborting);
                finishes++;tile_running=0;
            end
            qce=hls_query_ce;kce=hls_keys_ce;vce=hls_values_ce;sce=hls_state_in_ce;
            qa=int'(hls_query_address);ka=int'(hls_keys_address);va=int'(hls_values_address);sa=int'(hls_state_in_address);
            if(qce || kce || vce || sce)require(tile_running,"HLS memory read outside transaction");
            #1;
            if(qce)begin
                require(hls_query_q===fixture_q[(fixture_case-1)*64+qa],"Real query RAM mismatch");query_reads++;
                $display("AHE_Q cycle=%0d query=%0d start=%0d lane=%0d data=%08h",cycles,queries,hls_tile_start,qa,hls_query_q);
            end
            if(kce)begin
                if(aborting)require(!hls_keys_q_valid && hls_keys_q===0,"Revoked real K leaked");
                else require(hls_keys_q_valid && ka<int'(hls_tile_count)*64 && hls_keys_q===expected_kv(0,ka),"Real K RAM mismatch");
                key_reads++;$display("AHE_K cycle=%0d query=%0d start=%0d index=%0d valid=%0d data=%08h",cycles,queries,hls_tile_start,ka,hls_keys_q_valid,hls_keys_q);
            end
            if(vce)begin
                if(aborting)require(!hls_values_q_valid && hls_values_q===0,"Revoked real V leaked");
                else require(hls_values_q_valid && va<int'(hls_tile_count)*64 && hls_values_q===expected_kv(1,va),"Real V RAM mismatch");
                value_reads++;$display("AHE_V cycle=%0d query=%0d start=%0d index=%0d valid=%0d data=%08h",cycles,queries,hls_tile_start,va,hls_values_q_valid,hls_values_q);
            end
            if(sce)begin
                require(hls_tile_start!=0 && sa<104,"Real continuation read range");state_reads++;
                $display("AHE_STATE_R cycle=%0d query=%0d start=%0d index=%0d data=%016h",cycles,queries,hls_tile_start,sa,hls_state_in_q);
            end
        end
    end
    always @(posedge clk)if(rst_n && dut.arithmetic.raw_state_in_ce && hls_tile_start==0)begin
        require(tile_running && dut.arithmetic.raw_state_in_address<104 && !hls_state_in_ce &&
            dut.arithmetic.guarded_state_in_q===64'd0,"Isolated initial state prefetch");
        $display("HLS_PREFETCH cycle=%0d query=%0d start=%0d index=%0d forwarded=%0d data=%016h",
            cycles,queries,hls_tile_start,dut.arithmetic.raw_state_in_address,hls_state_in_ce,dut.arithmetic.guarded_state_in_q);
    end
    always @(posedge clk)if(rst_n && dut.arithmetic.discarded_state)begin
        require(tile_running && hls_tile_start!=0 && dut.arithmetic.raw_state_in_address==104 &&
            !hls_state_in_ce,"Isolated continuation sentinel request");
        #1;require(dut.arithmetic.guarded_state_in_q===64'd0,"Isolated continuation sentinel response");
        $display("HLS_STATE_SENTINEL cycle=%0d query=%0d start=%0d index=104 forwarded=0 data=%016h",
            cycles,queries,hls_tile_start,dut.arithmetic.guarded_state_in_q);
    end
    always @(posedge clk)if(rst_n && dut.arithmetic.discarded_keys)begin
        require(tile_running && hls_tile_count<64 && dut.arithmetic.raw_keys_address==hls_tile_count*64 &&
            !hls_keys_ce,"Isolated KV sentinel request");
        #1;require(dut.arithmetic.guarded_keys_q===32'd0,"Isolated KV sentinel response");
        $display("HLS_KV_PREFETCH cycle=%0d query=%0d start=%0d plane=0 index=%0d forwarded=0 data=%08h",
            cycles,queries,hls_tile_start,hls_tile_count*64,dut.arithmetic.guarded_keys_q);
    end
    always @(posedge clk)if(rst_n && dut.arithmetic.discarded_values)begin
        require(tile_running && hls_tile_count<64 && dut.arithmetic.raw_values_address==hls_tile_count*64 &&
            !hls_values_ce,"Isolated KV sentinel request");
        #1;require(dut.arithmetic.guarded_values_q===32'd0,"Isolated KV sentinel response");
        $display("HLS_KV_PREFETCH cycle=%0d query=%0d start=%0d plane=1 index=%0d forwarded=0 data=%08h",
            cycles,queries,hls_tile_start,hls_tile_count*64,dut.arithmetic.guarded_values_q);
    end
    always @(posedge clk)if(rst_n)begin
        if(dut.memory_path.finish_valid && dut.memory_path.finish_ready)
            $display("AHE_FINISH cycle=%0d query=%0d start=%0d status=%0d",cycles,queries,dut.hls_tile_start,dut.memory_path.finish_status);
        if(dut.memory_path.query_done_valid && dut.memory_path.query_done_ready)
            $display("AHE_CERTIFICATE cycle=%0d query=%0d status=%0d tokens=%0d tiles=%0d lease=%016h",cycles,queries,dut.memory_path.query_done_status,dut.memory_path.query_done_tokens,dut.memory_path.query_done_tiles,dut.memory_path.query_done_lease);
        if(dut.memory_path.kv_path.lease_cmd_valid && dut.memory_path.kv_path.lease_cmd_ready)
            $display("AHE_LEASE cycle=%0d query=%0d release=%0d lease=%016h",cycles,queries,dut.memory_path.kv_path.lease_cmd_release,dut.memory_path.kv_path.lease_cmd_id);
        if(dut.memory_path.kv_path.lease_done_valid && dut.memory_path.kv_path.lease_done_ready)
            $display("AHE_LEASE_DONE cycle=%0d query=%0d release=%0d status=%0d active=%0d",cycles,queries,dut.memory_path.kv_path.lease_done_release,dut.memory_path.kv_path.lease_done_status,lease_active);
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;
        reset_count++;$display("AHE_RESET cycle=%0d count=%0d query=%0d",cycles,reset_count,queries);
        bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;append_done_ready=0;
        command_valid=0;cancel_valid=0;completion_ready=0;result_ce=0;aborting=0;
        freeze_ar=0;freeze_r=0;freeze_aw=0;freeze_w=0;freeze_b=0;read_error_at=-1;write_error_mode=0;
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(command_ready && !completion_valid && !lease_active && !result_valid && !hls_start,"Coordinated HLS/context reset");
        for(int i=0;i<16;i++)prefix[i]=0;
    endtask
    task automatic select_image(input int image);
        int capacity,offset;
        current_image=image;require(image>=1 && image<=16,"Image range");
        require(fixture_image[image-1][31:0]==32'(image),"Image descriptor identity");
        capacity=int'(fixture_image[image-1][63:32]);offset=int'(fixture_image[image-1][127:96]);
        bind_cache(capacity,offset);
        for(int position=0;position<capacity;position++)append_token(0,position,0,0);
    endtask
    task automatic start_query(input int ident,requested_mode);
        fixture_case=ident;mode=requested_mode;queries++;reader_op=queries;query_tiles=0;aborting=0;
        query_start=cycles;require(ident>=1 && ident<=143,"Case range");
        require(fixture_command[ident-1][31:0]==32'(ident) && fixture_command[ident-1][63:32]==32'(current_image),"Command image");
        @(negedge clk);require(command_ready,"Root command busy");command_valid=1;
        command_position=fixture_command[ident-1][95:64];command_head=fixture_command[ident-1][127:96];command_layer=fixture_command[ident-1][159:128];
        command_sequence=seq_model;command_epoch=epoch_model;command_lease=next_lease;next_lease++;
        for(int lane=0;lane<64;lane++)command_query[lane*32+:32]=fixture_q[(ident-1)*64+lane];
        $display("AHE_QUERY cycle=%0d query=%0d fixture=%0d image=%0d generation=%0d layer=%0d head=%0d position=%0d mode=%0d sequence=%016h epoch=%016h lease=%016h",cycles,queries,ident,current_image,generation,command_layer,command_head,command_position,requested_mode,command_sequence,command_epoch,command_lease);
        tick();@(negedge clk);command_valid=0;command_query='1;command_sequence=0;command_lease=0;
    endtask
    task automatic collect(input int wanted,input bit late_revoke);
        int ticks;logic [3:0] saved;
        ticks=0;
        while(!completion_valid)begin @(negedge clk);ticks++;require(ticks<5000000,"Real core completion timeout");end
        saved=completion_status;require(saved==4'(wanted) && result_valid==(wanted==0),"Real completion status/authorization");
        require(!lease_active && !tile_running && !ar_pending && !m_axi_rvalid,"Completion before real drain");
        repeat(7)begin tick();require(completion_valid && completion_status==saved && !command_ready,"Held real completion changed");end
        if(wanted==0)for(int lane=0;lane<64;lane++)begin
            @(negedge clk);result_ce=1;result_address=6'(lane);tick();
            require(result_q_valid && result_q[30:23]!=8'hff && ulp_distance(result_q,fixture_expected[(fixture_case-1)*64+lane])<=1,"Real attention output mismatch");
            result_words++;$display("AHE_RESULT cycle=%0d query=%0d fixture=%0d lane=%0d data=%08h",cycles,queries,fixture_case,lane,result_q);
        end
        if(late_revoke)begin
            @(negedge clk);expected_epoch=epoch_model+1;result_ce=1;#1;
            require(!result_valid && !result_q_valid && result_q===0,"Late real epoch leak");
            $display("AHE_REVOKE cycle=%0d query=%0d epoch=%016h valid=%0d q_valid=%0d data=%08h",cycles,queries,expected_epoch,result_valid,result_q_valid,result_q);
            tick();@(negedge clk);expected_epoch=epoch_model;tick();
            require(!result_valid && !result_q_valid && completion_status==saved,"Real late revocation not sticky");
            $display("AHE_REVOKED cycle=%0d query=%0d epoch=%016h valid=%0d q_valid=%0d data=%08h",cycles,queries,expected_epoch,result_valid,result_q_valid,result_q);
        end
        $display("AHE_COMPLETE cycle=%0d query=%0d fixture=%0d status=%0d tiles=%0d elapsed=%0d revoke=%0d",cycles,queries,fixture_case,saved,query_tiles,cycles-query_start,late_revoke);
        completions++;@(negedge clk);completion_ready=1;result_ce=0;tick();@(negedge clk);completion_ready=0;
        expected_epoch=epoch_model;aborting=0;
    endtask
    task automatic inject;
        if(mode==1 || mode==2)begin
            wait(hls_keys_ce);@(negedge clk);aborting=1;
            if(mode==1)begin require(cancel_ready,"Real cancel not admitted");cancel_valid=1;end
            else expected_epoch=epoch_model+1;
            $display("AHE_CANCEL cycle=%0d query=%0d revoke=%0d",cycles,queries,mode==2);
            tick();@(negedge clk);cancel_valid=0;
        end
    endtask
    task automatic run_query(input int ident,requested_mode,wanted);
        start_query(ident,requested_mode);
        fork collect(wanted,requested_mode==3);inject();join
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        require($value$plusargs("FIXTURE=%s",fixture_dir),"Missing immutable fixture directory");
        $readmemh({fixture_dir,"/kv_beats.hex"},fixture_kv);
        $readmemh({fixture_dir,"/query.hex"},fixture_q);
        $readmemh({fixture_dir,"/expected.hex"},fixture_expected);
        $readmemh({fixture_dir,"/commands.hex"},fixture_command);
        $readmemh({fixture_dir,"/images.hex"},fixture_image);
        reset_all();
        for(int ident=1;ident<=143;ident++)begin
            if(current_image!=int'(fixture_command[ident-1][63:32]))select_image(int'(fixture_command[ident-1][63:32]));
            run_query(ident,0,0);
        end
        for(int fault_mode=1;fault_mode<=2;fault_mode++)begin
            run_query(143,fault_mode,fault_mode==1 ? 7:2);run_query(143,0,0);
        end
        run_query(143,3,0);run_query(143,0,0);
        for(int phase=0;phase<3;phase++)begin
            start_query(143,0);
            if(phase==0)begin freeze_r=1;wait(ar_pending);end
            else if(phase==1)wait(tile_running);
            else wait(completion_valid);
            @(negedge clk);$display("AHE_QUERY_RESET cycle=%0d query=%0d phase=%0d",cycles,queries,phase);
            reset_all();select_image(16);run_query(143,0,0);
        end
        require(queries==155 && completions==152 && result_words==9600 && reset_count==4 && append_count==756 && bind_count==19,"Real bank scenario coverage");
        $display("AHE_PASS seed=%0d watchdog=%0d cases=143 queries=%0d completions=%0d results=%0d resets=%0d binds=%0d appends=%0d starts=%0d dones=%0d q=%0d k=%0d v=%0d state_r=%0d state_w=%0d output_w=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d cycles=%0d",seed,WATCHDOG_CYCLES,queries,completions,result_words,reset_count,bind_count,append_count,starts,finishes,query_reads,key_reads,value_reads,state_reads,state_writes,output_writes,aw_count,w_count,b_count,ar_count,r_count,cycles);
        $finish;
    end
    initial begin #2000000000;$fatal(1,"AHE_FAIL global timeout");end
endmodule
`default_nettype wire
