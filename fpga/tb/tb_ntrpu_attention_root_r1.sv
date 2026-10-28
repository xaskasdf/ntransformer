`timescale 1ns/1ps
`default_nettype none
module tb_ntrpu_attention_root_r1 #(parameter integer WATCHDOG_CYCLES=65536);
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
    logic [10:0] result_address=0;
    logic [31:0] result_q;
    logic result_q_valid;
    logic [63:0] command_lease_base=0;
    logic q_valid=0,q_ready,q_last=0;
    logic [511:0] q_data=0;
    logic [7:0] q_index=0;
    logic [63:0] q_sequence=0,q_epoch=0;
    logic [63:0] completion_sequence,completion_epoch;
    logic cache_lock,maintenance_busy,metadata_fault;
    logic [31:0] committed_tokens;
    wire hls_start=dut.engine.hls_start;
    wire hls_ready=dut.engine.hls_ready;
    wire hls_done=dut.engine.hls_done;
    wire [31:0] hls_return=dut.engine.hls_return;
    wire [31:0] hls_tile_start=dut.engine.hls_tile_start;
    wire [31:0] hls_tile_count=dut.engine.hls_tile_count;
    wire [31:0] hls_query_position=dut.engine.hls_query_position;
    wire [31:0] hls_layer=dut.engine.hls_layer;
    wire [31:0] hls_query_head=dut.engine.hls_query_head;
    wire [31:0] hls_kv_head=dut.engine.hls_kv_head;
    wire [63:0] hls_sequence=dut.engine.hls_sequence;
    wire [63:0] hls_epoch=dut.engine.hls_epoch;
    wire hls_query_ce=dut.engine.hls_query_ce;
    wire [5:0] hls_query_address=dut.engine.hls_query_address;
    wire [31:0] hls_query_q=dut.engine.hls_query_q;
    wire hls_state_in_ce=dut.engine.hls_state_in_ce;
    wire [6:0] hls_state_in_address=dut.engine.hls_state_in_address;
    wire [63:0] hls_state_in_q=dut.engine.hls_state_in_q;
    wire hls_state_out_ce=dut.engine.hls_state_out_ce;
    wire hls_state_out_we=dut.engine.hls_state_out_we;
    wire [6:0] hls_state_out_address=dut.engine.hls_state_out_address;
    wire [63:0] hls_state_out_d=dut.engine.hls_state_out_d;
    wire hls_output_ce=dut.engine.hls_output_ce;
    wire hls_output_we=dut.engine.hls_output_we;
    wire [5:0] hls_output_address=dut.engine.hls_output_address;
    wire [31:0] hls_output_d=dut.engine.hls_output_d;
    wire hls_keys_ce=dut.engine.hls_keys_ce;
    wire hls_values_ce=dut.engine.hls_values_ce;
    wire [11:0] hls_keys_address=dut.engine.hls_keys_address;
    wire [11:0] hls_values_address=dut.engine.hls_values_address;
    wire [31:0] hls_keys_q=dut.engine.hls_keys_q;
    wire [31:0] hls_values_q=dut.engine.hls_values_q;
    wire hls_keys_q_valid=dut.engine.hls_keys_q_valid;
    wire hls_values_q_valid=dut.engine.hls_values_q_valid;
    always #2.5 clk=~clk;
    ntrpu_attention_root_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
    wire  tile_valid=dut.engine.memory_path.kv_path.tile_valid;
    wire  tile_ready=dut.engine.memory_path.kv_path.tile_ready;
    wire [511:0] tile_data=dut.engine.memory_path.kv_path.tile_data;
    wire  tile_plane=dut.engine.memory_path.kv_path.tile_plane;
    wire  tile_last=dut.engine.memory_path.kv_path.tile_last;
    wire [7:0] tile_index=dut.engine.memory_path.kv_path.tile_index;
    wire [63:0] tile_sequence=dut.engine.memory_path.kv_path.tile_sequence;
    wire [63:0] tile_epoch=dut.engine.memory_path.kv_path.tile_epoch;
    wire [63:0] tile_lease_id=dut.engine.memory_path.kv_path.tile_lease_id;
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
            $fatal(1,"AROOT_FAIL invalid fixture append index");
        return fixture_kv[int'(fixture_image[current_image-1][95:64])+pos*64+index];
    endfunction
    function automatic int slot(input logic [63:0] address);
        if(!(address>=BASE && address<BASE+64'd16777216 && address[5:0]==0))
            $fatal(1,"AROOT_FAIL op=%0d: Physical memory range",op);
        return int'((address-BASE)>>6);
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"AROOT_FAIL op=%0d: %s",op,why);
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
                $display("AROOT_B cycle=%0d id=%0d ordinal=%0d resp=%0d",cycles,writer_op,write_b_count,m_axi_bresp);
                b_count++;write_b_count++;m_axi_bvalid<=0;aw_pending=0;w_last_seen=0;write_applied=0;w_pending=0;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 &&
                    m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW protocol");
                aw_address=m_axi_awaddr;aw_beats=int'(m_axi_awlen)+1;aw_pending=1;aw_count++;
                require(aw_beats<=4 && int'(aw_address[11:0])+aw_beats*64<=4096,"AW page");
                $display("AROOT_AW cycle=%0d id=%0d address=%016h beats=%0d",cycles,writer_op,aw_address,aw_beats);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(w_pending<4 && !w_last_seen && m_axi_wstrb==64'hffffffffffffffff,"W protocol");
                pending_w[w_pending]=m_axi_wdata;w_pending++;w_count++;
                if(m_axi_wlast)w_last_seen=1;
                $display("AROOT_W cycle=%0d id=%0d last=%0d data=%0128h",cycles,writer_op,m_axi_wlast,m_axi_wdata);
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
                $display("AROOT_R cycle=%0d id=%0d address=%016h resp=%0d last=%0d data=%0128h",cycles,reader_op,ar_address+64'(ar_index*64),m_axi_rresp,m_axi_rlast,m_axi_rdata);
                r_count++;read_r_count++;ar_index++;m_axi_rvalid<=0;
                if(ar_index==ar_beats)ar_pending=0;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!ar_pending && !m_axi_rvalid && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 &&
                    m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR protocol");
                ar_address=m_axi_araddr;ar_beats=int'(m_axi_arlen)+1;ar_index=0;ar_pending=1;ar_count++;
                require(int'(ar_address[11:0])+ar_beats*64<=4096,"AR page");
                $display("AROOT_AR cycle=%0d id=%0d address=%016h beats=%0d",cycles,reader_op,ar_address,ar_beats);
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
        command_layer=0;command_valid=1;
        $display("AROOT_BIND cycle=%0d id=%0d generation=%0d capacity=%0d base=%016h sequence=%016h epoch=%016h",cycles,id,generation,cap,base_model,seq_model,epoch_model);
        do @(posedge clk);while(!bind_ready);
        @(negedge clk);bind_valid=0;bind_cache_base=0;bind_capacity=0;bind_sequence=0;bind_epoch=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Bind timeout");end
        require(append_done_bind && append_done_status==0 && dut.engine.cache_valid && cache_base==base_model && cache_capacity==32'(cap),"Bind result");
        require(append_done_sequence==seq_model && append_done_epoch==epoch_model,"Bind identity");
        require(!command_ready && maintenance_busy,"Root admitted during bind");
        command_valid=0;append_done_ready=1;@(negedge clk);append_done_ready=0;
        require(cache_valid && !maintenance_busy && !metadata_fault,"Trusted bind metadata not published");
        for(int i=0;i<16;i++)prefix[i]=0;
        $display("AROOT_BIND_DONE cycle=%0d id=%0d status=0",cycles,id);
    endtask
    task automatic append_token(input int layer,pos,mode,want_status);
        int id,index,ticks,old_count;
        op++;id=op;writer_op=id;append_count++;index=0;ticks=0;old_count=prefix[layer];write_error_mode=mode;write_b_count=0;
        @(negedge clk);append_layer=32'(layer);append_position=32'(pos);append_sequence=seq_model;append_epoch=epoch_model;append_valid=1;command_valid=1;
        $display("AROOT_APPEND cycle=%0d id=%0d generation=%0d layer=%0d position=%0d mode=%0d prefix=%0d",cycles,id,generation,layer,pos,mode,old_count);
        do @(posedge clk);while(!append_ready);
        @(negedge clk);append_valid=0;append_layer=0;append_position=0;append_sequence=0;append_epoch=0;
        while(!append_prepared && !append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Append admission timeout");end
        while(index<64 && append_prepared)begin
            data_valid=$urandom_range(0,3)!=0;data_bits=pattern(generation,layer,pos,index);data_index=6'(index);data_last=index==63;
            data_sequence=seq_model;data_epoch=epoch_model;commit_valid=data_valid && index==63;commit_status=0;commit_sequence=seq_model;commit_epoch=epoch_model;
            @(posedge clk);if(data_valid && data_ready)index++;
            @(negedge clk);
        end
        data_valid=0;commit_valid=0;require(!command_ready && maintenance_busy,"Root admitted during append");command_valid=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<4000,"Append completion timeout");end
        require(!append_done_bind && append_done_status==4'(want_status),"Append completion status");
        require(append_done_sequence==seq_model && append_done_epoch==epoch_model,"Append completion identity");
        if(want_status==0)begin require(append_done_bytes==4096,"Complete physical token");prefix[layer]=pos+1;end
        require(append_done_committed_tokens==32'(prefix[layer]),"Owned append counter");
        $display("AROOT_APPEND_DONE cycle=%0d id=%0d status=%0d bytes=%0d committed=%0d",cycles,id,append_done_status,append_done_bytes,append_done_committed_tokens);
        repeat(4)@(negedge clk);append_done_ready=1;@(negedge clk);append_done_ready=0;write_error_mode=0;
        require(committed_tokens==32'(prefix[0]),"Trusted committed prefix differs from cache");
    endtask
    logic [31:0] root_expected[0:8191],observed_child[0:2047];
    int roots=0,root_sample=0,root_mode=0,root_start=0,root_children=0,root_q_beats=0;
    int children=0,releases=0,completions=0,result_words=0,child_reads=0;
    int current_head=0,current_fixture=0,current_child_reads=0,hls_starts=0,hls_dones=0;
    int state_writes=0,output_writes=0,q_reads=0,k_reads=0,v_reads=0;
    int root_held_checks=0;
    bit child_owned=0,tile_running=0;
    logic [103:0] seen_state=0;
    logic [63:0] seen_output=0;
    logic [63:0] root_lease=0;
    task automatic tick;@(posedge clk);#1;endtask
    function automatic logic [31:0] ordered(input logic [31:0] bits);
        return bits[31] ? ~bits:(bits|32'h80000000);
    endfunction
    function automatic logic [32:0] ulp_distance(input logic [31:0] a,b);
        logic [31:0] x,y;
        if((a&32'h7fffffff)==0 && (b&32'h7fffffff)==0)return 0;
        x=ordered(a);y=ordered(b);return x>y ? {1'b0,x}-{1'b0,y}:{1'b0,y}-{1'b0,x};
    endfunction
    function automatic logic [31:0] expected_kv(input int plane,address);
        int position,head,lane;logic [511:0] beat;
        position=address/64;head=current_head/4;lane=address%64;
        beat=fixture_kv[position*64+plane*32+head*4+lane/16];
        return beat[(lane%16)*32+:32];
    endfunction

    always @(posedge clk)begin: observe_root
        bit read_ce,qce,kce,vce;
        int address,qa,ka,va;
        if(!rst_n)begin child_owned=0;tile_running=0;seen_state=0;seen_output=0;end
        else begin
            if(cache_lock)begin
                require(!bind_ready && !append_ready && !dut.engine_bind_valid && !dut.engine_append_valid,
                    "Cache maintenance escaped root lock");
                require(!m_axi_awvalid && !m_axi_wvalid && !m_axi_bvalid,"Write during root ownership");
                if(bind_valid && append_valid)root_held_checks++;
            end
            if(result_valid)require(completion_valid && cache_lock && !child_owned && !lease_active,
                "Root publication before all child ownership released");
            if(!result_valid)require(!result_q_valid && result_q===0,"Private root output leaked");
            if(dut.head_command_valid && dut.head_command_ready)begin
                require(cache_lock && !maintenance_busy && cache_valid && !child_owned && root_q_beats==128,
                    "Child without complete Q and cache ownership");
                require(dut.head_command_head==32'(root_children) && root_children<32 &&
                    dut.head_command_layer==0 && dut.head_command_position==32'(root_sample-1) &&
                    dut.head_command_sequence==seq_model && dut.head_command_epoch==epoch_model &&
                    dut.head_command_lease==root_lease+64'(root_children),"Child descriptor/lease");
                current_head=root_children;current_fixture=(root_sample-1)*32+current_head+1;
                for(int lane=0;lane<64;lane++)require(dut.head_command_query[lane*32+:32]===
                    fixture_q[(current_fixture-1)*64+lane],"Child Q vector/order");
                root_children++;children++;child_owned=1;current_child_reads=0;reader_op=children;read_r_count=0;
                if(root_mode==7 && current_head==1)read_error_at=1;else read_error_at=-1;
                $display("AROOT_CHILD cycle=%0d root=%0d child=%0d head=%0d fixture=%0d position=%0d sequence=%016h epoch=%016h lease=%016h",
                    cycles,roots,children,current_head,current_fixture,root_sample-1,seq_model,epoch_model,dut.head_command_lease);
            end
            if(dut.head_completion_valid && dut.head_completion_ready)begin
                require(child_owned && !lease_active && !tile_running && !ar_pending && !m_axi_rvalid,"Child released before drain");
                if(dut.head_completion_status==0 && root_mode!=6)
                    require(current_child_reads==64,"Successful child not completely copied");
                $display("AROOT_CHILD_DONE cycle=%0d root=%0d child=%0d status=%0d reads=%0d",
                    cycles,roots,children,dut.head_completion_status,current_child_reads);
                releases++;child_owned=0;
            end
            if(hls_start && !tile_running)begin
                require(child_owned && lease_active && hls_tile_start==0 && hls_tile_count==32'(root_sample),"HLS tile ownership");
                tile_running=1;hls_starts++;seen_state=0;seen_output=0;
                $display("AROOT_HLS_START cycle=%0d root=%0d child=%0d fixture=%0d count=%0d head=%0d sequence=%016h epoch=%016h",
                    cycles,roots,children,current_fixture,hls_tile_count,hls_query_head,hls_sequence,hls_epoch);
            end
            if(hls_state_out_ce && hls_state_out_we)begin
                require(tile_running && hls_state_out_address<104 && !seen_state[hls_state_out_address],"HLS state range/duplicate");
                seen_state[hls_state_out_address]=1;state_writes++;
                $display("AROOT_STATE cycle=%0d child=%0d index=%0d data=%016h",cycles,children,hls_state_out_address,hls_state_out_d);
            end
            if(hls_output_ce && hls_output_we)begin
                require(tile_running && !seen_output[hls_output_address],"HLS output ownership/duplicate");
                seen_output[hls_output_address]=1;output_writes++;
                $display("AROOT_OUTPUT cycle=%0d child=%0d lane=%0d data=%08h",cycles,children,hls_output_address,hls_output_d);
            end
            if(hls_done && tile_running)begin
                if(!dut.engine.memory_path.compute_abort)require(hls_return==0 && (&seen_state) && (&seen_output),"HLS coverage/status");
                require(hls_ready,"Real HLS completion handshake");
                $display("AROOT_HLS_DONE cycle=%0d child=%0d status=%0d aborting=%0d",cycles,children,hls_return,dut.engine.memory_path.compute_abort);
                hls_dones++;tile_running=0;
            end
            read_ce=dut.head_result_ce;address=int'(dut.head_result_address);
            qce=hls_query_ce;kce=hls_keys_ce;vce=hls_values_ce;
            qa=int'(hls_query_address);ka=int'(hls_keys_address);va=int'(hls_values_address);
            if(read_ce)require(child_owned && dut.head_completion_valid && dut.head_result_valid &&
                address==current_child_reads,"Child read handshake/order");
            #1;
            if(qce)begin
                require(hls_query_q===fixture_q[(current_fixture-1)*64+qa],"HLS Q content");q_reads++;
                $display("AROOT_Q cycle=%0d child=%0d lane=%0d data=%08h",cycles,children,qa,hls_query_q);
            end
            if(kce)begin
                require(ka<root_sample*64 && (hls_keys_q_valid ? hls_keys_q===expected_kv(0,ka):hls_keys_q===0),"HLS K content");
                require(hls_keys_q_valid || root_mode==5,"Unexpected K revocation");k_reads++;
                $display("AROOT_K cycle=%0d child=%0d index=%0d valid=%0d data=%08h",cycles,children,ka,hls_keys_q_valid,hls_keys_q);
            end
            if(vce)begin
                require(va<root_sample*64 && (hls_values_q_valid ? hls_values_q===expected_kv(1,va):hls_values_q===0),"HLS V content");
                require(hls_values_q_valid || root_mode==5,"Unexpected V revocation");v_reads++;
                $display("AROOT_V cycle=%0d child=%0d index=%0d valid=%0d data=%08h",cycles,children,va,hls_values_q_valid,hls_values_q);
            end
            if(read_ce)begin
                require(dut.head_result_q_valid && ulp_distance(dut.head_result_q,fixture_expected[(current_fixture-1)*64+address])<=1,
                    "Child read numerical mismatch");
                observed_child[current_head*64+address]=dut.head_result_q;current_child_reads++;child_reads++;
                $display("AROOT_COPY cycle=%0d root=%0d child=%0d index=%0d data=%08h",cycles,roots,children,address,dut.head_result_q);
            end
        end
    end
    task automatic reset_all;
        @(negedge clk);rst_n=0;reset_count++;
        $display("AROOT_RESET cycle=%0d count=%0d root=%0d",cycles,reset_count,roots);
        bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;append_done_ready=0;
        command_valid=0;cancel_valid=0;completion_ready=0;result_ce=0;q_valid=0;
        freeze_ar=0;freeze_r=0;freeze_aw=0;freeze_w=0;freeze_b=0;read_error_at=-1;write_error_mode=0;
        repeat(5)tick();@(negedge clk);rst_n=1;repeat(3)tick();
        require(command_ready && !completion_valid && !lease_active && !result_valid && !hls_start &&
            !maintenance_busy && !cache_lock && !cache_valid,"Coordinated root reset");
        for(int i=0;i<16;i++)prefix[i]=0;
    endtask
    task automatic select_image;
        current_image=1;bind_cache(4,4032);
        for(int position=0;position<4;position++)append_token(0,position,0,0);
    endtask
    task automatic start_root(input int sample,requested_mode);
        roots++;root_sample=sample;root_mode=requested_mode;root_start=cycles;root_children=0;root_q_beats=0;
        root_lease=64'h1122334400000000+64'(roots*64);
        @(negedge clk);require(command_ready && !metadata_fault,"Root admission busy/poisoned");
        command_valid=1;command_layer=requested_mode==1 ? 1:0;command_position=32'(sample-1);
        command_sequence=seq_model;command_epoch=epoch_model;command_lease_base=root_lease;
        $display("AROOT_BEGIN cycle=%0d root=%0d sample=%0d mode=%0d generation=%0d sequence=%016h epoch=%016h lease=%016h",
            cycles,roots,sample,requested_mode,generation,seq_model,epoch_model,root_lease);
        tick();@(negedge clk);command_valid=0;command_sequence=0;command_lease_base=0;
        // Both requests remain asserted until the root completion is consumed.
        // Neither may reach the actual engine, even between its child leases.
        bind_valid=1;append_valid=1;
        if(requested_mode!=1)begin
            for(int beat=0;beat<128;beat++)begin
                repeat($urandom_range(0,2))@(negedge clk);
                q_valid=1;q_index=8'(requested_mode==4 && beat==2 ? 1:beat);q_last=beat==127;
                q_sequence=requested_mode==3 ? seq_model+1:seq_model;q_epoch=epoch_model;
                for(int lane=0;lane<16;lane++)q_data[lane*32+:32]=fixture_q[(sample-1)*2048+beat*16+lane];
                require(q_ready,"Q unexpectedly blocked");
                @(posedge clk);if(q_valid && q_ready)root_q_beats++;
                @(negedge clk);q_valid=0;
                if(requested_mode==3 || (requested_mode==4 && beat==2) ||
                   ((requested_mode==2 || requested_mode==9) && beat==7))break;
            end
        end
        q_valid=0;
    endtask
    task automatic collect(input int wanted);
        int ticks;logic [3:0] saved;
        ticks=0;
        while(!completion_valid)begin @(negedge clk);ticks++;require(ticks<5000000,"Root timeout");end
        saved=completion_status;
        require(saved==4'(wanted) && result_valid==(wanted==0) && cache_lock &&
            !child_owned && !lease_active && !ar_pending && !m_axi_rvalid,"Root completion/drain");
        require(completion_sequence==seq_model && completion_epoch==epoch_model,"Root completion identity");
        repeat(7)begin tick();require(completion_valid && completion_status==saved && !command_ready,"Held root completion");end
        if(wanted==0)for(int index=0;index<2048;index++)begin
            @(negedge clk);result_ce=1;result_address=11'(index);tick();
            require(result_q_valid && result_q===observed_child[index] &&
                ulp_distance(result_q,root_expected[(root_sample-1)*2048+index])<=1,"Root concatenated output mismatch");
            result_words++;
            $display("AROOT_RESULT cycle=%0d root=%0d index=%0d data=%08h",cycles,roots,index,result_q);
            if(index%7==0)begin @(negedge clk);result_ce=0;tick();end
        end
        if(root_mode==8)begin
            @(negedge clk);expected_epoch=epoch_model+1;result_ce=1;#1;
            require(!result_valid && !result_q_valid && result_q===0,"Late root epoch leak");
            tick();@(negedge clk);expected_epoch=epoch_model;tick();
            require(!result_valid && !result_q_valid && completion_status==saved && !cache_valid,"Root rollback revived result/cache");
            $display("AROOT_REVOKED cycle=%0d root=%0d status=%0d",cycles,roots,completion_status);
        end
        $display("AROOT_COMPLETE cycle=%0d root=%0d status=%0d children=%0d elapsed=%0d",cycles,roots,saved,root_children,cycles-root_start);
        completions++;@(negedge clk);bind_valid=0;append_valid=0;completion_ready=1;result_ce=0;
        tick();@(negedge clk);completion_ready=0;read_error_at=-1;
        require(!cache_lock && !completion_valid && command_ready,"Root ownership not released");
    endtask
    task automatic run_root(input int sample,requested_mode,wanted);
        start_root(sample,requested_mode);
        if(requested_mode==2)begin cancel_valid=1;tick();@(negedge clk);cancel_valid=0;end
        else if(requested_mode==5)begin
            wait(current_head==1 && hls_keys_ce);@(negedge clk);cancel_valid=1;
            $display("AROOT_CANCEL cycle=%0d root=%0d",cycles,roots);
            tick();@(negedge clk);cancel_valid=0;
        end else if(requested_mode==6)begin
            wait(current_head==1 && current_child_reads==8);@(negedge clk);expected_epoch=epoch_model+1;
            $display("AROOT_EPOCH cycle=%0d root=%0d",cycles,roots);
        end
        if(requested_mode>=9)begin
            if(requested_mode==10)begin wait(current_head==1 && ar_pending);@(negedge clk);freeze_r=1;end
            else if(requested_mode==11)wait(current_head==1 && current_child_reads==8);
            else if(requested_mode==12)wait(completion_valid);
            $display("AROOT_RESET_ABORT cycle=%0d root=%0d mode=%0d children=%0d",cycles,roots,requested_mode,root_children);
            reset_all();
        end else collect(wanted);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end random_state=seed;random_state=$urandom(random_state);
        require($value$plusargs("FIXTURE=%s",fixture_dir),"Missing immutable fixture");
        $readmemh({fixture_dir,"/kv_beats.hex"},fixture_kv);$readmemh({fixture_dir,"/query.hex"},fixture_q);
        $readmemh({fixture_dir,"/expected.hex"},fixture_expected);$readmemh({fixture_dir,"/root-expected.hex"},root_expected);
        $readmemh({fixture_dir,"/images.hex"},fixture_image);
        reset_all();select_image();
        for(int sample=1;sample<=4;sample++)run_root(sample,0,0);
        run_root(1,1,1);run_root(1,2,7);run_root(1,3,2);run_root(1,4,4);
        run_root(1,5,7);run_root(1,0,0);
        run_root(2,6,2);select_image();run_root(2,0,0);
        run_root(3,7,3);run_root(3,0,0);
        run_root(4,8,0);select_image();
        run_root(1,9,0);select_image();run_root(1,0,0);
        run_root(2,10,0);select_image();run_root(2,0,0);
        run_root(3,11,0);select_image();run_root(3,0,0);
        run_root(4,12,0);select_image();run_root(4,0,0);
        require(roots==23 && completions==19 && result_words==24576 && children==426 && releases==424 &&
            reset_count==5 && bind_count==7 && append_count==28 && root_held_checks>0,"Root scenario coverage");
        $display("AROOT_PASS seed=%0d watchdog=%0d roots=%0d completions=%0d results=%0d children=%0d releases=%0d resets=%0d binds=%0d appends=%0d starts=%0d dones=%0d copies=%0d q=%0d k=%0d v=%0d state_w=%0d output_w=%0d held=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d cycles=%0d",
            seed,WATCHDOG_CYCLES,roots,completions,result_words,children,releases,reset_count,bind_count,append_count,hls_starts,hls_dones,child_reads,q_reads,k_reads,v_reads,state_writes,output_writes,root_held_checks,aw_count,w_count,b_count,ar_count,r_count,cycles);
        $finish;
    end
    initial begin #2000000000;$fatal(1,"AROOT_FAIL global timeout");end
endmodule
`default_nettype wire
