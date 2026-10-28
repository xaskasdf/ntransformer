`timescale 1ns/1ps
module tb_ntrpu_kv_cache_r1 #(parameter integer WATCHDOG_CYCLES=32);
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
    logic lease_cmd_valid=0;
    logic lease_cmd_ready;
    logic lease_cmd_release=0;
    logic [31:0] lease_cmd_layer=0;
    logic [31:0] lease_cmd_query_head=0;
    logic [31:0] lease_cmd_position=0;
    logic [63:0] lease_cmd_sequence=0;
    logic [63:0] lease_cmd_epoch=0;
    logic [63:0] lease_cmd_id=0;
    logic lease_active;
    logic lease_done_valid;
    logic lease_done_ready=0;
    logic lease_done_release;
    logic [3:0] lease_done_status;
    logic [31:0] lease_done_prefix;
    logic [63:0] lease_done_sequence;
    logic [63:0] lease_done_epoch;
    logic [63:0] lease_done_id;
    logic read_valid=0;
    logic read_ready;
    logic [31:0] read_start=0;
    logic [31:0] read_tokens=0;
    logic [63:0] read_sequence=0;
    logic [63:0] read_epoch=0;
    logic [63:0] read_lease_id=0;
    logic read_cancel_valid=0;
    logic read_cancel_ready;
    logic tile_valid;
    logic tile_ready=0;
    logic [511:0] tile_data;
    logic tile_plane;
    logic tile_last;
    logic [7:0] tile_index;
    logic [63:0] tile_sequence;
    logic [63:0] tile_epoch;
    logic [63:0] tile_lease_id;
    logic read_done_valid;
    logic read_done_ready=0;
    logic [3:0] read_done_status;
    logic [9:0] read_done_beats;
    logic [15:0] read_done_bytes;
    logic [63:0] read_done_sequence;
    logic [63:0] read_done_epoch;
    logic [63:0] read_done_lease_id;
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
    always #2.5 clk=~clk;
    ntrpu_kv_cache_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
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
            $fatal(1,"FAIL KV_CACHE op=%0d: Physical memory range",op);
        return int'((address-BASE)>>6);
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"FAIL KV_CACHE op=%0d: %s",op,why);
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
                $display("NTRPU_KVC_B cycle=%0d id=%0d ordinal=%0d resp=%0d",cycles,writer_op,write_b_count,m_axi_bresp);
                b_count++;write_b_count++;m_axi_bvalid<=0;aw_pending=0;w_last_seen=0;write_applied=0;w_pending=0;
            end
            if(m_axi_awvalid && m_axi_awready)begin
                require(!aw_pending && m_axi_awsize==6 && m_axi_awburst==1 && m_axi_awid==0 &&
                    m_axi_awcache==2 && m_axi_awprot==0 && !m_axi_awlock,"AW protocol");
                aw_address=m_axi_awaddr;aw_beats=int'(m_axi_awlen)+1;aw_pending=1;aw_count++;
                require(aw_beats<=4 && int'(aw_address[11:0])+aw_beats*64<=4096,"AW page");
                $display("NTRPU_KVC_AW cycle=%0d id=%0d address=%016h beats=%0d",cycles,writer_op,aw_address,aw_beats);
            end
            if(m_axi_wvalid && m_axi_wready)begin
                require(w_pending<4 && !w_last_seen && m_axi_wstrb==64'hffffffffffffffff,"W protocol");
                pending_w[w_pending]=m_axi_wdata;w_pending++;w_count++;
                if(m_axi_wlast)w_last_seen=1;
                $display("NTRPU_KVC_W cycle=%0d id=%0d last=%0d data=%0128h",cycles,writer_op,m_axi_wlast,m_axi_wdata);
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
                $display("NTRPU_KVC_R cycle=%0d id=%0d address=%016h resp=%0d last=%0d data=%0128h",cycles,reader_op,ar_address+64'(ar_index*64),m_axi_rresp,m_axi_rlast,m_axi_rdata);
                r_count++;read_r_count++;ar_index++;m_axi_rvalid<=0;
                if(ar_index==ar_beats)ar_pending=0;
            end
            if(m_axi_arvalid && m_axi_arready)begin
                require(!ar_pending && !m_axi_rvalid && m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 &&
                    m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AR protocol");
                ar_address=m_axi_araddr;ar_beats=int'(m_axi_arlen)+1;ar_index=0;ar_pending=1;ar_count++;
                require(int'(ar_address[11:0])+ar_beats*64<=4096,"AR page");
                $display("NTRPU_KVC_AR cycle=%0d id=%0d address=%016h beats=%0d",cycles,reader_op,ar_address,ar_beats);
            end
            if(!m_axi_rvalid && ar_pending && !freeze_r && $urandom_range(0,3)!=0)begin
                require(initialized[slot(ar_address+64'(ar_index*64))],"Read uninitialized physical bytes");
                m_axi_rdata<=memory[slot(ar_address+64'(ar_index*64))];m_axi_rvalid<=1;m_axi_rlast<=ar_index+1==ar_beats;
                m_axi_rresp<=read_r_count==read_error_at ? 2:0;m_axi_rid<=0;
            end
            if((aw_pending || m_axi_wvalid) && (ar_pending || m_axi_rvalid))simultaneous_rw++;
        end
    end
    task automatic reset_all;
        reset_count++;
        $display("NTRPU_KVC_RESET cycle=%0d count=%0d",cycles,reset_count);
        @(negedge clk);rst_n=0;bind_valid=0;append_valid=0;data_valid=0;commit_valid=0;append_cancel_valid=0;
        lease_cmd_valid=0;lease_done_ready=0;read_valid=0;read_done_ready=0;read_cancel_valid=0;tile_ready=0;append_done_ready=0;
        freeze_aw=0;freeze_w=0;freeze_ar=0;freeze_r=0;freeze_b=0;
        repeat(3)@(negedge clk);
        require(!cache_valid && !lease_active && !lease_done_valid && !read_done_valid && !append_done_valid && !tile_valid,"Reset visibility");
        require(!m_axi_awvalid && !m_axi_wvalid && !m_axi_arvalid,"Reset AXI");
        rst_n=1;lease_id_model=0;next_start_model=0;for(int i=0;i<16;i++)prefix[i]=0;
    endtask
    task automatic bind_cache(input int capacity,offset);
        int id,ticks;
        op++;id=op;bind_count++;generation++;ticks=0;
        @(negedge clk);require(bind_ready,"Bind blocked after releasing lease");
        base_model=BASE+64'(offset);cap=capacity;seq_model=64'h1234567800000000+64'(generation);epoch_model=64'hfedcba9800000000+64'(generation);
        bind_cache_base=base_model;bind_region_base=BASE;bind_region_bytes=64'(capacity)*65536+8192;bind_capacity=32'(capacity);
        bind_sequence=seq_model;bind_epoch=epoch_model;expected_epoch=epoch_model;bind_valid=1;
        $display("NTRPU_KVC_BIND cycle=%0d id=%0d generation=%0d capacity=%0d base=%016h sequence=%016h epoch=%016h",cycles,id,generation,cap,base_model,seq_model,epoch_model);
        do @(posedge clk);while(!bind_ready);
        @(negedge clk);bind_valid=0;bind_cache_base=0;bind_capacity=0;bind_sequence=0;bind_epoch=0;
        while(!append_done_valid)begin @(negedge clk);ticks++;require(ticks<100,"Bind timeout");end
        require(append_done_bind && append_done_status==0 && cache_valid && cache_base==base_model && cache_capacity==32'(cap),"Bind result");
        require(append_done_sequence==seq_model && append_done_epoch==epoch_model,"Bind identity");
        append_done_ready=1;@(negedge clk);append_done_ready=0;
        for(int i=0;i<16;i++)prefix[i]=0;
        $display("NTRPU_KVC_BIND_DONE cycle=%0d id=%0d status=0",cycles,id);
    endtask
    task automatic append_token(input int layer,pos,mode,want_status);
        int id,index,ticks,old_count;
        op++;id=op;writer_op=id;append_count++;index=0;ticks=0;old_count=prefix[layer];write_error_mode=mode;write_b_count=0;
        @(negedge clk);append_layer=32'(layer);append_position=32'(pos);append_sequence=seq_model;append_epoch=epoch_model;append_valid=1;
        $display("NTRPU_KVC_APPEND cycle=%0d id=%0d generation=%0d layer=%0d position=%0d mode=%0d prefix=%0d",cycles,id,generation,layer,pos,mode,old_count);
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
        $display("NTRPU_KVC_APPEND_DONE cycle=%0d id=%0d status=%0d bytes=%0d committed=%0d",cycles,id,append_done_status,append_done_bytes,append_done_committed_tokens);
        repeat(4)@(negedge clk);append_done_ready=1;@(negedge clk);append_done_ready=0;write_error_mode=0;
    endtask
    task automatic lease_command(input bit release_cmd,input int layer,qhead,position,mode,want_status);
        int id,ticks,prior_prefix;
        bit was_active;
        logic [63:0] sent_id,sent_seq,sent_epoch;
        logic [228:0] held_response;
        op++;id=op;lease_count++;ticks=0;was_active=lease_active;
        prior_prefix=was_active ? lease_prefix_model:0;
        sent_id=release_cmd ? lease_id_model:next_lease;
        if(!release_cmd)next_lease++;
        sent_seq=seq_model;sent_epoch=epoch_model;
        if(mode==1)sent_seq^=64'h100000000;
        if(mode==2)sent_epoch^=64'h100000000;
        if(mode==3)sent_id=0;
        if(mode==4)sent_id=lease_id_model;
        if(mode==5)sent_id^=64'h100000000;
        @(negedge clk);
        lease_cmd_valid=1;lease_cmd_release=release_cmd;lease_cmd_layer=32'(layer);
        lease_cmd_query_head=32'(qhead);lease_cmd_position=32'(position);
        lease_cmd_sequence=sent_seq;lease_cmd_epoch=sent_epoch;lease_cmd_id=sent_id;
        $display("NTRPU_KVC_LEASE cycle=%0d id=%0d release=%0d layer=%0d head=%0d position=%0d sequence=%016h epoch=%016h lease=%016h",cycles,id,release_cmd,layer,qhead,position,sent_seq,sent_epoch,sent_id);
        do begin @(posedge clk);ticks++;require(ticks<5000,"Lease command timeout");end while(!lease_cmd_ready);
        @(negedge clk);lease_cmd_valid=0;lease_cmd_release=0;lease_cmd_layer=32'hffffffff;
        lease_cmd_query_head=32'hffffffff;lease_cmd_position=32'hffffffff;
        lease_cmd_sequence=0;lease_cmd_epoch=0;lease_cmd_id=0;
        require(lease_done_valid && lease_done_release==release_cmd && lease_done_status==4'(want_status),"Lease completion status");
        require(lease_done_sequence==sent_seq && lease_done_epoch==sent_epoch && lease_done_id==sent_id,"Lease completion identity");
        if(want_status==0 && !release_cmd)begin
            require(lease_active && lease_done_prefix==32'(prefix[layer]),"Lease owns published prefix");
            lease_id_model=sent_id;lease_layer_model=layer;lease_head_model=qhead;lease_position_model=position;
            lease_prefix_model=prefix[layer];lease_generation=generation;next_start_model=0;
        end else begin
            require(lease_done_prefix==32'(prior_prefix),"Rejected/released prefix");
            require(lease_active==(was_active && !(release_cmd && want_status==0)),"Rejected command changed lease");
        end
        held_response={lease_done_release,lease_done_status,lease_done_prefix,lease_done_sequence,lease_done_epoch,lease_done_id};
        repeat(5)begin
            @(negedge clk);
            require(!bind_ready && !read_ready && !lease_cmd_ready,"Held lease completion ownership");
            require(lease_done_valid && held_response=={lease_done_release,lease_done_status,lease_done_prefix,lease_done_sequence,lease_done_epoch,lease_done_id},"Held lease response changed");
        end
        $display("NTRPU_KVC_LEASE_DONE cycle=%0d id=%0d status=%0d prefix=%0d active=%0d",cycles,id,lease_done_status,lease_done_prefix,lease_active);
        lease_done_ready=1;@(negedge clk);lease_done_ready=0;
    endtask
    task automatic read_tile(input int start,count,mode,want_status);
        int id,ticks,seen,ar_before,hold_ticks,word_index,position,plane,limit;
        bit cancel_sent;
        logic [63:0] sent_seq,sent_epoch,sent_id;
        logic [221:0] held_response;
        op++;id=op;reader_op=id;read_count++;ticks=0;seen=0;hold_ticks=0;cancel_sent=0;ar_before=ar_count;
        read_r_count=0;read_error_at=mode==5 ? 3:-1;limit=WATCHDOG_CYCLES+10000;
        sent_seq=seq_model;sent_epoch=epoch_model;sent_id=lease_id_model;
        if(mode==1)sent_seq^=64'h100000000;
        if(mode==2)sent_epoch^=64'h100000000;
        if(mode==3)sent_id^=64'h100000000;
        @(negedge clk);read_start=32'(start);read_tokens=32'(count);read_sequence=sent_seq;read_epoch=sent_epoch;
        read_lease_id=sent_id;read_valid=1;tile_ready=0;
        $display("NTRPU_KVC_READ cycle=%0d id=%0d start=%0d tokens=%0d mode=%0d sequence=%016h epoch=%016h lease=%016h",cycles,id,start,count,mode,sent_seq,sent_epoch,sent_id);
        do begin @(posedge clk);ticks++;require(ticks<100,"Read command timeout");end while(!read_ready);
        @(negedge clk);read_valid=0;read_start=32'hffffffff;read_tokens=32'hffffffff;read_sequence=0;read_epoch=0;read_lease_id=0;
        while(!read_done_valid)begin
            ticks++;require(ticks<limit,"Read completion timeout");
            if((mode==6 || mode==7) && !cancel_sent)begin
                tile_ready=0;
                if(tile_valid)begin
                    hold_ticks++;
                    bind_valid=1;lease_cmd_valid=1;lease_cmd_release=1;
                    require(!bind_ready && !lease_cmd_ready,"Read/held data ownership");
                    if(mode==6 && hold_ticks>=5)read_cancel_valid=1;
                    if(mode==7 && hold_ticks>=WATCHDOG_CYCLES+8)begin
                        cancel_sent=1;$display("NTRPU_KVC_STALL cycle=%0d id=%0d cycles=%0d",cycles,id,hold_ticks);
                    end
                end
            end else tile_ready=$urandom_range(0,3)!=0;
            @(posedge clk);
            if(read_cancel_valid && read_cancel_ready)begin
                cancel_sent=1;$display("NTRPU_KVC_CANCEL cycle=%0d id=%0d",cycles,id);
            end
            if(tile_valid && tile_ready)begin
                plane=seen/(count*4);word_index=seen%(count*4);position=start+word_index/4;
                require(tile_plane==1'(plane) && tile_index==8'(word_index) && tile_last==(seen==count*8-1),"Tile shape/order");
                require(tile_sequence==seq_model && tile_epoch==epoch_model && tile_lease_id==lease_id_model,"Tile identity");
                require(tile_data==pattern(lease_generation,lease_layer_model,position,plane*32+(lease_head_model/4)*4+word_index%4),"Physical write/read payload");
                $display("NTRPU_KVC_TILE cycle=%0d id=%0d plane=%0d index=%0d last=%0d lease=%016h data=%0128h",cycles,id,tile_plane,tile_index,tile_last,tile_lease_id,tile_data);
                seen++;tile_count++;
            end
            @(negedge clk);if(cancel_sent)read_cancel_valid=0;
        end
        tile_ready=0;read_cancel_valid=0;
        require(read_done_status==4'(want_status),"Read completion status");
        require(read_done_sequence==sent_seq && read_done_epoch==sent_epoch && read_done_lease_id==sent_id,"Read completion identity");
        require(read_done_beats==10'(seen) && read_done_bytes==16'(seen*64),"Read completion byte count");
        if(want_status==0)begin require(seen==count*8,"Full K/V delivery");next_start_model+=count;end
        if(want_status==1 || want_status==2)require(ar_count==ar_before && seen==0,"Rejected read touched AXI");
        held_response={read_done_status,read_done_beats,read_done_bytes,read_done_sequence,read_done_epoch,read_done_lease_id};
        bind_valid=1;lease_cmd_valid=1;lease_cmd_release=1;
        repeat(5)begin
            @(negedge clk);require(!bind_ready && !lease_cmd_ready && !read_ready,"Read completion ownership");
            require(read_done_valid && held_response=={read_done_status,read_done_beats,read_done_bytes,read_done_sequence,read_done_epoch,read_done_lease_id},"Held read completion changed");
        end
        bind_valid=0;lease_cmd_valid=0;lease_cmd_release=0;
        $display("NTRPU_KVC_READ_DONE cycle=%0d id=%0d status=%0d beats=%0d bytes=%0d next=%0d",cycles,id,read_done_status,seen,seen*64,next_start_model);
        read_done_ready=1;@(negedge clk);read_done_ready=0;read_error_at=-1;
    endtask
    task automatic reset_read_job(input int scenario);
        int id,ticks;
        op++;id=op;reader_op=id;read_count++;ticks=0;read_error_at=-1;read_r_count=0;
        @(negedge clk);freeze_ar=scenario==0;freeze_r=scenario==1;
        read_start=0;read_tokens=1;read_sequence=seq_model;read_epoch=epoch_model;read_lease_id=lease_id_model;
        read_valid=1;tile_ready=0;
        $display("NTRPU_KVC_READ cycle=%0d id=%0d start=0 tokens=1 mode=%0d sequence=%016h epoch=%016h lease=%016h",cycles,id,8+scenario,seq_model,epoch_model,lease_id_model);
        do @(posedge clk);while(!read_ready);
        @(negedge clk);read_valid=0;
        while(!((scenario==0 && m_axi_arvalid && !m_axi_arready) ||
                (scenario==1 && ar_pending && !m_axi_rvalid) || (scenario==2 && tile_valid)))begin
            @(negedge clk);ticks++;require(ticks<100,"Reset phase not reached");
        end
        bind_valid=1;lease_cmd_valid=1;lease_cmd_release=1;
        repeat(3)begin @(negedge clk);require(!bind_ready && !lease_cmd_ready,"Rebind/release before read reset");end
        $display("NTRPU_KVC_READ_ABORT cycle=%0d id=%0d phase=%0d",cycles,id,scenario);
        reset_all();
    endtask
    task automatic reset_held_grant;
        int id;
        op++;id=op;lease_count++;
        @(negedge clk);lease_cmd_valid=1;lease_cmd_release=0;lease_cmd_layer=15;lease_cmd_query_head=31;
        lease_cmd_position=0;lease_cmd_sequence=seq_model;lease_cmd_epoch=epoch_model;lease_cmd_id=next_lease;
        $display("NTRPU_KVC_LEASE cycle=%0d id=%0d release=0 layer=15 head=31 position=0 sequence=%016h epoch=%016h lease=%016h",cycles,id,seq_model,epoch_model,next_lease);
        next_lease++;
        do @(posedge clk);while(!lease_cmd_ready);
        @(negedge clk);lease_cmd_valid=0;
        require(lease_done_valid && lease_done_status==0 && lease_active,"Grant reset phase");
        repeat(3)begin @(negedge clk);require(!bind_ready && !read_ready,"Unacknowledged grant ownership");end
        $display("NTRPU_KVC_LEASE_ABORT cycle=%0d id=%0d active=%0d",cycles,id,lease_active);
        reset_all();
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end
        random_state=seed;random_state=$urandom(random_state);
        reset_all();bind_cache(4,4032);
        lease_command(0,0,0,0,0,1); // No published token.
        append_token(0,0,1,3);lease_command(0,0,0,0,0,1); // Failed B is not publication.
        append_token(0,0,0,0);
        lease_command(0,0,0,0,1,2);lease_command(0,0,0,0,2,2);lease_command(0,0,0,0,3,2);
        lease_command(0,16,0,0,0,1);lease_command(0,0,32,0,0,1);lease_command(0,0,0,4,0,1);
        lease_command(0,0,0,0,0,0);
        read_tile(0,1,1,2);read_tile(0,1,2,2);read_tile(0,1,3,2);
        read_tile(1,1,0,1);read_tile(0,0,0,1);read_tile(0,65,0,1);read_tile(0,2,0,1);
        lease_command(1,0,0,0,1,2);lease_command(1,0,0,0,2,2);lease_command(1,0,0,0,5,2);
        read_tile(0,1,5,3);read_tile(0,1,6,7);read_tile(0,1,7,8);read_tile(0,1,0,0);
        read_tile(0,1,0,1); // Successful tiles cannot be consumed twice.
        lease_command(1,0,0,0,0,0);lease_command(0,0,0,0,4,2);
        // Append publication may grow, but an active query keeps its old prefix.
        lease_command(0,0,3,0,0,0);
        fork
            append_token(0,1,0,0);
            begin wait(m_axi_awvalid);read_tile(0,1,0,0);end
        join
        require(prefix[0]==2 && lease_prefix_model==1,"Snapshot prefix changed during append");
        read_tile(1,1,0,1);lease_command(1,0,0,0,0,0);
        append_token(0,2,2,3);lease_command(0,0,0,2,0,1);append_token(0,2,0,0);
        // A pending query cannot observe a token before all B responses publish it.
        freeze_b=1;
        fork
            append_token(2,0,0,0);
            begin wait(m_axi_bready);lease_command(0,2,0,0,0,0);end
            begin wait(m_axi_bready);repeat(5)@(negedge clk);freeze_b=0;end
        join
        read_tile(0,1,0,0);lease_command(1,0,0,0,0,0);
        // Read every GQA query-head mapping from real writes in two layers.
        append_token(15,0,0,0);
        for(int head=0;head<32;head++)begin
            lease_command(0,head[0] ? 15:0,head,0,0,0);read_tile(0,1,0,0);lease_command(1,0,0,0,0,0);
        end
        // Epoch revocation rejects new reads while permitting release/drain.
        lease_command(0,0,31,2,0,0);
        expected_epoch=epoch_model+1;$display("NTRPU_KVC_EPOCH cycle=%0d expected=%016h",cycles,expected_epoch);
        read_tile(0,3,0,2);lease_command(1,0,0,0,0,0);
        lease_command(0,0,31,2,0,2);expected_epoch=epoch_model;$display("NTRPU_KVC_EPOCH cycle=%0d expected=%016h",cycles,expected_epoch);
        // Rebind clears all prefixes, even when physical memory contains old data.
        bind_cache(65,64);lease_command(0,0,0,0,0,1);
        for(int pos=0;pos<65;pos++)append_token(15,pos,0,0);
        lease_command(0,15,31,64,0,0);read_tile(0,64,0,0);read_tile(64,1,0,0);
        read_tile(65,1,0,1);lease_command(1,0,0,0,0,0);
        // Reset invalidates an active query lease; rebind recovers.
        lease_command(0,15,0,64,0,0);reset_all();
        bind_cache(1,0);lease_command(0,15,0,0,0,1);append_token(15,0,0,0);
        lease_command(0,15,31,0,0,0);read_tile(0,1,0,0);lease_command(1,0,0,0,0,0);
        for(int scenario=0;scenario<3;scenario++)begin
            lease_command(0,15,31,0,0,0);reset_read_job(scenario);
            bind_cache(1,64);lease_command(0,15,31,0,0,1);append_token(15,0,0,0);
            lease_command(0,15,31,0,0,0);read_tile(0,1,0,0);lease_command(1,0,0,0,0,0);
        end
        reset_held_grant();bind_cache(1,4032);lease_command(0,15,31,0,0,1);append_token(15,0,0,0);
        lease_command(0,15,31,0,0,0);read_tile(0,1,0,0);lease_command(1,0,0,0,0,0);
        require(aw_waits>0 && w_waits>0 && ar_waits>0 && tile_waits>0,"Backpressure coverage");
        require(simultaneous_rw>0,"Concurrent AXI reader/writer coverage");
        require(!aw_pending && !ar_pending && !m_axi_bvalid && !m_axi_rvalid,"Undrained AXI");
        $display("NTRPU_KVC_PASS seed=%0d watchdog=%0d binds=%0d appends=%0d leases=%0d reads=%0d resets=%0d aw=%0d w=%0d b=%0d ar=%0d r=%0d tiles=%0d overlap=%0d cycles=%0d",seed,WATCHDOG_CYCLES,bind_count,append_count,lease_count,read_count,reset_count,aw_count,w_count,b_count,ar_count,r_count,tile_count,simultaneous_rw,cycles);
        $finish;
    end
    initial begin #1000000000;$fatal(1,"FAIL KV_CACHE global timeout");end
endmodule
