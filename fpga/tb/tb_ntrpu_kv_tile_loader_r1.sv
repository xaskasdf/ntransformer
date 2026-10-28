`timescale 1ns/1ps
module tb_ntrpu_kv_tile_loader_r1 #(
    parameter integer WATCHDOG_CYCLES=32
);
    logic clk=0,rst_n=0;always #2.5 clk=~clk;
    logic cmd_valid=0,cmd_ready,cancel_valid=0,cancel_ready;
    logic [63:0] cmd_cache_base=0,cmd_region_base=0,cmd_region_bytes=0;
    logic [31:0] cmd_capacity=0,cmd_committed_tokens=0,cmd_position=0,cmd_tile_start=0;
    logic [31:0] cmd_tile_tokens=0,cmd_layer=0,cmd_query_head=0,cmd_kv_head=0;
    logic [63:0] cmd_sequence=0,cmd_epoch=0,cmd_cache_sequence=0,cmd_cache_epoch=0,expected_epoch=0;
    logic tile_valid,tile_ready=0,tile_plane,tile_last;
    logic [511:0] tile_data;
    logic [7:0] tile_index;
    logic [63:0] tile_sequence,tile_epoch;
    logic done_valid,done_ready=0;
    logic [3:0] done_status;
    logic [9:0] done_beats;
    logic [15:0] done_bytes;
    logic [63:0] done_sequence,done_epoch;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize,m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid,m_axi_arcache;
    logic m_axi_arlock,m_axi_arvalid,m_axi_arready=0;
    logic [511:0] m_axi_rdata=0;
    logic [1:0] m_axi_rresp=0;
    logic [3:0] m_axi_rid=0;
    logic m_axi_rlast=0,m_axi_rvalid=0,m_axi_rready;
    ntrpu_kv_tile_loader_r1 #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES)) dut(.*);
    localparam logic [63:0] BASE=64'ha000000000;
    int seed=1,random_state,case_id=0,total_outputs=0,total_bursts=0,resets=0;
    int ar_stalls=0,output_stalls=0,done_stalls=0;

    function automatic logic [63:0] location(input logic [63:0] base,input int cap,lay,head,plane,start);
        return base+(((64'(lay)*2+64'(plane))*8+64'(head))*64'(cap)+64'(start))*256;
    endfunction
    function automatic logic [511:0] memory_word(input logic [63:0] address,input int scenario);
        logic [31:0] word;
        for(int lane=0;lane<16;lane++) begin
            word=32'h3f000000 | (32'((address>>2)^(address>>33)^(address>>40)^64'(lane*29+case_id*17))&32'h007fffff);
            if(scenario==43) begin
                case(lane%6)
                    0:word=0;1:word=32'h80000000;2:word=1;3:word=32'h80000001;
                    4:word=32'h7f7fffff;5:word=32'hff7fffff;
                endcase
            end
            memory_word[lane*32+:32]=word;
        end
    endfunction
    task automatic require(input bit ok,input string why);
        if(!ok)$fatal(1,"FAIL KV_LOADER case=%0d: %s",case_id,why);
    endtask
    task automatic run_case(input int n,cap,pos,start,qhead,lay,offset,scenario,status);
        bit active,accepted,finished,cancelled,reset_done;
        bit ar_stalled,out_stalled,done_stalled,ar_hs,r_hs,out_hs,cmd_hs,cancel_hs;
        logic [71:0] saved_ar;
        logic [649:0] saved_output;
        logic [157:0] saved_done;
        logic [63:0] k_first,v_first,next_address,burst_address,seq,epoch,out_address;
        logic [511:0] want;
        int cycles,outputs,bursts,returned,beat,beat_count,done_wait,remaining,plan_plane,burst_plane;
        int burst_global_index,out_index,out_plane,burst_limit;
        case_id++;active=0;accepted=0;finished=0;cancelled=0;reset_done=0;
        ar_stalled=0;out_stalled=0;done_stalled=0;cycles=0;outputs=0;bursts=0;returned=0;
        beat=0;beat_count=0;done_wait=0;remaining=n*4;plan_plane=0;burst_plane=0;burst_global_index=0;burst_limit=0;
        @(negedge clk);require(cmd_ready,"Previous command not released");
        cmd_cache_base=BASE+64'(offset);cmd_region_base=BASE;cmd_region_bytes=64'(cap)*65536+8192;
        cmd_capacity=32'(cap);cmd_committed_tokens=32'(pos+1);cmd_position=32'(pos);cmd_tile_start=32'(start);
        cmd_tile_tokens=32'(n);cmd_layer=32'(lay);cmd_query_head=32'(qhead);cmd_kv_head=32'(qhead/4);
        cmd_sequence=64'h1234567800000000+64'(case_id);cmd_epoch=64'hfedcba9800000000+64'(case_id);
        cmd_cache_sequence=cmd_sequence;cmd_cache_epoch=cmd_epoch;expected_epoch=cmd_epoch;
        case(scenario)
            3:cmd_cache_base++;
            4:cmd_region_bytes=64'(offset)+64'(cap)*65536-1;
            5:cmd_region_base=cmd_cache_base+64;
            6:begin cmd_cache_base=64'hfffffffffffff000;cmd_region_base=cmd_cache_base;cmd_region_bytes=8192;end
            7:begin cmd_cache_base=64'hfffffffffffff000;cmd_region_base=0;cmd_region_bytes=64'hffffffffffffffff;end
            8:expected_epoch++;
            9:cmd_cache_epoch++;
            10:cmd_cache_sequence++;
            11:cmd_sequence=0;
            12:cmd_epoch=0;
            13:cmd_capacity=0;
            14:cmd_capacity=131073;
            15:cmd_position=cmd_capacity;
            16:cmd_committed_tokens=cmd_position;
            17:cmd_committed_tokens=cmd_capacity+1;
            18:cmd_tile_start=32'hffffffff;
            19:cmd_tile_start=cmd_position;
            20:cmd_layer=16;
            21:cmd_query_head=32;
            22:cmd_kv_head=cmd_kv_head^1;
            23:cmd_query_head=256;
        endcase
        k_first=location(cmd_cache_base,cap,lay,qhead/4,0,start);v_first=location(cmd_cache_base,cap,lay,qhead/4,1,start);
        next_address=k_first;seq=cmd_sequence;epoch=cmd_epoch;cmd_valid=1;cancel_valid=0;m_axi_rvalid=0;
        $display("NTRPU_KV_BEGIN id=%0d scenario=%0d n=%0d capacity=%0d position=%0d start=%0d layer=%0d qhead=%0d kvhead=%0d committed=%0d cache=%016h region=%016h bytes=%016h sequence=%016h epoch=%016h cache_sequence=%016h cache_epoch=%016h expected_epoch=%016h",
            case_id,scenario,cmd_tile_tokens,cmd_capacity,cmd_position,cmd_tile_start,cmd_layer,cmd_query_head,cmd_kv_head,cmd_committed_tokens,
            cmd_cache_base,cmd_region_base,cmd_region_bytes,cmd_sequence,cmd_epoch,cmd_cache_sequence,cmd_cache_epoch,expected_epoch);
        while(!finished) begin
            m_axi_arready=$urandom_range(0,3)!=0;
            if((scenario==31 || scenario==38) && cycles<30)m_axi_arready=0;
            if(scenario==34 && cycles<WATCHDOG_CYCLES+68)m_axi_arready=0;
            tile_ready=cycles>=15 && $urandom_range(0,3)!=0;
            if(scenario==36 && cycles<WATCHDOG_CYCLES+68)tile_ready=0;
            if((scenario==37 || scenario==41) && cycles<100)tile_ready=0;
            done_ready=done_wait>=6 && $urandom_range(0,2)!=0;
            if(scenario==42)done_ready=0;
            if(!cancelled && cancel_ready && ((scenario==30 && cycles==2) ||
                (scenario==31 && m_axi_arvalid && cycles>=10) || (scenario==32 && returned>=3) ||
                (scenario==33 && returned>=n*4+3) || (scenario==37 && tile_valid)))cancel_valid=1;
            if(!m_axi_rvalid && active && $urandom_range(0,3)!=0 && !(scenario==35 && cycles<WATCHDOG_CYCLES+88))begin
                m_axi_rdata=memory_word(burst_address+64'(beat*64),scenario);m_axi_rresp=0;m_axi_rid=0;
                if((scenario==24 && burst_plane==0 || scenario==25 && burst_plane==1) && burst_global_index+beat==n*4-1)
                    m_axi_rdata[511:480]=32'h7fc12345;
                if(scenario==26 && burst_plane==0 && burst_global_index+beat==n*4-1)m_axi_rresp=2;
                if(scenario==27 && burst_plane==1 && burst_global_index+beat==n*4-1)m_axi_rid=1;
                m_axi_rlast=beat==beat_count-1;
                if(scenario==28 && bursts==1)m_axi_rlast=beat==2;
                if(scenario==29 && bursts==1)m_axi_rlast=beat==beat_count;
                m_axi_rvalid=1;
            end
            @(posedge clk);cycles++;
            if(cycles==2*WATCHDOG_CYCLES+9999)$display("NTRPU_KV_TIMEOUT state=%0d status=%0d cmd=%b/%b reader=%0d ready=%b address=%h chunk=%0d received=%0d out=%b/%b",dut.state,done_status,cmd_valid,cmd_ready,dut.reader.state,dut.reader_cmd_ready,dut.address,dut.chunk_beats,dut.received,tile_valid,tile_ready);
            require(cycles<2*WATCHDOG_CYCLES+10000,"Test timeout");
            cmd_hs=cmd_valid&&cmd_ready;cancel_hs=cancel_valid&&cancel_ready;
            ar_hs=m_axi_arvalid&&m_axi_arready;r_hs=m_axi_rvalid&&m_axi_rready;out_hs=tile_valid&&tile_ready;
            if(ar_stalled)require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"AR stability");
            if(out_stalled)require(tile_valid && {tile_data,tile_plane,tile_last,tile_index,tile_sequence,tile_epoch}===saved_output,"Held output stability");
            if(done_stalled)require(done_valid && {done_status,done_beats,done_bytes,done_sequence,done_epoch}===saved_done,"Completion stability");
            ar_stalled=m_axi_arvalid&&!m_axi_arready;out_stalled=tile_valid&&!tile_ready;done_stalled=done_valid&&!done_ready;
            saved_ar={m_axi_araddr,m_axi_arlen};saved_output={tile_data,tile_plane,tile_last,tile_index,tile_sequence,tile_epoch};
            saved_done={done_status,done_beats,done_bytes,done_sequence,done_epoch};
            ar_stalls+=int'(ar_stalled);output_stalls+=int'(out_stalled);done_stalls+=int'(done_stalled);
            if(cmd_hs)begin require(!accepted,"Duplicate command");accepted=1;end
            if(cancel_hs)begin
                cancelled=1;burst_limit=bursts+(ar_stalled?1:0);
                $display("NTRPU_KV_CANCEL id=%0d cycle=%0d bursts=%0d pending_ar=%0d",case_id,cycles,bursts,ar_stalled);
            end
            if(ar_hs)begin
                require(accepted && !active,"Unexpected AXI request");
                if(cancelled)require(bursts<burst_limit,"New burst after cancellation");
                require(m_axi_araddr==next_address,"Address including high bits");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AXI attributes");
                beat_count=int'(m_axi_arlen)+1;
                require(beat_count==(remaining<(4096-int'(next_address[11:0]))/64 ? remaining:(4096-int'(next_address[11:0]))/64),"Burst split");
                require(int'(m_axi_araddr[11:0])+beat_count*64<=4096,"Page crossing");
                burst_address=next_address;burst_plane=plan_plane;burst_global_index=n*4-remaining;
                $display("NTRPU_KV_AR id=%0d plane=%0d index=%0d addr=%016h beats=%0d",case_id,burst_plane,burst_global_index,m_axi_araddr,beat_count);
                next_address+=64'(beat_count*64);remaining-=beat_count;
                if(remaining==0 && plan_plane==0)begin plan_plane=1;remaining=n*4;next_address=v_first;end
                active=1;beat=0;bursts++;
            end
            if(r_hs)begin
                require(active,"Unsolicited response");
                $display("NTRPU_KV_R id=%0d plane=%0d index=%0d addr=%016h resp=%0d rid=%0d last=%0d tail=%08h",
                    case_id,burst_plane,burst_global_index+beat,burst_address+64'(beat*64),m_axi_rresp,m_axi_rid,m_axi_rlast,m_axi_rdata[511:480]);
                beat++;returned++;if(m_axi_rlast)active=0;
            end
            if(out_hs)begin
                out_plane=outputs/(n*4);out_index=outputs%(n*4);
                out_address=(out_plane==0?k_first:v_first)+64'(out_index*64);want=memory_word(out_address,scenario);
                require(tile_plane==1'(out_plane) && tile_index==8'(out_index) && tile_last==(outputs+1==n*8),"Tile order/extent");
                require(tile_sequence==seq && tile_epoch==epoch && tile_data===want,"Tile identity/physical bytes");
                $display("NTRPU_KV_DATA id=%0d plane=%0d index=%0d last=%0d data=%0128h",case_id,tile_plane,tile_index,tile_last,tile_data);
                outputs++;
            end
            if(done_valid)begin
                done_wait++;require(!active && !tile_valid,"Completion before drain");
                require(done_status==4'(status),$sformatf("Status=%0d expected=%0d",done_status,status));
                require(done_beats==10'(outputs) && done_bytes==16'(outputs*64) && done_sequence==seq && done_epoch==epoch,"Completion count/tags");
                if(status==0)require(outputs==n*8,"Incomplete success");
                if(status==1 || status==2 || scenario==30)require(bursts==0 && outputs==0,"Rejected descriptor accessed memory");
                if(scenario==34 || scenario==35 || scenario==36)require(bursts==1,"Watchdog allowed a new burst");
                if(done_ready)finished=1;
            end
            #1;
            if(cmd_hs)begin
                cmd_valid=0;
                if(scenario==44)begin
                    cmd_cache_base=64'hffffffffffffffff;cmd_region_base=0;cmd_region_bytes=0;cmd_capacity=0;
                    cmd_tile_tokens=0;cmd_layer=99;cmd_query_head=99;cmd_kv_head=99;cmd_position=0;cmd_tile_start=0;
                    cmd_committed_tokens=0;cmd_sequence=0;cmd_epoch=0;cmd_cache_sequence=0;cmd_cache_epoch=0;expected_epoch=0;
                end
            end
            if(cancel_hs)cancel_valid=0;
            if(r_hs)m_axi_rvalid=0;
            if((scenario==38 && ar_stalled) || (scenario==39 && active && returned>=2) ||
                (scenario==40 && active && returned>=n*4+2) || (scenario==41 && out_stalled) || (scenario==42 && done_valid))begin
                @(negedge clk);rst_n=0;cmd_valid=0;cancel_valid=0;m_axi_rvalid=0;active=0;
                repeat(4)@(negedge clk);rst_n=1;repeat(8)@(negedge clk);
                require(cmd_ready && !done_valid && !tile_valid && !m_axi_arvalid,"Stale state after coordinated reset");
                reset_done=1;finished=1;resets++;
            end
            if(!finished)@(negedge clk);
        end
        if(reset_done)$display("NTRPU_KV_RESET id=%0d scenario=%0d outputs=%0d bursts=%0d returned=%0d",case_id,scenario,outputs,bursts,returned);
        else $display("NTRPU_KV_CASE id=%0d scenario=%0d status=%0d cycles=%0d outputs=%0d bursts=%0d returned=%0d",case_id,scenario,status,cycles,outputs,bursts,returned);
        total_outputs+=outputs;total_bursts+=bursts;repeat(3)@(negedge clk);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed))begin end
        $display("NTRPU_KV_SEED seed=%0d",seed);random_state=$urandom(seed);
        $display("NTRPU_KV_CONFIG watchdog=%0d",WATCHDOG_CYCLES);
        repeat(4)@(negedge clk);rst_n=1;@(posedge clk);#1;
        run_case(1,1,0,0,0,0,0,0,0);
        run_case(2,17,16,15,7,3,4032,0,0);
        run_case(17,65,64,48,28,15,64,0,0);
        run_case(64,131072,131071,131008,31,15,4032,0,0);
        for(int head=0;head<32;head++)run_case(64,131072,131071,131008,head,head%16,64,0,0);
        run_case(0,131072,63,0,0,0,0,1,1);run_case(65,131072,64,0,0,0,0,2,1);
        for(int scenario=3;scenario<=23;scenario++)run_case(2,131072,63,0,3,0,0,scenario,(scenario>=8 && scenario<=12)?2:1);
        for(int scenario=24;scenario<=29;scenario++)run_case(16,131072,63,0,31,15,0,scenario,scenario<=25?4:3);
        for(int scenario=30;scenario<=37;scenario++)run_case(16,131072,63,0,31,15,0,scenario,(scenario>=34 && scenario<=36)?8:7);
        for(int scenario=38;scenario<=42;scenario++)begin
            run_case(16,131072,63,0,31,15,0,scenario,0);
            run_case(1,1,0,0,0,0,0,0,0);
        end
        run_case(64,131072,131071,131008,31,15,4032,43,0);
        run_case(64,131072,131071,131008,31,15,4032,44,0);
        require(case_id==85 && resets==5,"Coverage count");
        require(ar_stalls>0 && output_stalls>0 && done_stalls>0,"Missing backpressure");
        $display("NTRPU_KV_PASS cases=%0d resets=%0d outputs=%0d bursts=%0d ar_stalls=%0d output_stalls=%0d done_stalls=%0d",
            case_id,resets,total_outputs,total_bursts,ar_stalls,output_stalls,done_stalls);$finish;
    end
endmodule
