`timescale 1ns/1ps
module tb_ntrpu_q8_gemv_memory;
    logic clk=0,rst_n=0;
    always #2.5 clk=~clk;
    logic job_valid=0,job_ready,tile_valid=0,tile_ready,cancel_valid=0,cancel_ready;
    logic [1:0] job_layout=2;
    logic [5:0] job_rows=0;
    logic [8:0] job_k_blocks=0,tile_k_base=0;
    logic [31:0] job_sequence=0,job_epoch=0,job_activation_epoch=0,tile_tag=0,tile_epoch=0;
    logic [63:0] job_region_base=0,job_region_bytes=0,job_activation_addr=0;
    logic [63:0] job_activation_region_base=0,job_activation_region_bytes=0,tile_addr=0;
    logic [5:0] tile_k_count=32;
    logic result_valid,result_ready=0,result_last,done_valid,done_ready=0;
    logic [63:0] result_data;
    logic [4:0] result_row;
    logic [31:0] result_sequence,result_epoch,done_sequence,done_epoch;
    logic [3:0] done_status;
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
`ifdef NTRPU_GEMV_QUAD
    localparam int LANES=4;
    ntrpu_q8_gemv_quad_memory #(.WATCHDOG_CYCLES(32)) dut(.*);
`else
    localparam int LANES=2;
    ntrpu_q8_gemv_memory #(.WATCHDOG_CYCLES(32)) dut(.*);
`endif
    localparam logic [63:0] ACT=64'h8000000000,WEIGHT=64'h9000000000;
    int seed=1,random_state,case_id=0,total_cycles=0,total_outputs=0,total_ab=0,total_wb=0;
    int ar_stalls=0,result_stalls=0,done_stalls=0,reset_cases=0;
    int act_beats,weight_beats,outputs,tiles,cycles,rows,k,mode;
    bit activation_complete;
    bit fixture_mode=0;
    string fixture_dir;
    logic [63:0] weight_base=WEIGHT;
    int tile_weight_beats,active_tile_beats,fixture_chunk=32;
    logic [3:0] lane_busy_mask,lane_wait_mask;
`ifdef NTRPU_MEMORY_MAC_FAULTS
    wire [3:0] raw_done_mask,shim_pending;
    wire [31:0] raw_status[0:3];
    for(genvar g=0;g<4;g++) begin
        assign raw_done_mask[g]=dut.core.workers[g].mac.raw_done;
        assign shim_pending[g]=dut.core.workers[g].mac.pending;
        assign raw_status[g]=dut.core.workers[g].mac.raw_status;
    end
    task automatic hold_mac_completions(input bit value);
        dut.core.workers[0].mac.hold_completion=value;
        dut.core.workers[1].mac.hold_completion=value;
        dut.core.workers[2].mac.hold_completion=value;
        dut.core.workers[3].mac.hold_completion=value;
    endtask
    task automatic inject_mac_scale(input int mask);
        dut.core.workers[0].mac.inject_invalid_scale=(mask&1)!=0;
        dut.core.workers[1].mac.inject_invalid_scale=(mask&2)!=0;
        dut.core.workers[2].mac.inject_invalid_scale=(mask&4)!=0;
        dut.core.workers[3].mac.inject_invalid_scale=(mask&8)!=0;
    endtask
`endif
    always_comb begin
        lane_busy_mask=0;lane_wait_mask=0;
        for(int lane=0;lane<LANES;lane++) begin
            lane_busy_mask[lane]=dut.core.lane_state[lane]!=0;
            lane_wait_mask[lane]=dut.core.lane_state[lane]==2;
        end
    end
`ifdef NTRPU_MEMORY_EXTENDED
    localparam int CASE_BUDGET=100000;
    wire fault_compute_ready=dut.core.mac_state==2 && (rows<LANES || lane_wait_mask==4'((1<<LANES)-1));
`else
    localparam int CASE_BUDGET=20000;
    wire fault_compute_ready=dut.core.mac_state==2;
`endif
`ifdef NTRPU_MEMORY_FIXTURE
    logic [7:0] weight_image [0:524287],activation_image [0:32767];
    logic [127:0] fixture_tiles [0:255];
    logic [63:0] fixture_reference [0:31];
    int weight_image_bytes,fixture_tile_count;
`endif
    function automatic void require(input bit ok,input string message);
        if(!ok) $fatal(1,"FAIL MEMORY case=%0d cycle=%0d: %s",case_id,cycles,message);
    endfunction
`ifdef NTRPU_QUAD_BLOCK_FIFO
    // Independent ordered payload scoreboard; observes both interfaces.
    logic [314:0] fifo_expected[0:1];
    int fifo_count=0,fifo_read=0,fifo_write=0;
    int fifo_pushes=0,fifo_pops=0,fifo_dropped=0,fifo_peak=0,fifo_full_stalls=0,fifo_both=0;
    always @(posedge clk) begin
        if (!rst_n) begin
            fifo_dropped+=fifo_count;
            fifo_count=0;fifo_read=0;fifo_write=0;
        end else begin
            require(dut.core.block_fifo_count===2'(fifo_count),"FIFO occupancy matches independent scoreboard");
            require(dut.core.block_valid===(fifo_count!=0),"FIFO valid reflects occupancy");
            if(dut.core.job_ready || dut.core.done_valid) require(fifo_count==0,"FIFO drained before completion/new job");
            if(fifo_count==2) begin
                require(!dut.core.decoded_block_ready,"full FIFO never combinationally accepts a replacement");
                fifo_full_stalls+=int'(dut.core.decoded_block_valid);
            end
            if(dut.core.block_fire) begin
                require(fifo_count>0,"FIFO cannot consume from empty");
                require({dut.core.block_scale,dut.core.block_weights,dut.core.block_index,
                         dut.core.block_tag,dut.core.block_last}===fifo_expected[fifo_read],"FIFO payload order and metadata");
                fifo_read=1-fifo_read;fifo_count--;fifo_pops++;
            end
            if(dut.core.decoded_block_fire) begin
                require(fifo_count<2,"FIFO cannot overrun");
                fifo_expected[fifo_write]={dut.core.decoded_block_scale,dut.core.decoded_block_weights,
                    dut.core.decoded_block_index,dut.core.decoded_block_tag,dut.core.decoded_block_last};
                fifo_write=1-fifo_write;fifo_count++;fifo_pushes++;
            end
            if(dut.core.block_fire && dut.core.decoded_block_fire) fifo_both++;
            if(fifo_count>fifo_peak) fifo_peak=fifo_count;
        end
    end
`endif
    function automatic logic [511:0] memory_word(input logic [63:0] addr);
        int off,block_id,lane,row,global_k;
        memory_word=0;
`ifdef NTRPU_MEMORY_FIXTURE
        if(fixture_mode) begin
            if(addr>=ACT && addr<ACT+64'h100000) begin
                off=int'(addr-ACT);
                require(off>=4032 && off+64<=4032+k*64,"activation image bounds");
                for(int j=0;j<64;j++) memory_word[j*8+:8]=activation_image[off+j];
            end else begin
                off=int'(addr-weight_base);
                require(off>=0 && off+64<=weight_image_bytes,"weight image bounds");
                for(int j=0;j<64;j++) memory_word[j*8+:8]=weight_image[off+j];
            end
            return memory_word;
        end
`endif
        if(addr>=ACT && addr<ACT+64'h100000) begin
            block_id=int'((addr-ACT-4032)/64);
            for(int j=0;j<32;j++) memory_word[j*8+:8]=8'((block_id+j)%5-2);
            memory_word[319:256]=64'h3ff0000000000000;
            if(mode==5 && block_id==k-1) memory_word[511]=1;
            if(mode==6 && block_id==k-1) memory_word[319:256]=64'h7ff8000000000000;
        end else begin
            off=int'(addr-WEIGHT-(64'(tiles-1)*65536)-4032);
            for(int j=0;j<64;j++) begin
                block_id=(off+j)/34;lane=(off+j)%34;
                row=block_id%rows;global_k=(tiles-1)*32+block_id/rows;
                if(lane==0) memory_word[j*8+:8]=0;
                else if(lane==1) begin
                    memory_word[j*8+:8]=8'h3c;
                end
                else memory_word[j*8+:8]=8'((global_k+row*3+lane-2)%7-3);
            end
        end
    endfunction
    function automatic real reference_value(input int row);
        int sum;
`ifdef NTRPU_MEMORY_FIXTURE
        if(fixture_mode) return $bitstoreal(fixture_reference[row]);
`endif
        sum=0;
        for(int block_id=0;block_id<k;block_id++)
            for(int lane=0;lane<32;lane++)
                sum+=((block_id+lane)%5-2)*((block_id+row*3+lane)%7-3);
        return real'(sum);
    endfunction
    task automatic run_case(input int nrows,input int nk,input int scenario,input int status);
        bit active,accepted,finished,cancelled,is_activation;
        bit ar_hold,result_hold,done_hold,job_hs,tile_hs,ar_hs,r_hs,cancel_hs;
        logic [71:0] saved_ar;
        logic [133:0] saved_result;
        logic [67:0] saved_done;
        logic [63:0] burst_address,expected_addr;
        int beat,beats,ab,wb,done_wait,first_result,result_wait,expected_count;
        int lane_starts[0:3],lane_dones[0:3],lane_busy[0:3],peak_busy,cancel_mask,reset_mask;
`ifdef NTRPU_MEMORY_MAC_FAULTS
        int raw_count[0:3],error_mask,error_busy_mask,held_start,released_at,held_mask,pending_seen;
        int timeout_at,timeout_idle,timeout_mask;
        bit held;
        for(int lane=0;lane<4;lane++) raw_count[lane]=0;
        error_mask=0;error_busy_mask=0;held_start=0;released_at=0;held_mask=0;pending_seen=0;
        timeout_at=0;timeout_idle=0;timeout_mask=0;held=0;
        hold_mac_completions(0);
        inject_mac_scale(0);
`endif
        for(int lane=0;lane<4;lane++) begin lane_starts[lane]=0;lane_dones[lane]=0;lane_busy[lane]=0;end
        peak_busy=0;cancel_mask=0;reset_mask=0;
        case_id++;rows=nrows;k=nk;mode=scenario;cycles=0;outputs=0;tiles=0;act_beats=0;weight_beats=0;
        active=0;accepted=0;finished=0;cancelled=0;activation_complete=0;tile_weight_beats=0;active_tile_beats=0;
        ar_hold=0;result_hold=0;done_hold=0;ab=0;wb=0;done_wait=0;first_result=0;result_wait=0;
        @(negedge clk);require(job_ready,"previous job not released");
        job_layout=mode==2 ? 1:2;job_rows=6'(rows);job_k_blocks=9'(k);
        job_sequence=32'(case_id);job_epoch=32'(case_id+200);job_activation_epoch=job_epoch;
        job_region_base=weight_base;job_region_bytes=64'h200000;
        job_activation_addr=ACT+4032;job_activation_region_base=ACT;job_activation_region_bytes=64'h100000;
        if(mode==3) job_activation_epoch++;
        if(mode==4) job_activation_region_bytes=4096;
        tile_addr=WEIGHT+4032;tile_k_base=0;tile_k_count=32;tile_tag=1000;tile_epoch=job_epoch;
`ifdef NTRPU_MEMORY_FIXTURE
        if(fixture_mode) begin
            tile_addr=fixture_tiles[0][127:64];tile_k_base=9'(fixture_tiles[0][63:32]);tile_k_count=6'(fixture_tiles[0][31:0]);
        end
`endif
        job_valid=1;tile_valid=1;cancel_valid=0;m_axi_rvalid=0;
        while(!finished) begin
`ifdef NTRPU_MEMORY_MAC_FAULTS
            // Inject at the HLS boundary: malformed external scales are rejected
            // earlier by the Q8 decoder and cannot exercise MAC_ERROR directly.
            if(mode>=17 && mode<=20 &&
               lane_starts[mode-17]==(rows+3-(mode-17))/4 &&
               dut.core.lane_row[mode-17]==6'(mode-17) && dut.core.lane_state[mode-17]==1)
                inject_mac_scale(1<<(mode-17));
            if(mode>=21 && !held_start && lane_wait_mask==15) begin
                require(raw_done_mask==0,"hold before genuine HLS completion");
                for(int lane=0;lane<4;lane++) require(lane_starts[lane]==1,"all HLS engines started once before hold");
                held_start=cycles;held_mask=int'(lane_busy_mask);held=1;hold_mac_completions(1);
            end
            if(held && mode==22 && !cancelled && cancel_ready) cancel_valid=1;
            if(held && shim_pending==15 && ((mode==21 && dut.core.error_code==8) ||
               (mode==22 && cancelled && cycles-held_start>=50))) begin
                require(!done_valid && !result_valid,"completion/output before delayed MAC drain");
                held=0;released_at=cycles;hold_mac_completions(0);
            end
`endif
            m_axi_arready=$urandom_range(0,3)!=0;
            if(mode==10 && cycles<30) m_axi_arready=0;
            result_ready=result_wait>=8 && $urandom_range(0,3)!=0;
            done_ready=done_wait>=6 && $urandom_range(0,3)!=0;
            if(!cancelled && cancel_ready && ((mode==9 && dut.state==2) ||
               (mode==10 && m_axi_arvalid) || (mode==11 && act_beats>=4) ||
               (mode==14 && fault_compute_ready))) cancel_valid=1;
            if(!m_axi_rvalid && active && $urandom_range(0,3)!=0 && !(mode==12 && cycles<90)) begin
                m_axi_rdata=memory_word(burst_address+64'(beat*64));m_axi_rresp=0;m_axi_rid=0;
                if(mode==7 && is_activation && act_beats==k-1) m_axi_rresp=2;
                if(mode==8 && is_activation && act_beats==0) m_axi_rid=1;
                if(mode==13 && !is_activation && beat==beats-1) m_axi_rresp=2;
                m_axi_rlast=beat==beats-1;m_axi_rvalid=1;
            end
            @(posedge clk);cycles++;require(cycles<(fixture_mode ? 250000:CASE_BUDGET),"timeout");
            for(int lane=0;lane<LANES;lane++) begin
                lane_starts[lane]+=int'(dut.core.lane_state[lane]==1);
                lane_dones[lane]+=int'(dut.core.lane_state[lane]==2 && dut.core.lane_done[lane]);
                lane_busy[lane]+=int'(lane_busy_mask[lane]);
            end
            if($countones(lane_busy_mask)>peak_busy) peak_busy=$countones(lane_busy_mask);
`ifdef NTRPU_MEMORY_MAC_FAULTS
            pending_seen|=int'(shim_pending);
            for(int lane=0;lane<4;lane++) if(raw_done_mask[lane]) begin
                raw_count[lane]++;
                if(raw_status[lane]!=0) begin
                    $display("NTRPU_MEMORY_FAULT_EVENT id=%0d cycle=%0d raw_error_lane=%0d raw_status=%0d core_error=%0d busy=%0d",case_id,cycles,lane,raw_status[lane],dut.core.error_code,lane_busy_mask);
                    require(mode>=17 && mode<=20 && lane==mode-17 && raw_status[lane]==1,"expected real HLS INVALID_SCALE");
                    require(error_mask==0,"one genuine HLS error");
                    error_mask=1<<lane;error_busy_mask=int'(lane_busy_mask);
                end
            end
            if(dut.core.watchdog_expired && !timeout_at) begin
                timeout_at=cycles;timeout_idle=int'(dut.core.idle_cycles);timeout_mask=int'(lane_busy_mask);
                $display("NTRPU_MEMORY_TIMEOUT_EVENT id=%0d cycle=%0d idle=%0d core_error=%0d busy=%0d raw=%0d pending=%0d",case_id,cycles,timeout_idle,dut.core.error_code,lane_busy_mask,raw_done_mask,shim_pending);
                require(!dut.core.progress_event && timeout_idle==31,"watchdog expires after actual idle interval");
            end
            if(held) require(!done_valid && !result_valid,"held completion must not publish");
`endif
            job_hs=job_valid&&job_ready;tile_hs=tile_valid&&tile_ready;
            ar_hs=m_axi_arvalid&&m_axi_arready;r_hs=m_axi_rvalid&&m_axi_rready;cancel_hs=cancel_valid&&cancel_ready;
            if(cancel_hs) cancel_mask=int'(lane_busy_mask);
            if(ar_hold) require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"AR stability");
            if(result_hold) require(result_valid && {result_data,result_row,result_last,result_sequence,result_epoch}===saved_result,"result stability");
            if(done_hold) require(done_valid && {done_status,done_sequence,done_epoch}===saved_done,"completion stability");
            ar_hold=m_axi_arvalid&&!m_axi_arready;result_hold=result_valid&&!result_ready;done_hold=done_valid&&!done_ready;
            saved_ar={m_axi_araddr,m_axi_arlen};saved_result={result_data,result_row,result_last,result_sequence,result_epoch};
            saved_done={done_status,done_sequence,done_epoch};ar_stalls+=int'(ar_hold);result_stalls+=int'(result_hold);done_stalls+=int'(done_hold);
            if(job_hs) begin require(!accepted,"duplicate job");accepted=1;end
            if(dut.loader_done_valid && dut.loader_status==0) activation_complete=1;
            if(tile_hs) begin
                require(activation_complete,"tile before activation validation");
                active_tile_beats=rows*int'(tile_k_count)*34/64;tile_weight_beats=0;tiles++;
            end
            if(cancel_hs) cancelled=1;
            if(ar_hs) begin
                require(accepted && !active,"unexpected/overlapping AR");
                is_activation=m_axi_araddr>=ACT && m_axi_araddr<ACT+64'h100000;
                if(is_activation) begin
                    require(wb==0,"activation after weights");expected_addr=ACT+4032+64'(act_beats*64);
                    expected_count=k-act_beats;ab++;
                    if(mode==10 || mode==12) require(ab==1,"burst after cancel/timeout");
                end else begin
                    require(activation_complete && tiles>0,"weight access before validated activations/tile");
                    require(!(mode>=3 && mode<=12),"failed activation job touched weights");
                    expected_addr=weight_base+4032+64'((tiles-1)*65536)+64'(tile_weight_beats*64);
`ifdef NTRPU_MEMORY_FIXTURE
                    if(fixture_mode) expected_addr=fixture_tiles[tiles-1][127:64]+64'(tile_weight_beats*64);
`endif
                    expected_count=active_tile_beats-tile_weight_beats;wb++;
                end
                require(m_axi_araddr==expected_addr,"address/high bits/order");
                if(expected_count>(4096-int'(expected_addr[11:0]))/64) expected_count=(4096-int'(expected_addr[11:0]))/64;
                require(int'(m_axi_arlen)+1==expected_count,"burst count/page split");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"AXI attributes");
                active=1;burst_address=m_axi_araddr;beats=int'(m_axi_arlen)+1;beat=0;
            end
            if(r_hs) begin
                require(active,"unexpected R");beat++;
                if(is_activation) act_beats++;else begin weight_beats++;tile_weight_beats++;end
                if(m_axi_rlast) active=0;
            end
            if(result_valid) begin
                result_wait++;require(status==0 && mode!=15 && mode!=16,"partial output on failure/reset");
                require(!cancel_ready,"cancel accepted after result commit");
                require(activation_complete && !active && !dut.loader.activation_valid,"result before drain");
                if(first_result==0) first_result=cycles;
                if(result_ready) begin
                    require(result_row==5'(outputs) && result_last==(outputs==rows-1),"result row/last");
                    require(result_sequence==32'(case_id) && result_epoch==32'(case_id+200),"registered result identity");
                    require(result_data==$realtobits(reference_value(outputs)),"independent integer-dot result");
                    $display("NTRPU_MEMORY_VALUE case=%0d row=%0d value=%0.17g reference=%0.17g",case_id,outputs,$bitstoreal(result_data),reference_value(outputs));
                    outputs++;
                end
            end
            if(done_valid) begin
                require(lane_busy_mask==0,"completion before all MAC lanes drained");
                done_wait++;require(!active && !m_axi_arvalid && !dut.loader.activation_valid,"done before drain");
                require(done_status==4'(status),$sformatf("status %0d expected %0d",done_status,status));
                require(done_sequence==32'(case_id) && done_epoch==32'(case_id+200),"registered completion identity");
                require(outputs==(status==0 ? rows:0),"missing/partial results");
                if(mode>=1 && mode<=4 || mode==9) require(ab==0 && wb==0,"invalid/early cancelled job touched memory");
                if(done_ready) finished=1;
            end
            #1;
            if(job_hs) begin
                job_valid=0;job_layout=0;job_rows=0;job_k_blocks=0;job_sequence=0;job_epoch=0;
                job_region_base=0;job_region_bytes=0;job_activation_addr=0;
                job_activation_region_base=0;job_activation_region_bytes=0;job_activation_epoch=0;
            end
            if(tile_hs) begin
`ifdef NTRPU_MEMORY_FIXTURE
                if(fixture_mode) begin
                    if(tiles==fixture_tile_count) tile_valid=0;
                    else begin
                        tile_addr=fixture_tiles[tiles][127:64];tile_k_base=9'(fixture_tiles[tiles][63:32]);
                        tile_k_count=6'(fixture_tiles[tiles][31:0]);tile_tag++;
                    end
                end else
`endif
                begin
                    if(tiles*32>=k) tile_valid=0;
                    else begin tile_k_base=9'(tiles*32);tile_addr=WEIGHT+4032+64'(tiles*65536);tile_tag++;end
                end
            end
            if(cancel_hs) cancel_valid=0;
            if(r_hs) m_axi_rvalid=0;
            if((mode==15 && act_beats>=4) || (mode==16 && fault_compute_ready)
`ifdef NTRPU_MEMORY_MAC_FAULTS
               || (mode==23 && held && pending_seen==15)
`endif
            ) begin
                reset_mask=int'(lane_busy_mask);
                @(negedge clk);rst_n=0;job_valid=0;tile_valid=0;cancel_valid=0;m_axi_rvalid=0;active=0;
                repeat(4) @(negedge clk);
`ifdef NTRPU_MEMORY_MAC_FAULTS
                require(shim_pending==0,"reset clears every delayed completion");
                held=0;hold_mac_completions(0);
`endif
                rst_n=1;repeat(8) @(negedge clk);
                require(job_ready && !done_valid && !result_valid && !m_axi_arvalid,"stale state after reset");
                finished=1;reset_cases++;
            end
            if(!finished) @(negedge clk);
        end
        $display("NTRPU_MEMORY_CASE id=%0d rows=%0d blocks=%0d scenario=%0d status=%0d cycles=%0d first_result=%0d activation_bursts=%0d weight_bursts=%0d activation_beats=%0d weight_beats=%0d outputs=%0d tiles=%0d",case_id,rows,k,mode,status,cycles,first_result,ab,wb,act_beats,weight_beats,outputs,tiles);
`ifdef NTRPU_MEMORY_MAC_FAULTS
        if(mode>=17 && mode<=20) begin
            require(error_mask==(1<<(mode-17)) && $countones(error_busy_mask)>=2 && peak_busy==4,"real HLS error overlaps other MAC work");
            require(timeout_at==0 && held_mask==0,"unmodified completion for invalid scale");
        end
        if(mode>=21) begin
            require(held_start>0 && held_mask==15 && pending_seen==15 && peak_busy==4,"four genuine completions held");
            for(int lane=0;lane<4;lane++) require(raw_count[lane]==1 && lane_starts[lane]==1,"one real operation per held lane");
            if(mode==21) require(timeout_at>held_start && timeout_mask==15 && released_at>=timeout_at,"timeout precedes releasing all MACs");
            if(mode==22) require(cancel_mask==15 && released_at-held_start>=50,"cancellation drains delayed completions");
            if(mode==23) require(reset_mask==15 && released_at==0,"reset discards pending completions");
        end
        for(int lane=0;lane<4;lane++)
            require(raw_count[lane]==lane_starts[lane] && lane_dones[lane]==(mode==23 ? 0:raw_count[lane]),"no genuine completion lost/duplicated except reset abort");
        $display("NTRPU_MEMORY_FAULT id=%0d error_mask=%0d error_busy=%0d held_start=%0d release=%0d held_mask=%0d pending_seen=%0d timeout_at=%0d timeout_idle=%0d timeout_mask=%0d raw0=%0d raw1=%0d raw2=%0d raw3=%0d",
            case_id,error_mask,error_busy_mask,held_start,released_at,held_mask,pending_seen,timeout_at,timeout_idle,timeout_mask,raw_count[0],raw_count[1],raw_count[2],raw_count[3]);
`endif
`ifdef NTRPU_MEMORY_METRICS
        if(mode==0) for(int lane=0;lane<LANES;lane++)
            require(lane_starts[lane]==((rows+LANES-1-lane)/LANES)*k && lane_dones[lane]==lane_starts[lane],"exact work per MAC lane");
        if(rows>=LANES && mode==14) require(cancel_mask==((1<<LANES)-1),"cancel with all MACs occupied");
        if(rows>=LANES && mode==16) require(reset_mask==((1<<LANES)-1),"reset with all MACs occupied");
        if(rows>=LANES && (mode==14 || mode==16)) begin
            require(peak_busy==LANES,"fault after observing all MACs active at a clock edge");
            for(int lane=0;lane<LANES;lane++) require(lane_starts[lane]>0,"fault after every HLS start edge");
        end
        $display("NTRPU_MEMORY_LANES id=%0d lanes=%0d peak=%0d cancel_mask=%0d reset_mask=%0d starts0=%0d starts1=%0d starts2=%0d starts3=%0d dones0=%0d dones1=%0d dones2=%0d dones3=%0d busy0=%0d busy1=%0d busy2=%0d busy3=%0d",
            case_id,LANES,peak_busy,cancel_mask,reset_mask,lane_starts[0],lane_starts[1],lane_starts[2],lane_starts[3],
            lane_dones[0],lane_dones[1],lane_dones[2],lane_dones[3],lane_busy[0],lane_busy[1],lane_busy[2],lane_busy[3]);
`endif
        total_cycles+=cycles;total_outputs+=outputs;total_ab+=ab;total_wb+=wb;
        tile_valid=0;cancel_valid=0;repeat(3) @(negedge clk);
    endtask
`ifdef NTRPU_MEMORY_FIXTURE
    task automatic run_fixtures;
        int descriptor,image_info,fields,count,id,n,nk,chunk;
        string magic,stem;
        weight_base=64'h100000000;
        descriptor=$fopen({fixture_dir,"/jobs.txt"},"r");require(descriptor!=0,"missing jobs");
        fields=$fscanf(descriptor,"%s %d",magic,count);require(fields==2 && magic=="NTRGEMV02" && count==58,"fixture header");
        for(int job=1;job<=count;job++) begin
            fields=$fscanf(descriptor,"%d %d %d %d",id,n,nk,chunk);
            require(fields==4 && id==job && n>=1 && n<=32 && nk>=1 && nk<=256 && chunk>=1 && chunk<=32,"fixture job");
            stem=$sformatf("%s/job-%03d",fixture_dir,id);
            image_info=$fopen({stem,"-image.txt"},"r");require(image_info!=0,"image header missing");
            fields=$fscanf(image_info,"%d %d",weight_image_bytes,fixture_tile_count);$fclose(image_info);
            require(fields==2 && weight_image_bytes>0 && weight_image_bytes<=524288 && fixture_tile_count==(nk+chunk-1)/chunk,"image dimensions");
            $readmemh({stem,"-image.hex"},weight_image,0,weight_image_bytes-1);
            $readmemh({stem,"-activation-image.hex"},activation_image,0,4032+nk*64-1);
            $readmemh({stem,"-tiles.hex"},fixture_tiles,0,fixture_tile_count-1);
            $readmemh({stem,"-reference.hex"},fixture_reference,0,n-1);
            for(int t=0;t<fixture_tile_count;t++)
                require(fixture_tiles[t][63:32]==32'(t*chunk) && fixture_tiles[t][31:0]==32'(nk-t*chunk<chunk ? nk-t*chunk:chunk),"tile geometry");
            run_case(n,nk,0,0);
        end
        fields=$fscanf(descriptor,"%s",magic);require(fields!=1,"trailing jobs");$fclose(descriptor);
    endtask
`endif
    initial begin
`ifdef NTRPU_MEMORY_FIXTURE
        fixture_mode=$value$plusargs("FIXTURE=%s",fixture_dir);require(fixture_mode,"fixture argument missing");
`endif
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_MEMORY_SEED seed=%0d",seed);random_state=$urandom(seed);
        repeat(4) @(negedge clk);rst_n=1;@(posedge clk);#1;
`ifdef NTRPU_MEMORY_FIXTURE
        if(fixture_mode) run_fixtures();else
`endif
        begin
`ifdef NTRPU_MEMORY_MAC_FAULTS
        run_case(4,32,0,0);
        for(int n=4;n<=5;n++) for(int lane=0;lane<4;lane++) begin
            run_case(n,32,17+lane,6);run_case(4,32,0,0);
        end
        run_case(31,32,17,6);run_case(4,32,0,0);
        run_case(32,32,20,6);run_case(4,32,0,0);
        run_case(4,32,21,8);run_case(4,32,0,0);
        run_case(5,32,21,8);run_case(4,32,0,0);
        run_case(32,32,21,8);run_case(4,32,0,0);
        for(int n=4;n<=5;n++) begin run_case(n,32,22,7);run_case(4,32,0,0);end
        for(int n=4;n<=5;n++) begin run_case(n,32,23,0);run_case(4,32,0,0);end
`else
        run_case(2,32,0,0);run_case(2,128,0,0);
        run_case(0,32,1,1);run_case(2,32,2,1);run_case(2,32,3,2);run_case(2,32,4,2);
        run_case(2,32,5,2);run_case(2,32,6,2);run_case(2,32,7,3);run_case(2,32,8,3);
        run_case(2,32,9,7);run_case(2,32,10,7);run_case(2,128,11,7);run_case(2,32,12,8);
        run_case(2,32,13,3);run_case(2,32,14,7);run_case(1,64,0,0);
        run_case(2,128,15,0);run_case(2,32,16,0);run_case(2,32,0,0);
`ifdef NTRPU_MEMORY_EXTENDED
        run_case(1,32,0,0);run_case(3,32,0,0);run_case(4,32,0,0);
        run_case(5,32,0,0);run_case(31,32,0,0);run_case(32,32,0,0);
        run_case(3,64,0,0);run_case(4,64,0,0);run_case(5,64,0,0);
        run_case(31,64,0,0);run_case(32,64,0,0);
        run_case(4,32,13,3);run_case(5,32,13,3);run_case(31,32,13,3);run_case(32,32,13,3);
        run_case(4,32,14,7);run_case(5,32,14,7);run_case(31,32,14,7);run_case(32,32,14,7);
        run_case(4,32,16,0);run_case(5,32,16,0);
`endif
`endif
        end
`ifdef NTRPU_MEMORY_EXTENDED
        require(!fixture_mode && reset_cases==4,"extended reset coverage");
`else
        require(ar_stalls>0 && result_stalls>0 && done_stalls>0 && reset_cases==(fixture_mode ? 0:2),"coverage counters");
`endif
        require(ar_stalls>0 && result_stalls>0 && done_stalls>0,"handshake pause coverage");
`ifdef NTRPU_QUAD_BLOCK_FIFO
        require(fifo_count==0 && fifo_pushes==fifo_pops+fifo_dropped,"FIFO final conservation including reset discards");
        require(fifo_peak==2 && fifo_full_stalls>0 && fifo_both>0,"FIFO occupancy/backpressure/simultaneous coverage");
        $display("NTRPU_BLOCK_FIFO pushes=%0d pops=%0d dropped=%0d peak=%0d full_stalls=%0d simultaneous=%0d pending=%0d",
            fifo_pushes,fifo_pops,fifo_dropped,fifo_peak,fifo_full_stalls,fifo_both,fifo_count);
`endif
        $display("NTRPU_MEMORY_PASS cases=%0d cycles=%0d outputs=%0d activation_bursts=%0d weight_bursts=%0d ar_stalls=%0d result_stalls=%0d done_stalls=%0d resets=%0d",case_id,total_cycles,total_outputs,total_ab,total_wb,ar_stalls,result_stalls,done_stalls,reset_cases);
        $finish;
    end
endmodule
