`timescale 1ns/1ps
module tb_ntrpu_q8_gemv_stream;
    logic clk=0, rst_n=0;
    always #2.5 clk=~clk;
    logic job_valid=0, job_ready;
    logic [5:0] job_rows=0;
    logic [8:0] job_k_blocks=0;
    logic [31:0] job_sequence=0, job_epoch=0;
    logic [63:0] job_region_base=0, job_region_bytes=0;
    logic activation_valid=0, activation_ready;
    logic [7:0] activation_index=0;
    logic [255:0] activation_values=0;
    logic [63:0] activation_scale=0;
    logic [31:0] activation_epoch=0;
    logic tile_valid=0, tile_ready;
    logic [63:0] tile_addr=0;
    logic [8:0] tile_k_base=0;
    logic [5:0] tile_k_count=0;
    logic [31:0] tile_tag=0, tile_epoch=0;
    logic cancel_valid=0, cancel_ready;
    logic result_valid, result_ready=0, result_last;
    logic [63:0] result_data;
    logic [4:0] result_row;
    logic [31:0] result_sequence, result_epoch;
    logic done_valid, done_ready=0;
    logic [3:0] done_status;
    logic [31:0] done_sequence, done_epoch;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize, m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid, m_axi_arcache;
    logic m_axi_arlock, m_axi_arvalid, m_axi_arready=0;
    logic [511:0] m_axi_rdata=0;
    logic [1:0] m_axi_rresp=0;
    logic [3:0] m_axi_rid=0;
    logic m_axi_rlast=0, m_axi_rvalid=0, m_axi_rready;
    logic [1:0] job_layout=2;
`ifdef NTRPU_DUAL
    localparam bit BLOCK_MAJOR=1;
    ntrpu_q8_gemv_parallel #(.WATCHDOG_CYCLES(96)) dut (.*);
    wire [1:0] monitor_busy={dut.lane_state[1]!=0,dut.lane_state[0]!=0};
    wire [1:0] monitor_done=dut.lane_done;
`else
    localparam bit BLOCK_MAJOR=0;
    ntrpu_q8_gemv_stream #(.WATCHDOG_CYCLES(96)) dut (.*);
    wire [1:0] monitor_busy={1'b0,dut.mac_state!=0};
    wire [1:0] monitor_done={1'b0,dut.mac_done};
`endif
    int phase_cycles[0:5], lane_busy[0:1];
    int q8_stalls, ar_stalls, r_stalls, r_waits, r_beats, block_ops, activation_writes, both_busy;

    localparam logic [63:0] REGION=64'h100000000;
    int mode=0, case_id=0, rows=0, total_k=0, expected_status=0;
    int cycles=0, case_start=0, case_count=0, bursts=0, mac_count=0, outputs=0;
    int tile_base=0, tile_count=0, tile_size=0, issued=0, returned=0;
    int burst_first=0, burst_size=0, beat=0, hold_memory=0;
    int result_wait=0, done_wait=0, result_stalls=0, completion_stalls=0;
    int seed=1, random_state, first_result=0;
    bit fixture_mode=0;
    bit maximum_shapes=0;
    string fixture_dir;
    int fixture_tile_k=32;
    logic [7:0] fixture_weights [0:278527];
    logic [255:0] fixture_activations [0:255];
    logic [63:0] fixture_scales [0:255], fixture_reference [0:31];
`ifdef NTRPU_PACKED
    logic [7:0] fixture_image [0:524287];
    logic [127:0] fixture_tiles [0:255];
    int fixture_image_bytes=0, fixture_tile_count=0;
`endif
    logic [63:0] memory_base;
    real expected [0:31];
    bit memory_active=0, r_accepted=0, finished=0, cancellation_sent=0;
    bit job_hs, activation_hs, tile_hs, forced_block=0, forced_decode=0, forced_lane_error=0;
    bit lane_fault_injected=0;
    bit output_stalled=0, done_stalled=0, ar_stalled=0;
    logic [133:0] saved_result;
    logic [67:0] saved_done;
    logic [71:0] saved_ar;

    task automatic require(input bit ok,input string why);
        if (!ok) $fatal(1,"FAIL GEMV case=%0d mode=%0d cycle=%0d: %s",case_id,mode,cycles,why);
    endtask
    function automatic int activation(input int k,input int lane);
        if (mode==24) return lane==0 ? 1:0;
        if (fixture_mode) return int'($signed(fixture_activations[k][lane*8+:8]));
        return ((k*13+lane*7+case_id*3)&255)-128;
    endfunction
    function automatic int weight(input int r,input int k,input int lane);
        if (mode==24) return lane==0 && (k==0 || k==32 || k==33) ? 1:0;
        if (fixture_mode) return int'($signed(fixture_weights[(r*total_k+k)*34+2+lane]));
        return ((r*29+k*11+lane*17+case_id)&255)-128;
    endfunction
    function automatic logic [15:0] scale_bits(input int r,input int k);
        if (mode==24) return k==33 ? 16'hbc00:16'h3c00;
        if (fixture_mode) return {fixture_weights[(r*total_k+k)*34+1],fixture_weights[(r*total_k+k)*34]};
        case ((r+k)%6)
            0: return 16'h3c00;
            1: return 16'hb800;
            2: return 16'h0001;
            3: return 16'h03ff;
            4: return 16'h4200;
            default: return 16'hbc01;
        endcase
    endfunction
    function automatic real half_value(input logic [15:0] b);
        int e,f; real magnitude;
        e=int'(b[14:10]); f=int'(b[9:0]);
        magnitude=e==0 ? real'(f)*(2.0**(-24)) : real'(1024+f)*(2.0**(e-25));
        return b[15] ? -magnitude : magnitude;
    endfunction
    function automatic real input_scale(input int block);
        // Pin the IEEE binary64 bits: this XSIM build parsed the large
        // decimal literal for 2^53 two units low in the isolated probe.
        if (mode==24) return (block==0 || block==33) ? $bitstoreal(64'h4340000000000000):1.0;
        return fixture_mode ? $bitstoreal(fixture_scales[block]):1.0/real'(block%5+1);
    endfunction
    function automatic logic [511:0] memory_word(input int index);
        int offset,block_number,r,k,position; logic [15:0] s;
`ifdef NTRPU_PACKED
        offset=int'(memory_base-REGION)+index*64;
        if (offset<0 || offset+64>fixture_image_bytes) $fatal(1,"AXI read outside immutable packed image");
        for (int j=0;j<64;j++) memory_word[j*8+:8]=fixture_image[offset+j];
        return memory_word;
`endif
        for (int j=0;j<64;j++) begin
            offset=index*64+j; block_number=offset/34; position=offset%34;
            if (BLOCK_MAJOR) begin r=block_number%rows; k=tile_base+block_number/rows; end
            else begin r=block_number/tile_count; k=tile_base+block_number%tile_count; end
            s=scale_bits(r,k);
            if (mode==2 && r==rows-1 && k==total_k-1) s=16'h7e00;
            if (position==0) memory_word[j*8+:8]=s[7:0];
            else if (position==1) memory_word[j*8+:8]=s[15:8];
            else memory_word[j*8+:8]=8'(weight(r,k,position-2));
        end
    endfunction

    task automatic step;
        real actual,delta,tolerance;
        @(negedge clk);
        m_axi_arready=$urandom_range(0,3)!=0;
        result_ready=result_wait>=7 && $urandom_range(0,3)!=0;
        done_ready=done_wait>=5 && $urandom_range(0,2)!=0;
        if (r_accepted) m_axi_rvalid=0;
        if (hold_memory>0) hold_memory--;
        if (!m_axi_rvalid && memory_active && hold_memory==0 && $urandom_range(0,3)!=0) begin
            m_axi_rvalid=1; m_axi_rdata=memory_word(burst_first+beat);
            m_axi_rresp=(mode==1 && tile_base+tile_count==total_k && burst_first+beat==tile_size/64-1)?2:0;
            m_axi_rid=(mode==11 && burst_first+beat==tile_size/64-1)?1:0;
            m_axi_rlast=beat==burst_size-1;
        end
        if (mode==9 && !cancellation_sent && cancel_ready && dut.mac_state==2 &&
            dut.computed_blocks>=2) cancel_valid=1;
        if (mode==14 && result_valid) begin
            cancel_valid=1; require(!cancel_ready,"cancel accepted after publication committed");
        end
        if (mode==15 && dut.mac_state!=0 && dut.computed_blocks>=1 && !finished) begin
            job_valid=1; job_sequence=32'hffffffff;
            activation_valid=1; activation_scale=64'h7ff8000000000000;
            require(!job_ready && !activation_ready,"busy command overwrote job or activations");
        end
        if (mode==12 && dut.block_valid && dut.received_blocks==7) begin
            force dut.block_tag=32'hdeadbeef; forced_block=1;
        end
`ifdef NTRPU_DUAL
        if (mode==23 && !lane_fault_injected && dut.lane_state[1]==1) begin
            require(&monitor_busy,"lane1 error did not overlap both workers");
            force dut.workers[1].mac.input_scale=64'h7ff8000000000000;
            forced_lane_error=1; lane_fault_injected=1;
        end
`endif
        if (mode==13 && dut.decode_done_valid) begin
            force dut.decode_done_tag=32'hdeadbeef; forced_decode=1;
        end
        @(posedge clk);
        cycles++; require(cycles<((fixture_mode || maximum_shapes) ? 2000000:300000),"suite timeout");
        case (dut.state)
            0: phase_cycles[0]++;
            1: phase_cycles[1]++;
            2,3,4,5: phase_cycles[2]++;
            6: phase_cycles[3]++;
            7: phase_cycles[4]++;
            8: phase_cycles[5]++;
            default: require(0,"unknown controller phase");
        endcase
        lane_busy[0]+=int'(monitor_busy[0]); lane_busy[1]+=int'(monitor_busy[1]);
        if (&monitor_busy) both_busy++;
        if (dut.block_valid && !dut.block_ready) q8_stalls++;
        if (m_axi_arvalid && !m_axi_arready) ar_stalls++;
        if (m_axi_rvalid && !m_axi_rready) r_stalls++;
        if (memory_active && !m_axi_rvalid) r_waits++;
        if (m_axi_rvalid && m_axi_rready) r_beats++;
        if (dut.block_fire && dut.error_now==0) block_ops++;
        if (activation_valid && activation_ready) activation_writes++;
        job_hs=job_valid&&job_ready;
        activation_hs=activation_valid&&activation_ready;
        tile_hs=tile_valid&&tile_ready;
        if (output_stalled) require(result_valid &&
            {result_data,result_row,result_last,result_sequence,result_epoch}===saved_result,"result stability");
        if (done_stalled) require(done_valid && {done_status,done_sequence,done_epoch}===saved_done,"done stability");
        if (ar_stalled) require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"AR stability");
        output_stalled=result_valid&&!result_ready;
        done_stalled=done_valid&&!done_ready;
        ar_stalled=m_axi_arvalid&&!m_axi_arready;
        saved_result={result_data,result_row,result_last,result_sequence,result_epoch};
        saved_done={done_status,done_sequence,done_epoch}; saved_ar={m_axi_araddr,m_axi_arlen};
        if (output_stalled) result_stalls++;
        if (done_stalled) completion_stalls++;
        if (result_valid) begin
            require(expected_status==0,"partial result published for a failed job");
            require(tile_base+tile_count==total_k && returned==tile_size/64,"result before final memory completion");
            require(result_sequence==32'(case_id) && result_epoch==32'(100+case_id),"result identity");
            require(int'(result_row)==outputs && result_last==(outputs==rows-1),"result row/last");
            actual=$bitstoreal(result_data); delta=actual-expected[outputs]; if (delta<0) delta=-delta;
            tolerance=expected[outputs]; if (tolerance<0) tolerance=-tolerance;
            if (tolerance<1.0) tolerance=1.0;
            require(delta<=tolerance*1.0e-12,$sformatf("numerical mismatch row=%0d actual=%g expected=%g",outputs,actual,expected[outputs]));
            result_wait++;
            if (first_result==0) first_result=cycles;
            if (result_ready) begin
                if (fixture_mode || maximum_shapes) $display("NTRPU_GEMV_VALUE id=%0d row=%0d bits=%016h",case_id,outputs,result_data);
                outputs++;
            end
        end
        if (done_valid) begin
            done_wait++;
            require(done_status==4'(expected_status),$sformatf("status actual=%0d expected=%0d",done_status,expected_status));
            require(done_sequence==32'(case_id) && done_epoch==32'(100+case_id),"done identity");
            require(!memory_active,"completion before draining issued AXI burst");
            require(outputs==(expected_status==0 ? rows : 0),"wrong output count");
            if (done_ready) finished=1;
        end
        if (cancel_valid && cancel_ready) begin
            if (BLOCK_MAJOR && mode==9) require(&monitor_busy,"cancel did not overlap both workers");
            cancellation_sent=1;
        end
        mac_count+=int'(monitor_done[0])+int'(monitor_done[1]);
        if (m_axi_arvalid && m_axi_arready) begin
            require(!memory_active,"two outstanding bursts");
            require(m_axi_araddr==memory_base+64'(issued*64),"address order/high bits");
            require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0,"AXI properties");
            require(int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"4KiB boundary");
            burst_first=issued; burst_size=int'(m_axi_arlen)+1;
            issued+=burst_size; beat=0; memory_active=1; bursts++;
            if (mode==10) hold_memory=112;
        end
        r_accepted=m_axi_rvalid&&m_axi_rready;
        if (r_accepted) begin
            require(memory_active,"unsolicited response");
            beat++; returned++; if (beat==burst_size) memory_active=0;
        end
        #1;
        if (cancellation_sent) cancel_valid=0;
        if (forced_block) begin release dut.block_tag; forced_block=0; end
        if (forced_decode) begin release dut.decode_done_tag; forced_decode=0; end
`ifdef NTRPU_DUAL
        if (forced_lane_error && dut.lane_state[1]==0) begin
            release dut.workers[1].mac.input_scale; forced_lane_error=0;
        end
`endif
        if (finished) begin job_valid=0; activation_valid=0; cancel_valid=0; end
    endtask

    task automatic run_case(input int id,input int n,input int k,input int scenario,input int status);
        int before_bursts,dot; real sum,term,tile_sum,regrouped;
        case_id=id; rows=n; total_k=k; mode=scenario; expected_status=status;
        finished=0; outputs=0; first_result=0; result_wait=0; done_wait=0; cancellation_sent=0;
        before_bursts=bursts; case_start=cycles;
        for (int i=0;i<6;i++) phase_cycles[i]=0;
        lane_busy[0]=0; lane_busy[1]=0;
        q8_stalls=0; ar_stalls=0; r_stalls=0; r_waits=0; r_beats=0;
        block_ops=0; activation_writes=0; both_busy=0; lane_fault_injected=0;
        require(job_ready && !memory_active,"previous job did not release resources");
        for (int r=0;r<n;r++) begin
            sum=0; tile_sum=0; regrouped=0;
            for (int block=0;block<k;block++) begin
                dot=0;
                for (int lane=0;lane<32;lane++) dot+=weight(r,block,lane)*activation(block,lane);
                term=(half_value(scale_bits(r,block))*input_scale(block))*real'(dot);
                sum+=term;
                if (scenario==24) begin
                    tile_sum+=term;
                    if ((block+1)%fixture_tile_k==0 || block==k-1) begin
                        regrouped+=tile_sum; tile_sum=0;
                    end
                end
            end
            expected[r]=fixture_mode ? $bitstoreal(fixture_reference[r]):sum;
            if (scenario==24) begin
                require(expected[r]==0.0,"non-associative reference must be zero");
                require(regrouped==1.0,"tile-subtotal oracle must differ from ordered accumulation");
            end
        end
        job_layout=scenario==22 ? 1:2;
        job_rows=scenario==20 ? 0 : 6'(n); job_k_blocks=9'(k);
        job_sequence=32'(id); job_epoch=32'(100+id);
        job_region_base=scenario==21 ? 64'hfffffffffffffff0 : REGION;
        job_region_bytes=64'h200000;
        job_valid=1; do step(); while(!job_hs); job_valid=0;
        if (scenario<20 || scenario>=23) begin
            for (int block=0;block<k;block++) begin
                if (scenario==5 && block==k-1) continue;
                activation_index=8'(block); activation_epoch=32'(100+id+(scenario==6 ? 1:0));
                for (int lane=0;lane<32;lane++) activation_values[lane*8+:8]=8'(activation(block,lane));
                activation_scale=(scenario==3 && block==k-1) ? 64'h7ff8000000000000 : $realtobits(input_scale(block));
                activation_valid=1; do step(); while(!activation_hs); activation_valid=0;
                if (done_valid) break;
            end
            if (!done_valid) begin
                for (int base=0;base<k;base+=fixture_tile_k) begin
                    tile_base=base; tile_count=k-base<fixture_tile_k ? k-base:fixture_tile_k; tile_size=n*tile_count*34;
                    // Distinct immutable weight region per K tile, also when
                    // fixture tiles contain fewer than 32 K blocks.
                    issued=0; returned=0; memory_base=REGION+64'hfc0+64'(base/fixture_tile_k)*64'h10000;
`ifdef NTRPU_PACKED
                    require(base/fixture_tile_k<fixture_tile_count,"missing packed tile descriptor");
                    memory_base=fixture_tiles[base/fixture_tile_k][127:64];
                    require(fixture_tiles[base/fixture_tile_k][63:32]==32'(base) &&
                            fixture_tiles[base/fixture_tile_k][31:0]==32'(tile_count),"packed tile geometry/order");
                    require(memory_base>=REGION && memory_base+64'(tile_size)<=REGION+64'(fixture_image_bytes),"packed descriptor image range");
`endif
                    tile_addr=scenario==4 ? REGION+64'h200000 : memory_base;
                    tile_k_base=9'(base+(scenario==8 ? 1:0)); tile_k_count=6'(tile_count);
                    tile_epoch=32'(100+id+(scenario==7 ? 1:0)); tile_tag=32'(id*1000+base);
                    tile_valid=1; do step(); while(!tile_hs); tile_valid=0;
                    while (!tile_ready && !done_valid && !result_valid) step();
                    if (done_valid || result_valid) break;
                end
            end
        end
        while (!finished) step();
        require(cycles-case_start<((maximum_shapes || fixture_mode) ? 250000:60000),"job exceeded cycle budget");
        if (scenario inside {4,5,6,7,8,20,21,22}) require(bursts==before_bursts,"invalid command issued AXI");
        $display("NTRPU_GEMV_CASE id=%0d rows=%0d columns=%0d scenario=%0d status=%0d cycles=%0d first_result=%0d bursts=%0d outputs=%0d",
                 id,n,k*32,scenario,status,cycles-case_start,first_result==0 ? 0:first_result-case_start,bursts-before_bursts,outputs);
        require(phase_cycles.sum()==cycles-case_start,"cycle partition");
        if (scenario==24) $display("NTRPU_GEMV_NONASSOCIATIVE_PASS rows=2 outputs=2 value=0 regrouped=1");
        $display("NTRPU_GEMV_METRIC id=%0d idle=%0d prepare=%0d descriptor=%0d run=%0d results=%0d complete=%0d lane0_busy=%0d lane1_busy=%0d both_busy=%0d q8_stall=%0d ar_stall=%0d r_stall=%0d r_wait=%0d r_beats=%0d block_ops=%0d activation_writes=%0d",
                 id,phase_cycles[0],phase_cycles[1],phase_cycles[2],phase_cycles[3],phase_cycles[4],phase_cycles[5],
                 lane_busy[0],lane_busy[1],both_busy,q8_stalls,ar_stalls,r_stalls,r_waits,r_beats,block_ops,activation_writes);
        case_count++;
        repeat (3) step();
    endtask
    task automatic reset_during_work(input int phase);
        bit during_mac;
        during_mac=phase!=0;
        case_id=phase+23; mode=0; rows=phase==2 ? 2:1; total_k=32; expected_status=99;
        finished=0; outputs=0; result_wait=0; done_wait=0; cancellation_sent=0;
        require(job_ready && !memory_active,"reset probe began while busy");
        job_rows=6'(rows); job_k_blocks=32; job_layout=2; job_sequence=32'(case_id); job_epoch=32'(100+case_id);
        job_region_base=REGION; job_region_bytes=64'h200000;
        job_valid=1; do step(); while(!job_hs); job_valid=0;
        for (int block=0;block<32;block++) begin
            activation_index=8'(block); activation_epoch=job_epoch; activation_scale=$realtobits(1.0);
            for (int lane=0;lane<32;lane++) activation_values[lane*8+:8]=8'(activation(block,lane));
            activation_valid=1; do step(); while(!activation_hs); activation_valid=0;
        end
        tile_base=0; tile_count=32; tile_size=rows*1088; issued=0; returned=0;
        memory_base=REGION+64'hfc0; tile_addr=memory_base;
        tile_k_base=0; tile_k_count=32; tile_epoch=job_epoch; tile_tag=32'(case_id*1000);
        tile_valid=1; do step(); while(!tile_hs); tile_valid=0;
        if (during_mac) begin
            while (dut.mac_state!=2 || dut.computed_blocks<2 || (phase==2 && !(&monitor_busy))) step();
            require(!memory_active,"MAC reset probe unexpectedly has AXI outstanding");
        end else begin
            while (!memory_active || returned<1) step();
        end
        require(!result_valid && !done_valid && outputs==0,"reset probe published early");
        // This explicitly models a system-level reset of BOTH AXI endpoints.
        @(negedge clk); rst_n=0; m_axi_rvalid=0; memory_active=0; r_accepted=0;
        job_valid=0; activation_valid=0; tile_valid=0; cancel_valid=0;
        output_stalled=0; done_stalled=0; ar_stalled=0; hold_memory=0;
        repeat (4) @(negedge clk);
        rst_n=1; @(posedge clk); #1;
        repeat (8) step();
        require(job_ready && !result_valid && !done_valid,"stale result/completion after coordinated reset");
        if (phase==2) $display("NTRPU_GEMV_RESET_PASS phase=BOTH outputs=0");
        else if (during_mac) $display("NTRPU_GEMV_RESET_PASS phase=MAC outputs=0");
        else $display("NTRPU_GEMV_RESET_PASS phase=AXI outputs=0");
        case_count++;
    endtask
    task automatic run_fixtures;
        int descriptor,jobs,id,n,k,chunk,fields;
        string magic,stem;
        descriptor=$fopen({fixture_dir,"/jobs.txt"},"r");
        require(descriptor!=0,"cannot open fixture descriptor");
        fields=$fscanf(descriptor,"%s %d\n",magic,jobs);
`ifdef NTRPU_PACKED
        require(fields==2 && magic=="NTRGEMV02" && jobs==58 && BLOCK_MAJOR,"invalid packed fixture header/layout");
`else
        require(fields==2 && magic=="NTRGEMV01" && jobs==54,"invalid fixture header");
`endif
        for (int j=0;j<jobs;j++) begin
            fields=$fscanf(descriptor,"%d %d %d %d\n",id,n,k,chunk);
            require(fields==4 && id==j+1 && n>0 && n<=32 && k>0 && k<=256 &&
                    chunk>0 && chunk<=32 && (n*chunk)%32==0 && (n*(k%chunk))%32==0,"invalid fixture job");
            stem=$sformatf("%s/job-%03d",fixture_dir,id);
            $readmemh({stem,"-weights.hex"},fixture_weights,0,n*k*34-1);
            $readmemh({stem,"-activations.hex"},fixture_activations,0,k-1);
            $readmemh({stem,"-scales.hex"},fixture_scales,0,k-1);
            $readmemh({stem,"-reference.hex"},fixture_reference,0,n-1);
`ifdef NTRPU_PACKED
            begin
                int packed_descriptor;
                packed_descriptor=$fopen({stem,"-image.txt"},"r");
                require(packed_descriptor!=0,"missing packed image descriptor");
                fields=$fscanf(packed_descriptor,"%d %d\n",fixture_image_bytes,fixture_tile_count);
                require(fields==2 && fixture_image_bytes>0 && fixture_image_bytes<=524288 &&
                        fixture_tile_count==(k+chunk-1)/chunk,"packed image size/tile count");
                fields=$fscanf(packed_descriptor,"%s",magic); require(fields!=1,"trailing image descriptor content");
                $fclose(packed_descriptor);
            end
            $readmemh({stem,"-image.hex"},fixture_image,0,fixture_image_bytes-1);
            $readmemh({stem,"-tiles.hex"},fixture_tiles,0,fixture_tile_count-1);
`endif
            fixture_tile_k=chunk;
            run_case(id,n,k,0,0);
        end
        fields=$fscanf(descriptor,"%s",magic); require(fields!=1,"trailing fixture descriptor content");
        $fclose(descriptor);
        require(result_stalls>0 && completion_stalls>0,"fixture backpressure coverage");
        $display("NTRPU_GEMV_FIXTURE_PASS cases=%0d bursts=%0d mac_ops=%0d result_stalls=%0d completion_stalls=%0d cycles=%0d",
                 case_count,bursts,mac_count,result_stalls,completion_stalls,cycles);
    endtask
    initial begin
        $display("NTRPU_GEMV_VARIANT lanes=%0d layout=%0d",BLOCK_MAJOR ? 2:1,BLOCK_MAJOR ? 2:1);
        fixture_mode=$value$plusargs("FIXTURE=%s",fixture_dir);
        maximum_shapes=$test$plusargs("MAX_SHAPES");
        require(!(fixture_mode && maximum_shapes),"conflicting workloads");
        if ($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_GEMV_SEED seed=%0d",seed);
        random_state=$urandom(seed);
        repeat (4) @(negedge clk); rst_n=1; @(posedge clk); #1;
        if (maximum_shapes) begin
            run_case(1,32,64,0,0);
            run_case(2,32,256,0,0);
            run_case(3,1,256,0,0);
            run_case(4,31,64,0,0);
            require(result_stalls>0 && completion_stalls>0,"maximum-shape backpressure coverage");
            $display("NTRPU_GEMV_MAX_PASS cases=%0d bursts=%0d mac_ops=%0d result_stalls=%0d completion_stalls=%0d cycles=%0d",
                     case_count,bursts,mac_count,result_stalls,completion_stalls,cycles);
        end else if (fixture_mode) begin
            run_fixtures();
        end else begin
        run_case(1,32,1,0,0);
        run_case(2,3,64,0,0);
        run_case(3,2,256,0,0);
        run_case(4,32,32,0,0);
        run_case(5,2,64,1,3);
        run_case(6,2,64,2,4);
        run_case(7,2,64,3,6);
        run_case(8,1,32,4,1);
        run_case(9,1,32,5,2);
        run_case(10,1,32,6,2);
        run_case(11,1,32,7,1);
        run_case(12,1,32,8,1);
        run_case(13,2,32,9,7);
        run_case(14,1,32,10,8);
        run_case(15,1,32,11,3);
        run_case(16,1,32,12,5);
        run_case(17,1,32,13,4);
        run_case(18,3,64,14,0);
        run_case(19,2,64,15,0);
        run_case(20,1,32,20,1);
        run_case(21,1,32,21,1);
        reset_during_work(0);
        reset_during_work(1);
        run_case(22,1,32,0,0);
        if (BLOCK_MAJOR) begin
            run_case(23,1,32,22,1);
            run_case(24,2,64,23,6);
            run_case(25,2,64,24,0);
            reset_during_work(2);
            run_case(26,1,32,0,0);
        end
        require(result_stalls>0 && completion_stalls>0,"backpressure coverage");
        $display("NTRPU_GEMV_STREAM_PASS cases=%0d bursts=%0d mac_ops=%0d result_stalls=%0d completion_stalls=%0d cycles=%0d",
                 case_count,bursts,mac_count,result_stalls,completion_stalls,cycles);
        end
        $finish;
    end
endmodule
