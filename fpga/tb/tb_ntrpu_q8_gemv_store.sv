`timescale 1ns/1ps
module tb_ntrpu_q8_gemv_store;
    logic clk=0,rst_n=0;
    logic job_output_fp32=0;
    always #2.5 clk=~clk;
    logic job_valid=0,job_ready,tile_valid=0,tile_ready,cancel_valid=0,cancel_ready;
    logic [1:0] job_layout=2;
    logic [5:0] job_rows=0;
    logic [8:0] job_k_blocks=0,tile_k_base=0;
    logic [31:0] job_sequence=0,job_epoch=0,job_activation_epoch=0,tile_tag=0,tile_epoch=0;
    logic [63:0] job_region_base=0,job_region_bytes=0,job_activation_addr=0;
    logic [63:0] job_activation_region_base=0,job_activation_region_bytes=0,tile_addr=0;
    logic [5:0] tile_k_count=32;
    logic result_valid,result_ready,result_last,done_valid,done_ready=0;
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
    logic [63:0] job_output_addr=0,job_output_region_base=0,job_output_region_bytes=0;
    logic [31:0] job_output_epoch=0;
    logic [8:0] done_bytes;
    logic [63:0] m_axi_awaddr,m_axi_wstrb;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize,m_axi_awprot;
    logic [1:0] m_axi_awburst,m_axi_bresp;
    logic [3:0] m_axi_awid,m_axi_awcache,m_axi_bid;
    logic m_axi_awlock,m_axi_awvalid,m_axi_awready,m_axi_wlast,m_axi_wvalid,m_axi_wready,m_axi_bvalid,m_axi_bready;
    logic [511:0] m_axi_wdata;
    logic [2047:0] expected_output=0;
    int write_aw,write_w,write_b,write_bytes,write_aw_stalls,write_w_stalls;
    assign result_valid=dut.result_valid;
    assign result_ready=dut.result_ready;
    assign result_data=dut.result_data;
    assign result_row=dut.result_row;
    assign result_last=dut.result_last;
    assign result_sequence=dut.result_sequence;
    assign result_epoch=dut.result_epoch;
    int seed=1,random_state,case_id=0,total_cycles=0,total_outputs=0,total_ab=0,total_wb=0;
    int act_beats,weight_beats,outputs,tiles,cycles,rows,k,mode;
    ntrpu_result_memory_model output_memory (
        .clk,.rst_n,.start(job_valid && job_ready),.rows(job_rows),.address(job_output_addr),
        .identity(case_id),.scenario(mode),.expected_output,.commit_seen(dut.engine_done_valid),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,
        .m_axi_awlock,.m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,.m_axi_wready,
        .m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready,
        .aw_count(write_aw),.w_count(write_w),.b_count(write_b),.byte_count(write_bytes),
        .aw_stalls(write_aw_stalls),.w_stalls(write_w_stalls)
    );
    ntrpu_q8_gemv_store #(.WATCHDOG_CYCLES(32)) dut(.*);
    localparam logic [63:0] OUT=64'ha000000000;
    localparam logic [63:0] ACT=64'h8000000000,WEIGHT=64'h9000000000;
    int ar_stalls=0,result_stalls=0,done_stalls=0,reset_cases=0;
    bit activation_complete;
    bit fixture_mode=0;
    string fixture_dir;
    logic [63:0] weight_base=WEIGHT;
    int tile_weight_beats,active_tile_beats,fixture_chunk=32;
`ifdef NTRPU_STORE_FIXTURE
    logic [7:0] weight_image [0:524287],activation_image [0:32767];
    logic [127:0] fixture_tiles [0:255];
    logic [63:0] fixture_reference [0:31];
    int weight_image_bytes,fixture_tile_count;
`endif
    function automatic void require(input bit ok,input string message);
        if(!ok) $fatal(1,"FAIL MEMORY case=%0d cycle=%0d: %s",case_id,cycles,message);
    endfunction
    function automatic logic [511:0] memory_word(input logic [63:0] addr);
        int off,block_id,lane,row,global_k;
        memory_word=0;
`ifdef NTRPU_STORE_FIXTURE
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
                else if(lane==1) memory_word[j*8+:8]=8'h3c;
                else memory_word[j*8+:8]=8'((global_k+row*3+lane-2)%7-3);
            end
        end
    endfunction
    function automatic real reference_value(input int row);
        int sum;
`ifdef NTRPU_STORE_FIXTURE
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
        logic [76:0] saved_done;
        logic [63:0] burst_address,expected_addr;
        int beat,beats,ab,wb,done_wait,first_result,result_wait,expected_count;
        int reset_aw,reset_w,reset_b,reset_bytes,reset_aw_stalls,reset_w_stalls;
        case_id++;rows=nrows;k=nk;mode=scenario;cycles=0;outputs=0;tiles=0;act_beats=0;weight_beats=0;
        active=0;accepted=0;finished=0;cancelled=0;activation_complete=0;tile_weight_beats=0;active_tile_beats=0;
        ar_hold=0;result_hold=0;done_hold=0;ab=0;wb=0;done_wait=0;first_result=0;result_wait=0;
        reset_aw=0;reset_w=0;reset_b=0;reset_bytes=0;reset_aw_stalls=0;reset_w_stalls=0;
        @(negedge clk);require(job_ready,"previous job not released");
        job_layout=mode==2 ? 1:2;job_rows=6'(rows);job_k_blocks=9'(k);
        job_sequence=32'(case_id);job_epoch=32'(case_id+200);job_activation_epoch=job_epoch;
        job_region_base=weight_base;job_region_bytes=64'h200000;
        job_activation_addr=ACT+4032;job_activation_region_base=ACT;job_activation_region_bytes=64'h100000;
        if(mode==3) job_activation_epoch++;
        if(mode==4) job_activation_region_bytes=4096;
        tile_addr=WEIGHT+4032;tile_k_base=0;tile_k_count=32;tile_tag=1000;tile_epoch=job_epoch;
`ifdef NTRPU_STORE_FIXTURE
        if(fixture_mode) begin
            tile_addr=fixture_tiles[0][127:64];tile_k_base=9'(fixture_tiles[0][63:32]);tile_k_count=6'(fixture_tiles[0][31:0]);
        end
`endif
        job_output_addr=OUT+4032;job_output_region_base=OUT;job_output_region_bytes=64'h100000;job_output_epoch=32'(case_id+200);
        if(mode==47) job_output_addr++;
        if(mode==48) job_output_region_bytes=4033;
        if(mode==49) job_output_epoch++;
        if(mode==50) begin job_output_addr=weight_base+4032;job_output_region_base=weight_base;end
        if(mode==51) begin job_output_addr=ACT+4032;job_output_region_base=ACT;end
        if(mode==54 || mode==55) job_output_addr=OUT;
        for(int r=0;r<32;r++) expected_output[r*64+:64]=r<rows ? $realtobits(reference_value(r)):64'd0;
        job_valid=1;tile_valid=1;cancel_valid=0;m_axi_rvalid=0;
        while(!finished) begin
            m_axi_arready=$urandom_range(0,3)!=0;
            if(mode==10 && cycles<30) m_axi_arready=0;

            done_ready=done_wait>=6 && $urandom_range(0,3)!=0;
            if(!cancelled && cancel_ready && ((mode==9 && dut.engine.state==2) ||
               (mode==10 && m_axi_arvalid) || (mode==11 && act_beats>=4) ||
               (mode==14 && dut.engine.core.mac_state==2) || (mode==52 && dut.state==1) ||
               (mode==53 && dut.state==4))) cancel_valid=1;
            if(mode==46 && m_axi_awvalid) cancel_valid=1;
            if(!m_axi_rvalid && active && $urandom_range(0,3)!=0 && !(mode==12 && cycles<90)) begin
                m_axi_rdata=memory_word(burst_address+64'(beat*64));m_axi_rresp=0;m_axi_rid=0;
                if(mode==7 && is_activation && act_beats==k-1) m_axi_rresp=2;
                if(mode==8 && is_activation && act_beats==0) m_axi_rid=1;
                if(mode==13 && !is_activation && beat==beats-1) m_axi_rresp=2;
                m_axi_rlast=beat==beats-1;m_axi_rvalid=1;
            end
            @(posedge clk);cycles++;require(cycles<(fixture_mode ? 250000:20000),"timeout");
            job_hs=job_valid&&job_ready;tile_hs=tile_valid&&tile_ready;
            ar_hs=m_axi_arvalid&&m_axi_arready;r_hs=m_axi_rvalid&&m_axi_rready;cancel_hs=cancel_valid&&cancel_ready;
            if(ar_hold) require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"AR stability");
            if(result_hold) require(result_valid && {result_data,result_row,result_last,result_sequence,result_epoch}===saved_result,"result stability");
            if(done_hold) require(done_valid && {done_status,done_bytes,done_sequence,done_epoch}===saved_done,"completion stability");
            ar_hold=m_axi_arvalid&&!m_axi_arready;result_hold=result_valid&&!result_ready;done_hold=done_valid&&!done_ready;
            saved_ar={m_axi_araddr,m_axi_arlen};saved_result={result_data,result_row,result_last,result_sequence,result_epoch};
            saved_done={done_status,done_bytes,done_sequence,done_epoch};ar_stalls+=int'(ar_hold);result_stalls+=int'(result_hold);done_stalls+=int'(done_hold);
            if(job_hs) begin require(!accepted,"duplicate job");accepted=1;end
            if(dut.engine.loader_done_valid && dut.engine.loader_status==0) activation_complete=1;
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
`ifdef NTRPU_STORE_FIXTURE
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
                result_wait++;require((status==0 || (mode>=40 && mode<=46)) && mode!=15 && mode!=16,"partial compute output on failure/reset");
                require(!cancel_ready,"cancel accepted after result commit");
                require(activation_complete && !active && !dut.engine.loader.activation_valid,"result before drain");
                if(first_result==0) first_result=cycles;
                if(result_ready) begin
                    require(result_row==5'(outputs) && result_last==(outputs==rows-1),"result row/last");
                    require(result_sequence==32'(case_id) && result_epoch==32'(case_id+200),"registered result identity");
                    require(result_data==$realtobits(reference_value(outputs)),"independent integer-dot result");
                    $display("NTRPU_STORE_COMPUTED case=%0d row=%0d value=%0.17g reference=%0.17g",case_id,outputs,$bitstoreal(result_data),reference_value(outputs));
                    outputs++;
                end
            end
            if(done_valid) begin
                require(mode!=54 && mode!=55,"write-reset job completed before reset");
                done_wait++;require(!active && !m_axi_arvalid && !dut.engine.loader.activation_valid,"done before drain");
                require(done_status==4'(status),$sformatf("status %0d expected %0d",done_status,status));
                require(done_sequence==32'(case_id) && done_epoch==32'(case_id+200),"registered completion identity");
                require(outputs==((status==0 || (mode>=40 && mode<=46)) ? rows:0),"missing/partial compute results");
                if(mode>=1 && mode<=4 || mode==9) require(ab==0 && wb==0,"invalid/early cancelled job touched memory");
                require(write_aw==write_b && !m_axi_awvalid && !m_axi_wvalid && !m_axi_bvalid,"completion before write response");
                require(done_bytes==9'(write_bytes),"completion write byte count");
                if(status==0) require(write_bytes==rows*8 && write_w==(rows+7)/8,"incomplete result memory");
                else if(mode<40 || mode>46) require(write_aw==0 && write_w==0,"failed compute/descriptor touched output");
                if(mode>=47 && mode<=53) require(ab==0 && wb==0,"bad output descriptor/early cancel touched inputs");
                if(done_ready) finished=1;
            end
            #1;
            if(job_hs) begin
                job_valid=0;job_layout=0;job_rows=0;job_k_blocks=0;job_sequence=0;job_epoch=0;
                job_region_base=0;job_region_bytes=0;job_activation_addr=0;
                job_activation_region_base=0;job_activation_region_bytes=0;job_activation_epoch=0;
                job_output_addr=0;job_output_region_base=0;job_output_region_bytes=0;job_output_epoch=0;
            end
            if(tile_hs) begin
`ifdef NTRPU_STORE_FIXTURE
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
            if((mode==15 && act_beats>=4) || (mode==16 && dut.engine.core.mac_state==2) ||
               (mode==54 && write_w==1 && m_axi_wvalid) ||
               (mode==55 && write_aw==1 && write_w==3 && write_b==0 && m_axi_bready)) begin
                reset_aw=write_aw;reset_w=write_w;reset_b=write_b;reset_bytes=write_bytes;
                reset_aw_stalls=write_aw_stalls;reset_w_stalls=write_w_stalls;
                if(mode==54 || mode==55) begin
                    require(outputs==17 && write_aw==1 && write_b==0 && !done_valid,"write-reset precondition");
                    require(mode==54 ? write_w==1 && write_bytes==64 : write_w==3 && write_bytes==136,"write-reset phase");
                    $display("NTRPU_STORE_RESET id=%0d scenario=%0d aw=%0d w=%0d b=%0d bytes=%0d",case_id,mode,write_aw,write_w,write_b,write_bytes);
                end
                @(negedge clk);rst_n=0;job_valid=0;tile_valid=0;cancel_valid=0;m_axi_rvalid=0;active=0;
                repeat(4) @(negedge clk);rst_n=1;repeat(8) @(negedge clk);
                require(job_ready && !done_valid && !result_valid && !m_axi_arvalid &&
                    !m_axi_awvalid && !m_axi_wvalid && !m_axi_bvalid &&
                    write_aw==0 && write_w==0 && write_b==0,"stale state after coordinated reset");
                finished=1;reset_cases++;
            end
            if(!finished) @(negedge clk);
        end
        $display("NTRPU_STORE_CASE id=%0d rows=%0d blocks=%0d scenario=%0d status=%0d cycles=%0d first_result=%0d activation_bursts=%0d weight_bursts=%0d activation_beats=%0d weight_beats=%0d outputs=%0d tiles=%0d",case_id,rows,k,mode,status,cycles,first_result,ab,wb,act_beats,weight_beats,outputs,tiles);
        if(mode==15 || mode==16 || mode==54 || mode==55)
            $display("NTRPU_STORE_WRITE id=%0d aw=%0d w=%0d b=%0d bytes=%0d aw_stalls=%0d w_stalls=%0d reset=1",case_id,reset_aw,reset_w,reset_b,reset_bytes,reset_aw_stalls,reset_w_stalls);
        else
            $display("NTRPU_STORE_WRITE id=%0d aw=%0d w=%0d b=%0d bytes=%0d aw_stalls=%0d w_stalls=%0d reset=0",case_id,write_aw,write_w,write_b,write_bytes,write_aw_stalls,write_w_stalls);
        total_cycles+=cycles;total_outputs+=outputs;total_ab+=ab;total_wb+=wb;
        tile_valid=0;cancel_valid=0;repeat(3) @(negedge clk);
    endtask
`ifdef NTRPU_STORE_FIXTURE
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
`ifdef NTRPU_STORE_FIXTURE
        fixture_mode=$value$plusargs("FIXTURE=%s",fixture_dir);require(fixture_mode,"fixture argument missing");
`endif
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_STORE_SEED seed=%0d",seed);random_state=$urandom(seed);
        repeat(4) @(negedge clk);rst_n=1;@(posedge clk);#1;
`ifdef NTRPU_STORE_FIXTURE
        if(fixture_mode) run_fixtures();else
`endif
        begin
        run_case(2,32,0,0);run_case(2,128,0,0);
        run_case(0,32,1,1);run_case(2,32,2,1);run_case(2,32,3,2);run_case(2,32,4,2);
        run_case(2,32,5,2);run_case(2,32,6,2);run_case(2,32,7,3);run_case(2,32,8,3);
        run_case(2,32,9,7);run_case(2,32,10,7);run_case(2,128,11,7);run_case(2,32,12,8);
        run_case(2,32,13,3);run_case(2,32,14,7);run_case(1,64,0,0);
        run_case(2,128,15,0);run_case(2,32,16,0);run_case(2,32,0,0);
        run_case(17,32,40,3);run_case(17,32,41,3);run_case(17,32,42,3);
        run_case(17,32,43,8);run_case(17,32,44,8);run_case(17,32,45,8);run_case(17,32,46,0);
        run_case(2,32,47,1);run_case(2,32,48,1);run_case(2,32,49,2);
        run_case(2,32,50,1);run_case(2,32,51,1);run_case(2,32,52,7);run_case(2,32,53,7);
        run_case(17,32,54,0);run_case(17,32,55,0);
        run_case(2,32,0,0);
        end
        require(ar_stalls>0 && done_stalls>0 && reset_cases==(fixture_mode ? 0:4),"coverage counters");
        $display("NTRPU_STORE_PASS cases=%0d cycles=%0d outputs=%0d activation_bursts=%0d weight_bursts=%0d ar_stalls=%0d result_stalls=%0d done_stalls=%0d resets=%0d",case_id,total_cycles,total_outputs,total_ab,total_wb,ar_stalls,result_stalls,done_stalls,reset_cases);
        $finish;
    end
endmodule
