`timescale 1ns/1ps
module tb_ntrpu_q8_matrix_memory;
    logic job_output_fp32=0;
    logic clk=0;
    logic rst_n=0;
    logic job_valid=0;
    logic job_ready;
    logic [1:0] job_layout=0;
    logic [13:0] job_rows=0;
    logic [8:0] job_k_blocks=0;
    logic [5:0] job_tile_k=0;
    logic [31:0] job_sequence=0;
    logic [31:0] job_epoch=0;
    logic [63:0] job_weight_addr=0;
    logic [63:0] job_region_base=0;
    logic [63:0] job_region_bytes=0;
    logic [63:0] job_activation_addr=0;
    logic [63:0] job_activation_region_base=0;
    logic [63:0] job_activation_region_bytes=0;
    logic [31:0] job_activation_epoch=0;
    logic [63:0] job_output_addr=0;
    logic [63:0] job_output_region_base=0;
    logic [63:0] job_output_region_bytes=0;
    logic [31:0] job_output_epoch=0;
    logic cancel_valid=0;
    logic cancel_ready;
    logic done_valid;
    logic done_ready=0;
    logic [3:0] done_status;
    logic [16:0] done_bytes;
    logic [13:0] done_rows;
    logic [31:0] done_sequence;
    logic [31:0] done_epoch;
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
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize;
    logic [2:0] m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid;
    logic [3:0] m_axi_awcache;
    logic m_axi_awlock;
    logic m_axi_awvalid;
    logic m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic m_axi_wlast;
    logic m_axi_wvalid;
    logic m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic m_axi_bvalid;
    logic m_axi_bready;

    always #2.5 clk=~clk;
    ntrpu_q8_matrix_memory #(.WATCHDOG_CYCLES(128)) dut(.*);
    localparam logic [63:0] W=64'h100000000,A=64'h8000000000,O=64'ha000000000;
    int seed=1,random_state,ident=0,n,k,chunk,mode,cycles,groups,tiles,child_rows,act_group_beats;
    int ar_count,activation_beats,weight_beats,aw_count,w_count,b_count,byte_count,computed,child_done;
    int total_cases=0,total_cycles=0,total_outputs=0,total_groups=0,total_reads=0,total_write_bytes=0,total_resets=0;
    int ar_stalls=0,done_stalls=0,model_id=0,model_mode=0;
    int write_aw,write_w,write_b,write_bytes,write_aw_stalls,write_w_stalls;
    logic [2047:0] expected_output=0;
    logic [63:0] refs[0:8191];
    logic [7:0] destination[0:65599];
    int group_write_beats,group_aw_cursor;
    bit fixture_mode=0;
    string fixture_dir;
`ifdef NTRPU_MATRIX_FIXTURE
`ifdef NTRPU_MATRIX_LARGE
    localparam IMAGE_LIMIT=20971520;
`else
    localparam IMAGE_LIMIT=524288;
`endif
    logic [7:0] weight_image[0:IMAGE_LIMIT-1],activation_image[0:32767];
    int weight_image_bytes;
`endif
    ntrpu_result_memory_model output_memory (
        .clk,.rst_n,.start(dut.e_job_valid && dut.e_job_ready),.rows(dut.e_job_rows),.address(dut.e_job_output_addr),
        .identity(model_id),.scenario(model_mode),.expected_output,.commit_seen(dut.engine.engine_done_valid),
        .m_axi_awaddr,.m_axi_awlen,.m_axi_awsize,.m_axi_awprot,.m_axi_awburst,.m_axi_awid,.m_axi_awcache,
        .m_axi_awlock,.m_axi_awvalid,.m_axi_awready,.m_axi_wdata,.m_axi_wstrb,.m_axi_wlast,.m_axi_wvalid,.m_axi_wready,
        .m_axi_bid,.m_axi_bresp,.m_axi_bvalid,.m_axi_bready,
        .aw_count(write_aw),.w_count(write_w),.b_count(write_b),.byte_count(write_bytes),
        .aw_stalls(write_aw_stalls),.w_stalls(write_w_stalls)
    );
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL MATRIX MEMORY case=%0d cycle=%0d: %s",ident,cycles,why);
    endfunction
    function automatic logic [511:0] read_word(input logic [63:0] addr);
        int off,group,gn,group_offset,ti,block_no,row,kb,lane;
        read_word=0;
`ifdef NTRPU_MATRIX_FIXTURE
        if(fixture_mode) begin
            if(addr>=A && addr<A+64'h100000) begin
                off=int'(addr-A);require(off>=4032 && off+64<=4032+k*64,"activation image bounds");
                for(int j=0;j<64;j++) read_word[j*8+:8]=activation_image[off+j];
            end else begin
                off=int'(addr-W);require(off>=4032 && off+64<=weight_image_bytes,"weight image bounds");
                for(int j=0;j<64;j++) read_word[j*8+:8]=weight_image[off+j];
            end
            return read_word;
        end
`endif
        if(addr>=A && addr<A+64'h100000) begin
            kb=int'((addr-A-4032)/64);
            for(int j=0;j<32;j++) read_word[j*8+:8]=8'((kb+j)%5-2);
            read_word[319:256]=64'h3ff0000000000000;
        end else begin
            off=int'(addr-W-4032);
            for(int j=0;j<64;j++) begin
                group=(off+j)/(32*k*34);gn=n-group*32>32 ? 32:n-group*32;
                group_offset=off+j-group*32*k*34;ti=group_offset/(gn*chunk*34);
                block_no=(group_offset%(gn*chunk*34))/34;lane=(off+j)%34;
                // Tile/group starts are multiples of whole 34-byte records.
                row=group*32+block_no%gn;kb=ti*chunk+block_no/gn;
                if(lane==0) read_word[j*8+:8]=0;
                else if(lane==1) read_word[j*8+:8]=8'h3c;
                else read_word[j*8+:8]=8'((kb+row*3+lane-2)%7-3);
            end
        end
    endfunction
    task automatic run_case(input int nr,input int kb,input int tk,input int scenario,input int status);
        bit active,is_activation,accepted,finished,cancelled,ar_hold,done_hold;
        bit job_hs,child_hs,tile_hs,ar_hs,r_hs,cancel_hs,child_done_hs;
        int beat,beats,done_wait,first_output,case_reset,group_number,expected_count;
        logic [63:0] burst_address,next_weight,expected_addr;
        logic [71:0] saved_ar;
        logic [98:0] saved_done;
        ident++;n=nr;k=kb;chunk=tk;mode=scenario;cycles=0;groups=0;tiles=0;act_group_beats=0;
        ar_count=0;activation_beats=0;weight_beats=0;aw_count=0;w_count=0;b_count=0;byte_count=0;computed=0;child_done=0;
        foreach(destination[j]) destination[j]=8'ha5;
        group_write_beats=0;group_aw_cursor=0;
        active=0;accepted=0;finished=0;cancelled=0;ar_hold=0;done_hold=0;done_wait=0;first_output=0;case_reset=0;next_weight=W+4032;
        if(!fixture_mode) for(int r=0;r<n;r++) begin
            int sum;sum=0;
            for(int block=0;block<k;block++) for(int lane=0;lane<32;lane++) sum+=((block+lane)%5-2)*((block+r*3+lane)%7-3);
            refs[r]=$realtobits(real'(sum));
        end
        @(negedge clk);require(job_ready,"previous matrix still active");
        job_layout=2;job_rows=14'(n);job_k_blocks=9'(k);job_tile_k=6'(chunk);job_sequence=32'hfffffff0+32'(ident);job_epoch=32'(ident+200);
        job_weight_addr=W+4032;job_region_base=W;job_region_bytes=64'h10000000;
        job_activation_addr=A+4032;job_activation_region_base=A;job_activation_region_bytes=64'h100000;job_activation_epoch=job_epoch;
        job_output_addr=O+4032;job_output_region_base=O;job_output_region_bytes=64'h100000;job_output_epoch=job_epoch;
        if(mode==5) job_output_region_bytes=64'(4032+((n+7)/8)*64-1);
        job_valid=1;cancel_valid=0;m_axi_rvalid=0;
        while(!finished) begin
            m_axi_arready=$urandom_range(0,3)!=0;done_ready=done_wait>=6 && $urandom_range(0,3)!=0;
            if(dut.e_job_valid) begin
                model_id=ident*1000+groups;model_mode=(mode==2 && groups==1) ? 40:(mode==4 && groups==1) ? 54:0;
                for(int r=0;r<32;r++) expected_output[r*64+:64]=groups*32+r<n ? refs[groups*32+r]:64'd0;
            end
            if(mode==3 && groups==2 && act_group_beats>=4 && !cancelled && cancel_ready) cancel_valid=1;
            if(active && !m_axi_rvalid && $urandom_range(0,3)!=0) begin
                m_axi_rdata=read_word(burst_address+64'(beat*64));m_axi_rresp=0;m_axi_rid=0;
                if(mode==1 && !is_activation && burst_address+64'(beat*64)==W+4032+64'(n*k*34-64)) m_axi_rresp=2;
                m_axi_rlast=beat==beats-1;m_axi_rvalid=1;
            end
            @(posedge clk);cycles++;require(cycles<(fixture_mode ? 10000000:200000),"timeout");
            job_hs=job_valid&&job_ready;child_hs=dut.e_job_valid&&dut.e_job_ready;tile_hs=dut.e_tile_valid&&dut.e_tile_ready;
            ar_hs=m_axi_arvalid&&m_axi_arready;r_hs=m_axi_rvalid&&m_axi_rready;cancel_hs=cancel_valid&&cancel_ready;
            child_done_hs=dut.e_done_valid&&dut.e_done_ready;
            if(ar_hold) require(m_axi_arvalid && {m_axi_araddr,m_axi_arlen}===saved_ar,"AR stability");
            if(done_hold) require(done_valid && {done_status,done_bytes,done_rows,done_sequence,done_epoch}===saved_done,"completion stable");
            ar_hold=m_axi_arvalid&&!m_axi_arready;done_hold=done_valid&&!done_ready;
            saved_ar={m_axi_araddr,m_axi_arlen};saved_done={done_status,done_bytes,done_rows,done_sequence,done_epoch};
            ar_stalls+=int'(ar_hold);done_stalls+=int'(done_hold);
            if(job_hs) begin require(!accepted,"duplicate matrix job");accepted=1;end
            if(child_hs) begin
                require(!active && (groups==0 || child_done==groups),"child group overlap");
                require(dut.e_job_output_addr==O+4032+64'(groups*256),"independent global destination address");
                group_write_beats=0;group_aw_cursor=0;
                group_number=groups;child_rows=n-groups*32>32 ? 32:n-groups*32;groups++;act_group_beats=0;
            end
            if(tile_hs) tiles++;
            if(cancel_hs) cancelled=1;
            if(ar_hs) begin
                require(accepted && !active && groups>0,"unexpected AR");
                is_activation=m_axi_araddr>=A && m_axi_araddr<A+64'h100000;
                expected_addr=is_activation ? A+4032+64'(act_group_beats*64):next_weight;
                require(m_axi_araddr==expected_addr,"matrix read address/alias");
                require(m_axi_araddr[5:0]==0 && int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64<=4096,"read page split");
                require(m_axi_arsize==6 && m_axi_arburst==1 && m_axi_arid==0 && m_axi_arcache==2 && m_axi_arprot==0 && !m_axi_arlock,"read attributes");
                require(is_activation || act_group_beats==k,"weights before all activations");
                active=1;burst_address=m_axi_araddr;beats=int'(m_axi_arlen)+1;beat=0;ar_count++;
            end
            if(r_hs) begin
                require(active,"unsolicited R");beat++;
                if(is_activation) begin activation_beats++;act_group_beats++;end
                else begin weight_beats++;next_weight+=64;end
                if(m_axi_rlast) active=0;
            end
            if(dut.engine.result_valid && dut.engine.result_ready) begin
                int row;row=(groups-1)*32+int'(dut.engine.result_row);
                require(row<n && dut.engine.result_data===refs[row],"computed full-matrix oracle");computed++;
            end
            if(m_axi_awvalid && m_axi_awready) begin
                require(m_axi_awaddr==O+4032+64'((groups-1)*256+group_aw_cursor),"global output burst address");
                group_aw_cursor+=(int'(m_axi_awlen)+1)*64;aw_count++;
            end
            if(m_axi_wvalid && m_axi_wready) begin
                int byte_offset;byte_offset=(groups-1)*256+group_write_beats*64;
                if(first_output==0) first_output=cycles;w_count++;
                for(int j=0;j<64;j++) if(m_axi_wstrb[j]) begin
                    require(byte_offset+j<n*8,"write beyond matrix output");
                    destination[byte_offset+j]=m_axi_wdata[j*8+:8];byte_count++;
                end
                group_write_beats++;
            end
            if(m_axi_bvalid && m_axi_bready) b_count++;
            if(child_done_hs) begin
                require(!active && write_aw==write_b && dut.e_done_bytes==9'(write_bytes),"child completed before reads/writes drained");
                $display("NTRPU_MATRIX_MEMORY_GROUP case=%0d group=%0d rows=%0d status=%0d bytes=%0d cycles=%0d",ident,groups-1,child_rows,dut.e_done_status,write_bytes,cycles);child_done++;
            end
            if(done_valid) begin
                done_wait++;require(!active && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && aw_count==b_count,"matrix done before AXI drain");
                require(done_status==4'(status),$sformatf("status %0d expected %0d",done_status,status));
                require(done_sequence==32'hfffffff0+32'(ident) && done_epoch==32'(ident+200),"matrix completion identity");
                require(done_bytes==17'(byte_count),"matrix strobed-byte accounting");
                if(status==0) begin
                    require(int'(done_rows)==n && computed==n && byte_count==n*8 && groups==(n+31)/32,"matrix completeness");
                    for(int j=0;j<n*8;j++) require(destination[j]===refs[j/8][(j%8)*8+:8],"persistent matrix bytes");
                    for(int j=n*8;j<((n+7)/8)*64+64;j++) require(destination[j]==8'ha5,"matrix tail guard");
                end
                else if(mode==1 || mode==2 || mode==3) require(done_rows==32 && groups==2,"late failure advanced/publication");
                else require(groups==0 && ar_count==0 && byte_count==0,"rejected descriptor touched memory");
                if(done_ready) finished=1;
            end
            #1;
            if(job_hs) begin
                job_valid=0;job_layout=0;job_rows=0;job_k_blocks=0;job_tile_k=0;job_sequence=0;job_epoch=0;
                job_weight_addr=0;job_region_base=0;job_region_bytes=0;job_activation_addr=0;job_activation_region_base=0;
                job_activation_region_bytes=0;job_activation_epoch=0;job_output_addr=0;job_output_region_base=0;job_output_region_bytes=0;job_output_epoch=0;
            end
            if(r_hs) m_axi_rvalid=0;
            if(cancel_hs) cancel_valid=0;
            if(mode==4 && groups==2 && write_w==1 && m_axi_wvalid) begin
                require(!done_valid && byte_count==320,"reset during second-group private write");
                @(negedge clk);rst_n=0;job_valid=0;cancel_valid=0;m_axi_rvalid=0;active=0;
                repeat(4) @(negedge clk);rst_n=1;repeat(8) @(negedge clk);
                require(job_ready && !done_valid && !m_axi_arvalid && !m_axi_awvalid && !m_axi_wvalid && !m_axi_bvalid,"stale state after reset");
                case_reset=1;total_resets++;finished=1;
            end
            if(!finished) @(negedge clk);
        end
        $display("NTRPU_MATRIX_MEMORY_CASE id=%0d rows=%0d blocks=%0d chunk=%0d mode=%0d status=%0d cycles=%0d groups=%0d tiles=%0d ar=%0d activation_beats=%0d weight_beats=%0d aw=%0d w=%0d b=%0d bytes=%0d computed=%0d first_output=%0d reset=%0d",ident,n,k,chunk,mode,status,cycles,groups,tiles,ar_count,activation_beats,weight_beats,aw_count,w_count,b_count,byte_count,computed,first_output,case_reset);
        total_cases++;total_cycles+=cycles;total_outputs+=computed;total_groups+=groups;total_reads+=activation_beats+weight_beats;total_write_bytes+=byte_count;
        cancel_valid=0;repeat(4) @(negedge clk);
    endtask
`ifdef NTRPU_MATRIX_FIXTURE
    task automatic run_fixtures;
        int file,fields,count,id,nr,kb,tk,size,selected;
        string magic,stem;
        file=$fopen({fixture_dir,"/jobs.txt"},"r");require(file!=0,"fixture jobs missing");
        fields=$fscanf(file,"%s %d",magic,count);require(fields==2 && magic=="NTRMATRIX01" && count>0 && count<=3,"fixture header");
        selected=0;
        if($value$plusargs("FIXTURE_JOB=%d",selected)) begin end
        require(selected>=0 && selected<=count,"fixture job selection");
        if(selected!=0) $display("NTRPU_MATRIX_MEMORY_FIXTURE selected=%0d jobs=%0d",selected,count);
        for(int j=1;j<=count;j++) begin
            fields=$fscanf(file,"%d %d %d %d %d",id,nr,kb,tk,size);
            require(fields==5 && id==j && size<=IMAGE_LIMIT && size==4032+nr*kb*34,"fixture geometry");
            if(selected!=0 && selected!=id) continue;
            ident=id-1;
            stem=$sformatf("%s/matrix-%03d",fixture_dir,id);weight_image_bytes=size;
            $readmemh({stem,"-image.hex"},weight_image,0,size-1);
            $readmemh({stem,"-activation.hex"},activation_image,0,4032+kb*64-1);
            $readmemh({stem,"-reference.hex"},refs,0,nr-1);
            run_case(nr,kb,tk,0,0);
        end
        fields=$fscanf(file,"%s",magic);require(fields!=1,"trailing jobs");$fclose(file);
    endtask
`endif
    initial begin
`ifdef NTRPU_MATRIX_FIXTURE
        fixture_mode=$value$plusargs("FIXTURE=%s",fixture_dir);require(fixture_mode,"fixture missing");
`endif
        if($value$plusargs("SEED=%d",seed)) begin end
        $display("NTRPU_MATRIX_MEMORY_SEED seed=%0d",seed);random_state=$urandom(seed);
        repeat(4) @(negedge clk);rst_n=1;@(posedge clk);#1;
`ifdef NTRPU_MATRIX_FIXTURE
        run_fixtures();
`else
        run_case(33,64,32,0,0);run_case(65,32,32,0,0);run_case(64,9,4,0,0);
        run_case(64,32,32,1,3);run_case(64,32,32,2,3);run_case(64,32,32,3,7);
        run_case(64,32,32,4,0);run_case(33,33,32,0,1);run_case(64,32,32,5,1);run_case(33,64,32,0,0);
`endif
        require(ar_stalls>0 && done_stalls>0 && total_resets==(fixture_mode ? 0:1),"coverage counters");
        $display("NTRPU_MATRIX_MEMORY_PASS cases=%0d cycles=%0d computed=%0d groups=%0d read_beats=%0d write_bytes=%0d resets=%0d ar_stalls=%0d done_stalls=%0d",total_cases,total_cycles,total_outputs,total_groups,total_reads,total_write_bytes,total_resets,ar_stalls,done_stalls);
        $finish;
    end
endmodule
