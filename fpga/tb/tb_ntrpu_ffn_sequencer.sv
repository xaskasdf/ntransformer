`timescale 1ns/1ps
module tb_ntrpu_ffn_sequencer;
    logic clk,rst_n;
    logic job_valid;
    logic job_ready;
    logic [1:0] job_layout;
    logic [13:0] job_dim,job_hidden;
    logic [5:0] job_tile_k [0:2];
    logic [63:0] job_addr [0:7],job_base [0:7],job_bytes [0:7];
    logic [31:0] job_buffer_epoch [0:7];
    logic [31:0] job_sequence,job_epoch,expected_epoch;
    logic cancel_valid;
    logic cancel_ready;
    logic done_valid;
    logic done_ready;
    logic [3:0] done_status,done_child_status;
    logic [2:0] done_stages;
    logic [1:0] done_phase;
    logic [16:0] done_bytes;
    logic [17:0] done_stage_bytes;
    logic [31:0] done_sequence,done_epoch,done_hls_status;
    logic activation_selected;
    logic m_job_valid;
    logic m_job_ready;
    logic [1:0] m_job_layout;
    logic [13:0] m_job_rows;
    logic [8:0] m_job_k_blocks;
    logic [5:0] m_job_tile_k;
    logic [31:0] m_job_sequence,m_job_epoch;
    logic [63:0] m_job_weight_addr,m_job_region_base,m_job_region_bytes;
    logic [63:0] m_job_activation_addr,m_job_activation_region_base,m_job_activation_region_bytes;
    logic [31:0] m_job_activation_epoch;
    logic [63:0] m_job_output_addr,m_job_output_region_base,m_job_output_region_bytes;
    logic [31:0] m_job_output_epoch;
    logic m_cancel_valid;
    logic m_cancel_ready,m_done_valid;
    logic m_done_ready;
    logic [3:0] m_done_status;
    logic [16:0] m_done_bytes;
    logic [13:0] m_done_rows;
    logic [31:0] m_done_sequence,m_done_epoch;
    logic a_job_valid;
    logic a_job_ready;
    logic [13:0] a_job_elements;
    logic [31:0] a_job_sequence,a_job_epoch,a_expected_epoch;
    logic [63:0] a_job_gate_addr,a_job_gate_base,a_job_gate_bytes;
    logic [63:0] a_job_up_addr,a_job_up_base,a_job_up_bytes;
    logic [63:0] a_job_output_addr,a_job_output_base,a_job_output_bytes;
    logic [31:0] a_job_gate_epoch,a_job_up_epoch,a_job_output_epoch;
    logic a_cancel_valid;
    logic a_cancel_ready,a_done_valid;
    logic a_done_ready;
    logic [3:0] a_done_status;
    logic [8:0] a_done_blocks;
    logic [14:0] a_done_bytes;
    logic [31:0] a_done_sequence,a_done_epoch,a_done_hls_status;

    always #2.5 clk=~clk;
    logic m_job_output_fp32;
`ifdef NTRPU_FFN_FP32
    localparam bit OUTPUT_FP32=1;
`else
    localparam bit OUTPUT_FP32=0;
`endif
    localparam int DOWN_BYTES=OUTPUT_FP32 ? 4:8;
    ntrpu_ffn_sequencer #(.OUTPUT_FP32(OUTPUT_FP32)) dut(.*);
    int seed=1,rng,ident=0,cycles=0,total_cycles=0,offers=0,total_offers=0;
    int cancels=0,total_cancels=0,resets=0,job_stalls=0,done_stalls=0;
    int d,h,tiles[0:2],phase_next,completed,byte_sum,output_bytes;
    string label;
    logic [63:0] addresses[0:7],bases[0:7],sizes[0:7];
    logic [31:0] root_sequence,root_epoch;
    function automatic void require(input bit ok,input string why);
        if(!ok) $fatal(1,"FAIL FFN SEQUENCER case=%0d label=%s cycle=%0d: %s",ident,label,cycles,why);
    endfunction
    task automatic tick;
        @(posedge clk);#0.1;@(negedge clk);
        cycles++;total_cycles++;
        require(cycles<20000,"timeout");
    endtask
    task automatic clear_children;
        m_job_ready=0;a_job_ready=0;m_done_valid=0;a_done_valid=0;
        m_cancel_ready=0;a_cancel_ready=0;m_done_status=0;a_done_status=0;
        m_done_bytes=0;m_done_rows=0;m_done_sequence=0;m_done_epoch=0;
        a_done_blocks=0;a_done_bytes=0;a_done_sequence=0;a_done_epoch=0;a_done_hls_status=0;
    endtask
    task automatic configure(input string name,input int nd=64,input int nh=96,input int tk=2);
        ident++;label=name;cycles=0;offers=0;cancels=0;phase_next=0;completed=0;byte_sum=0;output_bytes=0;
        d=nd;h=nh;root_sequence=32'hfffffffe;root_epoch=32'(ident+400);
        require(job_ready,"previous job busy");
        job_valid=0;cancel_valid=0;done_ready=0;clear_children();
        job_layout=2;job_dim=14'(d);job_hidden=14'(h);
        job_sequence=root_sequence;job_epoch=root_epoch;expected_epoch=root_epoch;
        for(int i=0;i<3;i++) begin job_tile_k[i]=6'(tk);tiles[i]=tk;end
        for(int i=0;i<8;i++) begin
            bases[i]=64'h100000000+64'(i)*64'h20000000;
            addresses[i]=bases[i]+4032;sizes[i]=64'h10000000;
            job_base[i]=bases[i];job_addr[i]=addresses[i];job_bytes[i]=sizes[i];job_buffer_epoch[i]=root_epoch;
        end
    endtask
    task automatic submit;
        // Save the actual descriptors, then poison every external input.
        for(int i=0;i<8;i++) begin addresses[i]=job_addr[i];bases[i]=job_base[i];sizes[i]=job_bytes[i];end
        for(int i=0;i<3;i++) tiles[i]=int'(job_tile_k[i]);
        job_valid=1;tick();job_valid=0;
        job_layout=0;job_dim=0;job_hidden=0;job_sequence=0;job_epoch=0;expected_epoch=0;
        for(int i=0;i<3;i++) job_tile_k[i]=0;
        for(int i=0;i<8;i++) begin job_addr[i]=0;job_base[i]=0;job_bytes[i]=0;job_buffer_epoch[i]=0;end
    endtask
    task automatic check_offer(input int p);
        int wi,ai,oi,rows,kb,ti;
        logic [31:0] seq;
        wi=p==0 ? 1:p==1 ? 2:3;ai=p==3 ? 6:0;oi=p==0 ? 4:p==1 ? 5:7;
        rows=p==3 ? d:h;kb=((p==3 ? h:d)+31)/32;ti=p==3 ? 2:p;
        seq=root_sequence+32'(p);
        require(activation_selected==(p==2),"AXI mux phase");
        require(!(a_job_valid && m_job_valid),"two children offered");
        require(!a_done_ready && !m_done_ready,"completion ready before command accepted");
        if(p==2) begin
            require(a_job_valid && !m_job_valid,"activation offer");
            require(a_job_elements==h && a_job_sequence==seq && a_job_epoch==root_epoch && a_expected_epoch==root_epoch,"activation identity/shape");
            require(a_job_gate_addr==addresses[4] && a_job_gate_base==bases[4] && a_job_gate_bytes==sizes[4] && a_job_gate_epoch==root_epoch,"gate descriptor");
            require(a_job_up_addr==addresses[5] && a_job_up_base==bases[5] && a_job_up_bytes==sizes[5] && a_job_up_epoch==root_epoch,"up descriptor");
            require(a_job_output_addr==addresses[6] && a_job_output_base==bases[6] && a_job_output_bytes==sizes[6] && a_job_output_epoch==root_epoch,"activation output descriptor");
        end else begin
            require(m_job_valid && !a_job_valid,"matrix offer");
            require(m_job_output_fp32==(OUTPUT_FP32 && p==3),"matrix output format");
            require(m_job_layout==2 && m_job_rows==rows && m_job_k_blocks==kb && m_job_tile_k==tiles[ti],"matrix shape/layout");
            require(m_job_sequence==seq && m_job_epoch==root_epoch,"matrix identity");
            require(m_job_weight_addr==addresses[wi] && m_job_region_base==bases[wi] && m_job_region_bytes==sizes[wi],"weights descriptor");
            require(m_job_activation_addr==addresses[ai] && m_job_activation_region_base==bases[ai] && m_job_activation_region_bytes==sizes[ai] && m_job_activation_epoch==root_epoch,"matrix activation descriptor");
            require(m_job_output_addr==addresses[oi] && m_job_output_region_base==bases[oi] && m_job_output_region_bytes==sizes[oi] && m_job_output_epoch==root_epoch,"matrix output descriptor");
        end
    endtask
    task automatic wait_offer(input int p);
        require(p==phase_next,"unexpected test phase");
        while(!(m_job_valid || a_job_valid)) begin
            require(!done_valid,"premature completion");tick();
        end
        check_offer(p);
        repeat(2+($random(rng)&7)) begin tick();check_offer(p);job_stalls++;end
    endtask
    task automatic accept_offer(input int p);
        check_offer(p);
        if(p==2) a_job_ready=1;else m_job_ready=1;
        tick();m_job_ready=0;a_job_ready=0;offers++;total_offers++;phase_next++;
        require(!a_job_valid && !m_job_valid && !done_valid,"command must enter active state");
        require((p==2 ? a_done_ready:m_done_ready),"selected completion ready");
        // Unselected child completion is noise, and must remain unacknowledged.
        if(p==2) m_done_valid=1;else a_done_valid=1;
        repeat(2+($random(rng)&3)) begin
            require(!(p==2 ? m_done_ready:a_done_ready) && !done_valid,"inactive completion accepted");tick();
        end
        m_done_valid=0;a_done_valid=0;
        $display("NTRPU_FFN_SEQ_CHILD case=%0d phase=%0d sequence=%0d",ident,p,root_sequence+32'(p));
    endtask
    task automatic complete_child(input int p,input int fault=0,input bit cancelled=0);
        int bytes,count,status;
        logic [31:0] seq,ep;
        bytes=p==2 ? ((h+31)/32)*64:(p==3 ? d*DOWN_BYTES:h*8);
        count=p==2 ? (h+31)/32:p==3 ? d:h;
        status=0;seq=root_sequence+32'(p);ep=root_epoch;
        case(fault)
            1:begin status=3;bytes=0;count=0;end
            2:begin status=8;bytes=p==2 ? 64:8;count=0;end
            3:seq++;
            4:ep++;
            5:count++;
            6:bytes+=(p==2 ? 64:8);
            7:begin bytes--;status=3;end
            8:bytes-=(p==2 ? 64:8);
            9:count--;
        endcase
        if(p==2) begin
            a_done_status=4'(status);a_done_bytes=15'(bytes);a_done_blocks=9'(count);
            a_done_sequence=seq;a_done_epoch=ep;a_done_hls_status=fault==10 ? 32'd2:32'd0;a_done_valid=1;
        end else begin
            m_done_status=4'(status);m_done_bytes=17'(bytes);m_done_rows=14'(count);
            m_done_sequence=seq;m_done_epoch=ep;m_done_valid=1;
        end
        require(p==2 ? a_done_ready:m_done_ready,"missing active ready");
        tick();m_done_valid=0;a_done_valid=0;
        if(fault<3) begin byte_sum+=bytes;if(p==3) output_bytes=bytes;end
        if(fault==0 && !cancelled) completed++;
    endtask
    task automatic prefix(input int phases);
        for(int p=0;p<phases;p++) begin wait_offer(p);accept_offer(p);complete_child(p);end
    endtask
    task automatic cancel_now;
        require(cancel_ready,"cancel not accepted");cancel_valid=1;tick();cancel_valid=0;cancels++;total_cancels++;
        require(!cancel_ready,"duplicate cancellation enabled");
    endtask
    task automatic finish(input int status,input int child=0,input int last_phase=0,input int hls=0);
        logic [159:0] snapshot;
        while(!done_valid) begin require(!m_job_valid && !a_job_valid,"extra child command");tick();end
        require(done_status==status && done_child_status==child,"root/child status");
        require(done_phase==last_phase && done_stages==completed,"phase/completed stages");
        require(done_sequence==root_sequence && done_epoch==root_epoch && done_hls_status==hls,"root identity/HLS diagnostic");
        require(done_stage_bytes==byte_sum && done_bytes==output_bytes,"root accounting");
        snapshot={done_status,done_child_status,done_phase,done_stages,done_bytes,done_stage_bytes,done_sequence,done_epoch,done_hls_status};
        repeat(3+($random(rng)&7)) begin
            cancel_valid=1;tick();done_stalls++;
            require(done_valid && !job_ready && !cancel_ready && !m_job_valid && !a_job_valid,"completion handshake isolation");
            require(snapshot=={done_status,done_child_status,done_phase,done_stages,done_bytes,done_stage_bytes,done_sequence,done_epoch,done_hls_status},"completion changed under backpressure");
        end
        cancel_valid=0;
        $display("NTRPU_FFN_SEQ_CASE id=%0d label=%s dim=%0d hidden=%0d status=%0d child=%0d phase=%0d stages=%0d bytes=%0d stage_bytes=%0d offers=%0d cancels=%0d cycles=%0d",ident,label,d,h,done_status,done_child_status,done_phase,done_stages,done_bytes,done_stage_bytes,offers,cancels,cycles);
        done_ready=1;tick();done_ready=0;require(job_ready && !done_valid,"completion not retired");
    endtask
    task automatic reset_case(input int phase,input bit running);
        configure($sformatf("reset_%0d_%0d",phase,running));submit();prefix(phase);wait_offer(phase);
        if(running) accept_offer(phase);
        rst_n=0;clear_children();tick();rst_n=1;resets++;
        require(job_ready && !done_valid && !m_job_valid && !a_job_valid,"coordinated reset state");
        repeat(4) tick();
        $display("NTRPU_FFN_SEQ_RESET id=%0d phase=%0d running=%0d offers=%0d",ident,phase,running,offers);
    endtask
    initial begin
        clk=0;rst_n=0;job_valid=0;cancel_valid=0;done_ready=0;clear_children();
        if($value$plusargs("SEED=%d",seed)) begin end
        rng=seed;$display("NTRPU_FFN_SEQ_SEED seed=%0d",seed);
        $display("NTRPU_FFN_SEQ_FORMAT fp32=%0d",OUTPUT_FP32);
        repeat(3) tick();rst_n=1;tick();
        configure("small",32,32,1);submit();prefix(4);finish(0,0,3);
        configure("unequal");submit();prefix(4);finish(0,0,3);
        configure("brandon",256,720,4);job_tile_k[2]=8;submit();prefix(4);finish(0,0,3);
        configure("llama",2048,8192,32);submit();prefix(4);finish(0,0,3);
        configure("reverse",8192,2048,32);submit();prefix(4);finish(0,0,3);
        configure("maximum",8192,8192,1);submit();prefix(4);finish(0,0,3);
        configure("tail",33,1024,32);submit();prefix(4);finish(0,0,3);
        for(int mode=0;mode<13;mode++) begin
            configure($sformatf("shape_%0d",mode));
            case(mode)
                0:job_layout=1;1:job_dim=0;2:job_hidden=0;3:job_dim=8193;4:job_hidden=8193;
                5,6,7:job_tile_k[mode-5]=0;
                8,9,10:job_tile_k[mode-8]=33;
                11:expected_epoch++;
                12:job_hidden=33;
            endcase
            submit();finish(mode==11 ? 2:1);
        end
        for(int b=0;b<8;b++) for(int mode=0;mode<7;mode++) begin
            configure($sformatf("buffer_%0d_%0d",b,mode));
            case(mode)
                0:job_addr[b]++;
                1:job_bytes[b]=0;
                2:job_bytes[b]=4033;
                3:job_base[b]=job_addr[b]+64;
                4:job_buffer_epoch[b]++;
                5:begin job_base[b]=64'hffffffffffffffc0;job_bytes[b]=128;end
                6:begin job_addr[b]=64'hffffffffffffffc0;job_base[b]=job_addr[b];job_bytes[b]=63;end
            endcase
            submit();finish(mode==4 ? 2:1);
        end
        for(int b=4;b<8;b++) for(int other=0;other<8;other++) if(b!=other) begin
            configure($sformatf("alias_%0d_%0d",b,other));
            job_addr[b]=job_addr[other];job_base[b]=job_base[other];submit();finish(1);
        end
        for(int p=0;p<4;p++) for(int fault=1;fault<=(p==2 ? 10:9);fault++) begin
            configure($sformatf("fault_%0d_%0d",p,fault));submit();prefix(p);wait_offer(p);accept_offer(p);complete_child(p,fault);
            finish(fault==1 ? 6:fault==2 ? 8:5,fault==1 || fault==7 ? 3:fault==2 ? 8:0,p,fault==10 ? 2:0);
        end
        configure("cancel_preflight");submit();cancel_now();finish(7);
        for(int p=0;p<4;p++) for(int mode=0;mode<3;mode++) begin
            configure($sformatf("cancel_%0d_%0d",p,mode));submit();prefix(p);wait_offer(p);
            if(mode==0) begin
                cancel_now();repeat(5) begin check_offer(p);require(!done_valid,"cancel abandoned offered command");tick();end
            end
            accept_offer(p);
            if(mode!=0) cancel_now();
            repeat(5) begin
                require(!done_valid && (p==2 ? a_cancel_valid:m_cancel_valid),"cancel abandoned running command");tick();
            end
            if(mode!=1) begin
                if(p==2) a_cancel_ready=1;else m_cancel_ready=1;
                tick();a_cancel_ready=0;m_cancel_ready=0;
                require(!a_cancel_valid && !m_cancel_valid,"duplicate child cancel");
            end
            complete_child(p,0,1);finish(7,0,p);
        end
        for(int p=0;p<3;p++) begin
            configure($sformatf("cancel_between_%0d",p));submit();prefix(p+1);cancel_now();finish(7,0,p);
        end
        for(int p=0;p<4;p++) begin reset_case(p,0);reset_case(p,1);end
        configure("recovery");submit();prefix(4);finish(0,0,3);
        $display("NTRPU_FFN_SEQ_PASS cases=%0d resets=%0d offers=%0d cancels=%0d cycles=%0d job_stalls=%0d done_stalls=%0d",ident,resets,total_offers,total_cancels,total_cycles,job_stalls,done_stalls);
        $finish;
    end
endmodule
