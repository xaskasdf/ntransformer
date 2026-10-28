`timescale 1ns/1ps
// AXI-Lite host and protocol-only compute peer. No inference or DMA is implied.
module tb_ntrpu_ffn_control;
`ifdef NTRPU_FFN_FP64
    localparam bit FP32=0;
`else
    localparam bit FP32=1;
`endif
    logic clk=0,rst_n=0;
    always #2.5 clk=~clk;
    logic [31:0] s_axil_awaddr=0,s_axil_wdata=0,s_axil_araddr=0;
    logic [2:0] s_axil_awprot=0,s_axil_arprot=0;
    logic [3:0] s_axil_wstrb=0;
    logic s_axil_awvalid=0,s_axil_wvalid=0,s_axil_bready=0,s_axil_arvalid=0,s_axil_rready=0;
    wire s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid;
    wire [1:0] s_axil_bresp,s_axil_rresp;
    wire [31:0] s_axil_rdata;
    wire irq,job_valid,cancel_valid,done_ready;
    logic job_ready=0,cancel_ready=0,done_valid=0;
    wire [1:0] job_layout;
    wire [13:0] job_dim,job_hidden;
    wire [5:0] job_tile_k[0:2];
    wire [63:0] job_addr[0:7],job_base[0:7],job_bytes[0:7];
    wire [31:0] job_buffer_epoch[0:7],job_sequence,job_epoch,expected_epoch;
    logic [3:0] done_status=0,done_child_status=0;
    logic [2:0] done_stages=0;
    logic [1:0] done_phase=0;
    logic [16:0] done_bytes=0;
    logic [17:0] done_stage_bytes=0;
    logic [31:0] done_sequence=0,done_epoch=0,done_hls_status=0;
    ntrpu_ffn_control #(.OUTPUT_FP32(FP32)) dut(.notify_event_ready(1'b0),.notify_service_ready(1'b0),.notify_service_rejected(1'b0),
        .notify_retire_valid(1'b0),.notify_retire_sequence(32'b0),.notify_retire_epoch(32'b0),
        .notify_event_valid(),.notify_event_irq(),.notify_service_valid(),.notify_retire_ready(),
        .notify_sequence(),.notify_epoch(),.*);
    int seed=1,reads=0,writes=0,checks=0,cases=0,resets=0;
    int aw_first=0,w_first=0,together=0,b_stalls=0,r_stalls=0;
    int observed_jobs=0,observed_done=0,observed_cancel=0;
    logic [31:0] rng,shadow[0:7][0:6],captured[0:7][0:6];
    logic [31:0] value,mask,merged;
    int mode,hold_cycles,idx,field;
    longint unsigned cycles=0,last_job_cycle=0,last_done_cycles=0;
    always @(posedge clk) begin
        cycles=cycles+1;
        if(rst_n && job_valid && job_ready) begin observed_jobs++;last_job_cycle=cycles;end
        if(rst_n && done_valid && done_ready) begin observed_done++;last_done_cycles=cycles-last_job_cycle;end
        if(rst_n && cancel_valid && cancel_ready) observed_cancel++;
    end
    function automatic logic [31:0] random_word();
        rng=rng^(rng<<13);rng=rng^(rng>>17);rng=rng^(rng<<5);return rng;
    endfunction
    task automatic tick(); @(posedge clk);#1;endtask
    task automatic verify(input bit ok,input string reason);
        checks++;if(!ok) $fatal(1,"NTRPU_CSR_FAIL %s",reason);
    endtask
    task automatic mark(input string name);
        cases++;$display("NTRPU_CSR_CASE id=%0d name=%s",cases,name);
    endtask
    task automatic reset_all();
        @(negedge clk);rst_n=0;
        s_axil_awvalid=0;s_axil_wvalid=0;s_axil_bready=0;s_axil_arvalid=0;s_axil_rready=0;
        job_ready=0;cancel_ready=0;done_valid=0;
        repeat(3) tick();
        verify(!job_valid && !cancel_valid && !done_ready && !irq && !s_axil_bvalid && !s_axil_rvalid,"reset state");
        @(negedge clk);rst_n=1;tick();resets++;
    endtask
    task automatic send_aw(input logic [31:0] address,input int delay_cycles);
        repeat(delay_cycles) tick();
        @(negedge clk);s_axil_awaddr=address;s_axil_awvalid=1;
        while(!s_axil_awready) @(negedge clk);
        tick();@(negedge clk);s_axil_awvalid=0;s_axil_awaddr=32'hdeadbeef;
    endtask
    task automatic send_w(input logic [31:0] data,input logic [3:0] strobes,input int delay_cycles);
        repeat(delay_cycles) tick();
        @(negedge clk);s_axil_wdata=data;s_axil_wstrb=strobes;s_axil_wvalid=1;
        while(!s_axil_wready) @(negedge clk);
        tick();@(negedge clk);s_axil_wvalid=0;s_axil_wdata=32'hbad0bad0;s_axil_wstrb=0;
    endtask
    task automatic write_reg(input logic [31:0] address,data,input logic [3:0] strobes,
                             input logic [1:0] response,input int order=0,input int stall=3);
        fork
            send_aw(address,order==2 ? 4:0);
            send_w(data,strobes,order==1 ? 4:0);
        join
        while(!s_axil_bvalid) tick();
        verify(s_axil_bresp===response,"write response");
        repeat(stall) begin
            tick();verify(s_axil_bvalid && s_axil_bresp===response,"B held stable");
            verify(!s_axil_awready && !s_axil_wready,"no write acceptance behind held B");b_stalls++;
        end
        $display("NTRPU_CSR_WRITE id=%0d addr=%08h data=%08h strb=%h resp=%0d order=%0d stall=%0d",
            writes,address,data,strobes,s_axil_bresp,order,stall);
        @(negedge clk);s_axil_bready=1;tick();
        verify(!s_axil_bvalid,"B consumed once");@(negedge clk);s_axil_bready=0;
        writes++;if(order==1) aw_first++;else if(order==2) w_first++;else together++;
    endtask
    task automatic read_reg(input logic [31:0] address,expected,input logic [1:0] response=0,input int stall=3);
        @(negedge clk);s_axil_araddr=address;s_axil_arvalid=1;
        while(!s_axil_arready) @(negedge clk);
        tick();@(negedge clk);s_axil_arvalid=0;s_axil_araddr=32'hffffbeef;
        while(!s_axil_rvalid) tick();
        verify(s_axil_rdata===expected && s_axil_rresp===response,"read data/response");
        repeat(stall) begin
            tick();verify(s_axil_rvalid && s_axil_rdata===expected && s_axil_rresp===response,"R held stable");
            verify(!s_axil_arready,"no read acceptance behind held R");r_stalls++;
        end
        $display("NTRPU_CSR_READ id=%0d addr=%08h data=%08h expected=%08h resp=%0d stall=%0d",
            reads,address,s_axil_rdata,expected,s_axil_rresp,stall);
        @(negedge clk);s_axil_rready=1;tick();verify(!s_axil_rvalid,"R consumed once");
        @(negedge clk);s_axil_rready=0;reads++;
    endtask
    task automatic descriptor();
        write_reg('h20,'hfedcba98,15,0,1);
        write_reg('h24,'h76543210,15,0,2);
        write_reg('h28,'h76543210,15,0);
        write_reg('h2c,'h20000800,15,0); // dim2048, hidden8192
        write_reg('h30,2,15,0);
        write_reg('h34,'h00201001,15,0);
        for(int b=0;b<8;b++) for(int f=0;f<7;f++) begin
            shadow[b][f]=random_word();captured[b][f]=shadow[b][f];
            write_reg('h100+b*32+f*4,shadow[b][f],15,0,(b+f)%3,1);
        end
    endtask
    task automatic verify_descriptor();
        verify(job_sequence==='hfedcba98 && job_epoch==='h76543210 && expected_epoch==='h76543210,"identity captured");
        verify(job_dim==2048 && job_hidden==8192 && job_layout==2,"shape captured");
        verify(job_tile_k[0]==1 && job_tile_k[1]==16 && job_tile_k[2]==32,"tiles captured");
        for(int b=0;b<8;b++) begin
            verify(job_addr[b]==={captured[b][1],captured[b][0]},"64-bit address captured");
            verify(job_base[b]==={captured[b][3],captured[b][2]},"64-bit base captured");
            verify(job_bytes[b]==={captured[b][5],captured[b][4]},"64-bit size captured");
            verify(job_buffer_epoch[b]===captured[b][6],"buffer epoch captured");
        end
    endtask
    task automatic accept_job();
        verify(job_valid && !done_ready,"pending until compute accepts");
        @(negedge clk);job_ready=1;tick();
        verify(!job_valid && done_ready,"job accepted");@(negedge clk);job_ready=0;
    endtask
    task automatic completion(input int status=0,input int faults=0);
        @(negedge clk);done_status=4'(status);done_child_status=faults==8 ? 6:0;
        done_sequence=job_sequence ^ (faults==1 ? 1:0);done_epoch=job_epoch ^ (faults==2 ? 1:0);
        done_stages=faults==3 ? 3:4;done_phase=faults==4 ? 2:3;
        done_bytes=17'(2048*(FP32 ? 4:8)+(faults==5 ? 4:0));done_stage_bytes=18'd155648;
        done_hls_status=faults==6 ? 1:0;done_valid=1;
        verify(done_ready,"completion slot ready");tick();
        verify(!done_ready && !job_valid && !cancel_valid,"completion retained");
        @(negedge clk);done_valid=0;
        $display("NTRPU_CSR_DONE status=%0d fault=%0d cycles=%0d",status,faults,last_done_cycles);
    endtask
    initial begin
        if($value$plusargs("SEED=%d",seed)) begin end
        rng=32'(seed);$display("NTRPU_CSR_SEED seed=%0d fp32=%0d",seed,FP32);
        reset_all();
        read_reg(0,'h4652544e);read_reg(4,'h00010000);read_reg(8,FP32 ? 7:6);
        read_reg('hc,0);read_reg('h10,0);read_reg('h14,0);read_reg('h18,0);read_reg('h1c,0);
        for(int a='h20;a<='h34;a+=4) read_reg(a,0);
        for(int a='h40;a<='h60;a+=4) read_reg(a,0);
        for(int b=0;b<8;b++) for(int f=0;f<7;f++) begin shadow[b][f]=0;read_reg('h100+b*32+f*4,0);end
        mark("reset_map");
        for(int a=0;a<=12;a+=4) write_reg(a,0,15,2);
        write_reg('h18,0,15,2);write_reg('h1c,0,15,2);
        for(int a='h40;a<='h60;a+=4) write_reg(a,0,15,2);
        for(int k=0;k<10;k++) begin
            case(k)
                0:value='h38;1:value='h3c;2:value='h64;3:value='hfc;4:value='h200;
                5:value='h10000100;6:value='h80000010;7:value='hffffffff;8:value=1;9:value='h102;
            endcase
            write_reg(value,'hffffffff,15,3,k%3);read_reg(value,0,3);
        end
        for(int b=0;b<8;b++) begin write_reg('h11c+b*32,0,15,3);read_reg('h11c+b*32,0,3);end
        write_reg('h14,2,15,2);write_reg('h2c,'h80000000,15,2);
        write_reg('h30,4,15,2);write_reg('h34,'h00400000,15,2);
        write_reg('h10,3,15,2);write_reg('h10,2,15,2);write_reg('h10,4,15,2);
        write_reg('h10,'hffffffff,0,0);read_reg('hc,0);
        mark("decode_rejections");
        for(int k=0;k<128;k++) begin
            idx=k%8;field=(k/8)%7;value=random_word();mode=k%3;hold_cycles=k%6;
            for(int lane=0;lane<4;lane++) mask[lane*8+:8]={8{(k%16)&(1<<lane) ? 1'b1:1'b0}};
            merged=(shadow[idx][field]&~mask)|(value&mask);shadow[idx][field]=merged;
            write_reg('h100+idx*32+field*4,value,4'(k%16),0,mode,hold_cycles);
            read_reg('h100+idx*32+field*4,merged,0,hold_cycles);
        end
        mark("strobes_and_channel_skew");
        descriptor();write_reg('h10,1,15,0);verify_descriptor();read_reg('hc,1);
        write_reg('h10,1,15,2);
        for(int b=0;b<8;b++) for(int f=0;f<7;f++) write_reg('h100+b*32+f*4,~captured[b][f],15,0,2,1);
        for(int a='h20;a<='h34;a+=4) write_reg(a,0,15,0);
        repeat(10) begin tick();verify_descriptor();verify(job_valid,"pending descriptor retained");end
        mark("atomic_descriptor");
        write_reg('h10,2,15,0);read_reg('hc,17);verify(!cancel_valid,"cancel queued before acceptance");
        write_reg('h10,2,15,2);accept_job();verify_descriptor();
        repeat(9) begin tick();verify(cancel_valid,"cancel stable until accepted");end
        @(negedge clk);cancel_ready=1;tick();@(negedge clk);cancel_ready=0;
        verify(!cancel_valid && observed_cancel==1,"one cancel handshake");read_reg('hc,18);
        write_reg('h10,1,15,2);write_reg('h10,2,15,2);completion(7);
        read_reg('hc,20);read_reg('h60,0);read_reg('h40,'h1c07);read_reg('h18,1);read_reg('h1c,1);
        verify(!irq,"IRQ masked");write_reg('h14,1,15,0);verify(irq,"pending IRQ unmasked");
        write_reg('h10,1,15,2);write_reg('h14,0,15,0);verify(!irq,"IRQ mask");
        write_reg('h14,1,15,0);read_reg('h4c,'hfedcba98);verify(irq,"reads retain completion IRQ");
        write_reg('h10,4,15,0);verify(!irq,"ACK clears level IRQ");write_reg('h10,4,15,2);
        mark("cancel_and_irq");
        descriptor();write_reg('h10,1,15,0);accept_job();
        repeat(12) begin tick();verify_descriptor();end
        completion();read_reg('h40,'h1c00);read_reg('h44,2048*(FP32 ? 4:8));read_reg('h48,155648);
        read_reg('h4c,'hfedcba98);read_reg('h50,'h76543210);read_reg('h54,0);
        read_reg('h58,32'(last_done_cycles));read_reg('h5c,32'(last_done_cycles>>32));read_reg('h60,0);
        read_reg('h18,2);read_reg('h1c,2);write_reg('h10,4,15,0);
        mark("success_metadata");
        for(int fault=1;fault<=8;fault++) begin
            write_reg('h10,1,15,0);accept_job();completion(fault==7 ? 9:0,fault);
            read_reg('h60,fault==1 ? 1:fault==2 ? 2:4);write_reg('h10,4,15,0);
        end
        for(int status=1;status<=8;status++) begin
            write_reg('h10,1,15,0);accept_job();completion(status);
            read_reg('h60,status==3 || status==4 ? 4:0);write_reg('h10,4,15,0);
        end
        mark("completion_faults");
        // A read response remains a snapshot while an independent write changes the register.
        @(negedge clk);s_axil_araddr='h20;s_axil_arvalid=1;tick();@(negedge clk);s_axil_arvalid=0;
        write_reg('h20,'h11223344,15,0);
        verify(s_axil_rvalid && s_axil_rdata==='hfedcba98,"held R preserves original snapshot");
        @(negedge clk);s_axil_rready=1;tick();@(negedge clk);s_axil_rready=0;
        read_reg('h20,'h11223344);mark("concurrent_read_write");
        // Reset each independently held protocol/state element, then prove recovery.
        for(int kind=0;kind<7;kind++) begin
            reset_all();
            case(kind)
                0:send_aw('h20,0);
                1:send_w('h55555555,15,0);
                2:begin fork send_aw('h20,0);send_w('h55555555,15,0);join while(!s_axil_bvalid) tick();end
                3:begin @(negedge clk);s_axil_arvalid=1;s_axil_araddr=0;tick();end
                4:write_reg('h10,1,15,0);
                5:begin write_reg('h10,1,15,0);accept_job();end
                6:begin write_reg('h10,1,15,0);accept_job();completion(1);end
            endcase
            reset_all();repeat(6) tick();read_reg('h20,0);read_reg('hc,0);read_reg('h18,0);read_reg('h1c,0);
            write_reg('h20,'habcdef01,15,0,kind%3);read_reg('h20,'habcdef01);
        end
        mark("reset_recovery");
        verify(aw_first>0 && w_first>0 && together>0 && b_stalls>0 && r_stalls>0,"bus coverage");
        verify(observed_jobs==20 && observed_done==19 && observed_cancel==1,"job lifecycle counts");
        $display("NTRPU_CSR_PASS cases=%0d reads=%0d writes=%0d checks=%0d resets=%0d jobs=%0d done=%0d cancel=%0d aw_first=%0d w_first=%0d together=%0d b_stalls=%0d r_stalls=%0d",
            cases,reads,writes,checks,resets,observed_jobs,observed_done,observed_cancel,aw_first,w_first,together,b_stalls,r_stalls);
        $finish;
    end
    initial begin #1000000;$fatal(1,"NTRPU_CSR_FAIL watchdog");end
endmodule
