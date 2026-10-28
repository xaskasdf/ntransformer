`timescale 1ns/1ps
// AXI-Lite host and protocol-only compute peer. No inference or DMA is implied.
module tb_ntrpu_ffn_irq_control;
`ifdef NTRPU_FFN_FP64
    localparam bit FP32=0;
`else
    localparam bit FP32=1;
`endif
    logic clk=0,rst_n=0;
    int profile=0;
    int source_half=2500,host_half=2000;
    logic host_clk=0,host_rst_n=0,usr_irq_ack=0,host_run=1;
    initial begin
        if($value$plusargs("PROFILE=%d",profile)) begin end
        if(profile==1) host_half=3100;
        if(profile==2) begin source_half=3100;host_half=1700;end
        forever #(source_half/1000.0) clk=~clk;
    end
    initial begin #700ps;forever begin #(host_half/1000.0);if(host_run) host_clk=~host_clk;else host_clk=0;end end
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
    wire raw_irq;
    ntrpu_ffn_irq_control #(.OUTPUT_FP32(FP32)) dut(.irq(raw_irq),.*);
    bit inject_early=0,ack_seen=0;
    initial inject_early=$test$plusargs("INJECT_EARLY");
    always @(posedge host_clk) begin
        if(!host_rst_n || !raw_irq) ack_seen<=0;
        else if(usr_irq_ack) ack_seen<=1;
    end
    assign irq=raw_irq && !(inject_early && ack_seen);
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
        @(negedge clk);rst_n=0;host_rst_n=0;usr_irq_ack=0;
        s_axil_awvalid=0;s_axil_wvalid=0;s_axil_bready=0;s_axil_arvalid=0;s_axil_rready=0;
        job_ready=0;cancel_ready=0;done_valid=0;host_run=1;
        repeat(20) tick();
        verify(!job_valid && !irq && !s_axil_bvalid && !s_axil_rvalid,"reset state");
        @(negedge host_clk);host_rst_n=1;
        @(negedge clk);rst_n=1;resets++;
        while(!dut.notify_event_ready) tick();
        repeat(3) tick();
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

    always @(posedge clk) begin
        $display("NTRPU_CIRQ_S ps=%0d reset=%0d awv=%0d awr=%0d awa=%08h wv=%0d wr=%0d wd=%08h ws=%h bv=%0d br=%0d bresp=%0d arv=%0d arr=%0d ara=%08h rv=%0d rr=%0d rd=%08h rresp=%0d jv=%0d jr=%0d js=%08h je=%08h dv=%0d dr=%0d ds=%08h de=%08h status=%0d ev=%0d er=%0d ei=%0d sv=%0d sr=%0d reject=%0d tv=%0d tr=%0d ts=%08h te=%08h",
            longint'($realtime*1000.0),rst_n,s_axil_awvalid,s_axil_awready,s_axil_awaddr,s_axil_wvalid,s_axil_wready,s_axil_wdata,s_axil_wstrb,
            s_axil_bvalid,s_axil_bready,s_axil_bresp,s_axil_arvalid,s_axil_arready,s_axil_araddr,s_axil_rvalid,s_axil_rready,
            s_axil_rdata,s_axil_rresp,job_valid,job_ready,job_sequence,job_epoch,done_valid,done_ready,done_sequence,done_epoch,done_status,
            dut.notify_event_valid,dut.notify_event_ready,dut.notify_event_irq,dut.notify_service_valid,dut.notify_service_ready,
            dut.notify_service_rejected,dut.notify_retire_valid,dut.notify_retire_ready,dut.notify_retire_sequence,dut.notify_retire_epoch);
    end
    always @(posedge host_clk) $display("NTRPU_CIRQ_H ps=%0d reset=%0d irq=%0d ack=%0d seq=%08h epoch=%08h",
        longint'($realtime*1000.0),host_rst_n,irq,usr_irq_ack,dut.irq_sequence,dut.irq_epoch);
    task automatic new_job(input logic [31:0] seq,epoch,input bit enable_irq=1,input bit accept=1);
        while(!dut.notify_event_ready) tick();
        write_reg('h20,seq,15,0);write_reg('h24,epoch,15,0);
        write_reg('h2c,'h20000800,15,0);write_reg('h14,32'(enable_irq),15,0);
        write_reg('h10,1,15,0);
        verify(job_valid && job_sequence==seq && job_epoch==epoch,"new descriptor");
        if(accept) accept_job();
    endtask
    task automatic identity(input logic [31:0] seq,epoch);
        write_reg('h38,seq,15,0,1,1);write_reg('h3c,epoch,15,0,2,1);
    endtask
    task automatic host_ack();
        while(!irq) @(negedge host_clk);
        @(negedge host_clk);usr_irq_ack=1;
        repeat(3) @(negedge host_clk);
        usr_irq_ack=0;
    endtask
    task automatic retired();
        while(!dut.notify_event_ready) tick();
        verify(!irq,"retirement drains host IRQ");
        read_reg('hc,'h100);
    endtask
    task automatic normal_finish(input bit enable_irq=1);
        identity(job_sequence,job_epoch);completion();
        write_reg('h10,4,15,0);
        if(enable_irq) host_ack();
        retired();
    endtask
    initial begin
        #1ps;rng=1;
        $display("NTRPU_CIRQ_CONFIG profile=%0d source_half_ps=%0d host_half_ps=%0d fault=%0d",profile,source_half,host_half,inject_early);
        reset_all();
        read_reg(0,'h4652544e);read_reg(4,'h20000);read_reg(8,15);read_reg('hc,'h100);
        read_reg('h38,0);read_reg('h3c,0);read_reg('h64,0);read_reg('h68,0);
        write_reg('h64,9,15,2);write_reg('h68,9,15,2);write_reg('h6c,9,15,3);read_reg('h6c,0,3);
        write_reg('h10,4,15,2);
        for(int k=0;k<16;k++) begin
            value=random_word();mask=0;
            for(int lane=0;lane<4;lane++) mask[lane*8+:8]={8{(k&(1<<lane))!=0}};
            merged=(merged&~mask)|(value&mask);
            if(k==0) merged=0;
            write_reg('h38,value,4'(k),0,k%3,k%5);read_reg('h38,merged);
            write_reg('h3c,value,4'(k),0,k%3,k%5);read_reg('h3c,merged);
        end
        mark("abi2_decode_and_strobes");

        new_job('h10203040,'h50607080);identity(job_sequence,job_epoch);completion();
        write_reg('h10,4,15,0);write_reg('h10,4,15,2);
        repeat(30) tick();verify(irq,"service cannot retire before transport ACK");
        read_reg('hc,'h64);write_reg('h10,1,15,2);read_reg('h4c,'h10203040);
        host_ack();retired();mark("service_before_transport");

        new_job('h10203041,'h50607081);completion();host_ack();repeat(20) tick();
        verify(irq,"transport ACK cannot retire before service");
        write_reg('h14,0,15,0);verify(irq,"mask change cannot withdraw captured IRQ");
        write_reg('h10,1,15,2);read_reg('hc,'h44);
        identity('h10203040,'h50607081);write_reg('h10,4,15,2);
        identity('h10203041,'h50607080);write_reg('h10,4,15,2);
        identity(job_sequence,job_epoch);write_reg('h10,4,15,0);retired();mark("transport_before_service_stale_ack");

        new_job('h10203042,'h50607082,0);completion();write_reg('h14,1,15,0);
        repeat(30) begin tick();verify(!irq,"polling mode captured at completion");end
        identity(job_sequence,job_epoch);write_reg('h10,4,15,0);retired();mark("captured_polling_mode");

        new_job('h10203043,'h50607083);identity(job_sequence,job_epoch);
        @(negedge host_clk);usr_irq_ack=1;repeat(30) tick();
        verify(!dut.notify_event_ready,"old host ACK holds admission");
        completion();write_reg('h10,4,15,0);read_reg('hc,'h24);write_reg('h10,1,15,2);
        repeat(20) begin tick();verify(dut.notify_event_valid && !irq,"unsent event retained after early service");end
        @(negedge host_clk);usr_irq_ack=0;
        host_ack();retired();mark("service_before_event_acceptance");

        for(int fault=1;fault<=2;fault++) begin
            new_job('h10203044+fault,'h50607084+fault);completion(0,fault);
            read_reg('h64,job_sequence);read_reg('h68,job_epoch);read_reg('h4c,done_sequence);read_reg('h50,done_epoch);
            read_reg('h60,32'(fault));identity(done_sequence,done_epoch);write_reg('h10,4,15,2);
            identity(job_sequence,job_epoch);write_reg('h10,4,15,0);host_ack();retired();
        end
        mark("malformed_returned_identity");

        new_job('h10203047,'h50607087,1,0);write_reg('h10,2,15,0);accept_job();
        repeat(5) begin tick();verify(cancel_valid,"queued cancellation held");end
        @(negedge clk);cancel_ready=1;tick();@(negedge clk);cancel_ready=0;
        completion(7);identity(job_sequence,job_epoch);write_reg('h10,4,15,0);host_ack();
        while(!dut.notify_event_ready) tick();read_reg('hc,'h110);mark("cancelled_completion");

        new_job('h10203048,'h50607088);completion(1);read_reg('h40,'h1c01);
        identity(job_sequence,job_epoch);write_reg('h10,4,15,0);host_ack();retired();mark("immediate_error_completion");

        new_job('h10203049,'h50607089);identity(job_sequence,job_epoch);
        @(negedge host_clk);host_run=0;completion();write_reg('h10,4,15,0);
        read_reg('h64,'h10203049);write_reg('h10,1,15,2);repeat(40) tick();
        verify(!irq,"host cannot notify while stopped");host_run=1;host_ack();retired();mark("stopped_host_clock");

        new_job('h1020304a,'h5060708a);identity(job_sequence,job_epoch);completion();host_ack();
        @(negedge clk);s_axil_araddr='h64;s_axil_arvalid=1;tick();@(negedge clk);s_axil_arvalid=0;
        write_reg('h10,4,15,0);while(!dut.notify_event_ready) tick();
        new_job('h1020304b,'h5060708b);
        verify(s_axil_rvalid && s_axil_rdata==='h1020304a,"held read snapshot survives retire/new START");
        @(negedge clk);s_axil_rready=1;tick();@(negedge clk);s_axil_rready=0;
        normal_finish();mark("held_read_across_retirement");

        for(int kind=0;kind<6;kind++) begin
            new_job('h200+kind,'h300+kind,1,kind!=0);
            if(kind>=2) begin identity(job_sequence,job_epoch);completion();end
            if(kind==3) write_reg('h10,4,15,0);
            if(kind==4) host_ack();
            if(kind==5) begin @(negedge host_clk);host_run=0;repeat(25) tick();end
            reset_all();read_reg('h18,0);read_reg('h1c,0);read_reg('hc,'h100);
            new_job('h400+kind,'h500+kind);normal_finish();
            mark($sformatf("reset_phase_%0d",kind));
        end

        new_job('h600,'h700);identity(job_sequence,job_epoch);completion();write_reg('h10,4,15,0);
        // An impossible transport response must fail closed, including if identities disagree.
        force dut.notify_retire_valid=1'b1;force dut.notify_retire_sequence=32'h601;force dut.notify_retire_epoch=32'h700;
        tick();verify(!dut.notify_retire_ready,"wrong retirement refused");
        release dut.notify_retire_valid;release dut.notify_retire_sequence;release dut.notify_retire_epoch;
        write_reg('h10,1,15,2);write_reg('h10,4,15,2);read_reg('hc,'he4);
        host_ack();repeat(60) tick();read_reg('hc,'he4);verify(!dut.notify_retire_ready,"fault remains sticky");
        reset_all();new_job('h602,'h702);normal_finish();mark("wrong_retirement_fault_and_reset");

        new_job('h603,'h703);completion();
        force dut.notify_service_rejected=1'b1;tick();release dut.notify_service_rejected;
        read_reg('hc,'hc4);identity(job_sequence,job_epoch);write_reg('h10,4,15,2);write_reg('h10,1,15,2);
        reset_all();new_job('h604,'h704);normal_finish();mark("service_rejection_fault_and_reset");
        verify(cases==18,"case count");
        $display("NTRPU_CIRQ_PASS cases=%0d reads=%0d writes=%0d checks=%0d resets=%0d jobs=%0d done=%0d cancel=%0d",cases,reads,writes,checks,resets,observed_jobs,observed_done,observed_cancel);
        $finish;
    end
    initial begin #1000000;$fatal(1,"NTRPU_CSR_FAIL watchdog");end
endmodule
