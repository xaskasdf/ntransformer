// Simulation-only host driver: all device control travels through AXI-Lite.
`timescale 1ns/1ps
module ntrpu_ffn_guarded_host_bfm #(parameter integer WATCHDOG_CYCLES=65536,parameter bit OUTPUT_FP32=1) (
`ifdef NTRPU_FFN_TRANSFER
    input wire host_dma_held,host_dma_ready,
    input logic[63:0] s_host_axi_araddr, s_host_axi_awaddr,
    input logic[7:0] s_host_axi_arlen, s_host_axi_awlen,
    input logic[2:0] s_host_axi_arsize, s_host_axi_arprot, s_host_axi_awsize, s_host_axi_awprot,
    input logic[1:0] s_host_axi_arburst, s_host_axi_awburst,
    input logic[3:0] s_host_axi_arid, s_host_axi_arcache, s_host_axi_awid, s_host_axi_awcache,
    input logic s_host_axi_arlock, s_host_axi_arvalid, s_host_axi_awlock, s_host_axi_awvalid,
    output logic s_host_axi_arready, s_host_axi_awready,
    output logic[511:0] s_host_axi_rdata,
    output logic[1:0] s_host_axi_rresp,
    output logic[3:0] s_host_axi_rid,
    output logic s_host_axi_rlast, s_host_axi_rvalid,
    input logic s_host_axi_rready,
    input logic[511:0] s_host_axi_wdata,
    input logic[63:0] s_host_axi_wstrb,
    input logic s_host_axi_wlast, s_host_axi_wvalid,
    output logic s_host_axi_wready,
    output logic[3:0] s_host_axi_bid,
    output logic[1:0] s_host_axi_bresp,
    output logic s_host_axi_bvalid,
    input logic s_host_axi_bready,
`endif

    input logic clk,host_clk,mgmt_clk,external_reset_async,rearm,
    input wire memory_peer_held,memory_peer_ready,
    output wire capacity_guard_idle,
    output wire rst_n,host_rst_n,device_admit,host_admit,running,
    output wire peers_reset_req,rearm_ready,peers_held,peers_ready,

    input logic job_valid,
    output logic job_ready,
    input logic [1:0] job_layout,
    input logic [13:0] job_dim,job_hidden,
    input logic [5:0] job_tile_k [0:2],
    input logic [63:0] job_addr [0:7],job_base [0:7],job_bytes [0:7],
    input logic [31:0] job_buffer_epoch [0:7],
    input logic [31:0] job_sequence,job_epoch,expected_epoch,
    input logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input logic done_ready,
    output logic [3:0] done_status,done_child_status,
    output logic [2:0] done_stages,
    output logic [1:0] done_phase,
    output logic [16:0] done_bytes,
    output logic [17:0] done_stage_bytes,
    output logic [31:0] done_sequence,done_epoch,done_hls_status,
    output logic activation_selected,
    output logic [63:0] m_axi_araddr,
    output logic [7:0] m_axi_arlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot,
    output logic [1:0] m_axi_arburst,
    output logic [3:0] m_axi_arid, m_axi_arcache,
    output logic m_axi_arlock, m_axi_arvalid,
    input logic m_axi_arready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid, m_axi_awcache,
    output logic m_axi_awlock, m_axi_awvalid,
    input logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready
 );
    logic [31:0] s_axil_awaddr,s_axil_wdata,s_axil_araddr;
    logic [2:0] s_axil_awprot,s_axil_arprot;
    logic [3:0] s_axil_wstrb;
    logic s_axil_awvalid,s_axil_wvalid,s_axil_bready,s_axil_arvalid,s_axil_rready;
    logic s_axil_awready,s_axil_wready,s_axil_bvalid,s_axil_arready,s_axil_rvalid;
    logic [1:0] s_axil_bresp,s_axil_rresp;
    logic [31:0] s_axil_rdata;
`ifdef NTRPU_FFN_IRQ
    localparam bit ABI2=1;
`else
    localparam bit ABI2=0;
`endif
    wire [36:0] guarded_araddr,guarded_awaddr;
    assign m_axi_araddr={27'b0,guarded_araddr};
    assign m_axi_awaddr={27'b0,guarded_awaddr};
    wire irq,raw_irq;
    logic usr_irq_ack=0;
    bit inject_irq_early=0,irq_ack_seen=0;
    initial inject_irq_early=$test$plusargs("INJECT_IRQ_EARLY");
    always @(posedge host_clk) begin
        if(!host_rst_n || !raw_irq) irq_ack_seen<=0;
        else if(usr_irq_ack) irq_ack_seen<=1;
    end
    assign irq=raw_irq && !(ABI2 && inject_irq_early && irq_ack_seen);


    logic busy=0,cancel_requested=0;
    longint unsigned host_clock=0;
    int host_cases=0,mmio_reads=0,mmio_writes=0;
    int host_aw_first=0,host_w_first=0,host_together=0;
    int host_b_stalls=0,host_r_stalls=0;
    assign job_ready=rst_n && device_admit && host_admit && !busy;
    assign cancel_ready=rst_n && device_admit && busy && !cancel_requested && !done_valid;
    always @(posedge clk) begin
        host_clock=host_clock+1;
        if(!rst_n) cancel_requested<=0;
        else if(job_valid && job_ready) cancel_requested<=0;
        else if(cancel_valid && cancel_ready) cancel_requested<=1;
    end
`ifdef NTRPU_FFN_TRANSFER
    ntrpu_ffn_guarded_fabric #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32)) device(.m_axi_araddr(guarded_araddr),.m_axi_awaddr(guarded_awaddr),.irq(raw_irq),.*);
`else
    ntrpu_ffn_guarded_fabric #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32)) device(.m_axi_araddr(guarded_araddr),.m_axi_awaddr(guarded_awaddr),
        .s_host_axi_araddr('0),
        .s_host_axi_awaddr('0),
        .s_host_axi_arlen('0),
        .s_host_axi_awlen('0),
        .s_host_axi_arsize('0),
        .s_host_axi_arprot('0),
        .s_host_axi_awsize('0),
        .s_host_axi_awprot('0),
        .s_host_axi_arburst('0),
        .s_host_axi_awburst('0),
        .s_host_axi_arid('0),
        .s_host_axi_arcache('0),
        .s_host_axi_awid('0),
        .s_host_axi_awcache('0),
        .s_host_axi_arlock('0),
        .s_host_axi_arvalid('0),
        .s_host_axi_awlock('0),
        .s_host_axi_awvalid('0),
        .s_host_axi_arready(),
        .s_host_axi_awready(),
        .s_host_axi_rdata(),
        .s_host_axi_rresp(),
        .s_host_axi_rid(),
        .s_host_axi_rlast(),
        .s_host_axi_rvalid(),
        .s_host_axi_rready('0),
        .s_host_axi_wdata('0),
        .s_host_axi_wstrb('0),
        .s_host_axi_wlast('0),
        .s_host_axi_wvalid('0),
        .s_host_axi_wready(),
        .s_host_axi_bid(),
        .s_host_axi_bresp(),
        .s_host_axi_bvalid(),
        .s_host_axi_bready('0),.irq(raw_irq),.*);
`endif

    // Simulation peer acknowledgement is derived from cleared protocol state.
    // It is not a hardware CDC substitute: the synthesizable wrapper receives
    // these registered local observations through its own level qualifiers.
    wire host_alive,device_alive;
    logic host_peer_held=0,host_peer_ready=0;
`ifdef NTRPU_FFN_TRANSFER
    wire dma_drained=host_dma_held && !s_host_axi_awvalid && !s_host_axi_wvalid &&
                     !s_host_axi_arvalid && !s_host_axi_bready && !s_host_axi_rready;
    wire dma_operational=host_dma_ready;
`else
    wire dma_drained=1'b1,dma_operational=1'b1;
`endif
    int host_quiet=0,host_up=0;
    always @(posedge host_clk) begin
        if(!host_rst_n && !busy && !s_axil_awvalid && !s_axil_wvalid && !s_axil_arvalid &&
           !s_axil_bready && !s_axil_rready && !usr_irq_ack && dma_drained) begin
            if(host_quiet==7) host_peer_held<=1;
            else host_quiet<=host_quiet+1;
        end else begin host_quiet<=0;host_peer_held<=0;end
        if(host_rst_n && dma_operational) begin
            if(host_up==7) host_peer_ready<=1;else host_up<=host_up+1;
        end else begin host_up<=0;host_peer_ready<=0;end
    end

    task automatic need(input bit ok,input string reason);
        if(!ok) $fatal(1,"FAIL FFN HOST case=%0d: %s",host_cases,reason);
    endtask
    task automatic tick(); @(posedge host_clk);#1;endtask
    task automatic bus_reset();
        busy=0;usr_irq_ack=0;
        s_axil_awaddr=0;s_axil_wdata=0;s_axil_araddr=0;s_axil_awprot=0;s_axil_arprot=0;s_axil_wstrb=0;
        s_axil_awvalid=0;s_axil_wvalid=0;s_axil_bready=0;s_axil_arvalid=0;s_axil_rready=0;
        done_valid=0;done_status=0;done_child_status=0;done_stages=0;done_phase=0;done_bytes=0;done_stage_bytes=0;
        done_sequence=0;done_epoch=0;done_hls_status=0;
    endtask
    task automatic send_aw(input logic [31:0] addr,input int delay_cycles);
        repeat(delay_cycles) tick();
        @(negedge host_clk);s_axil_awaddr=addr;s_axil_awvalid=1;
        while(!s_axil_awready) @(negedge host_clk);
        tick();@(negedge host_clk);s_axil_awvalid=0;s_axil_awaddr='hbadbad01;
    endtask
    task automatic send_w(input logic [31:0] data,input int delay_cycles);
        repeat(delay_cycles) tick();
        @(negedge host_clk);s_axil_wdata=data;s_axil_wstrb=15;s_axil_wvalid=1;
        while(!s_axil_wready) @(negedge host_clk);
        tick();@(negedge host_clk);s_axil_wvalid=0;s_axil_wstrb=0;s_axil_wdata='hbadbad02;
    endtask
    task automatic write_reg(input logic [31:0] addr,data,input logic [1:0] response=0);
        int order;
        order=mmio_writes%3;
        fork
            send_aw(addr,order==2 ? 2:0);
            send_w(data,order==1 ? 2:0);
        join
        while(!s_axil_bvalid) tick();
        repeat(2) begin
            tick();need(s_axil_bvalid && s_axil_bresp===response,"B response/stability");
            host_b_stalls++; // The converter may accept a later request while B is retained.
        end
        @(negedge host_clk);s_axil_bready=1;
        @(posedge host_clk);
        need(s_axil_bvalid && s_axil_bresp===response,"B handshake");
        // Log and count atomically at the observed handshake. Reset may abort
        // the task before its next edge, so counting later loses evidence.
        mmio_writes++;
        $display("NTRPU_FFN_HOST_WRITE id=%0d addr=%08h data=%08h resp=%0d order=%0d",host_cases,addr,data,s_axil_bresp,order);
        $display("NTRPU_SFFN_B id=%0d ps=%0d total=%0d addr=%08h data=%08h resp=%0d order=%0d",host_cases,longint'($realtime*1000),mmio_writes,addr,data,s_axil_bresp,order);
        #1;
        need(!s_axil_bvalid,"B consumed");@(negedge host_clk);s_axil_bready=0;
        if(order==1) host_aw_first++;else if(order==2) host_w_first++;else host_together++;
    endtask
    task automatic read_reg(input logic [31:0] addr,output logic [31:0] data);
        logic [31:0] saved;
        @(negedge host_clk);s_axil_araddr=addr;s_axil_arvalid=1;
        while(!s_axil_arready) @(negedge host_clk);
        tick();@(negedge host_clk);s_axil_arvalid=0;s_axil_araddr='hbadbad03;
        while(!s_axil_rvalid) tick();
        saved=s_axil_rdata;need(s_axil_rresp===0,"R response");
        repeat(2) begin
            tick();need(s_axil_rvalid && s_axil_rdata===saved && s_axil_rresp===0,"R response stable");
            host_r_stalls++; // READY here belongs to the CDC endpoint, not directly to CSR.
        end
        data=saved;
        @(negedge host_clk);s_axil_rready=1;
        @(posedge host_clk);
        need(s_axil_rvalid && s_axil_rdata===saved && s_axil_rresp===0,"R handshake");
        mmio_reads++;
        if(ABI2) $display("NTRPU_FFN_IRQ_READ id=%0d addr=%08h data=%08h time_ps=%0d",host_cases,addr,data,longint'($realtime*1000));
        $display("NTRPU_SFFN_R id=%0d ps=%0d total=%0d addr=%08h data=%08h resp=%0d",host_cases,longint'($realtime*1000),mmio_reads,addr,data,s_axil_rresp);
        #1;need(!s_axil_rvalid,"R consumed");
        @(negedge host_clk);s_axil_rready=0;
    endtask
    task automatic transport_ack();
        need(irq,"transport ACK requires active request");
        @(negedge host_clk);usr_irq_ack=1;
        @(negedge host_clk);usr_irq_ack=0;
    endtask
    // Log actual handshakes at compute/CSR/notifier boundaries, never FSM state.
    generate if(ABI2) begin : evidence
        bit previous_irq=0;
        always @(posedge host_clk) begin
            if(!host_rst_n) previous_irq=0;
            else begin
                if(irq!=previous_irq) $display("NTRPU_FFN_IRQ_EDGE id=%0d level=%0d time_ps=%0d",host_cases,irq,longint'($realtime*1000));
                if(irq && usr_irq_ack) $display("NTRPU_FFN_IRQ_ACK id=%0d time_ps=%0d",host_cases,longint'($realtime*1000));
                previous_irq=irq;
            end
        end
        always @(posedge clk) if(rst_n) begin
            if(device.compute.fabric.fabric.device.job_valid && device.compute.fabric.fabric.device.job_ready)
                $display("NTRPU_FFN_IRQ_JOB id=%0d sequence=%08h epoch=%08h time_ps=%0d",host_cases,device.compute.fabric.fabric.device.job_sequence,device.compute.fabric.fabric.device.job_epoch,longint'($realtime*1000));
            if(device.compute.fabric.fabric.device.done_valid && device.compute.fabric.fabric.device.done_ready)
                $display("NTRPU_FFN_IRQ_DONE id=%0d sequence=%08h epoch=%08h status=%0d stages=%0d bytes=%0d stage_bytes=%0d time_ps=%0d",host_cases,device.compute.fabric.fabric.device.done_sequence,device.compute.fabric.fabric.device.done_epoch,device.compute.fabric.fabric.device.done_status,device.compute.fabric.fabric.device.done_stages,device.compute.fabric.fabric.device.done_bytes,device.compute.fabric.fabric.device.done_stage_bytes,longint'($realtime*1000));
            if(device.compute.fabric.fabric.device.with_irq.control.notify_event_valid && device.compute.fabric.fabric.device.with_irq.control.notify_event_ready)
                $display("NTRPU_FFN_IRQ_EVENT id=%0d sequence=%08h epoch=%08h enable=%0d time_ps=%0d",host_cases,device.compute.fabric.fabric.device.with_irq.control.notify_sequence,device.compute.fabric.fabric.device.with_irq.control.notify_epoch,device.compute.fabric.fabric.device.with_irq.control.notify_event_irq,longint'($realtime*1000));
            if(device.compute.fabric.fabric.device.with_irq.control.notify_service_valid && device.compute.fabric.fabric.device.with_irq.control.notify_service_ready)
                $display("NTRPU_FFN_IRQ_SERVICE id=%0d sequence=%08h epoch=%08h time_ps=%0d",host_cases,device.compute.fabric.fabric.device.with_irq.control.notify_sequence,device.compute.fabric.fabric.device.with_irq.control.notify_epoch,longint'($realtime*1000));
            if(device.compute.fabric.fabric.device.with_irq.control.notify_retire_valid && device.compute.fabric.fabric.device.with_irq.control.notify_retire_ready)
                $display("NTRPU_FFN_IRQ_RETIRE id=%0d sequence=%08h epoch=%08h time_ps=%0d",host_cases,device.compute.fabric.fabric.device.with_irq.control.notify_retire_sequence,device.compute.fabric.fabric.device.with_irq.control.notify_retire_epoch,longint'($realtime*1000));
            need(!device.compute.fabric.fabric.device.with_irq.control.notify_service_rejected,"unexpected service rejection");
        end
    end endgenerate
    task automatic execute_job();
        logic [31:0] cfg[0:5],buffers[0:7][0:6],values[0:8];
        logic [31:0] state_word,accepted_before,completed_before,count_after;
        longint unsigned start_clock,programmed_clock,completed_clock,compute_cycles;
        int first_read,first_write,polls,irq_mode,retire_polls;
        bit cancel_sent;
        // Capture the host descriptor at its handshake; the outer TB poisons it next cycle.
        cfg[0]=job_sequence;cfg[1]=job_epoch;cfg[2]=expected_epoch;
        cfg[3]={2'b0,job_hidden,2'b0,job_dim};cfg[4]={30'b0,job_layout};
        cfg[5]={8'b0,2'b0,job_tile_k[2],2'b0,job_tile_k[1],2'b0,job_tile_k[0]};
        for(int b=0;b<8;b++) begin
            buffers[b][0]=job_addr[b][31:0];buffers[b][1]=job_addr[b][63:32];
            buffers[b][2]=job_base[b][31:0];buffers[b][3]=job_base[b][63:32];
            buffers[b][4]=job_bytes[b][31:0];buffers[b][5]=job_bytes[b][63:32];
            buffers[b][6]=job_buffer_epoch[b];
        end
        irq_mode=ABI2 ? (host_cases-1)%3:0;retire_polls=0;
        start_clock=host_clock;first_read=mmio_reads;first_write=mmio_writes;cancel_sent=0;polls=0;
        read_reg(0,state_word);need(state_word=='h4652544e,"device identity");
        read_reg(4,state_word);need(state_word==(ABI2 ? 'h00020000:'h00010000),"device ABI");
        read_reg(8,state_word);need(state_word==((OUTPUT_FP32 ? 7:6)|(ABI2 ? 8:0)),"output format capability");
        read_reg('h18,accepted_before);read_reg('h1c,completed_before);
        if(ABI2) begin
            read_reg('hc,state_word);need(state_word[8] && !(state_word&'h87),"notifier ready before START");
            $display("NTRPU_FFN_IRQ_BEGIN id=%0d mode=%0d sequence=%08h epoch=%08h time_ps=%0d fault=%0d",host_cases,irq_mode,cfg[0],cfg[1],longint'($realtime*1000),inject_irq_early);
        end
        write_reg('h14,irq_mode==2 ? 0:1);
        for(int i=0;i<6;i++) write_reg('h20+i*4,cfg[i]);
        for(int b=0;b<8;b++) for(int f=0;f<7;f++) write_reg('h100+b*32+f*4,buffers[b][f]);
        write_reg('h10,1);programmed_clock=host_clock;
        // Update every shadow after START; the active descriptor must remain immutable.
        for(int i=0;i<6;i++) write_reg('h20+i*4,0);
        for(int b=0;b<8;b++) for(int f=0;f<7;f++) write_reg('h100+b*32+f*4,0);
        do begin
            if(cancel_requested && !cancel_sent) begin write_reg('h10,2);cancel_sent=1;end
            read_reg('hc,state_word);polls++;
        end while(!state_word[2]);
        completed_clock=host_clock;
        if(irq_mode!=2) begin
            int wait_irq;wait_irq=0;
            while(!irq) begin tick();wait_irq++;need(wait_irq<128,"completion IRQ asserted");end
        end else need(!irq,"polling completion has no IRQ");
        if(!ABI2) $display("NTRPU_ASYNC_IRQ_SEEN id=%0d time_ps=%0d",host_cases,longint'($realtime*1000));
        if(ABI2 && irq_mode==0) begin
            transport_ack();
            repeat(24) begin tick();need(irq,"transport ACK must retain IRQ until software service");end
            write_reg('h14,0);need(irq,"captured IRQ survives mask change");
            write_reg('h38,cfg[0]^1);write_reg('h3c,cfg[1]);write_reg('h10,4,2);
        end
        if(ABI2 && irq_mode==2) begin write_reg('h14,1);need(!irq,"captured polling survives unmask");end
        // A new command cannot overwrite a completion while the host is collecting it.
        write_reg('h10,1,2);
        for(int i=0;i<9;i++) read_reg('h40+i*4,values[i]);
        need(values[8]==0,"completion identity/metadata diagnostic");
        need(values[3]===cfg[0] && values[4]===cfg[1],"MMIO completion identity");
        if(ABI2) begin
            read_reg('h64,state_word);need(state_word==cfg[0],"admitted sequence");
            read_reg('h68,state_word);need(state_word==cfg[1],"admitted epoch");
        end
        compute_cycles={values[7],values[6]};
        need(compute_cycles>0,"compute elapsed counter");
        read_reg('h18,count_after);need(count_after===accepted_before+32'd1,"one compute acceptance");
        read_reg('h1c,count_after);need(count_after===completed_before+32'd1,"one compute completion");
        done_status=values[0][3:0];done_child_status=values[0][7:4];done_stages=values[0][10:8];done_phase=values[0][12:11];
        done_bytes=values[1][16:0];done_stage_bytes=values[2][17:0];done_sequence=values[3];done_epoch=values[4];done_hls_status=values[5];
        done_valid=1;
        do begin @(posedge clk);end while(!done_ready);
        #1;done_valid=0;
        if(ABI2) begin
            need(irq==(irq_mode!=2),"captured notification retained through result consumption");
            write_reg('h38,cfg[0]);write_reg('h3c,cfg[1]);
            $display("NTRPU_FFN_IRQ_SERVICE_REQUEST id=%0d time_ps=%0d",host_cases,longint'($realtime*1000));
            write_reg('h10,4);
            if(irq_mode==1) begin
                read_reg('hc,state_word);need(state_word[2] && state_word[5],"software ACK retains completion until transport");
                write_reg('h10,1,2);write_reg('h10,4,2);
                repeat(24) begin tick();need(irq,"software service must retain IRQ until transport ACK");end
                transport_ack();
            end
            do begin
                read_reg('hc,state_word);retire_polls++;need(retire_polls<128,"retirement timeout");
                need(!state_word[7],"notifier fault");
            end while((state_word&7)!=0 || !state_word[8]);
            need(!irq,"retirement drains IRQ");
            $display("NTRPU_FFN_IRQ_END id=%0d mode=%0d retire_polls=%0d time_ps=%0d",host_cases,irq_mode,retire_polls,longint'($realtime*1000));
        end else begin
        need(irq,"completion IRQ retained until ACK");
        write_reg('h10,4);
        begin
            int irq_wait;irq_wait=0;
            while(irq) begin tick();irq_wait++;need(irq_wait<16,"ACK clears synchronized IRQ");end
            $display("NTRPU_ASYNC_IRQ_ACK id=%0d host_wait=%0d time_ps=%0d",host_cases,irq_wait,longint'($realtime*1000));
        end
        read_reg('hc,state_word);need((state_word&15)==0,"device idle after ACK");
        end
        $display("NTRPU_FFN_HOST_CASE id=%0d reset=0 status=%0d stages=%0d bytes=%0d stage_bytes=%0d sequence=%08h epoch=%08h flags=%08h program_cycles=%0d compute_cycles=%0d poll_cycles=%0d total_cycles=%0d reads=%0d writes=%0d polls=%0d cancel=%0d",
            host_cases,done_status,done_stages,done_bytes,done_stage_bytes,done_sequence,done_epoch,values[8],
            programmed_clock-start_clock,compute_cycles,completed_clock-programmed_clock,host_clock-start_clock,mmio_reads-first_read,mmio_writes-first_write,polls,cancel_sent);
        busy<=0;
    endtask
    initial begin
        bus_reset();
        forever begin
            wait(rst_n===1'b1);
            fork
                begin
                    forever begin
                        @(posedge clk);
                        if(job_valid && job_ready) begin busy<=1;host_cases++;execute_job();end
                    end
                end
                begin @(negedge rst_n);end
            join_any
            disable fork;
            if(busy) $display("NTRPU_FFN_HOST_CASE id=%0d reset=1",host_cases);
            bus_reset();
        end
    end
endmodule
