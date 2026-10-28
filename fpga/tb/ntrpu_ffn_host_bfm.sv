// Simulation-only host driver: all device control travels through AXI-Lite.
`timescale 1ns/1ps
module ntrpu_ffn_host_bfm #(parameter integer WATCHDOG_CYCLES=65536,parameter bit OUTPUT_FP32=1) (
`ifdef NTRPU_FFN_TRANSFER
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

    input logic clk,rst_n,
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
    logic irq;

    logic busy=0,cancel_requested=0;
    longint unsigned host_clock=0;
    int host_cases=0,mmio_reads=0,mmio_writes=0;
    int host_aw_first=0,host_w_first=0,host_together=0;
    int host_b_stalls=0,host_r_stalls=0;
    assign job_ready=rst_n && !busy;
    assign cancel_ready=rst_n && busy && !cancel_requested && !done_valid;
    always @(posedge clk) begin
        host_clock=host_clock+1;
        if(!rst_n) cancel_requested<=0;
        else if(job_valid && job_ready) cancel_requested<=0;
        else if(cancel_valid && cancel_ready) cancel_requested<=1;
    end
`ifdef NTRPU_FFN_TRANSFER
    ntrpu_ffn_fabric #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32)) device(.host_clk(clk),.host_rst_n(rst_n),.usr_irq_ack(1'b0),.*);
`else
    ntrpu_ffn_device #(.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.OUTPUT_FP32(OUTPUT_FP32)) device(.host_clk(clk),.host_rst_n(rst_n),.usr_irq_ack(1'b0),.*);
`endif

    task automatic need(input bit ok,input string reason);
        if(!ok) $fatal(1,"FAIL FFN HOST case=%0d: %s",host_cases,reason);
    endtask
    task automatic tick(); @(posedge clk);#1;endtask
    task automatic bus_reset();
        busy=0;
        s_axil_awaddr=0;s_axil_wdata=0;s_axil_araddr=0;s_axil_awprot=0;s_axil_arprot=0;s_axil_wstrb=0;
        s_axil_awvalid=0;s_axil_wvalid=0;s_axil_bready=0;s_axil_arvalid=0;s_axil_rready=0;
        done_valid=0;done_status=0;done_child_status=0;done_stages=0;done_phase=0;done_bytes=0;done_stage_bytes=0;
        done_sequence=0;done_epoch=0;done_hls_status=0;
    endtask
    task automatic send_aw(input logic [31:0] addr,input int delay_cycles);
        repeat(delay_cycles) tick();
        @(negedge clk);s_axil_awaddr=addr;s_axil_awvalid=1;
        while(!s_axil_awready) @(negedge clk);
        tick();@(negedge clk);s_axil_awvalid=0;s_axil_awaddr='hbadbad01;
    endtask
    task automatic send_w(input logic [31:0] data,input int delay_cycles);
        repeat(delay_cycles) tick();
        @(negedge clk);s_axil_wdata=data;s_axil_wstrb=15;s_axil_wvalid=1;
        while(!s_axil_wready) @(negedge clk);
        tick();@(negedge clk);s_axil_wvalid=0;s_axil_wstrb=0;s_axil_wdata='hbadbad02;
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
            need(!s_axil_awready && !s_axil_wready,"no writes behind held B");host_b_stalls++;
        end
        $display("NTRPU_FFN_HOST_WRITE id=%0d addr=%08h data=%08h resp=%0d order=%0d",host_cases,addr,data,s_axil_bresp,order);
        @(negedge clk);s_axil_bready=1;tick();
        need(!s_axil_bvalid,"B consumed");@(negedge clk);s_axil_bready=0;
        mmio_writes++;
        if(order==1) host_aw_first++;else if(order==2) host_w_first++;else host_together++;
    endtask
    task automatic read_reg(input logic [31:0] addr,output logic [31:0] data);
        logic [31:0] saved;
        @(negedge clk);s_axil_araddr=addr;s_axil_arvalid=1;
        while(!s_axil_arready) @(negedge clk);
        tick();@(negedge clk);s_axil_arvalid=0;s_axil_araddr='hbadbad03;
        while(!s_axil_rvalid) tick();
        saved=s_axil_rdata;need(s_axil_rresp===0,"R response");
        repeat(2) begin
            tick();need(s_axil_rvalid && s_axil_rdata===saved && s_axil_rresp===0,"R response stable");
            need(!s_axil_arready,"no AR behind held R");host_r_stalls++;
        end
        data=saved;
        @(negedge clk);s_axil_rready=1;tick();need(!s_axil_rvalid,"R consumed");
        @(negedge clk);s_axil_rready=0;mmio_reads++;
    endtask
    task automatic execute_job();
        logic [31:0] cfg[0:5],buffers[0:7][0:6],values[0:8];
        logic [31:0] state_word,accepted_before,completed_before,count_after;
        longint unsigned start_clock,programmed_clock,completed_clock,compute_cycles;
        int first_read,first_write,polls;
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
        start_clock=host_clock;first_read=mmio_reads;first_write=mmio_writes;cancel_sent=0;polls=0;
        read_reg(0,state_word);need(state_word=='h4652544e,"device identity");
        read_reg(4,state_word);need(state_word=='h00010000,"device ABI");
        read_reg(8,state_word);need(state_word==(OUTPUT_FP32 ? 7:6),"output format capability");
        read_reg('h18,accepted_before);read_reg('h1c,completed_before);
        write_reg('h14,1);
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
        need(irq,"completion IRQ asserted");
        // A new command cannot overwrite a completion while the host is collecting it.
        write_reg('h10,1,2);
        for(int i=0;i<9;i++) read_reg('h40+i*4,values[i]);
        need(values[8]==0,"completion identity/metadata diagnostic");
        need(values[3]===cfg[0] && values[4]===cfg[1],"MMIO completion identity");
        compute_cycles={values[7],values[6]};
        need(compute_cycles>0,"compute elapsed counter");
        read_reg('h18,count_after);need(count_after===accepted_before+32'd1,"one compute acceptance");
        read_reg('h1c,count_after);need(count_after===completed_before+32'd1,"one compute completion");
        done_status=values[0][3:0];done_child_status=values[0][7:4];done_stages=values[0][10:8];done_phase=values[0][12:11];
        done_bytes=values[1][16:0];done_stage_bytes=values[2][17:0];done_sequence=values[3];done_epoch=values[4];done_hls_status=values[5];
        done_valid=1;
        do begin @(posedge clk);end while(!done_ready);
        #1;done_valid=0;
        need(irq,"completion IRQ retained until ACK");
        write_reg('h10,4);need(!irq,"ACK clears IRQ");
        read_reg('hc,state_word);need((state_word&15)==0,"device idle after ACK");
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
