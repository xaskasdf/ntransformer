// Simulation-only XDMA application: unchanged FFN + bounded behavioral RAM.
`timescale 1ns/1ps
module ntrpu_pcie_ffn_app (
    input logic  host_clk,
    input logic  pcie_resetn,
    input logic  usr_irq_ack,
    input logic [31:0] s_axil_awaddr,
    input logic [31:0] s_axil_wdata,
    input logic [31:0] s_axil_araddr,
    input logic [2:0] s_axil_awprot,
    input logic [2:0] s_axil_arprot,
    input logic [3:0] s_axil_wstrb,
    input logic  s_axil_awvalid,
    input logic  s_axil_wvalid,
    input logic  s_axil_bready,
    input logic  s_axil_arvalid,
    input logic  s_axil_rready,
    output logic  s_axil_awready,
    output logic  s_axil_wready,
    output logic  s_axil_bvalid,
    output logic  s_axil_arready,
    output logic  s_axil_rvalid,
    output logic [1:0] s_axil_bresp,
    output logic [1:0] s_axil_rresp,
    output logic [31:0] s_axil_rdata,
    output logic  irq,
    input logic [63:0] s_host_axi_araddr,
    input logic [63:0] s_host_axi_awaddr,
    input logic [7:0] s_host_axi_arlen,
    input logic [7:0] s_host_axi_awlen,
    input logic [2:0] s_host_axi_arsize,
    input logic [2:0] s_host_axi_arprot,
    input logic [2:0] s_host_axi_awsize,
    input logic [2:0] s_host_axi_awprot,
    input logic [1:0] s_host_axi_arburst,
    input logic [1:0] s_host_axi_awburst,
    input logic [3:0] s_host_axi_arid,
    input logic [3:0] s_host_axi_arcache,
    input logic [3:0] s_host_axi_awid,
    input logic [3:0] s_host_axi_awcache,
    input logic  s_host_axi_arlock,
    input logic  s_host_axi_arvalid,
    input logic  s_host_axi_awlock,
    input logic  s_host_axi_awvalid,
    output logic  s_host_axi_arready,
    output logic  s_host_axi_awready,
    output logic [511:0] s_host_axi_rdata,
    output logic [1:0] s_host_axi_rresp,
    output logic [3:0] s_host_axi_rid,
    output logic  s_host_axi_rlast,
    output logic  s_host_axi_rvalid,
    input logic  s_host_axi_rready,
    input logic [511:0] s_host_axi_wdata,
    input logic [63:0] s_host_axi_wstrb,
    input logic  s_host_axi_wlast,
    input logic  s_host_axi_wvalid,
    output logic  s_host_axi_wready,
    output logic [3:0] s_host_axi_bid,
    output logic [1:0] s_host_axi_bresp,
    output logic  s_host_axi_bvalid,
    input logic  s_host_axi_bready
);
    logic clk=0,rst_n,host_rst_n;
    logic [63:0] host_reset_pipe=0,device_reset_pipe=0;
    always #2.5 clk=~clk;
    always @(posedge host_clk or negedge pcie_resetn)
        if(!pcie_resetn) host_reset_pipe<=0;else host_reset_pipe<={host_reset_pipe[62:0],1'b1};
    always @(posedge clk or negedge pcie_resetn)
        if(!pcie_resetn) device_reset_pipe<=0;else device_reset_pipe<={device_reset_pipe[62:0],1'b1};
    assign rst_n=device_reset_pipe[63];
    assign host_rst_n=host_reset_pipe[63];
    logic  activation_selected;
    logic [63:0] m_axi_araddr;
    logic [7:0] m_axi_arlen;
    logic [2:0] m_axi_arsize;
    logic [2:0] m_axi_arprot;
    logic [1:0] m_axi_arburst;
    logic [3:0] m_axi_arid;
    logic [3:0] m_axi_arcache;
    logic  m_axi_arlock;
    logic  m_axi_arvalid;
    logic  m_axi_arready;
    logic [511:0] m_axi_rdata;
    logic [1:0] m_axi_rresp;
    logic [3:0] m_axi_rid;
    logic  m_axi_rlast;
    logic  m_axi_rvalid;
    logic  m_axi_rready;
    logic [63:0] m_axi_awaddr;
    logic [7:0] m_axi_awlen;
    logic [2:0] m_axi_awsize;
    logic [2:0] m_axi_awprot;
    logic [1:0] m_axi_awburst;
    logic [3:0] m_axi_awid;
    logic [3:0] m_axi_awcache;
    logic  m_axi_awlock;
    logic  m_axi_awvalid;
    logic  m_axi_awready;
    logic [511:0] m_axi_wdata;
    logic [63:0] m_axi_wstrb;
    logic  m_axi_wlast;
    logic  m_axi_wvalid;
    logic  m_axi_wready;
    logic [3:0] m_axi_bid;
    logic [1:0] m_axi_bresp;
    logic  m_axi_bvalid;
    logic  m_axi_bready;
    ntrpu_ffn_async_fabric #(.OUTPUT_FP32(1),.EXTERNAL_RETIRE(1)) device(.*);

    byte unsigned memory [0:7][0:5119];
    bit written [0:7][0:5119];
    int cycle=0,read_left=0,write_left=0,b_wait=0;
    int input_bytes=0,core_bytes=0,result_read_bytes=0;
    bit reading=0,writing=0,b_pending=0,clear_r=0,clear_b=0;
    logic [63:0] read_address,write_address;
    logic [3:0] read_id,write_id;
    function automatic int locate(input logic [63:0] address);
        for(int i=0;i<8;i++)
            if(address>=64'h100000000+64'(i)*64'h20000000+4032 &&
               address<64'h100000000+64'(i)*64'h20000000+5120) return i;
        $fatal(1,"PCIE_FFN_RAM_UNMAPPED addr=%h",address);return -1;
    endfunction
    function automatic int offset(input logic [63:0] address,input int b);
        return int'(address-(64'h100000000+64'(b)*64'h20000000));
    endfunction
    function automatic int q(input int row);
        if(row%32==0) return 127;
        if(row%32==1) return 1;
        return (row%32)*7-113;
    endfunction
    function automatic byte unsigned expected(input int b,off);
        logic [63:0] bits;
        int r,k;
        k=off%64;r=off/(b==7 ? 4:8);
        if(b==4) bits=$realtobits(257.0);
        else if(b==5) bits=$realtobits(real'(q(r)));
        else if(b==6) begin
            bits=$realtobits(257.0);
            if(k<32) return 8'(q(k));
            if(k<40) return bits[(k-32)*8+:8];
            return 0;
        end else bits={32'b0,$shortrealtobits(shortreal'(257*q((r*7)%32)))};
        return bits[(off%(b==7 ? 4:8))*8+:8];
    endfunction
    initial begin
        for(int b=0;b<8;b++) for(int i=0;i<5120;i++) begin memory[b][i]='ha5;written[b][i]=0;end
        m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;
        m_axi_rvalid=0;m_axi_bvalid=0;m_axi_rdata=0;m_axi_rresp=0;m_axi_rid=0;m_axi_rlast=0;m_axi_bid=0;m_axi_bresp=0;
    end
    always @(negedge clk) begin
        if(clear_r) begin m_axi_rvalid=0;clear_r=0;end
        if(clear_b) begin m_axi_bvalid=0;clear_b=0;end
        if(!rst_n) begin
            m_axi_arready=0;m_axi_awready=0;m_axi_wready=0;m_axi_rvalid=0;m_axi_bvalid=0;
        end else begin
            m_axi_arready=!reading && cycle%5!=0;
            m_axi_awready=!writing && !b_pending && !m_axi_bvalid && cycle%5!=1;
            m_axi_wready=writing && cycle%7!=0;
            if(reading && !m_axi_rvalid && cycle%7!=1) begin
                for(int lane=0;lane<64;lane++) begin
                    if(!written[locate(read_address)][offset(read_address,locate(read_address))+lane])
                        $fatal(1,"PCIE_FFN_READ_UNINITIALIZED addr=%h",read_address+lane);
                    m_axi_rdata[lane*8+:8]=memory[locate(read_address)][offset(read_address,locate(read_address))+lane];
                end
                m_axi_rvalid=1;m_axi_rid=read_id;m_axi_rresp=0;m_axi_rlast=read_left==1;
            end
            if(b_pending && b_wait==0 && !m_axi_bvalid) begin
                m_axi_bvalid=1;m_axi_bid=write_id;m_axi_bresp=0;
            end
        end
    end
    always @(posedge clk) begin
        if(!rst_n) begin cycle=0;reading=0;writing=0;read_left=0;write_left=0;b_pending=0;b_wait=0;end
        else begin
            cycle++;
            if(b_wait>0) b_wait--;
            if(m_axi_bvalid && m_axi_bready) begin clear_b=1;b_pending=0;end
            if(m_axi_rvalid && m_axi_rready) begin
                $display("NTRPU_PCIE_FFN_RAM_R addr=%016h id=%0d data=%0128h time_ps=%0d",read_address,m_axi_rid,m_axi_rdata,longint'($realtime*1000));
                if(locate(read_address)==7) result_read_bytes+=64;
                read_address+=64;read_left--;clear_r=1;if(read_left==0) reading=0;
            end
            if(m_axi_arvalid && m_axi_arready) begin
                if(reading || m_axi_arsize!=6 || m_axi_arburst!=1 || m_axi_arlock || m_axi_araddr[5:0]!=0 ||
                   int'(m_axi_araddr[11:0])+(int'(m_axi_arlen)+1)*64>4096) $fatal(1,"PCIE_FFN_RAM_AR");
                reading=1;read_left=int'(m_axi_arlen)+1;read_address=m_axi_araddr;read_id=m_axi_arid;
                $display("NTRPU_PCIE_FFN_RAM_AR addr=%016h beats=%0d id=%0d time_ps=%0d",read_address,read_left,read_id,longint'($realtime*1000));
            end
            if(m_axi_awvalid && m_axi_awready) begin
                if(writing || b_pending || m_axi_awsize!=6 || m_axi_awburst!=1 || m_axi_awlock || m_axi_awaddr[5:0]!=0 ||
                   int'(m_axi_awaddr[11:0])+(int'(m_axi_awlen)+1)*64>4096)
                    $fatal(1,"PCIE_FFN_RAM_AW addr=%h len=%0d size=%0d burst=%0d lock=%0d writing=%0d pending=%0d",m_axi_awaddr,m_axi_awlen,m_axi_awsize,m_axi_awburst,m_axi_awlock,writing,b_pending);
                writing=1;write_left=int'(m_axi_awlen)+1;write_address=m_axi_awaddr;write_id=m_axi_awid;
                $display("NTRPU_PCIE_FFN_RAM_AW addr=%016h beats=%0d id=%0d time_ps=%0d",write_address,write_left,write_id,longint'($realtime*1000));
            end
            if(m_axi_wvalid && m_axi_wready) begin
                if(!writing || m_axi_wstrb!=='1 || m_axi_wlast!==(write_left==1) || $isunknown(m_axi_wdata)) $fatal(1,"PCIE_FFN_RAM_W");
                for(int lane=0;lane<64;lane++) begin
                    if(locate(write_address)>=4 && m_axi_wdata[lane*8+:8]!==expected(locate(write_address),offset(write_address,locate(write_address))+lane-4032))
                        $fatal(1,"PCIE_FFN_NUMERIC addr=%h lane=%0d",write_address,lane);
                    memory[locate(write_address)][offset(write_address,locate(write_address))+lane]=m_axi_wdata[lane*8+:8];
                    written[locate(write_address)][offset(write_address,locate(write_address))+lane]=1;
                end
                if(locate(write_address)<4) input_bytes+=64;else core_bytes+=64;
                $display("NTRPU_PCIE_FFN_RAM_W addr=%016h id=%0d data=%0128h time_ps=%0d",write_address,write_id,m_axi_wdata,longint'($realtime*1000));
                write_address+=64;write_left--;
                if(write_left==0) begin writing=0;b_pending=1;b_wait=5;end
            end
            if(device.fabric.device.job_valid && device.fabric.device.job_ready)
                $display("NTRPU_PCIE_FFN_JOB sequence=%08h epoch=%08h time_ps=%0d",device.fabric.device.job_sequence,device.fabric.device.job_epoch,longint'($realtime*1000));
            if(device.fabric.device.done_valid && device.fabric.device.done_ready)
                $display("NTRPU_PCIE_FFN_DONE sequence=%08h epoch=%08h status=%0d stages=%0d bytes=%0d stage_bytes=%0d time_ps=%0d",device.fabric.device.done_sequence,device.fabric.device.done_epoch,device.fabric.device.done_status,device.fabric.device.done_stages,device.fabric.device.done_bytes,device.fabric.device.done_stage_bytes,longint'($realtime*1000));
            if(device.fabric.device.with_irq.control.notify_retire_valid && device.fabric.device.with_irq.control.notify_retire_ready)
                $display("NTRPU_PCIE_FFN_RETIRE sequence=%08h epoch=%08h time_ps=%0d",device.fabric.device.with_irq.control.notify_retire_sequence,device.fabric.device.with_irq.control.notify_retire_epoch,longint'($realtime*1000));
            if(device.fabric.device.with_irq.control.notify_service_valid && device.fabric.device.with_irq.control.notify_service_ready)
                $display("NTRPU_PCIE_FFN_SERVICE_ACCEPT sequence=%08h epoch=%08h time_ps=%0d",device.fabric.device.with_irq.control.notify_sequence,device.fabric.device.with_irq.control.notify_epoch,longint'($realtime*1000));
        end
    end
    logic [31:0] held_aw,held_w,held_ar;
    int acknowledgements=0,retires=0;
    bit last_irq=0;
    always @(posedge host_clk) if(host_rst_n) begin
        if(s_axil_awvalid && s_axil_awready) held_aw=s_axil_awaddr;
        if(s_axil_wvalid && s_axil_wready) held_w=s_axil_wdata;
        if(s_axil_arvalid && s_axil_arready) held_ar=s_axil_araddr;
        if(s_axil_bvalid && s_axil_bready)
            $display("NTRPU_PCIE_FFN_MMIO_W addr=%08h data=%08h resp=%0d time_ps=%0d",held_aw,held_w,s_axil_bresp,longint'($realtime*1000));
        if(s_axil_rvalid && s_axil_rready)
            $display("NTRPU_PCIE_FFN_MMIO_R addr=%08h data=%08h resp=%0d time_ps=%0d",held_ar,s_axil_rdata,s_axil_rresp,longint'($realtime*1000));
        if(irq!=last_irq) $display("NTRPU_PCIE_FFN_IRQ level=%0d time_ps=%0d",irq,longint'($realtime*1000));
        if(irq && usr_irq_ack) begin
            acknowledgements++;
            $display("NTRPU_PCIE_FFN_ACK time_ps=%0d",longint'($realtime*1000));
        end
        last_irq=irq;
    end
endmodule
