// Two initiators, one memory target, one outstanding burst per direction.
// 512-bit data / 64-bit byte addresses / 4-bit IDs, common clock and reset.
// Round-robin grants stay owned until the initiator consumes RLAST or B.
// All five channels are registered; AW and W progress independently downstream.
// Two-entry R/W queues support simultaneous enqueue/dequeue without a
// combinational READY path. A full queue inserts a recovery bubble after stall.
// No address translation, timeout, CDC, exclusives, QoS or USER sidebands.
// Initiators must be AXI4-compliant (including AxLOCK=0). Reset must quiesce
// both initiators and the memory peer; reset alone cannot cancel external AXI.
module ntrpu_axi_arbiter (
    input logic clk, rst_n,
    input logic [1:0][63:0] s_axi_araddr, s_axi_awaddr,
    input logic [1:0][7:0] s_axi_arlen, s_axi_awlen,
    input logic [1:0][2:0] s_axi_arsize, s_axi_arprot, s_axi_awsize, s_axi_awprot,
    input logic [1:0][1:0] s_axi_arburst, s_axi_awburst,
    input logic [1:0][3:0] s_axi_arid, s_axi_arcache, s_axi_awid, s_axi_awcache,
    input logic [1:0] s_axi_arlock, s_axi_arvalid, s_axi_awlock, s_axi_awvalid,
    output logic [1:0] s_axi_arready, s_axi_awready,
    output logic [1:0][511:0] s_axi_rdata,
    output logic [1:0][1:0] s_axi_rresp,
    output logic [1:0][3:0] s_axi_rid,
    output logic [1:0] s_axi_rlast, s_axi_rvalid,
    input logic [1:0] s_axi_rready,
    input logic [1:0][511:0] s_axi_wdata,
    input logic [1:0][63:0] s_axi_wstrb,
    input logic [1:0] s_axi_wlast, s_axi_wvalid,
    output logic [1:0] s_axi_wready,
    output logic [1:0][3:0] s_axi_bid,
    output logic [1:0][1:0] s_axi_bresp,
    output logic [1:0] s_axi_bvalid,
    input logic [1:0] s_axi_bready,
    output logic [63:0] m_axi_araddr, m_axi_awaddr,
    output logic [7:0] m_axi_arlen, m_axi_awlen,
    output logic [2:0] m_axi_arsize, m_axi_arprot, m_axi_awsize, m_axi_awprot,
    output logic [1:0] m_axi_arburst, m_axi_awburst,
    output logic [3:0] m_axi_arid, m_axi_arcache, m_axi_awid, m_axi_awcache,
    output logic m_axi_arlock, m_axi_arvalid, m_axi_awlock, m_axi_awvalid,
    input logic m_axi_arready, m_axi_awready,
    input logic [511:0] m_axi_rdata,
    input logic [1:0] m_axi_rresp,
    input logic [3:0] m_axi_rid,
    input logic m_axi_rlast, m_axi_rvalid,
    output logic m_axi_rready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast, m_axi_wvalid,
    input logic m_axi_wready,
    input logic [3:0] m_axi_bid,
    input logic [1:0] m_axi_bresp,
    input logic m_axi_bvalid,
    output logic m_axi_bready
);
    typedef enum logic [1:0] {IDLE, CAPTURE, ACTIVE} state_t;
    state_t read_state, write_state;
    logic read_owner, write_owner, read_next, write_next;
    logic ar_pending, ar_issued, read_end;
    logic aw_pending, aw_issued, write_end, write_last_accepted, b_full;
    logic [1:0] r_count, w_count;
    logic r_head, r_tail, w_head, w_tail, r_push, r_pop, w_push, w_pop;
    logic [511:0] r_data[0:1], w_data[0:1];
    logic [63:0] w_strb[0:1];
    logic [1:0] r_resp[0:1], b_resp;
    logic [3:0] r_id[0:1], b_id;
    logic r_last[0:1], w_last[0:1];

    always_comb begin
        s_axi_arready='0; s_axi_awready='0; s_axi_wready='0;
        s_axi_rvalid='0; s_axi_bvalid='0;
        s_axi_rdata={2{r_data[r_head]}}; s_axi_rresp={2{r_resp[r_head]}};
        s_axi_rid={2{r_id[r_head]}}; s_axi_rlast={2{r_last[r_head]}};
        s_axi_bid={2{b_id}}; s_axi_bresp={2{b_resp}};
        m_axi_arvalid=rst_n && ar_pending;
        m_axi_awvalid=rst_n && aw_pending;
        m_axi_wdata=w_data[w_head]; m_axi_wstrb=w_strb[w_head]; m_axi_wlast=w_last[w_head];
        m_axi_wvalid=rst_n && (w_count!=0);
        m_axi_rready=rst_n && ar_issued && !read_end && (r_count<2);
        m_axi_bready=rst_n && aw_issued && write_end && !b_full;
        if(rst_n) begin
            s_axi_arready[read_owner]=(read_state==CAPTURE);
            s_axi_awready[write_owner]=(write_state==CAPTURE);
            s_axi_wready[write_owner]=(write_state==ACTIVE && !write_last_accepted && w_count<2);
            s_axi_rvalid[read_owner]=(r_count!=0);
            s_axi_bvalid[write_owner]=b_full;
        end
        r_push=m_axi_rvalid && m_axi_rready;
        r_pop=s_axi_rvalid[read_owner] && s_axi_rready[read_owner];
        w_push=s_axi_wvalid[write_owner] && s_axi_wready[write_owner];
        w_pop=m_axi_wvalid && m_axi_wready;
    end

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            read_state<=IDLE; write_state<=IDLE;
            read_owner<=0; write_owner<=0; read_next<=0; write_next<=0;
            ar_pending<=0; ar_issued<=0; read_end<=0;
            aw_pending<=0; aw_issued<=0; write_end<=0; write_last_accepted<=0; b_full<=0;
            r_count<=0; w_count<=0; r_head<=0; r_tail<=0; w_head<=0; w_tail<=0;
            m_axi_araddr<=0; m_axi_arlen<=0; m_axi_arsize<=0;
            m_axi_arburst<=0; m_axi_arid<=0; m_axi_arcache<=0;
            m_axi_arprot<=0; m_axi_arlock<=0;
            m_axi_awaddr<=0; m_axi_awlen<=0; m_axi_awsize<=0;
            m_axi_awburst<=0; m_axi_awid<=0; m_axi_awcache<=0;
            m_axi_awprot<=0; m_axi_awlock<=0;
            for(int i=0;i<2;i++) begin
                w_data[i]<=0; w_strb[i]<=0; w_last[i]<=0;
                r_data[i]<=0; r_resp[i]<=0; r_id[i]<=0; r_last[i]<=0;
            end
            b_resp<=0; b_id<=0;
        end else begin
            case(read_state)
                IDLE: if(|s_axi_arvalid) begin
                    read_owner<=s_axi_arvalid[read_next] ? read_next:!read_next;
                    read_state<=CAPTURE;
                end
                CAPTURE: if(s_axi_arvalid[read_owner]) begin
                    m_axi_araddr<=s_axi_araddr[read_owner];
                    m_axi_arlen<=s_axi_arlen[read_owner];
                    m_axi_arsize<=s_axi_arsize[read_owner];
                    m_axi_arburst<=s_axi_arburst[read_owner];
                    m_axi_arid<=s_axi_arid[read_owner];
                    m_axi_arcache<=s_axi_arcache[read_owner];
                    m_axi_arprot<=s_axi_arprot[read_owner];
                    m_axi_arlock<=s_axi_arlock[read_owner];
                    ar_pending<=1; read_state<=ACTIVE;
                end
                ACTIVE: if(r_pop && r_last[r_head]) begin
                    read_state<=IDLE; ar_issued<=0; read_end<=0;
                    read_next<=!read_owner;
                end
                default: read_state<=IDLE;
            endcase
            if(m_axi_arvalid && m_axi_arready) begin ar_pending<=0; ar_issued<=1; end
            if(r_push) begin
                r_data[r_tail]<=m_axi_rdata; r_resp[r_tail]<=m_axi_rresp;
                r_id[r_tail]<=m_axi_rid; r_last[r_tail]<=m_axi_rlast; r_tail<=!r_tail;
                if(m_axi_rlast) read_end<=1;
            end
            if(r_pop) r_head<=!r_head;
            case({r_push,r_pop})
                2'b10: r_count<=r_count+1'b1;
                2'b01: r_count<=r_count-1'b1;
                default: ;
            endcase

            case(write_state)
                IDLE: if(|s_axi_awvalid) begin
                    write_owner<=s_axi_awvalid[write_next] ? write_next:!write_next;
                    write_state<=CAPTURE;
                end
                CAPTURE: if(s_axi_awvalid[write_owner]) begin
                    m_axi_awaddr<=s_axi_awaddr[write_owner];
                    m_axi_awlen<=s_axi_awlen[write_owner];
                    m_axi_awsize<=s_axi_awsize[write_owner];
                    m_axi_awburst<=s_axi_awburst[write_owner];
                    m_axi_awid<=s_axi_awid[write_owner];
                    m_axi_awcache<=s_axi_awcache[write_owner];
                    m_axi_awprot<=s_axi_awprot[write_owner];
                    m_axi_awlock<=s_axi_awlock[write_owner];
                    aw_pending<=1; write_state<=ACTIVE;
                end
                ACTIVE: if(b_full && s_axi_bready[write_owner]) begin
                    write_state<=IDLE; aw_issued<=0; write_end<=0; write_last_accepted<=0;
                    write_next<=!write_owner;
                end
                default: write_state<=IDLE;
            endcase
            if(m_axi_awvalid && m_axi_awready) begin aw_pending<=0; aw_issued<=1; end
            if(w_push) begin
                w_data[w_tail]<=s_axi_wdata[write_owner]; w_strb[w_tail]<=s_axi_wstrb[write_owner];
                w_last[w_tail]<=s_axi_wlast[write_owner]; w_tail<=!w_tail;
                if(s_axi_wlast[write_owner]) write_last_accepted<=1;
            end
            if(w_pop) begin
                w_head<=!w_head;
                if(m_axi_wlast) write_end<=1;
            end
            case({w_push,w_pop})
                2'b10: w_count<=w_count+1'b1;
                2'b01: w_count<=w_count-1'b1;
                default: ;
            endcase
            if(m_axi_bvalid && m_axi_bready) begin b_full<=1; b_id<=m_axi_bid; b_resp<=m_axi_bresp; end
            if(b_full && s_axi_bready[write_owner]) b_full<=0;
        end
    end
endmodule
