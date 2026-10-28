// 64-bit AXI address firewall for one power-of-two memory channel.
// 512-bit data, 4-bit IDs, aligned full-width INCR bursts, no exclusives.
// One outstanding burst per direction; two-entry registered W/R queues.
// admit only gates new AR/AW captures. Accepted work must drain normally.
// Both peers must be quiesced for cold/coordinated reset: reset is not abort.
// AXI-compliant beat counts/LAST and compliant downstream responses required.
module ntrpu_axi_capacity_guard #(parameter integer MEMORY_ADDR_WIDTH=37) (
    input logic clk, rst_n, admit,
    output logic idle,
    input logic [63:0] s_axi_araddr, s_axi_awaddr,
    input logic [7:0] s_axi_arlen, s_axi_awlen,
    input logic [2:0] s_axi_arsize, s_axi_arprot, s_axi_awsize, s_axi_awprot,
    input logic [1:0] s_axi_arburst, s_axi_awburst,
    input logic [3:0] s_axi_arid, s_axi_arcache, s_axi_awid, s_axi_awcache,
    input logic s_axi_arlock, s_axi_arvalid, s_axi_awlock, s_axi_awvalid,
    output logic s_axi_arready, s_axi_awready,
    output logic [511:0] s_axi_rdata,
    output logic [1:0] s_axi_rresp,
    output logic [3:0] s_axi_rid,
    output logic s_axi_rlast, s_axi_rvalid,
    input logic s_axi_rready,
    input logic [511:0] s_axi_wdata,
    input logic [63:0] s_axi_wstrb,
    input logic s_axi_wlast, s_axi_wvalid,
    output logic s_axi_wready,
    output logic [3:0] s_axi_bid,
    output logic [1:0] s_axi_bresp,
    output logic s_axi_bvalid,
    input logic s_axi_bready,
    output logic [MEMORY_ADDR_WIDTH-1:0] m_axi_araddr, m_axi_awaddr,
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
    localparam logic [64:0] CAPACITY=65'd1 << MEMORY_ADDR_WIDTH;
    typedef enum logic [1:0] {R_IDLE, R_ADDRESS, R_ACTIVE, R_ERROR} read_state_t;
    typedef enum logic [1:0] {W_IDLE, W_ACTIVE, W_ERROR, W_REPLY} write_state_t;
    read_state_t read_state;
    write_state_t write_state;
    logic [8:0] read_left, write_left;
    logic [3:0] read_id, write_id;
    logic aw_pending, aw_issued, read_end, write_end;
    logic [1:0] r_count, w_count;
    logic r_head, r_tail, w_head, w_tail, r_push, r_pop, w_push, w_pop;
    logic [511:0] r_data[0:1], w_data[0:1];
    logic [63:0] w_strb[0:1];
    logic [1:0] r_resp[0:1], b_resp;
    logic [3:0] r_id[0:1], b_id;
    logic r_last[0:1], w_last[0:1];

    function automatic logic permitted(input logic [63:0] address,
        input logic [7:0] length, input logic [2:0] size,
        input logic [1:0] burst, input logic locked);
        logic [8:0] beats;
        logic [64:0] end_address;
        logic [14:0] page_end;
        beats={1'b0,length}+9'd1;
        end_address={1'b0,address}+{50'b0,beats,6'b0};
        page_end={3'b0,address[11:0]}+{beats,6'b0};
        return size==3'd6 && burst==2'b01 && !locked && address[5:0]==0 &&
               page_end<=15'd4096 && {1'b0,address}<CAPACITY && end_address<=CAPACITY;
    endfunction

    always_comb begin
        s_axi_arready=rst_n && admit && read_state==R_IDLE;
        s_axi_awready=rst_n && admit && write_state==W_IDLE;
        s_axi_wready=rst_n && write_left!=0 &&
            ((write_state==W_ACTIVE && w_count<2) || write_state==W_ERROR);
        m_axi_arvalid=rst_n && read_state==R_ADDRESS;
        m_axi_awvalid=rst_n && aw_pending;
        m_axi_wvalid=rst_n && write_state==W_ACTIVE && w_count!=0;
        m_axi_wdata=w_data[w_head];
        m_axi_wstrb=w_strb[w_head];
        m_axi_wlast=w_last[w_head];
        m_axi_rready=rst_n && read_state==R_ACTIVE && !read_end && r_count<2;
        m_axi_bready=rst_n && write_state==W_ACTIVE && aw_issued && write_end;
        s_axi_rvalid=rst_n && (read_state==R_ERROR || r_count!=0);
        s_axi_rdata=read_state==R_ERROR ? 512'b0:r_data[r_head];
        s_axi_rresp=read_state==R_ERROR ? 2'b11:r_resp[r_head];
        s_axi_rid=read_state==R_ERROR ? read_id:r_id[r_head];
        s_axi_rlast=read_state==R_ERROR ? read_left==1:r_last[r_head];
        s_axi_bvalid=rst_n && write_state==W_REPLY;
        s_axi_bid=b_id;
        s_axi_bresp=b_resp;
        idle=rst_n && read_state==R_IDLE && write_state==W_IDLE;
        r_push=m_axi_rvalid && m_axi_rready;
        r_pop=s_axi_rvalid && s_axi_rready && read_state!=R_ERROR;
        w_push=s_axi_wvalid && s_axi_wready && write_state==W_ACTIVE;
        w_pop=m_axi_wvalid && m_axi_wready;
    end

    always_ff @(posedge clk) begin
        if(!rst_n) begin
            read_state<=R_IDLE; write_state<=W_IDLE;
            read_left<=0; write_left<=0; read_id<=0; write_id<=0;
            aw_pending<=0; aw_issued<=0; read_end<=0; write_end<=0;
            r_count<=0; w_count<=0; r_head<=0; r_tail<=0; w_head<=0; w_tail<=0;
            b_id<=0; b_resp<=0;
            m_axi_araddr<=0; m_axi_arlen<=0; m_axi_arsize<=0; m_axi_arburst<=0;
            m_axi_arid<=0; m_axi_arcache<=0; m_axi_arprot<=0; m_axi_arlock<=0;
            m_axi_awaddr<=0; m_axi_awlen<=0; m_axi_awsize<=0; m_axi_awburst<=0;
            m_axi_awid<=0; m_axi_awcache<=0; m_axi_awprot<=0; m_axi_awlock<=0;
            // Payload storage is unobservable until its valid count is set.
        end else begin
            if(s_axi_arvalid && s_axi_arready) begin
                read_left<={1'b0,s_axi_arlen}+9'd1; read_id<=s_axi_arid; read_end<=0;
                m_axi_araddr<=s_axi_araddr[MEMORY_ADDR_WIDTH-1:0];
                m_axi_arlen<=s_axi_arlen; m_axi_arsize<=s_axi_arsize;
                m_axi_arburst<=s_axi_arburst; m_axi_arid<=s_axi_arid;
                m_axi_arcache<=s_axi_arcache; m_axi_arprot<=s_axi_arprot; m_axi_arlock<=s_axi_arlock;
                read_state<=permitted(s_axi_araddr,s_axi_arlen,s_axi_arsize,s_axi_arburst,s_axi_arlock) ? R_ADDRESS:R_ERROR;
            end
            if(m_axi_arvalid && m_axi_arready) read_state<=R_ACTIVE;
            if(read_state==R_ERROR && s_axi_rvalid && s_axi_rready) begin
                read_left<=read_left-1'b1;
                if(read_left==1) read_state<=R_IDLE;
            end
            if(r_push) begin
                r_data[r_tail]<=m_axi_rdata; r_resp[r_tail]<=m_axi_rresp;
                r_id[r_tail]<=m_axi_rid; r_last[r_tail]<=m_axi_rlast; r_tail<=!r_tail;
                if(m_axi_rlast) read_end<=1;
            end
            if(r_pop) begin
                r_head<=!r_head;
                if(r_last[r_head]) read_state<=R_IDLE;
            end
            case({r_push,r_pop})
                2'b10: r_count<=r_count+1'b1;
                2'b01: r_count<=r_count-1'b1;
                default: ;
            endcase

            if(s_axi_awvalid && s_axi_awready) begin
                write_left<={1'b0,s_axi_awlen}+9'd1; write_id<=s_axi_awid;
                aw_issued<=0; write_end<=0;
                m_axi_awaddr<=s_axi_awaddr[MEMORY_ADDR_WIDTH-1:0];
                m_axi_awlen<=s_axi_awlen; m_axi_awsize<=s_axi_awsize;
                m_axi_awburst<=s_axi_awburst; m_axi_awid<=s_axi_awid;
                m_axi_awcache<=s_axi_awcache; m_axi_awprot<=s_axi_awprot; m_axi_awlock<=s_axi_awlock;
                if(permitted(s_axi_awaddr,s_axi_awlen,s_axi_awsize,s_axi_awburst,s_axi_awlock)) begin
                    write_state<=W_ACTIVE; aw_pending<=1;
                end else begin write_state<=W_ERROR; aw_pending<=0; end
            end
            if(m_axi_awvalid && m_axi_awready) begin aw_pending<=0; aw_issued<=1; end
            if(s_axi_wvalid && s_axi_wready) begin
                write_left<=write_left-1'b1;
                if(write_state==W_ERROR && write_left==1) begin
                    b_id<=write_id; b_resp<=2'b11; write_state<=W_REPLY;
                end
            end
            if(w_push) begin
                w_data[w_tail]<=s_axi_wdata; w_strb[w_tail]<=s_axi_wstrb;
                w_last[w_tail]<=s_axi_wlast; w_tail<=!w_tail;
            end
            if(w_pop) begin
                w_head<=!w_head;
                if(w_last[w_head]) write_end<=1;
            end
            case({w_push,w_pop})
                2'b10: w_count<=w_count+1'b1;
                2'b01: w_count<=w_count-1'b1;
                default: ;
            endcase
            if(m_axi_bvalid && m_axi_bready) begin
                b_id<=m_axi_bid; b_resp<=m_axi_bresp; write_state<=W_REPLY;
            end
            if(s_axi_bvalid && s_axi_bready) write_state<=W_IDLE;
        end
    end
endmodule
