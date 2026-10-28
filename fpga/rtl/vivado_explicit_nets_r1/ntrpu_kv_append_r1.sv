// One active FP32 KV allocation: [layer][K/V][head][position][64].
// Stage all 64 beats of a layer/token before writing. Advance the owned
// per-layer prefix only after all sixteen heads/planes receive successful B.
// Rebinding requires the scheduler's maintenance_allowed certificate: no
// readers or users of the old lease may remain. Reset must quiesce AXI and
// producers/readers; it invalidates the lease and every committed prefix.
`default_nettype none
module ntrpu_kv_append_r1 #(
    parameter integer WATCHDOG_CYCLES=65536
) (
    input wire logic clk,rst_n,
    input wire logic maintenance_allowed,
    input wire logic bind_valid,
    output logic bind_ready,
    input wire logic [63:0] bind_cache_base,bind_region_base,bind_region_bytes,
    input wire logic [31:0] bind_capacity,
    input wire logic [63:0] bind_sequence,bind_epoch,expected_epoch,
    output logic cache_valid,
    output logic [63:0] cache_base,cache_region_base,cache_region_bytes,cache_sequence,cache_epoch,
    output logic [31:0] cache_capacity,
    input wire logic [3:0] inspect_layer,
    output logic [31:0] committed_tokens,
    input wire logic cmd_valid,
    output logic cmd_ready,prepared,
    input wire logic [31:0] cmd_layer,cmd_position,
    input wire logic [63:0] cmd_sequence,cmd_epoch,
    input wire logic data_valid,
    output logic data_ready,
    input wire logic [511:0] data_bits,
    input wire logic [5:0] data_index,
    input wire logic data_last,
    input wire logic [63:0] data_sequence,data_epoch,
    input wire logic commit_valid,
    output logic commit_ready,
    input wire logic [3:0] commit_status,
    input wire logic [63:0] commit_sequence,commit_epoch,
    input wire logic cancel_valid,
    output logic cancel_ready,
    output logic done_valid,
    input wire logic done_ready,
    output logic done_bind,
    output logic [3:0] done_status,
    output logic [12:0] done_bytes,
    output logic [31:0] done_committed_tokens,
    output logic [63:0] done_sequence,done_epoch,
    output logic [63:0] m_axi_awaddr,
    output logic [7:0] m_axi_awlen,
    output logic [2:0] m_axi_awsize,m_axi_awprot,
    output logic [1:0] m_axi_awburst,
    output logic [3:0] m_axi_awid,m_axi_awcache,
    output logic m_axi_awlock,m_axi_awvalid,
    input wire logic m_axi_awready,
    output logic [511:0] m_axi_wdata,
    output logic [63:0] m_axi_wstrb,
    output logic m_axi_wlast,m_axi_wvalid,
    input wire logic m_axi_wready,
    input wire logic [3:0] m_axi_bid,
    input wire logic [1:0] m_axi_bresp,
    input wire logic m_axi_bvalid,
    output logic m_axi_bready
);
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,WRITE_ERROR=3,
        BAD_FORMAT=4,CANCELLED=7,TIMEOUT=8;
    typedef enum logic [3:0] {IDLE,BIND_BOUNDS,BIND_CHECK,COLLECT,SLOT,
        ADDRESS,PLAN,WRITE,RESPONSE} state_t;
    state_t state;
    logic [17:0] prefixes[0:15];
    (* ram_style="block" *) logic [511:0] staging[0:63];
    logic [64:0] cache_end,region_end;
    logic [63:0] last_epoch,address;
    logic [3:0] layer,head;
    logic [17:0] position;
    logic [25:0] slot;
    logic [7:0] received;
    logic [8:0] received_after;
    logic [6:0] beat_index;
    logic [2:0] remaining,chunk,sent;
    logic aw_seen,aw_fire,w_fire,b_fire,data_fire,progress_event,watchdog_expired,bad_data;
    logic [31:0] idle_cycles;
    logic [3:0] error_now;

    function automatic logic [2:0] burst_size(input logic [63:0] addr,input logic [2:0] count);
        logic [6:0] page;
        page=7'd64-{1'b0,addr[11:6]};
        return {4'b0,count}<page ? count:page[2:0];
    endfunction
    assign bind_ready=state==IDLE && maintenance_allowed;
    assign cmd_ready=state==IDLE && !(bind_valid && maintenance_allowed);
    assign prepared=state==COLLECT;
    assign data_ready=state==COLLECT;
    assign commit_ready=state==COLLECT;
    assign cancel_ready=!done_bind && state!=IDLE && state!=RESPONSE && done_status==OK;
    assign done_valid=state==RESPONSE;
    assign committed_tokens=cache_valid ? {14'b0,prefixes[inspect_layer]}:32'b0;
    assign data_fire=data_valid && data_ready;
    assign received_after={1'b0,received}+9'(data_fire);
    assign m_axi_awaddr=address;
    assign m_axi_awlen={5'b0,chunk}-8'd1;
    assign m_axi_awsize=3'd6;
    assign m_axi_awburst=2'b01;
    assign m_axi_awid=0;
    assign m_axi_awcache=4'b0010;
    assign m_axi_awprot=0;
    assign m_axi_awlock=0;
    assign m_axi_awvalid=state==WRITE && !aw_seen;
    assign m_axi_wstrb=64'hffffffffffffffff;
    assign m_axi_wlast=sent+3'd1==chunk;
    assign m_axi_bready=state==WRITE && aw_seen && sent==chunk;
    assign aw_fire=m_axi_awvalid && m_axi_awready;
    assign w_fire=m_axi_wvalid && m_axi_wready;
    assign b_fire=m_axi_bvalid && m_axi_bready;
    assign progress_event=aw_fire || w_fire || b_fire || data_fire || (commit_valid && commit_ready);
    assign watchdog_expired=(state==COLLECT || state==WRITE) &&
        idle_cycles>=32'(WATCHDOG_CYCLES-1) && !progress_event;
    always_comb begin
        bad_data=received>=64 || data_index!=received[5:0] || data_last!=(received_after==64) ||
            data_sequence!=done_sequence || data_epoch!=done_epoch;
        for(int lane=0;lane<16;lane++) bad_data |= data_bits[lane*32+23+:8]==8'hff;
        error_now=done_status;
        if(done_status==OK) begin
            if(cancel_valid && cancel_ready) error_now=CANCELLED;
            else if(watchdog_expired) error_now=TIMEOUT;
            else if(b_fire && (m_axi_bresp!=0 || m_axi_bid!=0)) error_now=WRITE_ERROR;
            else if(data_fire && bad_data) error_now=BAD_FORMAT;
        end
    end
    // One synchronous read and one write port; no reset of the private RAM.
    always_ff @(posedge clk) begin
        if(rst_n && data_fire && error_now==OK) staging[received[5:0]]<=data_bits;
        if(rst_n && state==WRITE && !m_axi_wvalid && sent<chunk)
            m_axi_wdata<=staging[beat_index[5:0]];
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            state<=IDLE;cache_valid<=0;cache_base<=0;cache_region_base<=0;cache_region_bytes<=0;
            cache_sequence<=0;cache_epoch<=0;cache_capacity<=0;last_epoch<=0;cache_end<=0;region_end<=0;
            layer<=0;head<=0;position<=0;slot<=0;address<=0;received<=0;beat_index<=0;
            remaining<=0;chunk<=0;sent<=0;aw_seen<=0;m_axi_wvalid<=0;idle_cycles<=0;
            done_bind<=0;done_status<=OK;done_bytes<=0;done_committed_tokens<=0;done_sequence<=0;done_epoch<=0;
            for(int i=0;i<16;i++) prefixes[i]<=0;
        end else begin
            if(state!=IDLE && state!=RESPONSE) done_status<=error_now;
            if(state==COLLECT || state==WRITE) begin
                if(progress_event) idle_cycles<=0;
                else if(idle_cycles<32'(WATCHDOG_CYCLES)) idle_cycles<=idle_cycles+32'd1;
            end else idle_cycles<=0;
            if(data_fire && received<255) received<=received+8'd1;
            if(aw_fire) aw_seen<=1;
            if(state==WRITE && !m_axi_wvalid && sent<chunk) m_axi_wvalid<=1;
            if(w_fire) begin
                m_axi_wvalid<=0;sent<=sent+3'd1;beat_index<=beat_index+7'd1;done_bytes<=done_bytes+13'd64;
            end
            case(state)
                IDLE: begin
                    if(bind_valid && bind_ready) begin
                        cache_valid<=0;
                        for(int i=0;i<16;i++) prefixes[i]<=0;
                        cache_base<=bind_cache_base;cache_region_base<=bind_region_base;cache_region_bytes<=bind_region_bytes;
                        cache_sequence<=bind_sequence;cache_epoch<=bind_epoch;cache_capacity<=bind_capacity;
                        done_bind<=1;done_status<=OK;done_bytes<=0;done_committed_tokens<=0;
                        done_sequence<=bind_sequence;done_epoch<=bind_epoch;
                        if(bind_cache_base[5:0]!=0 || bind_region_bytes==0 || bind_capacity==0 || bind_capacity>131072) begin
                            done_status<=BAD_COMMAND;state<=RESPONSE;
                        end else if(bind_sequence==0 || bind_epoch==0 || bind_epoch<=last_epoch || bind_epoch!=expected_epoch) begin
                            done_status<=BAD_IDENTITY;state<=RESPONSE;
                        end else state<=BIND_BOUNDS;
                    end else if(cmd_valid && cmd_ready) begin
                        done_bind<=0;done_status<=OK;done_bytes<=0;
                        done_sequence<=cmd_sequence;done_epoch<=cmd_epoch;
                        done_committed_tokens<=cmd_layer<16 ? {14'b0,prefixes[cmd_layer[3:0]]}:32'b0;
                        layer<=cmd_layer[3:0];position<=cmd_position[17:0];head<=0;received<=0;beat_index<=0;
                        if(!cache_valid || cmd_sequence!=cache_sequence || cmd_epoch!=cache_epoch || cmd_epoch!=expected_epoch) begin
                            done_status<=BAD_IDENTITY;state<=RESPONSE;
                        end else if(cmd_layer>=16 || cmd_position>=cache_capacity ||
                            cmd_position!={14'b0,prefixes[cmd_layer[3:0]]}) begin done_status<=BAD_COMMAND;state<=RESPONSE;end
                        else state<=COLLECT;
                    end
                end
                BIND_BOUNDS: begin
                    cache_end<={1'b0,cache_base}+{17'b0,cache_capacity,16'b0};
                    region_end<={1'b0,cache_region_base}+{1'b0,cache_region_bytes};state<=BIND_CHECK;
                end
                BIND_CHECK: begin
                    if(cache_end[64] || region_end[64] || cache_base<cache_region_base || cache_end>region_end) done_status<=BAD_COMMAND;
                    else begin cache_valid<=1;last_epoch<=cache_epoch;end
                    state<=RESPONSE;
                end
                COLLECT: if(commit_valid) begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(commit_sequence!=done_sequence || commit_epoch!=done_epoch) begin done_status<=BAD_IDENTITY;state<=RESPONSE;end
                    else if(commit_status!=OK) begin done_status<=commit_status;state<=RESPONSE;end
                    else if(received_after!=64) begin done_status<=BAD_FORMAT;state<=RESPONSE;end
                    else state<=SLOT;
                end
                SLOT: begin
                    slot<=26'({layer,head})*26'(cache_capacity)+26'(position);
                    if(error_now!=OK) state<=RESPONSE;else state<=ADDRESS;
                end
                ADDRESS: begin
                    address<=cache_base+{30'b0,slot,8'b0};remaining<=4;
                    if(error_now!=OK) state<=RESPONSE;else state<=PLAN;
                end
                PLAN: begin
                    if(error_now!=OK) state<=RESPONSE;
                    else begin chunk<=burst_size(address,remaining);sent<=0;aw_seen<=0;state<=WRITE;end
                end
                WRITE: if(b_fire) begin
                    if(error_now!=OK) state<=RESPONSE;
                    else if(beat_index==64) begin
                        prefixes[layer]<=position+18'd1;
                        done_committed_tokens<={14'b0,position}+32'd1;
                        state<=RESPONSE;
                    end else if(chunk==remaining) begin head<=head+4'd1;state<=SLOT;end
                    else begin address<=address+{55'b0,chunk,6'b0};remaining<=remaining-chunk;state<=PLAN;end
                end
                RESPONSE: if(done_ready) state<=IDLE;
                default: state<=IDLE;
            endcase
        end
    end
endmodule
`default_nettype wire
