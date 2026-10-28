`timescale 1ns/1ps
`default_nettype none
// Experimental single-context, single-clock AXI write engine. Not a PCIe ABI.
// The mapping service MUST be the only writer of the bridge translation registers.
// A map response certifies completed programming/readback of exactly the offered image.
// cold_rst_n is configuration reset, never an operational error recovery mechanism.
module ntrpu_axi_bridge_writer #(
    parameter integer AXI_ADDR_BITS=48,
    parameter integer SOURCE_ADDR_BITS=39
) (
    input wire clk,cold_rst_n,link_up,stop,
    input wire cfg_valid,
    output wire cfg_ready,
    input wire [63:0] cfg_session,cfg_generation,cfg_axi_base,cfg_pcie_base,
    output reg cfg_rsp_valid=0,
    input wire cfg_rsp_ready,
    output reg cfg_rsp_error=0,
    output reg map_valid=0,
    input wire map_ready,
    output reg [63:0] map_session=0,map_generation=0,map_axi_base=0,map_pcie_base=0,
    input wire map_rsp_valid,map_rsp_ok,
    output wire map_rsp_ready,
    input wire cmd_valid,
    output wire cmd_ready,
    input wire [127:0] cmd_tag,
    input wire [63:0] cmd_generation,cmd_source,cmd_offset,
    input wire [12:0] cmd_bytes,
    output reg reject_valid=0,
    input wire reject_ready,
    output reg [3:0] reject_code=0,
    output reg [127:0] reject_tag=0,
    output reg src_req_valid=0,
    input wire src_req_ready,
    output reg [127:0] src_req_tag=0,
    output reg [63:0] src_req_address=0,
    output reg [12:0] src_req_bytes=0,
    input wire src_valid,src_last,
    output wire src_ready,
    input wire [511:0] src_data,
    input wire [127:0] src_tag,
    input wire src_cpl_valid,
    output wire src_cpl_ready,
    input wire [127:0] src_cpl_tag,
    input wire [12:0] src_cpl_bytes,
    input wire [1:0] src_cpl_status,
    output wire [AXI_ADDR_BITS-1:0] m_awaddr,
    output wire [3:0] m_awid,
    output wire [7:0] m_awlen,
    output wire [2:0] m_awsize,
    output wire [1:0] m_awburst,
    output reg m_awvalid=0,
    input wire m_awready,
    output wire [511:0] m_wdata,
    output wire [63:0] m_wstrb,
    output wire m_wlast,
    output reg m_wvalid=0,
    input wire m_wready,
    input wire m_bvalid,
    output wire m_bready,
    input wire [1:0] m_bresp,
    input wire [3:0] m_bid,
    input wire ack_valid,
    output wire ack_ready,
    input wire [127:0] ack_tag,
    input wire [63:0] ack_generation,
    input wire [12:0] ack_bytes,
    output reg result_valid=0,
    input wire result_ready,
    output wire [127:0] result_tag,
    output wire [12:0] result_bytes,
    output reg owned=0,
    output reg fault=0,
    output reg [7:0] fault_code=0,
    output wire window_held
);
    localparam [3:0] BAD_ID=1,REPLAY=2,RANGE=3,SPAN=4;
    localparam [7:0] LINK=1,SOURCE=2,RESPONSE=4,ACK=8,MAPPING=16;
    reg configured=0,map_pending=0,map_sent=0,seen_link=0;
    reg [63:0] session=0,generation=0,axi_base=0,watermark=0,held_generation=0;
    reg [63:0] address=0;
    reg [511:0] payload[0:63];
    reg [6:0] received=0,beats=0;
    reg [5:0] write_index=0;
    reg source_issued=0,source_complete=0,launched=0,aw_done=0,w_done=0;
    reg b_seen=0,b_ok=0,ack_seen=0,ack_ok=0;
    wire environment_fault=stop || (seen_link && !link_up);
    wire closed=fault || environment_fault;
    assign window_held=owned || map_pending || fault;
    assign cfg_ready=cold_rst_n && link_up && !closed && !owned && !map_pending &&
                     !cfg_rsp_valid && !result_valid;
    wire cfg_bad=cfg_session==0 || cfg_generation==0 ||
                 (configured && (cfg_session!=session || cfg_generation<=generation)) ||
                 |cfg_axi_base[11:0] || |cfg_pcie_base[11:0] ||
                 (cfg_axi_base >> AXI_ADDR_BITS)!=0;
    assign map_rsp_ready=map_pending && map_sent;
    assign cmd_ready=cold_rst_n && link_up && configured && !closed && !owned &&
                     !cfg_valid && !map_pending && !cfg_rsp_valid && !reject_valid && !result_valid;
    wire [64:0] source_end={1'b0,cmd_source}+{52'b0,cmd_bytes};
    reg [3:0] admission_error;
    always @* begin
        admission_error=0;
        if(cmd_tag[127:64]!=session || cmd_generation!=generation)admission_error=BAD_ID;
        else if(cmd_tag[63:0]<=watermark)admission_error=REPLAY;
        else if(cmd_bytes==0 || cmd_bytes>4096 || cmd_offset>4096 ||
                {52'b0,cmd_bytes}>65'd4096-{1'b0,cmd_offset} ||
                source_end>(65'd1 << SOURCE_ADDR_BITS))admission_error=RANGE;
        else if(|cmd_source[5:0] || |cmd_offset[5:0] || |cmd_bytes[5:0] ||
                ({1'b0,cmd_source[11:0]}+cmd_bytes)>4096)admission_error=SPAN;
    end
    assign src_ready=cold_rst_n && owned && source_issued && !closed && received<beats;
    assign src_cpl_ready=cold_rst_n && owned && source_issued && !source_complete;
    wire take_data=src_valid && src_ready;
    wire data_bad=take_data && (src_tag!=src_req_tag || src_last!=(received+1==beats));
    wire cpl_bad=src_cpl_valid && src_cpl_ready &&
        (src_cpl_status!=0 || src_cpl_tag!=src_req_tag || src_cpl_bytes!=src_req_bytes ||
         received+{6'b0,take_data}!=beats || data_bad);
    assign m_awaddr=address[AXI_ADDR_BITS-1:0];
    assign m_awid=0;
    assign m_awlen={1'b0,beats}-8'd1;
    assign m_awsize=3'd6;
    assign m_awburst=2'b01;
    assign m_wdata=payload[write_index];
    assign m_wstrb=64'hffffffffffffffff;
    assign m_wlast={1'b0,write_index}+1==beats;
    wire take_aw=m_awvalid && m_awready;
    wire take_w=m_wvalid && m_wready;
    assign m_bready=cold_rst_n && owned && launched && !b_seen;
    wire transfer_sent=(aw_done || take_aw) && (w_done || (take_w && m_wlast));
    wire response_bad=m_bvalid && m_bready && (!transfer_sent || m_bid!=0 || m_bresp!=0);
    assign ack_ready=cold_rst_n && owned && !ack_seen;
    wire ack_bad=ack_valid && ack_ready && (!transfer_sent || ack_tag!=src_req_tag ||
        ack_generation!=held_generation || ack_bytes!=src_req_bytes);
    wire mapping_bad=map_rsp_valid && map_rsp_ready && !map_rsp_ok;
    wire failure_now=environment_fault || data_bad || cpl_bad || response_bad || ack_bad || mapping_bad;
    assign result_tag=src_req_tag;
    assign result_bytes=src_req_bytes;

    always @(posedge clk)begin
        if(!cold_rst_n)begin
            cfg_rsp_valid<=0;cfg_rsp_error<=0;map_valid<=0;map_pending<=0;map_sent<=0;
            map_session<=0;map_generation<=0;map_axi_base<=0;map_pcie_base<=0;
            configured<=0;seen_link<=0;session<=0;generation<=0;axi_base<=0;watermark<=0;
            reject_valid<=0;reject_code<=0;reject_tag<=0;
            src_req_valid<=0;src_req_tag<=0;src_req_address<=0;src_req_bytes<=0;
            owned<=0;fault<=0;fault_code<=0;result_valid<=0;held_generation<=0;address<=0;
            received<=0;beats<=0;write_index<=0;source_issued<=0;source_complete<=0;
            launched<=0;m_awvalid<=0;m_wvalid<=0;aw_done<=0;w_done<=0;
            b_seen<=0;b_ok<=0;ack_seen<=0;ack_ok<=0;
        end else begin
            if(link_up)seen_link<=1;
            if(failure_now)begin
                fault<=1;
                fault_code<=fault_code | (environment_fault?LINK:0) |
                    ((data_bad || cpl_bad)?SOURCE:0) | (response_bad?RESPONSE:0) |
                    (ack_bad?ACK:0) | (mapping_bad?MAPPING:0);
            end
            if(cfg_rsp_valid && cfg_rsp_ready)cfg_rsp_valid<=0;
            if(reject_valid && reject_ready)reject_valid<=0;
            if(cfg_valid && cfg_ready)begin
                if(cfg_bad)begin cfg_rsp_valid<=1;cfg_rsp_error<=1;end
                else begin
                    map_pending<=1;map_sent<=0;map_valid<=1;
                    map_session<=cfg_session;map_generation<=cfg_generation;
                    map_axi_base<=cfg_axi_base;map_pcie_base<=cfg_pcie_base;
                end
            end
            // VALID offers persist through faults; only real handshakes remove them.
            if(map_valid && map_ready)begin map_valid<=0;map_sent<=1;end
            if(map_rsp_valid && map_rsp_ready)begin
                map_pending<=0;map_sent<=0;cfg_rsp_valid<=1;
                cfg_rsp_error<=!map_rsp_ok || closed;
                if(map_rsp_ok && !closed)begin
                    session<=map_session;generation<=map_generation;axi_base<=map_axi_base;configured<=1;
                end
            end
            if(cmd_valid && cmd_ready)begin
                if(admission_error!=0)begin
                    reject_valid<=1;reject_code<=admission_error;reject_tag<=cmd_tag;
                end else begin
                    owned<=1;src_req_valid<=1;src_req_tag<=cmd_tag;
                    src_req_address<=cmd_source;src_req_bytes<=cmd_bytes;held_generation<=generation;
                    watermark<=cmd_tag[63:0];address<=axi_base+cmd_offset;
                    received<=0;beats<=cmd_bytes[12:6];write_index<=0;
                    source_issued<=0;source_complete<=0;launched<=0;aw_done<=0;w_done<=0;
                    b_seen<=0;b_ok<=0;ack_seen<=0;ack_ok<=0;
                end
            end
            if(src_req_valid && src_req_ready)begin src_req_valid<=0;source_issued<=1;end
            if(take_data)begin
                if(!data_bad)begin payload[received[5:0]]<=src_data;received<=received+1'b1;end
            end
            if(src_cpl_valid && src_cpl_ready && !cpl_bad)source_complete<=1;
            if(owned && source_complete && received==beats && !launched && !closed && !failure_now)begin
                launched<=1;m_awvalid<=1;m_wvalid<=1;
            end
            if(take_aw)begin m_awvalid<=0;aw_done<=1;end
            if(take_w)begin
                if(m_wlast)begin m_wvalid<=0;w_done<=1;end
                else write_index<=write_index+1'b1;
            end
            if(m_bvalid && m_bready)begin b_seen<=1;b_ok<=!response_bad;end
            if(ack_valid && ack_ready)begin ack_seen<=1;ack_ok<=!ack_bad;end
            if(owned && b_ok && ack_ok && !closed && !failure_now && !result_valid)result_valid<=1;
            // A success already offered stays stable; a later fault still prevents reuse.
            if(result_valid && result_ready)begin
                result_valid<=0;
                if(!closed && !failure_now)owned<=0;
            end
        end
    end
endmodule
`default_nettype wire
