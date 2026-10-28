`timescale 1ns/1ps
`default_nettype none
// One retained source context. All ports are in clk, the bridge AXI domain.
// The HBM reader owns the memory-domain CDC. No operational reset is generated.
module ntrpu_axi_bridge_hbm_source (
    input wire clk,cold_rst_n,enable,stop,reader_started,reader_fault,
    input wire src_req_valid,
    output wire src_req_ready,
    input wire [127:0] src_req_tag,
    input wire [63:0] src_req_address,
    input wire [12:0] src_req_bytes,
    output reg src_valid=0,src_last=0,
    input wire src_ready,
    output reg [511:0] src_data=0,
    output wire [127:0] src_tag,
    output wire src_cpl_valid,
    input wire src_cpl_ready,
    output wire [127:0] src_cpl_tag,
    output wire [12:0] src_cpl_bytes,
    output wire [1:0] src_cpl_status,
    input wire retire_valid,
    input wire [127:0] retire_tag,
    output reg owned=0,
    output wire fault,reader_stop,
    output reg [1:0] r_cmd_valid=0,
    input wire [1:0] r_cmd_ready,
    output wire [127:0] r_cmd_addr,r_cmd_tag,
    output wire [15:0] r_cmd_len,
    input wire [1:0] r_data_valid,r_data_last,
    output wire [1:0] r_data_ready,
    input wire [1023:0] r_data,
    input wire [127:0] r_data_tag,
    input wire [1:0] r_cpl_valid,
    output wire [1:0] r_cpl_ready,
    input wire [127:0] r_cpl_tag,
    input wire [3:0] r_cpl_status,
    input wire [25:0] r_cpl_bytes
);
    reg failed=0,seen_ready=0,lane=0,issued=0,cpl_buffered=0;
    reg last_delivered=0,cpl_delivered=0;
    reg [127:0] tag=0;
    reg [63:0] address=0;
    reg [12:0] bytes=0;
    reg [6:0] received=0,beats=0;
    wire request_fire=src_req_valid&&src_req_ready;
    wire [64:0] end_address={1'b0,src_req_address}+{52'b0,src_req_bytes};
    wire bad_request=src_req_bytes==0||src_req_bytes>4096||
        |src_req_address[5:0]|| |src_req_bytes[5:0]||end_address>65'h400000000||
        ({1'b0,src_req_address[11:0]}+src_req_bytes)>4096;
    wire command_fire=r_cmd_valid[lane]&&r_cmd_ready[lane];
    wire incoming_data=r_data_valid[lane];
    wire incoming_cpl=r_cpl_valid[lane];
    wire wrong_lane=owned&&(r_data_valid[!lane]||r_cpl_valid[!lane]);
    wire bad_data=incoming_data&&(!owned||!issued||received>=beats||
        r_data_tag[lane*64+:64]!=tag[63:0]||r_data_last[lane]!=(received+1==beats));
    wire data_space=!src_valid||src_ready;
    // The selected frame reader publishes CPL after the final data handshake.
    // Validate only registered receipt counts: including src_ready here would
    // feed fault -> writer stop -> src_ready back into the fault predicate.
    wire bad_cpl=incoming_cpl&&(!owned||!issued||cpl_buffered||cpl_delivered||
        r_cpl_tag[lane*64+:64]!=tag[63:0]||r_cpl_status[lane*2+:2]!=0||
        r_cpl_bytes[lane*13+:13]!=bytes||received!=beats);
    wire bad_retire=retire_valid&&(!owned||retire_tag!=tag||!last_delivered||
        !cpl_delivered||!enable);
    wire environment_bad=stop||reader_fault||(seen_ready&&!reader_started);
    wire error_now=environment_bad||wrong_lane||bad_data||bad_cpl||bad_retire||
        (request_fire&&bad_request)||(!owned&&(|r_data_valid|| |r_cpl_valid));
    assign fault=failed||error_now;
    assign reader_stop=fault;
    // Admission never depends on source VALID or its content (no ready loop).
    assign src_req_ready=cold_rst_n&&enable&&reader_started&&!owned&&!failed&&!environment_bad;
    assign r_cmd_addr={address,address};
    assign r_cmd_tag={tag[63:0],tag[63:0]};
    assign r_cmd_len={2{8'(beats-1)}};
    assign r_data_ready=(!fault&&cold_rst_n&&owned&&issued&&data_space) ? (lane?2'b10:2'b01):2'b00;
    assign r_cpl_ready=(!fault&&cold_rst_n&&owned&&issued&&!cpl_buffered&&!cpl_delivered) ? (lane?2'b10:2'b01):2'b00;
    assign src_tag=tag;
    assign src_cpl_valid=cpl_buffered&&last_delivered;
    assign src_cpl_tag=tag;
    assign src_cpl_bytes=bytes;
    assign src_cpl_status=0;
    wire data_fire=incoming_data&&r_data_ready[lane];
    wire cpl_fire=incoming_cpl&&r_cpl_ready[lane];
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            owned<=0;failed<=0;seen_ready<=0;lane<=0;issued<=0;
            tag<=0;address<=0;bytes<=0;received<=0;beats<=0;
            r_cmd_valid<=0;src_valid<=0;src_last<=0;src_data<=0;
            cpl_buffered<=0;last_delivered<=0;cpl_delivered<=0;
        end else begin
            if(reader_started)seen_ready<=1;
            if(error_now)failed<=1;
            if(request_fire)begin
                owned<=1;tag<=src_req_tag;address<=src_req_address;bytes<=src_req_bytes;
                lane<=src_req_address[33];beats<=src_req_bytes[12:6];received<=0;
                issued<=0;cpl_buffered<=0;last_delivered<=0;cpl_delivered<=0;
                if(!bad_request)r_cmd_valid<=src_req_address[33]?2'b10:2'b01;
            end
            // Existing command/data/CPL offers persist across a later fault.
            if(command_fire)begin r_cmd_valid[lane]<=0;issued<=1;end
            if(src_valid&&src_ready)begin
                src_valid<=0;
                if(src_last)last_delivered<=1;
            end
            if(data_fire)begin
                src_valid<=1;src_data<=r_data[lane*512+:512];
                src_last<=r_data_last[lane];received<=received+1'b1;
            end
            if(cpl_fire)cpl_buffered<=1;
            if(src_cpl_valid&&src_cpl_ready)begin cpl_buffered<=0;cpl_delivered<=1;end
            if(retire_valid&&!fault)owned<=0;
        end
    end
endmodule
`default_nettype wire
