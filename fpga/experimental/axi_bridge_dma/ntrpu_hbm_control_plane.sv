`timescale 1ns/1ps
`default_nettype none
// Experimental internal CSR contract; not yet attached to a native PCIe BAR.
// bus_rst_n resets transport only. Accepted work survives until canonical cold reset.
module ntrpu_hbm_control_plane (
    input wire clk,cold_rst_n,bus_rst_n,
    input wire [11:0] s_awaddr,s_araddr,
    input wire [2:0] s_awprot,s_arprot,
    input wire s_awvalid,s_wvalid,s_bready,s_arvalid,s_rready,
    input wire [31:0] s_wdata,
    input wire [3:0] s_wstrb,
    output wire s_awready,s_wready,s_arready,
    output reg s_bvalid=0,s_rvalid=0,
    output reg [1:0] s_bresp=0,s_rresp=0,
    output reg [31:0] s_rdata=0,
    output reg cfg_valid=0,cmd_valid=0,ack_valid=0,
    input wire cfg_ready,cmd_ready,ack_ready,
    output wire [63:0] cfg_session,cfg_generation,cfg_axi_base,cfg_pcie_base,
    output wire [127:0] cmd_tag,ack_tag,
    output wire [63:0] cmd_generation,cmd_source,cmd_offset,ack_generation,
    output wire [12:0] cmd_bytes,ack_bytes,
    input wire cfg_rsp_valid,cfg_rsp_error,reject_valid,result_valid,
    output wire cfg_rsp_ready,reject_ready,result_ready,
    input wire [3:0] reject_code,
    input wire [127:0] reject_tag,result_tag,
    input wire [12:0] result_bytes,
    input wire [3:0] telemetry_clk,telemetry_cold_rst_n,
    input wire [511:0] telemetry_data
);
    wire transport_rst_n=cold_rst_n&&bus_rst_n;
    reg aw_full=0,w_full=0;
    reg [11:0] aw_address=0;
    reg [31:0] write_data=0;
    reg [3:0] write_strobe=0;
    wire write_fire=transport_rst_n&&aw_full&&w_full&&!s_bvalid;
    reg write_ok;
    reg [31:0] read_data;
    reg read_ok;
    reg [255:0] cfg_shadow=0,cfg_active=0;
    reg [351:0] cmd_shadow=0,cmd_active=0;
    reg [223:0] ack_shadow=0,ack_active=0;
    reg [31:0] cfg_written=0;
    reg [43:0] cmd_written=0;
    reg [27:0] ack_written=0;
    reg cfg_wait=0,cmd_wait=0,cfg_full=0,event_full=0,cfg_error=0,ack_issued=0;
    reg [1:0] event_kind=0;
    reg [159:0] event_data=0;
    wire cfg_busy=cfg_valid||cfg_wait||cfg_full;
    wire cmd_busy=cmd_valid||cmd_wait||event_full;
    wire snap_busy,snap_valid;
    wire [31:0] snap_epoch;
    wire [511:0] snap_data;
    wire [3:0] snap_progress;
    wire [31:0] status={24'd0,ack_issued,snap_valid,snap_busy,event_full,cfg_full,ack_valid,cmd_busy,cfg_busy};

    assign {cfg_pcie_base,cfg_axi_base,cfg_generation,cfg_session}=cfg_active;
    assign {cmd_bytes,cmd_offset,cmd_source,cmd_generation,cmd_tag}=cmd_active[332:0];
    assign {ack_bytes,ack_generation,ack_tag}=ack_active[204:0];
    assign cfg_rsp_ready=cold_rst_n&&cfg_wait&&!cfg_full;
    assign reject_ready=cold_rst_n&&cmd_wait&&!event_full;
    assign result_ready=cold_rst_n&&cmd_wait&&!event_full&&!reject_valid;
    assign s_awready=transport_rst_n&&!aw_full&&!s_bvalid;
    assign s_wready=transport_rst_n&&!w_full&&!s_bvalid;
    assign s_arready=transport_rst_n&&!s_rvalid;

    ntrpu_control_snapshot snapshot(
        .clk(clk),.cold_rst_n(cold_rst_n),
        .request(write_fire&&write_ok&&aw_address==12'h020),
        .release_sample(write_fire&&write_ok&&aw_address==12'h030),
        .src_clk(telemetry_clk),.src_cold_rst_n(telemetry_cold_rst_n),.src_data(telemetry_data),
        .busy(snap_busy),.valid(snap_valid),.epoch(snap_epoch),.data(snap_data),.progress(snap_progress));

    // AxPROT is intentionally not an authorization check in this internal prototype.
    always @* begin
        write_ok=0;
        if(aw_address[1:0]==0) begin
            if((aw_address>=12'h100&&aw_address<=12'h11c)||
               (aw_address>=12'h140&&aw_address<=12'h168)||
               (aw_address>=12'h180&&aw_address<=12'h198)) write_ok=1;
            else if(write_strobe==4'hf&&write_data==1) begin
                case(aw_address)
                    12'h00c: write_ok=!cfg_busy&&!cmd_busy&&!ack_valid&&(&cfg_written);
                    12'h010: write_ok=!cfg_busy&&!cmd_busy&&!ack_valid&&(&cmd_written)&&cmd_shadow[351:333]==0;
                    12'h014: write_ok=cmd_wait&&ack_ready&&!reject_valid&&!result_valid&&!ack_issued&&
                                         (&ack_written)&&ack_shadow[223:205]==0;
                    12'h018: write_ok=cfg_full;
                    12'h01c: write_ok=event_full;
                    12'h020: write_ok=!snap_busy&&!snap_valid;
                    12'h030: write_ok=snap_valid;
                    default: write_ok=0;
                endcase
            end
        end
    end
    always @* begin
        read_data=0;read_ok=1;
        if(s_araddr[1:0]!=0) read_ok=0;
        else if(s_araddr>=12'h100&&s_araddr<=12'h11c) read_data=cfg_shadow[((s_araddr-12'h100)>>2)*32+:32];
        else if(s_araddr>=12'h140&&s_araddr<=12'h168) read_data=cmd_shadow[((s_araddr-12'h140)>>2)*32+:32];
        else if(s_araddr>=12'h180&&s_araddr<=12'h198) read_data=ack_shadow[((s_araddr-12'h180)>>2)*32+:32];
        else if(s_araddr>=12'h200&&s_araddr<=12'h210) read_data=event_data[((s_araddr-12'h200)>>2)*32+:32];
        else if(s_araddr>=12'h300&&s_araddr<=12'h33c) read_data=snap_data[((s_araddr-12'h300)>>2)*32+:32];
        else if(s_araddr>=12'h400&&s_araddr<=12'h41c) read_data=cfg_active[((s_araddr-12'h400)>>2)*32+:32];
        else if(s_araddr>=12'h440&&s_araddr<=12'h468) read_data=cmd_active[((s_araddr-12'h440)>>2)*32+:32];
        else if(s_araddr>=12'h480&&s_araddr<=12'h498) read_data=ack_active[((s_araddr-12'h480)>>2)*32+:32];
        else case(s_araddr)
            12'h000: read_data=32'h4e545243;
            12'h004: read_data=1;
            12'h008: read_data=status;
            12'h024: read_data=snap_epoch;
            12'h028: read_data={31'd0,cfg_error};
            12'h02c: read_data={30'd0,event_kind};
            12'h034: read_data={28'd0,snap_progress};
            12'h038: read_data=cfg_written;
            12'h03c: read_data=cmd_written[31:0];
            12'h040: read_data={20'd0,cmd_written[43:32]};
            12'h044: read_data={4'd0,ack_written};
            12'h048: read_data={27'd0,ack_valid,cmd_wait,cmd_valid,cfg_wait,cfg_valid};
            default: read_ok=0;
        endcase
    end
    always @(posedge clk or negedge transport_rst_n) begin
        if(!transport_rst_n) begin
            aw_full<=0;w_full<=0;aw_address<=0;write_data<=0;write_strobe<=0;
            s_bvalid<=0;s_bresp<=0;s_rvalid<=0;s_rresp<=0;s_rdata<=0;
        end else begin
            if(s_awvalid&&s_awready) begin aw_full<=1;aw_address<=s_awaddr;end
            if(s_wvalid&&s_wready) begin w_full<=1;write_data<=s_wdata;write_strobe<=s_wstrb;end
            if(write_fire) begin aw_full<=0;w_full<=0;s_bvalid<=1;s_bresp<=write_ok?2'b00:2'b10;end
            if(s_bvalid&&s_bready) s_bvalid<=0;
            if(s_arvalid&&s_arready) begin s_rvalid<=1;s_rdata<=read_data;s_rresp<=read_ok?2'b00:2'b10;end
            if(s_rvalid&&s_rready) s_rvalid<=0;
        end
    end
    always @(posedge clk or negedge cold_rst_n) begin
        if(!cold_rst_n) begin
            cfg_shadow<=0;cfg_active<=0;cmd_shadow<=0;cmd_active<=0;ack_shadow<=0;ack_active<=0;
            cfg_written<=0;cmd_written<=0;ack_written<=0;cfg_valid<=0;cmd_valid<=0;ack_valid<=0;
            cfg_wait<=0;cmd_wait<=0;cfg_full<=0;event_full<=0;cfg_error<=0;event_kind<=0;event_data<=0;ack_issued<=0;
        end else begin
            if(cfg_valid&&cfg_ready) begin cfg_valid<=0;cfg_wait<=1;end
            if(cmd_valid&&cmd_ready) begin cmd_valid<=0;cmd_wait<=1;end
            if(ack_valid&&ack_ready) ack_valid<=0;
            if(cfg_rsp_valid&&cfg_rsp_ready) begin cfg_wait<=0;cfg_full<=1;cfg_error<=cfg_rsp_error;end
            if(reject_valid&&reject_ready) begin
                cmd_wait<=0;event_full<=1;event_kind<=1;event_data<={28'd0,reject_code,reject_tag};
            end
            if(result_valid&&result_ready) begin
                cmd_wait<=0;event_full<=1;event_kind<=2;event_data<={19'd0,result_bytes,result_tag};
            end
            if(write_fire&&write_ok) begin
                if(aw_address>=12'h100&&aw_address<=12'h11c) begin
                    for(integer b=0;b<4;b=b+1) if(write_strobe[b]) begin
                        cfg_shadow[((aw_address-12'h100)>>2)*32+b*8+:8]<=write_data[b*8+:8];
                        cfg_written[((aw_address-12'h100)>>2)*4+b]<=1;
                    end
                end else if(aw_address>=12'h140&&aw_address<=12'h168) begin
                    for(integer b=0;b<4;b=b+1) if(write_strobe[b]) begin
                        cmd_shadow[((aw_address-12'h140)>>2)*32+b*8+:8]<=write_data[b*8+:8];
                        cmd_written[((aw_address-12'h140)>>2)*4+b]<=1;
                    end
                end else if(aw_address>=12'h180&&aw_address<=12'h198) begin
                    for(integer b=0;b<4;b=b+1) if(write_strobe[b]) begin
                        ack_shadow[((aw_address-12'h180)>>2)*32+b*8+:8]<=write_data[b*8+:8];
                        ack_written[((aw_address-12'h180)>>2)*4+b]<=1;
                    end
                end else case(aw_address)
                    12'h00c: begin cfg_active<=cfg_shadow;cfg_valid<=1;cfg_written<=0;end
                    12'h010: begin cmd_active<=cmd_shadow;cmd_valid<=1;cmd_written<=0;ack_issued<=0;end
                    12'h014: begin ack_active<=ack_shadow;ack_valid<=1;ack_written<=0;ack_issued<=1;end
                    12'h018: begin cfg_full<=0;cfg_error<=0;end
                    12'h01c: begin event_full<=0;event_kind<=0;event_data<=0;end
                    default: begin end
                endcase
            end
        end
    end
endmodule
`default_nettype wire
