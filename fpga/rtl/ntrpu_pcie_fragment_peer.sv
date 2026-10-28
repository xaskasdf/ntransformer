`timescale 1ns / 1ps
// NTRPUF01: one preauthorized line lease; joint ingress/receiver epoch transitions.
// Application stop invalidates ingress but does NOT stop an exposed peer
// write: drain it and retain the lease through its final matching AXI B.
// rst_n is a coordinated cold reset of this entire transport domain.
module ntrpu_pcie_fragment_peer (
    input wire clk, rst_n, stop,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    input wire arm_valid,
    output wire arm_ready,
    input wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
    output wire [63:0] epoch,
    output wire lease_held, failed,
    output wire [3:0] error_code,
    output wire [31:0] rejected_frames, rejected_identity,
    output wire [3:0] last_reject_code,
    output wire [6:0] last_reject_faults,

    output wire transport_idle,
    output wire [63:0] valid_bytes,
    output wire [63:0] peer_epoch,
    output wire peer_quarantined, peer_axi_busy,
    output reg [2:0] bridge_phase,
    output reg [2:0] destination_status,

    input wire [63:0] s_axi_awaddr,
    input wire [3:0] s_axi_awid,
    input wire [7:0] s_axi_awlen,
    input wire [2:0] s_axi_awsize,
    input wire [1:0] s_axi_awburst,
    input wire s_axi_awlock, s_axi_awvalid,
    output wire s_axi_awready,
    input wire [511:0] s_axi_wdata,
    input wire [63:0] s_axi_wstrb,
    input wire s_axi_wlast, s_axi_wvalid,
    output wire s_axi_wready,
    output wire [3:0] s_axi_bid,
    output wire [1:0] s_axi_bresp,
    output wire s_axi_bvalid,
    input wire s_axi_bready,
    output wire result_valid,
    input wire result_ready,
    output wire [63:0] result_epoch, result_tag,
    output wire [3:0] result_status,

    output wire [63:0] m_axi_awaddr,
    output wire [7:0] m_axi_awlen,
    output wire [2:0] m_axi_awsize,
    output wire [1:0] m_axi_awburst,
    output wire [3:0] m_axi_awcache, m_axi_awid,
    output wire [2:0] m_axi_awprot,
    output wire m_axi_awlock, m_axi_awvalid,
    input wire m_axi_awready,
    output wire [511:0] m_axi_wdata,
    output wire [63:0] m_axi_wstrb,
    output wire m_axi_wlast, m_axi_wvalid,
    input wire m_axi_wready,
    input wire [1:0] m_axi_bresp,
    input wire [3:0] m_axi_bid,
    input wire m_axi_bvalid,
    output wire m_axi_bready
);
    localparam [2:0] EMPTY=0, COMMAND=1, DATA=2, COMPLETION=3, DRAIN=4, RETURN=5;
    reg active_context, foreign_completion;
    reg [63:0] saved_epoch,saved_tag,saved_address;
    reg [511:0] saved_data;
    wire ingress_arm_ready,peer_rearm_ready;
    wire peer_lease_held;
    wire line_valid,line_ready,done_valid,done_ready;
    wire [63:0] line_epoch,line_tag,line_address;
    wire [511:0] line_data;
    wire cmd_ready,data_ready,cpl_valid,cpl_ready;
    wire [63:0] cpl_epoch,cpl_tag;
    wire [2:0] cpl_status;
    // Stop the receiver only between drained leases to prepare its rearm.
    wire receiver_stop = rst_n && !active_context && bridge_phase==EMPTY &&
                         !peer_axi_busy && !peer_quarantined;
    assign arm_ready = rst_n && !active_context && bridge_phase==EMPTY &&
                       !peer_axi_busy && epoch==peer_epoch &&
                       ingress_arm_ready && peer_rearm_ready;
    wire joint_arm = arm_valid && arm_ready;
    assign line_ready = active_context && bridge_phase==EMPTY;
    assign cpl_ready = bridge_phase==COMPLETION;
    assign done_valid = bridge_phase==RETURN;

    ntrpu_pcie_fragment_ingress ingress (
        .clk(clk),.rst_n(rst_n),.stop(stop),.source_stopped(source_stopped),
        .ingress_isolated(ingress_isolated),.quiesce_epoch(quiesce_epoch),
        .arm_valid(joint_arm),.arm_ready(ingress_arm_ready),.arm_epoch(arm_epoch),
        .arm_tag(arm_tag),.arm_window(arm_window),.arm_destination(arm_destination),
        .epoch(epoch),.lease_held(lease_held),.failed(failed),.error_code(error_code),
        .rejected_frames(rejected_frames),.rejected_identity(rejected_identity),
        .last_reject_code(last_reject_code),.last_reject_faults(last_reject_faults),
        .transport_idle(transport_idle),.valid_bytes(valid_bytes),
        .s_axi_awaddr(s_axi_awaddr),.s_axi_awid(s_axi_awid),.s_axi_awlen(s_axi_awlen),
        .s_axi_awsize(s_axi_awsize),.s_axi_awburst(s_axi_awburst),.s_axi_awlock(s_axi_awlock),
        .s_axi_awvalid(s_axi_awvalid),.s_axi_awready(s_axi_awready),
        .s_axi_wdata(s_axi_wdata),.s_axi_wstrb(s_axi_wstrb),.s_axi_wlast(s_axi_wlast),
        .s_axi_wvalid(s_axi_wvalid),.s_axi_wready(s_axi_wready),
        .s_axi_bid(s_axi_bid),.s_axi_bresp(s_axi_bresp),.s_axi_bvalid(s_axi_bvalid),.s_axi_bready(s_axi_bready),
        .line_valid(line_valid),.line_ready(line_ready),.line_epoch(line_epoch),.line_tag(line_tag),
        .line_address(line_address),.line_data(line_data),.done_valid(done_valid),.done_ready(done_ready),
        .done_epoch(saved_epoch),.done_tag(saved_tag),
        .done_error(destination_status!=0 || foreign_completion),
        .result_valid(result_valid),.result_ready(result_ready),
        .result_epoch(result_epoch),.result_tag(result_tag),.result_status(result_status)
    );
    ntrpu_peer_rx_guard #(.MAX_BEATS(1)) receiver (
        .clk(clk),.rst_n(rst_n),.stop(receiver_stop),.source_stopped(source_stopped),
        .ingress_isolated(ingress_isolated),.quiesce_epoch(quiesce_epoch),
        .rearm_valid(joint_arm),.rearm_ready(peer_rearm_ready),.rearm_epoch(arm_epoch),
        .epoch(peer_epoch),.quarantined(peer_quarantined),.lease_held(peer_lease_held),.axi_busy(peer_axi_busy),
        .cmd_valid(bridge_phase==COMMAND),.cmd_ready(cmd_ready),.cmd_epoch(saved_epoch),
        .cmd_tag(saved_tag),.cmd_addr(saved_address),.cmd_len(8'd0),
        .data_valid(bridge_phase==DATA),.data_ready(data_ready),.data(saved_data),.data_last(1'b1),
        .cpl_valid(cpl_valid),.cpl_ready(cpl_ready),.cpl_epoch(cpl_epoch),.cpl_tag(cpl_tag),.cpl_status(cpl_status),
        .m_axi_awaddr(m_axi_awaddr),.m_axi_awlen(m_axi_awlen),.m_axi_awsize(m_axi_awsize),
        .m_axi_awburst(m_axi_awburst),.m_axi_awcache(m_axi_awcache),.m_axi_awid(m_axi_awid),
        .m_axi_awprot(m_axi_awprot),.m_axi_awlock(m_axi_awlock),.m_axi_awvalid(m_axi_awvalid),.m_axi_awready(m_axi_awready),
        .m_axi_wdata(m_axi_wdata),.m_axi_wstrb(m_axi_wstrb),.m_axi_wlast(m_axi_wlast),
        .m_axi_wvalid(m_axi_wvalid),.m_axi_wready(m_axi_wready),
        .m_axi_bresp(m_axi_bresp),.m_axi_bid(m_axi_bid),.m_axi_bvalid(m_axi_bvalid),.m_axi_bready(m_axi_bready)
    );
    always @(posedge clk) begin
        if(!rst_n) begin
            active_context<=0;bridge_phase<=EMPTY;foreign_completion<=0;
            saved_epoch<=0;saved_tag<=0;saved_address<=0;saved_data<=0;destination_status<=0;
        end else begin
            if(joint_arm) begin active_context<=1;destination_status<=0;foreign_completion<=0;end
            if(result_valid && result_ready) active_context<=0;
            case(bridge_phase)
                EMPTY: if(line_valid && line_ready) begin
                    saved_epoch<=line_epoch;saved_tag<=line_tag;saved_address<=line_address;saved_data<=line_data;
                    bridge_phase<=COMMAND;
                end
                COMMAND: if(cmd_ready) bridge_phase<=DATA;
                DATA: if(data_ready) bridge_phase<=COMPLETION;
                COMPLETION: if(cpl_valid && cpl_ready) begin
                    if(cpl_epoch!=saved_epoch || cpl_tag!=saved_tag) foreign_completion<=1;
                    else begin destination_status<=cpl_status;bridge_phase<=DRAIN;end
                end
                // Error CPL may be early (e.g. foreign BID); never equate it
                // to completion of the receiver's exposed AXI transaction.
                DRAIN: if(!peer_axi_busy) bridge_phase<=RETURN;
                RETURN: if(done_valid && done_ready) bridge_phase<=EMPTY;
                default: begin end
            endcase
        end
    end
endmodule
