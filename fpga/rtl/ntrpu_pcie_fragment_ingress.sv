`timescale 1ns / 1ps

// NTRPUF01: one authorized64-byte line, one AXI write frame outstanding.
// Frame-local format/range/identity rejection never poisons the current lease.
// Valid current-identity conflicts/late writes fail an unreported lease.
// Rejection counters are global and saturating; ARM does not clear them.
// Always drain admitted transport, including after stop. rst_n is a cold,
// coordinated transport reset, NEVER an application cancellation signal.
// Successful completion requires a trusted, epoch-specific external barrier.
module ntrpu_pcie_fragment_ingress (
    input wire clk, rst_n, stop,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    input wire arm_valid,
    output wire arm_ready,
    input wire [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
    output reg [63:0] epoch,
    output reg lease_held, failed,
    output reg [3:0] error_code, // Lease errors:1stop,4conflict,5destination,6late write; frame counters also2format,3range,7identity
    output wire transport_idle,
    output reg [63:0] valid_bytes,
    output reg [31:0] rejected_frames, rejected_identity,
    output reg [3:0] last_reject_code,
    output reg [6:0] last_reject_faults,

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
    output reg [3:0] s_axi_bid,
    output reg [1:0] s_axi_bresp,
    output reg s_axi_bvalid,
    input wire s_axi_bready,

    output reg line_valid,
    input wire line_ready,
    output wire [63:0] line_epoch, line_tag, line_address,
    output wire [511:0] line_data,
    input wire done_valid,
    output wire done_ready,
    input wire [63:0] done_epoch, done_tag,
    input wire done_error,
    output reg result_valid,
    input wire result_ready,
    output wire [63:0] result_epoch, result_tag,
    output reg [3:0] result_status
);
    reg [63:0] window_line, destination, tag;
    reg [511:0] payload;
    reg aw_held, w_held, exposed, sent, completed, reported;
    reg [63:0] awaddr, wstrb;
    reg [3:0] awid;
    reg [7:0] awlen;
    reg [2:0] awsize;
    reg [1:0] awburst;
    reg awlock, wlast;
    reg [511:0] wdata;
    reg extra_beat; // sticky until actual WLAST, no wrapping beat counter
    wire decode_accept;
    wire [6:0] decode_faults;
    reg [3:0] frame_reject_code;
    reg [6:0] frame_reject_faults;
    reg frame_identity_reject;
    wire [63:0] decoded_valid;
    wire [511:0] decoded_data;
    reg conflict;

    ntrpu_pcie_fragment_decode decode (
        .lease_valid(lease_held), .lease_window(window_line),
        .lease_epoch(epoch), .lease_tag(tag),
        .awaddr(awaddr), .awlen(awlen), .awsize(awsize),
        .awburst(awburst), .awlock(awlock), .wdata(wdata),
        .wstrb(wstrb), .wlast(wlast), .accept(decode_accept),
        .faults(decode_faults),
        .byte_valid(decoded_valid), .byte_data(decoded_data)
    );
    always @* begin
        conflict = 0;
        for (integer lane = 0; lane < 64; lane = lane + 1)
            if (decoded_valid[lane] && valid_bytes[lane] &&
                decoded_data[lane*8+:8] != payload[lane*8+:8]) conflict = 1;
    end

    assign s_axi_awready = rst_n && !aw_held && !s_axi_bvalid;
    assign s_axi_wready = rst_n && !w_held && !s_axi_bvalid;
    assign transport_idle = !aw_held && !w_held && !s_axi_bvalid;
    wire quiescent = source_stopped && ingress_isolated && quiesce_epoch == epoch;
    wire no_arrivals = !s_axi_awvalid && !s_axi_wvalid;
    assign arm_ready = rst_n && !stop && transport_idle && no_arrivals &&
                       quiescent && !line_valid && (!sent || completed) &&
                       !result_valid && (!lease_held || reported) &&
                       arm_epoch > epoch && arm_window[5:0] == 0 && arm_destination[5:0] == 0;
    assign line_epoch = epoch;
    assign line_tag = tag;
    assign line_address = destination;
    assign line_data = payload;
    assign done_ready = rst_n && sent && !completed;
    assign result_epoch = epoch;
    assign result_tag = tag;

    wire write_reject = stop || failed || exposed || extra_beat || !decode_accept || conflict;
    wire identity_reject = !(decode_faults[0] || decode_faults[1] || decode_faults[2] ||
                             decode_faults[3] || decode_faults[6]) &&
                            (decode_faults[4] || decode_faults[5]);
    wire [3:0] write_error = stop ? 4'd1 : failed ? error_code :
                             extra_beat ? 4'd2 :
                             decode_faults[0] || decode_faults[2] ? 4'd3 :
                             decode_faults[1] || decode_faults[3] || decode_faults[6] ? 4'd2 :
                             identity_reject ? 4'd7 : exposed ? 4'd6 :
                             conflict ? 4'd4 : 4'd0;
    wire [1:0] write_response = !lease_held || decode_faults[2] ? 2'b11 :
                                write_reject ? 2'b10 : 2'b00;
    // Untrusted or old frames may never change the current lease outcome.
    // A well-formed current frame after publication is a protocol violation;
    // already reported results are immutable, even if a late frame arrives.
    wire local_violation = lease_held && !reported && !failed && !stop &&
                           !extra_beat && decode_accept && (exposed || conflict);

    always @(posedge clk) begin
        if (!rst_n) begin
            epoch<=0;lease_held<=0;failed<=0;error_code<=0;
            window_line<=0;destination<=0;tag<=0;payload<=0;valid_bytes<=0;
            aw_held<=0;w_held<=0;exposed<=0;sent<=0;completed<=0;reported<=0;
            awaddr<=0;awid<=0;awlen<=0;awsize<=0;awburst<=0;awlock<=0;
            wdata<=0;wstrb<=0;wlast<=0;extra_beat<=0;
            s_axi_bid<=0;s_axi_bresp<=0;s_axi_bvalid<=0;
            line_valid<=0;result_valid<=0;result_status<=0;
            rejected_frames<=0;rejected_identity<=0;last_reject_code<=0;last_reject_faults<=0;
            frame_reject_code<=0;frame_reject_faults<=0;frame_identity_reject<=0;
        end else begin
            if (s_axi_bvalid && s_axi_bready) s_axi_bvalid<=0;
            if (result_valid && result_ready) result_valid<=0;

            if (s_axi_awvalid && s_axi_awready) begin
                aw_held<=1;awaddr<=s_axi_awaddr;awid<=s_axi_awid;
                awlen<=s_axi_awlen;awsize<=s_axi_awsize;
                awburst<=s_axi_awburst;awlock<=s_axi_awlock;
            end
            if (s_axi_wvalid && s_axi_wready) begin
                w_held<=1;wdata<=s_axi_wdata;wstrb<=s_axi_wstrb;wlast<=s_axi_wlast;
            end
            if (aw_held && w_held) begin
                w_held<=0;
                // Malformed/unsupported frames never update any byte. Drain
                // through actual WLAST; absence of WLAST intentionally blocks
                // retirement/rearm until coordinated transport recovery.
                if (!extra_beat) begin
                    frame_reject_code<=write_error;
                    frame_reject_faults<=decode_faults;
                    frame_identity_reject<=identity_reject;
                end
                if (local_violation) begin failed<=1;error_code<=write_error;end
                if (!write_reject) begin
                    for (integer lane=0;lane<64;lane=lane+1)
                        if (decoded_valid[lane]) payload[lane*8+:8]<=decoded_data[lane*8+:8];
                    valid_bytes<=valid_bytes | decoded_valid;
                end
                if (wlast) begin
                    aw_held<=0;extra_beat<=0;
                    s_axi_bid<=awid;s_axi_bresp<=extra_beat ? 2'b10 : write_response;s_axi_bvalid<=1;
                    if (write_reject) begin
                        if (rejected_frames != 32'hffffffff) rejected_frames<=rejected_frames+1;
                        if ((extra_beat ? frame_identity_reject : identity_reject) && rejected_identity != 32'hffffffff)
                            rejected_identity<=rejected_identity+1;
                        last_reject_code<=extra_beat ? frame_reject_code : write_error;
                        last_reject_faults<=extra_beat ? frame_reject_faults : decode_faults;
                    end
                end else extra_beat<=1;
            end

            // Publication is registered and waits for ingress responses to
            // drain. Once VALID is visible, stop cannot retract it.
            if (lease_held && !exposed && !failed && !stop &&
                valid_bytes == 64'hffffffffffffffff && transport_idle && no_arrivals) begin
                exposed<=1;line_valid<=1;
            end
            if (line_valid && line_ready) begin line_valid<=0;sent<=1;end
            if (done_valid && done_ready) begin
                if (done_epoch==epoch && done_tag==tag) begin
                    completed<=1;
                    if (done_error && !failed) begin failed<=1;error_code<=5;end
                end else if (!failed) begin failed<=1;error_code<=5;end
            end
            if (stop && lease_held && !reported && !failed) begin failed<=1;error_code<=1;end

            // An ingress B acknowledges staging only. Even a successful
            // destination response is insufficient without this barrier.
            if (lease_held && !reported && transport_idle && no_arrivals && quiescent &&
                !line_valid && (!exposed || completed) && (failed || completed)) begin
                reported<=1;result_valid<=1;result_status<=stop ? 4'd1 : error_code;
            end
            if (arm_valid && arm_ready) begin
                epoch<=arm_epoch;tag<=arm_tag;window_line<=arm_window;destination<=arm_destination;
                lease_held<=1;failed<=0;error_code<=0;payload<=0;valid_bytes<=0;
                exposed<=0;sent<=0;completed<=0;reported<=0;extra_beat<=0;
                result_status<=0;
            end
        end
    end
endmodule
