`timescale 1ns/1ps
// Public register sequencer in axi_aclk, channel0 H2C/C2H, AXI-MM.
// Requires exclusive register ownership and the coherent descriptor ledger.
// snapshot_valid is a local register observation, NEVER global drain/reset.
// An external supervisor supplies wait_expired (including a stopped clock).
// Once offered, a master request remains valid through any revocation/fault.
// The caller may release only after complete global retirement of this session.
module ntrpu_xdma_stop_registers (
    input wire clk, cold_rst_n, core_ready, exclusive_owner, external_fault, wait_expired,
    input wire start_valid,
    output wire start_ready,
    input wire [63:0] start_generation, start_tag,
    input wire close_request, host_closed,
    input wire [63:0] host_generation, host_tag,
    input wire ledger_active, ledger_closed, ledger_accounted,
    input wire [63:0] ledger_generation, ledger_tag, ledger_published, ledger_completed,
    input wire [8:0] ledger_faults,
    input wire release_valid,
    output reg active=0,
    output reg [63:0] generation=0, tag=0, expected_published=0,
    output reg [13:0] faults=0,
    output wire snapshot_valid,
    output reg [63:0] control_snapshot=0, status_snapshot=0, count_snapshot=0,
    output reg [1:0] stop_responses=0,
    output reg [5:0] read_responses=0,
    output wire transaction_pending,
    output wire m_admit, m_req_valid,
    input wire m_req_ready,
    output wire m_req_write,
    output wire [31:0] m_req_address,
    output wire [31:0] m_req_data,
    output wire [3:0] m_req_strobe,
    output wire [127:0] m_req_identity,
    input wire m_rsp_valid,
    output wire m_rsp_ready,
    input wire [31:0] m_rsp_data,
    input wire [1:0] m_rsp_code,
    input wire [127:0] m_rsp_identity,
    input wire m_rsp_timeout, m_timeout_latched
);
    localparam [2:0] IDLE=0, SEND=1, RECEIVE=2, ASSESS=3, CAPTURED=4, FAULTED=5;
    localparam [31:0] H2C_ERRORS=32'h00f8fe38, C2H_ERRORS=32'h00f80638;
    localparam [31:0] H2C_DEFINED=H2C_ERRORS|32'h47, C2H_DEFINED=C2H_ERRORS|32'h47;
    reg [2:0] state=IDLE;
    reg [2:0] step=0;
    reg have_identity=0;
    wire fresh = start_generation!=0 && start_tag!=0 && (!have_identity ||
        start_generation>generation || (start_generation==generation && start_tag>tag));
    wire ready_context = core_ready && exclusive_owner && !external_fault && !wait_expired;
    wire arm_context = ready_context && close_request && host_closed && ledger_active && ledger_closed &&
        !ledger_faults && !m_timeout_latched && host_generation==start_generation && host_tag==start_tag &&
        ledger_generation==start_generation && ledger_tag==start_tag;
    wire same_ledger = ledger_active && ledger_closed && ledger_generation==generation && ledger_tag==tag;
    wire same_host = close_request && host_closed && host_generation==generation && host_tag==tag;
    wire inventory_unchanged = same_ledger && !ledger_faults && ledger_published==expected_published;
    wire live_context = ready_context && same_host && inventory_unchanged && !m_timeout_latched;
    wire response_bad = m_rsp_code!=0 || m_rsp_identity!={generation,tag} || m_rsp_timeout;
    wire snapshots_quiet = !control_snapshot[0] && !control_snapshot[32] &&
        !status_snapshot[0] && !status_snapshot[32] && count_snapshot==expected_published &&
        ledger_completed==expected_published && ledger_accounted;
    assign start_ready = cold_rst_n && !active && !faults && fresh && arm_context;
    assign transaction_pending = state==SEND || state==RECEIVE;
    assign snapshot_valid = cold_rst_n && active && state==CAPTURED && !faults && live_context &&
        ledger_accounted && ledger_completed==expected_published && !start_valid && !m_rsp_valid;
    assign m_admit = active;
    assign m_req_valid = state==SEND;
    assign m_req_write = step<2;
    assign m_req_data = step<2 ? 32'd1 : 32'd0;
    assign m_req_strobe = step<2 ? 4'hf : 4'h0;
    assign m_req_identity = {generation,tag};
    assign m_rsp_ready = state==RECEIVE;
    // A continuous decode is defined from the declaration-initialized step,
    // including the first request before step has ever changed.
    assign m_req_address = step==0 ? 32'h000c : step==1 ? 32'h100c :
                           step==2 ? 32'h0004 : step==3 ? 32'h0040 :
                           step==4 ? 32'h0048 : step==5 ? 32'h1004 :
                           step==6 ? 32'h1040 : 32'h1048;
    always @(posedge clk) begin
        if(!cold_rst_n) begin
            state<=IDLE;step<=0;active<=0;have_identity<=0;generation<=0;tag<=0;
            expected_published<=0;faults<=0;control_snapshot<=0;status_snapshot<=0;
            count_snapshot<=0;stop_responses<=0;read_responses<=0;
        end else begin
            if(start_valid) begin
                if(active) faults[12]<=1;
                else if(!fresh) faults[1]<=1;
                else if(start_ready) begin
                    active<=1;have_identity<=1;generation<=start_generation;tag<=start_tag;
                    expected_published<=ledger_published;step<=0;state<=SEND;
                    control_snapshot<=0;status_snapshot<=0;count_snapshot<=0;
                    stop_responses<=0;read_responses<=0;
                end
            end
            if(active) begin
                if(!core_ready || !exclusive_owner || external_fault) faults[0]<=1;
                if(!same_ledger) faults[1]<=1;
                if(!same_host) faults[2]<=1;
                if(ledger_faults!=0 || ledger_published!=expected_published) faults[3]<=1;
                if(m_timeout_latched) faults[4]<=1;
                if(wait_expired) faults[10]<=1;
                if(state==CAPTURED && (!ledger_accounted || ledger_completed!=expected_published))
                    faults[13]<=1;
            end
            if(m_rsp_valid && state!=RECEIVE) faults[12]<=1;
            case(state)
                IDLE: begin end
                SEND: if(m_req_valid && m_req_ready) state<=RECEIVE;
                RECEIVE: if(m_rsp_valid && m_rsp_ready) begin
                    if(m_rsp_timeout) faults[4]<=1;
                    if(m_rsp_code!=0) faults[5]<=1;
                    if(m_rsp_identity!={generation,tag}) faults[6]<=1;
                    if(step<2) begin
                        if(!response_bad) stop_responses[step]<=1;
                    end else begin
                        if(!response_bad) begin
                            read_responses[step-2]<=1;
                            case(step)
                                2:control_snapshot[31:0]<=m_rsp_data;
                                3:status_snapshot[31:0]<=m_rsp_data;
                                4:count_snapshot[31:0]<=m_rsp_data;
                                5:control_snapshot[63:32]<=m_rsp_data;
                                6:status_snapshot[63:32]<=m_rsp_data;
                                7:count_snapshot[63:32]<=m_rsp_data;
                            endcase
                            if(step==3 || step==6) begin
                                if(|(m_rsp_data & (step==3?H2C_ERRORS:C2H_ERRORS))) faults[7]<=1;
                                if(|(m_rsp_data & ~(step==3?H2C_DEFINED:C2H_DEFINED))) faults[8]<=1;
                            end
                            if(step==4 || step==7) begin
                                if(m_rsp_data>(step==4?expected_published[31:0]:expected_published[63:32]) ||
                                   m_rsp_data<(step==4?count_snapshot[31:0]:count_snapshot[63:32])) faults[9]<=1;
                            end
                        end
                    end
                    // Stop both channels before any status polling, even when
                    // the first real B carries an error. Never fabricate B.
                    if(step==0) begin step<=1;state<=SEND;end
                    else if(faults || response_bad || !live_context) state<=FAULTED;
                    else if(step==7) state<=ASSESS;
                    else begin step<=step+1'b1;state<=SEND;end
                end
                ASSESS: begin
                    if(faults || !live_context) state<=FAULTED;
                    else if(stop_responses==2'b11 && read_responses==6'b111111 && snapshots_quiet)
                        state<=CAPTURED;
                    else begin step<=2;read_responses<=0;state<=SEND;end
                end
                CAPTURED: begin end
                FAULTED: begin end
                default: begin faults[12]<=1;state<=FAULTED;end
            endcase
            if(release_valid) begin
                if(snapshot_valid) begin active<=0;state<=IDLE;end
                else faults[11]<=1;
            end
        end
    end
endmodule
