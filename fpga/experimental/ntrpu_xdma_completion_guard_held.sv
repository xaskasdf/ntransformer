`timescale 1ns/1ps
`default_nettype none
// Isolated held-grant variant of the frozen completion policy.
// Fixes polling/READY phase starvation; native IP qualification is pending.
// Same axi_aclk as descriptor bypass, status ports and local AXI-Lite master.
// Exclusive ownership MUST cover host BAR writes and clear-on-read accesses too.
// This module never writes Run, clears errors, resets XDMA or proves PCIe drain.
module ntrpu_xdma_completion_guard_held #(
    parameter integer WAIT_LIMIT = 65536
) (
    input wire clk, cold_rst_n, core_ready, exclusive_owner,
    input wire external_fault, wait_expired, reset_request,
    input wire start_valid,
    output wire start_ready,
    input wire [63:0] start_epoch,
    input wire [1:0] native_run, descriptor_load, native_done,
    output wire [1:0] admit_descriptor, validated_done, dma_error,
    output reg active = 0,
    output reg [15:0] faults = 0,
    output wire [63:0] published_count, observed_count, delivered_count,
    output reg [63:0] control_snapshot = 0, status_snapshot = 0, count_snapshot = 0,
    output wire m_admit, m_req_valid, m_req_write,
    input wire m_req_ready,
    output wire [31:0] m_req_address, m_req_data,
    output wire [3:0] m_req_strobe,
    output wire [127:0] m_req_identity,
    input wire m_rsp_valid,
    output wire m_rsp_ready,
    input wire [31:0] m_rsp_data,
    input wire [1:0] m_rsp_code,
    input wire [127:0] m_rsp_identity,
    input wire m_rsp_timeout, m_timeout_latched
);
    localparam [2:0] OFF=0, SEND=1, RECEIVE=2, ASSESS=3, READY=4, CREDIT=5, FAILED=6;
    localparam [31:0] CONTROL = 32'h00f83e7f;
    // Fail closed on errors, stopped indications and reserved bits.
    // Busy and Descriptor_completed alone are allowed, neither proves drain.
    localparam [31:0] STATUS_ALLOWED = 32'h00000005;
    localparam integer TIMER_WIDTH = WAIT_LIMIT > 1 ? $clog2(WAIT_LIMIT) : 1;
    reg [2:0] state = OFF, step = 0;
    reg boot = 1, credit_channel = 0;
    reg [63:0] epoch = 0, serial = 0;
    reg [31:0] published[0:1], observed[0:1], delivered[0:1], target[0:1], last_count[0:1];
    reg [TIMER_WIDTH-1:0] elapsed = 0;
    reg [15:0] fault_now;
    wire outstanding = published[0]!=delivered[0] || published[1]!=delivered[1];
    wire waiting = active && (boot || outstanding);
    wire expires = waiting && elapsed == WAIT_LIMIT-1;
    wire channel = step[2];
    wire [1:0] field = step[1:0];
    wire context_ok = core_ready && exclusive_owner && !external_fault && !wait_expired && !reset_request;
    wire response = state==RECEIVE && m_rsp_valid;
    wire can_offer = active && state==READY && !faults && context_ok && !m_timeout_latched && !expires;
    reg [1:0] held_grant = 0;
    wire healthy = active && !faults && context_ok && !m_timeout_latched && !expires;
    wire response_bad = m_rsp_code!=0 || m_rsp_timeout || m_rsp_identity!={epoch,serial};

    assign start_ready = cold_rst_n && !active && !faults && context_ok && !m_timeout_latched && start_epoch!=0;
    assign published_count = {published[1],published[0]};
    assign observed_count = {observed[1],observed[0]};
    assign delivered_count = {delivered[1],delivered[0]};
    // Registered error avoids feedback through the bridge's descriptor_load.
    // validated_done still suppresses a newly observed fault on the same edge.
    assign dma_error = {2{active && |faults}};
    // No dependency on descriptor_load: avoid a combinational admission loop.
    // A load which violates this grant poisons the epoch before any credit.
    for(genvar g=0;g<2;g=g+1) begin: grants
        assign admit_descriptor[g] = healthy && (can_offer || held_grant[g]) && native_run[g] && published[g]!=32'hffffffff &&
                                     published[g]-delivered[g]<2;
        assign validated_done[g] = cold_rst_n && active && state==CREDIT && credit_channel==g &&
                                   !faults && !fault_now && delivered[g]<target[g];
    end
    // SEND is irrevocable even if ownership/link is lost while req_ready=0.
    // The local master retains an accepted AR until handshake, then awaits R.
    assign m_admit = active;
    assign m_req_valid = state==SEND;
    assign m_req_write = 0;
    assign m_req_data = 0;
    assign m_req_strobe = 0;
    assign m_req_identity = {epoch,serial};
    assign m_rsp_ready = state==RECEIVE;
    assign m_req_address = (channel ? 32'h1100 : 32'h1000) +
                           (field==1 ? 32'h48 : field==2 ? 32'h40 : 32'h04);

    always @* begin
        fault_now = 0;
        if(active) begin
            if(!core_ready || native_run!=2'b11) fault_now[0]=1;
            if(!exclusive_owner) fault_now[1]=1;
            if(external_fault || reset_request) fault_now[2]=1;
            if(wait_expired || expires || m_timeout_latched) fault_now[3]=1;
            if(start_valid) fault_now[4]=1;
            for(integer c=0;c<2;c=c+1) begin
                if(descriptor_load[c] && !admit_descriptor[c]) fault_now[5]=1;
                if(descriptor_load[c] && published[c]==32'hffffffff) fault_now[6]=1;
                if(native_done[c] && (observed[c]==32'hffffffff ||
                   {1'b0,observed[c]}+33'd1>{1'b0,published[c]}+{32'd0,descriptor_load[c]})) fault_now[7]=1;
            end
            if(m_rsp_valid && state!=RECEIVE) fault_now[8]=1;
            if(response) begin
                if(m_rsp_code!=0) fault_now[9]=1;
                if(m_rsp_timeout) fault_now[3]=1;
                if(m_rsp_identity!={epoch,serial}) fault_now[10]=1;
                if(!response_bad) begin
                    if((field==0 || field==3) && m_rsp_data!=CONTROL) fault_now[11]=1;
                    if(field==2 && |(m_rsp_data & ~STATUS_ALLOWED)) fault_now[12]=1;
                    if(field==1 && (m_rsp_data<last_count[channel] || m_rsp_data>published[channel] ||
                                   (boot && m_rsp_data!=0))) fault_now[13]=1;
                end
            end
            if(serial==64'hffffffffffffffff) fault_now[14]=1;
        end else if(|descriptor_load || |native_done) fault_now[15]=1;
    end

    always @(posedge clk) begin
        if(!cold_rst_n) begin
            state<=OFF;step<=0;boot<=1;credit_channel<=0;epoch<=0;serial<=0;
            active<=0;faults<=0;elapsed<=0;held_grant<=0;
            control_snapshot<=0;status_snapshot<=0;count_snapshot<=0;
            for(integer c=0;c<2;c=c+1) begin
                published[c]<=0;observed[c]<=0;delivered[c]<=0;target[c]<=0;last_count[c]<=0;
            end
        end else begin
            faults <= faults | fault_now;
            // A grant survives polling until its descriptor is actually accepted.
            // Only READY creates grants. Audit targets remain captured before
            // polling; descriptors published during a poll await a later audit.
            for(integer g=0;g<2;g=g+1)begin
                if(can_offer && native_run[g] && published[g]!=32'hffffffff &&
                   published[g]-delivered[g]<2)held_grant[g]<=1;
                if(descriptor_load[g] || faults || fault_now)held_grant[g]<=0;
            end
            if(waiting) begin
                if(!expires) elapsed<=elapsed+1'b1;
            end else elapsed<=0;
            if(active) begin
                for(integer c=0;c<2;c=c+1) begin
                    if(descriptor_load[c] && published[c]!=32'hffffffff) published[c]<=published[c]+1'b1;
                    if(native_done[c] && observed[c]!=32'hffffffff) observed[c]<=observed[c]+1'b1;
                    if(validated_done[c]) delivered[c]<=delivered[c]+1'b1;
                end
            end
            case(state)
                OFF: if(start_valid && start_ready && !fault_now) begin
                    active<=1;epoch<=start_epoch;serial<=1;state<=SEND;step<=0;
                end
                SEND: if(m_req_ready) state<=RECEIVE;
                RECEIVE: if(response) begin
                    if(!response_bad) begin
                        case(field)
                            0,3: control_snapshot[channel*32+:32]<=m_rsp_data;
                            1: begin count_snapshot[channel*32+:32]<=m_rsp_data;last_count[channel]<=m_rsp_data;end
                            2: status_snapshot[channel*32+:32]<=m_rsp_data;
                        endcase
                    end
                    if(faults || fault_now) state<=FAILED;
                    else if(step==7) state<=ASSESS;
                    else begin step<=step+1'b1;serial<=serial+1'b1;state<=SEND;end
                end
                ASSESS: begin
                    if(faults || fault_now) state<=FAILED;
                    else if(boot) begin boot<=0;state<=READY;end
                    else if(count_snapshot[31:0]>=target[0] && count_snapshot[63:32]>=target[1]) begin
                        state<=CREDIT;credit_channel<=0;
                    end else begin step<=0;serial<=serial+1'b1;state<=SEND;end
                end
                READY: begin
                    if(faults || fault_now) state<=FAILED;
                    else if(outstanding) begin
                        target[0]<=observed[0];target[1]<=observed[1];
                        step<=0;serial<=serial+1'b1;state<=SEND;
                    end
                end
                CREDIT: begin
                    if(faults || fault_now) state<=FAILED;
                    else if(delivered[credit_channel]==target[credit_channel]) begin
                        if(!credit_channel) credit_channel<=1;
                        else state<=READY;
                    end
                end
                FAILED: begin end
                default: begin faults[15]<=1;state<=FAILED;end
            endcase
        end
    end
endmodule
`default_nettype wire
