`timescale 1ns/1ps
// Experimental ordered AXI3 fragment tracker. The frame owns all payload space.
// Credits cover reserved/offered/accepted descriptors until final validation.
// Only a coordinated cold reset may discard quarantined obligations.
module ntrpu_hbm_read_tracker #(parameter integer MAX_PENDING=4) (
    input wire clk,rst_n,
    input wire p_arvalid,
    output wire p_arready,
    input wire [33:0] p_araddr,
    input wire [3:0] p_arlen,p_arid,
    input wire [2:0] p_arsize,
    input wire [1:0] p_arburst,
    output wire p_rvalid,
    input wire p_rready,
    output wire [255:0] validated_data,
    output wire [5:0] validated_rid,
    output wire [1:0] validated_resp,
    output wire validated_last,
    output wire h_arvalid,
    input wire h_arready,
    output reg [33:0] h_araddr,
    output reg [3:0] h_arlen,
    output wire [5:0] h_arid,
    output wire [2:0] h_arsize,
    output wire [1:0] h_arburst,
    input wire h_rvalid,
    output wire h_rready,
    input wire [255:0] h_rdata,
    input wire [5:0] h_rid,
    input wire [1:0] h_rresp,
    input wire h_rlast,
    output wire pending,
    output reg fault
);
    reg [3:0] reserved_count,accepted_count;
    reg [7:0] capture_remaining;
    reg [2:0] write_ptr,validate_ptr;
    reg [3:0] lengths[0:MAX_PENDING-1];
    reg [3:0] validate_index;
    reg offer_valid;
    wire reserve_fire=p_arvalid&&p_arready;
    wire ar_fire=h_arvalid&&h_arready;
    wire capture_fire=h_rvalid&&h_rready;
    wire q_valid,q_ready,q_space,q_empty,validated_space,validated_empty;
    wire [255:0] q_data;
    wire [5:0] q_rid;
    wire [1:0] q_resp;
    wire q_last;
    wire can_validate=accepted_count!=0;
    wire terminal=validate_index==lengths[validate_ptr];
    wire bad_response=q_valid&&(!can_validate||q_rid!=0||q_resp==2'b01||q_last!=terminal);
    wire validate_fire=q_valid&&q_ready;
    wire release_credit=validate_fire&&terminal;
    wire capture_enable=capture_remaining!=0&&!fault;
    // A zero-latency slave may offer R with the first AR handshake, but READY
    // waits for registered credit. An unsolicited offer otherwise quarantines.
    wire unexpected_offer=h_rvalid&&capture_remaining==0&&!ar_fire;

    assign p_arready=rst_n&&!fault&&!offer_valid&&reserved_count<MAX_PENDING;
    assign h_arvalid=offer_valid;
    assign h_arid=0;
    assign h_arsize=5;
    assign h_arburst=1;
    assign h_rready=q_space&&capture_enable;
    assign q_ready=validated_space&&can_validate&&!fault&&!bad_response;
    assign pending=reserved_count!=0||accepted_count!=0||capture_remaining!=0||
                   offer_valid||!q_empty||!validated_empty;

    ntrpu_hbm_read_response_queue #(.WIDTH(265)) response_cut (
        .clk(clk),.rst_n(rst_n),.s_valid(h_rvalid&&capture_enable),.s_ready(q_space),
        .s_data({h_rid,h_rresp,h_rlast,h_rdata}),.m_valid(q_valid),.m_ready(q_ready),
        .m_data({q_rid,q_resp,q_last,q_data}),.empty(q_empty));
    ntrpu_hbm_read_response_queue #(.WIDTH(265)) validated_cut (
        .clk(clk),.rst_n(rst_n),.s_valid(q_valid&&can_validate&&!fault&&!bad_response),
        .s_ready(validated_space),.s_data({q_rid,q_resp,q_last,q_data}),
        .m_valid(p_rvalid),.m_ready(p_rready),
        .m_data({validated_rid,validated_resp,validated_last,validated_data}),.empty(validated_empty));

    always @(posedge clk)begin
        if(!rst_n)begin
            reserved_count<=0;accepted_count<=0;capture_remaining<=0;
            write_ptr<=0;validate_ptr<=0;validate_index<=0;
            offer_valid<=0;h_araddr<=0;h_arlen<=0;fault<=0;
        end else begin
            if(bad_response||unexpected_offer)fault<=1;
            case({reserve_fire,release_credit})
                2'b10:reserved_count<=reserved_count+1'b1;
                2'b01:reserved_count<=reserved_count-1'b1;
                default:begin end
            endcase
            case({ar_fire,release_credit})
                2'b10:accepted_count<=accepted_count+1'b1;
                2'b01:accepted_count<=accepted_count-1'b1;
                default:begin end
            endcase
            case({ar_fire,capture_fire})
                2'b10:capture_remaining<=capture_remaining+{4'b0,h_arlen}+8'd1;
                2'b01:capture_remaining<=capture_remaining-1'b1;
                2'b11:capture_remaining<=capture_remaining+{4'b0,h_arlen};
                default:begin end
            endcase
            if(reserve_fire)begin
                lengths[write_ptr]<=p_arlen;
                write_ptr<=(write_ptr==MAX_PENDING-1)?0:write_ptr+1'b1;
                offer_valid<=1;h_araddr<=p_araddr;h_arlen<=p_arlen;
                if(p_arid!=0||p_arsize!=5||p_arburst!=1)fault<=1;
            end
            if(ar_fire)offer_valid<=0;
            if(validate_fire)begin
                if(terminal)begin
                    validate_index<=0;
                    validate_ptr<=(validate_ptr==MAX_PENDING-1)?0:validate_ptr+1'b1;
                end else validate_index<=validate_index+1'b1;
            end
        end
    end
    initial begin
        if(MAX_PENDING!=1&&MAX_PENDING!=2&&MAX_PENDING!=4&&MAX_PENDING!=8)
            $error("MAX_PENDING must be 1, 2, 4 or 8");
    end
endmodule
