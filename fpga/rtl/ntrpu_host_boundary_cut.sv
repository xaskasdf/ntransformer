`timescale 1ns / 1ps
// Source-domain boundary between resettable AMD CDCs and the retained core.
// Pack channels in order: data AW,W,B; control AW,W,B,AR,R.
// Loss is permanent for this configuration. No warm rearm/reset of inventory.
// The caller MUST feed invalid to the retained ownership invalidation path,
// wait for that invalidation acknowledgement, then reset both CDC ends and
// every facing interface with overlapping reset. This block is not a detector.
module ntrpu_host_boundary_cut (
    input wire clk,cold_rst_n,boundary_rst_n,loss,
    input wire [7:0] in_valid,out_ready,
    output wire [7:0] in_ready,out_valid,
    input wire [81:0] data_aw_in,
    output wire [81:0] data_aw_out,
    input wire [576:0] data_w_in,
    output wire [576:0] data_w_out,
    input wire [5:0] data_b_in,
    output wire [5:0] data_b_out,
    input wire [31:0] ctl_aw_in,ctl_ar_in,
    output wire [31:0] ctl_aw_out,ctl_ar_out,
    input wire [35:0] ctl_w_in,
    output wire [35:0] ctl_w_out,
    input wire [1:0] ctl_b_in,
    output wire [1:0] ctl_b_out,
    input wire [33:0] ctl_r_in,
    output wire [33:0] ctl_r_out,
    output wire invalid,
    output wire [7:0] fenced,
    output wire [15:0] pending,fault,
    output wire [511:0] accepted,delivered,abandoned
);
    reg dead;
    assign invalid=loss || dead || (|fault);
    always @(posedge clk) if(!cold_rst_n) dead<=0;else if(invalid) dead<=1;
    for(genvar c=0;c<8;c=c+1) begin:channels
        localparam integer WIDTH=c==0 ? 82 : c==1 ? 577 : c==2 ? 6 :
                                c==3 || c==6 ? 32 : c==4 ? 36 : c==5 ? 2 : 34;
        wire [WIDTH-1:0] input_data,output_data;
        if(c==0) begin assign input_data=data_aw_in;assign data_aw_out=output_data;end
        if(c==1) begin assign input_data=data_w_in;assign data_w_out=output_data;end
        if(c==2) begin assign input_data=data_b_in;assign data_b_out=output_data;end
        if(c==3) begin assign input_data=ctl_aw_in;assign ctl_aw_out=output_data;end
        if(c==4) begin assign input_data=ctl_w_in;assign ctl_w_out=output_data;end
        if(c==5) begin assign input_data=ctl_b_in;assign ctl_b_out=output_data;end
        if(c==6) begin assign input_data=ctl_ar_in;assign ctl_ar_out=output_data;end
        if(c==7) begin assign input_data=ctl_r_in;assign ctl_r_out=output_data;end
        ntrpu_retained_channel #(.WIDTH(WIDTH),.TO_BOUNDARY(c==2 || c==5 || c==7)) channel (
            .clk(clk),.cold_rst_n(cold_rst_n),.boundary_rst_n(boundary_rst_n),.loss(invalid),
            .s_valid(in_valid[c]),.s_ready(in_ready[c]),.s_data(input_data),
            .m_valid(out_valid[c]),.m_ready(out_ready[c]),.m_data(output_data),
            .invalid(),.fenced(fenced[c]),.pending(pending[c*2+:2]),.fault(fault[c*2+:2]),
            .accepted(accepted[c*64+:64]),.delivered(delivered[c*64+:64]),.abandoned(abandoned[c*64+:64]));
    end
endmodule
