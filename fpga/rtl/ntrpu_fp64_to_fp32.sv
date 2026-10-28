// Checked binary64 -> binary32, nearest/ties-to-even with gradual underflow.
// Reject magnitudes above FLT_MAX (even when ordinary rounding stays finite).
// Three globally stalled pipeline stages; one item/cycle when unblocked.
// Metadata follows data. Reset discards every in-flight item; the owner must
// discard its private destination. This block does not commit a vector or AXI.
module ntrpu_fp64_to_fp32 (
    input logic clk, rst_n,
    input logic s_valid,
    output logic s_ready,
    input logic [63:0] s_data,
    input logic [13:0] s_index,
    input logic s_last,
    input logic [31:0] s_sequence, s_epoch,
    output logic m_valid,
    input logic m_ready,
    output logic [31:0] m_data,
    output logic [1:0] m_status,
    output logic [13:0] m_index,
    output logic m_last,
    output logic [31:0] m_sequence, m_epoch
);
    localparam logic [1:0] OK=0, NONFINITE=2, OUT_OF_RANGE=3;
    logic advance, valid0, valid1;
    logic sign0, sign1, normal0, normal1;
    logic [1:0] status0, status1;
    logic [7:0] exponent0, exponent1;
    logic [5:0] shift0;
    logic [52:0] significand0, shifted, mask, remainder, halfway;
    logic [24:0] rounded, rounded1;
    logic [78:0] meta0, meta1;

    assign advance=!m_valid || m_ready;
    assign s_ready=rst_n && advance;
    always_comb begin
        shifted=significand0 >> shift0;
        mask=(53'd1 << shift0)-53'd1;
        remainder=significand0 & mask;
        halfway=53'd1 << (shift0-6'd1);
        rounded=shifted[24:0];
        if(remainder>halfway || (remainder==halfway && shifted[0]))
            rounded=rounded+25'd1;
    end
    always_ff @(posedge clk) begin
        if(!rst_n) begin
            valid0<=0;valid1<=0;m_valid<=0;
            sign0<=0;sign1<=0;normal0<=0;normal1<=0;
            status0<=OK;status1<=OK;exponent0<=0;exponent1<=0;
            shift0<=29;significand0<=0;rounded1<=0;meta0<=0;meta1<=0;
            m_data<=0;m_status<=OK;m_index<=0;m_last<=0;m_sequence<=0;m_epoch<=0;
        end else if(advance) begin
            valid0<=s_valid;
            valid1<=valid0;
            m_valid<=valid1;
            if(s_valid) begin
                meta0<={s_index,s_last,s_sequence,s_epoch};
                sign0<=s_data[63];
                normal0<=s_data[62:52]>=11'd897;
                exponent0<=8'(s_data[62:52]-11'd896);
                shift0<=s_data[62:52]>=11'd897 ? 6'd29 : 6'(11'd926-s_data[62:52]);
                significand0<={1'b1,s_data[51:0]};
                status0<=OK;
                // Every binary64 subnormal and every exponent below -150
                // rounds to signed zero. Clamp shift for the barrel shifter.
                if(s_data[62:52]<11'd873) begin significand0<=0;shift0<=29;end
                if(s_data[62:52]==11'h7ff) status0<=NONFINITE;
                else if(s_data[62:0]>63'h47efffffe0000000) status0<=OUT_OF_RANGE;
            end
            if(valid0) begin
                meta1<=meta0;sign1<=sign0;normal1<=normal0;
                exponent1<=exponent0;status1<=status0;rounded1<=rounded;
            end
            if(valid1) begin
                {m_index,m_last,m_sequence,m_epoch}<=meta1;
                m_status<=status1;
                if(status1!=OK) m_data<=0;
                else if(normal1)
                    m_data<={sign1,31'(({exponent1,23'b0}+32'(rounded1))-32'h00800000)};
                else m_data<={sign1,31'(rounded1)};
            end
        end
    end
endmodule
