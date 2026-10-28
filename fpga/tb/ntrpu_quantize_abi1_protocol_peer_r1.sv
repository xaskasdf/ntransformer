// Native protocol preflight ONLY. Emits oracle fixtures, performs no quantization.
// Do not include this module in generated-HLS simulation or synthesis.
`timescale 1ns/1ps
module ntrpu_quantize_abi1_r1(
    input wire ap_clk,ap_rst,ap_start,output reg ap_done=0,ap_idle=1,ap_ready=0,
    output reg [10:0] input_r_address0=0,output reg input_r_ce0=0,input wire [31:0] input_r_q0,
    output reg [8:0] output_r_address0=0,output reg output_r_ce0=0,output_r_we0=0,
    output reg [63:0] output_r_d0=0,input wire [31:0] elements,output reg [31:0] ap_return=0
);
    integer state=0,index=0,gap=0;reg bad=0;reg [63:0] reference_words[0:2047];string path;
    initial begin
        if(!$value$plusargs("FIXTURE=%s",path))$fatal(1,"Protocol peer fixture absent");
        $readmemh(path,reference_words);
    end
    always @(posedge ap_clk)begin
        if(ap_rst)begin state<=0;index<=0;gap<=0;bad<=0;ap_return<=0;ap_done<=0;ap_ready<=0;ap_idle<=1;
            input_r_ce0<=0;input_r_address0<=0;output_r_ce0<=0;output_r_we0<=0;output_r_address0<=0;output_r_d0<=0;end
        else begin
            ap_done<=0;ap_ready<=0;input_r_ce0<=0;output_r_ce0<=0;output_r_we0<=0;
            case(state)
                0:if(ap_start)begin
                    if(elements!=2048)$fatal(1,"Protocol peer extent");
                    ap_idle<=0;bad<=0;index<=0;ap_return<=0;state<=1;
                end
                1:begin input_r_ce0<=1;input_r_address0<=11'(index);state<=2;end
                2:state<=3;
                3:begin
                    if(input_r_q0[30:23]==8'hff)bad<=1;
                    if(index==2047)begin index<=0;gap<=20;state<=4;end
                    else begin index<=index+1;state<=1;end
                end
                4:if(gap==0)begin if(bad)begin ap_return<=2;state<=6;end else state<=5;end else gap<=gap-1;
                5:begin output_r_ce0<=1;output_r_we0<=1;output_r_address0<=9'(index);
                    output_r_d0<=reference_words[tb_ntrpu_attention_quantize_store_r1.sample_now*512+index];
                    if(index==511)state<=6;else index<=index+1;
                end
                6:begin ap_done<=1;ap_ready<=1;state<=7;end
                7:begin ap_idle<=1;state<=0;end
            endcase
        end
    end
endmodule
