// TEST ONLY: synchronous identity transport, deliberately no RoPE arithmetic.
// Replaces only the HLS port binder in native protocol tests. Never synthesize.
`default_nettype none
module ntrpu_rope_hls_ports_r1 (
    input wire clk,rst_n,hls_start,
    output wire hls_done,hls_ready,
    output wire [31:0] hls_return,
    input wire [31:0] hls_heads,hls_position,hls_coefficient_position,
    output wire [1:0] hls_input_ce,
    output wire [11:0] hls_input_address [0:1],
    input wire [31:0] hls_input_q [0:1],
    output wire hls_output_ce,hls_output_we,
    output wire [11:0] hls_output_address,
    output wire [31:0] hls_output_d
);
    typedef enum logic [2:0] {IDLE,REQUEST,WAIT_RAM,WRITE,DONE} state_t;
    state_t state;
    logic [11:0] index;
    assign hls_ready=state==IDLE || state==DONE;
    assign hls_done=state==DONE;
    assign hls_return=0;
    assign hls_input_ce=state==REQUEST ? 2'b01:2'b00;
    assign hls_input_address[0]=index;
    assign hls_input_address[1]=0;
    assign hls_output_ce=state==WRITE;
    assign hls_output_we=state==WRITE;
    assign hls_output_address=index;
    assign hls_output_d=hls_input_q[0];
    always_ff @(posedge clk)begin
        if(!rst_n)begin state<=IDLE;index<=0;end
        else case(state)
            IDLE:if(hls_start)begin
                if(hls_heads!=40 || hls_position!=hls_coefficient_position)
                    $fatal(1,"PROTOCOL_MODEL_BAD_COMMAND");
                index<=0;state<=REQUEST;
            end
            REQUEST:state<=WAIT_RAM;
            WAIT_RAM:state<=WRITE;
            WRITE:if(index==2559)state<=DONE;else begin index<=index+1;state<=REQUEST;end
            DONE:state<=IDLE;
            default:$fatal(1,"PROTOCOL_MODEL_BAD_STATE");
        endcase
    end
endmodule
`default_nettype wire
