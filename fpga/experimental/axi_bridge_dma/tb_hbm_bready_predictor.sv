`timescale 1ns/1ps
module tb_hbm_bready_predictor;
    reg offer,response,lane,aw_pending,w_pending;
    reg [1:0] awready,wready,bvalid;
    wire [1:0] next_ready;
    reg [1:0] expected;
    integer checks=0,next_state;
    ntrpu_hbm_bready_predictor dut(.*);
    initial begin
        for(integer state=0;state<8;state++)
        for(integer select=0;select<2;select++)
        for(integer pending=0;pending<4;pending++)
        for(integer a=0;a<4;a++)for(integer w=0;w<4;w++)for(integer b=0;b<4;b++)begin
            offer=state==2;response=state==3;lane=select;
            aw_pending=pending[0];w_pending=pending[1];awready=a;wready=w;bvalid=b;
            next_state=state;
            case(state)
                2:if((!pending[0]||a[select])&&(!pending[1]||w[select]))next_state=3;
                3:if(b[select])next_state=0;
                default:begin end
            endcase
            expected=0;if(next_state==3)expected[select]=1;
            #0.001;checks++;
            if(next_ready!==expected)$fatal(1,"BREADY_PREDICATE predictor truth table state=%0d lane=%0d pending=%0d a=%0d w=%0d b=%0d",state,select,pending,a,w,b);
        end
        $display("BREADY_PREDICATE PASS checks=%0d",checks);$finish;
    end
endmodule
