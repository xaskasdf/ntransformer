`timescale 1ns/1ps
`define RX_LOG 1'b0
`define PCI_EXP_CFG_READ0 7'b0000100
`define PCI_EXP_CFG_WRITE0 7'b1000100
`define PCI_EXP_COMPLETION_WO_DATA 7'b0001010
`define PCI_EXP_COMPLETION_DATA 7'b1001010
`define TX_TASKS board.RP.tx_usrapp
module ntrpu_rx_com;
    reg [7:0] frame_store_tx[0:5119],frame_store_rx[0:5119];
    integer frame_store_tx_idx=0,frame_store_rx_idx=0;
    integer cq_packets=0,rc_packets=0,compared=0;
    `include "rx_common.svh"
    task TSK_PARSE_FRAME(input bit txrx);
        integer f,i;
        bit is_rc;
        begin
            if(txrx) $fatal(1,"UNIT_RX_DIRECTION");
            is_rc=frame_store_rx[0]===8'h4a;
            if(frame_store_rx_idx!=(is_rc?16:272)) $fatal(1,"UNIT_RX_PACKET_SIZE");
            for(i=0;i<frame_store_rx_idx;i++)
                if(frame_store_rx[i]!==board.expected(is_rc,i)) $fatal(1,"UNIT_RX_BYTE index=%0d",i);
            compared+=frame_store_rx_idx;
            if(is_rc) begin
                rc_packets++;
                f=$fopen("callback.log","a");if(!f)$fatal(1,"UNIT_RX_FILE");
                TSK_3DW(2,10,0,0,0,0,1,1,16,f,0);
                $fclose(f);
                if(board.RP.tx_usrapp.read_data!==64'h12345678)$fatal(1,"UNIT_RX_CALLBACK");
            end else cq_packets++;
            if(board.mode!=7)frame_store_rx_idx=0;
        end
    endtask
endmodule
module ntrpu_rx_builders;
    localparam C_DATA_WIDTH=512,KEEP_WIDTH=16;
    `include "rx_builders.svh"
endmodule
module ntrpu_rx_tx;
    integer callbacks=0;
    reg [63:0] read_data=0;
    task TSK_SET_READ_DATA(input [3:0] be,input [63:0] value);
        begin callbacks++;read_data=value;end
    endtask
endmodule
module ntrpu_rx_rp;
    ntrpu_rx_com com_usrapp();
    ntrpu_rx_builders rx_usrapp();
    ntrpu_rx_tx tx_usrapp();
endmodule
module board;
    ntrpu_rx_rp RP();
    integer mode=0,phase=0,split_at,i,j;
    reg [63:0] word_data;
    localparam [63:0] CQ_QW1=(64'h1234<<16)|(64'h56<<32)|(64'd1<<11)|64'd64;
    localparam [63:0] RC_QW0=(64'h4567<<48)|(64'd1<<32)|(64'd4<<16);
    localparam [63:0] RC_QW1=(64'h12345678<<32)|(64'h1234<<8)|64'h77;
    function automatic [7:0] expected(input bit is_rc,input integer off);
        reg [127:0] header;
        begin
            header=is_rc ? 128'h4a000001_12340004_45677700_12345678:
                           128'h60000040_123456ff_00000004_00000fc0;
            if(off<16) return header[127-off*8-:8];
            return 8'(off*13+phase*7+19);
        end
    endfunction
    task legacy_reproduction;
        begin
            RP.rx_usrapp.TSK_LEGACY_CQ(64'h400000fc0,8'hff,CQ_QW1);
            for(int k=0;k<6;k++) RP.com_usrapp.TSK_READ_DATA(0,0,64'h1122334455667788,0);
            RP.rx_usrapp.TSK_LEGACY_RC(RC_QW0,RC_QW1,16'hf,1);
            if(RP.com_usrapp.frame_store_rx_idx!=80 || RP.com_usrapp.frame_store_rx[0]!==8'h60)
                $fatal(1,"UNIT_RX_LEGACY_NOT_REPRODUCED");
            if(mode==6)RP.com_usrapp.TSK_PARSE_FRAME(0);
            RP.com_usrapp.frame_store_rx_idx=0;
            $display("NTRPU_RX_LOG_LEGACY_REPRO partial_cq_bytes=64 rc_bytes=16 mixed_bytes=80");
        end
    endtask
    initial begin
        if(!$value$plusargs("MODE=%d",mode))mode=0;
        case(mode)
            1:RP.com_usrapp.TSK_NTRPU_RX_DATA(1'bx,0,0,0,0);
            2:begin RP.com_usrapp.ntrpu_log_cq_idx=5120;RP.com_usrapp.TSK_NTRPU_RX_DATA(0,0,0,0,0);end
            3:RP.com_usrapp.TSK_NTRPU_RX_PARSE(0,0);
            4:RP.com_usrapp.TSK_NTRPU_RX_PARSE(0,1);
            8:RP.com_usrapp.TSK_NTRPU_RX_DATA(0,1,0,0,1'bx);
            default:begin
                legacy_reproduction();
                for(phase=0;phase<8;phase++) begin
                    #1;RP.rx_usrapp.TSK_BUILD_CQ_TO_PCIE_PKT(64'h400000fc0,8'hff,CQ_QW1);
                    // RC is delivered between CQ beats, with varied offsets.
                    split_at=48+(phase%4)*64;
                    for(i=0;i<256;i+=8)begin
                        if(i==split_at)begin
                            #1;RP.rx_usrapp.TSK_BUILD_RC_TO_PCIE_PKT(RC_QW0,RC_QW1,16'hf,1);
                            RP.com_usrapp.TSK_NTRPU_RX_PARSE(1,0);
                            if(RP.com_usrapp.ntrpu_log_cq_idx!=16+i)$fatal(1,"UNIT_RX_PARTIAL_CQ");
                        end
                        word_data=0;
                        for(j=0;j<8;j++)word_data[63-j*8-:8]=expected(0,16+i+j);
                        if(mode==5 && i==64)word_data[0]=~word_data[0];
                        RP.com_usrapp.TSK_NTRPU_RX_DATA(0,i==248,0,word_data,0);
                        if((i+8)%64==48)#1;
                    end
                    RP.com_usrapp.TSK_NTRPU_RX_PARSE(0,0);
                    if(RP.com_usrapp.ntrpu_log_rc_idx || RP.com_usrapp.ntrpu_log_cq_idx)$fatal(1,"UNIT_RX_DRAIN");
                end
                if(RP.com_usrapp.cq_packets!=8 || RP.com_usrapp.rc_packets!=8 ||
                   RP.com_usrapp.compared!=2304 || RP.tx_usrapp.callbacks!=8)$fatal(1,"UNIT_RX_TOTAL");
                $display("NTRPU_RX_LOG_PASS cq=8 rc=8 bytes=2304 callbacks=8 legacy_mixture_reproduced=1");
                $finish;
            end
        endcase
        $fatal(1,"UNIT_RX_NEGATIVE_ESCAPED");
    end
    initial begin #1000;$fatal(1,"UNIT_RX_WATCHDOG");end
endmodule
