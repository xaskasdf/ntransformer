`timescale 1ns/1ps
`define TX_LOG 1'b1
`define SYNC_CC_RDY 1'b1
`define SYNC_RQ_RDY 1'b0
`define PCI_EXP_CFG_READ0 7'b0000100
`define PCI_EXP_CFG_WRITE0 7'b1000100
`define PCI_EXP_COMPLETION_WO_DATA 7'b0001010
`define PCI_EXP_COMPLETION_DATA 7'b1001010
`define TX_TASKS board.RP.tx_usrapp
module ntrpu_concurrent_com;
    reg [7:0] frame_store_tx [0:5119], frame_store_rx [0:5119];
    integer frame_store_tx_idx=0,frame_store_rx_idx=0;
    integer rq_packets=0,cc_packets=0,compared_bytes=0;
    `include "concurrent_logger.svh"
    // Byte observer for this unit test only; integration uses the AMD parser.
    task TSK_PARSE_FRAME(input bit txrx);
        integer i;
        begin
            if (!txrx) $fatal(1,"UNIT_DIRECTION");
            if (frame_store_tx[0]===8'h4a) begin
                if(frame_store_tx_idx!=268) $fatal(1,"UNIT_CC_SIZE");
                for(i=0;i<268;i++)
                    if(frame_store_tx[i]!==board.cc_byte(i)) $fatal(1,"UNIT_CC_DATA index=%0d",i);
                cc_packets++;
            end else begin
                if(frame_store_tx_idx!=board.rq_size()) $fatal(1,"UNIT_RQ_SIZE");
                for(i=0;i<frame_store_tx_idx;i++)
                    if(frame_store_tx[i]!==board.rq_byte(i)) $fatal(1,"UNIT_RQ_DATA index=%0d",i);
                rq_packets++;
            end
            compared_bytes+=frame_store_tx_idx;
            frame_store_tx_idx=0;
        end
    endtask
endmodule
module ntrpu_concurrent_tx;
    reg user_clk=0,user_lnk_up_n=0;
    reg s_axis_cc_tready=0,s_axis_rq_tready=0;
    reg [511:0] pcie_tlp_data=0,ntrpu_cc_tlp_data=0,ntrpu_rq_tlp_data=0;
    reg [3:0] pcie_tlp_rem=0,ntrpu_cc_tlp_rem=0,ntrpu_rq_tlp_rem=0;
    integer read_callbacks=0;
    reg [63:0] read_data=64'hdeadcafe;
    task TSK_SET_READ_DATA(input [3:0] be,input [63:0] value);
        begin read_callbacks++;read_data=value;end
    endtask
    always #2 user_clk=~user_clk;
    `include "concurrent_sync.svh"
endmodule
module ntrpu_concurrent_rp;
    ntrpu_concurrent_com com_usrapp();
    ntrpu_concurrent_tx tx_usrapp();
endmodule
module board;
    ntrpu_concurrent_rp RP();
    integer mode,phase=0,beat,i;
    reg [511:0] data,rq;
    function automatic [7:0] cc_byte(input integer index);
        case(index)
            0:cc_byte=8'h4a; 1:cc_byte=0; 2:cc_byte=0; 3:cc_byte=8'h40;
            default:cc_byte=8'(index*13+19+phase*7);
        endcase
    endfunction
    function automatic integer rq_size();rq_size=(phase%4==0 || phase%4==2)?12:16;endfunction
    function automatic [7:0] rq_byte(input integer index);
        case(index)
            0:case(phase%4) 0:rq_byte=8'h04;1:rq_byte=8'h44;2:rq_byte=0;3:rq_byte=8'h20;endcase
            1,2:rq_byte=0;3:rq_byte=1;
            default:rq_byte=8'(index*11+phase*3+5);
        endcase
    endfunction
    task callback_cases;
        integer f;
        begin
            f=$fopen("callback.log","w");if(!f)$fatal(1,"UNIT_CALLBACK_FILE");
            for(integer j=0;j<16;j++)begin
                RP.com_usrapp.frame_store_tx[j]=0;RP.com_usrapp.frame_store_rx[j]=0;
            end
            RP.com_usrapp.frame_store_rx[12]='h12;RP.com_usrapp.frame_store_rx[13]='h34;
            RP.com_usrapp.frame_store_rx[14]=mode==10?0:'h56;RP.com_usrapp.frame_store_rx[15]='h78;
            RP.tx_usrapp.read_callbacks=0;RP.tx_usrapp.read_data='hdeadcafe;
            RP.com_usrapp.TSK_UNGUARDED_3DW(2,10,0,0,0,0,1,1,16,f,1);
            if(mode==9 && RP.tx_usrapp.read_callbacks!=0)$fatal(1,"UNIT_CALLBACK_FALSE_WAKE");
            if(RP.tx_usrapp.read_callbacks!=1 || RP.tx_usrapp.read_data!==0)$fatal(1,"UNIT_CALLBACK_REPRO");
            RP.tx_usrapp.read_callbacks=0;RP.tx_usrapp.read_data='hdeadcafe;
            RP.com_usrapp.TSK_3DW(2,10,0,0,0,0,1,1,16,f,1);
            if(RP.tx_usrapp.read_callbacks!=0 || RP.tx_usrapp.read_data!=='hdeadcafe)$fatal(1,"UNIT_CALLBACK_FALSE_WAKE");
            RP.com_usrapp.TSK_3DW(2,10,0,0,0,0,1,1,16,f,0);
            if(RP.tx_usrapp.read_callbacks!=1 || RP.tx_usrapp.read_data!=='h12345678)$fatal(1,"UNIT_CALLBACK_RX_VALUE");
            RP.com_usrapp.TSK_3DW(2,10,0,0,0,0,1,1,16,f,1);
            if(RP.tx_usrapp.read_callbacks!=1 || RP.tx_usrapp.read_data!=='h12345678)$fatal(1,"UNIT_CALLBACK_RX_OVERWRITTEN");
            $fclose(f);
            $display("NTRPU_CONCURRENT_CALLBACK_PASS cases=4 legacy_false_wakeup=1 received_data=12345678");
        end
    endtask
    initial begin
        if(!$value$plusargs("MODE=%d",mode))mode=0;
        data=0;
        case(mode)
            1:RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(0,1,1,0,0);
            2:begin
                RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(1,0,1,0,0);
                RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(1,1,1,0,0);
            end
            3:RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(1,1,1'bx,0,0);
            4:begin
                RP.com_usrapp.ntrpu_log_cc_idx=5120;
                RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(0,1,1,0,0);
            end
            5:RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(1,1,1,0,4'bx001);
            6:begin data[511]=1;RP.com_usrapp.TSK_NTRPU_LOG_CHANNEL(1,1,0,data,0);end
            9,10:callback_cases();
            default:begin
                for(phase=0;phase<8;phase++)begin
                    @(negedge RP.tx_usrapp.user_clk);
                    RP.tx_usrapp.s_axis_cc_tready=0;RP.tx_usrapp.s_axis_rq_tready=0;
                    data=0;rq=0;
                    for(i=0;i<64;i++)data[511-8*i -:8]=cc_byte(i);
                    for(i=0;i<rq_size();i++)rq[255-8*i -:8]=rq_byte(i);
                    RP.tx_usrapp.ntrpu_rq_tlp_data=rq;
                    RP.tx_usrapp.ntrpu_rq_tlp_rem=13;
                    RP.tx_usrapp.ntrpu_cc_tlp_data=data;
                    RP.tx_usrapp.ntrpu_cc_tlp_rem=0;
                    // Reproduce the original shared image being overwritten
                    // by the other producer while RQ is waiting for ready.
                    RP.tx_usrapp.pcie_tlp_data=rq;
                    RP.tx_usrapp.pcie_tlp_data=data;
                    fork
                        begin
                            if(mode==8)RP.tx_usrapp.TSK_SHARED_TX_SYNCHRONIZE(1,1,1,`SYNC_RQ_RDY);
                            else RP.tx_usrapp.TSK_TX_SYNCHRONIZE(1,1,1,`SYNC_RQ_RDY);
                        end
                        begin
                            if(mode==8)RP.tx_usrapp.TSK_SHARED_TX_SYNCHRONIZE(1,1,0,`SYNC_CC_RDY);
                            else RP.tx_usrapp.TSK_TX_SYNCHRONIZE(1,1,0,`SYNC_CC_RDY);
                        end
                        begin
                            repeat(1+phase%3)@(negedge RP.tx_usrapp.user_clk);
                            RP.tx_usrapp.s_axis_cc_tready=1;
                            repeat(phase%2?2:0)@(negedge RP.tx_usrapp.user_clk);
                            RP.tx_usrapp.s_axis_rq_tready=1;
                        end
                    join
                    if(RP.com_usrapp.ntrpu_log_cc_idx!=64 || RP.com_usrapp.rq_packets!=phase+1)
                        $fatal(1,"UNIT_PARTIAL_PACKET");
                    for(beat=1;beat<5;beat++)begin
                        @(negedge RP.tx_usrapp.user_clk);data=0;
                        for(i=0;i<64;i++)data[511-8*i -:8]=cc_byte(beat*64+i);
                        if(mode==7 && beat==2)data[511]=~data[511];
                        RP.tx_usrapp.ntrpu_cc_tlp_data=data;
                        RP.tx_usrapp.ntrpu_cc_tlp_rem=beat==4?13:0;
                        RP.tx_usrapp.TSK_TX_SYNCHRONIZE(0,1,beat==4,`SYNC_CC_RDY);
                    end
                    if(RP.com_usrapp.ntrpu_log_cc_idx!=0 || RP.com_usrapp.ntrpu_log_rq_idx!=0 ||
                       RP.com_usrapp.cc_packets!=phase+1)$fatal(1,"UNIT_DRAIN");
                end
                if(RP.com_usrapp.compared_bytes!=2256)$fatal(1,"UNIT_TOTAL_BYTES");
                callback_cases();
                $display("NTRPU_CONCURRENT_LOG_PASS rq=8 cc=8 bytes=2256 phases=8");
                $finish;
            end
        endcase
        $fatal(1,"UNIT_NEGATIVE_ESCAPED mode=%0d",mode);
    end
    initial begin #10000;$fatal(1,"UNIT_WATCHDOG");end
endmodule
