// Bounded test-only root-port producer: one aligned 64-byte MemWr64 payload.
// PG213 512-bit RQ: descriptor+48 bytes, then16 bytes; no straddling.
task automatic NTRPU_TX_FRAME_512;
    input [9:0] tag_;
    input [63:0] addr_;
    input [3:0] last_be_;
    input [511:0] payload_;
    reg [127:0] descriptor;
    reg [136:0] sideband;
    reg [639:0] packet_image;
    integer i;
    begin
        if (C_DATA_WIDTH != 512 || AXISTEN_IF_RQ_ALIGNMENT_MODE != "FALSE" ||
            ATTR_AXISTEN_IF_ENABLE_CLIENT_TAG || tag_[9:8] != 0 || addr_[5:0] != 0 ||
            (last_be_ != 4'hf && last_be_ != 4'h7) || user_lnk_up_n)
            $fatal(1,"NTRPU_FRAME512_UNSUPPORTED_PROFILE");
        descriptor=0;
        descriptor[63:2]=addr_[63:2];
        descriptor[74:64]=11'd16;
        descriptor[78:75]=4'd1;
        descriptor[95:80]=RP_BUS_DEV_FNS;
        descriptor[103:96]=tag_[7:0];
        descriptor[119:104]=EP_BUS_DEV_FNS;
        descriptor[120]=(PF0_DEV_CAP2_10B_TAG_COMPLETER_SUPPORTED=="TRUE") ? 1'b0 : 1'b1;
        sideband=0;
        sideband[3:0]=4'hf;
        sideband[11:8]=last_be_;
        sideband[21:20]=2'b01;
        sideband[66:61]=6'b101010;
        sideband[72:67]=6'b101010;
        packet_image=0;
        packet_image[639:608]=32'h60000010;
        packet_image[607:576]={RP_BUS_DEV_FNS,tag_[7:0],last_be_,4'hf};
        packet_image[575:512]=addr_;
        for(i=0;i<64;i=i+1) packet_image[511-i*8-:8]=payload_[i*8+:8];
        TSK_TX_SYNCHRONIZE(0,0,0,`SYNC_RQ_RDY);
        s_axis_rq_tdata <= #(Tcq) {payload_[383:0],descriptor};
        s_axis_rq_tuser_wo_parity <= #(Tcq) sideband;
        s_axis_rq_tkeep <= #(Tcq) 16'hffff;
        s_axis_rq_tlast <= #(Tcq) 1'b0;
        s_axis_rq_tvalid <= #(Tcq) 1'b1;
        TSK_TX_SYNCHRONIZE(0,0,0,`SYNC_RQ_RDY);
        sideband[21:20]=0;
        sideband[27:26]=2'b01;
        sideband[31:28]=4'd3;
        s_axis_rq_tdata <= #(Tcq) {384'b0,payload_[511:384]};
        s_axis_rq_tuser_wo_parity <= #(Tcq) sideband;
        s_axis_rq_tkeep <= #(Tcq) 16'h000f;
        s_axis_rq_tlast <= #(Tcq) 1'b1;
        TSK_TX_SYNCHRONIZE(0,0,0,`SYNC_RQ_RDY);
        // Atomic complete packet logging after both actual handshakes.
        board.RP.com_usrapp.TSK_NTRPU_LOG_FRAME512(packet_image);
        s_axis_rq_tvalid <= #(Tcq) 1'b0;
        s_axis_rq_tlast <= #(Tcq) 1'b0;
        s_axis_rq_tkeep <= #(Tcq) 16'b0;
        s_axis_rq_tuser_wo_parity <= #(Tcq) 137'b0;
        s_axis_rq_tdata <= #(Tcq) 512'b0;
    end
endtask
