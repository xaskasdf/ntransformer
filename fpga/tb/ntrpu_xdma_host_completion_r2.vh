// Bounded public Root Port stimulus for the native stop/inventory experiment.
// Included inside AMD's public pci_exp_usrapp_tx module in a private copy.
// No protected endpoint state, no reset shortcuts and no manufactured status.
task TSK_TX_COMPLETION_DATA;
    input [15:0] req_id_;
    input [9:0] tag_;
    input [2:0] tc_;
    input [10:0] len_;
    input [11:0] byte_count_;
    input [10:0] lower_addr_;
    input [RP_BAR_SIZE:0] ram_ptr;
    input [2:0] comp_status_;
    input ep_;
    input [2:0] attr_;
    reg [511:0] beat;
    reg [82:0] sideband;
    reg [15:0] keep;
    integer offset, payload, header_bytes, words, j;
    reg final_beat;
    begin
        if (C_DATA_WIDTH != 512 || AXISTEN_IF_CC_ALIGNMENT_MODE != "FALSE" ||
            comp_status_ != 0 || ep_ != 0 || attr_ != 3'b010 || tc_ != 0 ||
            !((lower_addr_ == 11'h100 && len_ == 8 && byte_count_ == 32) ||
              (lower_addr_ == 11'h400 && len_ == 16 && byte_count_ == 64)))
            $fatal(1, "NTRPU_HOST_CPL_UNSUPPORTED width=%0d align=%s addr=%h len=%0d bytes=%0d attr=%h tc=%h status=%h poison=%b", C_DATA_WIDTH, AXISTEN_IF_CC_ALIGNMENT_MODE, lower_addr_, len_, byte_count_, attr_, tc_, comp_status_, ep_);
        $display("NTRPU_HOST_CPL_BEGIN t=%0t address=%h bytes=%0d tag=%h requester=%h ram_ptr=%h attr=%h tc=%h",
                 $time, lower_addr_, byte_count_, tag_, req_id_, ram_ptr, attr_, tc_);
        offset = 0;
        while (offset < byte_count_) begin
            header_bytes = offset == 0 ? 12 : 0;
            payload = byte_count_ - offset;
            if (payload > 64-header_bytes) payload = 64-header_bytes;
            words = (header_bytes+payload)/4;
            final_beat = offset+payload == byte_count_;
            beat = 0;
            if (offset == 0) begin
                beat[6:0] = lower_addr_[6:0];
                beat[27:16] = byte_count_;
                beat[31] = tag_[8];
                beat[42:32] = len_;
                beat[45:43] = comp_status_;
                beat[46] = ep_;
                beat[47] = tag_[9];
                beat[63:48] = req_id_;
                beat[71:64] = tag_[7:0];
                beat[87:72] = RP_BUS_DEV_FNS;
                beat[88] = 1'b1;
                beat[91:89] = tc_;
                beat[94:92] = attr_;
            end
            for (j=0; j<payload; j=j+1)
                beat[(header_bytes+j)*8 +: 8] = DATA_STORE[lower_addr_+offset+j];
            keep = 16'hffff >> (16-words);
            sideband = 0;
            sideband[1:0] = offset == 0 ? 2'b01 : 2'b00;
            sideband[7:6] = final_beat ? 2'b01 : 2'b00;
            sideband[11:8] = final_beat ? words-1 : 0;
            @(negedge user_clk);
            s_axis_cc_tdata = beat;
            s_axis_cc_tkeep = keep;
            s_axis_cc_tuser = sideband;
            s_axis_cc_tlast = final_beat;
            s_axis_cc_tvalid = 1;
            @(posedge user_clk);
            while (s_axis_cc_tready !== 1'b1) @(posedge user_clk);
            $display("NTRPU_HOST_CPL_BEAT t=%0t offset=%0d keep=%h last=%b user=%h data=%h",
                     $time, offset, keep, final_beat, sideband, beat);
            offset = offset+payload;
        end
        @(negedge user_clk);
        s_axis_cc_tvalid = 0;
        s_axis_cc_tlast = 0;
        s_axis_cc_tdata = 0;
        s_axis_cc_tkeep = 0;
        s_axis_cc_tuser = 0;
        $display("NTRPU_HOST_CPL_END t=%0t bytes=%0d tag=%h", $time, byte_count_, tag_);
    end
endtask
