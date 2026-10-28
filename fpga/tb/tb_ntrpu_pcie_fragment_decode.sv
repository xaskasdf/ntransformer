`timescale 1ns / 1ps
module tb_ntrpu_pcie_fragment_decode;
    reg lease_valid,awlock,wlast;
    reg [63:0] lease_window,lease_epoch,lease_tag,awaddr,wstrb;
    reg [7:0] awlen;
    reg [2:0] awsize;
    reg [1:0] awburst;
    reg [511:0] wdata;
    wire accept;
    wire [6:0] faults;
    wire [63:0] byte_valid;
    wire [511:0] byte_data;
    integer cases=0,allowed=0,rejected=0;
    ntrpu_pcie_fragment_decode dut (.*);
    task baseline;
        begin
            lease_valid=1;lease_window=64'h100080000;
            lease_epoch=64'h100000001;lease_tag=64'h8000000100200300;
            awaddr=lease_window;awlen=0;awsize=6;awburst=1;awlock=0;
            wlast=1;wstrb='1;wdata=0;
            wdata[63:0]=64'h313046555052544e;
            wdata[127:64]=lease_epoch;wdata[191:128]=lease_tag;
            for(integer i=0;i<32;i++) wdata[256+i*8+:8]=8'(i*37+11);
        end
    endtask
    task check(input [6:0] expected_faults);
        reg [63:0] expected_mask;
        reg [511:0] expected_data;
        integer offset;
        begin
            #10;cases++;
            if(faults!==expected_faults || accept!==(expected_faults==0))
                $fatal(1,"FRAGMENT_DECISION case=%0d wanted=%h actual=%h",cases,expected_faults,faults);
            expected_mask=0;expected_data=0;
            if(expected_faults==0) begin
                allowed++;offset=int'(wdata[255:192]);
                for(integer i=0;i<32;i++) begin
                    expected_mask[offset+i]=1;
                    expected_data[(offset+i)*8+:8]=wdata[256+i*8+:8];
                end
                if(byte_data!==expected_data) $fatal(1,"FRAGMENT_PAYLOAD case=%0d",cases);
            end else rejected++;
            if(byte_valid!==expected_mask) $fatal(1,"FRAGMENT_BITMAP case=%0d",cases);
            $display("NTRPU_FRAGMENT_CASE n=%0d faults=%h accept=%b mask=%h",cases,faults,accept,byte_valid);
        end
    endtask
    initial begin
        for(integer part=0;part<2;part++) begin
            baseline();wdata[255:192]=64'(part*32);check(0);
            for(integer i=0;i<256;i++) begin
                wdata[256+i]=~wdata[256+i];check(0);wdata[256+i]=~wdata[256+i];
            end
        end
        for(integer field=0;field<4;field++) for(integer i=0;i<64;i++) begin
            baseline();wdata[field*64+i]=~wdata[field*64+i];
            if(field==3 && i==5) check(0);
            else check(7'(1<<(field+3)));
        end
        for(integer i=0;i<64;i++) begin
            baseline();awaddr[i]=~awaddr[i];check(i<6 ? 7'h06 : 7'h04);
        end
        baseline();lease_valid=0;check(1);
        for(integer i=0;i<6;i++) begin baseline();lease_window[i]=1;check(5);end
        for(integer i=0;i<8;i++) if(i!=6) begin baseline();awsize=3'(i);check(2);end
        for(integer i=0;i<4;i++) if(i!=1) begin baseline();awburst=2'(i);check(2);end
        for(integer i=1;i<256;i++) begin baseline();awlen=8'(i);check(2);end
        for(integer i=0;i<64;i++) begin baseline();wstrb[i]=0;check(2);end
        baseline();awlock=1;check(2);
        baseline();wlast=0;check(2);
        // Same BAR, fresh lease: an old frame cannot become new payload.
        baseline();lease_epoch++;check(7'h10);
        wdata[127:64]=lease_epoch;lease_tag++;check(7'h20);
        wdata[191:128]=lease_tag;check(0);
        baseline();lease_valid=0;awsize=2;awaddr=awaddr^64;
        wdata[0]=~wdata[0];wdata[64]=~wdata[64];wdata[128]=~wdata[128];
        wdata[255:192]=3;check(7'h7f);
        if(cases!=1176 || allowed!=516 || rejected!=660) $fatal(1,"FRAGMENT_COUNTS");
        $display("NTRPU_FRAGMENT_PASS cases=%0d allowed=%0d rejected=%0d",cases,allowed,rejected);
        $finish;
    end
endmodule
