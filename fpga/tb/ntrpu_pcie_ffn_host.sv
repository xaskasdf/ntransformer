// Bounded host RAM and independent root-complex CQ observer; simulation only.
`timescale 1ps/1ps
module ntrpu_pcie_ffn_host;
    byte unsigned descriptors[0:159], inputs[0:3][0:1087], result[0:127];
    bit result_seen[0:127], initialized=0;
    int h2c_bytes=0,descriptor_bytes=0,c2h_bytes=0,msix_packets=0;
    int spans[0:3]='{64,1088,1088,1088};
    localparam logic [63:0] MSIX_ADDRESS=64'h500000800;
    localparam logic [31:0] MSIX_DATA=32'h8badc0de;

    function automatic int q(input int row);
        if(row==0) return 127;
        if(row==1) return 1;
        return row*7-113;
    endfunction
    function automatic logic [63:0] device_address(input int b);
        return 64'h100000fc0+64'(b)*64'h20000000;
    endfunction
    function automatic logic [63:0] source_address(input int b);
        return 64'h200000fc0+64'(b)*64'h10000;
    endfunction
    function automatic byte unsigned expected(input int off);
        logic [31:0] bits;
        bits=$shortrealtobits(shortreal'(257*q(((off/4)*7)%32)));
        return bits[(off%4)*8+:8];
    endfunction
    task initialize;
        logic [255:0] descriptor;
        logic [63:0] scale;
        logic [15:0] weight_scale;
        int row,k,selected;
        begin
            if(initialized) $fatal(1,"HOST_DOUBLE_INIT");
            for(int i=0;i<128;i++) begin result[i]='ha5;result_seen[i]=0;end
            for(int b=0;b<4;b++) begin
                descriptor={b==3 ? 64'b0:64'h100000100+64'((b+1)*32),
                    device_address(b),source_address(b),32'(spans[b]),
                    b==3 ? 32'had4b0013:32'had4b0010};
                for(int i=0;i<32;i++) descriptors[b*32+i]=descriptor[i*8+:8];
                for(int i=0;i<1088;i++) inputs[b][i]=0;
                if(b==0) begin
                    scale=$realtobits(1.0);
                    for(int i=0;i<32;i++) inputs[0][i]=8'(q(i));
                    for(int i=0;i<8;i++) inputs[0][32+i]=scale[i*8+:8];
                end else begin
                    weight_scale=(b==1 ? 16'h5c04:16'h3c00);
                    for(row=0;row<32;row++) begin
                        inputs[b][row*34]=weight_scale[7:0];
                        inputs[b][row*34+1]=weight_scale[15:8];
                        selected=(b==1 ? 1:(b==2 ? row:(row*7)%32));
                        inputs[b][row*34+2+selected]=1;
                    end
                end
            end
            descriptor={64'b0,64'h400000fc0,device_address(7),32'd128,32'had4b0013};
            for(int i=0;i<32;i++) descriptors[128+i]=descriptor[i*8+:8];
            initialized=1;
            $display("NTRPU_PCIE_FFN_HOST_INIT input_bytes=3328 output_bytes=128 dim=32 hidden=32");
        end
    endtask
    // Validate all 64 address bits and the entire request before indexing RAM.
    function automatic int region(input logic [63:0] address,input int count);
        if(!initialized || count<=0 || count>256 || address+64'(count)<address ||
           int'(address[11:0])+count>4096) $fatal(1,"HOST_READ_BOUNDS addr=%h bytes=%0d",address,count);
        if(address>=64'h100000100 && address+64'(count)<=64'h100000180) return 4;
        if(address>=64'h300000300 && address+64'(count)<=64'h300000320) return 5;
        for(int b=0;b<4;b++)
            if(address>=source_address(b) && address+64'(count)<=source_address(b)+64'(spans[b])) return b;
        $fatal(1,"HOST_READ_UNMAPPED addr=%h bytes=%0d",address,count);return -1;
    endfunction
    function automatic byte unsigned read_completion(input logic [63:0] address,input int count,index);
        int b,off;
        b=region(address,count);
        // AMD packs full CC beats: inactive lanes must not index past the request.
        if(index<0) $fatal(1,"HOST_NEGATIVE_INDEX");
        if(index>=count) return 0;
        if(b==4) return descriptors[int'(address-64'h100000100)+index];
        if(b==5) return descriptors[128+int'(address-64'h300000300)+index];
        return inputs[b][int'(address-source_address(b))+index];
    endfunction

    wire clk=board.RP.user_clk,reset=board.RP.user_reset;
    wire valid=board.RP.m_axis_cq_tvalid,ready=board.RP.m_axis_cq_tready;
    wire last=board.RP.m_axis_cq_tlast;
    wire [511:0] data=board.RP.m_axis_cq_tdata;
    wire [15:0] keep=board.RP.m_axis_cq_tkeep;
    wire [182:0] user=board.RP.m_axis_cq_tuser;
    bit held=0,in_packet=0,is_msix=0;
    logic [711:0] held_bus;
    logic [63:0] address;
    logic [15:0] expected_keep;
    logic [63:0] expected_be;
    int packet_bytes,packet_left,first_lane,beat_bytes,offset,b,msix_bytes=0;
    always @(posedge clk) if(!reset) begin
        if(held && (valid!==1'b1 || {data,keep,user,last}!==held_bus)) $fatal(1,"FFN_CQ_STALLED_CHANGED");
        held=valid===1'b1 && ready!==1'b1;held_bus={data,keep,user,last};
        if(valid===1'b1 && ready===1'b1) begin
            first_lane=0;
            if($isunknown({keep,last,user[85:16]})) $fatal(1,"FFN_CQ_UNKNOWN");
            if(!in_packet) begin
                if(user[81:80]!==2'b01 || user[83:82]!==0 || keep[3:0]!==4'hf || $isunknown(data[127:0]))
                    $fatal(1,"FFN_CQ_HEADER");
                address={data[63:2],2'b00};packet_bytes=int'(data[74:64])*4;
                if(data[79] || data[1:0]!=0 || user[3:0]!==4'hf ||
                   user[11:8]!==(packet_bytes==4 ? 4'h0:4'hf) || packet_bytes<=0 || packet_bytes>256 ||
                   int'(address[11:0])+packet_bytes>4096) $fatal(1,"FFN_CQ_RANGE_OR_BE");
                $display("NTRPU_PCIE_FFN_CQ_REQUEST type=%0d addr=%016h bytes=%0d tc=%0d attr=%0d time_ps=%0d",data[78:75],address,packet_bytes,data[123:121],data[126:124],$time);
                if(data[78:75]==0) begin
                    b=region(address,packet_bytes);
                    if(b>=4) descriptor_bytes+=packet_bytes;else h2c_bytes+=packet_bytes;
                    if(keep!==16'h000f || last!==1'b1 || user[79:16]!==0) $fatal(1,"FFN_CQ_READ_PAYLOAD");
                end else if(data[78:75]==1) begin
                    is_msix=address==MSIX_ADDRESS;
                    if(is_msix) begin
                        if(packet_bytes!=4) $fatal(1,"FFN_MSIX_SIZE");
                        msix_bytes=0;
                    end else if(address!=64'h400000fc0+64'(c2h_bytes) || c2h_bytes+packet_bytes>128)
                        $fatal(1,"FFN_CQ_C2H_RANGE");
                    packet_left=packet_bytes;first_lane=16;in_packet=1;
                end else $fatal(1,"FFN_CQ_TYPE");
            end else if(user[81:80]!==0) $fatal(1,"FFN_CQ_NESTED");
            if(in_packet) begin
                beat_bytes=packet_left<64-first_lane ? packet_left:64-first_lane;
                expected_keep=16'hffff>>(16-(first_lane+beat_bytes)/4);
                expected_be=(64'hffffffffffffffff>>(64-beat_bytes))<<first_lane;
                if(keep!==expected_keep || user[79:16]!==expected_be) $fatal(1,"FFN_CQ_PAYLOAD_BE");
                $display("NTRPU_PCIE_FFN_CQ_DATA addr=%016h left=%0d first=%0d bytes=%0d data=%0128h time_ps=%0d",address,packet_left,first_lane,beat_bytes,data,$time);
                for(int lane=first_lane;lane<first_lane+beat_bytes;lane++) begin
                    if(is_msix) begin
                        if(data[lane*8+:8]!==MSIX_DATA[msix_bytes*8+:8]) $fatal(1,"FFN_MSIX_DATA");
                        msix_bytes++;
                    end else begin
                        if(result_seen[c2h_bytes] || data[lane*8+:8]!==expected(c2h_bytes))
                            $fatal(1,"FFN_C2H_NUMERIC offset=%0d",c2h_bytes);
                        result[c2h_bytes]=data[lane*8+:8];result_seen[c2h_bytes]=1;c2h_bytes++;
                    end
                end
                packet_left-=beat_bytes;
                if(last!==(packet_left==0)) $fatal(1,"FFN_CQ_LAST");
                if(last) begin
                    in_packet=0;
                    if(is_msix) begin msix_packets++;$display("NTRPU_PCIE_FFN_MSIX count=%0d time_ps=%0d",msix_packets,$time);end
                end
            end
        end
    end
    task verify_complete;
        begin
            if(in_packet || held || h2c_bytes!=3328 || descriptor_bytes!=160 || c2h_bytes!=128 || msix_packets!=2)
                $fatal(1,"FFN_HOST_INCOMPLETE h2c=%0d desc=%0d c2h=%0d msix=%0d",h2c_bytes,descriptor_bytes,c2h_bytes,msix_packets);
            for(int i=0;i<128;i++) if(!result_seen[i] || result[i]!==expected(i)) $fatal(1,"FFN_HOST_RESULT");
            if(board.EP.ntrpu_app.input_bytes!=3328 || board.EP.ntrpu_app.core_bytes!=704 ||
               board.EP.ntrpu_app.result_read_bytes!=128 || board.EP.ntrpu_app.acknowledgements!=2)
                $fatal(1,"FFN_ENDPOINT_INCOMPLETE");
            $display("NTRPU_PCIE_FFN_PASS cases=3 computed=1 rejected=2 h2c_bytes=3328 core_bytes=704 c2h_bytes=128 msix=2");
        end
    endtask
    initial begin #1000000000;$fatal(1,"FFN_PCIE_TIMEOUT");end
endmodule
