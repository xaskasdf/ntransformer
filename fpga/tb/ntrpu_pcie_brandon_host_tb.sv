// Compile/load/read the actual specialized host RAM before the long PCIe run.
`timescale 1ns/1ps
module host_unit_root;
    logic user_clk=0,user_reset=1;
    logic m_axis_cq_tvalid=0,m_axis_cq_tready=0,m_axis_cq_tlast=0;
    logic [511:0] m_axis_cq_tdata=0;
    logic [15:0] m_axis_cq_tkeep=0;
    logic [182:0] m_axis_cq_tuser=0;
    always #2 user_clk=~user_clk;
endmodule
module host_unit_app;
    integer input_bytes=0,core_bytes=0,result_read_bytes=0,acknowledgements=0;
endmodule
module host_unit_endpoint;
    host_unit_app ntrpu_app();
endmodule
module board;
    host_unit_root RP();
    host_unit_endpoint EP();
    ntrpu_pcie_ffn_host ntrpu_host();
    `include "fixture_crc.svh"
    integer mode,total=0,requests=0,count,remaining;
    logic [63:0] address;
    logic [31:0] crc;
    byte unsigned value;
    function automatic logic [31:0] update_crc(input logic [31:0] state,input byte unsigned data);
        logic [31:0] next_state;
        next_state=state^data;
        for(int bit_index=0;bit_index<8;bit_index++)
            next_state=(next_state>>1)^(next_state[0]?32'hedb88320:0);
        return next_state;
    endfunction
    initial begin
        if(!$value$plusargs("MODE=%d",mode))mode=0;
        #1;ntrpu_host.initialize();
        case(mode)
            1:value=ntrpu_host.read_completion(64'h00000fc0,64,0);
            2:value=ntrpu_host.read_completion(64'h200010fc0,64,0);
            3:value=ntrpu_host.read_completion(64'h200000fc0,128,0);
            4:value=ntrpu_host.read_completion(64'h200100fc0+195840,64,0);
            5:ntrpu_host.initialize();
            6:value=ntrpu_host.read_completion(64'h200000fc0,64,-1);
            7:value=ntrpu_host.read_completion(64'h200000fc0,0,0);
            8:value=ntrpu_host.read_completion(64'h100000180,4,0);
            default:begin
                for(int b=0;b<6;b++)begin
                    address=b<4?ntrpu_host.source_address(b):(b==4?64'h100000100:64'h300000300);
                    remaining=b<4?ntrpu_host.spans[b]:(b==4?128:32);
                    crc='1;
                    while(remaining>0)begin
                        count=remaining<256?remaining:256;
                        if(count>4096-int'(address[11:0]))count=4096-int'(address[11:0]);
                        for(int i=0;i<count;i++)begin
                            value=ntrpu_host.read_completion(address,count,i);
                            if($isunknown(value))$fatal(1,"BRANDON_HOST_UNKNOWN");
                            crc=update_crc(crc,value);
                        end
                        for(int i=count;i<256;i++)
                            if(ntrpu_host.read_completion(address,count,i)!==0)$fatal(1,"BRANDON_HOST_PADDING");
                        total+=count;requests++;remaining-=count;address+=64'(count);
                    end
                    crc=~crc;
                    if(crc!==EXPECTED_CRC[b])$fatal(1,"BRANDON_HOST_CRC buffer=%0d got=%h expected=%h",b,crc,EXPECTED_CRC[b]);
                    $display("NTRPU_BRANDON_HOST_BUFFER id=%0d crc32=%08h",b,crc);
                end
                if(total!=592544)$fatal(1,"BRANDON_HOST_TOTAL");
                for(int i=0;i<1024;i++)if($isunknown(ntrpu_host.expected(i)))$fatal(1,"BRANDON_HOST_OUTPUT_UNKNOWN");
                $display("NTRPU_BRANDON_HOST_PASS bytes=%0d requests=%0d output_reference_bytes=1024",total,requests);
                $finish;
            end
        endcase
        $fatal(1,"BRANDON_HOST_NEGATIVE_ESCAPED");
    end
endmodule
