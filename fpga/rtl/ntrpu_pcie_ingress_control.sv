`timescale 1ns / 1ps
// Privileged AXI-Lite control in the ingress UI clock domain. Experimental ABI 1.
// This block never manufactures source_stopped/ingress_isolated/quiesce_epoch.
// arm_ready and result_valid must come from the complete guarded ingress path.
// A queued ARM remains valid until accepted; STOP does not reset any transport.
// Cold/coordinated reset only; host access control and control CDC are external.
module ntrpu_pcie_ingress_control (
    input wire clk, rst_n,
    input wire [31:0] s_axil_awaddr, s_axil_wdata, s_axil_araddr,
    input wire [3:0] s_axil_wstrb,
    input wire s_axil_awvalid, s_axil_wvalid, s_axil_bready,
    input wire s_axil_arvalid, s_axil_rready,
    output wire s_axil_awready, s_axil_wready, s_axil_arready,
    output reg s_axil_bvalid, s_axil_rvalid,
    output reg [1:0] s_axil_bresp, s_axil_rresp,
    output reg [31:0] s_axil_rdata,
    output reg arm_valid, stop,
    input wire arm_ready,
    output reg [63:0] arm_epoch, arm_tag, arm_window, arm_destination,
    input wire [63:0] epoch, valid_bytes,
    input wire lease_held, failed, transport_idle, guard_idle, peer_axi_busy,
    input wire [3:0] error_code,
    input wire source_stopped, ingress_isolated,
    input wire [63:0] quiesce_epoch,
    input wire result_valid,
    output wire result_ready,
    input wire [63:0] result_epoch, result_tag,
    input wire [3:0] result_status,
    output wire irq,
    output reg protocol_fault
);
    localparam [1:0] OKAY=0, SLVERR=2, DECERR=3;
    reg aw_pending, w_pending, running, irq_enable;
    reg [31:0] write_address, write_data;
    reg [3:0] write_strobes;
    reg [31:0] descriptor [0:7];
    reg [31:0] ack_words [0:3];
    reg [31:0] accepted_count, retired_count;
    reg [31:0] mask, command;
    reg [1:0] response;
    wire write_fire=aw_pending && w_pending && !s_axil_bvalid;
    wire identity_ok=result_epoch==arm_epoch && result_tag==arm_tag;
    wire ack_ok={ack_words[1],ack_words[0]}==result_epoch &&
                {ack_words[3],ack_words[2]}==result_tag;
    wire descriptor_ok={descriptor[1],descriptor[0]}>epoch &&
                       descriptor[4][5:0]==0 && descriptor[6][5:0]==0;
    wire unexpected_result=result_valid && (!running || !identity_ok);
    assign s_axil_awready=rst_n && !aw_pending && !s_axil_bvalid;
    assign s_axil_wready=rst_n && !w_pending && !s_axil_bvalid;
    assign s_axil_arready=rst_n && !s_axil_rvalid;
    assign result_ready=rst_n && write_fire && response==OKAY &&
                        write_address==32'h10 && command==4;
    assign irq=irq_enable && (result_valid || protocol_fault);

    function automatic address_valid(input [31:0] a);
        address_valid=a[1:0]==0 && (a<=32'h1c ||
            (a>=32'h20 && a<=32'h5c) || (a>=32'h60 && a<=32'h8c));
    endfunction
    function automatic [31:0] read_register(input [31:0] a);
        if(a>=32'h20 && a<=32'h3c) read_register=descriptor[(a-32'h20)>>2];
        else if(a>=32'h40 && a<=32'h4c) read_register=ack_words[(a-32'h40)>>2];
        else case(a)
            32'h00:read_register=32'h4952544e; // Little-endian NTRI.
            32'h04:read_register=32'h00010000;
            32'h08:read_register=32'h00000001; // External barrier REQUIRED.
            32'h0c:read_register={20'd0,peer_axi_busy,guard_idle,transport_idle,
                failed,lease_held,arm_ready,protocol_fault,stop,result_valid,running,arm_valid,irq};
            32'h10:read_register=0;
            32'h14:read_register={31'd0,irq_enable};
            32'h18:read_register=accepted_count;
            32'h1c:read_register=retired_count;
            32'h50:read_register=epoch[31:0];
            32'h54:read_register=epoch[63:32];
            32'h58:read_register=valid_bytes[31:0];
            32'h5c:read_register=valid_bytes[63:32];
            32'h60:read_register=result_epoch[31:0];
            32'h64:read_register=result_epoch[63:32];
            32'h68:read_register=result_tag[31:0];
            32'h6c:read_register=result_tag[63:32];
            32'h70:read_register={28'd0,result_status};
            32'h74:read_register={28'd0,error_code};
            32'h78:read_register={30'd0,ingress_isolated,source_stopped};
            32'h7c:read_register=quiesce_epoch[31:0];
            32'h80:read_register=quiesce_epoch[63:32];
            32'h84:read_register=arm_epoch[31:0];
            32'h88:read_register=arm_epoch[63:32];
            32'h8c:read_register={31'd0,protocol_fault};
            default:read_register=0;
        endcase
    endfunction
    always @* begin
        for(integer i=0;i<4;i=i+1) mask[i*8+:8]={8{write_strobes[i]}};
        command=write_data & mask;
        response=OKAY;
        if(!address_valid(write_address)) response=DECERR;
        else if(write_address==32'h10) begin
            case(command)
                0:begin end
                1:if(arm_valid || running || result_valid || protocol_fault ||
                     unexpected_result || !descriptor_ok) response=SLVERR;
                2:if(!running || stop) response=SLVERR;
                4:if(!running || !result_valid || !identity_ok || !ack_ok ||
                     protocol_fault) response=SLVERR;
                default:response=SLVERR;
            endcase
        end else if(write_address==32'h14) begin
            if((command & 32'hfffffffe)!=0) response=SLVERR;
        end else if(write_address>=32'h20 && write_address<=32'h3c) begin
            if(arm_valid || running || result_valid || protocol_fault) response=SLVERR;
        end else if(!(write_address>=32'h40 && write_address<=32'h4c)) response=SLVERR;
    end
    always @(posedge clk) begin
        if(!rst_n) begin
            aw_pending<=0;w_pending<=0;write_address<=0;write_data<=0;write_strobes<=0;
            s_axil_bvalid<=0;s_axil_bresp<=0;s_axil_rvalid<=0;s_axil_rresp<=0;s_axil_rdata<=0;
            arm_valid<=0;arm_epoch<=0;arm_tag<=0;arm_window<=0;arm_destination<=0;
            stop<=0;running<=0;irq_enable<=0;protocol_fault<=0;accepted_count<=0;retired_count<=0;
            for(integer i=0;i<8;i=i+1) descriptor[i]<=0;
            for(integer i=0;i<4;i=i+1) ack_words[i]<=0;
        end else begin
            if(s_axil_awvalid && s_axil_awready) begin aw_pending<=1;write_address<=s_axil_awaddr;end
            if(s_axil_wvalid && s_axil_wready) begin w_pending<=1;write_data<=s_axil_wdata;write_strobes<=s_axil_wstrb;end
            if(s_axil_bvalid && s_axil_bready) s_axil_bvalid<=0;
            if(s_axil_rvalid && s_axil_rready) s_axil_rvalid<=0;
            if(s_axil_arvalid && s_axil_arready) begin
                s_axil_rvalid<=1;
                s_axil_rresp<=address_valid(s_axil_araddr) ? OKAY:DECERR;
                s_axil_rdata<=address_valid(s_axil_araddr) ? read_register(s_axil_araddr):0;
            end
            if(unexpected_result) protocol_fault<=1;
            if(arm_valid && arm_ready) begin arm_valid<=0;running<=1;accepted_count<=accepted_count+1;end
            if(write_fire) begin
                aw_pending<=0;w_pending<=0;s_axil_bvalid<=1;s_axil_bresp<=response;
                if(response==OKAY) begin
                    if(write_address==32'h10) case(command)
                        1:begin
                            arm_valid<=1;stop<=0;
                            arm_epoch<={descriptor[1],descriptor[0]};arm_tag<={descriptor[3],descriptor[2]};
                            arm_window<={descriptor[5],descriptor[4]};arm_destination<={descriptor[7],descriptor[6]};
                        end
                        2:stop<=1;
                        4:begin running<=0;retired_count<=retired_count+1;end
                        default:begin end
                    endcase
                    else if(write_address==32'h14 && write_strobes[0]) irq_enable<=write_data[0];
                    else if(write_address>=32'h20 && write_address<=32'h3c)
                        descriptor[(write_address-32'h20)>>2]<=(descriptor[(write_address-32'h20)>>2]&~mask)|command;
                    else if(write_address>=32'h40 && write_address<=32'h4c)
                        ack_words[(write_address-32'h40)>>2]<=(ack_words[(write_address-32'h40)>>2]&~mask)|command;
                end
            end
        end
    end
endmodule
