`timescale 1ns/1ps
`default_nettype none
// Provisional PG194 bridge-table profile: bit 28=0, window 0 at 0x208/0x20c.
// Local AXI aperture is a generated-IP constant, NOT a programmable CSR here.
// Exclusive control-port wiring is mandatory. A runtime readback is not a firewall.
module ntrpu_axi_bridge_mapper #(
    parameter integer WAIT_LIMIT=65536,
    parameter [63:0] APERTURE_BASE=64'h10000000
) (
    input wire clk,cold_rst_n,link_up,stop,lease_busy,
    input wire map_valid,
    output wire map_ready,
    input wire [63:0] map_session,map_generation,map_axi_base,map_pcie_base,
    output reg map_rsp_valid=0,map_rsp_ok=0,
    input wire map_rsp_ready,
    output reg [63:0] rsp_session=0,rsp_generation=0,rsp_axi_base=0,rsp_pcie_base=0,
    output reg fault=0,
    output reg [7:0] fault_code=0,
    output wire verified,
    output wire [31:0] m_axil_awaddr,m_axil_wdata,m_axil_araddr,
    output wire [2:0] m_axil_awprot,m_axil_arprot,
    output wire [3:0] m_axil_wstrb,
    output wire m_axil_awvalid,m_axil_wvalid,m_axil_bready,m_axil_arvalid,m_axil_rready,
    input wire m_axil_awready,m_axil_wready,m_axil_bvalid,m_axil_arready,m_axil_rvalid,
    input wire [1:0] m_axil_bresp,m_axil_rresp,
    input wire [31:0] m_axil_rdata
);
    localparam [2:0] IDLE=0,SEND=1,WAIT_RESPONSE=2,REPORT=3,HALTED=4;
    localparam [7:0] REQUEST=1,AXI_ERROR=2,READBACK=4,ENVIRONMENT=8,TIMEOUT=16,IDENTITY=32;
    reg [2:0] state=IDLE,step=0;
    reg seen_link=0,qualified=0;
    wire req_valid=state==SEND;
    wire req_ready,rsp_valid,rsp_timeout,timeout_latched,master_busy;
    wire [31:0] rsp_data;
    wire [1:0] rsp_code;
    wire [127:0] rsp_identity;
    reg [31:0] req_address,req_data,expected,mask;
    wire req_write=step==2 || step==3;
    wire [127:0] transaction_identity={rsp_session,rsp_generation};
    wire env_bad=stop || (seen_link && !link_up) ||
                 (lease_busy && (state==SEND || state==WAIT_RESPONSE));
    wire closed=fault || env_bad || timeout_latched;
    assign verified=qualified && !closed;
    assign map_ready=cold_rst_n && link_up && !lease_busy && !closed && state==IDLE;
    wire request_bad=map_session==0 || map_generation==0 ||
                     map_axi_base!=APERTURE_BASE || |map_pcie_base[11:0];
    always @* begin
        req_address=0;req_data=0;expected=0;mask=32'hffffffff;
        case(step)
            0:begin req_address=32'h130;expected=32'h8;mask=32'ha;end // Gen3 endpoint, not RP.
            1:begin req_address=32'h204;expected=32'h03800000;mask=32'hffff0000;end // VSEC length/revision.
            2:begin req_address=32'h208;req_data=rsp_pcie_base[63:32];end
            3:begin req_address=32'h20c;req_data=rsp_pcie_base[31:0];end
            4:begin req_address=32'h208;expected=rsp_pcie_base[63:32];end
            5:begin req_address=32'h20c;expected=rsp_pcie_base[31:0];end
            default:begin req_address=0;mask=0;end
        endcase
    end
    wire response_bad=rsp_code!=0 || rsp_timeout || rsp_identity!=transaction_identity ||
                      (!req_write && (rsp_data & mask)!=(expected & mask));
    // Once offered, an internal request is drained even if admission later closes.
    // No subsequent operation is offered after a fault. The master never cancels AXI.
    ntrpu_xdma_local_master #(.WAIT_LIMIT(WAIT_LIMIT)) bus_master(
        .clk(clk),.cold_rst_n(cold_rst_n),.admit(1'b1),
        .req_valid(req_valid),.req_ready(req_ready),.req_write(req_write),
        .req_address(req_address),.req_data(req_data),.req_strobe(4'hf),.req_identity(transaction_identity),
        .rsp_valid(rsp_valid),.rsp_ready(state==WAIT_RESPONSE),.rsp_data(rsp_data),.rsp_code(rsp_code),
        .rsp_identity(rsp_identity),.rsp_timeout(rsp_timeout),.timeout_latched(timeout_latched),.busy(master_busy),
        .m_axil_awaddr(m_axil_awaddr),.m_axil_awprot(m_axil_awprot),.m_axil_awvalid(m_axil_awvalid),.m_axil_awready(m_axil_awready),
        .m_axil_wdata(m_axil_wdata),.m_axil_wstrb(m_axil_wstrb),.m_axil_wvalid(m_axil_wvalid),.m_axil_wready(m_axil_wready),
        .m_axil_bresp(m_axil_bresp),.m_axil_bvalid(m_axil_bvalid),.m_axil_bready(m_axil_bready),
        .m_axil_araddr(m_axil_araddr),.m_axil_arprot(m_axil_arprot),.m_axil_arvalid(m_axil_arvalid),.m_axil_arready(m_axil_arready),
        .m_axil_rdata(m_axil_rdata),.m_axil_rresp(m_axil_rresp),.m_axil_rvalid(m_axil_rvalid),.m_axil_rready(m_axil_rready));
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            state<=IDLE;step<=0;seen_link<=0;qualified<=0;fault<=0;fault_code<=0;
            map_rsp_valid<=0;map_rsp_ok<=0;rsp_session<=0;rsp_generation<=0;rsp_axi_base<=0;rsp_pcie_base<=0;
        end else begin
            if(link_up)seen_link<=1;
            if(env_bad || timeout_latched)begin
                fault<=1;qualified<=0;
                fault_code<=fault_code | (env_bad?ENVIRONMENT:0) | (timeout_latched?TIMEOUT:0);
            end
            case(state)
                IDLE:if(map_valid && map_ready)begin
                    rsp_session<=map_session;rsp_generation<=map_generation;
                    rsp_axi_base<=map_axi_base;rsp_pcie_base<=map_pcie_base;
                    qualified<=0;step<=0;map_rsp_ok<=0;
                    if(request_bad)begin
                        fault<=1;fault_code<=fault_code|REQUEST;map_rsp_valid<=1;state<=REPORT;
                    end else state<=SEND;
                end
                SEND:if(req_valid && req_ready)state<=WAIT_RESPONSE;
                WAIT_RESPONSE:if(rsp_valid)begin
                    if(closed || response_bad)begin
                        fault<=1;qualified<=0;map_rsp_valid<=1;map_rsp_ok<=0;state<=REPORT;
                        fault_code<=fault_code | (env_bad?ENVIRONMENT:0) |
                            ((timeout_latched || rsp_timeout)?TIMEOUT:0) | (rsp_code!=0?AXI_ERROR:0) |
                            (rsp_identity!=transaction_identity?IDENTITY:0) |
                            ((!req_write && (rsp_data & mask)!=(expected & mask))?READBACK:0);
                    end else if(step==5)begin
                        qualified<=1;map_rsp_valid<=1;map_rsp_ok<=1;state<=REPORT;
                    end else begin step<=step+1'b1;state<=SEND;end
                end
                REPORT:if(map_rsp_valid && map_rsp_ready)begin
                    map_rsp_valid<=0;state<=closed?HALTED:IDLE;
                end
                HALTED:begin qualified<=0;end
                default:begin fault<=1;qualified<=0;state<=HALTED;end
            endcase
        end
    end
endmodule
`default_nettype wire
