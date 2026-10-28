`timescale 1ns/1ps
`default_nettype none
// Experimental boot loader. One full 32-byte line per retained transaction.
// Both cold resets MUST assert together; transport reset never clears ownership.
// Bundled CDC buses require physical bounds before use on a board.
module ntrpu_hbm_weight_loader (
    input wire clk,cold_rst_n,bus_rst_n,mem_clk,mem_cold_rst_n,
    input wire memory_ready_async,
    input wire [11:0] s_awaddr,s_araddr,
    input wire [2:0] s_awprot,s_arprot,
    input wire s_awvalid,s_wvalid,s_bready,s_arvalid,s_rready,
    input wire [31:0] s_wdata,
    input wire [3:0] s_wstrb,
    output wire s_awready,s_wready,s_arready,
    output reg s_bvalid=0,s_rvalid=0,
    output reg [1:0] s_bresp=0,s_rresp=0,
    output reg [31:0] s_rdata=0,
    output reg [11:0] m_awaddr=0,m_araddr=0,
    output reg [2:0] m_awprot=0,m_arprot=0,
    output reg [31:0] m_wdata=0,
    output reg [3:0] m_wstrb=0,
    output reg m_awvalid=0,m_wvalid=0,m_arvalid=0,
    input wire m_awready,m_wready,m_arready,m_bvalid,m_rvalid,
    input wire [1:0] m_bresp,m_rresp,
    input wire [31:0] m_rdata,
    output wire m_bready,m_rready,
    output reg sealed=0,
    output wire [67:0] h_awaddr,
    output wire [3:0] h_awburst,
    output wire [11:0] h_awid,
    output wire [7:0] h_awlen,
    output wire [5:0] h_awsize,
    output wire [1:0] h_awvalid,h_wvalid,h_wlast,h_bready,
    input wire [1:0] h_awready,h_wready,h_bvalid,
    output wire [511:0] h_wdata,
    output wire [63:0] h_wstrb,
    input wire [11:0] h_bid,
    input wire [3:0] h_bresp
);
    wire transport_rst_n=cold_rst_n&&bus_rst_n;
    localparam [1:0] COLLECT=0,OFFER=1,RESPONSE=2,RETURN=3;
    reg [1:0] ws=COLLECT,rs=COLLECT;
    reg aw_full=0,w_full=0;
    reg [11:0] wa=0;
    reg [31:0] wd=0;
    reg [3:0] strobe=0;
    wire write_fire=transport_rst_n&&ws==COLLECT&&aw_full&&w_full;
    wire local_write=wa[11:8]==4'h8;
    wire blocked_command=wa==12'h010&&!sealed;
    wire local_read=s_araddr[11:8]==4'h8;
    reg write_ok,read_ok;
    reg [31:0] read_data;
    // {payload256, tag64, address64}; byte validity prevents stale staging reuse.
    reg [383:0] shadow=0,active=0;
    reg [47:0] written=0;
    reg busy=0,done=0,failed=0,rejected=0;
    reg request=0,settle_ack=0;
    reg [3:0] completion=0;
    reg [31:0] accepted=0;
    reg acknowledge=0;
    reg [3:0] memory_result=0;
    (* ASYNC_REG="TRUE" *) reg [2:0] ack_sync=0,ready_sync=0;
    (* ASYNC_REG="TRUE" *) reg [2:0] req_sync=0,mem_ready_sync=0;
    wire available=ready_sync[2];
    wire staging=wa>=12'h840&&wa<=12'h86c;
    wire idle=!busy&&!done;
    assign s_awready=transport_rst_n&&ws==COLLECT&&!aw_full;
    assign s_wready=transport_rst_n&&ws==COLLECT&&!w_full;
    assign s_arready=transport_rst_n&&rs==COLLECT;
    assign m_bready=transport_rst_n&&ws==RESPONSE;
    assign m_rready=transport_rst_n&&rs==RESPONSE;

    always @* begin
        write_ok=0;
        if(wa[1:0]==0) begin
            if(staging) write_ok=idle&&!sealed&&!failed;
            else if(strobe==15&&wd==1) case(wa)
                12'h80c:write_ok=idle&&!sealed&&!failed&&available&&(&written)&&
                    shadow[63:34]==0&&shadow[4:0]==0;
                12'h810:write_ok=done;
                12'h814:write_ok=idle&&!sealed&&!failed&&!rejected&&available&&accepted!=0&&written==0;
                12'h818:write_ok=1; // Clear rejected-MMIO indication only, not HBM failure.
                default:write_ok=0;
            endcase
        end
    end
    always @* begin
        read_data=0;read_ok=1;
        if(s_araddr[1:0]!=0)read_ok=0;
        else if(s_araddr>=12'h840&&s_araddr<=12'h86c)
            read_data=shadow[((s_araddr-12'h840)>>2)*32+:32];
        else case(s_araddr)
            12'h800:read_data=32'h4e544c44; // NTLD
            12'h804:read_data=1;
            12'h808:read_data={26'd0,rejected,available,failed,sealed,done,busy};
            12'h81c:read_data=accepted;
            12'h820:read_data=active[95:64];
            12'h824:read_data=active[127:96];
            12'h828:read_data={28'd0,completion};
            12'h82c:read_data=active[31:0];
            12'h830:read_data=active[63:32];
            12'h834:read_data=written[31:0];
            12'h838:read_data={16'd0,written[47:32]};
            default:read_ok=0;
        endcase
    end
    // Registered, independent AXI-Lite channels; exactly one read and one write.
    always @(posedge clk or negedge transport_rst_n) begin
        if(!transport_rst_n) begin
            ws<=COLLECT;rs<=COLLECT;aw_full<=0;w_full<=0;wa<=0;wd<=0;strobe<=0;
            s_bvalid<=0;s_rvalid<=0;s_bresp<=0;s_rresp<=0;s_rdata<=0;
            m_awaddr<=0;m_araddr<=0;m_awprot<=0;m_arprot<=0;m_wdata<=0;m_wstrb<=0;
            m_awvalid<=0;m_wvalid<=0;m_arvalid<=0;
        end else begin
            if(s_awvalid&&s_awready)begin aw_full<=1;wa<=s_awaddr;m_awprot<=s_awprot;end
            if(s_wvalid&&s_wready)begin w_full<=1;wd<=s_wdata;strobe<=s_wstrb;end
            case(ws)
                COLLECT:if(write_fire)begin
                    aw_full<=0;w_full<=0;
                    if(local_write||blocked_command)begin
                        s_bvalid<=1;s_bresp<=local_write&&write_ok?0:2;ws<=RETURN;
                    end else begin
                        m_awaddr<=wa;m_wdata<=wd;m_wstrb<=strobe;
                        m_awvalid<=1;m_wvalid<=1;ws<=OFFER;
                    end
                end
                OFFER:begin
                    if(m_awvalid&&m_awready)m_awvalid<=0;
                    if(m_wvalid&&m_wready)m_wvalid<=0;
                    if((!m_awvalid||m_awready)&&(!m_wvalid||m_wready))ws<=RESPONSE;
                end
                RESPONSE:if(m_bvalid)begin s_bvalid<=1;s_bresp<=m_bresp;ws<=RETURN;end
                RETURN:if(s_bvalid&&s_bready)begin s_bvalid<=0;ws<=COLLECT;end
            endcase
            if(s_arvalid&&s_arready)begin
                if(local_read)begin s_rvalid<=1;s_rresp<=read_ok?0:2;s_rdata<=read_data;rs<=RETURN;end
                else begin m_araddr<=s_araddr;m_arprot<=s_arprot;m_arvalid<=1;rs<=OFFER;end
            end
            case(rs)
                OFFER:if(m_arvalid&&m_arready)begin m_arvalid<=0;rs<=RESPONSE;end
                RESPONSE:if(m_rvalid)begin s_rvalid<=1;s_rresp<=m_rresp;s_rdata<=m_rdata;rs<=RETURN;end
                RETURN:if(s_rvalid&&s_rready)begin s_rvalid<=0;rs<=COLLECT;end
                default:begin end
            endcase
        end
    end
    // State below survives transport reset, including accepted but unacknowledged work.
    always @(posedge clk or negedge cold_rst_n) begin
        if(!cold_rst_n)begin
            shadow<=0;active<=0;written<=0;busy<=0;done<=0;failed<=0;rejected<=0;
            sealed<=0;request<=0;settle_ack<=0;completion<=0;accepted<=0;ack_sync<=0;ready_sync<=0;
        end else begin
            ack_sync<={ack_sync[1:0],acknowledge};ready_sync<={ready_sync[1:0],mem_ready_sync[2]};
            if(busy&&ack_sync[2]==request)begin
                if(!settle_ack)settle_ack<=1;
                else begin
                    busy<=0;done<=1;completion<=memory_result;failed<=failed||(memory_result!=0);settle_ack<=0;
                end
            end
            if(write_fire&&(local_write||blocked_command))begin
                if(blocked_command||!write_ok)rejected<=1;
                else if(staging)begin
                    for(integer b=0;b<4;b++)if(strobe[b])begin
                        shadow[((wa-12'h840)>>2)*32+b*8+:8]<=wd[b*8+:8];
                        written[((wa-12'h840)>>2)*4+b]<=1;
                    end
                end else case(wa)
                    12'h80c:begin active<=shadow;written<=0;busy<=1;request<=!request;accepted<=accepted+1;end
                    12'h810:done<=0;
                    12'h814:sealed<=1;
                    12'h818:rejected<=0;
                    default:begin end
                endcase
            end
        end
    end

    localparam [2:0] M_IDLE=0,M_SETTLE=1,M_OFFER=2,M_RESPONSE=3;
    reg [2:0] ms=M_IDLE;
    reg memory_token=0,lane=0,aw_pending=0,w_pending=0,lost_ready=0;
    reg [33:0] address=0;
    reg [255:0] payload=0;
    assign h_awaddr={address,address};
    assign h_awburst=4'b0101;
    assign h_awid=0;
    assign h_awlen=0;
    assign h_awsize={3'd5,3'd5};
    assign h_wdata={payload,payload};
    assign h_wstrb=64'hffffffffffffffff;
    assign h_wlast=2'b11;
    assign h_awvalid=(mem_cold_rst_n&&ms==M_OFFER&&aw_pending)?(lane?2'b10:2'b01):2'b00;
    assign h_wvalid=(mem_cold_rst_n&&ms==M_OFFER&&w_pending)?(lane?2'b10:2'b01):2'b00;
    assign h_bready=(mem_cold_rst_n&&ms==M_RESPONSE)?(lane?2'b10:2'b01):2'b00;
    always @(posedge mem_clk or negedge mem_cold_rst_n)begin
        if(!mem_cold_rst_n)begin
            req_sync<=0;mem_ready_sync<=0;ms<=M_IDLE;memory_token<=0;lane<=0;
            aw_pending<=0;w_pending<=0;lost_ready<=0;address<=0;payload<=0;
            acknowledge<=0;memory_result<=0;
        end else begin
            req_sync<={req_sync[1:0],request};mem_ready_sync<={mem_ready_sync[1:0],memory_ready_async};
            if((ms==M_OFFER||ms==M_RESPONSE)&&!mem_ready_sync[2])lost_ready<=1;
            case(ms)
                M_IDLE:if(req_sync[2]!=acknowledge&&mem_ready_sync[2])ms<=M_SETTLE;
                M_SETTLE:if(mem_ready_sync[2])begin
                    memory_token<=req_sync[2];address<=active[33:0];lane<=active[33];payload<=active[383:128];
                    aw_pending<=1;w_pending<=1;lost_ready<=0;ms<=M_OFFER;
                end
                M_OFFER:begin
                    if(h_awvalid[lane]&&h_awready[lane])aw_pending<=0;
                    if(h_wvalid[lane]&&h_wready[lane])w_pending<=0;
                    if((!aw_pending||h_awready[lane])&&(!w_pending||h_wready[lane]))ms<=M_RESPONSE;
                end
                M_RESPONSE:if(h_bvalid[lane])begin
                    memory_result<={lost_ready||!mem_ready_sync[2],h_bid[lane*6+:6]!=0,h_bresp[lane*2+:2]};
                    acknowledge<=memory_token;ms<=M_IDLE;
                end
                default:begin end
            endcase
        end
    end
endmodule
`default_nettype wire
