`timescale 1ns / 1ps
module tb_ntrpu_axi4_retained_admission_r2;
    reg clk=0;always #5 clk=~clk;
    integer case_id=-1,cycle=0;reg running=0;
    reg cold_rst_n=0;
    reg close_request=0;
    reg proof_request=0;
    reg [63:0] proof_tag=0;
    reg s_awvalid=0;
    reg [63:0] s_awaddr=0;
    reg [3:0] s_awid=0;
    reg [7:0] s_awlen=0;
    reg [2:0] s_awsize=0;
    reg [2:0] s_awprot=0;
    reg [1:0] s_awburst=0;
    reg [3:0] s_awcache=0;
    reg s_awlock=0;
    reg s_wvalid=0;
    reg [511:0] s_wdata=0;
    reg [63:0] s_wstrb=0;
    reg s_wlast=0;
    reg s_bready=0;
    reg s_arvalid=0;
    reg [63:0] s_araddr=0;
    reg [3:0] s_arid=0;
    reg [7:0] s_arlen=0;
    reg [2:0] s_arsize=0;
    reg [2:0] s_arprot=0;
    reg [1:0] s_arburst=0;
    reg [3:0] s_arcache=0;
    reg s_arlock=0;
    reg s_rready=0;
    reg [3:0] s_awregion=0;
    reg [3:0] s_awqos=0;
    reg [3:0] s_arregion=0;
    reg [3:0] s_arqos=0;
    reg m_awready=0;
    reg m_wready=0;
    reg m_bvalid=0;
    reg [3:0] m_bid=0;
    reg [1:0] m_bresp=0;
    reg m_arready=0;
    reg m_rvalid=0;
    reg [511:0] m_rdata=0;
    reg [3:0] m_rid=0;
    reg [1:0] m_rresp=0;
    reg m_rlast=0;
    wire source_fenced;
    wire drain_ack;
    wire local_idle;
    wire protocol_fault;
    wire [63:0] drain_tag;
    wire [6:0] fault_sources;
    wire [5:0] aw_pending;
    wire [5:0] w_pending;
    wire [7:0] ar_pending;
    wire [8:0] w_partial;
    wire [5:0] aw_left;
    wire [5:0] w_left;
    wire ar_left;
    wire [1:0] state_debug;
    wire s_awready;
    wire s_wready;
    wire s_bvalid;
    wire [3:0] s_bid;
    wire [1:0] s_bresp;
    wire s_arready;
    wire s_rvalid;
    wire [511:0] s_rdata;
    wire [3:0] s_rid;
    wire [1:0] s_rresp;
    wire s_rlast;
    wire m_awvalid;
    wire [63:0] m_awaddr;
    wire [3:0] m_awid;
    wire [7:0] m_awlen;
    wire [2:0] m_awsize;
    wire [2:0] m_awprot;
    wire [1:0] m_awburst;
    wire [3:0] m_awcache;
    wire m_awlock;
    wire m_wvalid;
    wire [511:0] m_wdata;
    wire [63:0] m_wstrb;
    wire m_wlast;
    wire m_bready;
    wire m_arvalid;
    wire [63:0] m_araddr;
    wire [3:0] m_arid;
    wire [7:0] m_arlen;
    wire [2:0] m_arsize;
    wire [2:0] m_arprot;
    wire [1:0] m_arburst;
    wire [3:0] m_arcache;
    wire m_arlock;
    wire m_rready;
    wire [3:0] m_awregion;
    wire [3:0] m_awqos;
    wire [3:0] m_arregion;
    wire [3:0] m_arqos;
    ntrpu_axi4_retained_admission dut(.clk(clk),.cold_rst_n(cold_rst_n),.close_request(close_request),.proof_request(proof_request),.proof_tag(proof_tag),.s_awvalid(s_awvalid),.s_awaddr(s_awaddr),.s_awid(s_awid),.s_awlen(s_awlen),.s_awsize(s_awsize),.s_awprot(s_awprot),.s_awburst(s_awburst),.s_awcache(s_awcache),.s_awlock(s_awlock),.s_wvalid(s_wvalid),.s_wdata(s_wdata),.s_wstrb(s_wstrb),.s_wlast(s_wlast),.s_bready(s_bready),.s_arvalid(s_arvalid),.s_araddr(s_araddr),.s_arid(s_arid),.s_arlen(s_arlen),.s_arsize(s_arsize),.s_arprot(s_arprot),.s_arburst(s_arburst),.s_arcache(s_arcache),.s_arlock(s_arlock),.s_rready(s_rready),.s_awregion(s_awregion),.s_awqos(s_awqos),.s_arregion(s_arregion),.s_arqos(s_arqos),.m_awready(m_awready),.m_wready(m_wready),.m_bvalid(m_bvalid),.m_bid(m_bid),.m_bresp(m_bresp),.m_arready(m_arready),.m_rvalid(m_rvalid),.m_rdata(m_rdata),.m_rid(m_rid),.m_rresp(m_rresp),.m_rlast(m_rlast),.source_fenced(source_fenced),.drain_ack(drain_ack),.local_idle(local_idle),.protocol_fault(protocol_fault),.drain_tag(drain_tag),.fault_sources(fault_sources),.aw_pending(aw_pending),.w_pending(w_pending),.ar_pending(ar_pending),.w_partial(w_partial),.aw_left(aw_left),.w_left(w_left),.ar_left(ar_left),.state_debug(state_debug),.s_awready(s_awready),.s_wready(s_wready),.s_bvalid(s_bvalid),.s_bid(s_bid),.s_bresp(s_bresp),.s_arready(s_arready),.s_rvalid(s_rvalid),.s_rdata(s_rdata),.s_rid(s_rid),.s_rresp(s_rresp),.s_rlast(s_rlast),.m_awvalid(m_awvalid),.m_awaddr(m_awaddr),.m_awid(m_awid),.m_awlen(m_awlen),.m_awsize(m_awsize),.m_awprot(m_awprot),.m_awburst(m_awburst),.m_awcache(m_awcache),.m_awlock(m_awlock),.m_wvalid(m_wvalid),.m_wdata(m_wdata),.m_wstrb(m_wstrb),.m_wlast(m_wlast),.m_bready(m_bready),.m_arvalid(m_arvalid),.m_araddr(m_araddr),.m_arid(m_arid),.m_arlen(m_arlen),.m_arsize(m_arsize),.m_arprot(m_arprot),.m_arburst(m_arburst),.m_arcache(m_arcache),.m_arlock(m_arlock),.m_rready(m_rready),.m_awregion(m_awregion),.m_awqos(m_awqos),.m_arregion(m_arregion),.m_arqos(m_arqos));
    always @(posedge clk)if(running)begin
        $display("AXI4_ADMISSION_SAMPLE case=%0d cycle=%0d time=%0t cold_rst_n=%h close_request=%h proof_request=%h proof_tag=%h s_awvalid=%h s_awaddr=%h s_awid=%h s_awlen=%h s_awsize=%h s_awprot=%h s_awburst=%h s_awcache=%h s_awlock=%h s_wvalid=%h s_wdata=%h s_wstrb=%h s_wlast=%h s_bready=%h s_arvalid=%h s_araddr=%h s_arid=%h s_arlen=%h s_arsize=%h s_arprot=%h s_arburst=%h s_arcache=%h s_arlock=%h s_rready=%h s_awregion=%h s_awqos=%h s_arregion=%h s_arqos=%h m_awready=%h m_wready=%h m_bvalid=%h m_bid=%h m_bresp=%h m_arready=%h m_rvalid=%h m_rdata=%h m_rid=%h m_rresp=%h m_rlast=%h source_fenced=%h drain_ack=%h local_idle=%h protocol_fault=%h drain_tag=%h fault_sources=%h aw_pending=%h w_pending=%h ar_pending=%h w_partial=%h aw_left=%h w_left=%h ar_left=%h state_debug=%h s_awready=%h s_wready=%h s_bvalid=%h s_bid=%h s_bresp=%h s_arready=%h s_rvalid=%h s_rdata=%h s_rid=%h s_rresp=%h s_rlast=%h m_awvalid=%h m_awaddr=%h m_awid=%h m_awlen=%h m_awsize=%h m_awprot=%h m_awburst=%h m_awcache=%h m_awlock=%h m_wvalid=%h m_wdata=%h m_wstrb=%h m_wlast=%h m_bready=%h m_arvalid=%h m_araddr=%h m_arid=%h m_arlen=%h m_arsize=%h m_arprot=%h m_arburst=%h m_arcache=%h m_arlock=%h m_rready=%h m_awregion=%h m_awqos=%h m_arregion=%h m_arqos=%h",case_id,cycle,$time,cold_rst_n,close_request,proof_request,proof_tag,s_awvalid,s_awaddr,s_awid,s_awlen,s_awsize,s_awprot,s_awburst,s_awcache,s_awlock,s_wvalid,s_wdata,s_wstrb,s_wlast,s_bready,s_arvalid,s_araddr,s_arid,s_arlen,s_arsize,s_arprot,s_arburst,s_arcache,s_arlock,s_rready,s_awregion,s_awqos,s_arregion,s_arqos,m_awready,m_wready,m_bvalid,m_bid,m_bresp,m_arready,m_rvalid,m_rdata,m_rid,m_rresp,m_rlast,source_fenced,drain_ack,local_idle,protocol_fault,drain_tag,fault_sources,aw_pending,w_pending,ar_pending,w_partial,aw_left,w_left,ar_left,state_debug,s_awready,s_wready,s_bvalid,s_bid,s_bresp,s_arready,s_rvalid,s_rdata,s_rid,s_rresp,s_rlast,m_awvalid,m_awaddr,m_awid,m_awlen,m_awsize,m_awprot,m_awburst,m_awcache,m_awlock,m_wvalid,m_wdata,m_wstrb,m_wlast,m_bready,m_arvalid,m_araddr,m_arid,m_arlen,m_arsize,m_arprot,m_arburst,m_arcache,m_arlock,m_rready,m_awregion,m_awqos,m_arregion,m_arqos);
        cycle=cycle+1;
    end

    reg allow_aw=1,allow_w=1,allow_ar=1,respond=1,allow_b=1,allow_r=1,stalls=0;
    reg inject_b=0,inject_r=0;
    integer aw_count=0,w_count=0,ar_count=0,w_run=0,b_pick=-1,r_pick=-1,next_id=0;
    integer aw_ids[0:1023],aw_lens[0:1023],w_lens[0:1023];
    integer ar_ids[0:1023],ar_lens[0:1023],ar_beats[0:1023];
    reg [63:0] ar_addrs[0:1023];reg b_done[0:1023],r_done[0:1023];
    integer picked,scan,idscan;
    function integer length(input integer n);
        case(n%8)0:length=1;1:length=2;2:length=4;3:length=16;4:length=64;5:length=256;6:length=3;default:length=1;endcase
    endfunction
    function [511:0] pattern(input integer seed);
        for(integer j=0;j<16;j=j+1)pattern[j*32+:32]=32'h9e3779b9*(seed+j+1)^32'hd00dbeef;
    endfunction
    // This responder is a fixture, not a DDR/controller/PCIe model. It pairs
    // accepted requests and returns deterministic payloads, reordered by ID.
    always @(posedge clk)begin
        if(!cold_rst_n)begin
            aw_count=0;w_count=0;ar_count=0;w_run=0;next_id=0;
            for(integer n=0;n<1024;n=n+1)begin b_done[n]=0;r_done[n]=0;ar_beats[n]=0;end
        end else begin
            if(m_awvalid && m_awready)begin aw_ids[aw_count]=m_awid;aw_lens[aw_count]=m_awlen+1;aw_count=aw_count+1;end
            if(m_wvalid && m_wready)begin
                w_run=w_run+1;if(m_wlast)begin w_lens[w_count]=w_run;w_count=w_count+1;w_run=0;end
            end
            if(m_arvalid && m_arready)begin
                ar_ids[ar_count]=m_arid;ar_lens[ar_count]=m_arlen+1;ar_addrs[ar_count]=m_araddr;ar_beats[ar_count]=0;ar_count=ar_count+1;
            end
            if(m_bvalid && m_bready && b_pick>=0)b_done[b_pick]=1;
            if(m_rvalid && m_rready && r_pick>=0)begin
                ar_beats[r_pick]=ar_beats[r_pick]+1;
                if(ar_beats[r_pick]==ar_lens[r_pick])r_done[r_pick]=1;
                next_id=(ar_ids[r_pick]+1)%16;
            end
        end
    end
    reg previous_b_fire=0,previous_r_fire=0;
    always @(posedge clk)begin previous_b_fire<=m_bvalid && m_bready;previous_r_fire<=m_rvalid && m_rready;end
    always @(negedge clk)begin
        m_awready=allow_aw && (!stalls || cycle%5!=0);
        m_wready=allow_w && (!stalls || cycle%7!=0);
        m_arready=allow_ar && (!stalls || cycle%3!=0);
        s_bready=allow_b && (!stalls || cycle%4!=0);
        s_rready=allow_r && (!stalls || cycle%6!=0);
        if(!cold_rst_n)begin m_bvalid=0;m_rvalid=0;b_pick=-1;r_pick=-1;end
        else begin
            if(!m_bvalid || previous_b_fire)begin
                m_bvalid=0;b_pick=-1;
                if(inject_b)begin m_bvalid=1;m_bid=15;m_bresp=2;end
                else if(respond)begin
                    picked=-1;
                    for(integer id=15;id>=0;id=id-1)
                        for(integer n=0;n<aw_count && n<w_count;n=n+1)
                            if(picked<0 && !b_done[n] && aw_ids[n]==id && aw_lens[n]==w_lens[n])picked=n;
                    if(picked>=0)begin b_pick=picked;m_bvalid=1;m_bid=aw_ids[picked];m_bresp=aw_ids[picked]%4;end
                end
            end
            if(!m_rvalid || previous_r_fire)begin
                m_rvalid=0;r_pick=-1;
                if(inject_r)begin m_rvalid=1;m_rid=15;m_rresp=2;m_rlast=1;m_rdata=pattern(999);end
                else if(respond)begin
                    picked=-1;
                    for(integer offset=0;offset<16;offset=offset+1)begin
                        idscan=(next_id+offset)%16;
                        for(integer n=0;n<ar_count;n=n+1)
                            if(picked<0 && !r_done[n] && ar_ids[n]==idscan)picked=n;
                    end
                    if(picked>=0)begin
                        r_pick=picked;m_rvalid=1;m_rid=ar_ids[picked];m_rresp=(ar_ids[picked]+ar_beats[picked])%4;
                        m_rlast=(ar_beats[picked]+1==ar_lens[picked]);
                        m_rdata=pattern(ar_addrs[picked][31:12]^ar_addrs[picked][63:32]^(ar_beats[picked]<<8)^ar_ids[picked]);
                    end
                end
            end
        end
    end
    task step;begin @(posedge clk);#1;end endtask
    task idle(input integer n);begin repeat(n)step;end endtask
    task check(input bit good,input [767:0] msg);
        if(!good)$fatal(1,"AXI4_ADMISSION_ASSERT case=%0d cycle=%0d %0s",case_id,cycle,msg);
    endtask
    task start_case(input integer n);
        begin
            @(negedge clk);running=0;cold_rst_n=0;close_request=0;proof_request=0;proof_tag=64'hfedc000000000000+n;
            s_awvalid=0;s_wvalid=0;s_arvalid=0;
            s_awid=0;s_arid=0;s_awlen=0;s_arlen=0;s_awaddr=0;s_araddr=0;
            s_awsize=6;s_arsize=6;s_awburst=1;s_arburst=1;s_awlock=0;s_arlock=0;
            s_awprot=0;s_arprot=0;s_awcache=3;s_arcache=3;s_awregion=0;s_arregion=0;s_awqos=0;s_arqos=0;
            s_wdata=0;s_wstrb=0;s_wlast=0;m_bid=0;m_bresp=0;m_rid=0;m_rresp=0;m_rlast=0;m_rdata=0;
            allow_aw=1;allow_w=1;allow_ar=1;allow_b=1;allow_r=1;respond=1;stalls=0;inject_b=0;inject_r=0;
            idle(4);@(negedge clk);cold_rst_n=1;case_id=n;cycle=0;running=1;
            $display("AXI4_ADMISSION_START case=%0d",n);idle(2);
        end
    endtask
    task set_aw(input integer n,input integer beats);
        begin s_awvalid=1;s_awaddr=64'h100000000+4096*n;s_awid=n%16;s_awlen=beats-1;s_awsize=(beats==256)?4:6;
            s_awprot=n%8;s_awcache=3+4*(n%4);s_awregion=n%16;s_awqos=15-(n%16);end
    endtask
    task set_ar(input integer n,input integer beats);
        begin s_arvalid=1;s_araddr=64'h200000000+4096*n;s_arid=n%16;s_arlen=beats-1;s_arsize=(beats==256)?4:6;
            s_arprot=n%8;s_arcache=3+4*(n%4);s_arregion=n%16;s_arqos=15-(n%16);end
    endtask
    task set_w(input integer n,input integer beat,input integer beats);
        begin s_wvalid=1;s_wlast=(beat==beats-1);s_wdata=pattern(4096*n+beat);
            s_wstrb=(beats==256)?(64'hffff<<((beat%4)*16)):(64'haa55aa55aa55aa55^(64'h0101010101010101*beat));end
    endtask
    task take_aw;
        integer budget;bit accepted;
        begin accepted=0;budget=0;
            while(!accepted)begin @(posedge clk);accepted=s_awready;#1;budget=budget+1;check(budget<30000,"AW progress");end
            @(negedge clk);s_awvalid=0;
        end
    endtask
    task take_w;
        integer budget;bit accepted;
        begin accepted=0;budget=0;
            while(!accepted)begin @(posedge clk);accepted=s_wready;#1;budget=budget+1;check(budget<30000,"W progress");end
            @(negedge clk);s_wvalid=0;
        end
    endtask
    task take_ar;
        integer budget;bit accepted;
        begin accepted=0;budget=0;
            while(!accepted)begin @(posedge clk);accepted=s_arready;#1;budget=budget+1;check(budget<30000,"AR progress");end
            @(negedge clk);s_arvalid=0;
        end
    endtask
    task aw(input integer n,input integer beats);begin @(negedge clk);set_aw(n,beats);take_aw;end endtask
    task ar(input integer n,input integer beats);begin @(negedge clk);set_ar(n,beats);take_ar;end endtask
    task wb(input integer n,input integer beat,input integer beats);begin @(negedge clk);set_w(n,beat,beats);take_w;end endtask
    task wburst(input integer n,input integer beats);begin for(integer b=0;b<beats;b=b+1)wb(n,b,beats);end endtask
    task close_now;
        begin @(negedge clk);close_request=1;proof_request=1;step;
            check(state_debug==1,"close captures finite prefix");
            $display("AXI4_ADMISSION_EVENT case=%0d kind=CLOSE aw_left=%0d w_left=%0d ar_left=%0d",case_id,aw_left,w_left,ar_left);
            @(negedge clk);close_request=0;
        end
    endtask
    task wait_reject;
        integer budget;
        begin budget=0;while(state_debug!=2 || !drain_ack)begin step;budget=budget+1;check(budget<40000,"drain and fresh proof progress");end
            check(source_fenced && !protocol_fault,"healthy downstream closed");
            $display("AXI4_ADMISSION_EVENT case=%0d kind=PROOF cycles=%0d",case_id,cycle);
        end
    endtask
    task wait_local_idle;
        integer budget;
        begin budget=0;while(!local_idle)begin step;budget=budget+1;check(budget<40000,"local replies progress");end end
    endtask
    task finish_case(input integer faults);
        begin idle(4);check(fault_sources==faults,"exact terminal faults");
            $display("AXI4_ADMISSION_CASE_PASS case=%0d faults=%0d cycles=%0d",case_id,fault_sources,cycle);
            @(negedge clk);running=0;
        end
    endtask
    integer i,j;integer begin_aw,begin_w,begin_ar;
    reg [96:0] stable_aw;
    initial begin
        // Fill every slot. Offer33/129 remains held while closure waits for room.
        start_case(0);respond=0;
        for(i=0;i<32;i=i+1)aw(i,length(i));
        for(i=0;i<32;i=i+1)wburst(i,length(i));
        for(i=0;i<16;i=i+1)for(j=0;j<8;j=j+1)ar(16*j+i,length(j));
        check(aw_pending==32 && w_pending==32 && ar_pending==128,"full32/128 capacity");
        @(negedge clk);set_aw(32,4);set_w(32,0,4);set_ar(128,4);idle(3);
        check(!(s_awready || s_wready || s_arready),"extra offers blocked at capacity");
        close_now;check(aw_left==1 && w_left==1 && ar_left,"exact extra prefix");respond=1;stalls=1;
        fork take_aw;begin take_w;for(integer b=1;b<4;b=b+1)wb(32,b,4);end take_ar;join
        wait_reject;aw(200,4);wburst(200,4);ar(201,4);wait_local_idle;finish_case(0);
        // W first:31 complete bursts and three beats of the final4-beat burst.
        start_case(1);respond=0;
        for(i=0;i<31;i=i+1)wburst(i,length(i));for(i=0;i<3;i=i+1)wb(31,i,4);
        allow_aw=0;allow_w=0;allow_ar=0;@(negedge clk);set_aw(0,1);set_w(31,3,4);set_ar(7,16);idle(3);
        close_now;check(aw_left==32 && w_left==1 && ar_left,"W-before-AW finite group");
        allow_aw=1;allow_w=1;allow_ar=1;respond=1;stalls=1;
        fork begin take_aw;for(integer n=1;n<32;n=n+1)aw(n,n==31?4:length(n));end take_w;take_ar;join
        wait_reject;wburst(202,16);aw(202,16);ar(203,16);wait_local_idle;finish_case(0);
        // AW first:32 descriptors, then offer33 and all33 data bursts.
        start_case(2);respond=0;for(i=0;i<32;i=i+1)aw(i,length(i));
        allow_w=0;@(negedge clk);set_aw(32,4);set_w(0,0,1);idle(3);close_now;
        check(aw_left==1 && w_left==33 && !ar_left,"AW-before-W finite group");
        allow_w=1;respond=1;stalls=1;
        fork take_aw;begin take_w;for(integer n=1;n<33;n=n+1)wburst(n,n==32?4:length(n));end join
        wait_reject;aw(204,256);wburst(204,256);ar(205,256);wait_local_idle;finish_case(0);
        // All currently offered VALIDs held, then local long error bursts.
        start_case(3);allow_aw=0;allow_w=0;allow_ar=0;
        @(negedge clk);set_aw(9,1);set_w(9,0,1);set_ar(10,4);idle(3);close_now;
        check(aw_left==1 && w_left==1 && ar_left,"three held offers included");
        allow_aw=1;allow_w=1;allow_ar=1;fork take_aw;take_w;take_ar;join
        wait_reject;stalls=1;
        for(i=0;i<8;i=i+1)begin
            if(i%2)begin wburst(300+i,length(i));aw(300+i,length(i));end
            else begin aw(300+i,length(i));wburst(300+i,length(i));end
            ar(400+i,length(i));wait_local_idle;
        end
        stalls=0;allow_b=0;allow_r=0;aw(500,1);wburst(500,1);ar(501,4);idle(5);
        check(s_bvalid && s_rvalid && drain_ack && !local_idle,"section proof excludes held local replies");
        @(negedge clk);proof_tag=proof_tag+1;#1;check(!drain_ack,"new tag masks old proof");idle(6);
        check(drain_ack && s_bvalid && s_rvalid && !local_idle,"fresh proof keeps local replies");
        @(negedge clk);proof_request=0;idle(2);proof_request=1;idle(6);check(drain_ack,"same tag needs fresh request");
        allow_aw=0;allow_w=0;allow_ar=0;allow_b=1;allow_r=1;wait_local_idle;
        check(aw_count==1 && w_count==1 && ar_count==1,"closed boundary never reopens");finish_case(0);
        // Close in the middle of W while its next beat and AR are stalled.
        start_case(4);aw(4,16);for(i=0;i<3;i=i+1)wb(4,i,16);
        allow_w=0;allow_ar=0;@(negedge clk);set_w(4,3,16);set_ar(6,4);idle(3);close_now;
        check(aw_left==0 && w_left==1 && ar_left,"partial W counted once");allow_w=1;allow_ar=1;
        fork begin take_w;for(integer b=4;b<16;b=b+1)wb(4,b,16);aw(600,2);wburst(600,2);end take_ar;join
        wait_reject;wait_local_idle;check(aw_count==1 && w_count==1,"next write stays local");finish_case(0);
        // A real orphan return triggers closure while healthy B and AW are held.
        start_case(5);allow_b=0;aw(1,1);wburst(1,1);idle(5);check(s_bvalid,"healthy B held");
        allow_aw=0;@(negedge clk);set_aw(2,1);idle(3);inject_r=1;idle(5);
        check(fault_sources==2 && state_debug==1 && m_awvalid && s_bvalid,"orphan closes without cutting healthy offers");
        allow_aw=1;fork take_aw;wburst(2,1);join
        allow_b=1;idle(12);check(!drain_ack && !m_rready && !s_rvalid,"orphan held below protected consumer");finish_case(2);
        // Corrupt W inventory drops capacity while another AW is already offered.
        start_case(6);allow_b=0;aw(0,1);wburst(0,1);idle(5);aw(1,2);
        allow_aw=0;@(negedge clk);set_aw(2,1);idle(3);wburst(1,1);idle(5);
        check(fault_sources==4 && m_awvalid && s_bvalid && state_debug==1,"held AW survives capacity loss");
        allow_aw=1;take_aw;allow_b=1;idle(6);check(!s_bvalid && !drain_ack,"healthy held B completes after W fault");finish_case(4);
        // Malformed local W cannot drop an already offered local R.
        start_case(7);close_now;wait_reject;allow_r=0;ar(8,4);aw(9,2);wburst(9,1);idle(5);
        check(fault_sources==32 && s_rvalid && s_rresp==2 && !drain_ack,"local W fault retains local R");
        allow_r=1;idle(8);check(!s_rvalid && !s_bvalid,"local R completes, malformed B withheld");finish_case(32);
        // Upstream corruption cannot change the payload already presented below.
        start_case(8);allow_aw=0;@(negedge clk);set_aw(11,4);idle(3);
        stable_aw={m_awaddr,m_awid,m_awlen,m_awsize,m_awprot,m_awburst,m_awcache,m_awlock,m_awregion,m_awqos};
        @(negedge clk);s_awaddr=s_awaddr^64'h10000000000;s_awqos=s_awqos^1;idle(4);
        check(fault_sources==16 && m_awvalid && {m_awaddr,m_awid,m_awlen,m_awsize,m_awprot,m_awburst,m_awcache,m_awlock,m_awregion,m_awqos}==stable_aw,"source corruption preserves old downstream offer");
        allow_aw=1;fork take_aw;wburst(11,4);join
        idle(10);check(!drain_ack && state_debug==1,"source fault prevents certification");finish_case(16);
        // Late bus faults revoke a proof but never truncate local replies.
        start_case(9);close_now;wait_reject;allow_b=0;allow_r=0;aw(12,1);wburst(12,1);ar(13,4);idle(5);
        check(drain_ack && s_bvalid && s_rvalid,"local replies held after proof");inject_b=1;inject_r=1;idle(5);
        check(fault_sources==3 && !drain_ack && s_bvalid && s_rvalid,"late raw orphan revokes proof, preserves local replies");
        allow_b=1;allow_r=1;idle(8);check(!s_bvalid && !s_rvalid && !local_idle,"local replies retire but fault remains");finish_case(3);

        // A partial W reserves a burst even with NO WVALID at the close edge.
        start_case(10);for(i=0;i<3;i=i+1)wb(21,i,4);idle(3);
        check(!s_wvalid && w_partial==3 && aw_pending==0,"partial W gap fixture");close_now;
        check(aw_left==1 && w_left==1 && !ar_left,"partial W gap reserves missing AW and WLAST");
        fork aw(21,4);wb(21,3,4);join
        wait_reject;ar(22,4);wait_local_idle;finish_case(0);
        // A late mismatched AW drops write capacity while W is already held.
        start_case(11);wburst(23,2);wb(24,0,2);allow_w=0;
        @(negedge clk);set_w(24,1,2);idle(3);aw(23,1);idle(5);
        check(fault_sources==4 && m_wvalid && m_wlast && m_wdata==pattern(4096*24+1),"held W survives capacity loss");
        allow_w=1;take_w;idle(5);check(!drain_ack && state_debug==1,"corrupt partial inventory never certifies");finish_case(4);
        // Missing local WLAST on beat256: freeze at255 and retain healthy R.
        start_case(12);close_now;wait_reject;allow_r=0;ar(25,4);aw(26,256);
        for(i=0;i<256;i=i+1)begin
            @(negedge clk);set_w(26,i,256);s_wlast=0;take_w;
        end
        idle(4);check(fault_sources==32 && s_rvalid && !s_wready && !drain_ack,"local missing WLAST256 retains R and stops W");
        allow_r=1;idle(8);check(!s_rvalid && !s_bvalid,"healthy local R survives missing WLAST");finish_case(32);
        $display("AXI4_ADMISSION_SUITE_PASS cases=13 write_depth=32 read_total=128 native_endpoint=0");$finish;
    end
    initial begin #4000000;$fatal(1,"AXI4_ADMISSION_TIMEOUT case=%0d",case_id);end
endmodule
