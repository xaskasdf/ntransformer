"""Generate a complete single-layer real-arithmetic bank with causal AXI memory."""
import json
from pathlib import Path
from build_ntrpu_llama32_layer_r1 import ROOT,render as hardware
from ntrpu_llama32_layer_fixture_r1 import digest

FIXTURE=Path('D:/AI/models/ntrpu/unsloth-llama32-1b-instruct-b69aef11/llama32-layer-real-r1')
MANIFEST_SHA='045f310a0b85c016ebee24cdb78c00d4132d4cf932b9ea2829e319959ddd1ba1'

BODY=r'''
    localparam [63:0] SEQ=64'h1234567800000001,EPOCH=64'hfedcba9800000001,LEASE=64'h8000000012340000;
    reg [511:0] memory[0:FRAMES-1],golden[0:FRAMES-1];
    bit valid_frame[0:FRAMES-1],written[0:FRAMES-1];
    integer starts[0:24],sizes[0:24];bit writable[0:24];
    reg [31:0] rope_expected[0:2559],attention_expected[0:2047];
    reg [63:0] read_hash[0:4][0:24];integer region_reads[0:4][0:24];
    integer writes[0:24];integer cycle=0,seed=1,limit,public_cycle,total_writes=0;
    integer r_count[0:4],r_index[0:4],w_count[0:4],w_beats[0:4];
    integer ars[0:4],rs[0:4],aws[0:4],ws[0:4],bs[0:4],r_stalls[0:4],w_stalls[0:4];
    bit read_active[0:4],aw_active[0:4],ar_stalled[0:4],aw_stalled[0:4],w_stalled[0:4];
    reg [63:0] r_base[0:4],w_base[0:4];reg [3:0] read_id[0:4],write_id[0:4];
    reg [511:0] w_buffer[0:4][0:63];bit w_last[0:4][0:63];
    reg [88:0] old_ar[0:4],old_aw[0:4];reg [576:0] old_w[0:4];
    reg [3:0] stage_seen=0,stage_started=0,stage_acked=0;
    integer heads=0,head_acks=0,head=-1,rope_words=0,at_starts=0,at_dones=0,attention_words=0;
    integer source_reads=0;bit source_pending=0;integer source_address=0;
    bit rp_running=0,at_running=0;reg [103:0] state_seen=0;reg [63:0] output_seen=0;
    string fixture;
    always #2.5 clk=~clk;
    always @(posedge clk)cycle<=cycle+1;
    function automatic [63:0] region_address(input integer r);return (64'(r+1)<<40)+4032;endfunction
    function automatic integer region_index(input [63:0] a);
        integer r;r=int'(a>>40)-1;
        if(r<0 || r>24 || a[5:0]!=0)$fatal(1,"LREAL_FAIL unmapped or unaligned address %016h",a);
        if(a<region_address(r) || a+64>region_address(r)+64'(sizes[r]*64))$fatal(1,"LREAL_FAIL address outside payload %016h",a);
        return r;
    endfunction
    function automatic integer frame_index(input [63:0] a);
        integer r;r=region_index(a);return starts[r]+int'((a-region_address(r))/64);
    endfunction
    function automatic [511:0] read_frame(input [63:0] a);
        integer index;index=frame_index(a);
        if(!valid_frame[index])$fatal(1,"LREAL_FAIL read before actual B commit %016h",a);
        return memory[index];
    endfunction
    function automatic bit read_permission(input integer p,r);
        case(p)
            0:return r<=5;
            1:return r==6 || r==7 || r==9;
            2:return r==8;
            3:return r==24;
            4:case(tail_selected)
                0:return r==10 || r==11;
                1:return r==0 || r==12 || r==13;
                2:return r==14 || (r>=15 && r<=22);
                default:return 0;
            endcase
            default:return 0;
        endcase
    endfunction
    function automatic bit write_permission(input integer p,r);
        case(p)
            0:return r==2 || (r>=6 && r<=8);
            3:return r==24;
            4:case(tail_selected)
                0:return r==11 || r==12;
                1:return r==14 || r==15;
                2:return r>=19 && r<=23;
                default:return 0;
            endcase
            default:return 0;
        endcase
    endfunction
    // Only the actual DUT write payload is installed, and only after B accepts.
    // Expected images are assertions; read_frame never returns them.
    always @(negedge clk)begin
        for(integer p=0;p<5;p=p+1)begin
            bus_arready[p]=rst_n && !read_active[p] && (cycle+p+seed)%5!=0;
            bus_awready[p]=rst_n && !aw_active[p] && (cycle+p+seed)%4!=0;
            bus_wready[p]=rst_n && w_count[p]<64 && (cycle+p+seed)%3!=0;
            if(rst_n)begin
                if(!bus_rvalid[p] && read_active[p] && (cycle+p+seed)%4!=0)begin
                    bus_rdata[p]=read_frame(r_base[p]+64'(r_index[p]*64));bus_rid[p]=read_id[p];
                    bus_rresp[p]=0;bus_rlast[p]=r_index[p]==r_count[p]-1;bus_rvalid[p]=1;
                end
                if(!bus_bvalid[p] && aw_active[p] && w_count[p]==w_beats[p] && (cycle+p+seed)%3!=0)begin
                    bus_bid[p]=write_id[p];bus_bresp[p]=0;bus_bvalid[p]=1;
                end
            end
        end
    end
    always @(posedge clk)begin: memory_service
        integer r,index,off;reg [63:0] a;reg [511:0] data;
        if(rst_n)for(integer p=0;p<5;p=p+1)begin
            if(ar_stalled[p])`CHECK(bus_arvalid[p] && {bus_araddr[p],bus_arlen[p],bus_arsize[p],bus_arburst[p],bus_arid[p],bus_arprot[p],bus_arcache[p],bus_arlock[p]}===old_ar[p],"AR changed under backpressure")
            if(aw_stalled[p])`CHECK(bus_awvalid[p] && {bus_awaddr[p],bus_awlen[p],bus_awsize[p],bus_awburst[p],bus_awid[p],bus_awprot[p],bus_awcache[p],bus_awlock[p]}===old_aw[p],"AW changed under backpressure")
            if(w_stalled[p])`CHECK(bus_wvalid[p] && {bus_wdata[p],bus_wstrb[p],bus_wlast[p]}===old_w[p],"W changed under backpressure")
            ar_stalled[p]=bus_arvalid[p] && !bus_arready[p];aw_stalled[p]=bus_awvalid[p] && !bus_awready[p];w_stalled[p]=bus_wvalid[p] && !bus_wready[p];
            old_ar[p]={bus_araddr[p],bus_arlen[p],bus_arsize[p],bus_arburst[p],bus_arid[p],bus_arprot[p],bus_arcache[p],bus_arlock[p]};
            old_aw[p]={bus_awaddr[p],bus_awlen[p],bus_awsize[p],bus_awburst[p],bus_awid[p],bus_awprot[p],bus_awcache[p],bus_awlock[p]};
            old_w[p]={bus_wdata[p],bus_wstrb[p],bus_wlast[p]};
            if(ar_stalled[p])r_stalls[p]++;if(w_stalled[p])w_stalls[p]++;
            if(bus_arvalid[p] && bus_arready[p])begin
                a=bus_araddr[p];r=region_index(a);off=int'((a-region_address(r))/64);
                `CHECK(!read_active[p] && bus_arlen[p]<64 && bus_arsize[p]==6 && bus_arburst[p]==1,"AR protocol")
                `CHECK(off+int'(bus_arlen[p])+1<=sizes[r] && int'(a[11:0])+(int'(bus_arlen[p])+1)*64<=4096,"AR bounds/page")
                `CHECK(read_permission(p,r),"Wrong master or tail owner read")
                for(integer i=0;i<=int'(bus_arlen[p]);i++)begin `CHECK(valid_frame[starts[r]+off+i],"AR before B commit") end
                r_base[p]=a;r_count[p]=int'(bus_arlen[p])+1;r_index[p]=0;read_id[p]=bus_arid[p];read_active[p]=1;
                $display("LREAL_AR cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,ars[p],r,off*64,r_count[p]);ars[p]++;
            end
            if(bus_rvalid[p] && bus_rready[p])begin
                a=r_base[p]+64'(r_index[p]*64);r=region_index(a);data=read_frame(a);
                `CHECK(read_active[p] && bus_rdata[p]===data && bus_rid[p]==read_id[p] && bus_rresp[p]==0 && bus_rlast[p]==(r_index[p]==r_count[p]-1),"R payload/framing")
                for(integer lane=0;lane<8;lane++)read_hash[p][r]=(read_hash[p][r]^data[lane*64+:64])*64'h100000001b3;
                if(writable[r])$display("LREAL_R cycle=%0d port=%0d region=%0d offset=%0d data=%0128h",cycle,p,r,int'(a-region_address(r)),data);
                region_reads[p][r]++;rs[p]++;r_index[p]++;bus_rvalid[p]<=0;
                if(r_index[p]==r_count[p])read_active[p]=0;
            end
            if(bus_awvalid[p] && bus_awready[p])begin
                a=bus_awaddr[p];r=region_index(a);off=int'((a-region_address(r))/64);
                `CHECK(!aw_active[p] && writable[r] && write_permission(p,r),"AW ownership")
                `CHECK(bus_awlen[p]<64 && bus_awsize[p]==6 && bus_awburst[p]==1,"AW protocol")
                `CHECK(off+int'(bus_awlen[p])+1<=sizes[r] && int'(a[11:0])+(int'(bus_awlen[p])+1)*64<=4096,"AW bounds/page")
                w_base[p]=a;w_beats[p]=int'(bus_awlen[p])+1;write_id[p]=bus_awid[p];aw_active[p]=1;
                $display("LREAL_AW cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,aws[p],r,off*64,w_beats[p]);aws[p]++;
            end
            if(bus_wvalid[p] && bus_wready[p])begin
                `CHECK(w_count[p]<64 && bus_wstrb[p]==64'hffffffffffffffff,"W extent/strobe")
                w_buffer[p][w_count[p]]=bus_wdata[p];w_last[p][w_count[p]]=bus_wlast[p];w_count[p]++;ws[p]++;
                $display("LREAL_W cycle=%0d port=%0d index=%0d last=%0d data=%0128h",cycle,p,ws[p]-1,bus_wlast[p],bus_wdata[p]);
            end
            if(bus_bvalid[p] && bus_bready[p])begin
                `CHECK(aw_active[p] && w_count[p]==w_beats[p] && bus_bresp[p]==0 && bus_bid[p]==write_id[p],"B before full write")
                r=region_index(w_base[p]);off=int'((w_base[p]-region_address(r))/64);
                for(integer i=0;i<w_beats[p];i++)begin
                    index=starts[r]+off+i;data=w_buffer[p][i];
                    `CHECK(!written[index] && w_last[p][i]==(i==w_beats[p]-1),"Duplicate commit or WLAST")
                    `CHECK(data===golden[index],"Actual intermediate differs from immutable reference")
                    memory[index]=data;valid_frame[index]=1;written[index]=1;writes[r]++;total_writes++;
                end
                $display("LREAL_B cycle=%0d port=%0d index=%0d region=%0d offset=%0d beats=%0d",cycle,p,bs[p],r,off*64,w_beats[p]);
                bs[p]++;aw_active[p]=0;w_count[p]=0;bus_bvalid[p]<=0;
            end
        end
    end
`define RP dut.front.consumer.front.rope
`define AT dut.front.consumer.attention.engine
    always @(posedge clk)begin: observe_real_arithmetic
        if(rst_n)begin
            if(busy)begin `CHECK(!command_ready && !bind_ready && cache_exclusive,"Global ownership lost") end
            if(`RP.hls_start && !rp_running)begin
                `CHECK(rs[1]==164 && rope_words==0,"RoPE must consume all actual Q/K/coefficient reads")
                rp_running=1;$display("LREAL_ROPE_START cycle=%0d",cycle);
            end
            if(`RP.hls_output_ce && `RP.hls_output_we)begin
                `CHECK(rp_running && `RP.hls_output_address==rope_words && rope_words<2560,"RoPE output order")
                `CHECK(`RP.hls_output_d===rope_expected[rope_words],"Real RoPE arithmetic")
                $display("LREAL_ROPE cycle=%0d index=%0d data=%08h",cycle,rope_words,`RP.hls_output_d);rope_words++;
            end
            if(`RP.hls_done && rp_running)begin
                `CHECK(rope_words==2560 && `RP.hls_ready && `RP.hls_return==0,"RoPE completion")
                rp_running=0;$display("LREAL_ROPE_DONE cycle=%0d",cycle);
            end
            if(dut.front.consumer.attention.head_command_valid && dut.front.consumer.attention.head_command_ready)begin
                `CHECK(heads==head_acks && heads<32 && dut.front.consumer.attention.head_command_head==heads,"Attention head ownership/order")
                head=heads;heads++;
                `CHECK(dut.front.consumer.attention.head_command_sequence==SEQ && dut.front.consumer.attention.head_command_epoch==EPOCH,"Attention head identity")
                `CHECK(dut.front.consumer.attention.head_command_lease==LEASE+64'(head) && cache_query_committed_tokens==1,"Attention committed prefix/lease")
                $display("LREAL_HEAD cycle=%0d head=%0d",cycle,head);
            end
            if(`AT.hls_start && !at_running)begin
                `CHECK(!at_running && heads==head_acks+1 && `AT.hls_tile_start==0 && `AT.hls_tile_count==1,"Attention tile ownership")
                at_running=1;at_starts++;state_seen=0;output_seen=0;
            end
            if(`AT.hls_state_out_ce && `AT.hls_state_out_we)begin
                `CHECK(at_running && `AT.hls_state_out_address<104 && !state_seen[`AT.hls_state_out_address],"Unique attention state word")
                state_seen[`AT.hls_state_out_address]=1;
                $display("LREAL_STATE cycle=%0d head=%0d index=%0d data=%016h",cycle,head,`AT.hls_state_out_address,`AT.hls_state_out_d);
            end
            if(`AT.hls_output_ce && `AT.hls_output_we)begin
                `CHECK(at_running && `AT.hls_output_address<64 && !output_seen[`AT.hls_output_address],"Unique attention output")
                `CHECK(`AT.hls_output_d===attention_expected[head*64+int'(`AT.hls_output_address)],"Attention FP32 bytes for downstream quantization")
                output_seen[`AT.hls_output_address]=1;attention_words++;
                $display("LREAL_ATTENTION cycle=%0d head=%0d index=%0d data=%08h",cycle,head,`AT.hls_output_address,`AT.hls_output_d);
            end
            if(`AT.hls_done && at_running)begin
                `CHECK((&state_seen) && (&output_seen) && `AT.hls_return==0 && `AT.hls_ready,"Complete attention state/output")
                at_running=0;at_dones++;
            end
            if(dut.front.consumer.attention.head_completion_valid && dut.front.consumer.attention.head_completion_ready)begin
                `CHECK(!at_running && heads==head_acks+1,"Attention head drain")head_acks++;
            end
            if(dut.front_result_ce)begin
                `CHECK(!source_pending && dut.front_result_valid && dut.front_completion_valid && dut.front_result_address==source_reads,"Projection source RAM ownership/order")
                source_pending=1;source_address=int'(dut.front_result_address);
            end
@STAGE_OBSERVERS@
            if(cycle%1000000==0)$display("LREAL_PROGRESS cycle=%0d stage=%0d writes=%0d",cycle,completion_stage,total_writes);
            `CHECK(cycle<40000000,"global cycle deadline")
            #1;
            if(source_pending)begin
                `CHECK(dut.front_result_q_valid && dut.front_result_q===attention_expected[source_address],"Actual attention RAM reaches O")
                $display("LREAL_SOURCE cycle=%0d index=%0d data=%08h",cycle,source_reads,dut.front_result_q);source_reads++;source_pending=0;
            end
        end
    end
    task tick;begin @(posedge clk);#1;@(negedge clk);#0.1;end endtask
    initial begin
@INITIALIZE@
        `CHECK($value$plusargs("FIXTURE=%s",fixture),"Missing frozen fixture")
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        for(integer p=0;p<5;p++)begin
            read_active[p]=0;aw_active[p]=0;r_count[p]=0;r_index[p]=0;w_count[p]=0;w_beats[p]=0;
            ars[p]=0;rs[p]=0;aws[p]=0;ws[p]=0;bs[p]=0;r_stalls[p]=0;w_stalls[p]=0;
            ar_stalled[p]=0;aw_stalled[p]=0;w_stalled[p]=0;
            for(integer r=0;r<25;r++)begin read_hash[p][r]=64'hcbf29ce484222325;region_reads[p][r]=0;end
        end
@LOAD_REGIONS@
        $readmemh({fixture,"/rope.hex"},rope_expected);$readmemh({fixture,"/attention.hex"},attention_expected);
        repeat(4)tick();rst_n=1;tick();
        expected_epoch=EPOCH;bind_sequence=SEQ;bind_epoch=EPOCH;bind_cache_base=region_address(24);
        bind_region_base=64'd25<<40;bind_region_bytes=262144+4032;bind_capacity=4;bind_valid=1;limit=0;
        while(!bind_ready)begin tick();limit++;`CHECK(limit<100,"bind readiness deadline") end
        tick();bind_valid=0;limit=0;
        while(!bind_completion_valid)begin tick();limit++;`CHECK(limit<100,"bind completion deadline") end
        `CHECK(bind_completion_status==0 && cache_valid && cache_capacity==4,"Bind succeeded")
        bind_completion_ready=1;tick();bind_completion_ready=0;
        command_sequence=SEQ;command_epoch=EPOCH;command_source_sequence=SEQ;command_source_epoch=EPOCH;
        command_source_valid=1;command_layer=0;command_position=0;command_coefficient_position=0;command_lease_base=LEASE;
@DESCRIPTORS@
        command_valid=1;limit=0;
        while(!command_ready)begin tick();limit++;`CHECK(limit<100,"admission deadline") end
        $display("LREAL_ADMIT cycle=%0d sequence=%016h epoch=%016h layer=0 position=0 token=0",cycle+1,SEQ,EPOCH);
        tick();command_valid=0;command_sequence=0;command_epoch=0;command_layer=31;command_position=999;
        command_source_valid=0;command_source_sequence=0;command_source_epoch=0;command_coefficient_position=999;command_lease_base=0;
        drive_command_addr=0;drive_command_base=0;drive_command_bytes=0;
        while(!completion_valid)tick();
        `CHECK(completion_status==0 && !completion_revoked && completion_sequence==SEQ && completion_epoch==EPOCH,"Whole layer status/identity")
        `CHECK(result_valid && result_address==region_address(23) && result_bytes==8192 && completion_stage==4,"Whole layer result")
        `CHECK(stage_started==15 && stage_seen==15 && stage_acked==1,"Retain tail and release consumed front")
        `CHECK(heads==32 && head_acks==32 && at_starts==32 && at_dones==32 && attention_words==2048 && source_reads==2048 && rope_words==2560,"Full source/attention coverage")
        `CHECK(cache_query_layer==0 && cache_query_committed_tokens==1 && total_writes==3264,"Complete memory/cache publication")
        for(integer r=0;r<24;r++)if(writable[r])begin
            `CHECK(writes[r]==sizes[r],"Complete written allocation")
            for(integer i=0;i<sizes[r];i++)begin `CHECK(written[starts[r]+i] && memory[starts[r]+i]===golden[starts[r]+i],"Committed final image") end
        end
        for(integer p=0;p<5;p++)begin
            `CHECK(!read_active[p] && !aw_active[p] && w_count[p]==0 && !bus_rvalid[p] && !bus_bvalid[p],"All memory masters drained")
            $display("LREAL_PORT port=%0d ar=%0d r=%0d aw=%0d w=%0d b=%0d ar_stalls=%0d w_stalls=%0d",p,ars[p],rs[p],aws[p],ws[p],bs[p],r_stalls[p],w_stalls[p]);
            for(integer r=0;r<25;r++)$display("LREAL_READ port=%0d region=%0d beats=%0d hash=%016h",p,r,region_reads[p][r],read_hash[p][r]);
        end
        public_cycle=cycle;$display("LREAL_COMPLETE cycle=%0d status=%0d sequence=%016h epoch=%016h address=%016h bytes=%0d prefix=%0d",cycle,completion_status,completion_sequence,completion_epoch,result_address,result_bytes,cache_query_committed_tokens);
        repeat(7)begin tick();`CHECK(completion_valid && result_valid && !completion_revoked && !command_ready,"Held result/ownership") end
        completion_ready=1;$display("LREAL_PUBLIC_ACK cycle=%0d",cycle+1);tick();completion_ready=0;limit=0;
        while(busy)begin tick();limit++;`CHECK(limit<100,"Final drain deadline") end
        `CHECK(stage_acked==15 && command_ready && bind_ready && !cache_exclusive && !completion_valid && !result_valid,"Whole layer reusable after drain")
        $display("LREAL_PASS seed=%0d jobs=1 cycles=%0d writes=%0d heads=%0d source_words=%0d",seed,cycle,total_writes,heads,source_reads);$finish;
    end
`undef RP
`undef AT
endmodule
`undef CHECK
`default_nettype wire
'''

def render():
    assert digest(FIXTURE/'manifest.json')==MANIFEST_SHA
    m=json.loads((FIXTURE/'manifest.json').read_text());_,proof=hardware();public=proof['public']
    lines=['`timescale 1ns/1ps','`default_nettype none',
        '`define CHECK(C,M) if((C)!==1\'b1) $fatal(1,"LREAL_FAIL cycle=%0d %s",cycle,M);',
        'module tb_ntrpu_llama32_layer_real_r1;',f"    localparam integer FRAMES={m['memory_frames']};"]
    init=[];bus={};prefixes=['input','rope','v','kv','tail']
    for n,(d,w,a) in public.items():
        if a:
            assert a=='[0:23]';lines += [f'    wire {w} {n} {a};',f'    reg [23:0][63:0] drive_{n};',
                f'    for(genvar g_{n}=0;g_{n}<24;g_{n}++)assign {n}[g_{n}]=drive_{n}[g_{n}];'];init.append(f'        drive_{n}=0;')
        elif '_axi_' in n:
            prefix,suffix=n.split('_axi_');p=prefixes.index(prefix);bus[suffix]=(d,w)
            lines.append(f'    wire {w} {n};')
            lines.append(f'    assign '+(f'bus_{suffix}[{p}]={n};' if d=='output' else f'{n}=bus_{suffix}[{p}];'))
        else:
            lines.append(f'    '+('reg' if d=='input' else 'wire')+f' {w} {n};')
            if d=='input':init.append(f'        {n}=0;')
    for suffix,(d,w) in bus.items():
        lines.append('    '+('wire' if d=='output' else 'reg')+f' {w} bus_{suffix}[0:4];')
        for p,prefix in enumerate(prefixes):
            if prefix+'_axi_'+suffix not in public and d=='output':lines.append(f"    assign bus_{suffix}[{p}]='0;")
            if d=='input':init.append(f"        bus_{suffix}[{p}]='0;")
    lines+=['    ntrpu_llama32_layer_r1 dut(']+['        .'+n+'('+n+')'+(',' if i<len(public)-1 else '') for i,n in enumerate(public)]+['    );']
    load=[];desc=[]
    for r in m['regions']:
        i=r['index'];start=r['memory_start'];end=start+r['frames']-1
        load.append(f"        starts[{i}]={start};sizes[{i}]={r['frames']};writable[{i}]={int(r['writable'])};writes[{i}]=0;")
        load.append(f'        $readmemh({{fixture,"/{r["file"]}"}},'+('golden' if r['writable'] else 'memory')+f',{start},{end});')
        load.append(f'        for(integer i={start};i<={end};i++)begin valid_frame[i]={int(not r["writable"])};written[i]=0;end')
        if i<24:desc.append(f"        drive_command_addr[{i}]=64'h{r['address']:016x};drive_command_base[{i}]=64'h{r['base']:016x};drive_command_bytes[{i}]={r['region_bytes']};")
    observe=[]
    for i,role in enumerate(('front','projection','post','ffn')):
        observe += [f'            if(dut.{role}_command_valid && dut.{role}_command_ready)begin',
            f'                `CHECK(!stage_started[{i}],"Stage admitted twice")stage_started[{i}]=1;',
            f'                $display("LREAL_START cycle=%0d stage={i+1}",cycle);end',
            f'            if(dut.{role}_completion_valid && !stage_seen[{i}])begin',
            f'                `CHECK(stage_started[{i}] && dut.{role}_completion_status==0 && dut.{role}_completion_sequence==SEQ && dut.{role}_completion_epoch==EPOCH,"Stage completion certificate")stage_seen[{i}]=1;',
            f'                $display("LREAL_DONE cycle=%0d stage={i+1}",cycle);end',
            f'            if(dut.{role}_completion_valid && dut.{role}_completion_ready)begin',
            f'                `CHECK(stage_seen[{i}] && !stage_acked[{i}],"Unowned stage ACK")stage_acked[{i}]=1;',
            (f'                `CHECK(source_reads==2048,"Front ACK before source RAM read")' if i==0 else f'                `CHECK(cycle>public_cycle && !completion_valid,"Tail ACK before public retirement")'),
            f'                $display("LREAL_ACK cycle=%0d stage={i+1}",cycle);end']
    body=BODY.replace('@INITIALIZE@','\n'.join(init)).replace('@LOAD_REGIONS@','\n'.join(load)).replace('@DESCRIPTORS@','\n'.join(desc)).replace('@STAGE_OBSERVERS@','\n'.join(observe))
    return '\n'.join(lines)+'\n'+body

if __name__=='__main__':
    p=ROOT/'fpga/tb/tb_ntrpu_llama32_layer_real_r1.sv'
    with p.open('x') as f:f.write(render())
    print(digest(p))
