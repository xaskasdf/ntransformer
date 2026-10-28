"""Generate a single-layer control bank with four explicit synthetic arithmetic peers.

The preflight and post/FFN lease are real RTL. No arithmetic accuracy or AXI
transaction throughput is claimed by this bank. AXI payloads exercise wiring.
"""
from pathlib import Path
import re
from build_ntrpu_llama32_layer_r1 import ROOT, PARENTS, render as layer_render
from build_ntrpu_input_attention_r1 import ports
from ntrpu_llama32_layer_layout_r1 import ALLOCATIONS

PEER_COMMON=r'''
    integer state=0,delay_count=0,starts=0,acks=0;
    integer cfg_delay=8,cfg_drain=5,cfg_status=0;
    reg cfg_badshape=0,cfg_foreign=0,cfg_revoke=0,cfg_block=0;
    reg [63:0] seq_reg=0,epoch_reg=0,out_reg=0,residual_reg=0,abi_reg=0;
    reg [31:0] layer_reg=0,position_reg=0;
    reg [3:0] status_reg=0;
    integer read_sent=0,read_received=0;
    always @(posedge clk)begin
        if(!rst_n)begin state<=0;starts<=0;acks<=0;delay_count<=0;status_reg<=0;read_sent<=0;read_received<=0;end
        else case(state)
            0:if(command_valid && command_ready)begin
                state<=1;delay_count<=cfg_delay;starts<=starts+1;status_reg<=0;
                seq_reg<=command_sequence;epoch_reg<=command_epoch;read_sent<=0;read_received<=0;
@CAPTURE@
            end
            1:begin
@WORK@
            end
            2:if(completion_valid && completion_ready)begin
                if(cfg_foreign)$fatal(1,"LAYER_FAIL foreign peer acknowledged");
                state<=3;delay_count<=cfg_drain;acks<=acks+1;
            end
            3:if(delay_count==0)state<=0;else delay_count<=delay_count-1;
        endcase
    end
'''


def peer(role):
    module=PARENTS[role][0];raw=(ROOT/'fpga/rtl'/f'{module}.sv').read_text();table=ports(raw)
    params=raw.split('#(',1)[1].split(') (',1)[0]
    lines=['// SYNTHETIC CONTROL PEER; never compile with real HLS/Vivado arithmetic.',f'module {module} #({params}) (']
    for i,(n,(d,w,a)) in enumerate(table.items()):
        lines.append('    '+d+' wire '+w+' '+n+' '+a+(',' if i<len(table)-1 else ''))
    lines+=[');']
    values=dict(command_ready='state==0 && !cfg_block',busy='state!=0',cancel_ready='state==1',completion_valid='state==2',
        completion_status='status_reg',completion_sequence="seq_reg ^ (cfg_foreign ? 64'h8000000000000000:64'd0)",
        completion_epoch='epoch_reg',result_valid='state==2 && status_reg==0 && !cfg_revoke')
    capture='';work='''                if(cancel_valid || expected_epoch!=epoch_reg)begin status_reg<=cancel_valid ? 7:2;state<=2;end
                else if(delay_count==0)begin status_reg<=cfg_status;state<=2;end
                else delay_count<=delay_count-1;'''
    if role=='front':
        lines += [r'''
    reg bound=0,bind_response=0;
    reg [63:0] c_base=0,c_region=0,c_bytes=0,c_seq=0,c_epoch=0;
    reg [31:0] c_capacity=0,prefix [0:15];
    reg [31:0] q=0;
    reg qvalid=0;
    reg cfg_cache_fault=0;
    integer i;
    always @(posedge clk)begin
        if(!rst_n)begin bound<=0;bind_response<=0;qvalid<=0;for(i=0;i<16;i=i+1)prefix[i]<=0;end
        else begin
            if(bind_valid && bind_ready)begin
                bound<=1;bind_response<=1;c_base<=bind_cache_base;c_region<=bind_region_base;c_bytes<=bind_region_bytes;
                c_seq<=bind_sequence;c_epoch<=bind_epoch;c_capacity<=bind_token_capacity;
                for(i=0;i<16;i=i+1)prefix[i]<=0;
            end
            if(bind_completion_valid && bind_completion_ready)bind_response<=0;
            if(state==1 && delay_count==0 && !cancel_valid && expected_epoch==epoch_reg && cfg_status==0)
                prefix[layer_reg]<=position_reg+1;
            qvalid<=result_ce && result_valid;
            if(result_ce && result_valid)q<=32'h51000000 ^ {21'd0,result_address} ^ seq_reg[31:0];
        end
    end
''']
        values.update(cache_exclusive='state!=0',cache_valid='bound',cache_metadata_fault='cfg_cache_fault',
            cache_base='c_base',cache_region_base='c_region',cache_region_bytes='c_bytes',cache_sequence='c_seq',cache_epoch='c_epoch',
            cache_capacity='c_capacity',cache_query_layer='command_layer',cache_query_valid='bound && command_layer<16',
            cache_query_committed_tokens='command_layer<16 ? prefix[command_layer]:0',
            bind_ready='state==0 && !bind_response',bind_completion_valid='bind_response',bind_completion_sequence='c_seq',bind_completion_epoch='c_epoch',
            completion_layer='layer_reg',completion_position='position_reg',completion_kv_committed='status_reg==0',completion_kv_bytes='4096',
            completion_kv_tokens='position_reg+1',completion_norm_read_beats='cfg_badshape ? 255:256',completion_norm_records='64',
            completion_phase='2',completion_matrices='3',completion_total_bytes='12288',result_q='q',result_q_valid='qvalid')
        capture='                layer_reg<=command_layer;position_reg<=command_position;'
    elif role=='projection':
        values.update(result_address='out_reg',result_bytes='cfg_badshape ? 4096:8192',
            source_result_ce='state==1 && read_sent<2048 && source_result_valid',source_result_address='read_sent[10:0]',
            source_completion_ready='state==2 && source_completion_valid')
        capture='                out_reg<=command_output_addr;'
        work=r'''
                if(source_result_ce)read_sent<=read_sent+1;
                if(source_result_q_valid)begin
                    if(source_result_q !== (32'h51000000 ^ read_received ^ seq_reg[31:0]))
                        $fatal(1,"LAYER_FAIL source RAM address/data mismatch word=%0d",read_received);
                    read_received<=read_received+1;
                end
                if(cancel_valid || expected_epoch!=epoch_reg)begin status_reg<=cancel_valid ? 7:2;state<=2;end
                else if(read_received==2048)begin status_reg<=cfg_status;state<=2;end
'''
    elif role=='post':
        capture='                residual_reg<=command_residual_address;abi_reg<=command_ffn_address;'
        values.update(completion_read_beats='cfg_badshape ? 383:384',completion_residual_beats='128',completion_ffn_beats='64',
            result_residual_address='residual_reg',result_ffn_address='abi_reg',result_residual_bytes='8192',result_ffn_bytes='4096')
    else:
        capture='                out_reg<=command_output_address;'
        values.update(completion_ffn_stages='4',completion_ffn_phase='3',completion_ffn_bytes='8192',completion_ffn_stage_bytes='155648',
            completion_read_beats='cfg_badshape ? 255:256',completion_write_beats='128',result_address='out_reg',result_bytes='8192')
    # Every downstream field has a distinct constant per peer and port. These
    # patterns test mux routing, not an AXI slave or actual memory transactions.
    marker=1+('front','projection','post','ffn').index(role)
    for i,(n,(d,w,a)) in enumerate(table.items()):
        if d!='output':continue
        assert not a
        value=values.get(n,"'0")
        if n.startswith('m_axi_'):
            width=int(w[1:].split(':')[0])+1 if w else 1
            value=f"{width}'h{((marker*53+i*7) & ((1<<width)-1)):x}"
        lines.append(f'    assign {n}={value};')
    lines += [PEER_COMMON.replace('@CAPTURE@',capture).replace('@WORK@',work),'endmodule','']
    return '\n'.join(lines)


TB_BODY=r'''
    integer seed=1,cycle=0,jobs=0,case_id=0,waits=0,mode=0,target=0;
    integer start_front,start_o,start_post,start_ffn,drain_start;
    reg [63:0] wanted_sequence,wanted_epoch;
    reg [31:0] wanted_layer,wanted_position;
    reg [3:0] wanted_status;
    reg [23:0][63:0] saved_address,saved_base,saved_bytes;
    reg injected=0;
    always #5 clk=~clk;
    always @(posedge clk)begin
        cycle<=cycle+1;
        if(rst_n && busy)begin
            `CHECK(!command_ready && !bind_ready,"ownership released while busy")
            `CHECK(cache_exclusive,"cache reservation missing")
        end
@MUX_CHECKS@
@DESCRIPTOR_CHECKS@
        if(cycle>1000000)$fatal(1,"LAYER_FAIL global test deadline");
    end
    task tick;begin @(posedge clk);#1;@(negedge clk);end endtask
    task defaults;begin
@CONFIG_DEFAULTS@
        dut.front.cfg_cache_fault=0;
        command_valid=0;cancel_valid=0;completion_ready=0;bind_valid=0;bind_completion_ready=0;
        command_sequence=64'h1234567887654321;command_epoch=64'hfedcba9876543210;
        expected_epoch=command_epoch;command_source_sequence=command_sequence;command_source_epoch=command_epoch;command_source_valid=1;
        command_lease_base=64'h8000000012340000;command_layer=jobs%16;command_position=0;command_coefficient_position=0;
@LAYOUT@
        bind_sequence=command_sequence;bind_epoch=command_epoch;bind_cache_base=(64'd25<<40)+4032;
        bind_region_base=(64'd25<<40);bind_region_bytes=262144+4032;bind_token_capacity=4;
        bind_valid=1;waits=0;
        while(!bind_ready)begin tick();waits=waits+1;`CHECK(waits<40,"bind readiness deadline") end
        tick();bind_valid=0;waits=0;
        while(!bind_completion_valid)begin tick();waits=waits+1;`CHECK(waits<40,"bind completion deadline") end
        bind_completion_ready=1;tick();bind_completion_ready=0;
    end endtask
    task admission;integer i;begin
        wanted_sequence=command_sequence;wanted_epoch=command_epoch;wanted_layer=command_layer;wanted_position=command_position;
        start_front=dut.front.starts;start_o=dut.projection.starts;start_post=dut.post.starts;start_ffn=dut.ffn.starts;
        for(i=0;i<24;i=i+1)begin saved_address[i]=addr_driver[i];saved_base[i]=base_driver[i];saved_bytes[i]=bytes_driver[i];end
        waits=0;command_valid=1;
        while(!command_ready)begin tick();waits=waits+1;`CHECK(waits<40,"admission deadline") end
        tick();command_valid=0;
        command_sequence=0;command_epoch=0;command_source_sequence=0;command_source_epoch=0;command_source_valid=0;
        command_layer=31;command_position=999;command_coefficient_position=999;command_lease_base=0;
        addr_driver='0;base_driver='0;bytes_driver='0;
        $display("LAYER_ADMIT case=%0d mode=%0d target=%0d cycle=%0d sequence=%016h epoch=%016h layer=%0d position=%0d",case_id,mode,target,cycle,wanted_sequence,wanted_epoch,wanted_layer,wanted_position);
    end endtask
    task wait_response;begin
        waits=0;injected=0;
        while(!completion_valid)begin
            if(!injected && mode==3 && completion_stage==target)begin cancel_valid=1;injected=1;end
            if(!injected && mode==4 && completion_stage==target)begin expected_epoch=wanted_epoch ^ 64'h8000000000000000;injected=1;end
            tick();waits=waits+1;
            `CHECK(waits<14000,"public response deadline")
        end
        `CHECK(completion_sequence==wanted_sequence && completion_epoch==wanted_epoch,"public identity")
        `CHECK(completion_layer==wanted_layer && completion_position==wanted_position,"public layer position")
        `CHECK(completion_status==wanted_status,"public status")
        if(wanted_status==0)begin
            `CHECK(result_valid && result_address==saved_address[23] && result_bytes==8192,"successful result descriptor")
            `CHECK(dut.front.starts==start_front+1 && dut.projection.starts==start_o+1 && dut.post.starts==start_post+1 && dut.ffn.starts==start_ffn+1,"single ordered launch")
            `CHECK(dut.projection.read_received==2048 && dut.front.acks==dut.front.starts,"front RAM consumed before early release")
            `CHECK(tail_selected==2,"FFN must retain tail bus")
        end else begin `CHECK(!result_valid,"fault exposed valid result") end
        $display("LAYER_RESULT case=%0d status=%0d stage=%0d revoked=%0d result=%0d front=%0d projection=%0d post=%0d ffn=%0d words=%0d cycle=%0d",case_id,completion_status,completion_stage,completion_revoked,result_valid,dut.front.starts-start_front,dut.projection.starts-start_o,dut.post.starts-start_post,dut.ffn.starts-start_ffn,dut.projection.read_received,cycle);
        repeat(3+seed%5)begin tick();`CHECK(completion_valid && completion_status==wanted_status,"held public response changed") end
    end endtask
    task release_job;begin
        completion_ready=1;tick();completion_ready=0;cancel_valid=0;expected_epoch=wanted_epoch;drain_start=cycle;waits=0;
        while(busy)begin
            if(dut.ffn.busy && dut.ffn.starts>start_ffn)begin `CHECK(tail_selected==2,"tail released before FFN drain") end
            tick();waits=waits+1;`CHECK(waits<100,"drain deadline")
        end
        `CHECK(!completion_valid && !result_valid && command_ready && bind_ready && !cache_exclusive,"idle after drain")
        `CHECK(dut.front.starts==dut.front.acks && dut.projection.starts==dut.projection.acks && dut.post.starts==dut.post.acks && dut.ffn.starts==dut.ffn.acks,"all and only owned peers acknowledged")
        $display("LAYER_RELEASE case=%0d drain=%0d cycle=%0d",case_id,cycle-drain_start,cycle);
        jobs=jobs+1;case_id=case_id+1;
    end endtask
    initial begin
@INITIALIZE@
        if(!$value$plusargs("SEED=%d",seed))seed=1;
        repeat(3)tick();rst_n=1;tick();
        mode=0;target=0;wanted_status=0;
        for(integer n=0;n<4;n=n+1)begin defaults();admission();wait_response();release_job();end
        // Reject each malformed allocation before any arithmetic peer starts.
        mode=1;wanted_status=1;
        for(integer n=0;n<24;n=n+1)begin
            defaults();target=n;bytes_driver[n]=1;admission();wait_response();
            `CHECK(dut.front.starts==start_front && dut.projection.starts==start_o && dut.post.starts==start_post && dut.ffn.starts==start_ffn,"preflight allowed malformed allocation")
            release_job();
        end
        // Shape certificates are checked at every boundary, including the real lease.
        mode=2;wanted_status=5;
        for(target=1;target<=4;target=target+1)begin
            defaults();case(target)
                1:dut.front.cfg_badshape=1;2:dut.projection.cfg_badshape=1;3:dut.post.cfg_badshape=1;4:dut.ffn.cfg_badshape=1;
            endcase
            admission();wait_response();`CHECK(completion_stage==target,"bad shape reached a later stage") release_job();
        end
        for(mode=3;mode<=4;mode=mode+1)begin
            wanted_status=mode==3 ? 7:2;
            for(target=1;target<=4;target=target+1)begin defaults();admission();wait_response();`CHECK(injected,"lifecycle fault not injected") release_job();end
        end
        mode=5;wanted_status=8;
        for(target=1;target<=3;target=target+1)begin
            defaults();case(target)1:dut.front.cfg_block=1;2:dut.projection.cfg_block=1;3:dut.post.cfg_block=1;endcase
            admission();wait_response();release_job();
        end
        // Held success is revoked immediately if a retained peer/cache changes.
        mode=6;wanted_status=0;
        for(target=0;target<3;target=target+1)begin
            defaults();admission();wait_response();
            case(target)0:dut.front.cfg_cache_fault=1;1:dut.projection.cfg_revoke=1;2:dut.post.cfg_revoke=1;endcase
            #1;`CHECK(!result_valid,"late fault visible through result_valid")
            tick();`CHECK(completion_revoked && completion_status==0 && !result_valid,"late fault not sticky or changed status")
            dut.front.cfg_cache_fault=0;dut.projection.cfg_revoke=0;dut.post.cfg_revoke=0;
            tick();`CHECK(!result_valid && completion_revoked,"transient late fault forgotten") release_job();
        end
        $display("LAYER_PASS seed=%0d jobs=%0d cycles=%0d arithmetic=synthetic",seed,jobs,cycle);$finish;
    end
'''


def render():
    text,proof=layer_render();public=proof['public'];lines=['`timescale 1ns/1ps',
        '`define CHECK(C,M) if((C)!==1\'b1) $fatal(1,"LAYER_FAIL case=%0d cycle=%0d %s",case_id,cycle,M);',
        'module tb_ntrpu_llama32_layer_protocol_r1;']
    init=[]
    for n,(d,w,a) in public.items():
        if a:
            assert d=='input' and a=='[0:23]'
            driver=n.replace('command_','')+'_driver'
            lines += [f'    reg [23:0][63:0] {driver};',f'    wire [63:0] {n} [0:23];',
                f'    for(genvar i_{driver}=0;i_{driver}<24;i_{driver}=i_{driver}+1)assign {n}[i_{driver}]={driver}[i_{driver}];']
            init.append(f"        {driver}='0;")
        else:
            lines.append('    '+('reg ' if d=='input' else 'wire ')+w+' '+n+';')
            if d=='input':init.append(f"        {n}='0;")
    lines += (['    ntrpu_llama32_layer_r1 #(.OWNER_TIMEOUT_CYCLES(6000)) dut(']+
        ['        .'+n+'('+n+')'+(',' if i<len(public)-1 else '') for i,n in enumerate(public)]+['    );'])
    defaults=[]
    for role in ('front','projection','post','ffn'):
        defaults.append(f'        dut.{role}.cfg_badshape=0;dut.{role}.cfg_foreign=0;dut.{role}.cfg_revoke=0;dut.{role}.cfg_block=0;dut.{role}.cfg_status=0;dut.{role}.cfg_delay=8+seed%7;dut.{role}.cfg_drain=5+seed%5;')
    layout=[]
    for i,(_,span,_) in enumerate(ALLOCATIONS):
        layout.append(f"        addr_driver[{i}]=(64'd{i+1}<<40)+4032;base_driver[{i}]=(64'd{i+1}<<40);bytes_driver[{i}]={span+4032};")
    # Full descriptor checks at actual handshakes, independent of the top's wiring generator.
    checks=[]
    def ck(condition,label):return f'            `CHECK({condition},"{label}")'
    for role in ('front','projection','post','ffn'):
        checks.append(f'        if(rst_n && dut.{role}.command_valid && dut.{role}.command_ready)begin')
        checks.append(ck(f'dut.{role}.command_sequence==wanted_sequence && dut.{role}.command_epoch==wanted_epoch',role+' admission identity'))
        if role=='front':
            checks.append(ck('dut.front.command_layer==wanted_layer && dut.front.command_position==wanted_position','captured layer position'))
            pairs=[(f'command_{field}[{i}]',kind,i) for i in range(10) for field,kind in [('addr','address'),('base','base'),('bytes','bytes')]]
        elif role=='projection':
            pairs=[(f'command_{label}_{suffix}',kind,i) for label,i in [('weight',10),('activation',11),('output',12)] for suffix,kind in [('addr','address'),('region_base','base'),('region_bytes','bytes')]]
        elif role=='post':
            pairs=[(f'command_{label}_{suffix}',kind,i) for label,i in [('input',0),('branch',12),('weight',13),('residual',14),('ffn',15)] for suffix,kind in [('address','address'),('base','base'),('bytes','bytes')]]
        else:
            pairs=[(f'command_{field}[{j}]',kind,i) for j,i in enumerate(range(15,23)) for field,kind in [('addr','address'),('base','base'),('bytes','bytes')]]
            pairs += [(f'command_{label}_{suffix}',kind,i) for label,i in [('residual',14),('output',23)] for suffix,kind in [('address','address'),('base','base'),('bytes','bytes')]]
        checks.extend(ck(f'dut.{role}.{n}==saved_{kind}[{i}]',f'{role} descriptor {n}') for n,kind,i in pairs)
        checks.append('        end')
    mux=['        if(rst_n)begin']
    for n,(d,w,a) in public.items():
        if not n.startswith('tail_axi_'):continue
        suffix=n[len('tail_axi_'):]
        if d=='output':
            expression='tail_selected==2 ? dut.ffn.m_axi_'+suffix+':tail_selected==1 ? dut.post.m_axi_'+suffix+':dut.projection.m_axi_'+suffix
            mux.append(ck(f'{n}===({expression})','tail output '+suffix))
        else:
            init.append(f"        {n}='1;")
            for owner,role in enumerate(('projection','post','ffn')):
                expression=f'(tail_selected=={owner} && {n})' if suffix.endswith(('ready','valid')) else n
                mux.append(ck(f'dut.{role}.m_axi_{suffix}===({expression})',f'tail input {role} {suffix}'))
    mux.append('        end')
    body=TB_BODY.replace('@INITIALIZE@','\n'.join(init)).replace('@CONFIG_DEFAULTS@','\n'.join(defaults)).replace('@LAYOUT@','\n'.join(layout)).replace('@MUX_CHECKS@','\n'.join(mux)).replace('@DESCRIPTOR_CHECKS@','\n'.join(checks))
    lines += [body,'endmodule','`undef CHECK','']
    return '\n'.join(lines),'\n'.join(peer(role) for role in ('front','projection','post','ffn'))


if __name__=='__main__':
    tb,peers=render()
    for name,text in [('tb_ntrpu_llama32_layer_protocol_r1.sv',tb),('ntrpu_llama32_layer_protocol_peers_r1.sv',peers)]:
        with (ROOT/'fpga/tb'/name).open('x') as f:f.write(text)
    print('Generated control bank; 46 jobs/seed, four synthetic arithmetic peers, real gate and lease.')
