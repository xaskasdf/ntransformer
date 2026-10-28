"""Compose actual normalization/QKV, lease control and RoPE/attention with preflight."""
import hashlib
import json
import re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
PARENTS={
    'fpga/rtl/ntrpu_input_qkv_real_r1.sv':'8e4787296756c1e657e6785beb6cd1af247c87d705bfcdb56575b7c9f5b16704',
    'fpga/rtl/ntrpu_qkv_rope_handoff_r1.sv':'41081796124ad5a8da5a6158861da4ffa8b4a631145c14b83344f9a9298cbd9b',
    'fpga/rtl/ntrpu_rope_attention_query_r1.sv':'ff3866d48025d625ca5e6055459f678ffe449aeb8cf1877f44fd8ea38dcac508',
}


def ports(text):
    header=text.split(') (',1)[1].split(');',1)[0]
    result={}
    for line in header.splitlines():
        line=line.strip().rstrip(',')
        if not line:continue
        m=re.fullmatch(r'(input|output)\s+(?:wire\s+)?(?:logic\s+)?(\[[0-9:]+\])?\s*(.+)',line)
        assert m,line
        direction,width,names=m.groups()
        for item in names.split(','):
            n=re.fullmatch(r'(\w+)\s*(\[[0-9:]+\])?',item.strip());assert n,item
            name,unpacked=n.groups();assert name not in result
            result[name]=(direction,width or '',unpacked or '')
    return result


BODY=r'''
    localparam logic [3:0] OK=0,BAD_COMMAND=1,BAD_IDENTITY=2,CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [3:0] {IDLE,CHECK_RANGES,CHECK_ALIASES,SOURCE_OFFER,SOURCE_WAIT,
        LEASE_OFFER,LEASE_WAIT,RESPONSE,ACK_LEASE,DRAIN_LEASE,ACK_SOURCE,DRAIN_SOURCE} state_t;
    state_t state;
    logic [9:0][63:0] address,region_base,region_size;
    logic [63:0] held_lease,held_source_sequence,held_source_epoch;
    logic held_source_valid,source_owned,lease_owned,source_cancel_sent,lease_cancel_sent,revoked,lease_seen;
    logic [31:0] held_coefficient_position,capacity,prefix,watchdog;
    logic [63:0] saved_cache_base,saved_cache_region,saved_cache_bytes;
    logic [3:0] range_index,write_index,other_index,fault,fault_now;
    logic [59:0] saved_lease;
    wire [59:0] lease_snapshot={lease_completion_status,lease_completion_source_status,lease_completion_child_status,
        lease_completion_kv_committed,lease_completion_kv_bytes,lease_completion_kv_tokens,lease_completion_revoked,lease_result_valid};
    wire owner_busy=state!=IDLE;
    wire source_identity=producer_completion_sequence==completion_sequence && producer_completion_epoch==completion_epoch;
    wire lease_identity=lease_completion_sequence==completion_sequence && lease_completion_epoch==completion_epoch &&
        lease_completion_layer==completion_layer && lease_completion_position==completion_position;
    wire cache_idle=!consumer_busy && !consumer_cache_exclusive && !consumer_completion_valid;
    wire cache_bad=owner_busy && (!cache_valid || cache_metadata_fault || cache_sequence!=completion_sequence ||
        cache_epoch!=completion_epoch || cache_base!=saved_cache_base || cache_region_base!=saved_cache_region ||
        cache_region_bytes!=saved_cache_bytes || cache_capacity!=capacity ||
        (!lease_owned && cache_query_committed_tokens!=prefix));
    wire source_bad=source_owned && !lease_owned &&
        ((producer_completion_valid && !source_identity) || (state==LEASE_OFFER && !producer_completion_valid));
    wire lease_bad=lease_owned && ((lease_completion_valid && !lease_identity) ||
        (lease_seen && state==RESPONSE && (!lease_completion_valid || lease_snapshot!=saved_lease)));
    wire waiting=state==SOURCE_OFFER || state==SOURCE_WAIT || state==LEASE_OFFER || state==LEASE_WAIT;
    wire progress=(producer_command_valid && producer_command_ready) ||
        (state==SOURCE_WAIT && producer_completion_valid && source_identity) ||
        (lease_command_valid && lease_command_ready) || (state==LEASE_WAIT && lease_completion_valid && lease_identity);
    wire timed_out=waiting && watchdog>=32'(OWNER_TIMEOUT_CYCLES-1) && !progress;
    wire [64:0] cache_limit={1'b0,saved_cache_region}+{1'b0,saved_cache_bytes};
    wire [64:0] cache_end={1'b0,saved_cache_base}+{17'd0,capacity,16'd0};
    wire [64:0] range_end={1'b0,region_base[range_index]}+{1'b0,region_size[range_index]};
    wire cache_overlap={1'b0,region_base[range_index]}<cache_limit && {1'b0,saved_cache_region}<range_end;
    wire [3:0] write_slot=write_index==0 ? 4'd2:write_index+4'd5;
    wire [64:0] write_end={1'b0,region_base[write_slot]}+{1'b0,region_size[write_slot]};
    wire [64:0] other_end={1'b0,region_base[other_index]}+{1'b0,region_size[other_index]};
    wire alias_write=write_slot!=other_index && {1'b0,region_base[write_slot]}<other_end &&
        {1'b0,region_base[other_index]}<write_end;
    function automatic logic [63:0] span(input logic [3:0] index);
        case(index)
            0,1,6:return 64'd8192;
            2:return 64'd4096;
            3:return 64'd4456448;
            4,5:return 64'd1114112;
            9:return 64'd256;
            default:return 64'd2048;
        endcase
    endfunction
    function automatic logic valid_span(input logic [63:0] addr,base,bytes,needed);
        logic [64:0] last,limit;
        last={1'b0,addr}+{1'b0,needed};limit={1'b0,base}+{1'b0,bytes};
        return addr[5:0]==0 && base[5:0]==0 && bytes[5:0]==0 && bytes!=0 &&
            !last[64] && !limit[64] && addr>=base && last<=limit;
    endfunction
    assign busy=owner_busy || producer_busy || lease_busy || consumer_busy;
    assign cache_exclusive=owner_busy || producer_busy || lease_busy || consumer_cache_exclusive;
    assign command_ready=!owner_busy && !bind_valid && cache_idle && !producer_busy &&
        !producer_completion_valid && !lease_busy && !lease_completion_valid;
    assign consumer_bind_valid=bind_valid && !owner_busy && !producer_busy && !lease_busy;
    assign bind_ready=consumer_bind_ready && !owner_busy && !producer_busy && !lease_busy;
    assign cache_query_layer=owner_busy ? completion_layer:command_layer;
    assign cancel_ready=owner_busy && state!=ACK_LEASE && state!=DRAIN_LEASE &&
        state!=ACK_SOURCE && state!=DRAIN_SOURCE && fault==OK && !revoked;
    assign producer_command_valid=state==SOURCE_OFFER && fault_now==OK;
    assign producer_command_sequence=completion_sequence;
    assign producer_command_epoch=completion_epoch;
    assign producer_command_source_valid=held_source_valid;
    assign producer_command_source_sequence=held_source_sequence;
    assign producer_command_source_epoch=held_source_epoch;
    assign producer_cancel_valid=source_owned && !lease_owned && fault_now!=OK && !source_cancel_sent && !producer_completion_valid;
    assign producer_completion_ready=lease_owned ? lease_source_completion_ready:
        (state==ACK_SOURCE && source_owned && producer_completion_valid && source_identity);
    assign lease_command_valid=state==LEASE_OFFER && fault_now==OK && producer_completion_valid && source_identity;
    assign lease_command_sequence=completion_sequence;
    assign lease_command_epoch=completion_epoch;
    assign lease_command_layer=completion_layer;
    assign lease_command_position=completion_position;
    assign lease_command_coefficient_position=held_coefficient_position;
    assign lease_command_lease_base=held_lease;
    assign lease_cancel_valid=lease_owned && fault_now!=OK && !lease_cancel_sent;
    assign lease_completion_ready=state==ACK_LEASE && lease_owned && lease_completion_valid && lease_identity;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked || (lease_owned && lease_completion_revoked);
    assign result_valid=completion_valid && completion_status==OK && !completion_revoked && fault_now==OK &&
        lease_owned && lease_completion_valid && lease_identity && lease_result_valid && !lease_bad;
    assign lease_result_ce=result_ce && result_valid;
    assign lease_result_address=result_address;
    assign result_q=lease_result_q;
    assign result_q_valid=result_valid && lease_result_q_valid;
    for(genvar i=0;i<10;i++)begin
        assign lease_command_addr[i]=address[i];
        assign lease_command_base[i]=region_base[i];
        assign lease_command_bytes[i]=region_size[i];
        if(i<9)begin
            assign producer_command_addr[i]=address[i];
            assign producer_command_base[i]=region_base[i];
            assign producer_command_bytes[i]=region_size[i];
        end
    end
    always_comb begin
        fault_now=fault;
        if(owner_busy && fault==OK)begin
            if(expected_epoch!=completion_epoch)fault_now=BAD_IDENTITY;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(cache_bad || source_bad || lease_bad)fault_now=BAD_PEER;
            else if(timed_out)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;address<=0;region_base<=0;region_size<=0;held_lease<=0;
            held_source_sequence<=0;held_source_epoch<=0;held_source_valid<=0;
            source_owned<=0;lease_owned<=0;source_cancel_sent<=0;lease_cancel_sent<=0;revoked<=0;lease_seen<=0;
            held_coefficient_position<=0;capacity<=0;prefix<=0;watchdog<=0;
            saved_cache_base<=0;saved_cache_region<=0;saved_cache_bytes<=0;
            range_index<=0;write_index<=0;other_index<=0;fault<=0;saved_lease<=0;
            completion_sequence<=0;completion_epoch<=0;completion_layer<=0;completion_position<=0;
            completion_status<=0;completion_source_status<=0;completion_child_status<=0;
            completion_kv_committed<=0;completion_kv_bytes<=0;completion_kv_tokens<=0;
            completion_norm_status<=0;completion_qkv_status<=0;completion_matrix_status<=0;completion_norm_hls_status<=0;
            completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
        end else begin
            if(waiting && !progress && fault==OK)watchdog<=watchdog+1;else watchdog<=0;
            if(owner_busy && fault_now!=OK)begin fault<=fault_now;revoked<=1;end
            if(producer_cancel_valid && producer_cancel_ready)source_cancel_sent<=1;
            if(lease_cancel_valid && lease_cancel_ready)lease_cancel_sent<=1;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;
                    held_coefficient_position<=command_coefficient_position;held_lease<=command_lease_base;
                    held_source_sequence<=command_source_sequence;held_source_epoch<=command_source_epoch;held_source_valid<=command_source_valid;
                    for(integer i=0;i<10;i++)begin address[i]<=command_addr[i];region_base[i]<=command_base[i];region_size[i]<=command_bytes[i];end
                    saved_cache_base<=cache_base;saved_cache_region<=cache_region_base;saved_cache_bytes<=cache_region_bytes;
                    capacity<=cache_capacity;prefix<=cache_query_committed_tokens;
                    source_owned<=0;lease_owned<=0;source_cancel_sent<=0;lease_cancel_sent<=0;revoked<=0;lease_seen<=0;fault<=0;
                    range_index<=0;write_index<=0;other_index<=0;saved_lease<=0;
                    completion_status<=0;completion_source_status<=0;completion_child_status<=0;
                    completion_kv_committed<=0;completion_kv_bytes<=0;completion_kv_tokens<=0;
                    completion_norm_status<=0;completion_qkv_status<=0;completion_matrix_status<=0;completion_norm_hls_status<=0;
                    completion_norm_read_beats<=0;completion_norm_records<=0;completion_phase<=0;completion_matrices<=0;completion_total_bytes<=0;
                    if(command_sequence==0 || command_epoch==0 || command_epoch!=expected_epoch || !command_source_valid ||
                        command_source_sequence!=command_sequence || command_source_epoch!=command_epoch ||
                        cache_sequence!=command_sequence || cache_epoch!=command_epoch)begin
                        completion_status<=BAD_IDENTITY;fault<=BAD_IDENTITY;state<=RESPONSE;
                    end else state<=CHECK_RANGES;
                end
                CHECK_RANGES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!cache_valid || cache_metadata_fault || !cache_query_valid || capacity==0 || capacity>131072 ||
                        completion_layer>=16 || completion_position>=capacity || completion_position!=prefix ||
                        held_coefficient_position!=completion_position || held_lease==0 || held_lease>64'hffffffffffffffe0 ||
                        !valid_span(saved_cache_base,saved_cache_region,saved_cache_bytes,64'd64) ||
                        cache_end[64] || cache_end>cache_limit ||
                        !valid_span(address[range_index],region_base[range_index],region_size[range_index],span(range_index)) || cache_overlap)begin
                        completion_status<=BAD_COMMAND;fault<=BAD_COMMAND;state<=RESPONSE;
                    end else if(range_index==9)state<=CHECK_ALIASES;
                    else range_index<=range_index+1;
                end
                CHECK_ALIASES:begin
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(alias_write)begin completion_status<=BAD_COMMAND;fault<=BAD_COMMAND;state<=RESPONSE;end
                    else if(other_index==9)begin
                        other_index<=0;
                        if(write_index==3)state<=SOURCE_OFFER;else write_index<=write_index+1;
                    end else other_index<=other_index+1;
                end
                SOURCE_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(producer_command_ready)begin source_owned<=1;state<=SOURCE_WAIT;end
                SOURCE_WAIT:if(producer_completion_valid && source_identity)begin
                    completion_source_status<=producer_completion_status;
                    completion_norm_status<=producer_completion_norm_status;completion_qkv_status<=producer_completion_qkv_status;
                    completion_matrix_status<=producer_completion_matrix_status;completion_norm_hls_status<=producer_completion_norm_hls_status;
                    completion_norm_read_beats<=producer_completion_norm_read_beats;completion_norm_records<=producer_completion_norm_records;
                    completion_phase<=producer_completion_phase;completion_matrices<=producer_completion_matrices;completion_total_bytes<=producer_completion_total_bytes;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else state<=LEASE_OFFER;
                end
                LEASE_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(lease_command_ready)begin lease_owned<=1;state<=LEASE_WAIT;end
                LEASE_WAIT:if(lease_completion_valid && lease_identity)begin
                    saved_lease<=lease_snapshot;lease_seen<=1;
                    completion_status<=fault_now==OK ? lease_completion_status:fault_now;
                    completion_source_status<=lease_completion_source_status;completion_child_status<=lease_completion_child_status;
                    completion_kv_committed<=lease_completion_kv_committed;completion_kv_bytes<=lease_completion_kv_bytes;completion_kv_tokens<=lease_completion_kv_tokens;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=lease_owned ? ACK_LEASE:ACK_SOURCE;
                ACK_LEASE:if(lease_completion_ready)state<=DRAIN_LEASE;
                DRAIN_LEASE:if(!lease_busy && !lease_completion_valid && !producer_busy && !producer_completion_valid &&
                    !consumer_busy && !consumer_completion_valid)begin source_owned<=0;lease_owned<=0;state<=IDLE;end
                ACK_SOURCE:if(!source_owned || producer_completion_ready)begin source_owned<=0;state<=DRAIN_SOURCE;end
                DRAIN_SOURCE:if(!producer_busy && !producer_completion_valid)state<=IDLE;
                default:begin fault<=BAD_PEER;revoked<=1;completion_status<=BAD_PEER;state<=RESPONSE;end
            endcase
        end
    end
'''


def render():
    tables=[]
    for name,sha in PARENTS.items():
        raw=(ROOT/name).read_bytes();assert hashlib.sha256(raw).hexdigest()==sha,name
        tables.append(ports(raw.decode()))
    p,l,c=tables
    public={n:v for n,v in l.items() if n in ('clk','rst_n','expected_epoch','cancel_valid','cancel_ready','busy','cache_exclusive') or
            n.startswith('command_') or n.startswith('completion_') or n.startswith('result_')}
    for n,v in p.items():
        if n.startswith('command_source_') or n in ('norm_tag_exhausted','qkv_tags_exhausted'):
            public[n]=v
        if n.startswith('completion_') and n not in public:public[n]=v
        if n.startswith('m_axi_'):public[n.replace('m_axi_','input_axi_')]=v
    for n,v in c.items():
        if n.startswith(('bind_','cache_','rope_axi_','v_axi_','kv_axi_')) or n=='root_cache_lock':public[n]=v
    public['cache_query_layer']=('output','[31:0]','')
    pm={n:('input_axi_'+n[6:] if n.startswith('m_axi_') else n if n in ('clk','rst_n','expected_epoch','norm_tag_exhausted','qkv_tags_exhausted') else 'producer_'+n) for n in p}
    lm={n:'lease_'+n for n in l}
    for n in ('clk','rst_n','expected_epoch'):lm[n]=n
    for n in l:
        if n.startswith('cache_'):lm[n]=n
        if n.startswith('source_'):
            q=n.removeprefix('source_')
            q={'sequence':'completion_sequence','epoch':'completion_epoch','status':'completion_status',
               'norm_status':'completion_norm_status','qkv_status':'completion_qkv_status','matrix_status':'completion_matrix_status',
               'hls_status':'completion_norm_hls_status','norm_read_beats':'completion_norm_read_beats','norm_records':'completion_norm_records',
               'phase':'completion_phase','matrices':'completion_matrices','total_bytes':'completion_total_bytes','revoked':'completion_revoked'}.get(q,q)
            lm[n]='producer_'+q
        if n.startswith('child_'):
            q=n.removeprefix('child_')
            if q in ('sequence','epoch','layer','position','status','append_status','root_status','capture_status','rope_status',
                     'rope_hls_status','rope_beats','capture_qk_words','capture_v_beats','kv_frames','q_frames','kv_bytes','kv_tokens','kv_committed','revoked'):
                q='completion_'+q
            lm[n]='consumer_'+q
    lm['source_completion_ready']='lease_source_completion_ready'
    lm['cache_exclusive']='lease_cache_exclusive'
    lm['cache_committed_tokens']='cache_query_committed_tokens'
    cm={n:n if n in ('clk','rst_n','expected_epoch') or n.startswith(('bind_','cache_','rope_axi_','v_axi_','kv_axi_')) or n=='root_cache_lock' else 'consumer_'+n for n in c}
    cm['bind_valid']='consumer_bind_valid';cm['bind_ready']='consumer_bind_ready';cm['cache_exclusive']='consumer_cache_exclusive'
    for label,index in [('q',0),('k',1),('v',2),('coefficient',3)]:
        for field,array in [('address','addr'),('base','base'),('bytes','bytes')]:cm['command_'+label+'_'+field]=f'consumer_command_{array}[{index}]'
    # Collect exact widths, array shapes and single drivers across the three children.
    signals={};drivers={}
    for table,mapping in zip(tables,(pm,lm,cm)):
        for n,(d,w,a) in table.items():
            target=mapping[n]
            if '[' in target:continue
            if target in public:assert public[target][1:]==(w,a),(n,target)
            elif target in signals:assert signals[target]==(w,a),(n,target)
            else:signals[target]=(w,a)
            if d=='output':
                assert target not in drivers,(target,drivers.get(target),n)
                drivers[target]=n
    # These are driven by owner logic rather than child outputs.
    for n in ('cache_idle',):signals.pop(n,None)
    regs={n for n,(d,w,a) in public.items() if d=='output' and n.startswith('completion_') and n not in ('completion_valid','completion_revoked')}
    lines=['// Actual embedding -> normalization/QKV -> retained RoPE/KV/32-head attention.',
        '// Preflight precedes producer admission; four independent AXI masters share external memory.',
        '// Reset must quiesce the complete hierarchy and all external AXI peers.',
        '`default_nettype none','module ntrpu_input_attention_r1 #(',
        '    parameter integer AXI_WATCHDOG_CYCLES=65536,',
        '    parameter integer HLS_WATCHDOG_CYCLES=262144,',
        '    parameter integer MATRIX_TIMEOUT_CYCLES=16777216,',
        '    parameter integer HANDOFF_TIMEOUT_CYCLES=67108864,',
        '    parameter integer OWNER_TIMEOUT_CYCLES=268435456',') (']
    for i,(n,(d,w,a)) in enumerate(public.items()):
        lines.append('    '+d+(' logic ' if n in regs else ' wire ')+(w+' ' if w else '')+n+(' '+a if a else '')+(',' if i<len(public)-1 else ''))
    lines+=[');']+['    wire '+(w+' ' if w else '')+n+(' '+a if a else '')+';' for n,(w,a) in signals.items()]
    lines.append(BODY)
    for module,instance,table,mapping,params in (
        ('ntrpu_input_qkv_real_r1','producer',p,pm,'.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.NORM_HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES),.MATRIX_TIMEOUT_CYCLES(MATRIX_TIMEOUT_CYCLES),.HANDOFF_TIMEOUT_CYCLES(HANDOFF_TIMEOUT_CYCLES)'),
        ('ntrpu_qkv_rope_handoff_r1','lease',l,lm,'.CHILD_TIMEOUT_CYCLES(HANDOFF_TIMEOUT_CYCLES)'),
        ('ntrpu_rope_attention_query_r1','consumer',c,cm,'.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)')):
        lines.append('    '+module+' #('+params+') '+instance+'(')
        lines+=['        .'+n+'('+mapping[n]+')'+(',' if i<len(table)-1 else '') for i,n in enumerate(table)]
        lines.append('    );')
    lines+=['endmodule','`default_nettype wire','']
    return '\n'.join(lines),dict(parents=PARENTS,public=public,signals=signals,producer_mapping=pm,lease_mapping=lm,consumer_mapping=cm)


if __name__=='__main__':
    text,record=render();path=ROOT/'fpga/rtl/ntrpu_input_attention_r1.sv'
    if path.exists():assert path.read_text()==text,'Refuse to overwrite an existing revision'
    else:path.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(dict(ports=len(record['public']),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
