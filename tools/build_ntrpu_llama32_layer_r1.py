"""Compose the complete single-layer datapath around one input/attention cache."""
import hashlib
import json
import re
from pathlib import Path
from build_ntrpu_input_attention_r1 import ports

ROOT=Path(__file__).resolve().parents[1]
PARENTS={
    'gate':('ntrpu_llama32_layer_preflight_r1','50032aa011be991a719c923582bb9851383b0a6613db7f83b6cbd2ca37d9c9b4'),
    'front':('ntrpu_input_attention_r1','20092111a632c735e8a0885891c242cc2b38271821923cfba7719be542f3053c'),
    'projection':('ntrpu_attention_output_path_r1','feda7f0ad2c7b52fec9e5c7c2c607992df9152d75204347d53b3b6976e0b9453'),
    'post':('ntrpu_post_attention_real_r2','e92ce4f9e68e3fbf3e5b256e7c5dafed2764b7cf1b0ec373ce2c43359962a0f9'),
    'lease':('ntrpu_post_ffn_handoff_r2','dc1f8525d7277095178040e5023cca7a70bf0f661b1a4c06148a4dc9c86d8ce5'),
    'ffn':('ntrpu_llama32_ffn_residual_r3','cc2ae32db00bdbA218d46cff43ebd9392e5049749b855900fde58b502d4f67c4'.lower()),
}

BODY=r'''
    localparam logic [3:0] OK=0,BAD_COMPLETION=5,CANCELLED=7,TIMED_OUT=8,BAD_PEER=11;
    typedef enum logic [4:0] {IDLE,WAIT_GATE,FRONT_OFFER,WAIT_FRONT,O_OFFER,WAIT_O,
        POST_OFFER,WAIT_POST,LEASE_OFFER,WAIT_LEASE,RESPONSE,ACK_LEASE,DRAIN_LEASE,
        ACK_POST,DRAIN_POST,ACK_O,DRAIN_O,ACK_FRONT,DRAIN_FRONT} state_t;
    state_t state;
    logic gate_passed,front_owned,projection_owned,post_owned,lease_owned,front_released;
    logic front_seen,projection_seen,post_seen,lease_seen,revoked;
    logic [3:0] fault,fault_now;
    logic [1:0] tail_owner;
    logic [31:0] watchdog,saved_capacity,prefix_after_front;
    logic [63:0] saved_cache_base,saved_cache_region,saved_cache_bytes,saved_cache_sequence,saved_cache_epoch;
    wire owner_busy=state!=IDLE;
    wire all_idle=!front_busy && !front_completion_valid && !front_cache_exclusive &&
        !projection_busy && !projection_completion_valid && !post_busy && !post_completion_valid &&
        !lease_busy && !lease_completion_valid && !ffn_busy && !ffn_completion_valid && !gate_busy;
    wire admission_allowed=state==IDLE && !bind_valid && !bind_completion_valid && all_idle;
    wire gate_identity=held_sequence==completion_sequence && held_epoch==completion_epoch;
    wire front_identity=front_completion_sequence==completion_sequence && front_completion_epoch==completion_epoch &&
        front_completion_layer==completion_layer && front_completion_position==completion_position;
    wire projection_identity=projection_completion_sequence==completion_sequence && projection_completion_epoch==completion_epoch;
    wire post_identity=post_completion_sequence==completion_sequence && post_completion_epoch==completion_epoch;
    wire lease_identity=lease_completion_sequence==completion_sequence && lease_completion_epoch==completion_epoch;
    wire ffn_identity=ffn_completion_sequence==completion_sequence && ffn_completion_epoch==completion_epoch;
    wire cache_changed=gate_passed && (!cache_valid || cache_metadata_fault || !cache_query_valid ||
        cache_base!=saved_cache_base || cache_region_base!=saved_cache_region || cache_region_bytes!=saved_cache_bytes ||
        cache_sequence!=saved_cache_sequence || cache_epoch!=saved_cache_epoch || cache_capacity!=saved_capacity ||
        cache_query_layer!=completion_layer ||
        (!front_owned && cache_query_committed_tokens!=initial_prefix) ||
        (front_owned && !front_seen && cache_query_committed_tokens!=initial_prefix &&
            cache_query_committed_tokens!=initial_prefix+1) ||
        (front_seen && cache_query_committed_tokens!=prefix_after_front));
    wire front_success=front_completion_status==OK && !front_completion_revoked && front_result_valid &&
        front_completion_source_status==OK && front_completion_child_status==OK &&
        front_completion_norm_status==OK && front_completion_qkv_status==OK && front_completion_matrix_status==OK &&
        front_completion_norm_hls_status==0 && front_completion_norm_read_beats==256 && front_completion_norm_records==64 &&
        front_completion_phase==2 && front_completion_matrices==3 && front_completion_total_bytes==12288 &&
        front_completion_kv_committed && front_completion_kv_bytes==4096 &&
        front_completion_kv_tokens==completion_position+1 && cache_query_committed_tokens==completion_position+1;
    wire projection_success=projection_completion_status==OK && projection_completion_hls_status==0 &&
        projection_result_valid && projection_result_address==address[12] && projection_result_bytes==8192;
    wire post_success=post_completion_status==OK && post_completion_hls_status==0 && post_result_valid &&
        post_completion_read_beats==384 && post_completion_residual_beats==128 && post_completion_ffn_beats==64 &&
        post_result_residual_address==address[14] && post_result_residual_bytes==8192 &&
        post_result_ffn_address==address[15] && post_result_ffn_bytes==4096;
    wire lease_success=lease_completion_status==OK && lease_result_valid &&
        lease_result_address==address[23] && lease_result_bytes==8192;
    wire foreign_peer=(front_owned && !front_released && front_completion_valid && !front_identity) ||
        (projection_owned && projection_completion_valid && !projection_identity) ||
        (post_owned && post_completion_valid && !post_identity) ||
        (lease_owned && lease_completion_valid && !lease_identity) ||
        (lease_ffn_bus_owned && ffn_completion_valid && !ffn_identity);
    wire changed_peer=(front_seen && !front_released &&
            (!front_completion_valid || front_snapshot!=saved_front_snapshot)) ||
        (projection_seen && projection_owned && state!=DRAIN_O && state!=ACK_FRONT && state!=DRAIN_FRONT &&
            (!projection_completion_valid || projection_snapshot!=saved_projection_snapshot)) ||
        (post_seen && post_owned && !lease_owned && state!=DRAIN_POST &&
            (!post_completion_valid || post_snapshot!=saved_post_snapshot)) ||
        (lease_seen && lease_owned && state==RESPONSE &&
            (!lease_completion_valid || lease_snapshot!=saved_lease_snapshot));
    wire waiting=state==FRONT_OFFER || state==WAIT_FRONT || state==O_OFFER || state==WAIT_O ||
        state==POST_OFFER || state==WAIT_POST || state==LEASE_OFFER || state==WAIT_LEASE;
    wire progress=(front_command_valid && front_command_ready) ||
        (state==WAIT_FRONT && front_completion_valid && front_identity) ||
        (projection_command_valid && projection_command_ready) ||
        (state==WAIT_O && projection_completion_valid && projection_identity) ||
        (post_command_valid && post_command_ready) ||
        (state==WAIT_POST && post_completion_valid && post_identity) ||
        (lease_command_valid && lease_command_ready) ||
        (state==WAIT_LEASE && lease_completion_valid && lease_identity);
    wire timed_out=waiting && watchdog>=32'(OWNER_TIMEOUT_CYCLES-1) && !progress;
    wire retiring=state>=ACK_LEASE;
    // Only stored peer faults feed child cancellation. Keep result/cancel feedback out.
    wire cancel_children=owner_busy && (fault!=OK || revoked || cancel_valid || expected_epoch!=completion_epoch);
    wire lease_can_release=post_completion_valid && post_identity &&
        (!lease_ffn_bus_owned || (ffn_completion_valid && ffn_identity));
    assign busy=owner_busy || !all_idle;
    assign cache_exclusive=owner_busy || !all_idle;
    assign command_ready=admission_allowed && gate_command_ready;
    assign gate_command_valid=command_valid && admission_allowed;
    assign gate_command_layer=owner_busy ? held_layer:command_layer;
    assign gate_cancel_valid=state==WAIT_GATE && cancel_children;
    assign gate_completion_ready=state==WAIT_GATE && gate_completion_valid && gate_identity;
    assign front_bind_valid=bind_valid && state==IDLE && all_idle;
    assign bind_ready=front_bind_ready && state==IDLE && all_idle;
    assign cancel_ready=owner_busy && !retiring && fault==OK && !revoked;
    assign completion_valid=state==RESPONSE;
    assign completion_revoked=revoked;
    assign result_valid=completion_valid && completion_status==OK && !revoked && fault_now==OK &&
        lease_owned && lease_completion_valid && lease_identity && lease_success;
    assign result_address=result_valid ? address[23]:64'd0;
    assign result_bytes=result_valid ? 14'd8192:14'd0;
    assign front_command_valid=state==FRONT_OFFER && fault_now==OK;
    assign projection_command_valid=state==O_OFFER && fault_now==OK && front_completion_valid && front_identity && front_success;
    assign post_command_valid=state==POST_OFFER && fault_now==OK && projection_completion_valid && projection_identity && projection_success;
    assign lease_command_valid=state==LEASE_OFFER && fault_now==OK && post_completion_valid && post_identity && post_success;
    assign front_cancel_valid=front_owned && !front_released && cancel_children;
    assign projection_cancel_valid=projection_owned && cancel_children;
    assign post_cancel_valid=post_owned && cancel_children;
    assign lease_cancel_valid=lease_owned && cancel_children;
    assign front_completion_ready=front_owned && !front_released && front_completion_valid && front_identity &&
        (projection_owned ? projection_source_completion_ready:state==ACK_FRONT);
    assign projection_completion_ready=state==ACK_O && projection_owned && projection_completion_valid && projection_identity;
    assign post_completion_ready=lease_owned ? lease_source_completion_ready:
        (state==ACK_POST && post_owned && post_completion_valid && post_identity);
    assign lease_completion_ready=state==ACK_LEASE && lease_owned && lease_completion_valid && lease_identity && lease_can_release;
    assign projection_source_completion_valid=front_owned && !front_released && front_completion_valid;
    assign projection_source_completion_status=front_completion_status;
    assign projection_source_sequence=front_completion_sequence;
    assign projection_source_epoch=front_completion_epoch;
    assign projection_source_result_valid=front_result_valid && front_identity && !front_released;
    assign front_result_ce=projection_owned && !front_released && front_identity && projection_source_result_ce;
    assign front_result_address=projection_source_result_address;
    assign projection_source_result_q=front_result_q;
    assign projection_source_result_q_valid=front_result_q_valid && !front_released;
    assign post_command_source_valid=projection_completion_valid && projection_identity && projection_success;
    assign post_command_source_sequence=projection_completion_sequence;
    assign post_command_source_epoch=projection_completion_epoch;
    assign lease_source_completion_valid=post_completion_valid;
    assign lease_source_completion_status=post_completion_status;
    assign lease_source_sequence=post_completion_sequence;
    assign lease_source_epoch=post_completion_epoch;
    assign lease_source_result_valid=post_result_valid;
    assign lease_source_residual_address=post_result_residual_address;
    assign lease_source_ffn_address=post_result_ffn_address;
    assign lease_source_residual_bytes=post_result_residual_bytes;
    assign lease_source_ffn_bytes=post_result_ffn_bytes;
    assign lease_source_root_hls_status=projection_completion_hls_status;
    assign lease_source_post_hls_status=post_completion_hls_status;
    assign lease_source_post_read_beats=post_completion_read_beats;
    assign lease_source_residual_beats=post_completion_residual_beats;
    assign lease_source_ffn_beats=post_completion_ffn_beats;
    assign tail_selected=tail_owner;
    always_comb begin
        fault_now=fault;
        if(owner_busy && fault==OK && !revoked)begin
            if(expected_epoch!=completion_epoch)fault_now=4'd2;
            else if(cancel_valid && cancel_ready)fault_now=CANCELLED;
            else if(cache_changed || foreign_peer || changed_peer)fault_now=BAD_PEER;
            else if(timed_out)fault_now=TIMED_OUT;
        end
    end
    always_ff @(posedge clk)begin
        if(!rst_n)begin
            state<=IDLE;gate_passed<=0;front_owned<=0;projection_owned<=0;post_owned<=0;lease_owned<=0;
            front_released<=0;front_seen<=0;projection_seen<=0;post_seen<=0;lease_seen<=0;revoked<=0;
            fault<=OK;tail_owner<=0;watchdog<=0;saved_capacity<=0;prefix_after_front<=0;
            saved_cache_base<=0;saved_cache_region<=0;saved_cache_bytes<=0;saved_cache_sequence<=0;saved_cache_epoch<=0;
            completion_sequence<=0;completion_epoch<=0;completion_layer<=0;completion_position<=0;completion_status<=OK;completion_stage<=0;
            saved_front_snapshot<=0;saved_projection_snapshot<=0;saved_post_snapshot<=0;saved_lease_snapshot<=0;
@RESET_DIAGNOSTICS@
        end else begin
            if(waiting && !progress && fault==OK)watchdog<=watchdog+1;else watchdog<=0;
            if(owner_busy && fault_now!=OK)begin fault<=fault_now;revoked<=1;end
            if(front_completion_valid && front_completion_ready)front_released<=1;
            if(ffn_command_valid && ffn_command_ready)tail_owner<=2;
            case(state)
                IDLE:if(command_valid && command_ready)begin
                    completion_sequence<=command_sequence;completion_epoch<=command_epoch;
                    completion_layer<=command_layer;completion_position<=command_position;completion_status<=OK;completion_stage<=0;
                    saved_cache_base<=cache_base;saved_cache_region<=cache_region_base;saved_cache_bytes<=cache_region_bytes;
                    saved_cache_sequence<=cache_sequence;saved_cache_epoch<=cache_epoch;saved_capacity<=cache_capacity;
                    prefix_after_front<=cache_query_committed_tokens;gate_passed<=0;
                    front_owned<=0;projection_owned<=0;post_owned<=0;lease_owned<=0;front_released<=0;
                    front_seen<=0;projection_seen<=0;post_seen<=0;lease_seen<=0;revoked<=0;fault<=OK;tail_owner<=0;
                    saved_front_snapshot<=0;saved_projection_snapshot<=0;saved_post_snapshot<=0;saved_lease_snapshot<=0;
@RESET_DIAGNOSTICS@
                    state<=WAIT_GATE;
                end
                WAIT_GATE:if(gate_completion_ready)begin
                    completion_preflight_status<=gate_completion_status;
                    completion_preflight_region<=gate_completion_region;completion_preflight_other_region<=gate_completion_other_region;
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(!gate_result_valid)begin completion_status<=gate_completion_status==OK ? BAD_PEER:gate_completion_status;state<=RESPONSE;end
                    else begin gate_passed<=1;state<=FRONT_OFFER;end
                end
                FRONT_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(front_command_ready)begin front_owned<=1;completion_stage<=1;state<=WAIT_FRONT;end
                WAIT_FRONT:if(front_completion_valid && front_identity)begin
                    front_seen<=1;saved_front_snapshot<=front_snapshot;prefix_after_front<=cache_query_committed_tokens;
@CAPTURE_FRONT@
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(front_completion_status!=OK)begin completion_status<=front_completion_status;state<=RESPONSE;end
                    else if(!front_success)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else state<=O_OFFER;
                end
                O_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(projection_command_ready)begin projection_owned<=1;completion_stage<=2;state<=WAIT_O;end
                WAIT_O:if(projection_completion_valid && projection_identity)begin
                    projection_seen<=1;saved_projection_snapshot<=projection_snapshot;
@CAPTURE_PROJECTION@
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(projection_completion_status!=OK)begin completion_status<=projection_completion_status;state<=RESPONSE;end
                    else if(!projection_success)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else state<=POST_OFFER;
                end
                POST_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(post_command_ready)begin post_owned<=1;tail_owner<=1;completion_stage<=3;state<=WAIT_POST;end
                WAIT_POST:if(post_completion_valid && post_identity)begin
                    post_seen<=1;saved_post_snapshot<=post_snapshot;
@CAPTURE_POST@
                    if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(post_completion_status!=OK)begin completion_status<=post_completion_status;state<=RESPONSE;end
                    else if(!post_success)begin completion_status<=BAD_COMPLETION;state<=RESPONSE;end
                    else state<=LEASE_OFFER;
                end
                LEASE_OFFER:if(fault_now!=OK)begin completion_status<=fault_now;state<=RESPONSE;end
                    else if(lease_command_ready)begin lease_owned<=1;completion_stage<=4;state<=WAIT_LEASE;end
                WAIT_LEASE:if(lease_completion_valid && lease_identity)begin
                    lease_seen<=1;saved_lease_snapshot<=lease_snapshot;
@CAPTURE_LEASE@
                    completion_status<=fault_now!=OK ? fault_now:lease_completion_status!=OK ? lease_completion_status:
                        lease_success ? OK:BAD_COMPLETION;
                    state<=RESPONSE;
                end
                RESPONSE:if(completion_ready)state<=lease_owned ? ACK_LEASE:ACK_POST;
                ACK_LEASE:if(lease_completion_ready)state<=DRAIN_LEASE;
                DRAIN_LEASE:if(!lease_busy && !lease_completion_valid && !ffn_busy && !ffn_completion_valid && !post_busy && !post_completion_valid)begin
                    lease_owned<=0;post_owned<=0;tail_owner<=0;state<=ACK_O;
                end
                ACK_POST:if(!post_owned || post_completion_ready)state<=DRAIN_POST;
                DRAIN_POST:if(!post_busy && !post_completion_valid)begin post_owned<=0;tail_owner<=0;state<=ACK_O;end
                ACK_O:if(!projection_owned || projection_completion_ready)state<=DRAIN_O;
                DRAIN_O:if(!projection_busy && !projection_completion_valid)begin projection_owned<=0;state<=ACK_FRONT;end
                ACK_FRONT:if(!front_owned || front_released || front_completion_ready)state<=DRAIN_FRONT;
                DRAIN_FRONT:if(!front_busy && !front_completion_valid && !front_cache_exclusive && !gate_busy &&
                    !projection_busy && !projection_completion_valid && !post_busy && !post_completion_valid &&
                    !lease_busy && !lease_completion_valid && !ffn_busy && !ffn_completion_valid)begin
                    front_owned<=0;gate_passed<=0;state<=IDLE;
                end
                default:begin fault<=BAD_PEER;revoked<=1;completion_status<=BAD_PEER;state<=RESPONSE;end
            endcase
        end
    end
'''


def render():
    tables={};parents={}
    for role,(module,sha) in PARENTS.items():
        name='fpga/rtl/'+module+'.sv';raw=(ROOT/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==sha,name
        text=raw.decode()
        if ') (' not in text:text=text.replace('module '+module+' (','module '+module+' #() (',1)
        tables[role]=ports(text);parents[name]=sha
    g,f,o,p,l,t=[tables[n] for n in PARENTS]
    public={n:v for n,v in f.items() if n in ('clk','rst_n','expected_epoch') or n.startswith('command_') or
            n.startswith(('bind_','cache_','input_axi_','rope_axi_','v_axi_','kv_axi_')) or n in ('root_cache_lock','norm_tag_exhausted','qkv_tags_exhausted')}
    for n in ('command_addr','command_base','command_bytes'):public[n]=('input','[63:0]','[0:23]')
    for n,w in [('cancel_valid',''),('completion_ready','')]:public[n]=('input',w,'')
    for n,w in [('command_ready',''),('cancel_ready',''),('busy',''),('completion_valid',''),('completion_revoked',''),
                ('completion_status','[3:0]'),('completion_sequence','[63:0]'),('completion_epoch','[63:0]'),
                ('completion_layer','[31:0]'),('completion_position','[31:0]'),('completion_stage','[2:0]'),
                ('result_valid',''),('result_address','[63:0]'),('result_bytes','[13:0]'),('tail_selected','[1:0]'),
                ('projection_tags_exhausted',''),('ffn_tags_exhausted',''),('completion_preflight_status','[3:0]'),
                ('completion_preflight_region','[4:0]'),('completion_preflight_other_region','[4:0]')]:public[n]=('output',w,'')
    for n,v in o.items():
        if n.startswith('m_axi_'):public[n.replace('m_axi_','tail_axi_')]=v
    maps={role:{n:n if n in ('clk','rst_n','expected_epoch') else role+'_'+n for n in table} for role,table in tables.items()}
    gm,fm,om,pm,lm,tm=[maps[n] for n in PARENTS]
    for n in g:
        if n.startswith('command_') and n not in ('command_valid','command_ready','command_layer'):gm[n]=n
        if n.startswith('cache_') and n!='cache_query_layer':gm[n]=n
    gm.update(command_layer='gate_command_layer',cache_query_layer='gate_query_layer',
        completion_sequence='held_sequence',completion_epoch='held_epoch',completion_layer='held_layer',completion_position='held_position',
        validated_addr='address',validated_base='allocation_base',validated_bytes='allocation_bytes',
        validated_lease_base='held_lease',validated_coefficient_position='held_coefficient_position',validated_prefix='initial_prefix')
    for n in f:
        if n in public and (n.startswith(('bind_','cache_','input_axi_','rope_axi_','v_axi_','kv_axi_')) or n in ('root_cache_lock','norm_tag_exhausted','qkv_tags_exhausted')):fm[n]=n
    fm.update(bind_valid='front_bind_valid',bind_ready='front_bind_ready',cache_exclusive='front_cache_exclusive')
    for mapping in (fm,om,pm,lm):
        mapping['command_sequence']='completion_sequence';mapping['command_epoch']='completion_epoch'
    fm.update(command_source_sequence='completion_sequence',command_source_epoch='completion_epoch',command_source_valid="1'b1",
        command_layer='gate_query_layer',command_position='held_position',command_lease_base='held_lease',command_coefficient_position='held_coefficient_position')
    om['tag_exhausted']='projection_tags_exhausted';tm['tags_exhausted']='ffn_tags_exhausted'
    # The existing post/FFN lease is the sole authority for FFN admission and ACK.
    aliases={'source_sequence':'command_source_sequence','source_epoch':'command_source_epoch','source_valid':'command_source_valid',
             'residual_address':'command_residual_address','residual_base':'command_residual_base','residual_bytes':'command_residual_bytes',
             'output_address':'command_output_address','output_base':'command_output_base','output_bytes':'command_output_bytes',
             'sequence':'completion_sequence','epoch':'completion_epoch','hls_status':'completion_hls_status',
             'ffn_hls_status':'completion_ffn_hls_status','ffn_status':'completion_ffn_status','ffn_child_status':'completion_ffn_child_status',
             'residual_status':'completion_residual_status','ffn_stages':'completion_ffn_stages','ffn_phase':'completion_ffn_phase',
             'ffn_bytes':'completion_ffn_bytes','ffn_stage_bytes':'completion_ffn_stage_bytes',
             'read_beats':'completion_read_beats','write_beats':'completion_write_beats'}
    for n in l:
        if n.startswith('child_'):lm[n]='ffn_'+aliases.get(n[6:],n[6:])
    signals={};drivers={};assignments=[]
    for role,table in tables.items():
        for n,(d,w,a) in table.items():
            target=maps[role][n]
            if "'" in target:continue
            if target in public:assert public[target][1:]==(w,a),(role,n,target)
            elif target in signals:assert signals[target]==(w,a),(role,n,target)
            else:signals[target]=(w,a)
            if d=='output':assert target not in drivers,(target,drivers.get(target),role);drivers[target]=role
    diagnostics={};snapshots=[];reset=[]
    for role in ('front','projection','post','lease'):
        table=tables[role]
        selected=[n for n,(d,w,a) in table.items() if d=='output' and n.startswith('completion_') and n not in ('completion_valid','completion_sequence','completion_epoch','completion_layer','completion_position')]
        diagnostic=[]
        for n in selected:
            public_name='completion_'+role+'_'+n[len('completion_'):]
            public[public_name]=table[n];reset.append('            '+public_name+"<=0;")
            diagnostic.append('                    '+public_name+'<='+maps[role][n]+';')
        diagnostics[role]='\n'.join(diagnostic)
        observed=selected+[n for n,(d,w,a) in table.items() if d=='output' and n.startswith('result_') and n not in ('result_q','result_q_valid')]
        width=sum((int(table[n][1][1:-1].split(':')[0])-int(table[n][1][1:-1].split(':')[1])+1) if table[n][1] else 1 for n in observed)
        snapshots+=['    wire ['+str(width-1)+':0] '+role+'_snapshot={'+','.join(maps[role][n] for n in observed)+'};',
                    '    logic ['+str(width-1)+':0] saved_'+role+'_snapshot;']
    reset+=['            completion_preflight_status<=0;completion_preflight_region<=31;completion_preflight_other_region<=31;']
    # Wire all captured descriptors with explicit indices and port shape checks.
    for i in range(10):
        for field,source in [('addr','address'),('base','allocation_base'),('bytes','allocation_bytes')]:
            assignments.append(f'    assign front_command_{field}[{i}]={source}[{i}];')
    for label,i in [('weight',10),('activation',11),('output',12)]:
        for suffix,source in [('addr','address'),('region_base','allocation_base'),('region_bytes','allocation_bytes')]:
            assignments.append(f'    assign projection_command_{label}_{suffix}={source}[{i}];')
    for label,i in [('input',0),('branch',12),('weight',13),('residual',14),('ffn',15)]:
        for suffix,source in [('address','address'),('base','allocation_base'),('bytes','allocation_bytes')]:
            assignments.append(f'    assign post_command_{label}_{suffix}={source}[{i}];')
    for suffix,source in [('address','address'),('base','allocation_base'),('bytes','allocation_bytes')]:
        assignments.append(f'    assign lease_command_output_{suffix}={source}[23];')
    for i in range(1,8):
        for field,source in [('addr','address'),('base','allocation_base'),('bytes','allocation_bytes')]:
            assignments.append(f'    assign lease_command_ffn_{field}[{i}]={source}[{15+i}];')
    for i,index in enumerate([None,10,11,12,0,13,14,15]):
        for field,source in [('base','allocation_base'),('bytes','allocation_bytes')]:
            value=('saved_cache_region' if field=='base' else 'saved_cache_bytes') if index is None else f'{source}[{index}]'
            assignments.append(f'    assign lease_command_protected_{field}[{i}]={value};')
    for n,(d,w,a) in o.items():
        if not n.startswith('m_axi_'):continue
        suffix=n[6:];outside='tail_axi_'+suffix
        if d=='output':
            assignments.append(f'    assign {outside}=tail_owner==2 ? ffn_{n}:tail_owner==1 ? post_{n}:projection_{n};')
        else:
            for index,role in enumerate(('projection','post','ffn')):
                value=f'(tail_owner=={index} && {outside})' if suffix.endswith(('ready','valid')) else outside
                assignments.append(f'    assign {role}_{n}={value};')
    body=BODY.replace('@RESET_DIAGNOSTICS@','\n'.join(reset))
    for role,text in diagnostics.items():body=body.replace('@CAPTURE_'+role.upper()+'@',text)
    assert '@' not in body.replace('@(posedge clk)','').replace('@(',''),'Unexpanded template'
    # Every input of every child must have a producer, a public input, or an explicit assignment.
    driven_text=body+'\n'+'\n'.join(assignments)
    for role,table in tables.items():
        for n,(d,_,_) in table.items():
            target=maps[role][n]
            if d!='input' or "'" in target or target in drivers or (target in public and public[target][0]=='input'):continue
            assert re.search(r'\b'+re.escape(target)+r'(?:\[[^]]+\])?\s*(?:=|<=)',driven_text),(role,n,target)
    lines=['// Complete single-layer datapath: input/RoPE/attention -> O -> post-O -> FFN/residual.',
        '// One cache authority; the parent retains every allocation through final ACK and drain.',
        '// Four upstream AXI masters plus one retained downstream AXI owner; shared reset quiesces all peers.',
        '`default_nettype none','module ntrpu_llama32_layer_r1 #(',
        '    parameter integer AXI_WATCHDOG_CYCLES=65536,',
        '    parameter integer HLS_WATCHDOG_CYCLES=262144,',
        '    parameter integer MATRIX_TIMEOUT_CYCLES=16777216,',
        '    parameter integer HANDOFF_TIMEOUT_CYCLES=67108864,',
        '    parameter integer OWNER_TIMEOUT_CYCLES=268435456',') (']
    for i,(n,(d,w,a)) in enumerate(public.items()):
        lines.append('    '+d+(' logic ' if d=='output' and n.startswith('completion_') and n not in ('completion_valid','completion_revoked') else ' wire ')+(w+' ' if w else '')+n+(' '+a if a else '')+(',' if i<len(public)-1 else ''))
    lines+=[');']+['    wire '+(w+' ' if w else '')+n+(' '+a if a else '')+';' for n,(w,a) in signals.items()]
    lines+=snapshots+assignments+[body]
    parameters={
        'gate':'','front':'.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES),.MATRIX_TIMEOUT_CYCLES(MATRIX_TIMEOUT_CYCLES),.HANDOFF_TIMEOUT_CYCLES(HANDOFF_TIMEOUT_CYCLES),.OWNER_TIMEOUT_CYCLES(OWNER_TIMEOUT_CYCLES)',
        'projection':'.WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES)',
        'post':'.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)',
        'lease':'','ffn':'.AXI_WATCHDOG_CYCLES(AXI_WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)'}
    for role,(module,_) in PARENTS.items():
        table=tables[role];mapping=maps[role];params=parameters[role]
        lines.append('    '+module+(' #('+params+')' if params else '')+' '+role+'(')
        lines+=['        .'+n+'('+mapping[n]+')'+(',' if i<len(table)-1 else '') for i,n in enumerate(table)]
        lines.append('    );')
    lines+=['endmodule','`default_nettype wire','']
    return '\n'.join(lines),dict(parents=parents,public=public,signals=signals,mappings=maps)


if __name__=='__main__':
    text,record=render();path=ROOT/'fpga/rtl/ntrpu_llama32_layer_r1.sv'
    if path.exists():assert path.read_text()==text,'Refuse to alter an existing revision'
    else:path.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(dict(ports=len(record['public']),sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
