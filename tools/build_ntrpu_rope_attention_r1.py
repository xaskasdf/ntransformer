"""Join the actual RoPE/capture dispatcher and actual 32-head attention root."""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def ports(path):
    header=path.read_text().split(') (',1)[1].split(');',1)[0];result=[]
    for line in header.splitlines():
        line=line.strip().rstrip(',')
        if not line:continue
        m=re.fullmatch(r'(input|output)\s+(?:wire\s+)?(?:logic\s+)?(\[[0-9:]+\])?\s*(.+)',line)
        assert m,line
        direction,width,names=m.groups()
        result.extend((direction,width or '',n.strip()) for n in names.split(','))
    assert len(result)==len({p[2] for p in result})
    return result

def render():
    front_path=ROOT/'fpga/rtl/ntrpu_rope_qkv_dispatch_r1.sv'
    root_path=ROOT/'fpga/rtl/ntrpu_attention_root_r1.sv'
    fp=ports(front_path);rp=ports(root_path)
    f={n:(d,w) for d,w,n in fp};r={n:(d,w) for d,w,n in rp}
    internal={n for n in f if n.startswith(('cache_','append_','data_','commit_','root_','q_'))}
    fm={n:n for n in f};rm={n:n for n in r}
    rm.update({n:'root_'+n for n in r if n.startswith(('command_','cancel_','result_'))})
    rm.update(completion_valid='root_completion_valid',completion_ready='root_completion_ready',
        completion_status='root_status',completion_sequence='root_sequence',completion_epoch='root_epoch',
        metadata_fault='cache_metadata_fault',committed_tokens='cache_committed_tokens',cache_lock='root_cache_lock',
        maintenance_busy='root_maintenance_busy',lease_active='root_lease_active',
        bind_valid='root_bind_valid',bind_ready='root_bind_ready',append_done_ready='root_append_done_ready')
    rm.update({n:n.replace('m_axi_','kv_axi_') for n in r if n.startswith('m_axi_')})
    fm.update({n:n.replace('m_axi_','v_axi_') for n in f if n.startswith('m_axi_')})
    fm.update(command_valid='front_command_valid',command_ready='front_command_ready',busy='front_busy',
        cache_exclusive='front_exclusive',append_done_valid='front_append_done_valid',append_done_ready='front_append_done_ready')
    public={fm[n]:(d,w) for d,w,n in fp if n not in internal and n not in ('command_valid','command_ready','busy')}
    # cache_exclusive is a host-visible output of the composition, not a child port.
    public.update(command_valid=('input',''),command_ready=('output',''),busy=('output',''),cache_exclusive=('output',''))
    for d,w,n in rp:
        if n.startswith(('bind_','cache_','m_axi_')):
            public[n if n.startswith('bind_') else rm[n]]=(d,w)
    public.update(cache_metadata_fault=('output',''),cache_committed_tokens=('output','[31:0]'),
        bind_completion_valid=('output',''),bind_completion_ready=('input',''),
        bind_completion_status=('output','[3:0]'),bind_completion_sequence=('output','[63:0]'),
        bind_completion_epoch=('output','[63:0]'))
    signals={}
    for pp,mapping in ((fp,fm),(rp,rm)):
        for _,w,n in pp:
            target=mapping[n]
            if target in public:assert public[target][1]==w,(n,target)
            elif target in signals:assert signals[target]==w,(n,target)
            else:signals[target]=w
    # The shared transport must be oppositely directed and exactly the same width.
    joined=[]
    for fn in internal:
        matches=[rn for rn in r if rm[rn]==fm[fn]]
        if fn in ('cache_idle','append_done_valid','append_done_ready','cache_exclusive'):continue
        assert len(matches)==1,(fn,matches)
        rn=matches[0];assert f[fn][0]!=r[rn][0] and f[fn][1]==r[rn][1],(fn,rn)
        joined.append((fn,rn))
    lines=['// Actual RoPE -> capture -> transactional KV append -> 32-head attention.',
        '// A bind has admission priority, but cannot interrupt an owned front-end job.',
        '// Three separate AXI masters; reset and memory quiesce remain coordinated.',
        '`default_nettype none','module ntrpu_rope_attention_r1 #(',
        '    parameter integer WATCHDOG_CYCLES=65536,',
        '    parameter integer HLS_WATCHDOG_CYCLES=262144',') (']
    pp=[(d,w,n) for n,(d,w) in public.items()]
    lines+=['    '+d+' wire '+(w+' ' if w else '')+n+(',' if i<len(pp)-1 else '') for i,(d,w,n) in enumerate(pp)]
    lines+=[');']+['    wire '+(w+' ' if w else '')+n+';' for n,w in signals.items()]
    lines+=['    wire admission_allowed;',
        '    assign cache_idle=!root_maintenance_busy && !root_lease_active && !root_cache_lock && !append_done_valid && !root_completion_valid;',
        '    assign admission_allowed=!bind_valid && cache_idle;',
        '    assign front_command_valid=command_valid && admission_allowed;',
        '    assign command_ready=front_command_ready && admission_allowed;',
        '    assign root_bind_valid=bind_valid && !front_busy && !front_exclusive;',
        '    assign bind_ready=root_bind_ready && !front_busy && !front_exclusive;',
        '    assign front_append_done_valid=append_done_valid && !append_done_bind;',
        '    assign root_append_done_ready=append_done_bind ? bind_completion_ready:front_append_done_ready;',
        '    assign bind_completion_valid=append_done_valid && append_done_bind;',
        '    assign bind_completion_status=append_done_status;',
        '    assign bind_completion_sequence=append_done_sequence;',
        '    assign bind_completion_epoch=append_done_epoch;',
        '    assign busy=front_busy || !cache_idle;',
        '    assign cache_exclusive=front_exclusive || !cache_idle;']
    for module,instance,pp,mapping,params in (
        ('ntrpu_rope_qkv_dispatch_r1','front',fp,fm,'.WATCHDOG_CYCLES(WATCHDOG_CYCLES),.HLS_WATCHDOG_CYCLES(HLS_WATCHDOG_CYCLES)'),
        ('ntrpu_attention_root_r1','attention',rp,rm,'.WATCHDOG_CYCLES(WATCHDOG_CYCLES)')):
        lines.append('    '+module+' #('+params+') '+instance+'(')
        lines+=['        .'+n+'('+mapping[n]+')'+(',' if i<len(pp)-1 else '') for i,(_,_,n) in enumerate(pp)]
        lines.append('    );')
    lines+=['endmodule','`default_nettype wire','']
    record=dict(public=public,signals=signals,front_mapping=fm,root_mapping=rm,joined=sorted(joined),
        parents={p.relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in (front_path,root_path)})
    return '\n'.join(lines),record

if __name__=='__main__':
    text,record=render();p=ROOT/'fpga/rtl/ntrpu_rope_attention_r1.sv'
    if p.exists():assert p.read_text()==text,'Refuse to overwrite an altered wrapper'
    else:p.write_text(text,encoding='utf-8',newline='\n')
    print(json.dumps(dict(ports=len(record['public']),joined=len(record['joined']),sha256=hashlib.sha256(p.read_bytes()).hexdigest())))
