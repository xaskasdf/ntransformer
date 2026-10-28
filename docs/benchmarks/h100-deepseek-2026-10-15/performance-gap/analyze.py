from pathlib import Path
import csv
import json
import math
from collections import defaultdict
import numpy as np
ROOT=Path(__file__).resolve().parent
def union(intervals):
    result=[]
    for a,b in sorted(intervals):
        if b<=a:continue
        if result and a<=result[-1][1]:result[-1]=(result[-1][0],max(result[-1][1],b))
        else:result.append((a,b))
    return sum(b-a for a,b in result),result
def trace(path):
    with path.open(encoding='utf-8') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    for row in rows:
        for key in ['start','end','bytes','copy_kind','device','stream','correlation']:row[key]=int(row[key])
    assert not any(r['kind']=='dropped' for r in rows),path
    windows=[r for r in rows if r['kind']=='window'];assert len(windows)==1
    a,b=windows[0]['start'],windows[0]['end']
    groups=defaultdict(lambda:dict(count=0,ms=0,bytes=0))
    intervals=defaultdict(list)
    for r in rows:
        if r['kind']=='window':continue
        assert r['end']>=r['start']>0,(path,r)
        # Clip times to the wall window, keeping counts and bytes for boundary operations.
        start,end=max(a,r['start']),min(b,r['end'])
        if end<start:continue
        group=groups[(r['kind'],r['name'],r['copy_kind'])]
        group['count']+=1;group['ms']+=(end-start)/1e6;group['bytes']+=r['bytes']
        intervals[r['kind']].append((start,end))
        if r['kind']=='api' and 'Synchronize' in r['name']:intervals['host_sync'].append((start,end))
    result=dict(wall_ms=(b-a)/1e6,records=len(rows),dropped=0)
    for kind in ['kernel','memcpy','api','host_sync']:
        result[kind+'_union_ms']=union(intervals[kind])[0]/1e6
    result['gpu_union_ms']=union(intervals['kernel']+intervals['memcpy'])[0]/1e6
    result['gpu_unobserved_ms']=result['wall_ms']-result['gpu_union_ms']
    result['groups']=[dict(kind=k[0],name=k[1],copy_kind=k[2],**v) for k,v in sorted(groups.items(),key=lambda kv:-kv[1]['ms'])]
    return result
def metrics(a,b):
    a=a.astype(np.float64);b=b.astype(np.float64);delta=a-b
    return dict(n=a.size,max_abs=float(np.max(np.abs(delta))),relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-30)),
        bitwise_equal=bool(np.array_equal(a,b)),top1_a=int(np.argmax(a)),top1_b=int(np.argmax(b)))
def main():
    result=dict(traces={},cases={},stages={})
    for p in ROOT.glob('*trace-p32/*.tsv'):
        command=json.loads((ROOT/(p.parent.name+'-command.json')).read_text())
        if command['returncode']==0:result['traces'][p.parent.name+'/'+p.stem]=trace(p)
    for p in ROOT.glob('*/measurements.jsonl'):
        rows=[json.loads(l) for l in p.read_text(encoding='utf-8').splitlines()]
        phases=[r for r in rows if r.get('kind')=='phase']
        completions=[r for r in rows if r.get('kind')=='completion']
        result['cases'][p.parent.name]=dict(phases=phases,completions=completions)
    for p in (ROOT/'nt-stages').glob('*.f32'):
        other=ROOT/'llama-stages'/p.name
        if not other.exists():continue
        a=np.fromfile(p,dtype='<f4');b=np.fromfile(other,dtype='<f4')
        if a.shape!=b.shape:result['stages'][p.stem]=dict(shape_nt=list(a.shape),shape_llama=list(b.shape));continue
        assert np.isfinite(a).all() and np.isfinite(b).all(),p
        result['stages'][p.stem]=metrics(a,b)
    for control,other in [('llama-control-p32','llama-no-graphs-p32'),('llama-control-p128','llama-no-op-offload-p128')]:
        if (ROOT/other/'final-logits-round0.f32').exists():
            a=np.fromfile(ROOT/other/'final-logits-round0.f32',dtype='<f4');b=np.fromfile(ROOT/control/'final-logits-round0.f32',dtype='<f4')
            result['cases'][other]['final_vs_control']=metrics(a,b)
    result['dump_controls']={}
    for engine,a,b in [('nt',ROOT/'nt-stages/first-logits.f32',ROOT/'nt-single-control/first-logits.f32'),
                       ('llama',ROOT/'llama-single/prompt-logits.f32',ROOT/'llama-single-control/prompt-logits.f32')]:
        if a.exists() and b.exists():result['dump_controls'][engine]=metrics(np.fromfile(a,dtype='<f4'),np.fromfile(b,dtype='<f4'))
    weights=ROOT/'operator-weights'
    if weights.exists() and (ROOT/'llama-stages/qr-0.f32').exists():
        config=json.loads((weights/'config.json').read_text())
        blocks=np.fromfile(weights/'blk.0.attn_q_a.weight',dtype=np.dtype([('d','<f2'),('qs','i1',(32,))])).reshape(config['rows'],-1)
        matrix=(blocks['d'].astype(np.float64)[:,:,None]*blocks['qs']).reshape(config['rows'],config['cols'])
        norm=np.fromfile(weights/'blk.0.attn_q_a_norm.weight',dtype='<f4').astype(np.float64)
        result['first_projection_replay']={}
        for engine in ['nt','llama']:
            x=np.fromfile(ROOT/f'{engine}-stages/attn_norm-0.f32',dtype='<f4')
            actual=np.fromfile(ROOT/f'{engine}-stages/qr-0.f32',dtype='<f4')
            grouped=x.reshape(-1,32)
            d=np.max(np.abs(grouped),axis=1)/np.float32(127)
            safe=np.where(d==0,np.float32(1),d)
            scaled=grouped/safe[:,None]
            q=np.copysign(np.floor(np.abs(scaled)+.5),scaled)
            qx=(q.astype(np.float64)*d.astype(np.float16).astype(np.float64)[:,None]).reshape(-1)
            for mode,activation in [('fp32_input',x.astype(np.float64)),('q8_1_input',qx)]:
                projected=matrix@activation
                projected=projected/np.sqrt(np.mean(projected**2)+config['eps'])*norm
                result['first_projection_replay'][engine+'_'+mode]=metrics(projected,actual)
        # Stored numerical results can be checked without bundling model weights.
        (ROOT/'projection-replay.json').write_text(json.dumps(result['first_projection_replay'],indent=2)+'\n',encoding='utf-8')
    elif (ROOT/'projection-replay.json').exists():
        result['first_projection_replay']=json.loads((ROOT/'projection-replay.json').read_text())
    (ROOT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    for name,t in result['traces'].items():
        print(name,{k:v for k,v in t.items() if k!='groups'})
        print('top',t['groups'][:4])
    for name,c in result['cases'].items():
        print(name,[(r['round'],r['phase'],round(r['seconds'],4)) for r in c['phases']],c.get('final_vs_control'))
    for name in ['hc_init','attn_norm-0','qr-0','q-0','attn_out-0','ffn_moe_out-0','l_last-0','l_last-10','l_last-39','result_output']:
        if name in result['stages']:print(name,result['stages'][name])
if __name__=='__main__':main()
