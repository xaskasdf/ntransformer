import json
from pathlib import Path
import statistics
import numpy as np

ROOT=Path(__file__).resolve().parent
FAIR=Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')

def read(path):return json.loads(path.read_text())
def values(path):
    rows=[json.loads(l) for l in (path/'measurements.jsonl').read_text().splitlines()]
    v=rows[0]['vocab'];a=np.fromfile(path/'all-logits.f32',dtype='<f4').reshape(65,v).astype('float64')
    assert np.isfinite(a).all()
    a-=a.max(axis=-1,keepdims=True);a-=np.log(np.exp(a).sum(axis=-1,keepdims=True))
    return a,rows
def comparison(a,b,targets):
    return dict(kl=float((np.exp(a)*(a-b)).sum(axis=-1).mean()),
        argmax_agreement=float(np.mean(a.argmax(axis=-1)==b.argmax(axis=-1))),
        nll_a=float(-a[np.arange(len(targets)),targets].mean()),nll_b=float(-b[np.arange(len(targets)),targets].mean()))
def main():
    result={}
    for w in ['p32','p128','p512','heldout-p512']:
        workload=read(FAIR/('workload-'+w+'.json'));target=workload['target_ids']
        old,_=values(FAIR/('nt-slru-'+w+'-r0'));llama,_=values(FAIR/('llama-'+w+'-r0'))
        entry=dict(old_vs_llama=comparison(old,llama,target))
        if (ROOT/('llama-serial-'+w)/'measurements.jsonl').exists():
            serial,rows=values(ROOT/('llama-serial-'+w))
            entry['llama_native_vs_serial']=comparison(llama,serial,target)
            entry['old_vs_llama_serial']=comparison(old,serial,target)
            entry['serial_times']={r['phase']:r['seconds'] for r in rows if r.get('round')==1 and r['kind']=='phase'}
        if (ROOT/('nt-hc-'+w)/'measurements.jsonl').exists():
            corrected,rows=values(ROOT/('nt-hc-'+w))
            entry['corrected_vs_llama']=comparison(corrected,llama,target)
            if 'llama_native_vs_serial' in entry:
                entry['corrected_vs_llama_serial']=comparison(corrected,serial,target)
            entry['corrected_times']={phase:statistics.median(r['seconds'] for r in rows if r.get('round') in [2,3,4] and r.get('phase')==phase) for phase in ['prefill','decode']}
            entry['measured_major_faults']=sum(r['major_page_faults'] for r in rows if r.get('round') in [2,3,4] and r['kind']=='phase')
        result[w]=entry
    (ROOT/'audit-summary.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
