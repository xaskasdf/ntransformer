"""Reconcile full-size FFN phase counters, progress samples and AXI traffic."""
from collections import Counter
import re

PHASES=('gate','up','swiglu_q8','down')
CASE_KEYS=set('id dim hidden mode target reset cycles ar r aw w b bytes down_bytes p0 p1 p2 p3'.split())
SUMMARY_KEYS=set('cases cycles bytes resets cancels ar_stalls aw_stalls w_stalls done_stalls'.split())

def records(text,prefix,keys,hex_fields=()):
    rows=[]
    for line in text.splitlines():
        if not line.startswith(prefix+' '):continue
        result={}
        for token in line.split()[1:]:
            match=re.fullmatch(r'([a-z0-9_]+)=([0-9a-f]+)',token)
            if not match:raise ValueError('Malformed '+prefix)
            key,value=match.groups()
            if key in result:raise ValueError('Duplicate field '+key)
            if key not in hex_fields and not value.isdecimal():raise ValueError('Nondecimal '+key)
            result[key]=int(value,16 if key in hex_fields else 10)
        if set(result)!=keys:raise ValueError('Unexpected fields in '+prefix)
        rows.append(result)
    return rows

def review(text):
    need=lambda ok,why:None if ok else (_ for _ in ()).throw(ValueError(why))
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Failure diagnostic')
    cases=records(text,'NTRPU_FFN_MEMORY_CASE',CASE_KEYS)
    summaries=records(text,'NTRPU_FFN_MEMORY_PASS',SUMMARY_KEYS)
    need(len(cases)==len(summaries)==1,'One closed root transaction')
    case=cases[0];summary=summaries[0]
    need(tuple(case[k] for k in ('id','dim','hidden','mode','target','reset'))==(1,2048,8192,0,0,0),'Full geometry/normal completion')
    need(tuple(summary[k] for k in ('cases','cycles','bytes','resets','cancels'))==(1,case['cycles'],case['bytes'],0,0),'Summary consistency')
    cycles=[case['p'+str(i)] for i in range(4)]
    need(all(c>0 for c in cycles) and sum(cycles)==case['cycles']<30000000,'Complete positive phase partition')
    ends=[];total=0
    for c in cycles:total+=c;ends.append(total)
    progress=records(text,'NTRPU_FFN_LARGE_PROGRESS',{'cycles','phase','bytes'})
    need([p['cycles'] for p in progress]==list(range(100000,case['cycles']+1,100000)),'Complete progress samples')
    output_bytes=[65536,65536,16384,8192];last_bytes=0;last_phase=0
    for row in progress:
        actual_phase=next(i for i,end in enumerate(ends) if row['cycles']<=end)
        need(row['phase']==actual_phase and row['phase']>=last_phase,'Progress contradicts phase counters')
        low=sum(output_bytes[:actual_phase]);high=low+output_bytes[actual_phase]
        need(low<=row['bytes']<=high and row['bytes']>=last_bytes and row['bytes']%64==0,'Monotonic committed bytes by phase')
        last_bytes=row['bytes'];last_phase=row['phase']
    reads=records(text,'NTRPU_FFN_LARGE_AR',{'id','phase','buffer','offset','beats'})
    writes=records(text,'NTRPU_FFN_MEMORY_W',{'id','buffer','offset','strb','data'},('strb','data'))
    ar=[0]*4;r=[0]*4;w=[0]*4;last_phase=0
    expected_read_bytes=[18874368,18874368,131072,18874368]
    expected_ar=[5376,5376,544,5184]
    allowed=[(0,1),(0,2),(4,5),(6,3)]
    for row in reads:
        p=row['phase'];need(row['id']==1 and 0<=p<4 and p>=last_phase,'Ordered read phases')
        need(row['buffer'] in allowed[p] and row['offset']%64==0 and 1<=row['beats']<=64,'Read ownership/shape')
        need((4032+row['offset'])%4096+row['beats']*64<=4096,'Read burst crosses page')
        ar[p]+=1;r[p]+=row['beats'];last_phase=p
    last_phase=0
    for row in writes:
        p=row['buffer']-4;need(row['id']==1 and 0<=p<4 and p>=last_phase,'Ordered write phases')
        need(row['strb']==(1<<64)-1 and row['data']<1<<512,'Complete write beats')
        need(row['offset']==w[p]*64,'Write coverage/offset')
        w[p]+=1;last_phase=p
    need(ar==expected_ar and [v*64 for v in r]==expected_read_bytes,'Read volume by phase')
    need([v*64 for v in w]==output_bytes,'Write volume by phase')
    need(tuple(case[k] for k in ('ar','r','aw','w','b','bytes','down_bytes'))==
         (sum(ar),sum(r),866,sum(w),866,sum(output_bytes),8192),'AXI totals')
    for i in range(4):need(max(r[i],w[i])<=cycles[i],'More beats than phase cycles')
    for key in ('ar_stalls','aw_stalls','w_stalls','done_stalls'):
        need(0<summary[key]<=case['cycles'],'Backpressure counter range')
    need(summary['done_stalls']<=cycles[3],'Completion stalls exceed final phase')
    rows=[]
    for i,name in enumerate(PHASES):
        # Weight matrices are read once per projection. Activation rereads are
        # accounted separately in actual R bytes, never called useful weights.
        weights=0 if i==2 else 17825792
        rows.append(dict(phase=i,name=name,cycles=cycles[i],fraction=cycles[i]/case['cycles'],
            read_bursts=ar[i],read_beats=r[i],read_bytes=r[i]*64,write_beats=w[i],write_bytes=w[i]*64,
            matrix_weight_bytes=weights,read_bytes_per_cycle=r[i]*64/cycles[i],
            weight_bytes_per_cycle=weights/cycles[i],r_beat_fraction=r[i]/cycles[i],w_beat_fraction=w[i]/cycles[i]))
    transitions=[]
    for i,end in enumerate(ends[:-1]):
        before=[p['cycles'] for p in progress if p['phase']<=i];after=[p['cycles'] for p in progress if p['phase']>i]
        low=max(before,default=0);high=min(after,default=case['cycles'])
        need(low<=end<high,'Transition sample bracket')
        transitions.append(dict(after_phase=i,counter_cycle=end,last_sample_before=low,first_sample_after=high,
            sampling_resolution_cycles=100000))
    return dict(status='PASS_LLAMA32_FULL_FFN_METRICS_R1',case=case,summary=summary,phases=rows,
        transition_brackets=transitions,progress_samples=len(progress),root_cycles=case['cycles'],
        read_bytes=sum(expected_read_bytes),write_bytes=sum(output_bytes),matrix_weight_bytes=53477376,
        matrix_weight_bytes_per_cycle=53477376/case['cycles'],physical_timing_qualified=False,
        scope='Simulation counters reconciled with sampled phase/byte progress and accepted AR/W records. Phase time includes control and stalls. Beat/stall fractions can overlap; not compute utilization, DDR bandwidth or tokens/s.')

def controls(text):
    def line_change(prefix,fn):
        return '\n'.join(fn(line) if line.startswith(prefix+' ') else line for line in text.splitlines())
    case=records(text,'NTRPU_FFN_MEMORY_CASE',CASE_KEYS)[0]
    altered=line_change('NTRPU_FFN_MEMORY_CASE',lambda line:line.replace('p0='+str(case['p0']), 'p0='+str(case['p0']+200000)).replace('p1='+str(case['p1']),'p1='+str(case['p1']-200000)))
    reads=[l for l in text.splitlines() if l.startswith('NTRPU_FFN_LARGE_AR ')];writes=[l for l in text.splitlines() if l.startswith('NTRPU_FFN_MEMORY_W ')]
    progress=[l for l in text.splitlines() if l.startswith('NTRPU_FFN_LARGE_PROGRESS ')]
    wrong_write=writes[0].replace('offset=0 ','offset=64 ',1)
    mutations={'redistributed_phase_cycles':altered,
        'duplicate_field':line_change('NTRPU_FFN_MEMORY_CASE',lambda l:l+' p0='+str(case['p0'])),
        'missing_progress':text.replace(progress[0]+'\n','',1),
        'duplicate_progress':text.replace(progress[0],progress[0]+'\n'+progress[0],1),
        'wrong_sample_phase':text.replace(progress[0],progress[0].replace('phase=0','phase=3'),1),
        'missing_read':text.replace(reads[0]+'\n','',1),
        'wrong_read_owner':text.replace(reads[0],reads[0].replace('buffer=0','buffer=3'),1),
        'page_crossing_read':text.replace(reads[0],reads[0].replace('beats=1','beats=64'),1),
        'missing_write':text.replace(writes[0]+'\n','',1),
        'duplicate_write':text.replace(writes[0],writes[0]+'\n'+writes[0],1),
        'wrong_write_offset':text.replace(writes[0],wrong_write,1),
        'missing_summary':text.replace('NTRPU_FFN_MEMORY_PASS','REMOVED_FFN_PASS'),
        'impossible_stalls':line_change('NTRPU_FFN_MEMORY_PASS',lambda l:re.sub(r'ar_stalls=\d+','ar_stalls=30000001',l)),
        'failure_diagnostic':text+'\nFAIL injected\n'}
    rejected=[]
    for name,altered in mutations.items():
        if altered==text:raise AssertionError('Ineffective corruption '+name)
        try:review(altered)
        except (AssertionError,ValueError,StopIteration):rejected.append(name)
        else:raise AssertionError('Accepted corrupted metrics '+name)
    return rejected
