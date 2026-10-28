"""Check lane work/occupancy and all-lane fault coverage alongside the Q8 oracle."""
import re
from check_ntrpu_gemv_memory import review as memory_review


def review(text,seed,lanes,fixture=None,extended=False,mac_faults=False):
    if lanes not in (2,4):raise ValueError('Expected dual or quad')
    result=memory_review(text,seed,fixture,extended,mac_faults)
    errors=result['errors']
    def need(ok,why):
        if not ok:errors.append(why)
    rows=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
          for line in text.splitlines() if line.startswith('NTRPU_MEMORY_LANES ')]
    cases=result['cases']
    need([r.get('id') for r in rows]==[r['id'] for r in cases],'one lane record per job')
    fields={'id','lanes','peak','cancel_mask','reset_mask'}|{n+str(i) for n in ('starts','dones','busy') for i in range(4)}
    for row,case in zip(rows,cases):
        need(set(row)==fields,'exact lane schema')
        if set(row)!=fields:continue
        n,k,mode=case['rows'],case['blocks'],case['scenario']
        need(row['lanes']==lanes and 0<=row['peak']<=min(lanes,n),'lane count and peak bounds')
        for i in range(4):
            starts,dones,busy=[row[key+str(i)] for key in ('starts','dones','busy')]
            if i>=lanes:
                need(starts==dones==busy==0,'inactive lane counters')
                continue
            need(0<=dones<=starts<=busy<=case['cycles'],'bounded lane lifetime counters')
            if mode==0:
                expected=((n+lanes-1-i)//lanes)*k
                need(starts==dones==expected,'one completion per ordered row/K block')
                need(busy>0 if expected else busy==0,'occupied iff used lane')
            elif mode not in (15,16,23):
                need(starts==dones,'all launched MACs drained before error/cancel completion')
        if mode==0:need(row['peak']==min(lanes,n),'all assigned lanes used simultaneously')
        if extended and n>=lanes and mode==14:
            need(row['cancel_mask']==(1<<lanes)-1,'cancellation while every MAC occupied')
        if extended and n>=lanes and mode==16:
            need(row['reset_mask']==(1<<lanes)-1,'reset while every MAC occupied')
        if extended and n>=lanes and mode in (14,16):
            need(row['peak']==lanes and all(row['starts'+str(i)]>0 for i in range(lanes)),
                 'fault after every HLS start and sampled active lane')
        if mode not in (9,10,11,14,22):need(row['cancel_mask']==0,'no unexpected cancellation')
        if mode not in (15,16,23):need(row['reset_mask']==0,'no unexpected reset')
    result.update(status='FAIL' if errors else 'PASS',lanes=lanes,lane_records=rows,
                  scope='Same Q8 order and independent numerical oracle, plus lane start/done/occupancy and fault masks. Simulation RAM only.')
    return result


def mutations(text):
    def change(prefix,field,value):
        return re.sub(r'(^'+prefix+r' [^\n]*\b'+field+r'=)\S+',lambda m:m[1]+str(value),text,count=1,flags=re.M)
    lane=next(l for l in text.splitlines() if l.startswith('NTRPU_MEMORY_LANES '))
    return {
        'missing_lane_record':text.replace(lane+'\n','',1),
        'duplicate_lane_record':text+'\n'+lane+'\n',
        'lane_count':change('NTRPU_MEMORY_LANES','lanes',3),
        'start_count':change('NTRPU_MEMORY_LANES','starts0',0),
        'done_count':change('NTRPU_MEMORY_LANES','dones0',0),
        'occupancy':change('NTRPU_MEMORY_LANES','busy0',0),
        'peak':change('NTRPU_MEMORY_LANES','peak',0),
        'output':change('NTRPU_MEMORY_VALUE','value','nan'),
        'read_beats':change('NTRPU_MEMORY_CASE','weight_beats',0),
        'missing_pass':text.replace('NTRPU_MEMORY_PASS ','REMOVED_PASS ',1),
        'fatal':text+'\nFatal: injected\n',
    }
