"""Audit real HLS scale failures and delayed completion drain/reset in quad GEMV."""
import re
from check_ntrpu_gemv_lanes import review as lane_review,mutations as lane_mutations


def review(text,seed):
    result=lane_review(text,seed,4,mac_faults=True)
    errors=result['errors']
    def need(ok,why):
        if not ok:errors.append(why)
    rows=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',l)} for l in text.splitlines() if l.startswith('NTRPU_MEMORY_FAULT ')]
    need([r.get('id') for r in rows]==[c['id'] for c in result['cases']],'one fault record per case')
    fields={'id','error_mask','error_busy','held_start','release','held_mask','pending_seen','timeout_at','timeout_idle','timeout_mask','raw0','raw1','raw2','raw3'}
    for r,c,l in zip(rows,result['cases'],result['lane_records']):
        need(set(r)==fields,'exact fault schema')
        if set(r)!=fields:continue
        mode=c['scenario'];cycles=c['cycles']
        need(all(0<=r[k]<cycles for k in ('held_start','release','timeout_at')),'fault timestamps within job')
        need(l['peak']==4,'four overlapping engines in fault bank')
        for lane in range(4):
            need(r['raw'+str(lane)]==l['starts'+str(lane)],'each start produces a genuine HLS completion')
            need(l['dones'+str(lane)]==(0 if mode==23 else r['raw'+str(lane)]),'delayed completions delivered once or discarded by reset')
        if 17<=mode<=20:
            mask=1<<(mode-17)
            need(r['error_mask']==mask and r['error_busy']&mask and 2<=r['error_busy'].bit_count()<=4,'real INVALID_SCALE with another engine occupied')
            need(all(r[k]==0 for k in ('held_start','release','held_mask','pending_seen','timeout_at','timeout_idle','timeout_mask')),'no delay/timeout injection for HLS scale error')
        else:need(r['error_mask']==r['error_busy']==0,'no unexpected HLS error')
        if mode>=21:
            need(r['held_start']>0 and r['held_mask']==r['pending_seen']==15,'all four genuine completions held')
            need(all(r['raw'+str(i)]==l['starts'+str(i)]==1 for i in range(4)),'one operation per held lane')
            if mode==21:
                # hold begins on the first MAC_WAIT negedge, after an idle
                # clock has already accrued. The expiration is sampled before
                # the NBA update at idle=31 (the 32nd no-progress edge).
                need(r['timeout_at']-r['held_start']>=31 and r['timeout_idle']==31 and r['timeout_mask']==15,'natural 32-cycle watchdog with four occupied MACs')
                need(r['release']>=r['timeout_at'],'timeout recorded before releasing completions')
            if mode==22:
                need(l['cancel_mask']==15 and r['release']-r['held_start']>=50,'cancellation remains first failure and drains late completions')
            if mode==23:
                need(l['reset_mask']==15 and r['release']==0,'coordinated reset discards unreleased completions')
        elif mode==0:
            need(all(r[k]==0 for k in fields-{'id','raw0','raw1','raw2','raw3'}),'healthy recovery has no fault or stale pending completion')
        if r['timeout_at']:
            need(mode in (21,22,23) and r['timeout_idle']==31,'watchdog evidence only in held-completion cases')
    result.update(status='FAIL' if errors else 'PASS',fault_records=rows,
                  scope='Real HLS arithmetic. Only its completion/status boundary is delayed by a simulation shim; no physical hung-clock recovery or board qualification.')
    return result


def mutations(text):
    tests=lane_mutations(text)
    def change(job,field,value):
        return re.sub(r'(^NTRPU_MEMORY_FAULT id='+str(job)+r' [^\n]*\b'+field+r'=)\d+',lambda m:m[1]+str(value),text,count=1,flags=re.M)
    tests.update(error_lane=change(2,'error_mask',0),error_overlap=change(2,'error_busy',1),
                 held_mask=change(22,'held_mask',7),pending_mask=change(22,'pending_seen',7),
                 early_timeout=change(22,'timeout_at',1),idle_counter=change(22,'timeout_idle',0),
                 missing_real_completion=change(22,'raw3',0),early_release=change(22,'release',1),
                 stale_recovery=change(23,'pending_seen',1),reset_delivered_completion=change(32,'release',1))
    row=next(l for l in text.splitlines() if l.startswith('NTRPU_MEMORY_FAULT '))
    tests['missing_fault_record']=text.replace(row+'\n','',1)
    return tests
