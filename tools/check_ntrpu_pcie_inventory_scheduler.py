"""Check the producer's every public state and canonical query, including faulty rows."""
import re
from check_ntrpu_pcie_request_inventory_r5 import review as protocol_review


def require(ok, message):
    if not ok:
        raise ValueError(message)


def rows(log, prefix):
    return [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in log.splitlines() if line.startswith(prefix)]


def review(logs, vectors, manifest, baseline_log, internal=True, mutants=True):
    normal=logs['normal/simulate.log']
    protocol=protocol_review(normal,vectors,manifest)
    count=manifest['rows']; scenarios=len(manifest['scenarios'])
    require(normal.splitlines().count('SCHED_MODE mode=0')==1,'normal producer mode')
    require(normal.splitlines().count(f'SCHED_SUITE_PASS rows={count} scenarios={scenarios} mode=0')==1,'complete producer bank')
    actual=rows(normal,'SCHED_ROW '); expected=rows(normal,'SCHED_EXPECT ')
    public=[s for s in normal.splitlines() if s.startswith('PCIE_INV_ROW ')]
    require(public==[s for s in baseline_log.splitlines() if s.startswith('PCIE_INV_ROW ')],'reference output changed')
    require(len(actual)==len(expected)==count,'missing/duplicate producer records')
    states=rows(normal,'PCIE_INV_ROW ')
    fields={'index','scenario','observing','tags','reads','bytes','rq_partial','rc_partial',
            'requests','completed','writes','bitmap','record','seq','valid','owned','metadata',
            'delivered','rc_context','rq_context','pending'}
    state_fields={'observing','tags','reads','bytes','rq_partial','rc_partial','requests','completed','writes','bitmap'}
    vector_rows=[line.split() for line in vectors.splitlines() if line.strip()]
    records=0; headers=0; wraps=0; words=0
    last_sequence=None
    for i,(a,e,state,v) in enumerate(zip(actual,expected,states,vector_rows)):
        require(set(e)==fields and set(a)==fields|{'mode','words','early_faults'},'unexpected producer record schema')
        require(a['index']==e['index']==str(i) and a['scenario']==e['scenario']==v[0],'producer identity/order')
        require(a['mode']=='0' and a['words']==('256' if internal else '0'),'wrong mode/table coverage')
        require(all(a[k]==e[k] for k in fields),'canonical response/state mismatch at '+str(i))
        require(all(a[k]==state[k] for k in state_fields),'canonical/reference public state at '+str(i))
        enabled=bool(int(v[2])) and bool(int(states[i-1]['observing'])) if i else False
        require(int(a['record'])==enabled,'capture validity/arming/reset')
        require(int(a['seq'],16)==((0xffffff80+i)&0xffffffff if enabled else 0),'diagnostic sequence alignment/wrap')
        if enabled:
            sequence=int(a['seq'],16)
            wraps+=int(last_sequence is not None and sequence<last_sequence)
            last_sequence=sequence
        records+=int(a['record']); headers+=int(a['valid'],16).bit_count(); words+=int(a['words'])
        require(int(a['owned'],16)&~int(a['valid'],16)==0,'ownership outside captured header slots')
        require(not (any(int(a[k]) for k in ['tags','reads','bytes','rq_partial','rc_partial']) and a['pending']=='0'),'omitted canonical obligation')
    require(wraps>=1,'diagnostic sequence wrap was not exercised')
    rejected=[]
    if mutants:
        for name,mode,kind,marker in [
            ('omit_rq',1,'TABLE|QUERY|STATE',''), ('omit_rc',2,'TABLE|QUERY|STATE',''),
            ('corrupt_table',3,'TABLE','SCHED_TABLE_CORRUPTION_INJECTED'),
            ('corrupt_context',4,'CONTEXT|STATE','SCHED_CONTEXT_CORRUPTION_INJECTED')]:
            log=logs[name+'/simulate.log']
            require(log.splitlines().count(f'SCHED_MODE mode={mode}')==1,'mutation not enabled: '+name)
            fatals=[s for s in log.splitlines() if s.startswith('Fatal:')]
            require(len(fatals)==1 and 'SCHED_SUITE_PASS' not in log,'mutant did not fail: '+name)
            match=re.fullmatch(r'Fatal: SCHED_('+kind+r')_MISMATCH row=(\d+)(?: tag=\d+)? mode='+str(mode),fatals[0])
            require(match is not None,'unexpected mutant rejection: '+name)
            failure_row=int(match[2]); require(0<=failure_row<count,'mutant failure outside bank')
            prior=rows(log,'SCHED_ROW ')
            require(len(prior)==failure_row,'mutant prefix coverage')
            for i,row in enumerate(prior):
                require(row==dict(actual[i],mode=str(mode)),'mutant changed before first rejection')
            require([s for s in log.splitlines() if s.startswith('PCIE_INV_ROW ')]==public[:failure_row],'mutant changed golden reference')
            if marker:
                require(log.splitlines().count(marker)==1 and log.index(marker)<log.index(fatals[0]),'missing actual corruption: '+name)
            rejected.append(dict(name=name,row=failure_row,fatal=fatals[0]))
    return dict(status='PASS_PCIE_CAUSAL_SCHEDULER_REFERENCE_AND_MUTATIONS' if mutants else 'PASS_PCIE_CAUSAL_SCHEDULER_BINARY',
                rows=count,scenarios=scenarios,table_word_comparisons=words,query_slots_compared=count*4,
                captured_records=records,headers=headers,diagnostic_wraps=wraps,protocol_reference=protocol,
                actual_mutants=rejected,all_canonical_states_compared_including_faults=True,
                metadata_source='Own table and ordered events from raw RQ/RC; reference is testbench-only.',
                payload_validation_integrated=False,full_pipeline_qualified=False)
