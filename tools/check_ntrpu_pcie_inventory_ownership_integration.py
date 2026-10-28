"""Check complete delayed state against an independently archived reference bank."""
import re
from check_ntrpu_pcie_inventory_scheduler import require, rows

MODES=[('omit_rq',1,''),('omit_rc',2,''),
       ('omit_validation',3,'JOIN_VALIDATION_OMISSION_INJECTED'),
       ('false_empty',4,'JOIN_FALSE_EMPTY_INJECTED'),('wrong_sequence',5,''),
       ('omit_cold_reset',6,''),('corrupt_metadata',7,'JOIN_METADATA_CORRUPTION_INJECTED')]
STATE={'observing','fault','tags','reads','bytes','rq_partial','rc_partial','requests','completed','writes','bitmap'}


def protocol_review(log,vectors,manifest):
    v=[line.split() for line in vectors.splitlines() if line.strip()]
    actual=rows(log,'PCIE_INV_ROW ')
    require(len(actual)==len(v)==manifest['rows'],'ownership reference row count')
    scenarios=set()
    state_names=['tags','reads','bytes','rq_partial','rc_partial','requests','completed','writes']
    for i,(record,source) in enumerate(zip(actual,v)):
        require(record['index']==str(i) and record['scenario']==source[0] and record['native_ps']==source[1]=='0','ownership input identity')
        scenarios.add(int(source[0]))
        require(record['observing']==source[16] and bool(int(record['fault'],16))==bool(int(source[17])),'ownership observing/fault expectation')
        if source[18]=='1':
            require(all(int(record[key])==int(source[19+j]) for j,key in enumerate(state_names)),'ownership software state expectation')
            require(int(record['bitmap'],16)==int(source[27],16),'ownership software bitmap')
        if int(record['fault'],16):require(record['empty']=='0','ownership faulty reference empty')
    require(scenarios==set(range(len(manifest['scenarios']))),'ownership scenario coverage')
    mark=f"PCIE_INV_SUITE_PASS rows={len(v)} scenarios={len(scenarios)} native_endpoint=0 native_trace_replayed=0 global_drain=0"
    require(log.splitlines().count(mark)==1,'ownership suite marker')
    return dict(status='PASS_OWNERSHIP_REFERENCE_EXPECTATIONS',rows=len(v),scenarios=len(scenarios),native_rows=0,
                scope='Python state before errors; immutable R7 RTL reference for all post-fault rows.')


def review(logs,vectors,manifest,baseline_log,internal=True,mutants=True):
    log=logs['normal/simulate.log']; n=manifest['rows']; scenarios=len(manifest['scenarios'])
    protocol=protocol_review(log,vectors,manifest)
    require(log.splitlines().count('JOIN_MODE mode=0')==1,'normal join mode')
    public=[s for s in log.splitlines() if s.startswith('PCIE_INV_ROW ')]
    require(public==[s for s in baseline_log.splitlines() if s.startswith('PCIE_INV_ROW ')],'independent reference differs')
    refs=rows(baseline_log,'PCIE_INV_ROW '); actual=rows(log,'JOIN_ROW ')
    vec=[s.split() for s in vectors.splitlines() if s.strip()]
    require(len(actual)==n+1 and len(refs)==len(vec)==n,'missing/duplicate retirement rows or final flush')
    empty_state={k:'0' for k in STATE}; empty_state.update(fault='0000',bitmap='0'*64)
    held_rq=held_rc=False; saved_requester=saved_attr=None
    retired=canceled=empty=wraps=words=revocations=0; last_sequence=None
    cancel_records=[]; previous_empty=False
    for i,a in enumerate(actual):
        require(set(a)==STATE|{'index','source','mode','words','valid','seq','empty','pending'},'unexpected joined schema')
        require(a['index']==str(i) and a['mode']=='0','join identity/order')
        v=vec[i] if i<n else list(vec[-1]); current=refs[i] if i<n else refs[-1]
        if i==n:
            v[2]='1'; v[8]=v[12]='0'
        cold=v[2]=='1'; previous_cold=i>0 and vec[i-1][2]=='1'
        valid=cold and previous_cold
        require(a['valid']==str(int(valid)) and a['source']==str(i-1 if valid else -1),'retirement/cold cancellation')
        sequence=(0xffffff80+i-1)&0xffffffff if valid else 0
        require(a['seq']==f'{sequence:08x}','delayed diagnostic sequence or wrap')
        expected=refs[i-1] if valid else empty_state
        require(all(a[k]==expected[k] for k in STATE),'full retired state/fault mismatch at '+str(i))
        pre_observing=i>0 and refs[i-1]['observing']=='1'
        if not cold:
            held_rq=held_rc=False; saved_requester=saved_attr=None
        elif not pre_observing:
            if (v[3],v[4],v[5])==('1','0','1'):saved_requester,saved_attr=v[6:8]
        else:
            held_rq=v[8]=='1' and int(v[9],16)==0
            held_rc=v[12]=='1' and v[13]=='0'
        activity=v[8]=='1' or v[12]=='1'
        pending=cold and (any(current[k]!='0' for k in ['tags','reads','bytes','rq_partial','rc_partial']) or
                           held_rq or held_rc or activity or (pre_observing and activity))
        require(a['pending']==str(int(pending)),'canonical or validation obligation omitted at '+str(i))
        expected_empty=(valid and current['empty']=='1' and (v[6],v[7])==(saved_requester,saved_attr) and
                        int(v[9],16) in (0,15) and not pending and expected['observing']=='1' and
                        expected['fault']=='0000' and all(expected[k]=='0' for k in ['tags','reads','bytes','rq_partial','rc_partial']))
        require(a['empty']==str(int(expected_empty)),'false/delayed indefinite empty at '+str(i))
        revocations+=int(previous_empty and activity); previous_empty=expected_empty
        empty+=int(expected_empty); retired+=int(valid)
        if valid:
            wraps+=int(last_sequence is not None and sequence<last_sequence); last_sequence=sequence
        if not cold and previous_cold:
            canceled+=1; cancel_records.append(dict(index=str(i),source=str(i-1),seq=f'{(0xffffff80+i-1)&0xffffffff:08x}'))
        word_count=256 if internal and i<n else 0
        require(a['words']==str(word_count),'internal table coverage'); words+=word_count
    require(rows(log,'JOIN_CANCELED ')==cancel_records,'cold-reset cancellation evidence')
    require(retired+canceled==sum(v[2]=='1' for v in vec),'unaccounted captured source row')
    require(empty>0 and revocations>0 and canceled>0 and wraps>0,'required quiet/traffic/reset/wrap coverage')
    marker=f'JOIN_SUITE_PASS rows={n} scenarios={scenarios} mode=0 retired={retired} canceled={canceled} revocations={revocations} empty={empty}'
    require(log.splitlines().count(marker)==1,'complete joined bank/counters')
    rejected=[]
    if mutants:
        for name,mode,injected in MODES:
            changed=logs[name+'/simulate.log']
            require(changed.splitlines().count(f'JOIN_MODE mode={mode}')==1,'actual mutant mode: '+name)
            fatal=[s for s in changed.splitlines() if s.startswith('Fatal:')]
            require(len(fatal)==1 and 'JOIN_SUITE_PASS' not in changed,'mutant falsely passed: '+name)
            match=re.fullmatch(r'Fatal: JOIN_(STATE|SEQUENCE|PENDING|EMPTY|REVOKE|FAULT)_MISMATCH row=(\d+) mode='+str(mode),fatal[0])
            require(match is not None,'unexpected mutant failure: '+name)
            at=int(match[2]); ownership=rows(changed,'JOIN_ROW ')
            require(0<=at<n and len(ownership)==at,'mutant ownership length: '+name)
            for j,r in enumerate(ownership):
                require(r==dict(actual[j],mode=str(mode),words='0'),'mutant ownership mismatch: '+name)
            require([s for s in changed.splitlines() if s.startswith('PCIE_INV_ROW ')]==public[:at],'mutant changed reference')
            if injected:
                require(changed.splitlines().count(injected)==1 and changed.index(injected)<changed.index(fatal[0]),'missing actual injection')
            rejected.append(dict(name=name,row=at,fatal=fatal[0]))
    return dict(status='PASS_PCIE_INVENTORY_OWNERSHIP_INTEGRATION',
                rows=n,scenarios=scenarios,retired_rows=retired,cold_canceled_rows=canceled,
                table_word_comparisons=words,empty_rows=empty,immediate_revocations=revocations,
                diagnostic_wraps=wraps,all_state_and_fault_rows_compared=True,protocol_reference=protocol,
                actual_mutants=rejected,reference_is_testbench_only=True,physical_timing_qualified=False)
