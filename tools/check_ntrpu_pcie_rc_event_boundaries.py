"""Check directed event-stage boundaries against the independent packet model."""
import re
from check_ntrpu_pcie_rc_event_stage import model_beat, active, entries, number


def review(log,vectors,manifest):
    if re.search(r'^\s*(Fatal:|Failure:|ERROR:|FAIL )',log,re.M):raise ValueError('Simulation diagnostic')
    rows=entries(log,'RC_BOUNDARY_ROW');inputs=vectors.splitlines()
    if len(rows)!=len(inputs) or len(rows)!=manifest['rows']:raise ValueError('Missing/duplicate boundary rows')
    marker=f'RC_BOUNDARY_SUITE_COMPLETE rows={len(rows)} phases={len(manifest["phases"])} latency=2 native=0'
    if log.splitlines().count(marker)!=1:raise ValueError('Missing boundary completion/scope')
    context=previous=None;lost=False;latched=0;capture_cancellations=context_cancellations=wrap_completions=0
    stats={p['id']:dict(name=p['name'],rows=0,accepted=0,updates=0,complete=0,checked=0,unresolved=0,fault_rows=0) for p in manifest['phases']}
    faults=[]
    keys={'tick','phase','sequence','accepted','headers','update','complete','unresolved','checked','tags','remaining','address','removed','lanes','origins','active','owned','loss','beat_faults','faults','pending','h_faults'}
    for i,(raw,row) in enumerate(zip(inputs,rows)):
        p,rst,obs,valid,ready,owned,rid,seq,mv,metadata,data,user=raw.split()
        phase=int(p);rst=int(rst);obs=int(obs);valid=int(valid);ready=int(ready);u=int(user,16)
        if set(row)!=keys or int(row['tick'])!=i or int(row['phase'])!=phase:raise ValueError('Boundary sequence/schema mismatch')
        accepted=int(bool(rst and obs and valid and ready))
        starts={0:0,1:1,3:2,7:3,15:4}.get((u>>64)&15,-1)
        mask=sum(1<<q for q in {((u>>(68+2*s))&3) for s in range(max(starts,0))}) if accepted else 0
        current=dict(accepted=accepted,headers=mask,owned=int(owned,16),requester=int(rid,16),sequence=int(seq,16),
            expected_valid=int(mv,16),metadata=int(metadata,16),data=format(int(data,16),'0512b'),user=u,phase=phase)
        if not rst:
            capture_cancellations+=int(previous is not None and previous['accepted'])
            context_cancellations+=int(active(context))
            expected,context,lost=model_beat(None,None,False);latched=0
        else:
            expected,context,lost=model_beat(previous,context,lost)
            latched|=expected['beat_faults']
        expected.update(active=int(active(context)),owned=int(active(context) and context['owned']),loss=int(lost),faults=latched)
        expected['pending']=int(bool(accepted or expected['accepted'] or active(context) or lost or (obs and valid)))
        for k,v in expected.items():
            if number(row,k)!=v:raise ValueError(f'Boundary mismatch row={i} phase={phase} field={k}: {row[k]} != {v:x}')
        st=stats[phase];st['rows']+=1;st['accepted']+=expected['accepted']
        for k,out in [('updates','update'),('complete','complete'),('checked','checked'),('unresolved','unresolved')]:st[k]+=expected[out].bit_count()
        if expected['beat_faults']:
            st['fault_rows']+=1;faults.append(dict(row=i,phase=phase,faults=hex(expected['beat_faults'])))
        for s in range(4):
            if expected['complete']&(1<<s) and ((expected['origins']>>(s*32))&0xffffffff)>expected['sequence']:
                wrap_completions+=1
        previous=current
    if any(stats[p['id']]['rows']!=p['rows'] for p in manifest['phases']):raise ValueError('Phase coverage missing')
    if capture_cancellations!=2 or context_cancellations!=1 or wrap_completions!=1:raise ValueError('Reset/wrap coverage missing')
    if (stats[5]['accepted'],stats[5]['updates'],stats[5]['complete'],stats[5]['checked'],stats[5]['fault_rows'])!=(256,1023,1,1024,0):
        raise ValueError('Sustained four-lane same-tag coverage missing')
    if any(stats[p]['fault_rows'] for p in [0,2,3,4,5,8,14]):raise ValueError('Unexpected healthy boundary fault')
    if any(stats[p]['fault_rows']==0 for p in [1,6,7,9,10,11,12,13]):raise ValueError('Missing negative boundary detection')
    return dict(status='PASS_PCIE_RC_EVENT_STAGE_BOUNDARIES',rows=len(rows),phases=stats,
        cold_reset_cancelled_captured_beats=capture_cancellations,cold_reset_cancelled_retained_contexts=context_cancellations,
        cross_wrap_completions=wrap_completions,sustained_four_same_tag_beats=256,
        fault_rows=faults,all_registered_outputs_checked=True,binary_only=True,
        full_inventory_pipeline_qualified=False,native_endpoint_qualified=False)
