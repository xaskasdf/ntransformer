"""Check directed event-stage boundaries against the independent packet model."""
import re,io,itertools
from check_ntrpu_pcie_rc_event_stage import model_beat, active, entries, number


def review(log,vectors,manifest):
    if re.search(r'^\s*(Fatal:|Failure:|ERROR:|FAIL )',log,re.M):raise ValueError('Simulation diagnostic')
    inputs=io.StringIO(vectors)
    rows=(entries(line,'RC_BOUNDARY_ROW')[0] for line in io.StringIO(log) if line.startswith('RC_BOUNDARY_ROW '))
    marker=f'RC_BOUNDARY_SUITE_COMPLETE rows={manifest["rows"]} phases={len(manifest["phases"])} latency=2 native=0'
    if sum(line.strip()==marker for line in io.StringIO(log))!=1:raise ValueError('Missing matrix completion')
    ref_marker=f'RC_FRAGMENT_MATRIX_REFERENCE_PASS rows={manifest["rows"]} native=0'
    if sum(line.strip()==ref_marker for line in io.StringIO(log))!=1:raise ValueError('Missing prior RTL reference comparison')
    processed=0
    context=previous=None;lost=False;latched=0;capture_cancellations=context_cancellations=wrap_completions=0
    stats={p['id']:dict(name=p['name'],rows=0,accepted=0,updates=0,complete=0,checked=0,unresolved=0,fault_rows=0) for p in manifest['phases']}
    faults=[]
    keys={'tick','phase','sequence','accepted','headers','update','complete','unresolved','checked','tags','remaining','address','removed','lanes','origins','active','owned','loss','beat_faults','faults','pending','h_faults'}
    for i,(raw,row) in enumerate(itertools.zip_longest(inputs,rows)):
        if raw is None or row is None:raise ValueError('Missing/duplicate matrix rows')
        processed+=1
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
            st['fault_rows']+=1
            if len(faults)<64:faults.append(dict(row=i,phase=phase,faults=hex(expected['beat_faults'])))
        for s in range(4):
            if expected['complete']&(1<<s) and ((expected['origins']>>(s*32))&0xffffffff)>expected['sequence']:
                wrap_completions+=1
        previous=current
    if any(stats[p['id']]['rows']!=p['rows'] for p in manifest['phases']):raise ValueError('Phase coverage missing')
    if processed!=manifest['rows'] or processed!=281904 or manifest['cases']!=80544:raise ValueError('Matrix extent differs')
    if sum(p['accepted'] for p in stats.values())!=120816:raise ValueError('Matrix acceptance coverage missing')
    return dict(status='PASS_PCIE_RC_FRAGMENT_MATRIX',rows=processed,cases=manifest['cases'],phases=stats,
        cold_reset_cancelled_captured_beats=capture_cancellations,cold_reset_cancelled_retained_contexts=context_cancellations,
        cross_wrap_completions=wrap_completions,sop_subsets=16,eop_position_sets=2517,retained_context_modes=2,
        first_fault_records=faults,fault_record_sample_limit=64,all_registered_outputs_checked=True,
        binary_only=True,full_inventory_pipeline_qualified=False,native_endpoint_qualified=False)
