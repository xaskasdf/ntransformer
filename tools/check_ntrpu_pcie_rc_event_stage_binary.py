"""Replay packet lifetime/validation in Python and compare every registered row.

The metadata provider is the unchanged R7 in the bound testbench. Its low
25 bits are crosschecked against the independently reviewed shadow context;
this does not replace production integration of the metadata bank.
"""
import re
from check_ntrpu_pcie_rc_proposal_shadow_binary import review as shadow_review, bits, field
from check_ntrpu_pcie_rc_proposal import calculate


def entries(log, prefix):
    result=[]
    for line in log.splitlines():
        if line.startswith(prefix+' '):
            pairs=[s.split('=') for s in line.split()[1:]]
            if any(len(p)!=2 for p in pairs) or len({p[0] for p in pairs})!=len(pairs):
                raise ValueError('Malformed '+prefix)
            result.append(dict(pairs))
    return result


def number(row,key):
    try:return int(row[key],16)
    except (ValueError,KeyError) as exc:raise ValueError('Unknown or missing output '+key) from exc


def active(context):
    return context is not None and context['active']


def model_beat(record, context, lost):
    out=dict.fromkeys(['accepted','headers','sequence','update','complete','unresolved','checked',
                      'tags','remaining','address','removed','lanes','origins','beat_faults','h_faults'],0)
    if record is None or not record['accepted']:
        return out,context,lost
    out.update(accepted=1,headers=record['headers'],sequence=record['sequence'])
    u=record['user'];word=record['data'];frame=0
    starts={0:0,1:1,3:2,7:3,15:4}.get((u>>64)&15,-1)
    ends={0:0,1:1,3:2,7:3,15:4}.get((u>>76)&15,-1)
    if starts<0 or ends<0:frame|=1<<2
    sop=[4*((u>>(68+2*s))&3) for s in range(max(starts,0))]
    eop=[(u>>(80+4*s))&15 for s in range(max(ends,0))]
    if sop!=sorted(set(sop)) or eop!=sorted(set(eop)):frame|=1<<2
    end_slots={lane:s for s,lane in enumerate(eop)}
    discontinue=bool((u>>96)&1)
    if (not active(context) and starts==0) or (discontinue and (starts!=0 or ends==0)):frame|=1<<2
    if discontinue:frame|=1<<12
    if context is not None:
        context['bad']|=frame
        if discontinue:context['normal']=False
    faults=frame
    for lane in range(16):
        if lane in sop:
            q=lane//4;h=bits(word,q*128,96);metadata=(record['metadata']>>(q*30))&((1<<30)-1)
            normal=bits(h,12,4)=='0000' and bits(h,43,3)=='000' and bits(h,46,1)=='0'
            owned=bool((record['owned']>>q)&1);bad=0
            if active(context):
                bad|=1<<2;lost|=context['owned']
            error=bits(h,12,4)
            if set(error)-{'0','1'}:bad|=1<<4
            check_bits=bits(h,64,8)+bits(h,48,8)+bits(h,30,1) if error=='1001' else h
            if set(check_bits)-{'0','1'}:bad|=1<<4
            if not owned:bad|=1<<6
            if bits(h,48,3)!=format(record['requester']&7,'03b'):bad|=1<<7
            if not normal:bad|=1<<11
            p=calculate(int(h,2)) if normal else dict(delivered=0,remaining=0,address=0,length=0,terminal=0)
            if normal and owned:
                if not (record['expected_valid']>>q)&1:bad|=1<<14
                if (field(h,48,16),field(h,89,3),field(h,92,3))!=(record['requester'],(metadata>>25)&7,(metadata>>28)&3):bad|=1<<7
                if (field(h,16,13),field(h,0,12))!=(metadata&8191,(metadata>>13)&4095):bad|=1<<8
                if p['length']:bad|=1<<10
                if p['terminal']:bad|=1<<9
            out['h_faults']|=bad<<(q*16);faults|=bad
            context=dict(active=True,owned=owned,normal=normal,ready=normal,terminal=bits(h,30,1)=='1',tag=field(h,64,8),
                original=field(h,16,13) if normal else None,lower=field(h,0,12) if normal else 0,
                delivered=p['delivered'],remaining=p['remaining'],address=p['address'],
                expected=(field(h,32,11)+3)%2048 if normal else None,seen=0,
                bad=bad|frame,origin=record['sequence'])
        enable=(u>>(4*lane))&15
        if active(context):
            seen=context['seen'];expected_enable=0
            if seen>=3 and context['normal']:
                offset=(seen-3)*4
                start=context['lower']%4;stop=start+context['delivered']
                expected_enable=sum(1<<j for j in range(4) if start<=offset+j<stop)
            if (seen<3 and enable!=0) or (seen>=3 and context['normal'] and enable!=expected_enable):
                faults|=1<<10;context['bad']|=1<<10
            if seen>=3 and context['normal']:
                for j in range(4):
                    if (expected_enable>>j)&1 and set(bits(word,lane*32+j*8,8))-{'0','1'}:
                        faults|=1<<4;context['bad']|=1<<4
            context['seen']=(seen+1)%2048
            if context['seen']>1027:faults|=1<<10;context['bad']|=1<<10
        elif enable!=0:
            faults|=1<<10
        if lane in end_slots:
            s=end_slots[lane]
            if context is None:
                faults|=1<<2
            else:
                if not active(context) or context['seen']<3:faults|=1<<2;context['bad']|=1<<2
                if context['normal'] and context['seen']!=context['expected']:
                    faults|=1<<10;context['bad']|=1<<10
                if active(context) and context['owned']:
                    out['tags']|=context['tag']<<(s*8);out['lanes']|=lane<<(s*4)
                    out['origins']|=context['origin']<<(s*32)
                    if not context['ready']:out['unresolved']|=1<<s;lost=True
                    elif context['terminal']:
                        out['complete']|=1<<s;out['removed']|=context['original']<<(s*13)
                    elif context['normal']:
                        out['update']|=1<<s;out['remaining']|=context['remaining']<<(s*13)
                        out['address']|=context['address']<<(s*12);out['removed']|=context['delivered']<<(s*13)
                    if context['ready'] and context['bad']==0 and (context['normal'] or context['terminal']):out['checked']|=1<<s
            if context is not None:
                context.update(active=False,owned=False,seen=0)
    lost|=bool(faults&(1<<2));out['beat_faults']=faults
    return out,context,lost


def review(logs,vectors,manifest,baseline_log,require_mutants=False):
    log=logs['normal/simulate.log']
    if log.splitlines().count('RC_STAGE_MODE mode=0')!=1 or log.splitlines().count('RC_STAGE_SUITE_PASS rows=22330 scenarios=67 mode=0 latency=2')!=1:
        raise ValueError('Missing stage scope/completion')
    shadow=shadow_review(log,vectors,manifest,baseline_log)
    rows=entries(log,'RC_STAGE_ROW');providers=entries(log,'RC_STAGE_EXPECT')
    contexts=entries(log,'RC_PROP_CONTEXT');headers=entries(log,'RC_PROP_ROW');public=entries(log,'PCIE_INV_ROW')
    inputs=vectors.splitlines()
    if len(rows)!=len(inputs) or len(providers)!=len(inputs):raise ValueError('Missing/duplicate stage records')
    previous=None;context=None;lost=False;latched=0
    accepted=updates=completions=unresolved=checked=retained=reset_flushes=0
    fault_records=[];bad_headers=[];origin_cross_beat=0
    row_keys={'tick','source','sequence','accepted','headers','update','complete','unresolved','checked','tags','remaining','address','removed','lanes','origins','active','owned','loss','beat_faults','faults','pending','h_faults'}
    for i,(line,row,provider,shadow_ctx,prop,pub) in enumerate(zip(inputs,rows,providers,contexts,headers,public)):
        v=line.split();rst=int(v[2]);scenario=int(v[0])
        if set(row)!=row_keys or int(row['tick'])!=i or int(row['source'])!=i-1 or int(provider['index'])!=i:
            raise ValueError('Stage sequence/schema mismatch')
        current=dict(accepted=number(provider,'accepted'),headers=number(provider,'headers'),owned=number(provider,'owned'),
                     metadata=number(provider,'metadata'),requester=number(provider,'requester'),sequence=number(provider,'sequence'),
                     expected_valid=number(provider,'headers'),user=int(v[15],16),data=v[14],scenario=scenario)
        if current['headers']!=number(prop,'seen') or current['owned']!=number(shadow_ctx,'owned') or current['sequence']!=(0xffffff00+i)%2**32:
            raise ValueError('Provider not correlated with raw input')
        prior_observing=int(public[i-1]['observing']) if i else 0
        if current['accepted']!=int(bool(rst and prior_observing and v[12]=='1' and v[13]=='1')):
            raise ValueError('Provider handshake mismatch')
        for q in range(4):
            expected25=((number(shadow_ctx,'address')>>(12*q))&4095)<<13 | ((number(shadow_ctx,'original')>>(13*q))&8191)
            if (current['metadata']>>(30*q))&((1<<25)-1)!=expected25:
                raise ValueError('Provider expected metadata mismatch')
        if not rst:
            reset_flushes+=int(previous is not None and previous['accepted'])
            expected,context,lost=model_beat(None,None,False);latched=0
        else:
            expected,context,lost=model_beat(previous,context,lost)
            latched|=expected['beat_faults']
        expected.update(active=int(active(context)),owned=int(active(context) and context['owned']),
                        loss=int(lost),faults=latched)
        expected['pending']=int(bool(current['accepted'] or expected['accepted'] or active(context) or lost or
                                     (int(pub['observing']) and v[12]=='1')))
        for key,val in expected.items():
            if number(row,key)!=val:raise ValueError(f'Stage mismatch tick={i} source={i-1} field={key}: {row[key]} != {val:x}')
        accepted+=expected['accepted'];updates+=expected['update'].bit_count();completions+=expected['complete'].bit_count()
        unresolved+=expected['unresolved'].bit_count();checked+=expected['checked'].bit_count();retained+=expected['active']
        for s in range(4):
            if (expected['update']|expected['complete']|expected['unresolved'])&(1<<s):
                origin_cross_beat+=int((expected['origins']>>(32*s))&0xffffffff != expected['sequence'])
        if expected['beat_faults']:
            fault_records.append(dict(tick=i,source=i-1,scenario=previous['scenario'],faults=hex(expected['beat_faults'])))
        if expected['h_faults']:
            bad_headers.append(dict(tick=i,source=i-1,scenario=previous['scenario'],faults=hex(expected['h_faults'])))
        previous=current
    mutants=[]
    normal_public=[s for s in log.splitlines() if s.startswith('PCIE_INV_ROW ')]
    for name,mode in ([('omit_rc',1),('missing_metadata',2)] if require_mutants else []):
        text=logs[name+'/simulate.log']
        if text.splitlines().count(f'RC_STAGE_MODE mode={mode}')!=1 or 'RC_STAGE_SUITE_PASS' in text:
            raise ValueError('Wrong mutant mode/completion')
        fatals=[s for s in text.splitlines() if s.startswith('Fatal:')]
        pattern=(r'Fatal: RC_STAGE_FALSE_IDLE tick=\d+' if mode==1 else
                 r'Fatal: RC_STAGE_VALIDATION_MISMATCH tick=\d+ source=\d+ mode=2')
        if len(fatals)!=1 or not re.fullmatch(pattern,fatals[0]):
            raise ValueError('Missing expected actual mutation failure')
        tick=int(re.search(r'tick=(\d+)',fatals[0])[1])
        prefix=[s for s in text.splitlines() if s.startswith('PCIE_INV_ROW ')]
        if prefix!=normal_public[:tick]:raise ValueError('Mutant reference prefix changed')
        mutants.append(dict(name=name,mode=mode,tick=tick,fatal=fatals[0]))
    return dict(status=('PASS_PCIE_RC_EVENT_STAGE_REFERENCE_AND_MUTATIONS' if require_mutants else
                        'REVIEWED_PCIE_RC_EVENT_STAGE_NORMAL_ONLY'),rows=len(rows),scenarios=67,
        accepted_retired_beats=accepted,partial_updates=updates,completions=completions,
        unresolved_events=unresolved,packet_checked_events=checked,context_active_cycles=retained,
        cross_beat_event_origins=origin_cross_beat,cold_reset_cancelled_captured_beats=reset_flushes,
        beat_fault_records=fault_records,header_fault_records=bad_headers,
        all_registered_rows_independently_replayed=True,latency_cycles=2,sequence_modulus=2**32,
        diagnostic_sequence_not_an_ownership_key=True,shadow=shadow,actual_mutants=mutants,
        metadata_and_ownership_provider='Unchanged R7 testbench reference',
        production_context_and_validation_rtl=True,full_pipeline_integrated=False)
