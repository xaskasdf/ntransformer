"""Independently replay the descriptor, reservation, fetch and fence boundary."""
import argparse
from collections import Counter,deque
import hashlib
import json
from pathlib import Path
import re

FIELDS='cycle case depth reset stop fault_in sv sr sd hv can queued take cv cr cc fc cd fv fr dr fd dv sink tag epoch beats bv br bid ov or oid fault empty drained'.split()
HEX={'sd','queued','cd','fd','tag','epoch','beats','bid','oid'}
DEC={'cycle','case','depth'}


def need(ok,message):
    if not ok:raise ValueError(message)


def edges(text):
    out=[]
    for line in text.splitlines():
        if not line.startswith('DQ E '):continue
        values=line.split()[2:]
        need(len(values)==len(FIELDS),'edge schema')
        out.append({k:int(v,16 if k in HEX else 10 if k in DEC else 2) for k,v in zip(FIELDS,values)})
    return out


def replay(rows,depth):
    q=[deque(),deque()];lease=None;rr=0;sticky=False;barrier=False;complete=False;fence=0
    counts=Counter();coverage=Counter();seen=set();last=-1;previous_case=None;case_counts=Counter();case_totals=[]
    def finish_case(number):
        jobs=4*depth if number==1 else 2*(5*depth+3) if number==2 else 2
        frames=jobs if number<=4 else 1
        need(case_counts['submitted']==jobs and case_counts['frames']==frames,'exact case workload')
        need(lease is None and [len(x) for x in q]==([0,0] if number<=4 else [0,1]),'case drain or explicit quarantine')
        need(not barrier and not complete and sticky==(number>=5),'case fault and barrier terminal state')
        case_totals.append(dict(case=number,submitted=jobs,completed_frames=frames,quarantined=jobs-frames))
    for row in rows:
        v=row;cycle=v['cycle'];case=v['case'];seen.add(case)
        if previous_case is not None and case!=previous_case:
            finish_case(previous_case);case_counts=Counter()
        previous_case=case
        need(cycle==last+1 and v['depth']==depth,'cycle/depth continuity');last=cycle
        if not v['reset']:
            need(not(v['sv']&v['sr'] or v['take'] or v['cv'] or v['fv'] or v['dr'] or v['ov']),'reset closes interfaces')
            q=[deque(),deque()];lease=None;rr=0;sticky=False;barrier=False;complete=False;fence=0
            continue
        fault=sticky or bool(v['fault_in'])
        empty=not any(q) and lease is None
        need(v['fault']==fault and v['empty']==empty,'fault/empty truth')
        need(v['drained']==(empty and bool(v['sink']) and not barrier),'drain includes queued, lease, sink and fence')
        need(v['hv']==sum((1<<ch) for ch in (0,1) if q[ch]),'queue visibility')
        need(v['queued']==len(q[0])+(len(q[1])<<depth.bit_length()),'queue accounting')
        ready=sum(1<<ch for ch in (0,1) if len(q[ch])<depth and not(v['stop'] or fault or barrier or v['bv']))
        need(v['sr']==ready,'submission admission')
        eligible=[ch for ch in (rr,1-rr) if q[ch] and v['can']>>ch&1]
        take=(1<<eligible[0]) if lease is None and not fault and eligible else 0
        need(v['take']==take,'atomic reservation and round robin')
        need(v['br']==(not fault and not barrier),'fence admission')
        need(v['ov']==complete and (not complete or v['oid']==fence),'stable fence result')
        phase=None if lease is None else lease['phase']
        need((v['cv'],v['fv'],v['dr'])==(phase=='command',phase=='fetch',phase=='frame'),'lease protocol phase')
        if lease:
            need(v['cc']==v['fc']==lease['channel'] and v['cd']==v['fd']==lease['desc'],'selected metadata stability')
        if v['cv'] and not v['cr']:coverage['command_stall']+=1
        if v['fv'] and not v['fr']:coverage['fetch_stall']+=1
        if v['dr'] and not v['dv']:coverage['frame_wait']+=1
        if complete and not v['or']:coverage['fence_result_stall']+=1
        if barrier and empty and not v['sink']:coverage['fence_wait_sink']+=1
        if case==1 and v['take']==2 and len(q[0])==depth:coverage['full_peer_bypassed']+=1
        if v['hv']==3 and v['can']==3 and take:coverage['both_eligible']+=1
        if v['stop'] and take:coverage['stop_drains_queue']+=1
        if fault and phase=='command' and not v['cr']:coverage['fault_command_held']+=1
        if fault and lease is None and any(q):coverage['fault_queue_quarantine']+=1
        # Fence captures a past drain event; its result remains valid afterward.
        if barrier and not complete and empty and v['sink'] and not fault:complete=True
        if v['bv'] and v['br']:
            barrier=True;fence=v['bid'];counts['fences']+=1
        if v['ov'] and v['or']:
            barrier=False;complete=False;counts['fence_results']+=1
        sticky=fault
        if take:
            ch=take.bit_length()-1;desc=q[ch].popleft()
            lease=dict(channel=ch,desc=desc,phase='command');rr=1-ch
            counts['reservations']+=1
        elif phase=='command' and v['cr']:
            lease['phase']='fetch';counts['commands']+=1
        elif phase=='fetch' and v['fr']:
            lease['phase']='frame';counts['fetches']+=1
        elif phase=='frame' and v['dv']:
            d=lease['desc']
            match=(v['tag']==(d>>136)&((1<<64)-1) and v['epoch']==d>>200 and v['beats']==(d&255)+1)
            if match:lease=None;counts['frames']+=1;case_counts['frames']+=1
            else:sticky=True;counts['bad_done']+=1
        for ch in (0,1):
            if (v['sv'] & v['sr']) & (1<<ch):
                d=(v['sd']>>(ch*232))&((1<<232)-1)
                need((d&255)<64,'prevalidated fixture length');q[ch].append(d);counts['submitted']+=1;case_counts['submitted']+=1
        need(all(len(x)<=depth for x in q),'bounded queues')
    finish_case(previous_case)
    need(seen==set(range(1,9)),'eight cases present')
    need(counts['bad_done']==3 and counts['fences']==counts['fence_results']==1,'fault and fence coverage')
    need(counts['reservations']==counts['commands']==counts['fetches']==counts['frames'],'every reserved lease completed')
    need(all(coverage[k]>0 for k in ['command_stall','fetch_stall','frame_wait','fence_result_stall',
        'fence_wait_sink','full_peer_bypassed','both_eligible','stop_drains_queue','fault_command_held','fault_queue_quarantine']),'required scenarios exercised')
    return dict(depth=depth,cycles=len(rows),counts=dict(counts),coverage=dict(coverage),cases=case_totals,
                scope='Receiver capacity and payload completion are supplied by a test double, not real R4/MIG/DMA.')


def negatives(rows,depth):
    out=[]
    for name in ('unstable_command','duplicate_reservation','premature_fetch','false_drain','early_fence'):
        changed=[dict(r) for r in rows]
        if name=='unstable_command':next(r for r in changed if r['cv'] and not r['cr'])['cd']^=1
        elif name=='duplicate_reservation':next(r for r in changed if r['take'])['take']=3
        elif name=='premature_fetch':next(r for r in changed if r['cv'])['fv']=1
        elif name=='false_drain':next(r for r in changed if r['reset'] and r['hv'])['drained']=1
        else:next(r for r in changed if r['case']==4 and r['reset'] and not r['sink'])['ov']=1
        try:replay(changed,depth)
        except ValueError as e:out.append(dict(mutation=name,rejection=str(e)))
        else:raise ValueError('negative control missed '+name)
    return out


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path)
    args=ap.parse_args();root=args.run;output=root/'output'
    record=json.loads((output/'result.json').read_text())
    need(record['status']=='EXPORTED_PENDING_REVIEW' and record['returncode']==0 and not record['timed_out'],'runner completion')
    need(record['sources']==record['sources_after'],'snapshot preserved')
    for name,h in record['sources'].items():
        need(hashlib.sha256((root/'inputs'/name).read_bytes()).hexdigest()==h,'input source hash')
    reports=[];controls=[]
    for depth in (1,3,8):
        text=(output/f'suite/depth-{depth}.log').read_text()
        need(not re.search(r'(?mi)^\s*(?:Fatal:|ERROR:|FAIL )',text),'simulation diagnostics')
        need(f'DQ SUITE_PASS depth={depth} cases=8 ' in text,'terminal PASS')
        need(sorted(map(int,re.findall(r'DQ CASE_PASS (\d+)',text)))==list(range(1,9)),'case markers')
        rows=edges(text);reports.append(replay(rows,depth))
        controls+=negatives(rows,depth)
    structure=(output/'suite/structure.txt').read_text()
    need(structure=='LATCHES 0\nBLACKBOXES 0\n','no latches or blackboxes')
    console=(output/'console.log').read_text()
    need('NTRPU_DESTINATION_DISPATCH_EXPORTED' in console and not re.search(r'(?mi)^\s*(ERROR:|CRITICAL WARNING:)',console),'Vivado completion')
    result=dict(status='ISOLATED_DISPATCH_PROTOCOL_PASS_NOT_INTEGRATED',depths=reports,
                negative_controls=controls,seconds=record['seconds'],cpus=record['cpus'],
                logical_descriptor_bits=2*8*232,
                checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                limitations=['Atomic reservation adapter is required; banks_used or cmd_ready are not sufficient.',
                    'No actual payload/DMA/AXI/MIG traffic, CDC, board integration or routed timing is tested.',
                    'Epoch is opaque at submission; full epoch/window validation remains upstream.',
                    'Faulted unreserved queues remain quarantined; reset in the test starts an independent modeled epoch.'])
    with (root/'REVIEW.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
