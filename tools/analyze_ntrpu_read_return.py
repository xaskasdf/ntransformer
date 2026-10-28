"""Measure accepted command-to-completion windows in reviewed simulation traces.

These are finite testbench service times, not sustained throughput or board measurements.
"""
import argparse,json
from pathlib import Path
from check_ntrpu_read_return import sha,replay

def analyze(output,review_path):
    r=json.loads((output/'result.json').read_text());review=json.loads(review_path.read_text())
    assert review['result_sha256']==sha(output/'result.json')
    native=r['experiment']=='read_return_native'
    assert review['status']==('READ_RETURN_NATIVE_TRAFFIC_REVIEWED' if native else 'READ_RETURN_PREFLIGHT_REVIEWED')
    log=output/('evidence/simulate.log' if native else 'suite/simulation.log')
    lines=log.read_text().splitlines();replay(lines,1 if native else r['native_profile'])
    events=[s.split()[1:] for s in lines if s.startswith('RR ')]
    case=0;commands={};active={};rows=[]
    for e in events:
        if e[0]=='CASE':case=int(e[1])
        if case!=1 or e[0] not in ('CMD','AR','NR','DATA','CPL'):continue
        kind=e[0];ts,epoch,lane=map(int,e[1:4]);assert epoch==1
        if kind=='CMD':
            tag,n=int(e[4]),int(e[5]);assert tag not in commands and lane not in active
            commands[tag]=dict(lane=lane,beats=n,accepted_ps=ts,ar=[],nr=[],data=[]);active[lane]=tag
        else:
            c=commands[active[lane]]
            if kind in ('AR','NR','DATA'):c[kind.lower()].append(ts)
            else:
                assert kind=='CPL' and int(e[5])==0 and int(e[4])==active[lane]
                assert int(e[6])==c['beats']*64 and len(c['nr'])==c['beats']*2 and len(c['data'])==c['beats']
                c['completed_ps']=ts;del active[lane]
    assert not active and set(commands)==set(range(8))
    for tag,c in sorted(commands.items()):
        latency=c['completed_ps']-c['accepted_ps'];assert latency>0
        rows.append(dict(tag=tag,lane=c['lane'],beats=c['beats'],bytes=c['beats']*64,
            command_to_completion_ns=latency/1000,
            observed_window_gb_s=c['beats']*64*1000/latency,
            command_to_first_native_data_ns=(c['nr'][0]-c['accepted_ps'])/1000,
            last_native_to_first_output_ns=(c['data'][0]-c['nr'][-1])/1000,
            native_fragments=len(c['ar']),largest_native_beat_gap_ns=max((b-a for a,b in zip(c['nr'],c['nr'][1:])),default=0)/1000,
            output_span_including_last_cycle_ns=(c['data'][-1]-c['data'][0]+5000)/1000))
    pairs=[]
    for tag in range(0,8,2):
        a,b=commands[tag],commands[tag+1];assert a['beats']==b['beats']
        span=max(a['completed_ps'],b['completed_ps'])-min(a['accepted_ps'],b['accepted_ps'])
        pairs.append(dict(beats_per_channel=a['beats'],window_ns=span/1000,
                          observed_aggregate_window_gb_s=2*a['beats']*64*1000/span))
    return dict(status='READ_RETURN_SIMULATION_WINDOWS_MEASURED',result_sha256=review['result_sha256'],
                trace_sha256=sha(log),native_hbm_controller=native,source_clock_ns=5,commands=rows,pairs=pairs,
                scope='Case 1 only; already-written memory; no deliberate consumer stalls; finite service windows',
                excludes=['write preparation','steady state command stream','PCIe','physical hardware'],
                single_frame_per_channel_ideal_ceiling_gb_s=6.4,two_channel_ideal_ceiling_gb_s=12.8,
                ceiling_derivation='64 bytes * 200 MHz / 2 nonoverlapping receive/publish phases; actual overhead lowers it',
                board_throughput_measured=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();print(json.dumps(analyze(a.output,a.review),indent=2))
