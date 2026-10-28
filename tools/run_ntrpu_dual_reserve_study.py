"""Compare reserve depth and bounded upstream selection in a local cycle model."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from ntrpu_buffer_pool_model import Config, Job, Stalls, simulate


def cases():
    for count in (16,128):
        for split in ('balanced','3to1','clustered'):
            jobs = [Job(i, i%2 if split=='balanced' else int(i%4==3) if split=='3to1'
                        else int(i>=count//2), 64) for i in range(count)]
            for kind, pause_us in [('none',0)]+[(k,u) for k in ('w','b') for u in (4,8,16)]:
                stalls = Stalls(**({kind:((0,0,pause_us*200),)} if pause_us else {}))
                for reserve in (4,6,8):
                    for policy, window in [('ordered',None),('window2',2),('window8',8),('unbounded',None)]:
                        config = Config(reserve=reserve, descriptors=16, result_slots=16,
                                        ingress='ordered' if policy=='ordered' else 'eligible',
                                        ingress_window=window)
                        yield dict(count=count, split=split, pause_kind=kind,
                                   pause_us=pause_us, policy=policy), config, jobs, stalls


def summarize(result, config, pause_us):
    rows = result['trace']
    period_ns = 1000/config.clock_mhz
    end = pause_us*config.clock_mhz
    return dict(
        logical_pool_KiB=(2*config.channels+config.reserve)*config.bank_bytes//1024,
        blocked_ns=result['metrics']['counters'].get('ingress_blocked_cycles',0)*period_ns,
        completion_ns=result['metrics']['cycles']*period_ns,
        last_ingress_ns=(result['metrics']['last_ingress_cycle']+1)*period_ns,
        peer_completions_before_release=sum(r['channel']==1 and r['complete']<end for r in rows),
        peer_last_completion_ns=(max(r['complete'] for r in rows if r['channel']==1)+1)*period_ns,
        peak_descriptors=[max(sum(q['channel']==ch and q['admitted']<=r['admitted']<q['b'] for q in rows)
                              for r in rows) for ch in (0,1)],
        peak_outstanding=[max(sum(q['channel']==ch and q['aw']<=r['aw']<q['b'] for q in rows)
                              for r in rows) for ch in (0,1)])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    sources = ['tools/ntrpu_buffer_pool_model.py','tools/run_ntrpu_dual_reserve_study.py',
               'tools/check_ntrpu_dual_reserve_study.py','tests/test_ntrpu_pool_lookahead.py',
               'tests/test_ntrpu_buffer_pool_model.py','docs/fpga/PEER_RX_DUAL_PERFORMANCE.md']
    hashes = {}
    for name in sources:
        payload = (root/name).read_bytes()
        p = args.output/'inputs'/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(payload)
        hashes[name] = hashlib.sha256(payload).hexdigest()
    started = time.perf_counter()
    output = []
    for i,(dimensions,config,jobs,stalls) in enumerate(cases()):
        result = simulate(config,jobs,stalls,max_cycles=30000)
        assert result['metrics']['drained'] and result['metrics']['payload_bytes']==len(jobs)*4096
        output.append(dict(id=i,**dimensions,config=asdict(config),stalls=asdict(stalls),
                           jobs=[asdict(j) for j in jobs],**result,
                           derived=summarize(result,config,dimensions['pause_us'])))
        if (i+1)%84==0:
            print(json.dumps(dict(completed=i+1,seconds=round(time.perf_counter()-started,2))),flush=True)
    report = dict(status='MODEL_SWEEP_COMPLETE_PENDING_REVIEW_NOT_RTL',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  wall_seconds=time.perf_counter()-started,inputs_sha256=hashes,
                  case_count=len(output),cases=output,
                  assumptions=[
                      'Ideal arbitration, same-cycle resource reuse, two independent read services; no command-cycle, RAM pipeline or CDC cost.',
                      'All descriptors are offered at cycle zero, and payload remains fetchable upstream in selected order.',
                      'A finite window sees only the first N unadmitted jobs; per-destination order is preserved.',
                      'Fixed 32-cycle B delay, 4 outstanding writes and 16 descriptor/result reservations per destination.',
                      'Model banks release on modeled WLAST; routed hardware has additional elastic and CDC queue storage.',
                      'One 4/8/16-us pause at the start of each case; 200 MHz converts model cycles only.',
                      'No RTL, MIG, PCIe, HBM, timing closure, power or added hardware queue cost is measured.'])
    (args.output/'RESULTS.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    assert all(hashlib.sha256((root/n).read_bytes()).hexdigest()==h for n,h in hashes.items())
    print(json.dumps(dict(status=report['status'],cases=len(output),seconds=report['wall_seconds'])))


if __name__=='__main__':
    main()
