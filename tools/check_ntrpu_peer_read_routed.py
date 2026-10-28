"""Audit external descriptor/payload/DMA/ACK/result events of the routed DUT."""
import argparse
import hashlib
import json
from pathlib import Path
from check_ntrpu_peer_read_oracle import replay as oracle


def replay(text):
    assert 'PP ROUTED_PUBLIC_ONLY' in text and 'PP ALL_PASS cases=21 full=1' in text
    assert not any(s in text for s in ('PP_FAIL','PP_WATCHDOG','Fatal:'))
    for marker in ('CONFIG_PRIORITY_PASS','CONFIG_REVOKE_PASS','GLOBAL_STOP_PASS','REUSE_PASS transfers=16'):
        assert 'PP '+marker in text
    cases=set();quarantines=set();jobs={};rejected=results=payload=0;model_lines=[];model_jobs=set()
    for line in text.splitlines():
        p=line.split()
        if p[:2]==['PP','CASE']:
            assert p[3]=='PASS';cases.add(int(p[2]))
        if p[:2]==['PP','REJECT']:rejected+=1
        if p[:2]==['PP','QUARANTINE']:
            assert int(p[4])>0;quarantines.add(int(p[2]))
        if len(p)<6 or p[0]!='PP' or p[1] not in ('REQUEST','DATA','DMA','ACK','RESULT'):continue
        kind=p[1];case,epoch,lane,seq=map(int,p[2:6]);key=(epoch,lane,seq)
        if kind=='REQUEST':
            # Rejected replay attempts do not replace the original submission.
            jobs.setdefault(key,dict(address=int(p[6]),length=int(p[7]),received=0,dma=False,ack=False,result=False))
            if case==2 and key not in model_jobs:
                assert int(p[7])%64==0
                model_lines.append(' '.join(['PR','CMD',*p[2:7],str(int(p[7])//64)]));model_jobs.add(key)
            continue
        if case==2:model_lines.append(' '.join(['PR',*p[1:]]))
        if kind=='RESULT' and int(p[6])!=0:
            assert int(p[7])==0;results+=1;continue
        assert key in jobs,(kind,key)
        job=jobs[key]
        if kind=='DATA':
            beat,last=map(int,p[6:8]);raw=bytes.fromhex(p[8])[::-1]
            assert beat*64==job['received']+64 and len(raw)==64
            assert bool(last)==(beat*64==job['length'])
            for offset,value in enumerate(raw):
                address=job['address']+job['received']+offset
                assert value==((address^(address>>8)^(address>>16)^(lane*91))&255)
            job['received']+=64;payload+=64
        elif kind in ('DMA','ACK') and case==2:
            assert int(p[6])==0 and int(p[7])==job['length']==job['received']
            assert not job[kind.lower()];job[kind.lower()]=True
        elif kind=='RESULT':
            assert int(p[6])==0 and int(p[7])==job['length']==job['received']
            assert job['dma'] and job['ack'] and not job['result'];job['result']=True;results+=1
    assert cases==set(range(1,22)) and quarantines==set(range(4,22)) and rejected==29
    assert sum(j['result'] for j in jobs.values())==20
    return dict(cases=21,rejections=rejected,quarantines=len(quarantines),result_handshakes=results,
                payload_bytes=payload,oracle=oracle('\n'.join(model_lines)),
                oracle_scope='Public descriptors and payload; no internal command/CPL observation')


def controls(text):
    variants=[text.replace('PP ALL_PASS','PP NOT_PASS',1)]
    for prefix in ('PP REQUEST ','PP ACK 2 ','PP DMA 2 ','PP QUARANTINE 20 '):
        variants.append('\n'.join(line for line in text.splitlines() if not line.startswith(prefix)))
    for line in text.splitlines():
        if line.startswith('PP DATA 2 '):
            p=line.split();p[-1]=format(int(p[-1],16)^1,'0128x');variants.append(text.replace(line,' '.join(p),1));break
    assert len(variants)==6
    for variant in variants:
        try:replay(variant)
        except (AssertionError,KeyError,ValueError):continue
        raise AssertionError('Negative control accepted')
    return len(variants)


def review(out):
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    result=json.loads((out/'result.json').read_text())
    assert result['status']=='EXPORTED_PENDING_REVIEW' and result['returncode']==0 and not result['timed_out']
    assert result['sources']==result['sources_after'] and result['donor_unchanged']
    for name,digest in json.loads((out/'FILES.json').read_text()).items():assert sha(out/name)==digest,name
    comparison=json.loads((out/'suite/COMPARISON.json').read_text())
    assert comparison['netlist_sha256']==comparison['isolated_sha256']==sha(out/'suite/route_isolated.v')==result['netlist_sha256']
    assert comparison['public_ports_only'] and not comparison['netlist_edited']
    prior=json.loads((out/'PHYSICAL_REVIEW.json').read_text())
    assert prior['evidence_validated'] and prior['result_sha256']==result['prior_result_sha256']
    text=(out/'suite/simulation.log').read_text()
    return dict(status='PEER_READ_ROUTED_PUBLIC_REVIEWED',result_sha256=sha(out/'result.json'),
                **replay(text),negative_controls=controls(text),netlist_sha256=result['netlist_sha256'],
                timing_status=prior['status'],sdf=False,cycle_equivalence=False,board_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('output',type=Path);args=parser.parse_args()
    print(json.dumps(review(args.output),indent=2))
