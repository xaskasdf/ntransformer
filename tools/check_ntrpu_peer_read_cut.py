"""Replay peer transactions and independently audit the added return queue."""
import argparse
import hashlib
import json
from pathlib import Path

from check_ntrpu_peer_read import replay, controls
from check_ntrpu_read_cut import boundary, negative_controls


def retained_faults(lines):
    last, quarantines = {}, {}
    for line in lines:
        fields=line.split()
        if fields[:2]==['RC','STATE']:
            _,epoch,lane,count,_,_,sv,sr,mv,mr,pending=map(int,fields[2:])
            last[epoch,lane]=(count+int(bool(sv and sr))-int(bool(mv and mr)),pending)
        if fields[:2]==['PR','QUARANTINE']:
            quarantines[int(fields[3])]=int(fields[2])
    retained=[]
    for (epoch,lane),(count,pending) in last.items():
        if count:
            assert epoch in quarantines and pending==1,'Unaccounted queue data at epoch end'
            retained.append(dict(epoch=epoch,lane=lane,scenario=quarantines[epoch],count=count))
    return retained


def review(out, require_full=False):
    result = json.loads((out/'result.json').read_text())
    assert result['status']=='EXPORTED_PENDING_REVIEW' and result['returncode']==0
    assert result['full'] and result['pair'] and not result['timed_out']
    assert result['sources']==result['sources_after']
    for name, digest in json.loads((out/'FILES.json').read_text()).items():
        assert hashlib.sha256((out/name).read_bytes()).hexdigest()==digest, name
    text = (out/'suite/simulation.log').read_text()
    queue = boundary(text.splitlines())
    if require_full:
        assert queue['full']>0 and queue['held']>0
    return dict(status='PEER_READ_CUT_FUNCTIONAL_REVIEWED',
                result_sha256=hashlib.sha256((out/'result.json').read_bytes()).hexdigest(),
                peer=replay(text, True), peer_negative_controls=controls(text, True),
                boundary=queue, boundary_negative_controls=negative_controls(text.splitlines()),
                retained_fault_epochs=retained_faults(text.splitlines()),
                full_queue_required=require_full)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output',type=Path)
    parser.add_argument('--require-full',action='store_true')
    args=parser.parse_args()
    print(json.dumps(review(args.output,args.require_full),indent=2))
