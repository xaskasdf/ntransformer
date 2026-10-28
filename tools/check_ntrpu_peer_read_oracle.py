"""Replay successful RTL transfers through the independent transport model.

The RTL's opaque handles are fixture-local, not the Python model's allocated
handles. Source node, sequence, offsets, bytes and observable lifetime are mapped
explicitly; this is not a general equivalence proof or a PCIe model.
"""
import argparse
import hashlib
import json
from pathlib import Path
from ntrpu_transport_model import TransportModel, TransportError


def replay(text):
    model = TransportModel(capacities=(1 << 42, 1 << 42), queue_depth=2,
                           fragment_bytes=64, max_transfer=65536)
    origins = [model.register(l, l*65536, 65536) for l in range(2)]
    targets = [model.register(l^1, (1 << 40)+l*65536, 65536) for l in range(2)]
    windows = [model.bind_window(t) for t in targets]
    for lane, source in enumerate(origins):
        for offset in range(0,65536,4096):
            model.host_write(source, offset, bytes(((a ^ (a>>8) ^ (a>>16) ^ (lane*91)) & 255)
                for a in range(lane*65536+offset,lane*65536+offset+4096)))
    jobs = {}
    confirmed = set()
    copied = retired = 0
    for line in text.splitlines():
        p = line.split()
        if len(p)<6 or p[0]!='PR' or p[2]!='2' or p[1] not in ('CMD','DATA','LOCAL','DMA','ACK','RESULT'):
            continue
        kind=p[1];lane=int(p[4]);seq=int(p[5]);key=(lane,seq)
        if kind=='CMD':
            address=int(p[6]);length=int(p[7])*64
            identity=model.submit(1,seq,origins[lane],windows[lane],length,
                                  source_offset=address-lane*65536,target_offset=address-lane*65536)
            jobs[key]=identity
        elif kind=='DATA':
            packet=model.send(jobs[key],int(p[6])-1)
            # Public RTL bytes must equal the oracle's actual issued payload.
            actual=bytes.fromhex(p[8])[::-1]
            assert actual==model._packets[packet].payload
            model.receive(packet);model.commit(packet);copied+=len(actual)
        elif kind=='ACK':
            result=model.confirm_remote(jobs[key]);confirmed.add(key)
            assert result.status=='SUCCESS' and result.valid_bytes==int(p[7])
        elif kind in ('LOCAL','DMA'):
            # These boundaries alone must not publish an unacknowledged result.
            try:
                model.completion(jobs[key]);assert key in confirmed
            except TransportError as error:
                assert error.code=='NOT_COMPLETE' and key not in confirmed
        elif kind=='RESULT':
            result=model.retire(jobs[key]);assert result.status=='SUCCESS'
            assert result.valid_bytes==int(p[7]);retired+=1
    assert retired==20 and copied==37376
    assert model.resources()['queue_used']==(0,0) and model.resources()['packets']==0
    assert all(not r.readers and not r.writers for regions in model._regions for r in regions.values())
    return dict(status='PEER_READ_NOMINAL_ORACLE_REPLAY_PASS',transfers=retired,
                compared_bytes=copied,scope='case 2 only; exact payload and remote-confirmation lifetime')


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('log',type=Path);a=ap.parse_args()
    result=replay(a.log.read_text());result['log_sha256']=hashlib.sha256(a.log.read_bytes()).hexdigest()
    print(json.dumps(result,indent=2))
