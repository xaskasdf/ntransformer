"""Replay unchanged data reference plus the exact privileged MMIO fixture.

STOP marks command initiation at the named phase, not same-cycle application.
External isolation and memory responders remain testbench models.
"""
import argparse
import json
from pathlib import Path
import re
from check_ntrpu_pcie_ingress_guarded import review as review_data, destination, CASES


def need(ok, message):
    if not ok:
        raise ValueError(message)


def review(text, seed):
    result = review_data(text, seed)
    writes = []
    reads = []
    for job in range(1, 22):
        win = 0x1000fffc0 if job % 2 else 0x100000000
        tag = 0x1000+job
        dest = destination(job)

        def desc(epoch, window, address):
            for index, value in enumerate((epoch, tag, window, address)):
                writes.extend([(0x20+index*8,value & 0xffffffff,0),
                               (0x24+index*8,value >> 32,0)])

        desc(job, win, dest)
        writes.extend([(0x10,1,0),(0x20,0,2),(0x10,4,2)])
        if job in {4,6,7,8,9,10,17}:
            writes.append((0x10,2,0))
        if job == 15:
            writes.extend([(0x10,1,2),(0x10,4,2)])
        reads.extend([(0x60,job),(0x64,0),(0x68,tag),(0x6c,0),(0x70,CASES[job-1][2])])
        writes.extend([(0x40,job,0),(0x44,0,0),(0x48,tag,0),(0x4c,1,0),
                       (0x10,4,2),(0x4c,0,0),(0x10,4,0)])
        for epoch,window,address in [(job,win,dest),(job+1,win+1,dest),(job+1,win,dest+1)]:
            desc(epoch,window,address)
            writes.append((0x10,1,2))
    actual_writes = []
    actual_reads = []
    terminal = []
    last_cycle = 0
    for line in text.splitlines():
        if not line.startswith('NTRPU_MANAGED_'):
            continue
        label,*words=line.split()
        f=dict(word.split('=',1) for word in words)
        need(len(f)==len(words), 'duplicate field')
        if label=='NTRPU_MANAGED_MMIO':
            need(set(f)==set('n addr data resp cycle'.split()), 'write schema')
            actual_writes.append((int(f['addr'],16),int(f['data'],16),int(f['resp'])))
            need(int(f['n'])==len(actual_writes), 'write numbering')
        elif label=='NTRPU_MANAGED_READ':
            need(set(f)==set('n addr data cycle'.split()), 'read schema')
            actual_reads.append((int(f['addr'],16),int(f['data'],16)))
            need(int(f['n'])==len(actual_reads), 'read numbering')
        elif label=='NTRPU_MANAGED_PASS':
            need(set(f)==set('writes reads jobs seed'.split()), 'summary schema')
            terminal.append({k:int(v) for k,v in f.items()})
        else:
            raise ValueError('unknown managed event')
        if 'cycle' in f:
            need(int(f['cycle'])>=last_cycle, 'MMIO chronology')
            last_cycle=int(f['cycle'])
    need(actual_writes==writes and len(writes)==954, 'MMIO fixture')
    need(actual_reads==reads and len(reads)==105, 'readback fixture')
    need(terminal==[dict(writes=954,reads=105,jobs=21,seed=seed)], 'managed summary')
    result.update(control=terminal[0], rejected_control_writes=sum(r!=0 for _,_,r in writes))
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('log',type=Path);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args();print(json.dumps(review(a.log.read_text(),a.seed),indent=2))
