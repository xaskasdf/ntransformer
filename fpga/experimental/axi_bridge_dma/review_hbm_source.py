"""Independent public-transaction audit of the HBM source frontier bench."""
import argparse
import hashlib
import json
from pathlib import Path
import re

CASES=list(range(18))+[20,21]
READER_SHA='28863cc75ac91edf265540cdb8a6c02dc110620aad4a88e11b86f97506c14b2f'
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def review(text,case,seed):
    require(f'HS PASS case={case} seed={seed}' in text,'Missing verdict')
    require(not re.search(r'(?im)^\s*(?:Fatal|Error|FATAL|ERROR):',text),'Simulation error')
    rows={kind:[] for kind in ('CMD','AW','W','RESULT','RETAIN')}
    for kind,fields in re.findall(r'^HS (CMD|AW|W|RESULT|RETAIN) (.+)$',text,re.M):
        rows[kind].append(dict(re.findall(r'(\w+)=(\S+)',fields)))
    byte_count=0
    for w in rows['W']:
        txn=int(w['txn']);index=int(w['index'])
        address=(0x200000000 if txn%2 else 0x100000000)+0x1000+txn*4096
        expected=bytes(((n*37)^(n>>7)^((address>>32)&255)^((address>>8)&255)^0x5a)&255 for n in range(index*64,(index+1)*64))
        require(int(w['data'],16).to_bytes(64,'little')==expected,'Wrong HBM-to-PCIe payload');byte_count+=64
    for cmd in rows['CMD']:
        txn=int(cmd['txn']);address=(0x200000000 if txn%2 else 0x100000000)+0x1000+txn*4096
        require(int(cmd['addr'],16)==address and int(cmd['lane'])==txn%2,'Lost high address or stack')
        require(int(cmd['tag'],16)==(0x12345678abcdef09<<64)+0x100000001+txn,'Lost source identity')
    if case==0:
        require(len(rows['CMD'])==len(rows['AW'])==len(rows['RESULT'])==4,'Missing successful transfer')
        require(byte_count==4544 and not rows['RETAIN'],'Missing successful bytes')
        for txn,length in enumerate((64,128,256,4096)):
            payload=[r for r in rows['W'] if int(r['txn'])==txn]
            require([int(r['index']) for r in payload]==list(range(length//64)),'Missing/reordered write beat')
            require([int(r['last']) for r in payload]==[0]*(length//64-1)+[1],'Wrong final beat')
            result=rows['RESULT'][txn]
            require(int(result['bytes'])==length and int(result['tag'],16)==(0x12345678abcdef09<<64)+0x100000001+txn,'Wrong retired identity')
    elif case==16:require(not any(rows.values()),'Rejected address reached the reader')
    else:
        require(len(rows['RETAIN'])==1 and all(rows['RETAIN'][0].get(k)=='1' for k in ('owned','source','fault','cold')),'Lost ownership or operational reset')
        require(len(rows['RESULT'])==(1 if case in (10,14) else 0),'Unexpected result')
        if case in (4,5,6,7,9,12):require(not rows['AW'] and not rows['W'],'Bad source launched PCIe')
    return dict(case=case,seed=seed,bytes_checked=byte_count,commands=len(rows['CMD']),results=len(rows['RESULT']),retained=bool(rows['RETAIN']))

def bundle(root):
    files=json.loads((root/'FILES.json').read_text())
    # The runner excludes its rebuildable compiler work directory from evidence.
    require(set(files)|{'FILES.json'}=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.is_relative_to(root/'work')},'File set mismatch')
    for n,h in files.items():require(sha(root/n)==h,'Hash mismatch: '+n)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='HBM_SOURCE_FRONTIER_PASS' and result['sources_unchanged'],'Phase incomplete')
    require(sha(root/'source/ntrpu_hbm_read_frames.sv')==READER_SHA,'Reader is not the qualified donor')
    reviews=[review((root/f'case-{case:02}-seed-{seed}.log').read_text(),case,seed) for seed in range(2) for case in CASES]
    return dict(status='HBM_SOURCE_FRONTIER_REVIEWED',files=len(files),cases=len(reviews),bytes_checked=sum(r['bytes_checked'] for r in reviews),retention_cases=sum(r['retained'] for r in reviews),native_hbm_pcie_simulation=False,physical_timing_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(bundle(ap.parse_args().output),indent=2))
