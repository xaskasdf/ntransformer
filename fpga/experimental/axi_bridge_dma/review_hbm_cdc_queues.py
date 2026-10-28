"""Audit transport stress transactions independently of the HDL scoreboard."""
import argparse,json,re
from pathlib import Path
from review_hbm_source import sha,require
CASES=[0]
FIFO_SHA='be809fe998385e69a52eb2fa8687b5b50e52433a5b1a42694331efc07f1f0634'
def tag(lane,frame):return 0xabcd123400000001+(lane<<40)+frame

def review(text,case,seed):
    require(f'HQ PASS seed={seed} cold_recovery=1' in text,'Missing stress verdict')
    require(not re.search(r'(?im)^\s*(?:Fatal|Error|FATAL|ERROR):',text),'Tool error')
    rows={k:[] for k in ('CMD','EVENT','RETAIN','COVER')}
    for kind,fields in re.findall(r'^HQ (CMD|EVENT|RETAIN|COVER) (.+)$',text,re.M):
        rows[kind].append(dict(re.findall(r'(\w+)=(\S+)',fields)))
    expected_cmd=[];expected_events=[]
    for lane in range(2):
        for stage,indices in [(1,range(8)),(3,[8]),(5,[10])]:
            for i in indices:expected_cmd.append((stage,lane,i if stage!=5 else 0,0x100001000+(lane<<33)+i*4096,tag(lane,i),i))
        for stage,frame,beats,offset in [(2,0,64,0),(3,1,2,65),(5,3,1,0)]:
            for beat in range(beats):
                data=bytes(((beat*64+b)*37 ^ (frame*13) ^ (lane*91) ^ seed)&255 for b in range(64))
                expected_events.append((stage,lane,offset+beat,0,int(beat==beats-1),tag(lane,frame),0,0,int.from_bytes(data,'little')))
            expected_events.append((stage,lane,offset+beats,1,0,tag(lane,frame),frame%3,beats*64,0))
    commands=[tuple(int(r[k],16 if k in ('addr','tag') else 10) for k in ('stage','lane','index','addr','tag','len')) for r in rows['CMD']]
    events=[tuple(int(r[k],16 if k in ('tag','data') else 10) for k in ('stage','lane','index','cpl','last','tag','status','bytes','data')) for r in rows['EVENT']]
    # Keep per-lane sequence; ordering between independent lanes is unspecified.
    for lane in range(2):
        require([r for r in commands if r[1]==lane]==[r for r in expected_cmd if r[1]==lane],'Lost/replayed/reordered command')
        require([r for r in events if r[1]==lane]==[r for r in expected_events if r[1]==lane],'Lost/reordered/corrupt data or CPL')
    require(rows['RETAIN']==[dict(commands='2',returns='6',fault='1')],'Missing queued ownership evidence')
    require(len(rows['COVER'])==1 and all(int(rows['COVER'][0][k])>0 for k in ('command_stalls','return_stalls')) and int(rows['COVER'][0]['simultaneous'])>=8,'Missing stress coverage')
    return dict(seed=seed,commands=len(commands),events=len(events),bytes_checked=8576,cold_recovery=True)

def bundle(root):
    files=json.loads((root/'FILES.json').read_text())
    require(set(files)|{'FILES.json'}=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.is_relative_to(root/'work')},'File set mismatch')
    for n,h in files.items():require(sha(root/n)==h,'Hash mismatch: '+n)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='HBM_CDC_QUEUES_PASS' and result['sources_unchanged'],'Incomplete stress phase')
    require(sha(root/'source/ntrpu_bridge_async_fifo.sv')==FIFO_SHA,'FIFO mismatch')
    require(all(s['returncode']==0 and not s['timed_out'] for s in result['steps']),'Failed step')
    checks=[review((root/f'case-00-seed-{seed}.log').read_text(),0,seed) for seed in range(4)]
    require(checks==result['reviews'],'Stored verdict mismatch')
    return dict(status='HBM_CDC_QUEUES_REVIEWED',cases=len(checks),bytes_checked=sum(c['bytes_checked'] for c in checks),commands=sum(c['commands'] for c in checks),events=sum(c['events'] for c in checks),physical_cdc_qualified=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(bundle(ap.parse_args().output),indent=2))
