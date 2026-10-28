"""Original RoPE vectors plus honest KV-prefix preparation for composition tests."""
import hashlib,json,shutil
from pathlib import Path
from ntrpu_rope_input_fixture_r1 import load as load_parent

RESETS=(50,51,52,53,90,91)
FAULTS=[10,11,20,21,24,25,26,28,30,31,32,33,34,35,40,43,50,51,52,53,*range(80,89),90,91]
FIELDS=('mode','sample','reuse','group','layer','capacity','prefix','want','capture_status','rope_status')

def fault_status(mode):
    if mode in (10,87):return 1
    if mode in (11,26,84,88):return 2
    if mode in (20,34,85):return 3
    if mode in (21,86):return 4
    if mode==35:return 5
    if mode in (24,25,28,30,31,32,33,80,81,82,83):return 7
    return 0

def plan(rows):
    jobs=[];group=0
    def add(mode,sample,reuse=False,layer=0,capacity=3,prefix=0):
        nonlocal group
        if not reuse:group=len(jobs)+1
        row=rows[sample]
        if mode>=100:
            rope=row['public_status']
            capture=1 if row['position']>=131072 else 2 if row['position']!=row['coefficient_position'] else rope
            want=1 if row['position']!=prefix or row['position']>=capacity else capture
        else:
            want=fault_status(mode)
            rope=want if 80<=mode<=88 and mode!=83 else 0
            capture=want if 10<=mode<=27 or 80<=mode<=88 else 0
        jobs.append(dict(zip(FIELDS,(mode,sample,int(reuse),group,layer,capacity,prefix,want,capture,rope))))
    for index,row in enumerate(rows):
        if row['heads']!=40:continue
        p=row['position'];layer=15 if index>=45 else 0
        if p in (1,2):
            add(100,0,layer=layer)
            if p==2:add(101,1,True,layer=layer,prefix=1)
            add(100+index,index,True,layer=layer,prefix=p)
        else:add(100+index,index,layer=layer,capacity=min(max(p+1,3),131072))
    for mode in FAULTS:
        add(mode,0)
        if mode in RESETS:add(100,0)
    assert len(jobs)==82
    return jobs

def load(folder):
    m=json.loads((folder/'manifest.json').read_text())
    for n,h in m['files'].items():assert hashlib.sha256((folder/n).read_bytes()).hexdigest()==h,n
    rows=load_parent(folder/'rope_input');jobs=plan(rows)
    assert m['plan']==jobs and m['fields']==list(FIELDS)
    assert (folder/'plan.hex').read_text()==''.join(f'{r[k]:08x}\n' for r in jobs for k in FIELDS)
    return rows,jobs

def prepare(parent,destination):
    rows=load_parent(parent);jobs=plan(rows)
    destination.mkdir(exist_ok=False)
    shutil.copytree(parent,destination/'rope_input')
    (destination/'plan.hex').write_text(''.join(f'{r[k]:08x}\n' for r in jobs for k in FIELDS))
    hashes={p.relative_to(destination).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in destination.rglob('*') if p.is_file()}
    summary=dict(jobs=len(jobs),resets=sum(j['mode'] in RESETS for j in jobs),
        expected_errors=sum(j['want']!=0 for j in jobs),
        public_successes=sum(j['want']==0 and j['mode'] not in RESETS for j in jobs),
        revocations=2,original_fixed_40_vectors=39,full_path_real_projection_vectors=2,
        large_position_projection_vectors_with_prefix_rejection=4)
    m=dict(files=hashes,fields=list(FIELDS),plan=jobs,summary=summary,arithmetic_executed=False,
        scope='Original unmodified 39 fixed-40-head vectors. Real priming for positions 1 and 2; high-position capture is followed by required prefix rejection, never forced metadata. Six reset recoveries.')
    (destination/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
    assert load(destination)==(rows,jobs)
    return m
