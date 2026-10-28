"""Full-size synthetic reviewer controls; these are not RTL results or timings."""
import argparse,json,time
from pathlib import Path
import check_ntrpu_final_vocab_arithmetic_r1 as c
import check_ntrpu_vocab_memory_r2 as old
from test_ntrpu_vocab_memory_review_r2 import synthetic as head_synthetic

def synthetic(fixture):
    events=old.parse(head_synthetic(fixture));out=[];cycle=1;starts={};matrix_starts={}
    abi=fixture['norm']['abi64.hex'];source=fixture['norm']['source32.hex']
    def emit(kind,**row):
        nonlocal cycle
        if 'cycle' in c.SCHEMA[kind].split():row['cycle']=cycle;cycle+=1
        if kind=='BEGIN':starts[row['job']]=row['cycle']
        if kind=='MATRIX':matrix_starts[row['job'],row['phase']]=row['cycle']
        if kind=='MATRIX_DONE':row['elapsed']=row['cycle']-matrix_starts[row['job'],row['phase']]
        if kind=='COMPLETE':row['elapsed']=row['cycle']-starts[row['job']]
        if kind=='PASS':row.update(cycles=cycle,source_acks=2)
        assert set(row)==set(c.SCHEMA[kind].split()),kind
        out.append('FVREAL_'+kind+' '+' '.join(k+'='+('%016x'%row[k] if k in c.HEX else str(row[k])) for k in c.SCHEMA[kind].split()))
    for kind,row in events:
        job=row.get('job');mode=1 if job==1 else 2 if job==2 else 0
        if kind=='SOURCE_ACK' and job==1:continue
        if kind=='PUBLIC':row['revoked']=int(mode!=0)
        if kind=='COMPLETE':
            active=int(job!=1)
            emit('NCOMPLETE',job=job,status=0,hls=0,reads=active*256,records=active*64,
                starts=active,dones=active,words=active*512,ar=active*6,r=active*256,aw=active*64,w=active*64,b=active*64)
        if kind=='PASS':emit('NSUMMARY',starts=2,dones=2,words=1024,ar=12,r=512,w=128,b=128)
        emit(kind,**row)
        if kind=='BEGIN' and job!=1:
            cursor=ar=0
            for plane in range(2):
                for address,count in old.bursts(0x6000000000+plane*0x10000000+4032,8192):
                    emit('NAR',job=job,ar=ar,address=address,beats=count)
                    for beat in range(count):
                        value=sum(source[cursor*16+n]<<(32*n) for n in range(16))
                        emit('NR',job=job,ar=ar,beat=beat,address=address+beat*64,data=value,response=0,last=int(beat==count-1));cursor+=1
                    ar+=1
            emit('NSTART',job=job)
            for index,value in enumerate(abi):emit('NWRITE',job=job,index=index,data=value)
            emit('NDONE',job=job,status=0,words=512)
            for index in range(64):
                emit('NAW',job=job,aw=index,address=0x8000000000+4032+64*index,beats=1)
                emit('NW',job=job,word=index,data=sum(abi[index*8+n]<<(64*n) for n in range(8)),last=1)
                emit('NB',job=job,aw=index,response=0)
        if kind=='GROUP':
            for index in range(64):emit('R',job=job,phase=row['phase'],group=row['group'],index=index,
                address=0x8000000000+4032+index*64,data=sum(abi[index*8+n]<<(64*n) for n in range(8)))
    return '\n'.join(out)+'\n'

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fixture','norm-fixture','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--norm-sha',required=True);a=p.parse_args()
    assert not a.output.exists();started=time.monotonic()
    fixture=c.load_fixture(a.fixture);c.load_norm_fixture(a.norm_fixture,a.norm_sha,fixture)
    text=synthetic(fixture);report=c.review(text,fixture);rejected=c.controls(text,fixture)
    assert report['rows']==128256 and len(report['segments'])==17 and report['live_normalization_qualified']
    report=dict(status='PASS_FINAL_VOCAB_REVIEWER_SYNTHETIC_CONTROLS_R1',controls=rejected,
        full_synthetic_rows=128256,synthetic_abi_reads=4009*64,native_rtl_executed=False,
        fixture_manifest_sha256=c.MANIFEST_SHA,norm_manifest_sha256=a.norm_sha,
        reviewer_sha256=c.sha(Path(c.__file__)),wall_seconds=time.monotonic()-started)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__':main()
