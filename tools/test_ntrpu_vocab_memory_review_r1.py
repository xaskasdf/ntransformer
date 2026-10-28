"""Synthetic transcript controls with real gold bytes; no RTL execution."""
import argparse
import json
from pathlib import Path
import tempfile
import check_ntrpu_vocab_memory_r1 as c


def synthetic(fixture):
    lines=[]
    cycle=1
    def emit(kind, **values):
        keys=c.SCHEMA[kind].split()
        assert set(keys)==set(values), (kind, set(keys)^set(values))
        lines.append('VOCABM_'+kind+' '+' '.join(k+'='+('%016x'%values[k] if k in c.HEX else str(values[k])) for k in keys))
    emit('CONFIG',seed=1,rows=128256,segments=16,tail_rows=5376,k_blocks=64,tile_k=32,watchdog=65536,phase_limit=16777216)
    total_ar=total_r=0
    for job,mode in ((1,1),(2,2),(3,0)):
        start=cycle;tag=c.FIRST_TAG+(4096 if job==3 else 0)
        seq=0x1234567800000000+job;epoch=0xfedcba9800000000+job
        emit('BEGIN',cycle=cycle,job=job,mode=mode,sample=0,target=0,sequence=seq,epoch=epoch,tag=tag);cycle+=1
        sums=dict(ar=0,r=0,aw=0,w=0,b=0,results=0)
        phases=[] if mode==1 else [0] if mode==2 else list(range(16))
        for phase in phases:
            first=cycle;rows=5376 if phase==15 else 8192;groups=1 if mode==2 else rows//32
            emit('MATRIX',cycle=cycle,job=job,phase=phase,tag=tag+phase*256);cycle+=1
            for group in range(groups):
                emit('GROUP',cycle=cycle,job=job,phase=phase,group=group,tag=tag+phase*256+group);cycle+=1
                for k in range(1 if mode==2 else 2):
                    tile=group*2+k
                    emit('TILE',cycle=cycle,job=job,phase=phase,tile=tile,address=0x100000000+4032+phase*17825792+tile*34816);cycle+=1
                if mode==2:
                    emit('CANCEL',cycle=cycle,job=job,phase=phase,reads=70);cycle+=1
                    emit('CHILD_CANCEL',cycle=cycle,job=job,phase=phase);cycle+=1
                else:
                    for offset in range(32):
                        row=group*32+offset
                        emit('MAC',cycle=cycle,job=job,phase=phase,row=row,data=fixture['fp64'][phase*8192+row]);cycle+=1
                    addresses=c.bursts(0xa000000000+4032+phase*32768+group*128,128)
                    wi=group*2
                    ai=group+(group+31)//32
                    for address,count in addresses:
                        emit('AW',cycle=cycle,job=job,phase=phase,aw=ai,address=address,beats=count);cycle+=1
                        for beat in range(count):
                            emit('W',cycle=cycle,job=job,phase=phase,word=wi,data=fixture['fp32'][phase*512+wi],last=int(beat==count-1));cycle+=1;wi+=1
                        emit('B',cycle=cycle,job=job,phase=phase,aw=ai,response=0);cycle+=1;ai+=1
                emit('GROUP_DONE',cycle=cycle,job=job,phase=phase,group=group,status=7 if mode==2 else 0,bytes=0 if mode==2 else 128);cycle+=1
            metrics=dict(ar=3,r=128,aw=0,w=0,b=0,results=0) if mode==2 else dict(ar=groups*21,r=groups*1152,aw=groups+(groups+31)//32,w=rows//16,b=groups+(groups+31)//32,results=rows)
            for k,v in metrics.items():sums[k]+=v
            emit('MATRIX_DONE',cycle=cycle,job=job,phase=phase,status=7 if mode==2 else 0,bytes=0 if mode==2 else rows*4,rows=0 if mode==2 else rows,tag=tag+phase*256,groups=groups,tiles=1 if mode==2 else groups*2,elapsed=cycle-first,**metrics);cycle+=1
            if phase<15 and mode==0:
                emit('MATRIX_ACK',cycle=cycle,job=job,phase=phase,public=0);cycle+=1
        total_ar+=sums['ar'];total_r+=sums['r']
        emit('COMPLETE',cycle=cycle,job=job,mode=mode,matrices=len(phases),good=16 if mode==0 else 0,elapsed=cycle-start,**sums);cycle+=1
        emit('PUBLIC',cycle=cycle,job=job,status=1 if mode==1 else 7 if mode==2 else 0,sequence=seq,epoch=epoch,valid=int(mode==0),revoked=0,matrices=16 if mode==0 else 0,bytes=513024 if mode==0 else 0)
        if phases:emit('MATRIX_ACK',cycle=cycle,job=job,phase=phases[-1],public=1)
        cycle+=1;emit('SOURCE_ACK',cycle=cycle,job=job);cycle+=1
        emit('END',cycle=cycle,job=job);cycle+=1
    emit('PASS',seed=1,jobs=3,publics=3,source_acks=3,cancels=1,matrices=17,groups=4009,results=128256,ar=total_ar,r=total_r,aw=4134,w=8016,b=4134,ar_stalls=1,r_stalls=1,aw_stalls=1,w_stalls=1,cycles=cycle)
    return '\n'.join(lines)+'\n'


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    assert not a.output.exists()
    fixture=c.load_fixture(a.fixture)
    text=synthetic(fixture)
    reference=c.review(text,fixture)
    assert reference['rows']==128256 and len(reference['segments'])==17
    rejected=c.controls(text,fixture)
    for label,bad in [('foreign_phase',text.replace('phase=0 row=0','phase=16 row=0',1)),
                      ('unknown_event',text+'VOCABM_ALIEN job=3\n')]:
        assert bad!=text
        try:c.review(bad,fixture)
        except (AssertionError,ValueError,KeyError,IndexError):rejected.append(label)
        else:raise AssertionError('Accepted '+label)
    with tempfile.TemporaryDirectory() as directory:
        folder=Path(directory)
        def write(name,raw):
            (folder/name).write_text(''.join(raw[i:i+64][::-1].hex()+'\n' for i in range(0,len(raw),64)))
        raw=fixture['logits']
        write('vocab-result.hex',raw)
        for phase in range(16):write('vocab-phase-%02d.hex'%phase,raw[phase*32768:(phase+1)*32768])
        assert c.output_files(folder,fixture)['bytes']==513024
        for name in ('vocab-result.hex','vocab-phase-15.hex'):
            original=(folder/name).read_text()
            damaged=original[:-2]+('0' if original[-2]!='0' else '1')+'\n'
            (folder/name).write_text(damaged)
            try:c.output_files(folder,fixture)
            except AssertionError:rejected.append('damaged_'+name)
            else:raise AssertionError('Accepted corrupted output file')
            (folder/name).write_text(original)
    report=dict(status='PASS_VOCAB_REVIEWER_SYNTHETIC_CONTROLS',rejected=rejected,
        fixture_manifest_sha256=c.MANIFEST_SHA,full_synthetic_rows=128256,
        native_rtl_executed=False,latency_measured=False,reviewer_sha256=c.sha(Path(c.__file__)))
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
