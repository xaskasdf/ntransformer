"""Synthetic reviewer controls; never an RTL/arithmetic execution certificate."""
import argparse
import copy
import json
from pathlib import Path
import struct
import check_ntrpu_llama32_layer_real_r3 as checker
import ntrpu_llama32_layer_fixture_r2 as fixture

def transcript(ref):
    rows=[];cycle=10;counts=[dict(ar=0,r=0,aw=0,w=0,b=0) for _ in range(5)]
    def emit(tag,advance=True,**fields):
        nonlocal cycle
        if advance:cycle+=1
        if 'cycle' in checker.SCHEMA[tag].split():fields=dict(cycle=cycle,**fields)
        rows.append((tag,fields))
    def reads(port,requests):
        for req in requests:
            emit('AR',port=port,index=counts[port]['ar'],**req);counts[port]['ar']+=1;counts[port]['r']+=req['beats']
            r=req['region']
            if ref['manifest']['regions'][r]['writable']:
                for beat in range(req['beats']):
                    off=req['offset']+beat*64
                    emit('R',port=port,region=r,offset=off,data=int.from_bytes(ref['data'][r][off:off+64],'little'))
    def writes(port,region):
        offsets=[b*64 for b in ref['manifest']['cache_written_beats']] if region==24 else range(0,len(ref['data'][region]),64)
        for off in offsets:
            emit('AW',port=port,index=counts[port]['aw'],region=region,offset=off,beats=1);counts[port]['aw']+=1
            emit('W',port=port,index=counts[port]['w'],last=1,data=int.from_bytes(ref['data'][region][off:off+64],'little'));counts[port]['w']+=1
            emit('B',port=port,index=counts[port]['b'],region=region,offset=off,beats=1);counts[port]['b']+=1
    emit('ADMIT',sequence=checker.SEQ,epoch=checker.EPOCH,layer=0,position=0,token=0)
    emit('START',stage=1);reads(0,ref['plans'][0][:6]);writes(0,2);reads(0,ref['plans'][0][6:])
    for r in (6,7,8):writes(0,r)
    reads(1,ref['plans'][1]);reads(2,ref['plans'][2]);emit('ROPE_START')
    for i,v in enumerate(ref['rope']):emit('ROPE',index=i,data=v)
    emit('ROPE_DONE');writes(3,24)
    for h in range(32):
        emit('HEAD',head=h)
        for plane in range(2):reads(3,fixture.bursts(24,((plane*8+h//4)*4)*256,256))
        oracle=ref['refs'][h];prefix=oracle['prefixes'][0]
        bits=lambda x:struct.unpack('<Q',struct.pack('<d',float(x)))[0]
        state=[0x4e54524154543031,checker.SEQ,checker.EPOCH,0,h,1,prefix['maximum'],bits(prefix['normalizer'])]
        state += [oracle['q'][2*i]|oracle['q'][2*i+1]<<32 for i in range(32)]
        state += [bits(v) for v in prefix['accumulators']]
        for i,v in enumerate(state):emit('STATE',head=h,index=i,data=v)
        for i,v in enumerate(ref['attention'][h*64:(h+1)*64]):emit('ATTENTION',head=h,index=i,data=v)
    emit('DONE',stage=1);emit('START',stage=2)
    for i,v in enumerate(ref['attention']):
        emit('SOURCE',index=i,data=v);emit('HLS_INPUT',advance=False,index=i,data=v)
    # HLS can request the same RAM address more than once. Every return must be consumed.
    for i in range(64):
        emit('SOURCE',index=i,data=ref['attention'][i]);emit('HLS_INPUT',advance=False,index=i,data=ref['attention'][i])
    emit('ACK',stage=1);writes(4,11)
    plans=ref['plans'][4];cursor=0
    while cursor<len(plans) and plans[cursor]['region'] in (10,11):reads(4,[plans[cursor]]);cursor+=1
    writes(4,12);emit('DONE',stage=2);emit('START',stage=3)
    while cursor<len(plans) and plans[cursor]['region'] in (0,12,13):reads(4,[plans[cursor]]);cursor+=1
    writes(4,14);writes(4,15);emit('DONE',stage=3);emit('START',stage=4)
    for allowed,destination in [((15,16),19),((15,17),20),((19,20),21),((21,18),22),((14,22),23)]:
        while cursor<len(plans) and plans[cursor]['region'] in allowed:reads(4,[plans[cursor]]);cursor+=1
        writes(4,destination)
    assert cursor==len(plans);emit('DONE',stage=4)
    for p,c in enumerate(counts):
        emit('PORT',port=p,**c,ar_stalls=42*int(p in (0,4)),w_stalls=99*int(p in (0,3,4)))
        for r in range(25):
            stats=ref['manifest']['read_stats'][p][str(r)];emit('READ',port=p,region=r,beats=stats['beats'],hash=int(stats['hash'],16))
    emit('COMPLETE',status=0,sequence=checker.SEQ,epoch=checker.EPOCH,address=fixture.address(23),bytes=8192,prefix=1)
    cycle+=7;emit('PUBLIC_ACK');emit('ACK',stage=3);emit('ACK',advance=False,stage=4);emit('ACK',stage=2)
    emit('PASS',seed=1,jobs=1,cycles=cycle+1,writes=3264,heads=32,source_words=2112)
    return rows

def render(rows):
    return '\n'.join('LREAL_'+tag+' '+' '.join(f'{k}={r[k]:x}' if k in ('sequence','epoch','address','data','hash') else f'{k}={r[k]}' for k in checker.SCHEMA[tag].split()) for tag,r in rows)+'\n'

def controls(rows,ref):
    changes=[('ADMIT','sequence',1<<40),('ADMIT','epoch',1<<40),('ADMIT','token',1),
        ('AR','offset',64),('AR','beats',1),('AW','region',1),('W','data',1),('W','last',1),('B','offset',64),
        ('R','data',1),('ROPE','data',1),('HEAD','head',1),('STATE','data',1),('ATTENTION','data',1),
        ('SOURCE','data',1),('HLS_INPUT','index',1),('PORT','r',1),('PORT','ar_stalls',1),('PORT','w_stalls',1),('READ','hash',1),('READ','beats',1),
        ('COMPLETE','prefix',1),('COMPLETE','status',1),('COMPLETE','bytes',64),('PASS','writes',1),('PASS','source_words',1)]
    rejected=[]
    for tag,key,mask in changes:
        altered=rows.copy();i=next(i for i,(k,_) in enumerate(rows) if k==tag)
        r=dict(rows[i][1]);r[key]=0 if key in ('ar_stalls','w_stalls') else r[key]^mask;altered[i]=(tag,r)
        try:checker.review_rows(altered,ref)
        except (ValueError,AssertionError,KeyError):rejected.append(tag+'.'+key)
        else:raise AssertionError('Accepted corrupted '+tag+'.'+key)
    for tag in ('B','DONE','ACK','HLS_INPUT','STATE','PASS'):
        altered=rows.copy();i=next(i for i,(k,_) in enumerate(rows) if k==tag);altered.pop(i)
        try:checker.review_rows(altered,ref)
        except (ValueError,AssertionError,KeyError):rejected.append('missing_'+tag)
        else:raise AssertionError('Accepted missing '+tag)
    # A fully formed read cannot be placed before the write that produces it.
    altered=rows.copy();i=next(i for i,(tag,r) in enumerate(rows) if tag=='AR' and r['region']==2)
    tag,r=altered.pop(i);r=dict(r);j=next(i for i,(tag,r) in enumerate(altered) if tag=='AW' and r['region']==2)
    r['cycle']=altered[j][1]['cycle'];altered.insert(j,(tag,r))
    try:checker.review_rows(altered,ref)
    except (ValueError,AssertionError,KeyError):rejected.append('read_before_commit')
    else:raise AssertionError('Accepted unproduced activation')
    return rejected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();ref=checker.load(a.fixture);rows=transcript(ref);text=render(rows)
    review=checker.review(text,ref);rejected=controls(rows,ref)
    a.output.mkdir(parents=True);(a.output/'synthetic.log').write_text(text)
    result=dict(status='PASS_SYNTHETIC_LAYER_REVIEWER_CONTROLS_R4',arithmetic_executed=False,rtl_executed=False,
        fixture_manifest_sha256=fixture.digest(a.fixture/'manifest.json'),summary=review,rejected=rejected,
        checker_sha256=fixture.digest(Path(checker.__file__)),generator_sha256=fixture.digest(Path(__file__)))
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],controls=len(rejected),events=len(rows),rtl_executed=False)))

if __name__=='__main__':main()
