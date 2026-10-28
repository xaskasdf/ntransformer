"""Review full numerical results, public identity and B-before-read materialization."""
import re
from check_ntrpu_llama32_owned_ffn_arithmetic_r1 import reference,metadata,review as arithmetic_review

SCHEMA={'OWNFFN_BEGIN':'id sequence epoch source_sample token residual','OWNFFN_CHILD':'id cycle tag',
    'OWNFFN_AW':'id cycle phase buffer offset beats','OWNFFN_B':'id cycle phase buffer offset beats resp',
    'OWNFFN_COMPLETE':'id cycle sequence epoch status child stages phase bytes stage_bytes hls ffn residual valid',
    'OWNFFN_ACK':'id cycle child_ready','OWNFFN_PASS':'jobs public_bits retained_bytes input_abi_requantized',
    'NTRPU_FFN_MEMORY_W':'id buffer offset strb data','NTRPU_FFN_LARGE_AR':'id phase buffer offset beats'}
HEX={'sequence','epoch','tag','hls','ffn','residual','strb','data'}

def parse(text):
    rows=[]
    assert not re.search(r'(?mi)^\s*(FATAL:|ERROR:|FAIL\b)',text)
    for line in text.splitlines():
        parts=line.split()
        if not parts:continue
        tag=parts[0]
        if tag not in SCHEMA:
            assert not tag.startswith('OWNFFN_'),tag
            continue
        pairs=[p.split('=') for p in parts[1:]]
        assert all(len(p)==2 for p in pairs) and [k for k,v in pairs]==SCHEMA[tag].split(),line
        row={k:int(v,16 if k in HEX else 10) for k,v in pairs};assert all(v>=0 for v in row.values())
        rows.append((tag,row))
    return rows

def review_ownership(text,ref):
    rows=parse(text);assert rows[0]==('OWNFFN_BEGIN',dict(id=1,sequence=0x12345678fffffffe,epoch=0xfedcba9800000259,source_sample=1,token=0,residual=0x700000000fc0))
    assert rows[-1]==('OWNFFN_PASS',dict(jobs=1,public_bits=64,retained_bytes=8192,input_abi_requantized=0))
    assert sum(k=='OWNFFN_BEGIN' for k,r in rows)==sum(k=='OWNFFN_PASS' for k,r in rows)==1
    buffers={b:bytearray(len(ref['buffers'][b])) for b in range(4,8)}
    committed={b:bytearray(len(raw)) for b,raw in buffers.items()}
    pending=None;child=completion=ack=None;last=0;aw_count=b_count=write_bytes=read_bytes=0
    for tag,row in rows[1:-1]:
        assert row['id']==1 and ack is None
        if 'cycle' in row:
            assert last<row['cycle']<30000000;last=row['cycle']
        if tag=='OWNFFN_CHILD':
            assert child is None and row['tag']==0xfffffffc;child=row;continue
        assert child is not None
        if tag=='OWNFFN_COMPLETE':
            assert completion is None and pending is None and all(all(v) for v in committed.values())
            expected=dict(id=1,cycle=row['cycle'],sequence=0x12345678fffffffe,epoch=0xfedcba9800000259,status=0,child=0,
                stages=4,phase=3,bytes=8192,stage_bytes=155648,hls=0,ffn=0x1e0000fc0,residual=0x700000000fc0,valid=1)
            assert row==expected and all(bytes(raw)==ref['buffers'][b] for b,raw in buffers.items())
            completion=row;continue
        if tag=='OWNFFN_ACK':
            assert completion and row==dict(id=1,cycle=completion['cycle']+6,child_ready=1)
            ack=row;continue
        assert completion is None
        if tag=='OWNFFN_AW':
            assert pending is None and row['phase'] in range(4) and row['buffer']==row['phase']+4
            assert 1<=row['beats']<=4 and row['offset']%64==0
            assert row['offset']+row['beats']*64<=len(buffers[row['buffer']])
            assert (4032+row['offset'])%4096+row['beats']*64<=4096
            pending=dict(row,words=[]);aw_count+=1
        elif tag=='NTRPU_FFN_MEMORY_W':
            assert pending and len(pending['words'])<pending['beats']
            assert row['buffer']==pending['buffer'] and row['offset']==pending['offset']+64*len(pending['words'])
            assert row['strb']==(1<<64)-1 and row['data']<1<<512
            word=row['data'].to_bytes(64,'little')
            assert word==ref['buffers'][row['buffer']][row['offset']:row['offset']+64]
            pending['words'].append(word);write_bytes+=64
        elif tag=='OWNFFN_B':
            assert pending and len(pending['words'])==pending['beats']
            assert row=={k:v for k,v in pending.items() if k not in ('cycle','words')}|dict(cycle=row['cycle'],resp=0)
            b=row['buffer'];off=row['offset'];raw=b''.join(pending['words']);end=off+len(raw)
            assert not any(committed[b][off:end]);buffers[b][off:end]=raw;committed[b][off:end]=b'\1'*len(raw)
            b_count+=1;pending=None
        elif tag=='NTRPU_FFN_LARGE_AR':
            b=row['buffer'];off=row['offset'];size=row['beats']*64
            assert row['phase'] in range(4) and b in ((0,1),(0,2),(4,5),(6,3))[row['phase']]
            assert off%64==0 and 0<size<=4096 and off+size<=len(ref['buffers'][b])
            if b>=4:
                assert all(committed[b][off:off+size]),'Read before successful B'
                assert buffers[b][off:off+size]==ref['buffers'][b][off:off+size];read_bytes+=size
        else:raise AssertionError(tag)
    assert completion and ack and pending is None and aw_count==b_count and write_bytes==155648 and read_bytes==1179648
    return dict(public_bits=64,private_tag=0xfffffffc,committed_write_bytes=write_bytes,reads_from_committed_bytes=read_bytes,
        committed_bursts=b_count,completion_cycle=completion['cycle'],ack_cycle=ack['cycle'],input_abi_requantized=False)

def review(text,ref,seed=1):
    result=arithmetic_review(text,ref,seed)
    try:result['ownership']=review_ownership(text,ref)
    except (AssertionError,ValueError,KeyError,IndexError,OverflowError) as exc:result['errors'].append('Ownership/materialization: '+repr(exc))
    result['status']='FAIL' if result['errors'] else 'PASS'
    result['scope']='Actual full owned FFN against exact post-O reference ABI1. Live post-O producer, second residual transport, full inference and physical timing remain separate gates.'
    return result

def controls(text,ref,seed=1):
    lines=text.splitlines();rejected=[]
    trials=[('OWNFFN_BEGIN','sequence',1<<40),('OWNFFN_BEGIN','epoch',1<<40),('OWNFFN_BEGIN','source_sample',1),
        ('OWNFFN_BEGIN','token',1),('OWNFFN_BEGIN','residual',64),('OWNFFN_CHILD','tag',4),
        ('OWNFFN_AW','buffer',1),('OWNFFN_AW','offset',64),('OWNFFN_AW','beats',1),
        ('OWNFFN_B','resp',1),('OWNFFN_B','offset',64),('OWNFFN_B','beats',1),
        ('NTRPU_FFN_MEMORY_W','data',1),('NTRPU_FFN_MEMORY_W','strb',1),
        ('OWNFFN_COMPLETE','sequence',1<<40),('OWNFFN_COMPLETE','epoch',1<<40),('OWNFFN_COMPLETE','status',1),
        ('OWNFFN_COMPLETE','child',1),('OWNFFN_COMPLETE','stages',1),('OWNFFN_COMPLETE','phase',1),
        ('OWNFFN_COMPLETE','bytes',1),('OWNFFN_COMPLETE','stage_bytes',1),('OWNFFN_COMPLETE','hls',1),
        ('OWNFFN_COMPLETE','ffn',64),('OWNFFN_COMPLETE','residual',64),('OWNFFN_COMPLETE','valid',1),
        ('OWNFFN_ACK','child_ready',1),('OWNFFN_ACK','cycle',1),('OWNFFN_PASS','jobs',1),('OWNFFN_PASS','input_abi_requantized',1)]
    for prefix,key,mask in trials:
        bad=list(lines);i=next(i for i,line in enumerate(bad) if line.startswith(prefix+' '))
        def alter(m):
            value=int(m[1],16 if key in HEX else 10)^mask
            return key+'='+format(value,'x' if key in HEX else 'd')
        bad[i],count=re.subn(r'\b'+key+r'=([0-9a-f]+)',alter,bad[i]);assert count==1
        result=review('\n'.join(bad),ref,seed);assert result['status']=='FAIL',prefix+'.'+key;rejected.append(prefix+'.'+key)
    for prefix in ('OWNFFN_B','OWNFFN_ACK','OWNFFN_COMPLETE'):
        bad=list(lines);i=next(i for i,line in enumerate(bad) if line.startswith(prefix+' '));del bad[i]
        assert review('\n'.join(bad),ref,seed)['status']=='FAIL';rejected.append('missing-'+prefix)
    bad=list(lines);i=next(i for i,line in enumerate(bad) if line.startswith('NTRPU_FFN_LARGE_AR ') and 'buffer=4 ' in line)
    moved=bad.pop(i);at=next(i for i,line in enumerate(bad) if line.startswith('OWNFFN_B '));bad.insert(at,moved)
    assert review('\n'.join(bad),ref,seed)['status']=='FAIL';rejected.append('read-before-commit')
    return rejected
