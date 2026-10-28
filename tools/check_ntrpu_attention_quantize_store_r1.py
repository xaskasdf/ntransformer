"""Check FP32/ABI1 binding, source releases, AXI drain and publication traces."""
import hashlib
import json
from pathlib import Path
import re

PLAN=[0]*4+[m for mode in range(1,11) for m in (mode,0)]
SCHEMA={
 'CONFIG':'seed watchdog','RESET':'cycle job count',
 'BEGIN':'cycle job mode sample sequence epoch address','READ':'cycle job index data','INPUT':'cycle job index data',
 'START':'cycle job','WORD':'cycle job index data','DONE':'cycle job status words',
 'AW':'cycle job record address','W':'cycle job record data','B':'cycle job record response',
 'ROOT_ACK':'cycle job sequence epoch','QUANT_ACK':'cycle job sequence epoch',
 'INJECT':'cycle job mode','FOREIGN_HELD':'cycle job','REUSE':'cycle job sequence epoch','ABORT':'cycle job mode aw w b',
 'REVOKED':'cycle job','COMPLETE':'cycle job mode status hls sequence epoch records valid address bytes aw w b',
 'PASS':'seed jobs publics root quant resets reads writes starts dones aw w b aw_stalls w_stalls held aw_first w_first cycles'}
HEX={'sequence','epoch','address','data'}

def load_fixture(folder):
    m=json.loads((folder/'manifest.json').read_text());samples=[]
    for i in range(1,5):
        row={}
        for kind,width in [('fp32',4),('abi1',8)]:
            p=folder/f'sample-{i:03d}-attention-{kind}.bin';raw=p.read_bytes()
            assert hashlib.sha256(raw).hexdigest()==m['files'][p.name]
            assert len(raw)==(8192 if kind=='fp32' else 4096)
            row[kind]=[int.from_bytes(raw[n:n+width],'little') for n in range(0,len(raw),width)]
        samples.append(row)
    return samples

def parse(text):
    assert not re.search(r'(?im)^\s*(?:Fatal:|FATAL:|ERROR:|QSTORE_FAIL)',text),'Diagnostic'
    result=[];cycle=-1
    for line in text.splitlines():
        if not line.startswith('QSTORE_'):continue
        parts=line.split();kind=parts[0][7:];assert kind in SCHEMA,kind
        pairs=[s.split('=') for s in parts[1:]]
        assert all(len(s)==2 for s in pairs)
        assert [s[0] for s in pairs]==SCHEMA[kind].split(),(kind,pairs)
        row={k:int(v,16 if k in HEX else 10) for k,v in pairs}
        if 'cycle' in row:assert row['cycle']>=cycle;cycle=row['cycle']
        result.append((kind,row))
    return result

def review(text,samples,seed=1):
    events=parse(text)
    def selected(kind,job=None):return [r for k,r in events if k==kind and (job is None or r.get('job')==job)]
    assert selected('CONFIG')==[dict(seed=seed,watchdog=65536)]
    begins=selected('BEGIN');assert len(begins)==24
    summaries=selected('PASS');assert len(summaries)==1
    assert events[-1][0]=='PASS'
    summary=summaries[0];assert summary['seed']==seed
    assert [r['job'] for r in begins]==list(range(1,25)) and [r['mode'] for r in begins]==PLAN
    assert [r['job'] for r in selected('RESET')]==[0,17,19,21]
    assert [r['count'] for r in selected('RESET')]==[1,2,3,4]
    assert [r['job'] for r in selected('INJECT')]==[5,7,13,17,21]
    assert [r['job'] for r in selected('REVOKED')]==[15]
    assert [r['job'] for r in selected('REUSE')]==[23]
    assert [r['job'] for r in selected('FOREIGN_HELD')]==[21]
    results=[]
    for begin in begins:
        j=begin['job'];mode=begin['mode'];sample=samples[(j-1)%4]
        seq=0x1234567800000000+j;epoch=0xfedcba9800000000+j;address=0x2200000fc0+j*8192
        assert (begin['sample'],begin['sequence'],begin['epoch'],begin['address'])==((j-1)%4,seq,epoch,address)
        e={kind:selected(kind,j) for kind in SCHEMA}
        assert len(e['START'])==1 and e['START'][0]['cycle']>begin['cycle']
        assert len(e['DONE'])==(0 if mode==7 else 1)
        reads=e['READ'];inputs=e['INPUT'];words=e['WORD']
        assert reads
        for row in reads:
            assert 0<=row['index']<2048
            value=0x7fc00001 if mode==3 and row['index']==2030 else sample['fp32'][row['index']]
            assert row['data']==value
        requested={(r['cycle']+1,r['index']):r['data'] for r in reads}
        assert len(requested)==len(reads)
        for row in inputs:assert requested[(row['cycle'],row['index'])]==row['data']
        assert len({(r['cycle'],r['index']) for r in inputs})==len(inputs)
        if mode not in (1,2,3,7,9):
            assert set(r['index'] for r in reads)==set(range(2048))
            assert len(reads)==len(inputs)
        assert len(words)==(0 if mode in (3,7) else 512)
        assert [r['index'] for r in words]==list(range(len(words)))
        if mode not in (1,2,9):assert [r['data'] for r in words]==sample['abi1'][:len(words)]
        if e['DONE']:
            done=e['DONE'][0];assert done['status']==(2 if mode==3 else 0) and done['words']==len(words)
            assert done['cycle']>e['START'][0]['cycle']
            assert all(r['cycle']<=done['cycle'] for r in words)
        assert len(e['ROOT_ACK'])==(0 if mode in (7,9) else 1)
        assert len(e['QUANT_ACK'])==(0 if mode in (7,8,9) else 1)
        for kind in ('ROOT_ACK','QUANT_ACK'):
            for row in e[kind]:assert (row['sequence'],row['epoch'])==(seq,epoch)
        if e['ROOT_ACK']:assert e['ROOT_ACK'][0]['cycle']>e['DONE'][0]['cycle']
        expected_count=0 if mode in (1,2,3,7,9) else 8 if mode==4 else 1 if mode in (5,8) else 64
        expected_aw=0 if mode==8 else expected_count
        assert (len(e['AW']),len(e['W']),len(e['B']))==(expected_aw,expected_count,expected_aw)
        for kind in ('AW','W','B'):assert [r['record'] for r in e[kind]]==list(range(len(e[kind])))
        for r in e['AW']:assert r['address']==address+64*r['record'] and r['cycle']>e['ROOT_ACK'][0]['cycle']
        for r in e['W']:
            record=r['record'];want=sum(v<<(64*i) for i,v in enumerate(sample['abi1'][record*8:record*8+8]))
            assert r['data']==want and r['cycle']>e['ROOT_ACK'][0]['cycle']
        for r in e['B']:
            n=r['record'];assert r['response']==(2 if mode==4 and n==7 else 0)
            assert r['cycle']>max(e['AW'][n]['cycle'],e['W'][n]['cycle'])
        if e['QUANT_ACK']:
            ack=e['QUANT_ACK'][0]['cycle']
            assert ack>e['ROOT_ACK'][0]['cycle'] and all(ack>r['cycle'] for r in e['B'])
        if mode in (7,8,9):
            assert not e['COMPLETE'] and len(e['ABORT'])==1 and len(e['RESET'])==1
            assert all(e['ABORT'][0][k]==v for k,v in [('mode',mode),('aw',expected_aw),('w',expected_count),('b',expected_aw)])
            assert e['RESET'][0]['cycle']>=e['ABORT'][0]['cycle']
            if mode==9:assert e['FOREIGN_HELD'][0]['cycle']>=e['DONE'][0]['cycle']+32
        else:
            assert not e['ABORT'] and len(e['COMPLETE'])==1
            r=e['COMPLETE'][0];status={1:7,2:2,3:5,4:3,5:7}.get(mode,0)
            valid=int(status==0 and mode!=6);records=64 if status==0 else 7 if mode==4 else 0
            assert (r['mode'],r['status'],r['hls'],r['sequence'],r['epoch'],r['records'],r['valid'],r['address'],r['bytes'])==(mode,status,2 if mode==3 else 0,seq,epoch,records,valid,address if valid else 0,4096 if valid else 0)
            assert (r['aw'],r['w'],r['b'])==(expected_aw,expected_count,expected_aw)
            assert r['cycle']>e['QUANT_ACK'][0]['cycle']
            if mode==6:assert r['cycle']>=e['REVOKED'][0]['cycle']
            if mode==10:
                reuse=e['REUSE'][0];assert (reuse['sequence'],reuse['epoch'])==(seq^(1<<40),epoch^(1<<44))
                assert e['ROOT_ACK'][0]['cycle']<=reuse['cycle']<e['QUANT_ACK'][0]['cycle']
            results.append(dict(job=j,mode=mode,cycles=r['cycle']-begin['cycle'],published_bytes=r['bytes']))
    mapping={'jobs':'BEGIN','publics':'COMPLETE','root':'ROOT_ACK','quant':'QUANT_ACK','resets':'RESET','reads':'READ','writes':'WORD','starts':'START','dones':'DONE','aw':'AW','w':'W','b':'B'}
    for key,kind in mapping.items():assert summary[key]==len(selected(kind)),key
    assert all(summary[k]>0 for k in ('aw_stalls','w_stalls','held','aw_first','w_first','cycles'))
    assert summary['cycles']>=max(r.get('cycle',0) for _,r in events)
    assert not any(r.get('job',0) not in range(25) for _,r in events)
    return dict(status='PASS_ATTENTION_QUANTIZE_STORE_R1_TRACE',summary=summary,jobs=24,completions=results,published_bytes=sum(r['published_bytes'] for r in results))

def controls(text,samples,seed=1):
    variants={}
    for kind in ('BEGIN','START','READ','INPUT','WORD','DONE','AW','W','B','ROOT_ACK','QUANT_ACK','ABORT','RESET','REUSE','FOREIGN_HELD','REVOKED','COMPLETE','PASS'):
        variants['missing_'+kind.lower()]=re.sub(r'(?m)^QSTORE_'+kind+r' [^\n]*\n','',text,count=1)
    def alter(kind,key,fn):
        pattern=r'(?m)^(QSTORE_'+kind+r' [^\n]*?\b'+key+r'=)([0-9a-f]+)'
        return re.sub(pattern,lambda m:m[1]+fn(m[2]),text,count=1)
    variants['input_bits']=alter('INPUT','data',lambda v:f'{int(v,16)^1:08x}')
    variants['output_bits']=alter('WORD','data',lambda v:f'{int(v,16)^1:016x}')
    variants['axi_bits']=alter('W','data',lambda v:f'{int(v,16)^1:0128x}')
    variants['parent_high_sequence']=alter('COMPLETE','sequence',lambda v:f'{int(v,16)^(1<<40):016x}')
    variants['parent_high_epoch']=alter('COMPLETE','epoch',lambda v:f'{int(v,16)^(1<<44):016x}')
    variants['early_root_ack']=alter('ROOT_ACK','cycle',lambda v:'0')
    variants['early_quant_ack']=alter('QUANT_ACK','cycle',lambda v:'0')
    variants['bus_response']=alter('B','response',lambda v:'2')
    variants['coverage']=alter('PASS','jobs',lambda v:'23')
    variants['diagnostic']=text+'\nQSTORE_FAIL injected\n'
    variants['unknown_event']=text+'\nQSTORE_UNKNOWN value=1\n'
    variants['duplicate_pass']=text+next(s for s in text.splitlines() if s.startswith('QSTORE_PASS '))+'\n'
    rejected=[]
    for name,corrupt in variants.items():
        assert corrupt!=text,name
        try:review(corrupt,samples,seed)
        except (AssertionError,KeyError,ValueError,IndexError):rejected.append(name)
        else:raise AssertionError('Accepted corruption: '+name)
    return rejected
