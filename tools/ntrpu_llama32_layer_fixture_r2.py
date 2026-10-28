"""Join full layer-0/token-0 references without re-quantizing any intermediate."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import ntrpu_input_attention_fixture_r1 as front
import check_ntrpu_attention_ffn_real_r2 as tail
from ntrpu_llama32_layer_layout_r1 import ALLOCATIONS

PARENT_MANIFESTS={'front':'04df4a83b7c6ef8cae36904fdc44d54a1e56ca8088e97d9352a892c8597ade68',
    'output':'73a727d73b1fe6024b1346bf23abda93b66b194d08a53a87799a5867a4b3091e',
    'root':'4ac0ed169bbf5617e24662783182844d6cdbaa07cc5f8214b9d1f6ca01937cc1',
    'post':'7a59fdb2832b0b5547872ecff7b685dbc839e49c8cb0054436441efc276a0f75',
    'ffn':'bf36c6afbe46df663aa9823769f6c6c74e1608826fce7182e21ffe42646e0194',
    'extra':'b14e8731ce63420cbd1d775f241b8f621e491caf189d8deab2a803136f7a49e0'}
MASK=(1<<64)-1
FNV=0xcbf29ce484222325

def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()

def address(region):return ((region+1)<<40)+4032

def pack(values,width):return b''.join(v.to_bytes(width,'little') for v in values)

def bursts(region,offset,count):
    rows=[]
    while count:
        n=min(count//64,64,(4096-(4032+offset)%4096)//64)
        assert n>0;rows.append(dict(region=region,offset=offset,beats=n));offset+=64*n;count-=64*n
    return rows

def read_plan(ffn_reads):
    plans=[[],[],[],[],[]]
    for r in (0,1):plans[0]+=bursts(r,0,8192)
    for r,rows in [(3,2048),(4,512),(5,512)]:
        for group in range(rows//32):
            plans[0]+=bursts(2,0,4096)
            for tile in range(2):plans[0]+=bursts(r,group*69632+tile*34816,34816)
    for r,size in [(6,8192),(7,2048),(9,256)]:plans[1]+=bursts(r,0,size)
    plans[2]+=bursts(8,0,2048)
    for head in range(32):
        for plane in range(2):plans[3]+=bursts(24,((plane*8+head//4)*4)*256,256)
    for group in range(64):
        plans[4]+=bursts(11,0,4096)
        for tile in range(2):plans[4]+=bursts(10,group*69632+tile*34816,34816)
    for r in (0,12,13):plans[4]+=bursts(r,0,8192)
    mapping=[15,16,17,18,19,20,21,22,14,23]
    plans[4]+=[dict(region=mapping[r['buffer']],offset=r['offset'],beats=r['beats']) for r in ffn_reads]
    return plans

def build(parents,output):
    parents={k:Path(v) for k,v in parents.items()};output=Path(output);assert not output.exists()
    for k,h in PARENT_MANIFESTS.items():assert digest(parents[k]/'manifest.json')==h,k
    fm,producer,samples,refs=front.load(parents['front'])
    values,ffn=tail.reference(*[parents[k] for k in ('output','root','post','ffn','extra')])
    sample=samples[0];o=values['root']['samples'][0];post=values['post']
    assert sample['attention']==o['input_fp32']
    assert producer['norm']['source32.hex'][:2048]==post['source32.hex'][:2048]
    assert pack(post['abi64.hex'][:512],8)==ffn['buffers'][0]
    assert pack(post['residual32.hex'][:2048],4)==ffn['buffers'][8]
    regions={0:pack(producer['norm']['source32.hex'][:2048],4),
        1:pack(producer['norm']['source32.hex'][2048:4096],4),2:pack(producer['samples'][0]['activation'],64),
        9:pack(sample['input'][2560:],4),10:pack(values['root']['weight'][63:],64),
        11:pack(o['activation'],64),12:pack(o['fp32'],64),13:pack(post['source32.hex'][4096:6144],4),
        14:ffn['buffers'][8],15:ffn['buffers'][0],23:ffn['buffers'][9]}
    for i in range(3):
        regions[3+i]=pack(producer['weights'][i][63:],64)
        regions[6+i]=pack(producer['samples'][0]['fp32'][i],64)
    for i in range(1,8):regions[15+i]=ffn['buffers'][i]
    assert set(regions)==set(range(24))
    for i,(_,span,_) in enumerate(ALLOCATIONS):assert len(regions[i])==span,(i,len(regions[i]),span)
    cache=bytearray(262144);cache_beats=[]
    for plane in range(2):
        for head in range(8):
            words=sample['rope'][2048+head*64:2048+(head+1)*64] if plane==0 else sample['v'][head*64:(head+1)*64]
            offset=(plane*8+head)*4*256;cache[offset:offset+256]=pack(words,4)
            cache_beats.extend(range(offset//64,offset//64+4))
    regions[24]=bytes(cache);plans=read_plan(ffn['reads'])
    output.mkdir(parents=True)
    def write(name,data):
        p=output/name;p.parent.mkdir(parents=True,exist_ok=True)
        with p.open('xb') as f:f.write(data)
    entries=[];cursor=0
    for i in range(25):
        name,span,writable=ALLOCATIONS[i] if i<24 else ('kv_cache',262144,True)
        raw=regions[i];file=('expected' if writable else 'source')+f'-{i:02d}.hex'
        write(file, ''.join(f'{int.from_bytes(raw[n:n+64],"little"):0128x}\n' for n in range(0,len(raw),64)).encode())
        entries.append(dict(index=i,name=name,address=address(i),base=(i+1)<<40,region_bytes=span+4032,
            span=span,writable=writable,memory_start=cursor,frames=span//64,file=file,raw_sha256=hashlib.sha256(raw).hexdigest()))
        cursor+=span//64
    write('rope.hex',''.join(f'{v:08x}\n' for v in sample['rope']).encode())
    write('attention.hex',''.join(f'{v:08x}\n' for v in sample['attention']).encode())
    for name in ('manifest.json','llama32_attention_r1.bin'):
        write('attention-reference/'+name,(parents['front']/'consumer/attention-reference'/name).read_bytes())
    write('read-plan.json',(json.dumps(plans,separators=(',',':'))+'\n').encode())
    read_stats=[]
    for port,plan in enumerate(plans):
        stats={r:dict(beats=0,hash=FNV) for r in range(25)}
        for request in plan:
            r,off,n=(request[k] for k in ('region','offset','beats'));data=regions[r][off:off+64*n];assert len(data)==64*n
            for word, in struct.iter_unpack('<Q',data):stats[r]['hash']=((stats[r]['hash']^word)*0x100000001b3)&MASK
            stats[r]['beats']+=n
        read_stats.append({r:dict(beats=v['beats'],hash=f"{v['hash']:016x}") for r,v in stats.items()})
    manifest=dict(format='NTRPU_LLAMA32_LAYER_FIXTURE_R1',model_sha256=front.MODEL_SHA,layer=0,token=0,position=0,
        dim=2048,hidden=8192,cache_capacity=4,regions=entries,memory_frames=cursor,cache_written_beats=cache_beats,
        parent_manifests=PARENT_MANIFESTS,parents={k:str(p) for k,p in parents.items()},read_stats=read_stats,
        read_bursts=[len(p) for p in plans],read_beats=[sum(r['beats'] for r in p) for p in plans],
        write_beats=3264,source_input_reused=True,input_abi_requantized=False,source_projections_equal=True,
        attention_equal=True,post_ffn_abi_equal=True,post_ffn_residual_equal=True,
        generator_sha256=digest(Path(__file__)),
        files={str(p.relative_to(output)).replace('\\','/'):digest(p) for p in sorted(output.rglob('*')) if p.is_file()},
        scope='Reference identity and causal simulation inputs only. Writable memory must start invalid and be filled solely by actual AXI B commits. Full-layer RTL execution remains pending.')
    write('manifest.json',(json.dumps(manifest,indent=2)+'\n').encode());return manifest

def load(folder):
    folder=Path(folder);m=json.loads((folder/'manifest.json').read_text())
    assert m['format']=='NTRPU_LLAMA32_LAYER_FIXTURE_R1' and m['parent_manifests']==PARENT_MANIFESTS and m['model_sha256']==front.MODEL_SHA
    for n,h in m['files'].items():assert digest(folder/n)==h,n
    data={}
    for r in m['regions']:
        raw=b''.join(int(s,16).to_bytes(64,'little') for s in (folder/r['file']).read_text().split())
        assert len(raw)==r['span'] and hashlib.sha256(raw).hexdigest()==r['raw_sha256'];data[r['index']]=raw
    return m,data,json.loads((folder/'read-plan.json').read_text())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in PARENT_MANIFESTS:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--destination',type=Path,required=True);a=p.parse_args()
    m=build({k:getattr(a,k) for k in PARENT_MANIFESTS},a.destination)
    print(json.dumps(dict(manifest_sha256=digest(a.destination/'manifest.json'),read_beats=m['read_beats'],writes=m['write_beats'],files=len(m['files']))))
