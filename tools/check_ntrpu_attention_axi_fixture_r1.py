"""Independently reconstruct physical KV layout from append beats and commands."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import struct

PARENT_SHA='6c992c47e5a3e9f2d6fc95c3db291b6d83a96cf3483bb2ba5e96f3f54976f361'
PARENT_MANIFEST='5338eeb1d0fa4291f67ecc5e7bf06ec3fa55e2846160aab89bc04e561709b722'


def parent_jobs(folder):
    raw=(folder/'llama32_attention_r1.bin').read_bytes()
    assert hashlib.sha256(raw).hexdigest()==PARENT_SHA
    metadata=(folder/'manifest.json').read_bytes();assert hashlib.sha256(metadata).hexdigest()==PARENT_MANIFEST
    m=json.loads(metadata);assert raw[:12]==b'NTRATTN1'+struct.pack('<I',143)
    offset=12;jobs=[]
    for spec in m['jobs']:
        header=struct.unpack_from('<6I2Q',raw,offset);offset+=40
        assert header==tuple(spec[k] for k in ('id','tokens','layer','query_head','kv_head','tile_width','sequence','epoch'))
        job=dict(spec)
        for name,n in [('query',64),('keys',spec['tokens']*64),('values',spec['tokens']*64),('expected',64)]:
            data=raw[offset:offset+4*n];offset+=4*n
            assert hashlib.sha256(data).hexdigest()==spec['hashes'][name]
            job[name]=struct.unpack('<'+'I'*n,data)
        jobs.append(job)
    assert offset==len(raw) and len(jobs)==143
    return jobs,raw


def hexwords(data,digits,count):
    lines=data.decode('ascii').splitlines()
    assert len(lines)==count and all(re.fullmatch('[0-9a-f]{'+str(digits)+'}',line) for line in lines)
    return [int(line,16) for line in lines]


def review_payload(manifest,files,jobs,parent):
    assert manifest['format']=='NTRPU_ATTENTION_AXI_R1'
    assert manifest['parent_fixture_sha256']==PARENT_SHA and manifest['parent_manifest_sha256']==PARENT_MANIFEST
    assert (manifest['cases'],manifest['real_cases'],manifest['tiles'],manifest['appends'],manifest['append_beats'])==(143,128,147,747,47808)
    assert len(manifest['images'])==16 and len(manifest['jobs'])==143
    beats=hexwords(files['kv_beats.hex'],128,47808)
    queries=hexwords(files['query.hex'],8,9152);outputs=hexwords(files['expected.hex'],8,9152)
    commands=hexwords(files['commands.hex'],40,143);descriptors=hexwords(files['images.hex'],32,16)
    fixed=bytearray(parent);offset=12
    for job in jobs:
        struct.pack_into('<I',fixed,offset+20,64);offset+=40+512+job['tokens']*512
    assert files['native-fixture/llama32_attention_r1.bin']==fixed, 'partition changed numerical input'
    nm=json.loads(files['native-fixture/manifest.json'])
    assert nm['transactions']==147 and nm['files']['llama32_attention_r1.bin']==hashlib.sha256(fixed).hexdigest()
    for a,b in zip(nm['jobs'],jobs):
        assert a['tile_width']==64
        assert all(a[k]==b[k] for k in ('id','tokens','layer','query_head','kv_head','sequence','epoch','hashes'))
    physical={};next_beat=0;compared=0;read_addresses=set()
    for image in manifest['images']:
        ident=image['id'];capacity=4 if ident==1 else jobs[126+ident]['tokens']
        assert 1<=ident<=16 and image['capacity']==capacity and image['first_beat']==next_beat
        offset=4032 if ident%2 else 64
        assert image['cache_offset']==offset and image['sequence']==0x1234567800000000+ident and image['epoch']==0xfedcba9800000000+ident
        assert descriptors[ident-1]==ident+(capacity<<32)+(next_beat<<64)+(offset<<96)
        assert image['beats']==capacity*64
        # Reconstruct byte addresses using channel strides, independently of
        # the generator's nested plane/head/word loop.
        memory={}
        for beat_index in range(capacity*64):
            position,token_index=divmod(beat_index,64)
            channel,word=divmod(token_index,4)
            address=0xa000000000+offset+channel*capacity*256+position*256+word*64
            assert address not in memory and address%64==0 and address+64<=0xa001000000
            memory[address]=beats[next_beat+beat_index]
        physical[ident]=memory;next_beat+=capacity*64
    assert next_beat==47808
    for i,(command,spec,job) in enumerate(zip(commands,manifest['jobs'],jobs)):
        ident=i+1;image=1 if i<128 else i-126;capacity=manifest['images'][image-1]['capacity']
        assert command==ident+(image<<32)+((job['tokens']-1)<<64)+(job['query_head']<<96)+(job['layer']<<128)
        assert (spec['id'],spec['image'],spec['position'],spec['head'],spec['layer'],spec['query_offset'],spec['output_offset'])==(
            ident,image,job['tokens']-1,job['query_head'],job['layer'],i*64,i*64)
        assert spec['lease']==0x1122334400000000+i and spec['tiles']==(job['tokens']+63)//64
        assert tuple(queries[i*64:i*64+64])==job['query'] and tuple(outputs[i*64:i*64+64])==job['expected']
        base=0xa000000000+manifest['images'][image-1]['cache_offset'];memory=physical[image]
        for plane,key in enumerate(('keys','values')):
            for index,wanted in enumerate(job[key]):
                position,lane=divmod(index,64)
                address=base+(plane*8+job['kv_head'])*capacity*256+position*256+(lane//16)*64
                actual=(memory[address]>>(32*(lane%16)))&0xffffffff
                assert actual==wanted, ('physical KV layout',ident,plane,index)
                compared+=1;read_addresses.add((image,address))
    assert compared==1063*128 and sum(s['tiles'] for s in manifest['jobs'])==147
    return dict(status='PASS_ATTENTION_AXI_R1_FIXTURE_LAYOUT',cases=143,real_cases=128,images=16,
        appends=747,physical_write_bytes=3059712,causal_tokens=1063,kv_words_reconstructed=compared,
        distinct_physical_read_addresses=len(read_addresses),query_words=9152,output_words=9152,
        native_fixed64_sha256=hashlib.sha256(fixed).hexdigest(),rtl_qualified=False)


def controls(manifest,files,jobs,parent):
    rejected=[]
    for name in ('kv_bit','kv_plane_swap','kv_head_swap','query_bit','result_bit','wrong_command_head',
                 'image_offset','image_capacity','missing_beat','extra_word','changed_tile_partition','late_real_prefix'):
        m=copy.deepcopy(manifest);f=files.copy()
        if name in ('kv_bit','kv_plane_swap','kv_head_swap','missing_beat','late_real_prefix'):
            lines=f['kv_beats.hex'].decode().splitlines()
            if name=='kv_bit':lines[0]=f'{int(lines[0],16)^1:0128x}'
            elif name=='kv_plane_swap':lines[:32],lines[32:64]=lines[32:64],lines[:32]
            elif name=='kv_head_swap':lines[:4],lines[4:8]=lines[4:8],lines[:4]
            elif name=='late_real_prefix':lines[192]=f'{int(lines[192],16)^1:0128x}'
            else:lines.pop()
            f['kv_beats.hex']=('\n'.join(lines)+'\n').encode()
        elif name in ('query_bit','result_bit','wrong_command_head','extra_word'):
            key={'query_bit':'query.hex','result_bit':'expected.hex','wrong_command_head':'commands.hex','extra_word':'query.hex'}[name]
            lines=f[key].decode().splitlines()
            if name=='extra_word':lines.append('00000000')
            else:
                bits=1<<96 if name=='wrong_command_head' else 1
                lines[0]=format(int(lines[0],16)^bits,'040x' if name=='wrong_command_head' else '08x')
            f[key]=('\n'.join(lines)+'\n').encode()
        elif name=='image_offset':m['images'][0]['cache_offset']+=64
        elif name=='image_capacity':m['images'][0]['capacity']+=1
        else:
            data=bytearray(f['native-fixture/llama32_attention_r1.bin']);struct.pack_into('<I',data,32,63)
            f['native-fixture/llama32_attention_r1.bin']=bytes(data)
        try:review_payload(m,f,jobs,parent)
        except (AssertionError,KeyError,IndexError,ValueError):rejected.append(name)
        else:raise AssertionError('Accepted layout corruption: '+name)
    return rejected


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--fixture',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    jobs,raw=parent_jobs(args.parent);manifest=json.loads((args.fixture/'manifest.json').read_text())
    files={name:(args.fixture/name).read_bytes() for name in manifest['files']}
    for name,data in files.items():assert hashlib.sha256(data).hexdigest()==manifest['files'][name]
    result=review_payload(manifest,files,jobs,raw);result['rejected_semantic_controls']=controls(manifest,files,jobs,raw)
    result.update(manifest_sha256=hashlib.sha256((args.fixture/'manifest.json').read_bytes()).hexdigest(),
        checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),files=manifest['files'])
    with args.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))


if __name__=='__main__':main()
