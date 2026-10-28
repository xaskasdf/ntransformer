"""Convert pinned attention inputs to full-head KV append images and commands.

Produces data for the actual memory/HLS composition; this does not qualify RTL.
The original 143 numerical cases are retained with production 64-token tiles.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

PARENT_SHA = '6c992c47e5a3e9f2d6fc95c3db291b6d83a96cf3483bb2ba5e96f3f54976f361'
MANIFEST_SHA = '5338eeb1d0fa4291f67ecc5e7bf06ec3fa55e2846160aab89bc04e561709b722'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def parent(folder):
    raw = (folder/'llama32_attention_r1.bin').read_bytes()
    metadata = (folder/'manifest.json').read_bytes()
    assert digest(raw) == PARENT_SHA and digest(metadata) == MANIFEST_SHA
    manifest = json.loads(metadata); assert raw[:12] == b'NTRATTN1'+struct.pack('<I',143)
    cursor = 12; jobs = []; fixed = bytearray(raw)
    for spec in manifest['jobs']:
        header = struct.unpack_from('<6I2Q',raw,cursor)
        assert header == tuple(spec[k] for k in ('id','tokens','layer','query_head','kv_head','tile_width','sequence','epoch'))
        struct.pack_into('<I',fixed,cursor+20,64); cursor += 40
        sections = {}
        for key, words in [('query',64),('keys',spec['tokens']*64),('values',spec['tokens']*64),('expected',64)]:
            block = raw[cursor:cursor+words*4]; cursor += words*4
            assert len(block) == words*4 and digest(block) == spec['hashes'][key]
            sections[key] = list(struct.unpack('<'+'I'*words,block))
            assert all(math.isfinite(struct.unpack('<f',struct.pack('<I',v))[0]) for v in sections[key])
        jobs.append(dict(spec, **sections))
    assert cursor == len(raw) and len(jobs) == 143
    return jobs, bytes(fixed), manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent',type=Path,required=True); parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(); assert not args.output.exists()
    jobs,fixed,manifest=parent(args.parent)
    real=jobs[:128]
    assert {(j['tokens'],j['query_head']) for j in real} == {(n,h) for n in range(1,5) for h in range(32)}
    assert all(j['category']=='real_embedding_norm_q8_qkv_rope_layer0' for j in real)
    # All four Q heads sharing one KV head must agree across every prefix.
    real_words={}
    for job in real:
        for plane,key in enumerate(('keys','values')):
            for token in range(job['tokens']):
                for lane in range(64):
                    identity=(plane,job['kv_head'],token,lane); value=job[key][token*64+lane]
                    assert identity not in real_words or real_words[identity] == value
                    real_words[identity]=value
    assert len(real_words)==2*8*4*64
    images=[];beats=[];qwords=[];expected=[];commands=[];image_descriptors=[];specs=[]
    def image(capacity, selected=None):
        ident=len(images)+1; first=len(beats); offset=4032 if ident%2 else 64
        for pos in range(capacity):
            for plane in range(2):
                for head in range(8):
                    for word in range(4):
                        value=0
                        for sub in range(16):
                            lane=word*16+sub
                            if selected is None:
                                bits=real_words[plane,head,pos,lane]
                            elif head==selected['kv_head']:
                                bits=selected['keys' if plane==0 else 'values'][pos*64+lane]
                            else:
                                bits=0x3e000000|((ident*131071+plane*65537+head*4099+pos*97+lane)&0x7fffff)
                            value |= bits << (sub*32)
                        beats.append(value)
        row=dict(id=ident,capacity=capacity,first_beat=first,beats=capacity*64,cache_offset=offset,
                 sequence=0x1234567800000000+ident,epoch=0xfedcba9800000000+ident)
        images.append(row)
        image_descriptors.append(ident|(capacity<<32)|(first<<64)|(offset<<96))
        return ident
    image(4)
    for job in jobs:
        ident=1 if job['id']<=128 else image(job['tokens'],job)
        spec=dict(id=job['id'],image=ident,position=job['tokens']-1,layer=job['layer'],head=job['query_head'],
                  category=job['category'],label=job['label'],query_offset=len(qwords),output_offset=len(expected),
                  lease=0x1122334400000000+job['id']-1,tiles=(job['tokens']+63)//64)
        qwords.extend(job['query']);expected.extend(job['expected']);specs.append(spec)
        commands.append(spec['id']|(ident<<32)|(spec['position']<<64)|(spec['head']<<96)|(spec['layer']<<128))
    assert len(images)==16 and sum(r['capacity'] for r in images)==747
    assert len(beats)==47808 and len(qwords)==len(expected)==9152 and sum(r['tiles'] for r in specs)==147
    args.output.mkdir();native=args.output/'native-fixture';native.mkdir()
    (native/'llama32_attention_r1.bin').write_bytes(fixed)
    native_manifest=dict(manifest)
    native_manifest.update(jobs=[dict(j,tile_width=64) for j in manifest['jobs']],transactions=147,
        parent_fixture_sha256=PARENT_SHA,parent_manifest_sha256=MANIFEST_SHA,
        files={'llama32_attention_r1.bin':digest(fixed)},scope='Same 143 pinned inputs; production 64-token partition. Native arithmetic and future AXI/HLS composition only.')
    (native/'manifest.json').write_text(json.dumps(native_manifest,indent=2)+'\n')
    for name,words,digits in [('kv_beats.hex',beats,128),('query.hex',qwords,8),('expected.hex',expected,8),
                               ('commands.hex',commands,40),('images.hex',image_descriptors,32)]:
        with (args.output/name).open('x',newline='\n') as f:
            f.writelines(f'{v:0{digits}x}\n' for v in words)
    files={str(p.relative_to(args.output)).replace('\\','/'):digest(p.read_bytes()) for p in args.output.rglob('*') if p.is_file()}
    out=dict(format='NTRPU_ATTENTION_AXI_R1',parent_fixture_sha256=PARENT_SHA,parent_manifest_sha256=MANIFEST_SHA,
        model_sha256=manifest['model_sha256'],source_tensors=manifest['source_tensors'],files=files,
        generator_sha256=digest(Path(__file__).read_bytes()),images=images,jobs=specs,
        cases=143,real_cases=128,tiles=147,appends=747,append_beats=47808,publishable_write_bytes=747*4096,
        queries_words=9152,expected_words=9152,causal_tokens=sum(j['tokens'] for j in jobs),
        layout=dict(region_base=0xa000000000,region_bytes=16*1024*1024,
                    address='base + (((layer*2+plane)*8+kv_head)*capacity+position)*256 + word*64',
                    append_index='plane*32 + kv_head*4 + word',command_words=['id','image','position','head','layer'],
                    image_words=['id','capacity','first_beat','cache_offset']),
        scope='143 original numerical cases; physical input images only. No RTL or hardware result claimed.')
    with (args.output/'manifest.json').open('x') as f:
        json.dump(out,f,indent=2);f.write('\n')
    print(json.dumps({k:v for k,v in out.items() if k not in ('images','jobs','files','source_tensors')}))


if __name__=='__main__':
    main()
