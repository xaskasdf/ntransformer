#!/usr/bin/env python3
"""Check every conversion against native IEEE rounding, plus stream identities."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct

VECTOR_SHA = 'eb04850b7dc1709b2a0ad837cb5771200a5e1bd4143113164cd269b183480272'
LEGACY_CRLF_SHA = 'c71e5c3c00a07b7aa49cb34e88cc3448feb32f77d5101eca6dcef1bf77dbb4a1'


def load_vectors(path):
    raw=(path/'vectors.txt').read_bytes()
    # The first Windows producer wrote CRLF but hashed LF. Its exact raw hash
    # is separately pinned here and in the upload/archive manifest. Accept
    # only that immutable legacy file, not arbitrary newline transformations.
    actual=hashlib.sha256(raw).hexdigest()
    if actual==LEGACY_CRLF_SHA:
        raw=raw.replace(b'\r\n',b'\n')
    if hashlib.sha256(raw).hexdigest()!=VECTOR_SHA:
        raise ValueError('Pinned IEEE/real Brandon fixture identity')
    manifest=json.loads((path/'manifest.json').read_text())
    if manifest['vector_sha256']!=VECTOR_SHA or manifest['count']!=51928 or \
            manifest['brandon_start']!=51160 or manifest['brandon_count']!=768 or \
            manifest['brandon_manifest_sha256']!='bf01bf1a0aeb709d0da6b14bb0856b0395b03f490bf976224c37a01f0e0be33a':
        raise ValueError('Manifest identity')
    vectors=[]
    maximum=float.fromhex('0x1.fffffep+127')
    for line in raw.decode('ascii').splitlines():
        a,b,c=line.split();word=int(a,16);status=int(b);result=int(c,16)
        value=struct.unpack('>d',word.to_bytes(8,'big'))[0]
        expected_status=2 if not math.isfinite(value) else 3 if abs(value)>maximum else 0
        expected=int.from_bytes(struct.pack('>f',value),'big') if expected_status==0 else 0
        if (status,result)!=(expected_status,expected):raise ValueError('Native conversion oracle')
        vectors.append((word,status,result))
    if len(vectors)!=51928:raise ValueError('Count')
    return vectors


def review(text, vectors, seed):
    errors=[]
    def need(ok,message):
        if not ok:errors.append(message)
    def records(prefix):
        return [{k:int(v,16) if k in {'input','data','sequence','epoch'} else int(v)
                 for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)}
                for line in text.splitlines() if line.startswith(prefix+' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(records('NTRPU_FPCAST_SEED')==[dict(seed=seed,count=len(vectors))],'Seed/fixture count')
    need(records('NTRPU_FPCAST_RESET')==[dict(id=i+1,depth=d) for i,d in enumerate([1,2,3,11])],'Reset pipeline coverage')
    values=records('NTRPU_FPCAST_VALUE');need(len(values)==len(vectors),'Result count')
    for i,(v,(word,status,result)) in enumerate(zip(values,vectors)):
        expected=dict(id=i,input=word,data=result,status=status,index=i%16384,last=int(i%32==31),
                      sequence=i^0xfffffffe,epoch=(~i)&0xffffffff)
        actual={k:n for k,n in v.items() if k!='latency'}
        if actual!=expected:errors.append('Numerical/metadata/order mismatch '+str(i));break
        if not 3<=v.get('latency',0)<200:errors.append('Latency bound '+str(i));break
        if i<250 and v['latency']!=3:errors.append('Unstalled latency '+str(i));break
    summaries=records('NTRPU_FPCAST_PASS');need(len(summaries)==1,'Suite completion')
    summary=summaries[0] if summaries else {}
    need(summary.get('count')==summary.get('accepted')==51928 and summary.get('resets')==4,'Accepted/completed/reset totals')
    need(summary.get('stalls',0)>0 and summary.get('gaps',0)>0,'Backpressure/source gaps')
    latency=[v.get('latency',0) for v in values]
    need(bool(latency) and summary.get('min_latency')==min(latency,default=0)==3 and
         summary.get('max_latency')==max(latency,default=0)>3,'Measured latency totals')
    need(51928<summary.get('cycles',0)<500000,'Cycle accounting bound')
    return dict(status='FAIL' if errors else 'PASS',errors=errors,summary=summary,
                statuses={str(s):sum(t[1]==s for t in vectors) for s in [0,2,3]},
                brandon_down_values=768,scope='Standalone checked scalar conversion; no AXI/FFN integration or physical throughput.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('fixture',type=Path);p.add_argument('log',type=Path);p.add_argument('--seed',type=int,default=1)
    a=p.parse_args();result=review(a.log.read_text(),load_vectors(a.fixture),a.seed)
    print(json.dumps(result,indent=2));raise SystemExit(result['status']!='PASS')
