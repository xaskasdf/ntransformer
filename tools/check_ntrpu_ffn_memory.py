"""Independent physical-byte reference for the synthetic memory FFN bank."""
import re
import struct

CASES=[(32,32,0,0),(64,96,0,0),(64,80,0,0),
       *[(64,96,m,p) for m in range(1,5) for p in range(4)],(64,96,0,0),(64,96,5,0)]
RECOVERY_CASES = CASES[:15] + [case for phase in range(4)
    for case in [(64,96,4,phase),(64,96,0,0)]] + [(64,96,5,0)]


def reference(dim,hidden,fp32=False):
    def q(index):
        lane=index%32
        return 127 if lane==0 else 1 if lane==1 else lane*7-113
    def doubles(values):return b''.join(struct.pack('<d',v) for v in values)
    records=[]
    for block in range((hidden+31)//32):
        payload=bytes(q(block*32+i)&255 if block*32+i<hidden else 0 for i in range(32))
        records.append(payload+struct.pack('<d',257+2*block)+bytes(24))
    down=[(257+2*(((i*7)%hidden)//32))*q((i*7)%hidden) for i in range(dim)]
    return {4:doubles(257+2*(i//32) for i in range(hidden)),
            5:doubles(q(i) for i in range(hidden)),6:b''.join(records),
            7:b''.join(struct.pack('<f' if fp32 else '<d',v) for v in down)}


def review(text,seed,fp32=False,*,delayed_reset=False,recovery_each=False):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    def records(prefix):
        return [{k:int(v,16) if k in {'data','strb'} else int(v) for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)}
                for line in text.splitlines() if line.startswith(prefix+' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(records('NTRPU_FFN_MEMORY_SEED')==[dict(seed=seed)],'Seed')
    if fp32:need(records('NTRPU_FFN_MEMORY_FORMAT')==[dict(fp32=1)],'FP32 output format')
    specs=RECOVERY_CASES if recovery_each else CASES
    cases=records('NTRPU_FFN_MEMORY_CASE');writes=records('NTRPU_FFN_MEMORY_W')
    need([r.get('id') for r in cases]==list(range(1,len(specs)+1)),'Case identities')
    need(all(1<=w.get('id',0)<=len(specs) for w in writes),'Write case identity')
    verified_bytes=0
    for ident,(case,contract) in enumerate(zip(cases,specs),1):
        d,h,mode,target=contract;refs=reference(d,h,fp32)
        need(tuple(case.get(k) for k in ['dim','hidden','mode','target'])==contract,'Case shape '+str(ident))
        need(case.get('reset')==int(mode==4),'Reset coverage')
        need(0<case.get('cycles',0)<250000,'Latency')
        need(sum(case.get('p'+str(p),0) for p in range(4))==case.get('cycles'),'Phase latency total')
        chunks={b:[] for b in range(4,8)}
        for w in [w for w in writes if w.get('id')==ident]:
            b=w.get('buffer');off=w.get('offset',-1)
            if b not in refs:
                errors.append('Invalid destination buffer');continue
            need(off==64*len(chunks[b]),'Sequential physical writes')
            need(w.get('strb')==2**64-1,'Byte mask')
            try:actual=w.get('data',0).to_bytes(64,'little')
            except OverflowError:actual=b''
            need(actual==refs[b][off:off+64] and len(actual)==64,'Independent numerical reference '+str((ident,b,off)))
            chunks[b].append(actual);verified_bytes+=64
        lengths={b:len(v)*64 for b,v in chunks.items()}
        if mode==0:expected={b:len(v) for b,v in refs.items()}
        elif mode==5:expected={b:0 for b in refs}
        else:
            partial=0 if mode==1 else 64 if mode==2 or target==2 else 128 if fp32 and target==3 else 256
            if mode==4:
                partial=lengths[target+4]
                # Automatic clock-loss detection takes time. The other clock
                # may complete additional beats before reset. Numerical bytes,
                # contiguous offsets, phase ownership and counters stay strict.
                # The caller must independently qualify that detection interval.
                if delayed_reset:need(64<=partial<=len(refs[target+4]),'Delayed reset write prefix')
                else:need(partial in {0,64},'Reset write prefix')
            expected={b:len(refs[b]) if b<target+4 else partial if b==target+4 else 0 for b in refs}
        need(lengths==expected,'Phase write counts '+str(ident))
        physical=sum(lengths.values())
        need(case.get('bytes')==physical if mode!=4 else physical<=case.get('bytes',-1)<=physical+64,'Strobed bytes')
        need(case.get('bytes')==case.get('w',-1)*64,'W byte accounting')
        need(case.get('down_bytes')==lengths[7] if mode!=4 else lengths[7]<=case.get('down_bytes',-1)<=lengths[7]+64,'Final private byte count')
        need(case.get('aw')==case.get('b') if mode!=4 else case.get('b',0)<=case.get('aw',0)<=case.get('b',0)+1,'B drain')
        if mode==0:
            dk=(d+31)//32;hk=(h+31)//32
            reads=2*((h+31)//32*dk+h*dk*34//64)+2*((h+7)//8)+(d+31)//32*hk+d*hk*34//64
            need(case.get('r')==reads,'Successful read beats')
        if mode==5:need(case.get('ar')==case.get('r')==case.get('aw')==case.get('w')==case.get('b')==0,'Preflight traffic')
    summaries=records('NTRPU_FFN_MEMORY_PASS');need(len(summaries)==1,'Summary count')
    summary=summaries[0] if summaries else {}
    need(summary.get('cases')==len(specs) and summary.get('resets')==4 and summary.get('cancels')==4,'Coverage totals')
    for key in ['cycles','bytes']:need(summary.get(key)==sum(c.get(key,0) for c in cases),'Total '+key)
    for key in ['ar_stalls','aw_stalls','w_stalls','done_stalls']:need(summary.get(key,0)>0,'Backpressure '+key)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=cases,summary=summary,
                verified_physical_bytes=verified_bytes,
                scope='Actual HLS MAC/SwiGLU RTL and shared synthetic AXI RAM; sparse synthetic weights, exact closed-form reference; no real-model inference.')
