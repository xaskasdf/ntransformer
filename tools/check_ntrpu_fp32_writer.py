"""Independently reconstruct all physical FP32/FP64 writer beats and masks."""
import re
import struct
from check_ntrpu_result_writer import CASES as BASE_CASES

CASES=[(*c,1) for c in BASE_CASES]+[
    (16,4032,35,0,1),(17,4032,35,0,1),(31,4032,35,0,1),
    *[(9,0,m,5,1) for m in range(36,41)],(9,0,41,0,0),
    (9,0,42,5,1),(9,0,43,7,1),(9,0,44,0,1),(17,4032,35,0,1),
    (31,4032,0,0,0),(31,4032,0,0,1)]


def value(ident,row,mode):
    if mode==35:
        vals=[0.0,-0.0,*[float.fromhex(s) for s in [
            '0x1p-149','-0x1p-149','0x1p-150','0x1.0000000000001p-150',
            '0x1.fffffcp-127','0x1.fffffep-127','0x1p-126','0x1.000001p+0',
            '0x1.000003p+0','0x1.fffffep+127','-0x1.fffffep+127','0x1p-1074','-0x1p-1074']],1.25]
        return vals[row%16]
    if mode==41 and row==8:return struct.unpack('<d',bytes.fromhex('010000e0ffffef47'))[0]
    if mode==33 and row==0:return -0.0
    return (ident*100+row)*0.25


def review(text,seed,netlist=False):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    def records(prefix):
        return [{k:int(v,16) if k in {'data','strb'} else int(v) for k,v in re.findall(r'(\w+)=([0-9a-f]+)',line)}
                for line in text.splitlines() if line.startswith(prefix+' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(records('NTRPU_FPWRITER_SEED')==[dict(seed=seed)],'Seed')
    need(records('NTRPU_FPWRITER_CONFIG')==[dict(watchdog=65536 if netlist else 32)],'Watchdog')
    cases=records('NTRPU_FPWRITER_CASE');writes=records('NTRPU_FPWRITER_W')
    need([c.get('id') for c in cases]==list(range(1,58)),'Case identities')
    expected_writes=[];physical=0
    for ident,(c,(n,offset,mode,status,fp32)) in enumerate(zip(cases,CASES),1):
        reset=mode in {28,29,44};size=4 if fp32 else 8
        need(tuple(c.get(k) for k in ['rows','offset','scenario','status','fp32'])==(n,offset,mode,status,fp32),'Case contract '+str(ident))
        need(c.get('reset')==int(reset),'Reset')
        need(0<c.get('cycles',0)<(80000 if netlist else 2000),'Latency bound')
        inputs=0 if status in {1,2} or mode==24 else n-1 if mode==19 else n+1 if mode==20 else 70 if mode==42 else 2 if mode==44 else n
        need(c.get('inputs')==inputs,'Input drain '+str(ident))
        if status in {1,2,5,6,7}:
            beats=0;count=0
            need(all(c.get(k)==0 for k in ['aw','w','b','bytes']),'No writes on rejection')
        else:
            beats=(n*size+63)//64;count=n*size;bursts=1+int(offset+beats*64>4096)
            if mode in {7,9,11,12,13}:beats=1;bursts=1;count=64
            if reset:
                beats=1 if mode==28 else 2 if mode==29 else 0;count=beats*64
                need(c.get('w')==beats and c.get('bytes')==count and c.get('b')==0,'Reset traffic')
                need(0<=c.get('aw',-1)<=int(mode!=44),'Reset address')
            else:
                need(tuple(c.get(k) for k in ['aw','w','b','bytes'])==(bursts,beats,bursts,count),'AXI accounting '+str(ident))
                if status==8:need(c.get('cycles',0)>=(65584 if netlist else 80),'Timeout stimulus')
        for beat in range(beats):
            start=beat*64;raw=b''.join(struct.pack('<f' if fp32 else '<d',value(ident,r,mode))
                                       for r in range(start//size,min(n,(start+64)//size)))
            expected_writes.append(dict(id=ident,beat=beat,data=int.from_bytes(raw.ljust(64,b'\0'),'little'),strb=(1<<len(raw))-1))
            physical+=len(raw)
    need(writes==expected_writes,'All numerical W data, masks and ordered beats')
    summaries=records('NTRPU_FPWRITER_PASS');need(len(summaries)==1,'Suite completion')
    summary=summaries[0] if summaries else {}
    need(summary.get('cases')==57 and summary.get('resets')==3,'Coverage totals')
    for k in ['aw','w','bytes']:need(summary.get(k)==sum(c.get(k,0) for c in cases),'Total '+k)
    need(summary.get('bytes')==physical==2608,'Physical payload bytes')
    for k in ['aw_stalls','w_stalls','done_stalls','aw_before_w','w_before_aw']:need(summary.get(k,0)>0,'Coverage '+k)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,summary=summary,cases=cases,
                verified_w_payload_bytes=physical,scope='Mixed FP32/FP64 vector writer with conversion, private staging and synthetic AXI peer.')
