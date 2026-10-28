"""Freeze independent software-reference vectors plus the reviewed native capture."""
import argparse, hashlib, json, sys
from pathlib import Path

from ntrpu_pcie_request_inventory import RequestInventory, AcceptedBeatMonitor
from ntrpu_pcie_straddle import LogicWord
from ntrpu_pcie_vcd import logic_vcd_rows
from check_ntrpu_xdma_pcie_inventory import CORE
from check_ntrpu_xdma_stop_inventory_r4 import markers


def place(*segments):
    data=bytearray(64)
    for offset,raw in segments:data[offset:offset+len(raw)]=raw
    return int.from_bytes(data,'little')


def rq_header(tag, address, count, write=False):
    return (address | count<<64 | int(write)<<75 | tag<<96).to_bytes(16,'little')


def rc_header(tag,count,remaining,address,terminal=True,error=0):
    return (address | error<<12 | remaining<<16 | int(terminal)<<30 | count<<32 |
            0x1a0<<48 | tag<<64).to_bytes(12,'little')


def rq(tag,address=0x100,count=1,first=15,last=None):
    if last is None:last=0 if count==1 else 15
    return place((0,rq_header(tag,address,count))), first | last<<8 | 1<<20 | 1<<26 | 3<<28


def rc(tag,payload=b'abcd',address=0x100,remaining=None,terminal=True,error=0):
    if remaining is None:remaining=len(payload)
    count=len(payload)//4
    delivered=min(remaining,len(payload)-(address&3)) if not error else 0
    return place((0,rc_header(tag,count,remaining,address,terminal,error)+payload)),\
           ((1<<delivered)-1)<<(12+(address&3)) | 1<<64 | 1<<76 | (2+count)<<80


def binary(value,width):
    if isinstance(value,LogicWord):
        return ''.join(str((value.value>>n)&1) if value.known>>n&1 else 'x' for n in reversed(range(width)))
    return format(value,'0'+str(width)+'b')


class Vectors:
    def __init__(self):
        self.rows=[];self.scenarios=[];self.case=None;self.time=0

    def case_begin(self,name):
        self.case=dict(id=len(self.scenarios),name=name,first_vector=len(self.rows),fault_expected=False)
        self.scenarios.append(self.case)
        self.model=RequestInventory({0:0x1a0})
        self.monitors={n:AcceptedBeatMonitor(n) for n in ('rq','rc')}
        self.observing=False;self.fault=False;self.untrusted=False
        self.step(cold=0);self.step(cold=0);self.step();self.step()

    def step(self,*,rq_beat=None,rc_beat=None,rq_ready=15,rc_ready=1,
             rq_valid=None,rc_valid=None,cold=1,profile=1,reset=0,link=1,
             requester=0x1a0,attr=3,expect_error=False,native_ps=0):
        self.time+=4000
        rd,ru=rq_beat or (0,0);cd,cu=rc_beat or (0,0)
        rv=int(rq_beat is not None) if rq_valid is None else rq_valid
        cv=int(rc_beat is not None) if rc_valid is None else rc_valid
        if not cold:self.observing=False
        elif not self.observing:
            if profile and not reset and link:
                self.observing=True
                if rv or cv: self.fault=True
        elif not self.untrusted:
            if not profile or reset or not link or requester!=0x1a0 or attr!=3:
                self.fault=True
            try:
                rb=self.monitors['rq'].sample(rv,rq_ready,rd,ru)
                cb=self.monitors['rc'].sample(cv,rc_ready,cd,cu)
                self.model.cycle(self.time,rq=rb,rc=cb)
                self.fault |= bool(self.model.faults)
            except ValueError:
                if not expect_error:raise
                self.fault=True;self.untrusted=True
        if expect_error:
            assert self.fault, ('negative vector unexpectedly passed',self.case['name'])
            self.case['fault_expected']=True
        trustworthy=not self.untrusted
        bitmap=sum(1<<tag for function,tag in self.model.tags)
        span=sum(entry['remaining'] for entry in self.model.live.values())
        values=[self.case['id'],native_ps,cold,profile,reset,link,format(requester,'x'),format(attr,'x'),
                rv,format(rq_ready,'x'),binary(rd,512),format(ru,'x'),cv,rc_ready,binary(cd,512),format(cu,'x'),
                int(self.observing),int(self.fault),int(trustworthy),len(self.model.tags),len(self.model.live),span,
                int(self.model.rq.pending is not None),int(self.model.rc.pending is not None),
                len(self.model.requests),len(self.model.completed_reads),sum(r['kind']=='write' for r in self.model.requests),
                format(bitmap,'x')]
        self.rows.append(' '.join(map(str,values)))

    def end(self):
        self.step()
        if not self.fault:
            self.model.finish()
            for monitor in self.monitors.values():monitor.finish()
        else:
            self.step();self.step()
        self.case.update(last_vector=len(self.rows)-1,reference_fault=self.fault,
                         peak_tags=self.model.peak_tags,peak_reads=self.model.peak_reads,
                         peak_span_bytes=self.model.peak_span_bytes)


def generate(native):
    v=Vectors()
    v.case_begin('all_256_tags_RQ2_RC4_reversed')
    for pair in range(128):
        data=place((0,rq_header(2*pair,0x100,1)),(32,rq_header(2*pair+1,0x100,1)))
        user=0xff | 3<<20 | 2<<24 | 3<<26 | 3<<28 | 11<<32
        v.step(rq_beat=(data,user))
    for batch in range(64):
        pieces=[];user=15<<64 | 0xe4<<68 | 15<<76 | 0xfb73<<80
        for slot in range(4):
            tag=255-(4*batch+slot)
            pieces.append((16*slot,rc_header(tag,1,4,0x100)+tag.to_bytes(4,'little')))
            user|=15<<(16*slot+12)
        v.step(rc_beat=(place(*pieces),user))
    v.end()
    v.case_begin('tag_reuse_with_old_terminal_payload')
    v.step(rq_beat=rq(7,count=16))
    v.step(rc_beat=(place((0,rc_header(7,16,64,0x100)),(12,bytes(range(52)))),((1<<52)-1)<<12 | 1<<64),rq_beat=rq(7,0x200))
    v.step(rc_beat=(place((0,bytes(range(52,64)))),0xfff | 1<<76 | 2<<80))
    v.step(rc_beat=rc(7,address=0x200));v.end()
    v.case_begin('same_tag_two_splits_one_beat')
    v.step(rq_beat=rq(8,count=8))
    raw=place((0,rc_header(8,4,32,0x100,False)+bytes(range(16))),
              (32,rc_header(8,4,16,0x110)+bytes(range(16,32))))
    v.step(rc_beat=(raw,0xffff<<12 | 0xffff<<44 | 3<<64 | 2<<70 | 3<<76 | 6<<80 | 14<<84));v.end()
    v.case_begin('4KiB_completion_65_beats')
    v.step(rq_beat=rq(9,0,count=1024))
    raw=rc_header(9,1024,4096,0)+bytes(n%251 for n in range(4096))
    for index,offset in enumerate(range(0,len(raw),64)):
        chunk=raw[offset:offset+64]
        user=sum(1<<n for n in range(len(chunk)) if offset+n>=12) | (1<<64 if index==0 else 0)
        if offset+64>=len(raw):user|=1<<76 | (len(chunk)//4-1)<<80
        v.step(rc_beat=(place((0,chunk)),user))
    v.end()
    for first,last,count,low,span in [(5,0,1,0x100,3),(6,0,1,0x101,2),(0,0,1,0x100,1),(12,3,2,0x102,4)]:
        v.case_begin('byte_enables_'+str(first)+'_'+str(count))
        v.step(rq_beat=rq(1,count=count,first=first,last=last))
        v.step(rc_beat=rc(1,bytes(range(count*4)),address=low,remaining=span));v.end()
    v.case_begin('offered_rq_rc_backpressure')
    for _ in range(7):v.step(rq_beat=rq(3),rq_ready=0)
    v.step(rq_beat=rq(3))
    for _ in range(9):v.step(rc_beat=rc(3),rc_ready=0)
    v.step(rc_beat=rc(3));v.end()
    v.case_begin('posted_write_continuation_plus_new_read')
    pattern=bytes(range(64))
    v.step(rq_beat=(place((0,rq_header(0,0x600,16,True)),(16,pattern[:48])),0xf0f | 1<<20))
    v.step(rq_beat=(place((0,pattern[48:]),(32,rq_header(8,0x100,1))),15 | 1<<20 | 2<<22 | 3<<26 | 3<<28 | 11<<32))
    v.step(rc_beat=rc(8));v.end()
    v.case_begin('native_dma_bar_endpoint_capture')
    sim=next(text for name,text in native['logs'].items() if name.endswith('simulate.log'))
    start=int(markers(sim,'XDMA_LOCAL_BEGIN')[0]['t'])
    for t,b,a in logic_vcd_rows(native['logs']['native.vcd'],{CORE+n+'_tdata' for n in ('s_axis_rq','m_axis_rc')}):
        if t<start or b[CORE+'user_clk']!=0 or a[CORE+'user_clk']!=1:continue
        values={}
        for label,prefix in [('rq','s_axis_rq_'),('rc','m_axis_rc_')]:
            values[label+'_valid']=b[CORE+prefix+'tvalid'];values[label+'_ready']=b[CORE+prefix+'tready']
            # Idle data can be wholly unknown and is semantically unobserved.
            if values[label+'_valid']:
                values[label+'_beat']=(b[CORE+prefix+'tdata'],b[CORE+prefix+'tuser'])
        v.step(**values,native_ps=t)
    v.end()
    bad_rc=[('count',lambda d,u:(d^(1<<16),u)),('address',lambda d,u:(d^1,u)),
            ('requester',lambda d,u:(d^(1<<56),u)),('tc',lambda d,u:(d^(1<<89),u)),
            ('attribute',lambda d,u:(d^(1<<92),u)),('terminal',lambda d,u:(d^(1<<30),u)),
            ('enable',lambda d,u:(d,u^(1<<12))),('sop_mask',lambda d,u:(d,u | 4<<64)),
            ('orphan_tag',lambda d,u:(d^(1<<64),u)),
            ('unknown_payload',lambda d,u:(LogicWord(d,((1<<512)-1)^(1<<96),512),u))]
    for name,mutate in bad_rc:
        v.case_begin('fault_rc_'+name);v.step(rq_beat=rq(1));v.step(rc_beat=mutate(*rc(1)),expect_error=True);v.end()
    v.case_begin('fault_duplicate_tag');v.step(rq_beat=rq(1));v.step(rq_beat=rq(1),expect_error=True);v.end()
    v.case_begin('fault_request_4KiB_crossing');v.step(rq_beat=rq(1,0xffc,count=2),expect_error=True);v.end()
    v.case_begin('fault_rq_ready_not_replicated');v.step(rq_beat=rq(1),rq_ready=1,expect_error=True);v.end()
    v.case_begin('fault_stalled_valid_withdrawal');v.step(rq_beat=rq(1),rq_ready=0);v.step(expect_error=True);v.end()
    for signal,kwargs in [('link',dict(link=0)),('user_reset',dict(reset=1)),('profile',dict(profile=0)),('bdf',dict(requester=0x200))]:
        v.case_begin('fault_'+signal+'_preserves_pending');v.step(rq_beat=rq(1));v.step(**kwargs,expect_error=True);v.end()
    for error in (2,9):
        v.case_begin('fault_completion_error_'+str(error));v.step(rq_beat=rq(1))
        if error==9:
            data=place((0,rc_header(1,0,0,0,error=9)))
            beat=(LogicWord(data,15<<12 | 1<<30 | 255<<48 | 255<<64,512),1<<64 | 1<<76 | 2<<80)
        else:beat=rc(1,b'',remaining=0,error=error)
        v.step(rc_beat=beat,expect_error=True);v.end()
    v.case_begin('fault_late_rc_discontinue');v.step(rq_beat=rq(7,count=16))
    v.step(rc_beat=(place((0,rc_header(7,16,64,0x100))),((1<<52)-1)<<12 | 1<<64))
    v.step(rc_beat=(0,0xfff | 1<<76 | 2<<80 | 1<<96),expect_error=True);v.end()
    return v


def main():
    p=argparse.ArgumentParser();p.add_argument('native_evidence',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args();raw=args.native_evidence.read_bytes();native=json.loads(raw)
    assert native['invocation']['exit_code']==0
    vectors=generate(native)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f:f.write('\n'.join(vectors.rows)+'\n')
    manifest=dict(schema=1,rows=len(vectors.rows),scenarios=vectors.scenarios,
                  vectors_sha256=hashlib.sha256(args.output.read_bytes()).hexdigest(),
                  generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  native_evidence_sha256=hashlib.sha256(raw).hexdigest(),native_source=native['invocation']['source_sha256'])
    with args.output.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(vectors.rows),scenarios=len(vectors.scenarios),sha256=manifest['vectors_sha256'])))


if __name__=='__main__':main()
