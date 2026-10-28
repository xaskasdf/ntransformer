"""Independent transaction queues for the isolated 512-bit AXI4 drain bank.

Pairs complete write transactions without modeling the RTL pairing pipeline.
Only positive cases qualify protocol traffic; negatives must expose precisely
their injected violation and retain the fault across a fresh drain identity.
"""
from collections import Counter, deque
import json
import re
import sys


def require(ok, message):
    if not ok:
        raise ValueError(message)


def pattern(seed):
    return sum(((seed ^ ((0x9e3779b9 * j) & 0xffffffff)) & 0xffffffff) << (32*j) for j in range(16))


def mask(beat, size):
    return (0xa5f3 << ((beat % 4)*16)) if size == 4 else 0xf0ff55aa12345678 ^ (0x0101010101010101*(beat % 16))


def review(text):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|Warning:|WARNING:)', text, re.M), 'simulator diagnostic')
    require(text.splitlines().count('AXI4_DRAIN_SUITE_PASS cases=22 native_endpoint=0') == 1, 'suite marker')
    starts=re.findall(r'^AXI4_CASE case=(\d+) write_depth=8 read_depth_per_id=8 ids=16 data_bits=512$', text,re.M)
    require(list(map(int,starts))==list(range(22)), 'case inventory')
    expected=[0]*5+[2,4,2,4,8,8,16,16,1,1,1,32,32,32,32,32,32]
    ends=re.findall(r'^AXI4_CASE_PASS case=(\d+) fault=(\d+)$',text,re.M)
    require([(int(a),int(b)) for a,b in ends]==list(enumerate(expected)), 'terminal case markers')
    hexkeys={'awaddr','wd','ws','araddr','rd','tag','acktag'}
    groups={i:[] for i in range(22)}
    for line in text.splitlines():
        if not line.startswith('AXI4_SAMPLE '):continue
        fields=dict(x.split('=') for x in line.split()[1:])
        require(all(re.fullmatch('[0-9a-fA-F]+' if k in hexkeys else '[0-9]+',v) for k,v in fields.items()), 'unknown sample value')
        row={k:int(v,16 if k in hexkeys else 10) for k,v in fields.items()}
        groups[row['case']].append(row)
    summaries=[]
    lengths=set();ids=set();response_codes=set();high_addresses=0
    for case,rows in groups.items():
        require(rows and [r['cycle'] for r in rows]==list(range(1,len(rows)+1)),f'complete trace {case}')
        awq=deque();wq=deque();partial=[];writes={i:deque() for i in range(16)};reads={i:deque() for i in range(16)}
        outstanding_aw=outstanding_w=0
        handshakes=Counter();stalls=Counter();errors=[];fault_at=None;frozen=None
        captured=None;quiet=0;ack_tags=set();last=None;peaks=Counter()
        def violation(bit):
            errors.append((r['cycle'],bit))
        for r in rows:
            for channel,v,ready in [('aw','awv','awr'),('w','wv','wr'),('b','bv','br'),('ar','arv','arr'),('r','rv','rr')]:
                if r[v] and not r[ready]:stalls[channel]+=1
                if r[v] and r[ready]:handshakes[channel]+=1
            # A prior stalled offer must survive, including its acceptance edge.
            payloads=[('awv','awr',['awid','awaddr','awlen','awsize']),('wv','wr',['wd','ws','wl']),
                      ('bv','br',['bid','bp']),('arv','arr',['arid','araddr','arlen','arsize']),
                      ('rv','rr',['rid','rd','rp','rl'])]
            if last:
                for v,ready,fields in payloads:
                    if last[v] and not last[ready] and (not r[v] or any(last[f]!=r[f] for f in fields)):violation(32)
            # Outstanding queues here reflect PRIOR accepted events. Never grant
            # causal credit from an address or data accepted on this same edge.
            before_empty=not(outstanding_aw or outstanding_w or partial or any(reads.values()))
            if r['bv']:
                if not writes[r['bid']]:violation(2)
                elif r['br']:
                    writes[r['bid']].popleft();outstanding_aw-=1;outstanding_w-=1
                    if case<5:response_codes.add(r['bp'])
            if r['rv']:
                if not reads[r['rid']]:violation(4)
                else:
                    desc=reads[r['rid']][0]
                    if r['rl'] != (desc['beat']==desc['length']-1):violation(16)
                    if case<5:
                        require(r['rd']==pattern((desc['addr']&0xffffffff)^(desc['beat']<<8)^r['rid']), f'R payload {case}/{r["cycle"]}')
                        require(r['rp']==desc['beat']%4,'R response')
                    if r['rr']:
                        desc['beat']+=1
                        if r['rl']:reads[r['rid']].popleft()
            if r['awv'] and r['awr']:
                if outstanding_aw>=8:violation(1)
                awq.append(dict(id=r['awid'],length=r['awlen']+1,size=r['awsize'],addr=r['awaddr']))
                outstanding_aw+=1
            if r['wv'] and r['wr']:
                if not partial and outstanding_w>=8:violation(1)
                partial.append((r['wd'],r['ws']))
                if r['wl']:
                    wq.append(partial);partial=[];outstanding_w+=1
                elif len(partial)>=256:violation(8)
            while awq and wq:
                addr=awq.popleft();beats=wq.popleft()
                if len(beats)!=addr['length']:violation(8)
                if case<5:
                    require(addr['size'] in {4,6},'tested write size')
                    require((addr['addr']%4096)+addr['length']*(1<<addr['size'])<=4096,'write 4K boundary')
                    for i,(data,strb) in enumerate(beats):
                        require(data==pattern((addr['addr']&0xffffffff)^(i<<8)),f'W payload {case}/{r["cycle"]}/{i}')
                        require(strb==mask(i,addr['size']), 'W strobe')
                    lengths.add(addr['length']);ids.add(addr['id']);high_addresses+=addr['addr']>0xffffffff
                writes[addr['id']].append(addr)
            if awq and len(partial)>=awq[0]['length']:violation(8)
            if r['arv'] and r['arr']:
                if len(reads[r['arid']])>=8:violation(1)
                reads[r['arid']].append(dict(length=r['arlen']+1,beat=0,addr=r['araddr']))
                if case<5:
                    require((r['araddr']%4096)+(r['arlen']+1)*(1<<r['arsize'])<=4096,'read 4K boundary')
            if case<5:
                require(not errors,f'legal traffic reported violation {case}: {errors}')
                require(r['faults']==0,f'false RTL fault {case}/{r["cycle"]}')
                total_reads=sum(map(len,reads.values()))
                require((r['ap'],r['wp'],r['rpnd'],r['partial'])==(outstanding_aw,outstanding_w,total_reads,len(partial)),f'inventory {case}/{r["cycle"]}')
                any_valid=any(r[k] for k in ['awv','wv','bv','arv','rv'])
                if not r['req']:captured=None;quiet=0
                elif captured!=r['tag']:captured=r['tag'];quiet=0
                elif not r['fence'] or not before_empty or any_valid:quiet=0
                else:quiet=min(2,quiet+1)
                empty=not(outstanding_aw or outstanding_w or total_reads or partial)
                ack=bool(r['req'] and captured==r['tag'] and r['fence'] and empty and not any_valid and quiet==2)
                require(r['ack']==ack,f'ACK temporal contract {case}/{r["cycle"]}')
                if r['ack']:
                    require(r['acktag']==r['tag'],'ACK identity');ack_tags.add(r['tag'])
            else:
                require(not r['ack'],'fault scenario falsely certified')
                require(r['faults'] in {0,expected[case]},f'unexpected fault source {case}/{r["cycle"]}')
                if r['faults']:
                    if fault_at is None:
                        fault_at=r['cycle'];frozen=tuple(r[x] for x in ['ap','wp','rpnd','partial'])
                    require(tuple(r[x] for x in ['ap','wp','rpnd','partial'])==frozen,'corrupted inventory not retained')
                elif fault_at is not None:require(False,'fault cleared')
            for name in ['ap','wp','rpnd','partial']:peaks[name]=max(peaks[name],r[name])
            last=r
        if case<5:
            require(not(outstanding_aw or outstanding_w or partial or any(reads.values()) or any(writes.values())),'not fully drained')
            require(len(ack_tags)==2,'two distinct fresh identities')
        else:
            matching=[cycle for cycle,bit in errors if bit==expected[case]]
            require(matching and fault_at is not None,'missing independent violation')
            require(min(matching)<=fault_at<=min(matching)+1,'fault recognition latency')
            require(all(r['faults']==expected[case] for r in rows[-8:]),'fault not sticky through fresh identity')
        summaries.append(dict(case=case,cycles=len(rows),handshakes=dict(handshakes),stalled_observations=dict(stalls),peaks=dict(peaks),fault=expected[case],fault_cycle=fault_at))
    require(lengths=={1,2,4,16,64,256},'length coverage')
    require(ids==set(range(16)),'full four-bit ID coverage')
    require(response_codes=={0,1,2,3} and high_addresses>70,'response/high-address coverage')
    require(summaries[2]['peaks']['rpnd']==8 and summaries[1]['peaks']['ap']==8,'capacity boundaries')
    return dict(status='PASS_TWENTY_TWO_AXI4_DRAIN_SCENARIOS',cases=summaries,write_lengths=sorted(lengths),ids=sorted(ids),high_write_addresses=high_addresses,
                scope='Isolated passive observer, independent transaction/payload reconstruction. No AMD CDC, active fence, internal XDMA queues, reset or physical qualification.')


if __name__=='__main__':
    from pathlib import Path
    print(json.dumps(review(Path(sys.argv[1]).read_text()),indent=2))
