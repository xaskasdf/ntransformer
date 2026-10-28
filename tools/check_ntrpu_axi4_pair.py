"""Independent per-interface transaction queues and end-to-end CDC evidence."""
from collections import Counter,deque
import re
from check_ntrpu_axi4_drain import pattern,mask,require

LENS=[1,2,4,16,64,256,3]
FIELDS={'aw':['id','addr','len','size','prot','burst','cache','lock'],
        'w':['data','strb','last'],'b':['id','resp'],
        'ar':['id','addr','len','size','prot','burst','cache','lock'],
        'r':['data','id','resp','last']}
HEX={'awaddr','wdata','wstrb','araddr','rdata','gen','tag'}


def parse(line):
    fields=dict(x.split('=') for x in line.split()[1:])
    return {k:(v if k in {'side','name'} else None if re.search('[xzXZ]',v) else int(v,16 if k in HEX else 10)) for k,v in fields.items()}


def review(text,period):
    require(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|Error:|Warning:|WARNING:)',text,re.M),'runtime diagnostic')
    require(text.splitlines().count(f'AXI4_PAIR_SUITE_PASS device_period_ps={period} writes=24 reads=24 proofs=3 native_endpoint=0')==1,'suite marker')
    require(text.splitlines().count(f'AXI4_PAIR_PROFILE device_period_ps={period} host_period_ps=7000 mgmt_period_ps=10000 transactions=24 write_depth=32')==1,'profile')
    buses={side:[parse(x) for x in text.splitlines() if x.startswith('AXI4_CDC_BUS side='+side+' ')] for side in ['H','D']}
    events=[parse(x) for x in text.splitlines() if x.startswith('AXI4_PAIR_EVENT ')]
    reports=[e for e in events if e['name']=='FAULT_REPORT']
    events=[e for e in events if e['name']!='FAULT_REPORT']
    names=['START','FENCED_REQUEST','HELD_RESPONSES','STOP_DEVICE','DEVICE_WAIT','RESUME_DEVICE','PROOF_ONE',
           'STOP_HOST_NEW_TAG','HOST_WAIT','RESUME_HOST','PROOF_TWO','NEW_GENERATION','PROOF_THREE','INJECT_ORPHAN_R',
           'STOP_DEVICE_FAULT','DEVICE_FAULT_REVOKED','FAULT_FRESH_REQUEST','RESUME_DEVICE_FAULT','TERMINAL']
    require([e['name'] for e in events]==names,'event order')
    by={e['name']:e for e in events}
    require(all(a['t']<=b['t'] for a,b in zip(events,events[1:])),'monotonic events')
    stats={};transfers={};edges={}
    for side,rows in buses.items():
        require(rows and all(a['t']<b['t'] for a,b in zip(rows,rows[1:])), 'edge timestamps')
        edges[side]=[r['t'] for r in rows]
        half=3500 if side=='H' else period//2;offset=0 if side=='H' else 600
        require(all((t-offset)%half==0 for t in edges[side]),'clock edge grid')
        pauses=[(by[a]['t'],by[b]['t']) for s,a,b in [('D','STOP_DEVICE','RESUME_DEVICE'),
            ('H','STOP_HOST_NEW_TAG','RESUME_HOST'),('D','STOP_DEVICE_FAULT','RESUME_DEVICE_FAULT')] if s==side]
        for a,b in zip(edges[side],edges[side][1:]):
            if not any(a<=end and start<=b for start,end in pauses):
                require(b-a==2*half,'missing clock observation outside a documented stop')
        aw=deque();w=deque();partial=[];writes={i:deque() for i in range(16)};reads={i:deque() for i in range(16)}
        ap=wp=0;aseq=rseq=0;handshakes=Counter();stalls=Counter();observed={ch:[] for ch in FIELDS}
        max_pending=Counter();last=None;orphan_seen=False;orphan_accepts=0;first_w=None;first_aw=None
        for row in rows:
            require(all(row[ch+suffix] in {0,1} for ch in FIELDS for suffix in ['valid','ready']),'unknown handshake')
            require(row['faults'] in {0,4} and (row['phase']==3 or row['faults']==0),'unexpected local fault')
            require((row['ap'],row['wp'],row['ar'],row['partial'])==(ap,wp,sum(map(len,reads.values())),len(partial)),f'pre-edge ledger {side}/{row["t"]}')
            if last:
                if orphan_seen:require(row['faults']==4,'orphan fault not retained')
                for ch,fields in FIELDS.items():
                    if last[ch+'valid'] and not last[ch+'ready']:
                        require(row[ch+'valid'] and all(row[ch+f]==last[ch+f] for f in fields),'stalled payload changed')
            for ch,fields in FIELDS.items():
                if row[ch+'valid'] and not row[ch+'ready']:stalls[ch]+=1
                if row[ch+'valid'] and row[ch+'ready']:
                    handshakes[ch]+=1;observed[ch].append(tuple(row[ch+f] for f in fields))
            if row['bvalid']:
                require(writes[row['bid']],f'orphan B {side}/{row["t"]}')
                require(row['bresp']==writes[row['bid']][0]['seq']%4,'B response payload')
                if row['bready']:writes[row['bid']].popleft();ap-=1;wp-=1
            if row['rvalid']:
                q=reads[row['rid']]
                if not q:
                    require(row['phase']==3 and row['rid']==3 and row['rresp']==2 and row['rlast']==1 and row['rdata']==pattern(0xbad4c0de),'invalid orphan injection')
                    orphan_seen=True
                    if row['rready']:orphan_accepts+=1
                else:
                    desc=q[0]
                    require(row['rlast']==(desc['beat']==desc['len']-1),'R LAST association')
                    require(row['rdata']==pattern(desc['seq']*4096^(desc['beat']<<8)^row['rid']),'R payload')
                    require(row['rresp']==desc['beat']%4,'R response payload')
                    if row['rready']:
                        desc['beat']+=1
                        if row['rlast']:q.popleft()
            for ch in ['aw','ar']:
                if row[ch+'valid'] and row[ch+'ready']:
                    seq=aseq if ch=='aw' else rseq
                    expected=[seq%16,(0x100000000 if ch=='aw' else 0x200000000)+seq*4096,LENS[seq%7]-1,
                              4 if seq%7==5 else 6,seq%8,1,3+4*(seq%4),0]
                    require([row[ch+f] for f in FIELDS[ch]]==expected,'complete address attributes')
                    require((row[ch+'addr']%4096)+(row[ch+'len']+1)*(1<<row[ch+'size'])<=4096,'legal 4K burst')
                    if ch=='aw':
                        first_aw=row['t'] if first_aw is None else first_aw
                        aw.append(dict(seq=seq,id=seq%16,len=LENS[seq%7],size=expected[3]));aseq+=1;ap+=1
                    else:reads[seq%16].append(dict(seq=seq,len=LENS[seq%7],beat=0));rseq+=1
            if row['wvalid'] and row['wready']:
                first_w=row['t'] if first_w is None else first_w
                partial.append((row['wdata'],row['wstrb']))
                if row['wlast']:w.append(partial);partial=[];wp+=1
            while aw and w:
                desc=aw.popleft();beats=w.popleft()
                require(len(beats)==desc['len'],'W LAST association')
                for beat,(data,strb) in enumerate(beats):
                    require(data==pattern(desc['seq']*4096^(beat<<8)),'W payload')
                    require(strb==mask(beat,desc['size']),'W strobe')
                writes[desc['id']].append(desc)
            max_pending['aw']=max(max_pending['aw'],ap);max_pending['w']=max(max_pending['w'],wp)
            max_pending['ar']=max(max_pending['ar'],sum(map(len,reads.values())))
            last=row
        beats=sum(LENS[i%7] for i in range(24))
        require(dict(handshakes)==dict(aw=24,w=beats,b=24,ar=24,r=beats+1),'handshake totals')
        require(aseq==rseq==24 and not(ap or wp or partial or aw or w or any(reads.values()) or any(writes.values())),'terminal empty inventory')
        require(first_w<first_aw,'actual W before AW')
        require(orphan_accepts==1 and last['faults']==4,'one real orphan accepted and retained')
        transfers[side]=observed
        stats[side]=dict(samples=len(rows),handshakes=dict(handshakes),stalled_observations=dict(stalls),max_pending=dict(max_pending),first_w_ps=first_w,first_aw_ps=first_aw)
    require(transfers['H']==transfers['D'],'full end-to-end channel conservation')
    for side,start,end in [('D','STOP_DEVICE','RESUME_DEVICE'),('H','STOP_HOST_NEW_TAG','RESUME_HOST'),('D','STOP_DEVICE_FAULT','RESUME_DEVICE_FAULT')]:
        require(not any(by[start]['t']<t<by[end]['t'] for t in edges[side]),'clock continued while stopped')
    for name in ['HELD_RESPONSES','DEVICE_WAIT','HOST_WAIT','NEW_GENERATION','DEVICE_FAULT_REVOKED','FAULT_FRESH_REQUEST','TERMINAL']:
        require(by[name]['ack']==0,'forbidden certificate')
    proofs=[by[n] for n in ['PROOF_ONE','PROOF_TWO','PROOF_THREE']]
    require(len({(p['gen'],p['tag']) for p in proofs})==3,'three distinct identities')
    require(proofs[0]['gen']==proofs[1]['gen'] and proofs[0]['tag']!=proofs[1]['tag'],'tag transition')
    require(proofs[1]['tag']==proofs[2]['tag'] and proofs[1]['gen']!=proofs[2]['gen'],'generation transition')
    for p in proofs:
        require(p['ack']==p['hc']==p['dc']==1 and not p['fm'],'valid pair proof')
        require(all(p[k]==0 for k in ['hap','hwp','har','dap','dwp','dar']),'proof requires both external inventories empty')
        require(p['hbs']==24 and p['hrs']==p['rs']==beats,'complete responses before proof')
    require(all(e['fm']==0 for e in events[:13]),'premature fault')
    require(by['STOP_DEVICE_FAULT']['df']==4,'registered device fault')
    require(reports and reports[0]['fm']&256 and reports[0]['ack']==0,'first fault revokes cached proof')
    latency=reports[0]['t']-by['STOP_DEVICE_FAULT']['t']
    require(0<=latency<=30001,'fault crossing needs no extra source edge')
    terminal=by['TERMINAL'];require(terminal['fm']==260 and terminal['hf']==terminal['df']==4 and terminal['hb']==terminal['db']==0,'sticky fault and no new mailboxes')
    return dict(status='PASS_AXI4_REAL_CDC_PAIR',device_period_ps=period,profiles=stats,
                write_beats=beats,read_beats=beats,proofs=proofs,fault_reporting_latency_ps=latency,
                scope='Real AMD AXI4 CDC and XPM pair evidence. Passive external observers with explicit finite test producer fence; no active admission, response containment, protected XDMA inventory or endpoint-reset permission.')
