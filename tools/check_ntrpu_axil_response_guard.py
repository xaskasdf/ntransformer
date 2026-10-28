"""Independent request-queue/payload audit for the isolated guard bank."""
import collections
import re


def review(text):
    def need(ok, message):
        if not ok:
            raise ValueError(message)
    need(not re.search(r'^\s*(?:Fatal:|Failure:|ERROR:|CRITICAL WARNING:|FAIL )',text,re.M),'simulation failure')
    need(text.splitlines().count('GUARD_SUITE_PASS cases=5 counter_width=8 native_endpoint=0')==1,'complete suite')
    need('$finish called at time' in text,'simulator completion')
    samples=collections.defaultdict(list)
    passes={}
    for line in text.splitlines():
        if not line.startswith(('GUARD_SAMPLE ','GUARD_CASE_PASS ')):
            continue
        row={k:int(v,16 if k=='rd' else 10) for k,v in re.findall(r'(\w+)=(\S+)',line)}
        case=row['case'];need(case in range(5),'case identity')
        if line.startswith('GUARD_SAMPLE '):
            samples[case].append(row)
        else:
            need(case not in passes,'unique terminal');passes[case]=row
    need(set(samples)==set(passes)==set(range(5)),'five cases')
    expected=[(0,0,0,0,34,34),(1,1,1,0,0,1),(2,0,0,1,1,0),
              (4,255,0,0,0,0),(3,1,1,1,0,0)]
    result=[]
    for case in range(5):
        q={k:collections.deque() for k in ['AW','W','AR']}
        quarantine={'B':False,'R':False}
        overflow=False;b=[];r=[];held_b=None;held_r=None;prior=0;coverage=set()
        for row in samples[case]:
            t=row['ps'];need(t%7000==3500 and (not prior or t-prior==7000),'continuous clock trace');prior=t
            need(row['fault']==(int(quarantine['B'])|int(quarantine['R'])<<1|int(overflow)<<2),'registered fault timing')
            need((row['ap'],row['wp'],row['rp'])==tuple(min(255,len(q[k])) for k in q),'outstanding independent request queues')
            wb=bool(q['AW'] and q['W']);rd=bool(q['AR'])
            need(row['hbv']==int(bool(row['bv']) and wb and not quarantine['B']),'B prior-credit and quarantine contract')
            need(row['hrv']==int(bool(row['rv']) and rd and not quarantine['R']),'R prior-credit and quarantine contract')
            need(row['br']*row['bv']==row['hbv']*row['hbr'] and row['rr']*row['rv']==row['hrv']*row['hrr'],'two-sided acceptance')
            if held_b is not None:
                need(row['hbv'] and row['bc']==held_b,'held B payload')
                if row['fault']&2:coverage.add('valid_B_survives_R_fault')
            if held_r is not None:
                need(row['hrv'] and (row['rd'],row['rc'])==held_r,'held R payload')
                if row['fault']&1:coverage.add('valid_R_survives_B_fault')
            held_b=row['bc'] if row['hbv'] and not row['hbr'] else None
            held_r=(row['rd'],row['rc']) if row['hrv'] and not row['hrr'] else None
            if row['bv'] and not wb:quarantine['B']=True
            if row['rv'] and not rd:quarantine['R']=True
            if quarantine['B'] and wb and row['bv']:coverage.add('B_not_reassociated')
            if quarantine['R'] and rd and row['rv']:coverage.add('R_not_reassociated')
            if row['hbv'] and row['hbr']:
                q['AW'].popleft();q['W'].popleft();b.append(row['bc'])
            if row['hrv'] and row['hrr']:
                q['AR'].popleft();r.append((row['rd'],row['rc']))
            for flag,key in [('aw','AW'),('w','W'),('ar','AR')]:
                if row[flag]:q[key].append(t)
            if any(len(x)>255 for x in q.values()):overflow=True
        end=passes[case]
        need(tuple(end[k] for k in ['fault','ap','wp','rp','b_count','r_count'])==expected[case],'terminal expectations')
        need(end['b_count']==len(b) and end['r_count']==len(r) and end['ps']==prior+3500,'observed terminal completions')
        if case==0:
            need(b==[n%4 for n in range(32)]+[3,3],'all B response codes conserved')
            need(r==[(0xface0000+n,(n+1)%4) for n in range(32)]+[(0xface001f,0)]*2,'all R payloads conserved')
        if case==1:
            need(coverage>={'valid_R_survives_B_fault','B_not_reassociated'} and r==[(0x12345678,3)],'cross-channel B fault coverage')
        if case==2:
            need(coverage>={'valid_B_survives_R_fault','R_not_reassociated'} and b==[3],'cross-channel R fault coverage')
        if case==4:
            need(coverage>={'B_not_reassociated','R_not_reassociated'},'both quarantine histories')
        result.append(dict(case=case,cycles=len(samples[case]),coverage=sorted(coverage),terminal=end))
    return dict(status='PASS_FIVE_RESPONSE_GUARD_SCENARIOS',cases=result,native_endpoint=False)
