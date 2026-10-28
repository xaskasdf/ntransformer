"""Check ownership retained across actual cold-supervisor runtime pool resets."""
import re
from collections import Counter

def need(ok,message):
    if not ok:raise ValueError(message)

def fields(line):
    p=re.findall(r'(\w+)=([^\s]+)',line);need(len(p)==len(dict(p)),'duplicate field');return dict(p)

def review(text):
    need(not re.search(r'^(Fatal:|ERROR:|CRITICAL WARNING:)',text,re.M),'simulation failure')
    need(text.splitlines().count('LIFETIME_SUITE_PASS cases=6')==1,'one suite finish')
    results=[]
    for case in range(6):
        events=[]
        for l in text.splitlines():
            if l.startswith(f'LIFETIME case={case} POOL_XPORT_'):
                kind=l.split()[2].replace('POOL_XPORT_','');f=fields(l)
                events.append((kind,f))
        resets=[fields(l) for l in text.splitlines() if l.startswith(f'LIFETIME_RESET case={case} ')]
        faults=[fields(l) for l in text.splitlines() if l.startswith(f'LIFETIME_FAULT case={case} ')]
        passes=[fields(l) for l in text.splitlines() if l.startswith(f'LIFETIME_PASS case={case} ')]
        need(len(passes)==1 and len(faults)==int(case!=0),'case terminals/faults')
        p=passes[0];job=case%2;epoch=0x100000001+job;tag=0x800000010000abcd
        expected=dict(ARM=1,LINE=int(case!=1),CMD=int(case!=1),DATA=int(case!=1),
            AW=int(case!=1),W=int(case not in [1,5]),B=int(case in [0,3,4]),CPL=int(case in [0,3,4]),DONE=int(case in [0,4]),RETIRE=int(case==0))
        counts=Counter(k for k,f in events if k not in ['SRC','JOB'])
        need(all(counts[k]==v for k,v in expected.items()) and set(counts)<=set(expected),'per-case transfer conservation')
        e={k:f for k,f in events if k in expected};arm=int(e['ARM']['ps']);line=int(e['LINE']['ps']) if 'LINE' in e else None
        cookie=0x100000000 if line else 0xffffffff
        payload=int.from_bytes(bytes((job*37+i*23+case*11)&255 for i in range(64)),'little')
        for k,f in e.items():
            need(int(f['job'])==job,'fixture job')
            if 'epoch' in f:need(int(f['epoch'],16)==epoch and int(f['tag'],16)==tag,'full64 context')
            if 'channel' in f:need(int(f['channel'])==job,'destination channel')
            if 'address' in f:need(int(f['address'],16)==(0x1fffffffc0 if job else 0x100000040),'destination address')
            if k in ['LINE','DATA','W']:need(int(f['data'],16)==payload,'all payload bytes')
            if k in ['CMD','CPL']:need(int(f['cookie'],16)==0xffffffff,'transport cookie')
            if 'status' in f:need(int(f['status'],16)==0,'successful peer status')
        for first,second in [('ARM','LINE'),('LINE','CMD'),('CMD','DATA'),('DATA','W'),('CMD','AW'),('AW','B'),('W','B'),('B','CPL'),('CPL','DONE'),('DONE','RETIRE')]:
            if first in e and second in e:need(int(e[first]['ps'])<int(e[second]['ps']),'handshake causality')
        mapping=dict(arms='ARM',lines='LINE',commands='CMD',data='DATA',cpl='CPL',done='DONE',retires='RETIRE')
        need(all(int(p[k])==expected[v] for k,v in mapping.items()),'terminal counts')
        need(int(p['epoch'],16)==epoch and int(p['next'],16)==cookie and p['held']==p['invalid']==str(int(case!=0)),'terminal retained identity/cookie')
        need(resets and len({r['ps'] for r in resets})==len(resets),'unique reset samples')
        need(all(int(a['ps'])<int(b['ps']) for a,b in zip(resets,resets[1:])),'reset trace chronology')
        need(all(set(r)==set('case ps pool_reset bridge_reset admit restart synced_restart held epoch cookie invalid done status phase'.split()) for r in resets),'exact reset schema')
        need(all(re.fullmatch(r'[0-9a-f]+',v) for r in resets for v in r.values()),'known retained state')
        source=[f for k,f in events if k=='SRC'];need([r['ps'] for r in resets]==[r['ps'] for r in source],'paired source/reset samples')
        retire=int(e['RETIRE']['ps']) if 'RETIRE' in e else None
        for r,s in zip(resets,source):
            now=int(r['ps']);held=int(arm<now and (retire is None or now<=retire))
            need(r['bridge_reset']=='1' and int(r['held'])==held and r['held']==s['held'],'retained lease despite pool reset')
            need(int(r['epoch'],16)==(epoch if arm<now else 0),'epoch preservation')
            need(int(r['cookie'],16)==(0x100000000 if line is not None and line<now else 0xffffffff),'cookie preservation')
            need(r['invalid']==s['invalid'] and r['done']==s['ov'] and r['status']==s['os'],'paired output evidence')
            if case and now>int(faults[0]['ps']) and r['pool_reset']=='0':
                need(r['admit']=='0' and r['invalid']=='1' and held==1,'runtime reset closes but retains')
                need(r['done']==str(int(case==3)) and int(r['status'],16)==0,'no newly published success after reset')
        if case:
            need(int(faults[0]['epoch'],16)==epoch and int(faults[0]['cookie'],16)==cookie,'fault snapshot identity')
            raw=next(i for i,r in enumerate(resets) if r['restart']=='1')
            synced=next(i for i,r in enumerate(resets) if r['synced_restart']=='1')
            # Three destination stages plus up to one 10ns source-register
            # interval (two destination periods); paused source resumes at 3.
            need(3<=synced-raw<=5,'restart crosses actual multi-stage source synchronizer')
            need(all(r['synced_restart']=='1' for r in resets[synced:]),'restart remains sticky')
            final=resets[-1]
            need(final['pool_reset']==final['admit']=='0' and final['invalid']=='1','no warm reopening')
            need(int(p['ps'])-int(faults[0]['ps'])>=3750000,'restoration and rearm observation window')
        else:need(all(r['invalid']==r['restart']==r['synced_restart']=='0' for r in resets),'healthy case remains live')
        results.append(dict(case=case,handshakes=expected,reset_samples=len(resets),held=case!=0,next_cookie=cookie))
    need(len(re.findall(r'^LIFETIME_PASS ',text,re.M))==6 and len(re.findall(r'^LIFETIME_FAULT ',text,re.M))==5,'global fixture coverage')
    return dict(status='PASS_FRAGMENT_DUAL_LIFETIME',qualified=True,cases=results,
        scope='Actual cold supervisor, bridge, pool and CDC. Clock/calibration/memory responses are drivers. Lease, epoch, cookie and held DONE survive runtime pool resets; five fault/restoration cases plus healthy transfer. No actual MIG, reconfiguration, routed timing or metastability qualification.')
