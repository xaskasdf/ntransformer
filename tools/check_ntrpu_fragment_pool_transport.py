"""Independent event conservation, bytes and causality for real pool/CDC runs.

The testbench separately checks every exposed AW/W payload during stalls.
This oracle checks recorded source handshakes against destination events;
it does not use the DUT phase encoding to infer transaction ownership.
"""
import re
from collections import Counter

def require(ok,message):
    if not ok:raise ValueError(message)

def fields(line):
    pairs=re.findall(r'(\w+)=([^\s]+)',line)
    require(len(pairs)==len(dict(pairs)),'duplicate fields')
    return dict(pairs)

SCHEMA={
    'BEGIN':'profile half0 half1','COLD':'ps n','JOB':'ps job',
    'ARM':'ps job epoch tag channel address','LINE':'ps job epoch tag address data',
    'CMD':'ps job channel address cookie','DATA':'ps job data',
    'AW':'ps job channel address len id size burst cache','W':'ps job channel data strb last',
    'B':'ps job channel id status','CPL':'ps job channel cookie status',
    'DONE':'ps job epoch tag status raw','RETIRE':'ps job epoch tag',
    'LOST':'ps job held invalid cplvalid cplready drained',
    'SRC':'ps cycle job stop held invalid poolfault drained phase banks pending cv cr ct dv dr pv pr pt psr ov ore os',
    'PASS':'profile arms lines commands data cpl done retires aw w b cold awstall0 awstall1 wstall0 wstall1 end_ps',
}

def review(text,profile):
    require(not re.search(r'^(Fatal:|ERROR:|CRITICAL WARNING:)',text,re.M),'simulation failure')
    records=[]
    for line in text.splitlines():
        if not line.startswith('POOL_XPORT_'):continue
        name=line.split()[0][len('POOL_XPORT_'):];f=fields(line)
        require(name in SCHEMA and set(f)==set(SCHEMA[name].split()),'exact record schema')
        records.append((name,f))
    kinds=Counter(n for n,f in records)
    require(all(kinds[k]==1 for k in ['BEGIN','PASS','LOST']),'one header/summary/session loss')
    grouped={k:[f for n,f in records if n==k] for k in SCHEMA}
    begin=grouped['BEGIN'][0];end=grouped['PASS'][0]
    require(begin==dict(profile=str(profile),half0=str(2000 if profile==0 else 4250),half1=str(3650 if profile==0 else 1750)),'clock profile')
    timed=[int(f['ps']) for n,f in records if 'ps' in f]
    require(timed==sorted(timed),'event chronology')
    require([int(f['job']) for f in grouped['JOB']]==list(range(13)),'all jobs exactly once')
    require([int(f['n']) for f in grouped['COLD']]==list(range(1,6)),'cold sessions')
    times=[int(f['ps']) for f in grouped['COLD']]
    source=grouped['SRC'];require(source,'missing source samples')
    expected_sample_times=[]
    for index,tm in enumerate(times):
        limit=times[index+1] if index+1<len(times) else int(end['end_ps'])
        expected_sample_times.extend(range(tm+1002500,limit,5000))
    require([int(f['ps']) for f in source]==expected_sample_times,'complete reset-delimited source trace')
    require(all(int(f['cycle'])==(int(f['ps'])-2500)//5000+1 for f in source),'absolute source cycle')
    require(all(int(a['cycle'])<int(b['cycle']) and int(b['ps'])-int(a['ps'])==5000*(int(b['cycle'])-int(a['cycle'])) for a,b in zip(source,source[1:])),'source clock spacing')
    require(all(re.fullmatch(r'[0-9a-f]+',v) for f in source for k,v in f.items() if k!='job'),'known source trace')
    for f in source:
        require(all(f[k] in ['0','1'] for k in 'stop held invalid poolfault drained cv cr dv dr ov ore'.split()),'Boolean widths')
        require(all(re.fullmatch(r'[01]{2}',f[k]) for k in ['pv','pr']),'completion width')
    expected_counts=dict(arms=13,lines=12,commands=11,data=11,cpl=10,done=11,retires=12,aw=11,w=11,b=11,cold=5)
    labels=dict(arms='ARM',lines='LINE',commands='CMD',data='DATA',cpl='CPL',done='DONE',retires='RETIRE',aw='AW',w='W',b='B',cold='COLD')
    require(int(end['profile'])==profile and all(int(end[k])==v==kinds[labels[k]] for k,v in expected_counts.items()),'global conservation')
    require(all(int(end[k])>0 for k in ['awstall0','wstall1']),'forced destination stalls')
    metrics=[];known_cookies=[];previous_session=-1;cookie=0xffffffff
    for job in range(13):
        events={k:[f for f in v if f.get('job')==str(job)] for k,v in grouped.items() if k not in ['SRC','BEGIN','PASS','COLD']}
        accepted=job not in [2,8];completed=accepted and job!=11
        expected={'JOB':1,'ARM':1,'LINE':int(job!=2),'CMD':int(accepted),'DATA':int(accepted),'AW':int(accepted),'W':int(accepted),'B':int(accepted),'CPL':int(completed),'DONE':int(job not in [2,11]),'RETIRE':int(job!=11),'LOST':int(job==11)}
        require(all(len(events[k])==v for k,v in expected.items()),f'per-job conservation {job}')
        e={k:v[0] for k,v in events.items() if v}
        t=lambda name:int(e[name]['ps'])
        require(t('JOB')<t('ARM'),'ARM fixture causality')
        session=sum(tm<t('ARM') for tm in times)-1
        require(session==0 if job<9 else session==job-8,'cold session boundary')
        if session!=previous_session:cookie=0xffffffff;previous_session=session
        epoch=0x100000001+job;tag=0x800000010000abcd;channel=job%2
        address=0x1fffffffc0 if channel else 0x100000040
        payload=int.from_bytes(bytes((job*37+i*23+profile*11)&255 for i in range(64)),'little')
        status=1 if job in range(3,7) else 2 if job==8 else 4 if job in [9,10] else 0
        raw=2 if job==9 else 3 if job==10 else 0
        for kind in ['ARM','LINE','DONE','RETIRE']:
            if kind in e:
                expected_epoch=epoch^(0x100000000 if kind=='LINE' and job==8 else 0)
                require(int(e[kind]['epoch'],16)==expected_epoch and int(e[kind]['tag'],16)==tag,'full64 identity '+kind)
        for kind in ['ARM','LINE','CMD','AW']:
            if kind in e:require(int(e[kind]['address'],16)==address,'full destination '+kind)
        for kind in ['ARM','CMD','AW','W','B','CPL']:
            if kind in e:require(int(e[kind]['channel'])==channel,'destination channel '+kind)
        for kind in ['LINE','DATA','W']:
            if kind in e:require(int(e[kind]['data'],16)==payload,'independent bytes '+kind)
        if accepted:
            require(t('ARM')<t('LINE')<t('CMD')<t('DATA')<t('W')<t('B') and t('CMD')<t('AW')<t('B'),'destination write causality')
            require(int(e['CMD']['cookie'],16)==cookie,'nonrepeating transport cookie')
            known_cookies.append(dict(job=job,cookie=cookie));cookie+=1
            require(all(int(e['AW'][k],16)==v for k,v in dict(len=0,id=0,size=6,burst=1,cache=0).items()),'AW attributes')
            require(int(e['W']['strb'],16)==(1<<64)-1 and e['W']['last']=='1','full single beat')
            require(int(e['B']['status'],16)==raw and int(e['B']['id'],16)==0,'response fixture')
        if completed:
            require(t('B')<t('CPL')<t('DONE')<t('RETIRE'),'completion before result/retirement')
            require(int(e['CPL']['cookie'],16)==cookie-1 and int(e['CPL']['status'],16)==raw,'full completion identity/status')
            metrics.append(dict(job=job,channel=channel,line_to_b_ps=t('B')-t('LINE'),b_to_cpl_ps=t('CPL')-t('B'),cpl_to_done_handshake_ps=t('DONE')-t('CPL'),line_to_retirement_ps=t('RETIRE')-t('LINE')))
        if 'DONE' in e:require(int(e['DONE']['status'],16)==status and int(e['DONE']['raw'],16)==raw,'result mapping')
        if job==2:require(t('ARM')<t('RETIRE'),'partial STOP retirement')
        if job==8:require(t('LINE')<t('DONE')<t('RETIRE'),'local error retirement')
        if job==11:
            require(t('B')<t('LOST') and all(e['LOST'][k]==v for k,v in dict(held='1',invalid='1',cplvalid='10',cplready='00',drained='0').items()),'lost session keeps pending ownership')
        rows=[f for f in source if f['job']==str(job)]
        for f in rows:
            tm=int(f['ps'])
            next_cold=times[session+1] if session+1<len(times) else int(end['end_ps'])+1
            require(int(f['held'])==int(t('ARM')<tm<next_cold and ('RETIRE' not in e or tm<=t('RETIRE'))),'source ownership interval')
            if f['cv']=='1':require(accepted and int(f['ct'],16)==cookie-1,'held command cookie')
            if f['ov']=='1':
                require('DONE' in e and int(f['os'],16)==status,'published DONE invariant')
                if completed:require(t('CPL')<tm and f['drained']=='1','publication after drain')
        for valid,ready,kind in [('cv','cr','CMD'),('dv','dr','DATA'),('ov','ore','DONE')]:
            actual=[int(f['ps']) for f in rows if f[valid]==f[ready]=='1']
            require(actual==([t(kind)] if kind in e else []),'source event matches samples '+kind)
        actual=[]
        for f in rows:
            for ch in range(2):
                if (int(f['pv'],2)&int(f['pr'],2))>>ch&1:
                    actual.append(int(f['ps']))
                    require(ch==channel and int(f['pt'],16)>>(64*ch)&((1<<64)-1)==cookie-1 and int(f['psr'],16)>>(2*ch)&3==raw,'sampled CPL context')
        require(actual==([t('CPL')] if completed else []),'sampled completion conservation')
    require(int(end['end_ps'])>max(timed),'terminal fixture ordering')
    require(all(int(source[-1][k],16)==0 for k in ['held','invalid','poolfault','banks','pending','cv','dv','pv','ov']),'terminal quiescence')
    require(source[-1]['drained']=='1','terminal drain')
    return dict(status='PASS_FRAGMENT_REAL_POOL_CDC_PROTOCOL',qualified=True,profile=profile,counts=expected_counts,
        bytes_written=11*64,source_samples=len(source),cookies=known_cookies,latencies=metrics,
        scope='Actual multibuffer/credits and two AMD AXI CDCs. Memory and supervisor are BFMs. One exclusive lease, no multiple-reservation throughput, real MIG, PCIe, warm reset or physical timing qualification.')
