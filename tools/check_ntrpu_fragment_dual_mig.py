"""Independent full64 bridge ownership and two-MIG byte/readback ledger."""
import re
from collections import Counter
from check_ntrpu_fragment_pool_transport import fields,require,SCHEMA as BASE_SCHEMA

SCHEMA={k:v for k,v in BASE_SCHEMA.items() if k not in ['COLD','LOST']}
SCHEMA.update(BEGIN='first jobs',PASS='arms lines commands data cpl done retires aw w b reads one_ready end_ps',
    CLOCK='ps channel period phase',AR='ps channel index identity address',R='ps channel index identity data status id last')
MASK=(1<<64)-1

def address(job):
    if job==13:return 0x100001040
    if job==14:return 0x100001080
    if job>=12:return 0x1fffffffc0
    return [0,0x100000040,0x800000000,0x1000000000,0x1800000000,0x1fffffffc0][job//2]

def pattern(job):
    return int.from_bytes(bytes((job*37+i*23)&255 for i in range(64)),'little')

def review(text):
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text),'simulator/model failure')
    records=[]
    for line in text.splitlines():
        if not line.startswith('FRAG_DUAL_'):continue
        kind=line.split()[0][len('FRAG_DUAL_'):];f=fields(line)
        require(kind in SCHEMA and set(f)==set(SCHEMA[kind].split()),'exact trace schema')
        records.append((kind,f))
    groups={k:[f for n,f in records if n==k] for k in SCHEMA}
    require(len(groups['BEGIN'])==len(groups['PASS'])==1 and groups['BEGIN'][0]==dict(first='00000000ffffffff',jobs='17'),'one fixture header/summary')
    end=groups['PASS'][0]
    require([int(f['job']) for f in groups['JOB']]==list(range(17)),'ordered job inventory')
    require(len(groups['CLOCK'])==2,'two clock witnesses')
    clocks={int(f['channel']):f for f in groups['CLOCK']}
    require(set(clocks)=={0,1} and all(int(f['period'])==5000 and int(f['ps'])%5000==int(f['phase']) for f in clocks.values()),'actual UI clocks')
    require((int(clocks[1]['phase'])-int(clocks[0]['phase']))%5000==731,'independent UI phase')
    # CLOCK observers and source monitors may be scheduled in either order at
    # a shared simulation timestamp; only nondecreasing time is required.
    timed=[int(f['ps']) for k,f in records if 'ps' in f]
    require(timed==sorted(timed),'event chronology')
    for k in ['AW','W','B','AR','R']:
        for f in groups[k]:
            ch=int(f['channel']);require(ch in clocks and int(f['ps'])%5000==int(clocks[ch]['phase']),'destination UI edge '+k)
    counts=dict(arms=17,lines=16,commands=15,data=15,cpl=15,done=16,retires=17,aw=15,w=15,b=15,reads=28)
    mapping=dict(arms='ARM',lines='LINE',commands='CMD',data='DATA',cpl='CPL',done='DONE',retires='RETIRE',aw='AW',w='W',b='B',reads='R')
    require(all(int(end[k])==v==len(groups[mapping[k]]) for k,v in counts.items()),'global conservation')
    require(int(end['one_ready'])>500 and len(groups['AR'])==28,'cold barrier and read coverage')
    source=groups['SRC'];require(source,'source samples')
    require([int(f['ps']) for f in source]==list(range(int(source[0]['ps']),int(source[-1]['ps'])+1,5000)),'complete active source trace')
    require(all(int(f['cycle'])==(int(f['ps'])-2500)//5000+1 for f in source),'absolute source cycle')
    require(all(re.fullmatch(r'[0-9a-f]+',v) for f in source for k,v in f.items() if k!='job'),'known source ports')
    require(all(f['invalid']==f['poolfault']=='0' for f in source),'one live cold session')
    jobs={};cookie=0xffffffff;metrics=[]
    for job in range(17):
        e={k:[f for f in values if f.get('job')==str(job)] for k,values in groups.items() if k not in ['SRC','BEGIN','PASS','CLOCK','AR','R']}
        accepted=job not in [12,16]
        expected=dict(JOB=1,ARM=1,LINE=int(job!=12),CMD=int(accepted),DATA=int(accepted),AW=int(accepted),W=int(accepted),B=int(accepted),CPL=int(accepted),DONE=int(job!=12),RETIRE=1)
        require(all(len(e[k])==n for k,n in expected.items()),'per-job conservation '+str(job))
        e={k:v[0] for k,v in e.items() if v};t=lambda k:int(e[k]['ps'])
        require(t('JOB')<t('ARM') and t('ARM')>max(int(f['ps']) for f in clocks.values()),'all-ready ARM')
        epoch=0x100000001+job;tag=0x800000010000abcd;ch=job%2
        for k in ['ARM','LINE','DONE','RETIRE']:
            if k in e:
                le=epoch^(0x100000000 if k=='LINE' and job==16 else 0)
                require(int(e[k]['epoch'],16)==le and int(e[k]['tag'],16)==tag,'full64 original identity '+k)
        for k in ['ARM','LINE','CMD','AW']:
            if k in e:require(int(e[k]['address'],16)==address(job),'full address '+k)
        for k in ['ARM','CMD','AW','W','B','CPL']:
            if k in e:require(int(e[k]['channel'])==ch,'channel ownership '+k)
        for k in ['LINE','DATA','W']:
            if k in e:require(int(e[k]['data'],16)==pattern(job),'independent bytes '+k)
        status=1 if job in [13,14,15] else 2 if job==16 else 0
        if accepted:
            require(t('ARM')<t('LINE')<t('CMD')<t('DATA')<t('W')<t('B')<t('CPL')<t('DONE')<t('RETIRE') and t('CMD')<t('AW')<t('B'),'write/result causality')
            require(int(e['CMD']['cookie'],16)==int(e['CPL']['cookie'],16)==cookie,'cookie provenance')
            require(int(e['CPL']['status'],16)==int(e['B']['status'],16)==int(e['B']['id'],16)==0,'MIG response')
            require(all(int(e['AW'][k],16)==v for k,v in dict(len=0,id=0,size=6,burst=1,cache=0).items()),'AW attributes')
            require(int(e['W']['strb'],16)==MASK and e['W']['last']=='1','one complete beat')
            jobs[job]=dict(channel=ch,address=address(job),cookie=cookie,b_ps=t('B'),payload=pattern(job));cookie+=1
            metrics.append(dict(job=job,channel=ch,line_to_b_ps=t('B')-t('LINE'),b_to_cpl_ps=t('CPL')-t('B'),cpl_to_done_handshake_ps=t('DONE')-t('CPL'),line_to_retirement_ps=t('RETIRE')-t('LINE')))
        if 'DONE' in e:require(int(e['DONE']['status'],16)==status and int(e['DONE']['raw'],16)==0,'result mapping')
        if job==12:require(t('ARM')<t('RETIRE'),'partial STOP retirement')
        if job==16:require(t('LINE')<t('DONE')<t('RETIRE'),'local rejection retirement')
        rows=[f for f in source if f['job']==str(job)]
        for f in rows:
            tm=int(f['ps']);require(int(f['held'])==int(t('ARM')<tm<=t('RETIRE')),'source reservation lifetime')
            if f['cv']=='1':require(accepted and int(f['ct'],16)==cookie-1,'held command cookie')
            if f['ov']=='1':
                require('DONE' in e and int(f['os'],16)==status,'held DONE')
                if accepted:require(t('CPL')<tm and f['drained']=='1','completion/drain before publication')
        for valid,ready,k in [('cv','cr','CMD'),('dv','dr','DATA'),('ov','ore','DONE')]:
            require([int(f['ps']) for f in rows if f[valid]==f[ready]=='1']==([t(k)] if k in e else []),'sampled '+k)
        completions=[]
        for f in rows:
            require(re.fullmatch(r'[01]{2}',f['pv']) and re.fullmatch(r'[01]{2}',f['pr']),'completion width')
            for channel in range(2):
                if (int(f['pv'],2)&int(f['pr'],2))>>channel&1:
                    completions.append(int(f['ps']))
                    require(accepted and channel==ch and (int(f['pt'],16)>>(64*channel)&MASK)==cookie-1 and (int(f['psr'],16)>>(2*channel)&3)==0,'sampled completion identity')
        require(completions==([t('CPL')] if accepted else []),'sampled completion count')
    # A separate destination ledger reconstructs memory from B-completed writes.
    golden=[{},{}];pending=[None,None];read_order=[[],[]];readbacks=[]
    for kind,f in records:
        if kind=='B':
            job=int(f['job']);j=jobs[job];golden[j['channel']][j['address']]=dict(identity=job,data=j['payload'],b=int(f['ps']))
        if kind=='AR':
            ch=int(f['channel']);a=int(f['address'],16);identity=int(f['identity']);index=int(f['index'])
            require(pending[ch] is None and a in golden[ch] and a==address(identity),'read issued to independently written location')
            g=golden[ch][a];require(g['identity']==identity and g['b']<int(f['ps']) and index==len(read_order[ch])+1,'readback version/index')
            pending[ch]=dict(address=a,identity=identity,index=index,ar=int(f['ps']),data=g['data'])
        if kind=='R':
            ch=int(f['channel']);p=pending[ch];require(p is not None,'response without read')
            require(int(f['identity'])==p['identity'] and int(f['index'])==p['index'] and int(f['data'],16)==p['data'] and int(f['status'],16)==int(f['id'],16)==0 and f['last']=='1','readback identity/bytes/status')
            require(p['ar']<int(f['ps']),'read latency')
            read_order[ch].append(p['identity']);readbacks.append(dict(channel=ch,identity=p['identity'],ar_to_r_ps=int(f['ps'])-p['ar']));pending[ch]=None
    require(read_order==[[0,2,4,6,8,10,10,0,2,4,6,8,10,14],[1,3,5,7,9,11,11,1,3,5,7,9,15,13]],'readback coverage including STOP/reject preservation and overwrite')
    require(pending==[None,None] and [len(g) for g in golden]==[7,7],'all destinations drained')
    require(int(end['end_ps'])>=max(timed) and source[-1]['drained']=='1' and all(int(source[-1][k],16)==0 for k in ['held','banks','pending','cv','dv','pv','ov']),'terminal drain')
    return dict(status='PASS_FRAGMENT_ACTUAL_DUAL_MIG_BYTES',qualified=True,counts=counts,write_bytes=960,read_bytes=1792,source_samples=len(source),clocks=clocks,latencies=metrics,readback_latencies=readbacks,
        scope='Single exclusive lease through actual pool, two AMD CDCs, cold supervisor and two generated MIG/RDIMM generic models. No native PCIe collector, exact commercial DIMM, warm DDR reset, routed implementation, board performance or inference qualification.')

# Reuse the independently qualified two-controller periodic lifecycle method;
# this fixture changes only the exact user read/write inventory to14 and7/8.
from decimal import Decimal
from check_ntrpu_multibuffer_periodic import review_periodic
from check_ntrpu_pcie_ddr_periodic import WARNING,WARNING_DELAY,BEAT_PS,records
need=require

def periodic(text):
    parts = [[],[]]; candidates = {}
    for ch,record in re.findall(r'^DUAL_PER ([01]) (.*)$',text,re.M): parts[int(ch)].append(record)
    need(len(re.findall(r'^DUAL_PER ',text,re.M))==sum(map(len,parts)),'Known periodic port tags')
    for ch,lines in enumerate(parts):
        for pin in records('\n'.join(lines),'NTRPU_DDR_PIN_CMD', ['ps','slot','rank','act_n','cke'], ['cmd','adr','bg','bank']):
            if pin['act_n']==1 and (pin['adr']>>14)&7==5:
                for beat in range(8):candidates.setdefault(pin['ps']+WARNING_DELAY+beat*BEAT_PS,set()).add(ch)
    warning_counts = [0,0]
    for line in text.splitlines():
        if re.search(r':\s*WARNING\s*:',line):
            match=WARNING.fullmatch(line);need(match is not None,'Unknown model warning')
            owners=candidates.get(int(match[7]),set());need(len(owners)==1,'Unique physical owner of model warning')
            ch=next(iter(owners));parts[ch].append(line);warning_counts[ch]+=1
    finish=re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    times={int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in finish}
    need(len(times)==1,'One global finish time');finish_ps=times.pop()
    result=[]
    for ch,lines in enumerate(parts):
        local='\n'.join(lines); witness=review_periodic(local)
        rows=records(local,'NTRPU_DDR_PERIODIC_DRAIN',['begin_ps','end_ps','falling_edges','requests','returns'])
        need(len(rows)==1,'One final periodic drain per controller');d=rows[0]
        phase=(witness['configuration']['ps']+2500)%5000
        first=d['begin_ps']+(phase-d['begin_ps'])%5000
        if first==d['begin_ps']:first+=5000
        need(0<d['falling_edges']<200 and d['end_ps']==first+5000*(d['falling_edges']-1),'Consecutive local drain edges')
        need(0<=finish_ps-d['end_ps']<1000000,'Global finish within bounded peer drain window')
        counts=witness['counters']
        need(d['requests']==d['returns']==counts['requests']==counts['accepts']==counts['returns']==counts['periodic_cas'],'Drain counters')
        returns=records(local,'NTRPU_DDR_READ_RETURN',['ps'],['phy','injected','rmw','native','periodic_done'])
        need(returns[-1]['ps']<d['end_ps'] and all(x['return_ps']<d['end_ps'] for x in witness['lifecycle']),'All returns before local drain')
        need(counts['user_cas']==14 and witness['physical_writes']==[7,8][ch],'Per-controller physical transfers')
        result.append(dict(port=ch,witness=witness,drain=d))
    return dict(status='DUAL_PERIODIC_WITNESS_PASS',controllers=result,warning_counts=warning_counts,finish_ps=finish_ps)


def rank_witness(text):
    acts=records(text,'DUAL_ACT',['port','ps','rank','count'])
    counters=Counter();actual=[];expected=[]
    for a in acts:
        key=(a['port'],a['rank']);require(a['port'] in [0,1] and a['rank'] in range(4),'physical rank')
        counters[key]+=1;require(a['count']==counters[key],'monotonic rank activation counter')
        actual.append((a['port'],a['ps'],a['rank']))
    for ch in [0,1]:
        local='\n'.join(line.split(' ',2)[2] for line in text.splitlines() if line.startswith(f'DUAL_PER {ch} '))
        pins=records(local,'NTRPU_DDR_PIN_CMD',['ps','slot','rank','act_n','cke'],['cmd','adr','bg','bank'])
        expected.extend((ch,p['ps'],p['rank']) for p in pins if p['act_n']==0)
    require(sorted(actual)==sorted(expected),'rank counters match physical ACT pins')
    require(set(counters)=={(ch,rank) for ch in [0,1] for rank in range(4)},'four ranks in each controller')
    return {str(ch):{str(rank):counters[(ch,rank)] for rank in range(4)} for ch in [0,1]}
