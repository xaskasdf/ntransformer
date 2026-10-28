"""Periodic DDR witness for 18 healthy jobs plus one retained channel0 write.

Uses the existing independent pin/return lifecycle checker. The extra job adds
one physical write and one user read on channel0; model warnings remain checked.
"""
import re
from decimal import Decimal
from check_ntrpu_multibuffer_periodic import review_periodic
from check_ntrpu_pcie_ddr_periodic import WARNING,WARNING_DELAY,BEAT_PS,records
from check_ntrpu_pcie_fragment_pool import require


def review(text):
    parts=[[],[]];candidates={}
    for ch,record in re.findall(r'^DUAL_PER ([01]) (.*)$',text,re.M):parts[int(ch)].append(record)
    require(len(re.findall(r'^DUAL_PER ',text,re.M))==sum(map(len,parts)),'known periodic port tags')
    for ch,lines in enumerate(parts):
        for pin in records('\n'.join(lines),'NTRPU_DDR_PIN_CMD',['ps','slot','rank','act_n','cke'],['cmd','adr','bg','bank']):
            if pin['act_n']==1 and (pin['adr']>>14)&7==5:
                for beat in range(8):candidates.setdefault(pin['ps']+WARNING_DELAY+beat*BEAT_PS,set()).add(ch)
    warning_counts=[0,0]
    for line in text.splitlines():
        if re.search(r':\s*WARNING\s*:',line):
            match=WARNING.fullmatch(line);require(match is not None,'unknown model warning')
            owners=candidates.get(int(match[7]),set());require(len(owners)==1,'unique physical owner of model warning')
            ch=next(iter(owners));parts[ch].append(line);warning_counts[ch]+=1
    times={int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)}
    require(len(times)==1,'one global finish');finish_ps=times.pop();result=[]
    for ch,lines in enumerate(parts):
        local='\n'.join(lines);witness=review_periodic(local)
        rows=records(local,'NTRPU_DDR_PERIODIC_DRAIN',['begin_ps','end_ps','falling_edges','requests','returns'])
        require(len(rows)==1,'one final periodic drain per controller');d=rows[0]
        phase=(witness['configuration']['ps']+2500)%5000
        first=d['begin_ps']+(phase-d['begin_ps'])%5000
        if first==d['begin_ps']:first+=5000
        require(0<d['falling_edges']<200 and d['end_ps']==first+5000*(d['falling_edges']-1),'consecutive local drain edges')
        require(0<=finish_ps-d['end_ps']<1000000,'finish within bounded peer drain window')
        counts=witness['counters']
        require(d['requests']==d['returns']==counts['requests']==counts['accepts']==counts['returns']==counts['periodic_cas'],'periodic drain counters')
        returns=records(local,'NTRPU_DDR_READ_RETURN',['ps'],['phy','injected','rmw','native','periodic_done'])
        require(returns[-1]['ps']<d['end_ps'] and all(x['return_ps']<d['end_ps'] for x in witness['lifecycle']),'all periodic returns before drain')
        require(counts['user_cas']==[15,14][ch] and witness['physical_writes']==[9,8][ch],'all physical DDR reads and writes')
        result.append(dict(port=ch,witness=witness,drain=d))
    return dict(status='PASS_GUARDED_MIG_PERIODIC_DDR',controllers=result,warning_counts=warning_counts,finish_ps=finish_ps)
