"""CMD-only physical DDR witness: eight writes per channel, no job18 write.

Preserves the full native pin/CAS/warning/return/drain audit. Channel0 has
fifteen reads because its final forensic read returns old job16 bytes;
there is no ninth write. Other retained-DATA/CPL reviewers are unchanged.
"""
import re
from decimal import Decimal
from check_ntrpu_multibuffer_periodic import review_periodic
from check_ntrpu_pcie_ddr_periodic import WARNING, WARNING_DELAY, BEAT_PS, records
from check_ntrpu_retained_paths_mig import require as need

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
        need(counts['user_cas']==(15 if ch==0 else 14) and witness['physical_writes']==8,'CMD scenario requires exactly eight physical writes per controller')
        result.append(dict(port=ch,witness=witness,drain=d))
    return dict(status='DUAL_PERIODIC_WITNESS_PASS',controllers=result,warning_counts=warning_counts,finish_ps=finish_ps)

