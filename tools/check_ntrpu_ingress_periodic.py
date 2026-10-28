"""Reuse the qualified passive MIG causality proof without an FFN fixture."""
from decimal import Decimal
import re
from check_ntrpu_pcie_ddr_periodic import records, require, review_periodic


def review(text):
    periodic=review_periodic(text)
    rows=records(text,'NTRPU_DDR_PERIODIC_DRAIN',['begin_ps','end_ps','falling_edges','requests','returns'])
    require(len(rows)==1,'one final periodic drain')
    d=rows[0];phase=(periodic['configuration']['ps']+2500)%5000
    first=d['begin_ps']+(phase-d['begin_ps'])%5000
    if first==d['begin_ps']:first+=5000
    require(0<d['falling_edges']<200 and d['end_ps']==first+5000*(d['falling_edges']-1),'bounded consecutive drain edges')
    counts=periodic['counters']
    require(d['requests']==d['returns']==counts['requests']==counts['accepts']==counts['returns']==counts['periodic_cas'],'periodic drain counters')
    ends=re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    require(ends and {int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in ends}=={d['end_ps']},'finish at completed drain')
    returns=records(text,'NTRPU_DDR_READ_RETURN',['ps'],['phy','injected','rmw','native','periodic_done'])
    require(returns[-1]['ps']<d['end_ps'] and all(life['return_ps']<d['end_ps'] for life in periodic['lifecycle']),'all observed periodic returns precede finish')
    return dict(periodic=periodic,drain=d)
