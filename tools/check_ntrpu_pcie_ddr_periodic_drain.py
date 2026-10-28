"""Require an observed bounded final drain in addition to full DDR causality."""
from decimal import Decimal
import re

from check_ntrpu_pcie_ddr_periodic import records, require, review as review_periodic_bank


def review(text):
    result = review_periodic_bank(text)
    drain = records(text, 'NTRPU_DDR_PERIODIC_DRAIN',
                    ['begin_ps','end_ps','falling_edges','requests','returns'])
    require(len(drain) == 1, 'one final passive drain')
    d = drain[0]
    phase = (result['periodic']['configuration']['ps']+2500) % 5000
    first = d['begin_ps'] + (phase-d['begin_ps']) % 5000
    if first == d['begin_ps']:
        first += 5000
    require(0 < d['falling_edges'] < 200 and d['end_ps'] == first+5000*(d['falling_edges']-1),
            'bounded consecutive falling-edge wait')
    counts = result['periodic']['counters']
    require(d['requests'] == d['returns'] == counts['requests'] == counts['returns'] ==
            counts['accepts'] == counts['periodic_cas'], 'complete periodic drain counters')
    ends = re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)', text)
    require(ends, 'actual simulator finish timestamp')
    require({int(Decimal(value)*{'ps':1,'ns':1000,'us':1000000}[unit]) for value,unit in ends}
            == {d['end_ps']}, 'finish only at completed drain')
    returns = records(text, 'NTRPU_DDR_READ_RETURN', ['ps'], ['phy','injected','rmw','native','periodic_done'])
    require(returns[-1]['ps'] < d['end_ps'] and
            all(life['return_ps'] < d['end_ps'] for life in result['periodic']['lifecycle']),
            'all actual responses observed before finish')
    result['periodic_drain'] = d
    return result


def mutations(text):
    line = next(l for l in text.splitlines() if l.startswith('NTRPU_DDR_PERIODIC_DRAIN '))
    out = {'missing_drain':text.replace(line+'\n','',1),
           'duplicate_drain':text.replace(line,line+'\n'+line,1)}
    for key in ['begin_ps','end_ps','falling_edges','requests','returns']:
        token = re.search(key+r'=(\d+)',line)[0]
        value = int(token.split('=')[1])
        # Shift begin beyond its actual first falling edge, not merely within
        # the same clock half-period, which would still be a valid wait.
        step = 5000 if key == 'begin_ps' else 1
        out['drain_'+key] = text.replace(line,line.replace(token,key+'='+str(value+step),1),1)
    final = re.search(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    out['early_finish'] = text.replace(final[0],'$finish called at time : 1 ps')
    return out
