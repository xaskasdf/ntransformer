"""Reconcile every cell/net increment on the retained worst setup path; no new Vivado run."""
from collections import Counter,defaultdict
from decimal import Decimal
import json
from pathlib import Path
import re

def analyze(text):
    block=text.split('Max Delay Paths',1)[1].split('Min Delay Paths',1)[0]
    get=lambda pattern:re.search(pattern,block).group(1)
    source=get(r'Source:\s+(\S+)');destination=get(r'Destination:\s+(\S+)')
    summary=re.search(r'Data Path Delay:\s+([\d.]+)ns\s+\(logic ([\d.]+)ns .*?route ([\d.]+)ns',block)
    total,logic,nets=map(Decimal,summary.groups());levels=int(get(r'Logic Levels:\s+(\d+)'))
    body=block[block.index('FDRE (Prop_FDRE_C_Q)'):].split('\n  -------------------------------------------------------------------',1)[0]
    rows=[];pending=''
    for line in body.splitlines():
        match=re.match(r'^\s*(?:(.*?)\s+)?(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(?:[rf]\s+)?(\S+)\s*$',line)
        if match:
            desc,incr,path,resource=match.groups();desc=desc or pending;pending=''
            kind='net' if desc.startswith('net ') else desc.split()[0]
            assert kind in ('FDRE','CARRY8','LUT2','LUT4','LUT5','LUT6','net'),desc
            rows.append(dict(kind=kind,increment_ns=incr,cumulative_ns=path,resource=resource,description=desc))
        elif 'CARRY8 (Prop_' in line:pending=line.strip()
    assert sum(Decimal(r['increment_ns']) for r in rows)==total
    assert sum(Decimal(r['increment_ns']) for r in rows if r['kind']=='net')==nets
    assert sum(Decimal(r['increment_ns']) for r in rows if r['kind']!='net')==logic
    counts=Counter(r['kind'] for r in rows if r['kind'] not in ('net','FDRE'));assert sum(counts.values())==levels
    hierarchy=defaultdict(lambda:dict(logic_ns=Decimal(0),estimated_net_ns=Decimal(0),rows=0))
    for row in rows:
        resource=row['resource'];owner=resource.rsplit('/',1)[0] if '/' in resource else '<top>'
        group=hierarchy[owner];group['estimated_net_ns' if row['kind']=='net' else 'logic_ns']+=Decimal(row['increment_ns']);group['rows']+=1
    return dict(source=source,destination=destination,setup_slack_ns=get(r'Slack \(VIOLATED\) :\s+(-?[\d.]+)ns'),
        clock_requirement_ns=get(r'Requirement:\s+([\d.]+)ns'),data_path_ns=str(total),logic_ns=str(logic),
        estimated_net_ns=str(nets),logic_levels=levels,cells=dict(counts),
        hierarchy={k:{n:str(v) if isinstance(v,Decimal) else v for n,v in r.items()} for k,r in hierarchy.items()},
        entries=rows,physical_route_measured=False,
        scope='One worst path from completed OOC synthesis. Unplaced net delays are estimates. Registering cancellation is a candidate; improvement must be measured in a separate synthesis and routing.')

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('report',type=Path);a=p.parse_args()
    print(json.dumps(analyze(a.report.read_text()),indent=2))
