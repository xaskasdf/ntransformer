"""Audit physical changes, timing preservation and compact netlist validation."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from check_ntrpu_candidate_transport import require


def timing(path):
    text=path.read_text()
    found=re.search(r'^\s*WNS\(ns\)[^\n]*\n[^\n]*\n([^\n]+)',text,re.M)
    require(found and len(found[1].split())==12,'Timing summary missing')
    fields=list(map(float,found[1].split()))
    return dict(setup_ns=fields[0],tns_ns=fields[1],setup_failing_endpoints=int(fields[2]),
                hold_ns=fields[4],hold_failing_endpoints=int(fields[6]),pulse_ns=fields[8])


def cells(path):
    with path.open(newline='') as f:rows=list(csv.DictReader(f,delimiter='\t'))
    require(rows and all(set(r)=={'cell','ref','loc','bel','modifications'} for r in rows),'Primitive state schema')
    result={r['cell']:r for r in rows};require(len(result)==len(rows),'Duplicate primitive')
    return result


def audit(out):
    record=json.loads((out/'result.json').read_text())
    require(record['experiment']=='post_route_physopt','Wrong experiment')
    s=out/'suite';before=timing(s/'pre_timing.rpt');after=timing(s/'timing.rpt')
    require(before==timing(out/'inputs/timing.rpt'),'Input timing changed before optimization')
    budget=lambda p:[line.strip() for line in p.read_text().splitlines()
                     if re.match(r'^\s*(create_clock|set_clock_uncertainty)\s',line)]
    require(budget(out/'inputs/constraints_after.xdc')==budget(s/'constraints_before.xdc')==budget(s/'constraints_after.xdc'),'Timing budgets changed')
    opts=(s/'optimization.txt').read_text().splitlines()
    command='COMMAND phys_opt_design '+('-restruct_opt ' if record.get('restructure') else '')+'-critical_cell_opt -placement_opt -routing_opt -critical_pin_opt -hold_fix'
    require(opts==[command,
                   'RETIME 0','CLOCK_OPT 0','FORCED_REPLICATION 0','RTL_CHANGED 0'],'Optimization scope changed')
    a=cells(s/'cells_before.tsv');b=cells(s/'cells_after.tsv');shared=set(a)&set(b)
    delta=dict(added=sorted(set(b)-set(a)),removed=sorted(set(a)-set(b)),
        moved=sorted(n for n in shared if (a[n]['loc'],a[n]['bel'])!=(b[n]['loc'],b[n]['bel'])),
        modification_flags_changed=sorted(n for n in shared if a[n]['modifications']!=b[n]['modifications']),
        type_changed=sorted(n for n in shared if a[n]['ref']!=b[n]['ref']))
    require(record.get('restructure') or not delta['type_changed'],'Unexpected primitive type change; requires explicit review')
    require(Counter(r['ref'] for r in a.values())['BUFGCE']==Counter(r['ref'] for r in b.values())['BUFGCE']==1,'Clock buffer changed')
    require(all(not any(k in row['modifications'] for k in ('RETIM','INSERT_NEGEDGE','CLOCK_OPT')) for row in b.values()),'Unexpected sequential/clock optimization')
    return dict(before=before,after=after,setup_improvement_ns=round(after['setup_ns']-before['setup_ns'],3),
                before_primitives=len(a),after_primitives=len(b),counts={k:len(v) for k,v in delta.items()},changes=delta)


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args()
    subprocess.run([sys.executable,'-B',str(Path(__file__).with_name('check_ntrpu_overlap_physical.py')),
                    str(a.output),'--review',str(a.review)],check=True,capture_output=True,text=True)
    result=json.loads(a.review.read_text());record=json.loads((a.output/'result.json').read_text())
    require(record['experiment']=='post_route_physopt','Wrong experiment')
    if result['stage']=='route':result['physical_optimization']=audit(a.output)
    result['optimization_checker_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','timing_checks','drc','negative_controls','physical_optimization')}))
    if 'physical_optimization' in result:
        print(json.dumps({k:v for k,v in result['physical_optimization'].items() if k!='changes'}))


if __name__=='__main__':main()
