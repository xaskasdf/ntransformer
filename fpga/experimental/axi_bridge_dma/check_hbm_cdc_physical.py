"""Negative controls for physical CDC path, coverage and exception review."""
import argparse
import copy
import json
from pathlib import Path
import re
from review_hbm_cdc_physical import paths_review, xdc_review, table, report_review, comparable_xdc, require


def check(root):
    audit=root/'output/audit'
    paths=table(audit/'paths.tsv');stages=table(audit/'interstage.tsv')
    xdc=(audit/'effective.xdc').read_text()
    paths_review(paths,stages);xdc_review(xdc)
    results=[]

    def reject(name, action):
        try:
            action()
        except ValueError as exc:
            results.append(dict(name=name,rejected=True,reason=str(exc)))
        else:
            raise ValueError('Invalid evidence accepted: '+name)

    def alter(rows,index,key,value):
        changed=copy.deepcopy(rows);changed[index][key]=value
        return changed

    reject('missing CDC endpoint',lambda:paths_review(paths[:-1],stages))
    reject('duplicated CDC endpoint',lambda:paths_review(paths[:-1]+[paths[0]],stages))
    reject('wrong launch clock',lambda:paths_review(alter(paths,0,'start_clock','mgmt_clk'),stages))
    reject('relaxed bound',lambda:paths_review(alter(paths,0,'requirement_ns','5.0'),stages))
    reject('excessive data delay',lambda:paths_review(alter(paths,0,'datapath_ns','4.01'),stages))
    reject('negative CDC slack',lambda:paths_review(alter(paths,0,'slack_ns','-0.001'),stages))
    reject('wrong RAM driver',lambda:paths_review(alter(paths,next(i for i,r in enumerate(paths) if r['kind']=='capture'),'from','unrelated/C'),stages))
    reject('missing synchronizer path',lambda:paths_review(paths,stages[:-1]))
    reject('negative synchronizer hold',lambda:paths_review(paths,alter(stages,next(i for i,r in enumerate(stages) if r['type']=='min'),'slack_ns','-0.001')))
    reject('unplaced synchronizer',lambda:paths_review(paths,alter(stages,0,'dest_loc','')))
    reject('NaN timing',lambda:paths_review(alter(paths,0,'datapath_ns','nan'),stages))
    reject('broad asynchronous cut',lambda:xdc_review(xdc+'\nset_clock_groups -asynchronous -group pcie_clk -group reader_clk\n'))
    reject('false path override',lambda:xdc_review(xdc+'\nset_false_path -from [get_clocks pcie_clk]\n'))
    reject('missing skew constraint',lambda:xdc_review(re.sub(r'^set_bus_skew.*\n','',xdc,count=1,flags=re.M)))
    reject('relaxed skew bound',lambda:xdc_review(re.sub(r'(^set_bus_skew.*)2\.000\s*$',r'\g<1>5.000',xdc,count=1,flags=re.M)))
    route=(audit/'route_status.rpt').read_text();skew=(audit/'bus_skew.rpt').read_text();timing=(audit/'timing.rpt').read_text()
    report_review(route,skew,timing)
    reject('routing error',lambda:report_review(re.sub(r'(nets with routing errors\.+\s*:)\s*0',r'\g<1> 1',route),skew,timing))
    reject('extra route gap',lambda:report_review(re.sub(r'(nets with explicit gaps\.+\s*:)\s*3',r'\g<1> 4',route),skew,timing))
    reject('skew report relaxed',lambda:report_review(route,re.sub(r'(Slow\s+)2\.000',r'\g<1>5.000',skew,count=1),timing))
    reject('skew group missing',lambda:report_review(route,re.sub(r'^\s*Slow\s+2\.000[^\n]*\n','',skew,count=1,flags=re.M),timing))
    section=timing.split('| Design Timing Summary')[1].split('| Clock Summary')[0]
    bad_section=re.sub(r'^(\s*)\d+\.\d+(\s+0\.000\s+0\s+\d+)',r'\g<1>-0.001\g<2>',section,count=1,flags=re.M)
    reject('negative global setup',lambda:report_review(route,skew,timing.replace(section,bad_section,1)))
    if (root/'output/before.xdc').exists():
        before=(root/'output/before.xdc').read_text();after=(root/'output/after.xdc').read_text()
        require(comparable_xdc(before)==comparable_xdc(after),'Read-only baseline mismatch')
        changed=after.replace('4.000','4.100',1)
        reject('constraints changed on reopen',lambda:require(comparable_xdc(before)==comparable_xdc(changed),'Effective constraints changed'))
    return dict(status='HBM_CDC_PHYSICAL_NEGATIVES_PASS',rejected=len(results),controls=results)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('phase',type=Path)
    print(json.dumps(check(parser.parse_args().phase),indent=2))
