"""Review closed board-startup, elaboration and placement evidence without accepting board timing."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

OPTIMIZED_SHA='65e2a7ae84397ad674e1691e1029c6410c625a034f685cf48246f5a052eb90a8'
STARTUP_SHA='3bb2b38e714871359d490c92cf4ae86209bc45d1348bf2480f243e059d6f7386'


def require(ok,message):
    if not ok:raise ValueError(message)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))


def report_timing(path):
    s=path.read_text();s=s[s.index('Design Timing Summary'):]
    v=next(l.split() for l in s.splitlines() if re.match(r'^\s*-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\s+',l))
    require(len(v)==12,'Timing format')
    return dict(wns_ns=float(v[0]),whs_ns=float(v[4]),wpws_ns=float(v[8]),
                setup_failing=int(v[2]),hold_failing=int(v[6]),pulse_failing=int(v[10]))


def review(root, mode):
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Closed inventory')
    for n,h in files.items():require(sha(root/n)==h,'Closed file '+n)
    r=json.loads((root/'RESULT.json').read_text())
    require(r['sources_unchanged'],'Sources changed')
    for n,h in r['sources'].items():require(sha(root/'source'/n)==h,'Source identity '+n)
    out=root/'output'
    value=dict(mode=mode,tool_status=r['status'],seconds=r['seconds'],files_verified=len(files),
               native_traffic_executed=False,physical_route_executed=False,board_qualified=False)
    if mode=='startup':
        require(r['status']=='BOARD_STARTUP_SIM_EXPORTED','Startup run failed')
        require(sha(root/'source/ntrpu_hbm_board_startup.sv')==STARTUP_SHA,'Selected startup revision')
        steps=r['steps'];require([s['name'] for s in steps]==['compile','elaborate','positive','mutation-compile','mutation-elaborate','mutation-run','synth'],'Stages')
        require(all(s['returncode']==0 and not s['timed_out'] for s in steps),'Stage exit/timeout')
        pos=(root/'positive.log').read_text();neg=(root/'mutation-run.log').read_text()
        matches=re.findall(r'BS CASE PASS hold=(\d+) stable=(\d+) checks=(\d+) groups=(\d+)',pos)
        cases=sorted(tuple(map(int,x)) for x in matches)
        require(cases==[(1,1,56,11),(4,3,81,12),(16,32,240,12),(17,9,177,12)],'Positive coverage')
        require(pos.count('BS PASS cases=4')==1 and not re.search(r'(?im)^\s*(fatal|error):',pos),'Positive result')
        require('Fatal: BS FAIL hold=1 stable=1 lock loss preserves epoch' in neg and 'BS PASS cases=4' not in neg,'Mutation detection')
        original=(root/'source/ntrpu_hbm_board_startup.sv').read_text()
        anchor='else if (!locked_sync[2]) lock_fault <= 1;'
        require(original.count(anchor)==1 and (root/'mutation/ntrpu_hbm_board_startup.sv').read_text()==original.replace(anchor,'else if (!locked_sync[2]) begin lock_fault <= 1; configuration_rst_n <= 0; end'),'Actual RTL mutation')
        refs=Counter(x['ref'] for x in table(out/'primitives.tsv'))
        require(refs['IBUFDS']==3 and refs['IBUF']==1 and refs['BUFGCE']==3 and refs['DIFFINBUF']==3,'Pad/buffer primitives')
        require(not any(n.startswith('LD') for n in refs),'Latch primitive')
        require((out/'STATUS.txt').read_text().strip()=='BOARD_FRONTEND_STANDALONE_SYNTHESIZED','Synthesis marker')
        require("/output/frontend.dcp' has been generated." in (root/'synth.log').read_text(),'Checkpoint export')
        value.update(checks=sum(x[2] for x in cases),scenario_groups=sum(x[3] for x in cases),cases=cases,
                     mutation_rejected=True,checkpoint_sha256=sha(out/'frontend.dcp'),primitives=dict(refs),
                     standalone_async_input_cdc_qualified=False,native_frontend_synthesized=False)
    elif mode=='elaborate':
        require(r['status']=='HBM_NATIVE_JOIN_EXPORTED' and r['returncode']==0 and not r['timed_out'],'Elaboration failed')
        require(sha(root/'source/ntrpu_hbm_board_startup.sv')==STARTUP_SHA,'Selected startup revision')
        require((out/'STATUS.txt').read_text().strip()=='HBM_NATIVE_JOIN_elaborate_EXPORTED','Elaboration marker')
        ports=table(out/'ports.tsv');names={x['port'] for x in ports}
        require(len(names)==75,'Board port bits')
        require({'mgmt_clk_p','mgmt_clk_n','hbm_ref0_clk_p','hbm_ref0_clk_n','hbm_ref1_clk_p','hbm_ref1_clk_n','supervisor_rst_n_pad'}<=names,'New physical ports')
        require(not names&{'mgmt_clk_in','hbm_ref0_clk_in','hbm_ref1_clk_in','configuration_rst_n','fabric_clock_reset','fabric_clock_locked'},'Ideal boundary remains')
        clocks=(out/'primary_clocks.xdc').read_text()
        for name in ('mgmt_clk_p','hbm_ref0_clk_p','hbm_ref1_clk_p'):require('[get_ports '+name+']' in clocks,'Clock boundary')
        instances=table(out/'native_instances.tsv');require(len(instances)==4,'Native blocks')
        value.update(port_bits=75,native_instances=instances,native_frontend_elaborated=True,native_frontend_synthesized=False)
    else:
        require(r['checkpoint_sha256']==OPTIMIZED_SHA,'Placement donor')
        log=(root/'tool.log').read_text()
        exported=(out/'placed.dcp').exists() and "/output/placed.dcp' has been generated." in log
        complete=r['status']=='HBM_NATIVE_JOIN_EXPORTED'
        if complete:
            require(exported and r['returncode']==0 and not r['timed_out'],'Placement completion')
            require((out/'STATUS.txt').read_text().strip()=='HBM_BOARD_PLACED_REVIEW_REQUIRED','Placement marker')
        value.update(placement_completed=complete,placed_checkpoint_exported=exported,timed_out=r['timed_out'],
                     checkpoint_sha256=sha(out/'placed.dcp') if exported else None,
                     errors=re.findall(r'^ERROR:.*$',log,re.M),constraints_equivalence_qualified=False)
        if (out/'timing.rpt').exists():value['timing']=report_timing(out/'timing.rpt')
        if (out/'cdc.rpt').exists():value['cdc']=[dict(rule=n,severity=s,count=int(c)) for n,s,c in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+',(out/'cdc.rpt').read_text(),re.M)]
        if (out/'drc.rpt').exists():
            value['drc']=[dict(rule=n,severity=s,count=int(c)) for n,s,c in re.findall(
                r'^\| ([A-Z]+-\d+)\s*\| (Critical Warning|Warning|Error|Info)\s*\| [^|]+\| (\d+)\s*\|$',
                (out/'drc.rpt').read_text(),re.M)]
        if (out/'methodology.rpt').exists():
            value['methodology']=[dict(rule=n,severity=s,count=int(c)) for n,s,c in re.findall(
                r'^\| ([A-Z]+-\d+)\s*\| (Critical Warning|Warning|Error|Info|Advisory)\s*\| [^|]+\| (\d+)\s*\|$',
                (out/'methodology.rpt').read_text(),re.M)]
        if (out/'waivers.rpt').exists():
            require('No waivers found.' in (out/'waivers.rpt').read_text(),'Unexpected waiver')
            value['waivers']=0
        if (out/'bus_skew.rpt').exists():
            text=(out/'bus_skew.rpt').read_text()
            summary=re.split(r'\n2\. Bus Skew Report Per Constraint\n-+',text)[0]
            skew=[dict(corner=c,requirement_ns=float(r),actual_ns=float(a),slack_ns=float(s)) for c,r,a,s in re.findall(
                r'^\s+(Slow|Fast)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)\s*$',summary,re.M)]
            require(skew,'Bus skew summary')
            value['bus_skew']=dict(constraints_reported=len(skew),failing=sum(x['slack_ns']<0 for x in skew),
                                    minimum_slack_ns=min(x['slack_ns'] for x in skew))
        if (out/'readiness_paths.tsv').exists():
            paths=table(out/'readiness_paths.tsv');require(len(paths)==6,'Readiness path coverage')
            require(all(float(x['requirement_ns'])==(10 if x['type']=='max' else 0) for x in paths),'Readiness timing requirements')
            value['readiness_paths']=paths
        value['timing_qualified']=False
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['startup','elaborate','placement']);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();value=json.dumps(review(a.root,a.mode),indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(value)
    else:print(value,end='')
