"""Review candidate pin constraints and inventory completed implementation stages."""
import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from review_hbm_bar_control import rows
from review_hbm_loader_readiness import expand_xdc
from review_hbm_board_boundary import DONOR_SHA, require


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def assigned_constraints(before, candidate):
    a,b=expand_xdc(before),expand_xdc(candidate)
    require(not a-b, 'Original constraints removed')
    expected=Counter([
        'set_property PACKAGE_PIN AR40 [get_ports sys_clk_p]',
        'set_property PACKAGE_PIN AR41 [get_ports sys_clk_n]',
        'set_property PACKAGE_PIN BF41 [get_ports sys_rst_n]',
        'set_property PACKAGE_PIN BK31 [get_ports fabric_ref_clk_in]',
        'set_property IOSTANDARD LVCMOS18 [get_ports sys_rst_n]',
        'set_property IOSTANDARD LVCMOS18 [get_ports fabric_ref_clk_in]',
        'set_property LOC GTYE4_COMMON_X0Y1 [get_cells refclk_ibuf]'])
    require(b-a==expected, 'Candidate changed more than the seven explicit pin/standard/site constraints')


def review(root):
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'}, 'Closed inventory')
    for name,digest in files.items():require(sha(root/name)==digest, 'Closed file hash '+name)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['sources_unchanged'] and result['checkpoint_sha256']==DONOR_SHA, 'Donor/source integrity')
    for name,digest in result['sources'].items():require(sha(root/'source'/name)==digest, 'Source identity '+name)
    out=root/'output'
    before=(out/'before.xdc').read_text();candidate=(out/'candidate.xdc').read_text()
    assigned_constraints(before,candidate)
    pins=rows(root,'candidate_pins.tsv')
    require([(r['port'],r['ball'],r['bank'],r['standard']) for r in pins]==[
        ('sys_clk_p','AR40','125','NONE'),('sys_clk_n','AR41','125','NONE'),
        ('sys_rst_n','BF41','65','LVCMOS18'),('fabric_ref_clk_in','BK31','64','LVCMOS18')], 'Candidate pin identity')
    require(all(r['status']=='EXPERIMENT_ONLY' for r in pins), 'Unqualified pin release')
    site=rows(root,'reference_site.tsv')
    require(site==[dict(cell='refclk_ibuf',site='GTYE4_COMMON_X0Y1',ball='AR40',status='EXPERIMENT_ONLY')], 'Reference site identity')
    pinned=(out/'pinned/drc.rpt').read_text()
    require(re.search(r'Checks found:\s+0\b',pinned), 'Initial pin DRCs remain')
    stages={}
    for label in ('pinned','optimized','placed'):
        folder=out/label
        if not folder.exists():continue
        record={}
        if (folder/'drc.rpt').exists():
            text=(folder/'drc.rpt').read_text()
            record['drc']=[dict(rule=r,severity=s,count=int(n)) for r,s,n in re.findall(
                r'^\| ([A-Z]+-\d+)\s*\| (Critical Warning|Warning|Error|Info)\s*\| [^|]+\| (\d+)\s*\|$',text,re.M)]
        if (folder/'timing.rpt').exists():
            text=(folder/'timing.rpt').read_text();text=text[text.index('Design Timing Summary'):]
            values=next(l.split() for l in text.splitlines() if re.match(r'^\s*-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\s+',l))
            require(len(values)==12, 'Timing report format')
            record['timing']=dict(wns_ns=float(values[0]),whs_ns=float(values[4]),wpws_ns=float(values[8]),
                failing_setup_endpoints=int(values[2]),failing_hold_endpoints=int(values[6]))
        if (folder/'cdc.rpt').exists():
            text=(folder/'cdc.rpt').read_text()
            record['cdc']=[dict(rule=r,severity=s,count=int(n)) for r,s,n in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+',text,re.M)]
        stages[label]=record
    log=(root/'tool.log').read_text(errors='replace')
    checkpoints={}
    for label in ('candidate','optimized','placed'):
        path=out/(label+'.dcp')
        if path.exists():
            require(re.search(r"The checkpoint '[^'\n]+/output/"+label+r"\.dcp' has been generated\.",log), 'Incomplete checkpoint '+label)
            checkpoints[label]=sha(path)
    success=result['status']=='HBM_NATIVE_JOIN_EXPORTED'
    require(result['status'] in ('FAILED','HBM_NATIVE_JOIN_EXPORTED'), 'Unknown execution status')
    if success:
        require(result['returncode']==0 and not result['timed_out'] and 'placed' in checkpoints, 'Incomplete placement')
        require((out/'STATUS.txt').read_text().strip()=='HBM_EXPLORATORY_PLACEMENT_REVIEW_REQUIRED', 'Wrong final marker')
    negative=[]
    for name,changed in [('wrong_voltage',candidate.replace('IOSTANDARD LVCMOS18','IOSTANDARD LVCMOS33',1)),
                         ('broad_clock_cut',candidate+'\nset_clock_groups -asynchronous -group [get_clocks *]\n'),
                         ('wrong_reference_site',candidate.replace('LOC GTYE4_COMMON_X0Y1 [get_cells refclk_ibuf]','LOC GTYE4_COMMON_X0Y2 [get_cells refclk_ibuf]'))]:
        try:assigned_constraints(before,changed)
        except ValueError:negative.append(name)
        else:raise ValueError('Negative assignment control escaped: '+name)
    return dict(status='HBM_PIN_ASSIGNMENT_REVIEWED_WITH_PLACEMENT_DIAGNOSTICS',
        tool_status=result['status'],tool_returncode=result['returncode'],timed_out=result['timed_out'],
        checkpoint_sha256=DONOR_SHA,candidate_constraint_additions=7,initial_pad_drc_findings=0,
        input_constraints_preserved=True,checkpoint_exports=checkpoints,stages=stages,
        placement_completed=success,physical_route_executed=False,
        errors=re.findall(r'^ERROR:.*$',log,re.M),
        failure_tail=log[-3000:] if not success else None,negative_controls=negative,
        post_optimization_constraint_equivalence_qualified=False,
        routed_timing_qualified=False,board_qualified=False,pinout_released=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();value=json.dumps(review(a.root),indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(value)
    else:print(value,end='')
