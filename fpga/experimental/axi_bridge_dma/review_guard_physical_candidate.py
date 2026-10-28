"""Review an exploratory OOC route without mistaking it for board closure."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
from review_guard_integration import manifest,require,cdc


def timing(text):
    section=text.split('Design Timing Summary',1)[1]
    match=re.search(r'^\s*(-?\d+\.\d+\s+.*)$',section,re.M)
    require(match is not None,'Timing summary missing')
    values=match[1].split();require(len(values)==12,'Timing summary columns changed')
    return dict(wns_ns=float(values[0]),tns_ns=float(values[1]),setup_failing=int(values[2]),
                whs_ns=float(values[4]),ths_ns=float(values[5]),hold_failing=int(values[6]),
                wpws_ns=float(values[8]),pulse_failing=int(values[10]))


def exceptions(text):
    return Counter(line for line in text.splitlines() if re.match(r'set_(false_path|max_delay|min_delay|clock_groups|multicycle_path)\b',line))


def route_status(text):
    values={key:int(value) for key,value in re.findall(r'# of ([^.\n]+)\.+\s*:\s*(\d+)\s*:',text)}
    require(values.get('routable nets',0)>0,'Missing routable net count')
    require(values.get('fully routed nets')==values['routable nets'] and values.get('nets with routing errors')==0,'Incomplete or erroneous route')
    return values


def scoped_cdc(text):
    # Preserve the new physical findings as blockers. This is not a waiver:
    # the earlier synthesis reviewer remains strict and unchanged.
    summary=re.findall(r'^CDC-(\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',text,re.M)
    require(summary==[('3','Info','3'),('9','Info','8'),('11','Critical','2')],'Unexpected routed CDC inventory')
    critical=re.findall(r'^\s*\d+\s+CDC-11\s+Critical\s+.+?\s+3\s+False Path\s+(\S+)\s+(\S+)\s*$',text,re.M)
    expected={('ntrpu_bridge_i/inst/pcie4c_ip_i/inst/user_reset_reg/C',f'guarded/guard/availability[{i}].release_m_reg[0]/CLR') for i in (1,2)}
    require(len(critical)==2 and set(critical)==expected,'CDC-11 endpoints changed')
    info='\n'.join(line for line in text.splitlines() if not re.search(r'\bCDC-11\b',line))
    cdc(info)
    return dict(summary=[dict(rule='CDC-'+r,severity=s,count=int(n)) for r,s,n in summary],critical_endpoints=critical,accepted=False)


def review(root,checkpoint,inventory_only=False):
    count=manifest(root);result=json.loads((root/'RESULT.json').read_text());out=root/'output'
    require(result['status']=='NATIVE_PHYSICAL_CANDIDATE_EXPORTED' and result['returncode']==0 and not result['timed_out'] and result['inputs_unchanged'],'Physical candidate did not complete')
    require(result['placement_executed'] and result['physical_route_executed'] and not result['board_timing_closed'],'Wrong physical scope')
    require(hashlib.sha256(checkpoint.read_bytes()).hexdigest()==json.loads((root/'INPUT.json').read_text())['sha256'],'Wrong source checkpoint')
    pins=list(csv.DictReader((out/'candidate_pins.tsv').open(),delimiter='\t'))
    require({(p['port'],p['ball'],p['bank']) for p in pins}=={('sys_clk_p','AR40','125'),('sys_clk_n','AR41','125'),('sys_rst_n','BF41','65')} and len(pins)==3,'Unexpected pin assignment')
    require(exceptions((out/'input.xdc').read_text())==exceptions((out/'candidate.xdc').read_text()),'Candidate changed timing exceptions')
    effective=(out/'routed/effective.xdc').read_text()
    require(exceptions((out/'candidate.xdc').read_text())==exceptions(effective),'Physical flow changed timing exceptions')
    require('HD.CLK_SRC' not in effective and 'set_input_delay' not in effective and 'set_output_delay' not in effective,'Unreviewed clock origin or I/O delays')
    require('No waivers found.' in (out/'routed/waivers.rpt').read_text(),'Unexpected waivers')
    cdc_counts=scoped_cdc((out/'routed/cdc.rpt').read_text())
    regs=list(csv.DictReader((out/'guard_placement.tsv').open(),delimiter='\t'))
    sync=[r for r in regs if r['async_reg'] in ('1','TRUE')]
    require(len(sync)==42 and all(r['loc'].startswith('SLICE_') and r['bel'] for r in sync),'Synchronizers not physically placed')
    checked=dict(re.findall(r'\d+\. checking (\w+) \((\d+)\)',(out/'routed/check_timing.rpt').read_text()))
    require(all(checked.get(k)=='0' for k in ('no_clock','multiple_clock','unconstrained_internal_endpoints','loops')),'Internal timing coverage changed')
    require(int(checked['no_input_delay'])>0 and int(checked['no_output_delay'])>0,'Unexpected external timing coverage')
    placed=timing((out/'placed/timing.rpt').read_text());routed=timing((out/'routed/timing.rpt').read_text())
    route=route_status((out/'route_status.rpt').read_text())
    drc=(out/'routed/drc.rpt').read_text()
    rules=re.findall(r'^\|\s*([A-Z0-9]+-\d+)\s*\|\s*(Critical Warning|Warning|Error|Info)\s*\|',drc,re.M)
    methodology=re.findall(r'^\|\s*([A-Z0-9]+-\d+)\s*\|\s*(Critical Warning|Warning|Error|Info)\s*\|[^|]+\|\s*(\d+)\s*\|',(out/'routed/methodology.rpt').read_text(),re.M)
    blockers=['CDC-11: duplicate synchronization of native user_reset in management domain',
              'TIMING-16: four warnings retained; correlate with the independent native-multicycle path probe',
              'Management input lacks physical clock origin (HD.CLK_SRC)',
              'External I/O timing budgets and joint HBM implementation absent']
    require(inventory_only,'Candidate NOT accepted: '+'; '.join(blockers))
    return dict(status='EXPLORATORY_OOC_PHYSICAL_EVIDENCE_WITH_BLOCKERS',files=count,
                placement=placed,routing=routed,scoped_cdc=cdc_counts,placed_async_reg=42,
                route_counts=route,drc_rules=rules,methodology_rules=methodology,blockers=blockers,accepted=False,
                missing_input_delays=int(checked['no_input_delay']),missing_output_delays=int(checked['no_output_delay']),
                internal_summary_met=all(routed[k]>=0 for k in ('wns_ns','whs_ns','wpws_ns')),
                board_timing_closed=False,joint_hbm_implemented=False,native_postroute_simulation_executed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('candidate',type=Path);ap.add_argument('source_checkpoint',type=Path)
    ap.add_argument('--inventory-only',action='store_true',help='Record findings without accepting the physical candidate')
    a=ap.parse_args();print(json.dumps(review(a.candidate,a.source_checkpoint,a.inventory_only),indent=2))
