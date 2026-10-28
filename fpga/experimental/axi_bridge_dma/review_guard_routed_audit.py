"""Check reset-buffer provenance and constrained synchronizer stages offline.

Successful evidence review does not accept the candidate's CDC-11 findings.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import re
from review_guard_integration import manifest, require, rows, sha, check_stages
from review_guard_physical_candidate import timing, route_status


def reset_buffer(buffers, enables):
    name='guarded/guard/cold_a_reg[2]_bufg_place'
    found=[r for r in buffers if r['cell']==name]
    require(len(found)==1,'Missing reset distribution buffer')
    r=found[0]
    require(r['ref']=='BUFGCE' and r['loc'].startswith('BUFGCE_'),'Unexpected reset buffer primitive/location')
    require(r['input_startpoints']=='guarded/guard/cold_a_reg[2]/C' and r['input_drivers']=='guarded/guard/cold_a_reg[2]/Q','Reset provenance changed')
    require(r['clock_loads']=='0' and r['data_loads']=='0' and int(r['reset_loads'])==1451,'Reset buffer consumer role changed')
    require(len(enables)==1,'Missing reset buffer enable')
    e=enables[0]
    require(e['buffer']==name and e['driver_ref']=='VCC' and e['driver']=='guarded/guard/VCC/P' and e['ce_type']=='ASYNC','Reset buffer enable changed')
    # Vivado exports unset inversion properties as empty strings. UG974
    # documents their defaults as 0; report both raw values, do not invent reads.
    require(e['i_inverted'] in ('',"1'b0",'0') and e['ce_inverted'] in ('',"1'b0",'0'),'Inverted reset buffer')
    mgmt=[v for v in buffers if v['cell']=='management_buffer']
    require(len(mgmt)==1 and mgmt[0]['ref']=='BUFGCE' and mgmt[0]['input_startpoints']=='mgmt_clk_in' and int(mgmt[0]['clock_loads'])==33,'Management clock provenance changed')
    axi=[v for v in buffers if v['cell'].endswith('/phy_clk_i/bufg_gt_userclk')]
    require(len(axi)==1 and axi[0]['ref']=='BUFG_GT' and axi[0]['input_startpoints'].endswith('.GTYE4_CHANNEL_PRIM_INST/TXOUTCLK') and int(axi[0]['clock_loads'])>0,'Native AXI clock provenance changed')
    return dict(cell=name,source=r['input_drivers'],reset_loads=1451,clock_loads=0,
                enable=e,unset_inversions_use_documented_primitive_defaults=True,
                management_buffer=mgmt[0]['loc'],native_axi_buffer=axi[0]['loc'])


def stages(text):
    result=check_stages(text,text)
    require(all(math.isfinite(float(r['slack'])) and float(r['slack'])>=0 for r in rows(text)),'Unmet or nonfinite synchronizer timing')
    return dict(paths=56,min_setup_ns=result['max'],min_hold_ns=result['min'])


def active_xdc(text):
    return '\n'.join(line.strip() for line in text.splitlines() if line.strip() and not line.startswith('#'))


def review(root,checkpoint):
    count=manifest(root);out=root/'output';r=json.loads((root/'RESULT.json').read_text())
    require(r['status']=='ROUTED_GUARD_AUDIT_EXPORTED' and r['returncode']==0 and not r['timed_out'] and r['inputs_unchanged'],'Incomplete routed audit')
    require(sha(checkpoint)==json.loads((root/'INPUT.json').read_text())['sha256'],'Wrong routed checkpoint')
    require(active_xdc((out/'before.xdc').read_text())==active_xdc((out/'after.xdc').read_text()),'Read-only audit changed constraints')
    b=reset_buffer(rows((out/'buffers.tsv').read_text()),rows((out/'reset_enable.tsv').read_text()))
    return dict(status='ROUTED_AUDIT_EVIDENCE_REVIEWED',files=count,reset_buffer=b,
                synchronizers=stages((out/'interstage.tsv').read_text()),
                timing=timing((out/'timing.rpt').read_text()),route=route_status((out/'route_status.rpt').read_text()),
                candidate_accepted=False,board_timing_closed=False)


def controls(root):
    out=root/'output';b=rows((out/'buffers.tsv').read_text());e=rows((out/'reset_enable.tsv').read_text());s=(out/'interstage.tsv').read_text()
    rejected=[]
    def reject(name,fn):
        try:fn()
        except ValueError:rejected.append(name)
        else:raise ValueError('Corrupt evidence accepted: '+name)
    for name,field,value in [('reset_as_clock','clock_loads','1'),('wrong_reset_driver','input_drivers','other/Q'),('lost_reset_consumers','reset_loads','0')]:
        changed=copy.deepcopy(b);next(v for v in changed if 'cold_a_reg[2]_bufg_place' in v['cell'])[field]=value
        reject(name,lambda:reset_buffer(changed,e))
    for name,field,value in [('disabled_buffer','driver_ref','GND'),('inverted_input','i_inverted',"1'b1"),('inverted_enable','ce_inverted',"1'b1")]:
        changed=copy.deepcopy(e);changed[0][field]=value
        reject(name,lambda:reset_buffer(b,changed))
    reject('missing_stage',lambda:stages('\n'.join(s.splitlines()[:-1])))
    reject('wrong_stage_clock',lambda:stages(s.replace('mgmt_clk','axi_aclk_out',1)))
    first=rows(s)[0]['slack']
    reject('negative_stage_slack',lambda:stages(s.replace('\t'+first+'\t','\t-0.1\t',1)))
    reject('nonfinite_stage_slack',lambda:stages(s.replace('\t'+first+'\t','\tinf\t',1)))
    return rejected


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('audit',type=Path);ap.add_argument('routed_checkpoint',type=Path)
    a=ap.parse_args();r=review(a.audit,a.routed_checkpoint);r['negative_controls']=controls(a.audit)
    print(json.dumps(r,indent=2))
