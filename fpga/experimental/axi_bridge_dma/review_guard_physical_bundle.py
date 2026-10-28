"""Review a restored physical evidence bundle, retaining all design blockers."""
import argparse
import csv
import json
from pathlib import Path
import re
from review_guard_integration import manifest, require, rows, sha, PINNED
from review_guard_physical_candidate import review as candidate_review, route_status, scoped_cdc, timing
from review_guard_routed_audit import review as audit_review, controls, active_xdc
from check_guard_physical_contract import check as contract_controls
from build_guard_physical_contract import decode_lanes, properties


def findings(root, checkpoint):
    count=manifest(root);out=root/'output';r=json.loads((root/'RESULT.json').read_text())
    require(r['status']=='ROUTED_FINDINGS_PROBE_EXPORTED' and r['returncode']==0 and not r['timed_out'] and r['inputs_unchanged'],'Incomplete findings probe')
    require(sha(checkpoint)==json.loads((root/'INPUT.json').read_text())['sha256'],'Probe read a different checkpoint')
    require(active_xdc((out/'before.xdc').read_text())==active_xdc((out/'after.xdc').read_text()),'Probe changed constraints')
    require('No waivers found.' in (out/'waivers.rpt').read_text(),'Probe added waiver')
    sources=rows((out/'reset_sources.tsv').read_text())
    require(len(sources)==12 and len({r['destination'] for r in sources})==12,'Missing reset source coverage')
    for r in sources:
        m=re.fullmatch(r'guarded/guard/availability\[([0-3])\]\.release_m_reg\[([0-2])\]/CLR',r['destination'])
        require(m is not None,'Unknown reset consumer')
        i=int(m[1]);prefix='ntrpu_bridge_i/inst/pcie4c_ip_i/inst/'
        expected='user_lnk_up_reg' if i==0 else 'user_reset_reg'
        require(r['startpoints']==('sys_rst_n' if i==3 else prefix+expected+'/C'),'Reset source alias changed')
        require(r['drivers']==('sys_reset_n_ibuf/IBUFCTRL_INST/O' if i==3 else prefix+expected+'/Q'),'Reset driver changed')
    paths=rows((out/'methodology_paths.tsv').read_text())
    require(len(paths)==8 and {(p['bit'],p['mode']) for p in paths}=={(str(b),m) for b in (9,11,21,26) for m in ('timed','ignored')},'Methodology path coverage changed')
    measured={}
    for p in paths:
        if p['mode']=='ignored':
            require(p['count']=='0','Methodology path was cut')
            continue
        require(p['count']=='1' and p['start_clock']=='GTYE4_CHANNEL_TXOUTCLK[3]_3' and p['end_clock']=='axi_aclk_out','Methodology path clock changed')
        expected=.916 if p['bit'] in ('9','21') else .918
        require(float(p['slack'])==expected,'Routed targeted slack changed')
        report=(out/f"path_{p['bit']}_timed.rpt").read_text()
        require('MultiCycle Path   Setup -start 2' in report and 'CFGMGMTREADWRITEDONE' in report,'Native multicycle path context missing')
        measured[p['bit']]=expected
    require(len(re.findall(r'^TIMING-16#\d+ Warning$',(out/'methodology.rpt').read_text(),re.M))==4,'Methodology warnings hidden or changed')
    return dict(files=count,scoped_cdc=scoped_cdc((out/'cdc.rpt').read_text()),
                timing=timing((out/'timing.rpt').read_text()),methodology_targeted_setup_ns=measured,
                timed_paths=4,ignored_paths=0,native_multicycle_setup_start=2,
                methodology_warnings_retained=4,candidate_accepted=False)


def review(bundle):
    count=manifest(bundle);pre=bundle/'preflight';contract=bundle/'contract'
    manifest(pre);manifest(contract)
    c=json.loads((contract/'contract.json').read_text())
    require(c['part']=='xcvu47p-fsvh2892-2-e' and not c['board_approved'] and not c['joint_hbm_implemented'],'Architecture scope changed')
    require(c['preflight_manifest_sha256']==sha(pre/'FILES.json'),'Contract does not match preflight')
    require(c['checkpoint']['sha256']==sha(bundle/'inputs/native_audited.dcp'),'Preflight checkpoint mismatch')
    lanes=decode_lanes(rows((pre/'output/serial_lanes.tsv').read_text()),properties(pre/'output/package.tsv'))
    require(lanes==c['pcie']['serial_ports'],'Contract lane map changed')
    for name,digest in PINNED.items():require(sha(bundle/'source'/name)==digest,'Pinned RTL changed: '+name)
    candidate=bundle/'candidate';routed=candidate/'output/routed.dcp'
    physical=candidate_review(candidate,bundle/'inputs/native_audited.dcp',True)
    audit=audit_review(bundle/'routed-audit',routed)
    probe=findings(bundle/'findings',routed)
    require(physical['routing']==audit['timing']==probe['timing'],'Reopened checkpoint timing changed')
    require(physical['scoped_cdc']==probe['scoped_cdc'],'Reopened checkpoint CDC changed')
    rejected=[]
    def reject(name,fn):
        try:fn()
        except ValueError:rejected.append(name)
        else:raise ValueError('Unsafe evidence accepted: '+name)
    reject('strict_candidate_gate',lambda:candidate_review(candidate,bundle/'inputs/native_audited.dcp'))
    route=(candidate/'output/route_status.rpt').read_text()
    reject('unrouted_net',lambda:route_status(re.sub(r'(fully routed nets\.+\s*:\s*)66564',r'\g<1>66563',route)))
    reject('routing_error',lambda:route_status(re.sub(r'(nets with routing errors\.+\s*:\s*)0',r'\g<1>1',route)))
    cdc=(candidate/'output/routed/cdc.rpt').read_text()
    reject('downgraded_cdc',lambda:scoped_cdc(cdc.replace('Critical','Warning')))
    reject('removed_cdc',lambda:scoped_cdc('\n'.join(l for l in cdc.splitlines() if 'CDC-11' not in l)))
    reject('foreign_cdc_source',lambda:scoped_cdc(cdc.replace('user_reset_reg/C','foreign_reg/C')))
    return dict(status='RESTORED_PHYSICAL_BUNDLE_REVIEWED_WITH_BLOCKERS',files=count,
                physical=physical,audit=audit,findings=probe,
                contract_controls=contract_controls(pre),audit_controls=controls(bundle/'routed-audit'),
                physical_controls=rejected,accepted=False,board_timing_closed=False,joint_hbm_implemented=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('bundle',type=Path)
    print(json.dumps(review(ap.parse_args().bundle),indent=2))
