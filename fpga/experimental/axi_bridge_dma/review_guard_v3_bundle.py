"""Verify restored V3 evidence and its chain of immutable native inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from review_guard_integration import manifest, require, sha, PINNED
from review_guard_v3 import rtl, recovered, route, probe
from check_guard_v3 import check


def synthesizable(text):
    body,count=re.subn(r'// synthesis translate_off.*?// synthesis translate_on','',text,flags=re.S)
    require(count==1,'Unexpected simulation-only regions')
    return body


def review(root):
    count=manifest(root);reference=root/'reference/routed_v2.dcp'
    origin=root/'native-origin';recovery=root/'recovery';physical=root/'route'
    local=rtl(root/'local');xsim=rtl(root/'xsim');portable=rtl(root/'xsim',root/'topology')
    native=recovered(recovery,origin,reference)
    routed=route(physical,recovery/'output/optimized.dcp',reference)
    findings=probe(root/'probe',physical/'output/routed.dcp')
    require(findings['timing']==routed['evidence']['timing'] and findings['cdc']==routed['evidence']['cdc'],'Read-only probe changed routed results')
    require(json.loads((origin/'SEED_FILES.json').read_text())==json.loads((root/'reference/SEED_FILES.json').read_text()),'Different native seed from the loaded V2 reference')
    variants=[origin/'source/ntrpu_axi_bridge_guard_v3.sv',root/'xsim/source/ntrpu_axi_bridge_guard_v3.sv',root/'topology/source/ntrpu_axi_bridge_guard_v3.sv',root/'local/source/ntrpu_axi_bridge_guard_v3.sv',root/'source/ntrpu_axi_bridge_guard_v3.sv']
    bodies=[synthesizable(p.read_text()) for p in variants]
    require(len(set(bodies))==1,'Synthesizable V3 source differs across phases')
    require(sha(variants[-1])==sha(variants[-2])==sha(variants[-3]),'Portable assertion not tested on final source')
    require(sha(root/'source/ntrpu_axi_bridge_guarded_client_v3.sv')==sha(origin/'source/ntrpu_axi_bridge_guarded_client_v3.sv'),'Native wrapper differs')
    for name,digest in PINNED.items():require(sha(root/'source'/name)==digest,'Frozen V2/client source changed: '+name)
    require(native['evidence']['cdc']==routed['evidence']['cdc'],'CDC regressed during physical implementation')
    t=routed['evidence']['timing']
    internal=all(t[k]>=0 for k in ('wns_ns','whs_ns','wpws_ns')) and all(t[k]==0 for k in ('setup_failing','hold_failing','pulse_failing'))
    failures={}
    for p in (root/'retained-attempts').iterdir():
        manifest(p);r=json.loads((p/'RESULT.json').read_text());failures[p.name]=dict(status=r['status'],error=r.get('error'))
    processes=json.loads((root/'reviews/V3_PROCESS_COMPLETION.json').read_text())
    require(not any(p['active'] for phase in processes['phases'] for p in phase['processes']),'Private process still active')
    return dict(status='V3_SHARED_RESET_EVIDENCE_RESTORED',files=count,local=local,xsim=xsim,portable_topology=portable,
                optimized=native,routed=routed,methodology_probe=findings,negative_controls=check(recovery,root/'topology'),
                preserved_attempts=failures,synthesizable_v3_sha256=hashlib.sha256(bodies[0].encode()).hexdigest(),
                shared_reset_cdc_fix_verified=True,internal_timing_summary_met=internal,
                board_timing_closed=False,joint_hbm_implemented=False,native_postroute_simulation=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('bundle',type=Path)
    print(json.dumps(review(ap.parse_args().bundle),indent=2))
