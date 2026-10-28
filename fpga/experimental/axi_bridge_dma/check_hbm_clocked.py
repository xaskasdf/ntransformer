"""Reject altered control evidence and modified reset-export RTL."""
import argparse
import json
from pathlib import Path
import shutil
from review_hbm_clocked import review,review_case,review_source

def check(root,output,vm_root=None,packaging_audit=None):
    if output.exists():raise ValueError('New negative-control directory required')
    output.mkdir(parents=True)
    good=(root/'case-01-seed-0.log').read_text()
    review_case(good,1,0)
    accepted=[]
    mutations={
        'missing_completion':good.replace('HC PASS','HC REMOVED'),
        'wrong_case':good.replace('HC PASS case=1','HC PASS case=2'),
        'wrong_seed':good.replace('HC PASS case=1 seed=0','HC PASS case=1 seed=1'),
        'duplicate_completion':good+'\n'+next(s for s in good.splitlines() if s.startswith('HC PASS')),
        'fatal_before_pass':'FATAL: injected failure\n'+good,
        'missing_fault':good.replace('HC FAULT','HC REMOVED'),
        'wrong_watchdog':good.replace('mem=1','mem=0'),
    }
    import re
    mutations['unbounded_detection']=re.sub(r'latency_ns=[\d.]+','latency_ns=1500.0',good)
    for name,text in mutations.items():
        (output/(name+'.log')).write_text(text)
        try:review_case(text,1,0)
        except ValueError as exc:accepted.append(dict(name=name,rejected=True,reason=str(exc)))
        else:raise ValueError('Altered evidence accepted: '+name)
    copy=output/'modified-source';shutil.copytree(root/'source',copy)
    target=copy/'ntrpu_axi_bridge_guard_v3_reset_export.sv'
    target.write_text(target.read_text().replace('assign management_cold_rst_n=cold_m[2];','assign management_cold_rst_n=cold_m[1];'))
    try:review_source(copy)
    except ValueError as exc:accepted.append(dict(name='early_management_release',rejected=True,reason=str(exc)))
    else:raise ValueError('Modified reset release accepted')
    if vm_root is not None:
        original=json.loads(packaging_audit.read_text())
        for name in ('real_rtl_changed','snapshot_changed','unclassified_input'):
            altered=json.loads(json.dumps(original))
            for rec in altered:
                if name=='real_rtl_changed':rec['stage_hashes']['ntrpu_axi_bridge_hbm_cdc.sv']='0'*64
                elif name=='snapshot_changed':rec['bad_copies']=['ntrpu_axi_bridge_hbm_cdc.sv']
                else:rec['changed'].append('other.log')
            path=output/(name+'.json');path.write_text(json.dumps(altered,indent=2)+'\n')
            try:review(vm_root,path)
            except ValueError as exc:accepted.append(dict(name=name,rejected=True,reason=str(exc)))
            else:raise ValueError('Invalid recovery accepted: '+name)
    result=dict(status='HBM_CLOCKED_NEGATIVES_REJECTED',count=len(accepted),checks=accepted)
    (output/'REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);ap.add_argument('output',type=Path)
    ap.add_argument('--vm-root',type=Path);ap.add_argument('--packaging-audit',type=Path)
    a=ap.parse_args();print(json.dumps(check(a.root,a.output,a.vm_root,a.packaging_audit),indent=2))
