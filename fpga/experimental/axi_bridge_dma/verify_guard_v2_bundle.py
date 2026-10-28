"""Recheck a restored V2 evidence bundle entirely offline, without Vivado."""
import argparse
import hashlib
import json
from pathlib import Path
from review_guard import require, review as review_legacy
from review_guard_v2_controls import review_controls
from review_guard_v2_cdc import load, review as review_cdc
from review_clock_loss_native import review as review_native
from check_guard_v2_cdc import check as check_cdc
from check_guard_v2_controls import check as check_controls


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text())
def normalized(value):return json.loads(json.dumps(value))


def verify(root):
    root=root.resolve();files=read(root/'BUNDLE_FILES.json')
    for name,digest in files.items():
        path=(root/name).resolve()
        require(path.is_relative_to(root) and path.is_file() and sha(path)==digest,'Restored file integrity: '+name)
    runs=read(root/'RUNS.json');roots={phase:root/path for phase,path in runs.items()}
    require(set(roots)=={'rtl','cdc','native'},'Three completed phases')
    results={phase:read(path/'RESULT.json') for phase,path in roots.items()}
    require(results['rtl']['status']=='GUARD_V2_RTL_TRACE_PASS','Completed RTL matrix')
    require(results['cdc']['status']=='GUARD_V2_SYNTHESIS_EXPORTED','Completed synthesis')
    require(results['native']['status']=='NATIVE_AXI_BOUNDARY_CLOCK_INJECTION_PASS','Completed native clock cases')
    variant=sha(root/'source/ntrpu_axi_bridge_guard_v2.sv')
    for phase,path in roots.items():
        require(sha(path/'source/ntrpu_axi_bridge_guard_v2.sv')==variant,'Same V2 in every phase')
        for name,digest in read(path/'FILES.json').items():require(sha(path/name)==digest,'Original phase manifest: '+name)
        require(results[phase]['error'] is None,'No phase errors')
    for phase in ('rtl','cdc'):
        require(results[phase]['sources_unchanged'],'Frozen source inputs')
        for name,digest in results[phase]['sources'].items():require(sha(roots[phase]/'source'/name)==digest,'Frozen source: '+name)
    native=roots['native'];prepared=read(native/'source/PREPARED.json')
    require(prepared['variant']=='ntrpu_axi_bridge_guard_v2','Explicit native variant identity')
    require(sha(native/'source/ntrpu_axi_bridge_guard.sv')==variant,'Private filename alias matches V2')
    require('Compiling module xil_defaultlib.ntrpu_axi_bridge_guard_v2' in (native/'elaborate.log').read_text(),
            'Native elaboration actually compiles the V2 module')
    for name,digest in prepared['files'].items():require(sha(native/'source'/name)==digest,'Prepared native source: '+name)
    require(results['native']['seed_unchanged'],'Native seed preserved')
    checked=[];rtl=roots['rtl']
    profiles=[(2,5,1),(4,7,3),(7,2,2),(2,2,5)]
    for p,clocks in enumerate(profiles):
        for case in range(17):
            name=f'legacy-{p}-{case:02}';log=rtl/(name+'.log');vcd=rtl/name/'guard.vcd'
            r=review_legacy(log.read_text(),vcd.read_text(),case,clocks)
            r.update(name=name,log_sha256=sha(log),vcd_sha256=sha(vcd))
            require(normalized(r)==read(rtl/name/'REVIEW.json'),'Restored legacy review equals VM: '+name);checked.append(r)
    for case in range(16):
        name=f'controls-{case:02}';log=rtl/(name+'.log');vcd=rtl/name/'guard.vcd'
        r=review_controls(log.read_text(),vcd.read_text(),case)
        r.update(name=name,log_sha256=sha(log),vcd_sha256=sha(vcd))
        require(normalized(r)==read(rtl/name/'REVIEW.json'),'Restored pulse review equals VM: '+name);checked.append(r)
    require(normalized(checked)==results['rtl']['reviews'],'All 84 reviews match remote aggregate')
    native_reviews=[]
    for level in (0,1):
        log=native/f'simulate-{level}.log';vcd=native/f'level-{level}/clock_loss.vcd'
        r=review_native(log.read_text(),vcd.read_text());r.update(log_sha256=sha(log),vcd_sha256=sha(vcd))
        require(r['stop_level']==level and normalized(r)==read(native/f'level-{level}/REVIEW.json'),'Restored native review equals VM')
        native_reviews.append(r)
    require(normalized(native_reviews)==results['native']['reviews'],'Both native reviews match remote aggregate')
    reports=roots['cdc']/'reports';baseline=load(reports/'baseline');scoped=load(reports/'scoped')
    cdc=review_cdc(baseline,scoped);negative_cdc=check_cdc(baseline,scoped);negative_pulse=check_controls(rtl)
    return dict(status='GUARD_V2_RESTORED_EVIDENCE_PASS',files_verified=len(files),guard_sha256=variant,
                rtl_cases=len(checked),native_cases=len(native_reviews),native_reviews=native_reviews,cdc=cdc,
                cdc_corruptions_rejected=len(negative_cdc['rejected_controls']),
                pulse_corruptions_rejected=len(negative_pulse['rejected_controls']),physical_timing_closed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=verify(a.root)
    require(not a.output.resolve().is_relative_to(a.root.resolve()),'Keep the receipt outside the immutable evidence bundle')
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
