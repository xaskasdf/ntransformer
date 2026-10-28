"""Audit immutable CDC simulation evidence and ordered public transactions."""
import argparse,json,re
from pathlib import Path
from review_hbm_source import require,sha,READER_SHA,review as source_review
CASES=list(range(18))+[20,21,22,23]
FIFO_SHA='be809fe998385e69a52eb2fa8687b5b50e52433a5b1a42694331efc07f1f0634'

def review(text,case,seed):
    result=source_review(text,case,seed)
    require('HC CLOCKS p=250 r=200 m=100 phase_ps=700' in text,'Missing separate clock profile')
    if case in (22,23):
        require(f'HC WATCHDOG case={case} level={case-22} management=1' in text,'Missing clock-stop observation')
    return result

def bundle(root):
    files=json.loads((root/'FILES.json').read_text())
    require(set(files)|{'FILES.json'}=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and not p.is_relative_to(root/'work')},'File set mismatch')
    for n,h in files.items():require(sha(root/n)==h,'Hash mismatch: '+n)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='HBM_CDC_FUNCTIONAL_PASS' and result['sources_unchanged'],'Incomplete phase')
    require(not result['native_hbm_pcie_simulation'],'Invalid native IP claim')
    require(sha(root/'source/ntrpu_hbm_read_frames.sv')==READER_SHA,'Reader donor mismatch')
    require(sha(root/'source/ntrpu_bridge_async_fifo.sv')==FIFO_SHA,'FIFO donor mismatch')
    for n,h in result['sources'].items():require(sha(root/'source'/n)==h,'Source result mismatch: '+n)
    require(len(result['steps'])==len(CASES)*2+(1 if result['engine']=='icarus' else 2),'Incomplete steps')
    require(all(s['returncode']==0 and not s['timed_out'] for s in result['steps']),'Failed tool step')
    reviews=[review((root/f'case-{case:02}-seed-{seed}.log').read_text(),case,seed) for seed in range(2) for case in CASES]
    require(result['reviews']==reviews,'Stored verdict mismatch')
    return dict(status='HBM_CDC_FUNCTIONAL_REVIEWED',files=len(files),cases=len(reviews),bytes_checked=sum(r['bytes_checked'] for r in reviews),retention_cases=sum(r['retained'] for r in reviews),native_hbm_pcie_simulation=False,physical_cdc_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    print(json.dumps(bundle(ap.parse_args().output),indent=2))
