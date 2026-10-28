"""Numerically review completed RTL simulation before spending time on placement."""
import argparse,hashlib,json
from pathlib import Path
import check_ntrpu_attention_projected_root_real_r2 as checker

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);a=p.parse_args()
    log=a.run/'output/seed-1.log';text=log.read_text()
    fixture=checker.load_root_fixture(a.source/'root-fixture',checker.load_fixture(a.source/'fixture'))
    result=checker.review(text,fixture,1);controls=checker.controls(text,fixture,1)
    assert len(controls)==69
    report=dict(status='PASS_PROJECTED_ROOT_BRAM_ARITHMETIC_BEFORE_ROUTE',suite=result,rejected_controls=controls,
        log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),
        checkpoint_sha256=hashlib.sha256((a.run/'output/attention_projected_root_real_synth.dcp').read_bytes()).hexdigest(),
        physical_timing_qualified=False)
    with (a.run/'arithmetic-before-route.json').open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(report))
if __name__=='__main__':main()
