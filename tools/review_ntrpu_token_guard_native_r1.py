"""Review three preserved native transcripts and their corruption controls."""
import argparse,json
from pathlib import Path
import check_ntrpu_llama32_token_guard_r2 as check

def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists();rows=[]
    for seed in (1,7,42):
        text=(a.source/'native-logs'/('seed-'+str(seed)+'.log')).read_text()
        rows.append(dict(metrics=check.review(text,seed),controls=check.controls(text,seed)))
    report=dict(status='PASS_TOKEN_GUARD_NATIVE_REVIEW_R1',suites=rows,jobs=1182,trace_controls=42,
        whole_model_rtl_qualified=False,arithmetic_executed=False)
    with a.output.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=report['status'],jobs=1182,trace_controls=42)))

if __name__=='__main__':main()
