import datetime
import json
from pathlib import Path
import shutil
import sys

FAIR=Path('/home/ubuntu/ntransformer-h100/ds41-fair-20261016')
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(FAIR))
import common as b
from admission import admitted
import run as fair_run

def main():
    b.verify()
    b.ROOT=ROOT
    fair_run.ROOT=ROOT
    b.ENV=b.BASE_ENV | {'GGML_OP_OFFLOAD_MIN_BATCH':'1000000'}
    b.save('control-plan.json',dict(scope='Diagnostic ubatch ablation, not a replacement benchmark',
        baseline=str(FAIR),workloads=['p32','p512'],ubatch=1,repeats=1,
        threads=52,cpu_expert_layers=20,gpu_count=2,op_offload=False,
        binary_sha256=b.sha(FAIR/'bench-llama'),model_identity=b.model_identity()))
    with admitted(b),fair_run.monitoring():
        for name in ['p32','p512']:
            label='llama-serial-'+name
            b.command(label,[FAIR/'bench-llama',b.MODEL,FAIR/('workload-'+name+'.json'),
                ROOT/label,1,52,20,1,2])
            data=[json.loads(line) for line in (ROOT/label/'measurements.jsonl').read_text().splitlines()]
            assert len([r for r in data if r['kind']=='completion'])==2
            assert all(r['repeat_bitwise_equal'] for r in data if r['kind']=='completion')
    b.status('complete')
    destination=Path('/lambda/nfs/btiny/ntransformer-training/deepseek-audit-20261016/controls')
    destination.mkdir(parents=True,exist_ok=True)
    for path in ROOT.iterdir():
        if path.is_file():shutil.copy2(path,destination/path.name)
        elif path.name.startswith('llama-serial-'):shutil.copytree(path,destination/path.name,dirs_exist_ok=True)

if __name__=='__main__':
    try:main()
    except BaseException as error:
        b.ROOT=ROOT;b.status('failed',error=repr(error));raise
