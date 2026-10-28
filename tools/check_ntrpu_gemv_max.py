"""Compare maximum-shape RTL values to independently evaluated Python FP64."""
import argparse
import json
from pathlib import Path
from check_ntrpu_gemv_fixture import review as fixture_review
from prepare_ntrpu_gemv_packed_fixture import SHAPES, synthetic


def reference_manifest():
    jobs=[]
    for ident,(rows,k) in enumerate(SHAPES,1):
        _,_,_,values=synthetic(rows,k,ident)
        jobs.append(dict(id=ident,rows=rows,k_blocks=k,tile_k=32,
                         tensor=f'synthetic/{rows}x{k*32}',reference_fp64_bits=values))
    return dict(format='NTRGEMV_MAX01',jobs=jobs,matrices=[],
                scope='Synthetic maximum/odd-row dimensions, not real Llama weights or inference')


def review(text,seed):
    return fixture_review(text,reference_manifest(),seed)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--seed',type=int,required=True)
    args=parser.parse_args();result=review(args.log.read_text(),args.seed)
    print(json.dumps(result));raise SystemExit(result['status']!='PASS')
