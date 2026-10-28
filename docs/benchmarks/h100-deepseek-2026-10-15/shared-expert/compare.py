from pathlib import Path
import json, hashlib
import numpy as np
from common import root

# Fixed before inspecting full-model results. Baseline control must be exact;
# migrated expf is bounded numerically and must preserve all sampled greedy IDs.
MAX_ABS = 1e-3
MAX_REL_L2 = 1e-4
V = 129280

def compare(a, b):
    x = np.fromfile(root / (a + '.bin'), dtype='<f4').reshape(-1, V)
    y = np.fromfile(root / (b + '.bin'), dtype='<f4').reshape(-1, V)
    assert x.shape == y.shape and np.isfinite(x).all() and np.isfinite(y).all()
    rows = []
    for i, (left, right) in enumerate(zip(x, y)):
        diff = right.astype(np.float64) - left.astype(np.float64)
        absmax = float(np.max(np.abs(diff)))
        rel = float(np.linalg.norm(diff) / max(np.linalg.norm(left.astype(np.float64)), 1e-30))
        top1 = int(np.argmax(left)) == int(np.argmax(right))
        top8 = bool(np.array_equal(np.argsort(left)[-8:][::-1], np.argsort(right)[-8:][::-1]))
        assert absmax <= MAX_ABS and rel <= MAX_REL_L2 and top1, (a,b,i,absmax,rel,top1)
        rows.append(dict(index=i, max_abs=absmax, rel_l2=rel, top1_equal=top1, top8_order_equal=top8,
                         changed=int(np.count_nonzero(left.view(np.uint32) != right.view(np.uint32)))))
    return dict(reference=a,candidate=b,vectors=len(x),floats=int(x.size),rows=rows,
                max_abs=max(r['max_abs'] for r in rows),max_rel_l2=max(r['rel_l2'] for r in rows),
                all_top1_equal=all(r['top1_equal'] for r in rows),all_top8_order_equal=all(r['top8_order_equal'] for r in rows))

if __name__ == '__main__':
    record=dict(thresholds=dict(max_abs=MAX_ABS,max_rel_l2=MAX_REL_L2,all_greedy_ids_equal=True),
                benchmark=compare('cpu1','gpu1'), benchmark_exp32=compare('cpu1','exp32'))
    for a,b in [('cpu1','cpu2'),('cpu1','trace-cpu'),('cpu1','previous'),('gpu1','gpu2'),('gpu1','trace-gpu')]:
        assert (root/(a+'.bin')).read_bytes()==(root/(b+'.bin')).read_bytes(),(a,b)
    record['controls_bitwise_exact']=True
    record['gpu_runs_bitwise_repeatable']=True
    if (root/'probe-gpu-command.json').exists():record['contexts']=compare('probe-cpu','probe-gpu')
    record['sha256']={name:hashlib.sha256((root/(name+'.bin')).read_bytes()).hexdigest() for name in ['cpu1','gpu1']}
    (root/'numerics.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ['benchmark','benchmark_exp32','contexts']},indent=2))
    for key in ['benchmark','benchmark_exp32','contexts']:
        if key in record:print(key,{k:v for k,v in record[key].items() if k!='rows'})
