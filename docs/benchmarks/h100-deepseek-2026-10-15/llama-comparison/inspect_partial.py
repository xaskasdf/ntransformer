import json
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parent
for n,label in [(32,'p32-d64'),(128,'p128-d64'),(512,'p512-d64'),(128,'serial-p128-d64')]:
    path=root/label/'final-logits-round0.f32'
    if not path.exists():continue
    a=np.fromfile(path,dtype='<f4').astype(np.float64)
    b=np.fromfile(root/'baseline'/f'p{n}-d64'/'final-logits.f32',dtype='<f4').astype(np.float64)
    d=a-b
    rows=[json.loads(l) for l in (root/label/'measurements.jsonl').read_text().splitlines()]
    print(label,dict(max_abs=float(np.max(np.abs(d))),rel_l2=float(np.linalg.norm(d)/np.linalg.norm(b)),
        cosine=float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b))),argmax=[int(a.argmax()),int(b.argmax())],
        agreement=[r['agreement_with_ntransformer'] for r in rows if r['kind']=='completion']),flush=True)
