import json
from pathlib import Path
import sys
sys.path.insert(0,'/home/ubuntu/ntransformer-h100/llama-bench-20261015/source/gguf-py')
from gguf import GGUFReader
root=Path(__file__).resolve().parent
rows=[]
for path in sorted((root.parent/'models/deepseek-v41-flash-q8').glob('*.gguf')):
    reader=GGUFReader(str(path))
    for t in reader.tensors:
        if t.name.endswith(('hc_attn_fn.weight','hc_ffn_fn.weight')):
            rows.append(dict(name=t.name,dtype=t.tensor_type.name,shape=list(map(int,t.shape)),bytes=int(t.n_bytes)))
    del reader
(root/'weight-metadata.json').write_text(json.dumps(rows,indent=2)+'\n')
print(json.dumps(dict(count=len(rows),types=sorted(set(t['dtype'] for t in rows)),shapes=sorted(set(str(t['shape']) for t in rows)),bytes=sum(t['bytes'] for t in rows))))
