import json
from pathlib import Path
import sys
sys.path.insert(0,'/home/ubuntu/ntransformer-h100/llama-bench-20261015/source/gguf-py')
from gguf import GGUFReader
ROOT=Path(__file__).resolve().parent
folder=ROOT.parent/'models/deepseek-v41-flash-q8'
rows=[]
for path in sorted(folder.glob('*.gguf')):
    reader=GGUFReader(str(path))
    for t in reader.tensors:
        if t.name.endswith('ffn_gate_inp.weight'):
            rows.append(dict(name=t.name,dtype=t.tensor_type.name,shape=list(map(int,t.shape)),bytes=int(t.n_bytes)))
    del reader
assert len(rows)==40
(ROOT/'router-weight-metadata.json').write_text(json.dumps(dict(tensors=rows,total_bytes=sum(r['bytes'] for r in rows)),indent=2)+'\n')
print(json.dumps(dict(layers=len(rows),types=sorted(set(r['dtype'] for r in rows)),bytes=sum(r['bytes'] for r in rows))))
