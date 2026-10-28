import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, '/home/ubuntu/ntransformer-h100/source/scripts')
from train_brandon_experiment import import_brandon
from ntransformer_training import brandon
from ntransformer_training.ops import rmsnorm_compatible
root = Path('/lambda/nfs/btiny/brandon-mini/sources/stage1-v1')
TinyLlama, ModelConfig, _, _, trainer = import_brandon(root)
torch.set_num_threads(2)
torch.set_float32_matmul_precision('high')
payload = torch.load('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1/sft/best.pt', map_location='cpu', weights_only=False)
model = TinyLlama(ModelConfig(**payload['config'])).cuda()
model.load_state_dict(payload['model_state_dict'])
data = trainer.read_instruction_jsonl(Path('/lambda/nfs/btiny/brandon-mini/distillation/text-v1/filtered/train.jsonl'))
tokenizer = trainer.Tokenizer(str(root / 'data/tokenizer_8k.model'))
rng = np.random.default_rng(np.random.SeedSequence([20261013, 0, 0, 0]))
batch = [t.cuda() for t in trainer.instruction_batch([data[i] for i in rng.integers(len(data), size=8)], tokenizer, 1024)]
records = []
def check(x, w, eps):
    out = rmsnorm_compatible(x, w, eps)
    expected = (x * torch.rsqrt(x.pow(2).mean(-1,keepdim=True) + eps)) * w
    entry = dict(shape=list(x.shape), forward=(out != expected).sum().item())
    records.append(entry)
    def backward(dy):
        with torch.enable_grad():
            a, b = x.detach().requires_grad_(), w.detach().requires_grad_()
            actual = torch.autograd.grad(rmsnorm_compatible(a,b,eps), (a,b), dy)
            ref = torch.autograd.grad((a*torch.rsqrt(a.pow(2).mean(-1,keepdim=True)+eps))*b, (a,b), dy)
        entry['mismatches'] = [(u != v).sum().item() for u,v in zip(actual,ref)]
        entry['examples'] = []
        for name,u,v in zip(('dx','dw'),actual,ref):
            for i in torch.nonzero(u.flatten()!=v.flatten()).flatten()[:2]:
                entry['examples'].append(dict(name=name,index=i.item(),native=u.flatten()[i].item(),reference=v.flatten()[i].item()))
        return dy
    out.register_hook(backward)
    return out
brandon.rmsnorm_compatible = check
brandon.install(model,norm_mode='compatible')
with torch.autocast('cuda',dtype=torch.bfloat16):
    loss=model(*batch,targets_shifted=False)[1]
loss.backward()
print(json.dumps(records,indent=2))
