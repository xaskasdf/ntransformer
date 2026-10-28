import copy
import json
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0, '/home/ubuntu/ntransformer-h100/source/scripts')
from train_brandon_experiment import import_brandon
from ntransformer_training import brandon
from ntransformer_training.ops import swiglu as native_swiglu

root = Path('/lambda/nfs/btiny/brandon-mini/sources/stage1-v1')
TinyLlama, ModelConfig, _, _, trainer = import_brandon(root)
torch.set_num_threads(2)
torch.set_float32_matmul_precision('high')
payload = torch.load('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1/sft/best.pt', map_location='cpu', weights_only=False)
seed = TinyLlama(ModelConfig(**payload['config']))
seed.load_state_dict(payload['model_state_dict'])
data = trainer.read_instruction_jsonl(Path('/lambda/nfs/btiny/brandon-mini/distillation/text-v1/filtered/train.jsonl'))
tokenizer = trainer.Tokenizer(str(root / 'data/tokenizer_8k.model'))
rng = np.random.default_rng(np.random.SeedSequence([20261013, 0, 0, 0]))
batch = [tensor.cuda() for tensor in trainer.instruction_batch([data[i] for i in rng.integers(len(data), size=8)], tokenizer, 1024)]
records = []
def checked_swiglu(gate, up):
    out = native_swiglu(gate, up)
    expected = F.silu(gate) * up
    records.append({'forward_mismatches': (out != expected).sum().item()})
    index = len(records) - 1
    def check_backward(dy):
        with torch.enable_grad():
            g, u = gate.detach().requires_grad_(), up.detach().requires_grad_()
            a = torch.autograd.grad(native_swiglu(g, u), (g, u), dy)
            b = torch.autograd.grad(F.silu(g) * u, (g, u), dy)
        records[index]['backward_mismatches'] = [(x != y).sum().item() for x, y in zip(a, b)]
        records[index]['examples'] = []
        for operand, x, y in zip(('gate', 'up'), a, b):
            indices = torch.nonzero(x.flatten() != y.flatten()).flatten()[:2]
            for i in indices:
                records[index]['examples'].append(dict(operand=operand, gate=g.flatten()[i].item(),
                    up=u.flatten()[i].item(), dy=dy.flatten()[i].item(), native=x.flatten()[i].item(), reference=y.flatten()[i].item()))
        return dy
    out.register_hook(check_backward)
    return out

report = []
for mode in ('reference', 'swiglu', 'norms'):
    model = copy.deepcopy(seed).cuda()
    if mode != 'reference':
        brandon.install(model, native_norms=mode == 'norms', native_swiglu=mode == 'swiglu')
    if mode == 'swiglu':
        brandon.swiglu = checked_swiglu
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = model(*batch, targets_shifted=False)[1]
    loss.backward()
    gradients = {n: p.grad.detach().cpu() for n, p in model.named_parameters()}
    if mode == 'reference':
        reference = gradients
    errors = []
    squared_diff, squared_ref = 0., 0.
    for name, g in gradients.items():
        r = reference[name]
        diff = (g.double()-r.double()).square().sum().item()
        scale = r.double().square().sum().item()
        squared_diff += diff
        squared_ref += scale
        if diff:
            errors.append(dict(name=name, relative=(diff/max(scale,1e-30))**.5))
    report.append(dict(mode=mode, loss=loss.item(), relative=(squared_diff/squared_ref)**.5,
                       differing_parameters=len(errors), worst=sorted(errors,key=lambda r:r['relative'],reverse=True)[:8]))
    del model
print(json.dumps(dict(model=report, primitives=records), indent=2))
