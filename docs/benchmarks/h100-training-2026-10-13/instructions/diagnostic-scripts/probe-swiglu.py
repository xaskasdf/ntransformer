import json
import sys
import torch
import torch.nn.functional as F
sys.path.insert(0, '/home/ubuntu/ntransformer-h100/source/python')
from ntransformer_training.ops import swiglu

torch.set_num_threads(2)
torch.manual_seed(727)
values = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).cuda()
values = values[torch.isfinite(values) & (values.abs() <= 40)]
g = values.repeat_interleave(32).requires_grad_()
u = torch.randn_like(g).requires_grad_()
dy = torch.randn_like(g)
actual = swiglu(g, u)
expected = F.silu(g) * u
native_grad = torch.autograd.grad(actual, (g, u), dy)
torch_grad = torch.autograd.grad(expected, (g, u), dy)
results = []
for name, a, b in [('forward', actual, expected), ('gate_grad', native_grad[0], torch_grad[0]),
                   ('up_grad', native_grad[1], torch_grad[1])]:
    indices = torch.nonzero(a != b).flatten()
    results.append(dict(name=name, tested=g.numel(), mismatches=indices.numel(), examples=[
        dict(g=g[i].item(), u=u[i].item(), dy=dy[i].item(), native=a[i].item(), reference=b[i].item())
        for i in indices[:12]]))
print(json.dumps(results, indent=2))
