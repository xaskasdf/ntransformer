import ctypes
import json
from pathlib import Path
import torch

root=Path('/home/ubuntu/ntransformer-h100/training/rope-rounding-20261014-r2')
lib=ctypes.CDLL(str(root/'probe.so'))
lib.run_probe.argtypes=[ctypes.c_void_p]*3+[ctypes.c_int64,ctypes.c_int,ctypes.c_void_p]
lib.run_probe.restype=ctypes.c_int
torch.manual_seed(20261014)
torch.set_num_threads(2)
records=[]
for dtype in (torch.float32,torch.bfloat16):
    for scale in (1.,1e-5,1e-10):
        x=(torch.randn(1000003,2,device='cuda')*scale).to(dtype).float()
        angle=torch.randn(1000003,device='cuda')*100
        f=torch.polar(torch.ones_like(angle),angle)
        for conjugate in (False,True):
            freq=f.conj().resolve_conj() if conjugate else f
            expected=torch.view_as_real(torch.view_as_complex(x)*freq)
            out=torch.empty_like(x)
            for mode in range(5):
                assert lib.run_probe(out.data_ptr(),x.data_ptr(),freq.data_ptr(),x.shape[0],mode,
                                     torch.cuda.current_stream().cuda_stream)==0
                records.append(dict(dtype=str(dtype),scale=scale,conjugate=conjugate,mode=mode,
                    fp32_mismatches=int((out!=expected).sum()),
                    output_mismatches=int((out.to(dtype)!=expected.to(dtype)).sum())))
(root/'results.json').write_text(json.dumps(records,indent=2)+'\n')
print(json.dumps({str(m):sum(r['fp32_mismatches'] for r in records if r['mode']==m) for m in range(5)}),flush=True)
