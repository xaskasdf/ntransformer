import fcntl
import json
import os
import sys
from contextlib import ExitStack
from pathlib import Path

os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
root=Path('/home/ubuntu/ntransformer-h100')
stage=Path('/lambda/nfs/btiny/brandon-mini/stages/stage1-v1')
with ExitStack() as stack:
    for directory in (stage,stage.parent/'behavior-v1'):
        lock=stack.enter_context((directory/'pipeline.lock').open('a'))
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    import subprocess
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend,sdpa_kernel
    sys.path.insert(0,str(root/'source/scripts'))
    from train_brandon_experiment import import_brandon
    from probe_training_differences import differences
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    TinyLlama,Config,Corpus,_,_=import_brandon(stage.parent.parent/'sources/stage1-v1')
    checkpoint=torch.load(stage/'pretrain/last.pt',map_location='cpu',weights_only=False)
    model=TinyLlama(Config(**checkpoint['config'])).cuda()
    model.load_state_dict(checkpoint['model_state_dict'])
    data=Corpus(stage/'data','train',1024,checkpoint['settings']['seed'])
    batch=[t.cuda() for t in data.batch(16,checkpoint['step'],0,0)]
    original=F.scaled_dot_product_attention
    captured={}
    calls=[0]
    def intercept(*values,**options):
        output=original(*values,**options)
        if calls[0]==29:
            captured['inputs']=[t.detach().clone() for t in values[:3]]
            captured['options']=dict(options)
            def upstream(grad):
                captured['upstream']=grad.detach().clone()
                return grad
            output.register_hook(upstream)
        calls[0]+=1
        return output
    F.scaled_dot_product_attention=intercept
    with torch.autocast('cuda',dtype=torch.bfloat16):
        loss=model(*batch,targets_shifted=True)[1]
    loss.backward()
    F.scaled_dot_product_attention=original
    del model,loss,checkpoint
    torch.cuda.empty_cache()
    result=dict(layer=29,input_shapes=[list(v.shape) for v in captured['inputs']],
                upstream='Actual layer-29 pretraining gradient; fixed across all replays',runs=[])
    for backend,deterministic in [(SDPBackend.CUDNN_ATTENTION,False),(SDPBackend.FLASH_ATTENTION,False),
                                  (SDPBackend.FLASH_ATTENTION,True)]:
        torch.use_deterministic_algorithms(deterministic,warn_only=False)
        reference=None
        for repeat in range(10):
            inputs=[t.detach().requires_grad_() for t in captured['inputs']]
            with sdpa_kernel(backend):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=original(*inputs,**captured['options'])
                grads=torch.autograd.grad(output,inputs,captured['upstream'])
            actual={k:t.cpu() for k,t in zip(('q','k','v'),grads)}
            record=dict(backend=str(backend),deterministic=deterministic,repeat=repeat)
            if reference is None:
                reference=actual
            else:
                record['difference']=differences(actual,reference)
            result['runs'].append(record)
    destination=root/'training/difference-review-20261013/real-attention.json'
    destination.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps([dict(backend=r['backend'],deterministic=r['deterministic'],repeat=r['repeat'],
                          mismatches=r.get('difference',{}).get('mismatches')) for r in result['runs']],indent=2))
