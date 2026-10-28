#!/usr/bin/env python3
"""Compare graph accumulation with independent multi-tensor communication buckets."""
import argparse
import copy
import gc
import json
import os
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'),str(ROOT/'scripts')]
from ntransformer_training.gradients import GradientBuckets
from ntransformer_training.graphs import install_training_graph
from train_brandon_experiment import import_brandon,install
from analyze_verified_instructions import exact_state


def compare(args,full):
    TinyLlama,ModelConfig,PackedCorpus,_,_ = import_brandon(args.brandon_root)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config = dict(checkpoint['config'])
    if not full:
        config.update(n_layers=2,max_seq_len=64)
    seed = TinyLlama(ModelConfig(**config))
    if full:
        seed.load_state_dict(checkpoint['model_state_dict'])
        settings = checkpoint['settings']
        corpus = PackedCorpus(args.data_dir,'train',1024,settings['seed'])
        batches = [corpus.batch(settings['batch_size'],checkpoint['step']+i,0,m)
                   for i in range(2) for m in range(2)]
    else:
        batches = [(torch.randint(8192,(2,64)),torch.randint(8192,(2,64))) for _ in range(4)]
    del checkpoint
    reference = None
    records = []
    for capacity in (None,25,512):
        model = copy.deepcopy(seed).cuda()
        install(model,shared_projection_casts=True,direct_gqa=True,fused_rope=True,
                loss_mode='fused',fused_loss_labels=True)
        keys = tuple(model.state_dict())
        graph = install_training_graph(model,*(x.cuda() for x in batches[0]))
        buckets = GradientBuckets(graph,2,capacity) if capacity else None
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.95),weight_decay=.1,fused=True)
        if buckets:
            first = next(model.parameters())
            saved = first.grad
            first.grad = None
            try:
                buckets.begin_step()
            except RuntimeError as error:
                assert 'ownership' in str(error)
            else:
                raise AssertionError('Reset gradient ownership accepted')
            first.grad = saved
        observed = []
        for step in range(2):
            if buckets:
                buckets.begin_step()
                try:
                    buckets.finish()
                except RuntimeError as error:
                    assert 'microbatches' in str(error)
                else:
                    raise AssertionError('Premature reduction accepted')
            else:
                optimizer.zero_grad(set_to_none=False)
            losses = []
            gradients = []
            for x,y in batches[step*2:step*2+2]:
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = model(*graph.stage(x,y),targets_shifted=True,loss_only=True)[1]
                if buckets:
                    buckets.backward(loss/2)
                else:
                    (loss/2).backward()
                losses.append(loss.detach())
                gradients.append({n:p.grad.cpu().clone() for n,p in model.named_parameters()})
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            observed.append(dict(losses=[x.item() for x in losses],gradients=gradients,norm=norm.item()))
        state = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        assert tuple(model.state_dict()) == keys and model.output.weight is model.tok_embeddings.weight
        if reference is None:
            reference = observed,state,copy.deepcopy(optimizer.state_dict())
            # The optimizer snapshot must not retain GPU storage across variants.
            for value in reference[2]['state'].values():
                for key,tensor in value.items():
                    if isinstance(tensor,torch.Tensor):value[key]=tensor.cpu()
        else:
            for a,b in zip(observed,reference[0]):
                assert a['losses']==b['losses'] and a['norm']==b['norm']
                for ga,gb in zip(a['gradients'],b['gradients']):
                    for name in ga:
                        torch.testing.assert_close(ga[name],gb[name],rtol=0,atol=0,msg=name)
            assert exact_state(state,reference[1])
            assert exact_state(optimizer.state_dict(),reference[2])
            records.append(dict(full_model=full,bucket_mb=capacity,exact=True,
                                parameters=len(state),statistics=buckets.statistics()))
        del model,graph,buckets,optimizer,observed,state,loss,losses
        gc.collect()
        torch.cuda.empty_cache()
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('brandon-root','checkpoint','data-dir','output'):parser.add_argument('--'+name,required=True)
    args=parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    torch.manual_seed(20261015)
    result=dict(small=compare(args,False),full=compare(args,True),status='passed')
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
