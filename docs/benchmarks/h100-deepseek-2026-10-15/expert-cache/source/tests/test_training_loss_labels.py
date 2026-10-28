#!/usr/bin/env python3
"""Exact label fusion, Release validation and graph lifetime/accumulation checks."""
import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'scripts')]
from ntransformer_training.loss import linear_cross_entropy
from ntransformer_training.graphs import install_training_graph
from train_brandon_experiment import import_brandon, install
from test_training_loss import primitives, semantics, model_checks, difference


def label_layouts():
    records = []
    for shifted in (False, True):
        for strided in (False, True):
            h = torch.randn(2, 33, 64, device='cuda', requires_grad=True)
            w = torch.randn(8192, 64, device='cuda', requires_grad=True)
            labels = torch.randint(8192, (2, 66 if strided else 33), device='cuda')
            if strided:
                labels = labels[:, ::2]
            labels[:, :7] = -100
            values, gradients = [], []
            for enabled in (False, True):
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    loss = linear_cross_entropy(h, w, labels, targets_shifted=shifted, fused_labels=enabled)
                values.append(loss)
                gradients.append(torch.autograd.grad(loss, (h,w)))
            torch.testing.assert_close(*values, rtol=0, atol=0)
            for a,b in zip(*gradients):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            records.append(dict(shifted=shifted,strided=strided,exact=True))
    # Outside BF16 autocast, ordinary CE must retain its own dtype/reduction.
    for dtype in (torch.bfloat16, torch.float64):
        h = torch.randn(2, 4, 7, device='cuda', dtype=dtype, requires_grad=True)
        w = torch.randn(8192, 7, device='cuda', dtype=dtype, requires_grad=True)
        y = torch.randint(8192, (2, 4), device='cuda')
        a = linear_cross_entropy(h,w,y,fused_labels=True)
        b = F.cross_entropy(F.linear(h,w).view(-1,8192),y.view(-1))
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        for x,z in zip(torch.autograd.grad(a,(h,w)),torch.autograd.grad(b,(h,w))):
            torch.testing.assert_close(x,z,rtol=0,atol=0)
        records.append(dict(fallback_dtype=str(dtype),exact=True))
    return records


def invalid_child(kind):
    h = torch.randn(2, 4, 64, device='cuda', requires_grad=True)
    w = torch.randn(8192, 64, device='cuda', requires_grad=True)
    y = torch.zeros((2, 4), device='cuda', dtype=torch.int64)
    y[0,1] = -1 if kind == 'negative' else 8192
    with torch.autocast('cuda', dtype=torch.bfloat16):
        linear_cross_entropy(h,w,y,fused_labels=True)
    torch.cuda.synchronize()
    raise RuntimeError('Invalid target escaped validation')


def validation_checks():
    records = []
    for kind in ('negative','too-large'):
        result = subprocess.run([sys.executable,__file__,'--invalid-child',kind],
                                text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,timeout=90)
        assert result.returncode != 0 and 'device-side assert' in result.stdout, result.stdout
        records.append(dict(case=kind,release_device_assert=True,exit_code=result.returncode))
    return records


def graph_checks(args, full=False):
    TinyLlama, ModelConfig, PackedCorpus, _, _ = import_brandon(args.brandon_root)
    checkpoint = torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if full:
        seed = TinyLlama(ModelConfig(**checkpoint['config']))
        seed.load_state_dict(checkpoint['model_state_dict'])
        settings = checkpoint['settings']
        corpus = PackedCorpus(args.data_dir,'train',seed.config.max_seq_len,settings['seed'])
        batches = [corpus.batch(settings['batch_size'],checkpoint['step']+i,0,m)
                   for i in range(2) for m in range(2)]
    else:
        config = dict(checkpoint['config'],n_layers=2,max_seq_len=64)
        seed = TinyLlama(ModelConfig(**config))
        batches = [(torch.randint(8192,(2,64)),torch.randint(8192,(2,64))) for _ in range(4)]
    del checkpoint
    expected = None
    records = []
    for enabled in (False,True):
        model = copy.deepcopy(seed).cuda()
        install(model,shared_projection_casts=True,direct_gqa=True,fused_rope=True,
                loss_mode='fused',fused_loss_labels=True)
        keys = tuple(model.state_dict())
        graph = install_training_graph(model,*(x.cuda() for x in batches[0])) if enabled else None
        assert tuple(model.state_dict()) == keys
        assert model.output.weight is model.tok_embeddings.weight
        optimizer = torch.optim.AdamW(model.parameters(),lr=3e-5,betas=(.9,.95),weight_decay=.1,fused=True)
        observed = []
        # Two updates with two microsteps exercise replay after parameter writes,
        # None -> accumulated parameter gradients and deferred scalar readback.
        for step in range(2):
            optimizer.zero_grad(set_to_none=graph is None)
            losses = []
            first_gradients = None
            for x,y in batches[step*2:step*2+2]:
                inputs = graph.stage(x,y) if graph else (x.cuda(),y.cuda())
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    loss = model(*inputs,targets_shifted=True,loss_only=True)[1]
                (loss/2).backward()
                if first_gradients is None:
                    first_gradients = {n:p.grad.cpu().clone() for n,p in model.named_parameters()}
                losses.append(loss.detach())
            observed.append(dict(losses=[x.item() for x in losses],
                                 first_gradients=first_gradients,
                                 gradients={n:p.grad.cpu().clone() for n,p in model.named_parameters()}))
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
        state = {n:p.detach().cpu().clone() for n,p in model.named_parameters()}
        if expected is None:
            expected = observed,state
        else:
            for a,b in zip(observed,expected[0]):
                print(json.dumps(dict(graph_loss=a['losses'],reference_loss=b['losses'],
                    first_gradient_differences={n:difference(a['first_gradients'][n],b['first_gradients'][n])
                                                for n in a['gradients']},
                    accumulated_gradient_differences={n:difference(a['gradients'][n],b['gradients'][n])
                                                      for n in a['gradients']})),flush=True)
                assert a['losses'] == b['losses'], (a['losses'],b['losses'])
                for n in a['gradients']:
                    torch.testing.assert_close(a['first_gradients'][n],b['first_gradients'][n],rtol=0,atol=0,msg=n)
                    torch.testing.assert_close(a['gradients'][n],b['gradients'][n],rtol=0,atol=0,msg=n)
            for n in state:
                torch.testing.assert_close(state[n],expected[1][n],rtol=0,atol=0,msg=n)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                try:
                    model(*graph.stage(*batches[0]),targets_shifted=True,loss_only=True)
                except RuntimeError as error:
                    assert 'set_to_none=False' in str(error)
                else:
                    raise AssertionError('Missing accumulation storage accepted')
            for parameter in model.parameters():
                parameter.grad = torch.zeros_like(parameter)
            # A second live forward must fail before overwriting captured state.
            x,y = graph.stage(*batches[0])
            with torch.autocast('cuda',dtype=torch.bfloat16):
                loss = model(x,y,targets_shifted=True,loss_only=True)[1]
                try:
                    model(x,y,targets_shifted=True,loss_only=True)
                except RuntimeError as error:
                    assert 'backward' in str(error)
                else:
                    raise AssertionError('Second live graph forward accepted')
            loss.backward()
            try:
                graph.stage(batches[0][0][:,:8],batches[0][1][:,:8])
            except ValueError:
                pass
            else:
                raise AssertionError('Shape change accepted')
            model.eval()
            with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                a = model(x,y,targets_shifted=True,loss_only=True)[1]
                b = graph.original_forward(x,y,targets_shifted=True,loss_only=True)[1]
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            records.append(dict(full_model=full,exact=True,updates=2,accumulation=2,
                                parameter_gradients=len(state),graph=graph.statistics()))
        del model,optimizer,graph,observed,state
        # The graph owner and callable refer back to the model; collect cycles.
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('brandon-root','checkpoint','data-dir','output'):
        parser.add_argument('--'+name)
    parser.add_argument('--invalid-child')
    parser.add_argument('--small-only',action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('high')
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory=False
    torch.manual_seed(20261014)
    if args.invalid_child:
        invalid_child(args.invalid_child)
        return
    result = dict(primitives=primitives(),semantics=semantics(fused_labels=True),
                  labels=label_layouts(),validation=validation_checks())
    result['graph_small'] = graph_checks(args)
    if not args.small_only:
        result['model'] = model_checks(args,fused_labels=True)
        result['graph_full'] = graph_checks(args,full=True)
    result['status'] = 'passed'
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    main()
