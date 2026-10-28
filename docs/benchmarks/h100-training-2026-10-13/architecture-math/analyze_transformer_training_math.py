#!/usr/bin/env python3
"""Small NumPy checks for training algebra; no model-quality or GPU-speed claims."""
import argparse
import json
from pathlib import Path

import numpy as np


def norm(x, eps=1e-5):
    return x / np.sqrt(np.mean(x*x) + eps)


def norm_jacobian(x, eps=1e-5):
    radius = np.sqrt(np.mean(x*x) + eps)
    return np.eye(x.size)/radius - np.outer(x, x)/(x.size*radius**3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    rng = np.random.default_rng(20261014)
    result = dict(scope='Small algebra/rounding probes; not trained Transformer results',
                  seed=20261014, numpy=np.__version__)

    # Check a linear repeat operator's adjoint before reasoning about GQA kernels.
    compact, upstream = rng.normal(size=(3,5,4)), rng.normal(size=(9,5,4))
    repeated = np.repeat(compact, 3, axis=0)
    adjoint = upstream.reshape(3,3,5,4).sum(axis=1)
    error = abs(np.vdot(repeated, upstream) - np.vdot(compact, adjoint))
    assert error < 1e-12
    result['gqa_adjoint_absolute_error'] = float(error)

    x, probe = rng.normal(size=8), rng.normal(size=8)
    analytic = norm_jacobian(x) @ probe
    numeric = (norm(x+1e-6*probe)-norm(x-1e-6*probe))/(2e-6)
    error = np.max(np.abs(analytic-numeric))
    assert error < 1e-8
    result['rmsnorm_jvp_max_error'] = float(error)

    # Streaming exact-real cross entropy and a second-pass gradient calculation.
    h, w = rng.normal(size=(7,8)), rng.normal(size=(13,8))
    target = rng.integers(0,13,size=7)
    logits = h @ w.T
    maximum = logits.max(axis=1)
    lse = maximum + np.log(np.exp(logits-maximum[:,None]).sum(axis=1))
    m, total = np.full(7,-np.inf), np.zeros(7)
    for start in range(0,13,4):
        block = h @ w[start:start+4].T
        new_m = np.maximum(m,block.max(axis=1))
        total = np.exp(m-new_m)*total + np.exp(block-new_m[:,None]).sum(axis=1)
        m = new_m
    streamed_lse = m+np.log(total)
    loss = np.mean(lse-logits[np.arange(7),target])
    streamed_loss = np.mean(streamed_lse-np.sum(h*w[target],axis=1))
    dz = np.exp(logits-lse[:,None])
    dz[np.arange(7),target] -= 1
    dz /= 7
    dh, dw = np.zeros_like(h), np.zeros_like(w)
    for start in range(0,13,4):
        stop = min(start+4,13)
        block = np.exp(h@w[start:stop].T-streamed_lse[:,None])
        rows = np.flatnonzero((target>=start)&(target<stop))
        block[rows,target[rows]-start] -= 1
        block /= 7
        dh += block @ w[start:stop]
        dw[start:stop] = block.T @ h
    errors = dict(loss=float(abs(loss-streamed_loss)),
                  dh=float(np.max(np.abs(dh-dz@w))),
                  dw=float(np.max(np.abs(dw-dz.T@h))))
    assert max(errors.values()) < 1e-12
    result['streamed_cross_entropy'] = errors

    # A parallel block omits a real first-order interaction from the serial block.
    wa, wf = rng.normal(scale=.2,size=(8,8)), rng.normal(scale=.2,size=(8,8))
    a = np.tanh(wa @ norm(x))
    f = np.tanh(wf @ norm(x))
    jf = (1-f*f)[:,None] * (wf @ norm_jacobian(x))
    records = []
    for alpha in (.1,.05,.025,.0125):
        serial = x+alpha*a+np.tanh(wf@norm(x+alpha*a))
        parallel = x+alpha*a+f
        discrepancy = serial-parallel
        remainder = discrepancy-alpha*(jf@a)
        records.append(dict(alpha=alpha, discrepancy_l2=float(np.linalg.norm(discrepancy)),
                            first_order_remainder_l2=float(np.linalg.norm(remainder))))
    ratios = [records[i]['first_order_remainder_l2']/records[i+1]['first_order_remainder_l2']
              for i in range(len(records)-1)]
    assert all(3.5 < ratio < 4.5 for ratio in ratios)
    result['parallel_block_taylor'] = dict(samples=records, halving_remainder_ratios=ratios)

    def bf16_round(value):
        value = np.asarray(value, dtype=np.float32)
        bits = value.view(np.uint32)
        return ((bits+np.uint32(0x7fff)+((bits>>16)&1)) & np.uint32(0xffff0000)).view(np.float32)

    # BF16 variant: branch functions use FP32; each scaled branch output and
    # residual state is rounded to BF16 before the next operation.
    # This is a controlled coupling toy, not an emulation of a full Transformer.
    matrices = rng.normal(scale=.2, size=(30,2,8,8))
    initial = rng.normal(size=(2,8))
    reversible = []
    for name, quantize, dtype in (('fp64',np.float64,np.float64),
                                 ('fp32',np.float32,np.float32),
                                 ('bf16_state',bf16_round,np.float32)):
        weights = matrices.astype(dtype)
        for scale in (.1,1.):
            states = [(quantize(initial[0]),quantize(initial[1]))]
            for a_w,f_w in weights:
                x1,x2 = states[-1]
                y1 = quantize(x1+quantize(scale*np.tanh(a_w@norm(x2))))
                y2 = quantize(x2+quantize(scale*np.tanh(f_w@norm(y1))))
                states.append((y1,y2))
            for interval in (30,5,1):
                worst, worst_relative = 0.,0.
                for begin in range(0,30,interval):
                    end = min(begin+interval,30)
                    y1,y2 = states[end]
                    for layer in reversed(range(begin,end)):
                        a_w,f_w = weights[layer]
                        x2 = quantize(y2-quantize(scale*np.tanh(f_w@norm(y1))))
                        x1 = quantize(y1-quantize(scale*np.tanh(a_w@norm(x2))))
                        actual = np.array((x1,x2),dtype=np.float64)
                        expected = np.array(states[layer],dtype=np.float64)
                        delta = actual-expected
                        worst = max(worst,float(np.max(np.abs(delta))))
                        worst_relative = max(worst_relative,float(np.linalg.norm(delta)/np.linalg.norm(expected)))
                        y1,y2 = x1,x2
                reversible.append(dict(state=name,residual_scale=scale,anchor_interval=interval,
                                       worst_abs=worst,worst_relative_l2=worst_relative))
    assert max(r['worst_abs'] for r in reversible if r['state']=='fp64') < 1e-10
    assert any(r['worst_abs'] > 0 for r in reversible if r['state']=='bf16_state')
    result['reversible_reconstruction'] = reversible
    result['status'] = 'passed'
    Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
