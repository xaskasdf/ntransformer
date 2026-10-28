#!/usr/bin/env python3
"""Attribute saved continuation differences by tensor, layer and reference noise."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import torch


def read(path):
    return json.loads(path.read_text())


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initial', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--candidate', action='append', required=True, help='LABEL=DIRECTORY')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    reference = Path(args.reference)
    initial = torch.load(args.initial, map_location='cpu', weights_only=False)['model_state_dict']
    expected = torch.load(reference/'last.pt', map_location='cpu', weights_only=False)['model_state_dict']
    summary = read(reference/'summary.json')
    metrics = [json.loads(line) for line in (reference/'metrics.jsonl').read_text().splitlines()]
    result = dict(reference=str(reference), initial_sha256=sha(Path(args.initial)),
                  reference_sha256=sha(reference/'last.pt'), cases={})
    for value in args.candidate:
        label, directory = value.split('=', 1)
        directory = Path(directory)
        candidate = torch.load(directory/'last.pt', map_location='cpu', weights_only=False)['model_state_dict']
        other_summary = read(directory/'summary.json')
        other_metrics = [json.loads(line) for line in (directory/'metrics.jsonl').read_text().splitlines()]
        assert summary['batch_sha256'] == other_summary['batch_sha256']
        assert initial.keys() == expected.keys() == candidate.keys()
        assert [r['step'] for r in metrics] == [r['step'] for r in other_metrics]
        tensors, seen = [], {}
        for name, tensor in expected.items():
            a, b, source = tensor.double(), candidate[name].double(), initial[name].double()
            assert torch.isfinite(a).all() and torch.isfinite(b).all()
            storage = (tensor.untyped_storage().data_ptr(), tensor.storage_offset(), tuple(tensor.shape))
            record = dict(name=name, elements=tensor.numel(), mismatches=int((a != b).sum()),
                          error_squared=(a-b).square().sum().item(),
                          update_squared=(a-source).square().sum().item(),
                          weight_squared=a.square().sum().item(), max_abs=(a-b).abs().max().item(),
                          tied_alias=seen.get(storage))
            record['relative_update_l2'] = math.sqrt(record['error_squared']/max(record['update_squared'],1e-300))
            seen[storage] = name
            tensors.append(record)
        def aggregate(rows):
            error = sum(r['error_squared'] for r in rows)
            update = sum(r['update_squared'] for r in rows)
            weight = sum(r['weight_squared'] for r in rows)
            return dict(tensors=len(rows), elements=sum(r['elements'] for r in rows),
                        differing_tensors=sum(r['mismatches'] > 0 for r in rows),
                        mismatches=sum(r['mismatches'] for r in rows), max_abs=max(r['max_abs'] for r in rows),
                        error_squared=error, update_squared=update,
                        relative_update_l2=math.sqrt(error/max(update,1e-300)),
                        relative_weight_l2=math.sqrt(error/max(weight,1e-300)))
        unique = [r for r in tensors if r['tied_alias'] is None]
        layers, families = defaultdict(list), defaultdict(list)
        for row in unique:
            name = row['name']
            pieces = name.split('.')
            layers['.'.join(pieces[:2]) if pieces[0]=='layers' else pieces[0]].append(row)
            families['.'.join(pieces[2:-1]) if pieces[0]=='layers' else pieces[0]].append(row)
        comparison = dict(directory=str(directory), checkpoint_sha256=sha(directory/'last.pt'),
                          state_dict=aggregate(tensors), unique_parameters=aggregate(unique),
                          layers={k:aggregate(v) for k,v in layers.items()},
                          families={k:aggregate(v) for k,v in families.items()},
                          tensors=sorted(tensors, key=lambda r:r['error_squared'],reverse=True), metrics={})
        for key in ('loss', 'grad_norm'):
            comparison['metrics'][key] = dict(
                first_difference=next((dict(step=a['step'], reference=a[key],candidate=b[key])
                                       for a,b in zip(metrics,other_metrics) if a[key]!=b[key]),None),
                max_abs=max(abs(a[key]-b[key]) for a,b in zip(metrics,other_metrics)))
        comparison['validation_delta'] = other_summary['final_validation']['mean']-summary['final_validation']['mean']
        result['cases'][label] = comparison
        Path(args.output).write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(dict(case=label,**comparison['unique_parameters'],metrics=comparison['metrics'])),flush=True)


if __name__ == '__main__':
    main()
