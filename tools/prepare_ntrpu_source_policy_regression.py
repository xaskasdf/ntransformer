"""Prepare a fresh private replay of existing guard regressions with held grants."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FILES = ['fpga/rtl/ntrpu_xdma_local_master.sv',
         'fpga/experimental/ntrpu_hbm_xdma_guarded.sv',
         'fpga/experimental/ntrpu_xdma_completion_guard.sv',
         'fpga/experimental/ntrpu_xdma_completion_guard_held.sv',
         'fpga/tb/tb_ntrpu_hbm_xdma_guarded.sv',
         'fpga/tb/tb_ntrpu_xdma_completion_guard.sv',
         *['tools/'+name+'.py' for name in (
             'run_ntrpu_hbm_xdma_guarded_local', 'check_ntrpu_hbm_xdma_guarded',
             'check_ntrpu_hbm_xdma_native', 'check_ntrpu_xdma_stop_inventory_r4',
             'ntrpu_pcie_vcd', 'ntrpu_pcie_straddle', 'run_ntrpu_xdma_completion_guard',
             'check_ntrpu_xdma_completion_guard', 'prepare_ntrpu_source_policy_regression')]]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(out):
    out.mkdir(parents=True, exist_ok=False)
    before = {name: sha(ROOT/name) for name in FILES}
    for name in FILES:
        target = out/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    base = out/'fpga/experimental/ntrpu_xdma_completion_guard.sv'
    held = out/'fpga/experimental/ntrpu_xdma_completion_guard_held.sv'
    reference = out/'GUARD_REFERENCE.sv'
    shutil.copy2(base, reference)
    text = held.read_text(encoding='utf-8')
    old = 'module ntrpu_xdma_completion_guard_held #('
    if text.count(old) != 1:
        raise ValueError('Guard declaration mismatch')
    base.write_text(text.replace(old, 'module ntrpu_xdma_completion_guard #('), encoding='utf-8', newline='\n')
    after = {name: sha(out/name) for name in FILES}
    assert {name for name in FILES if before[name] != after[name]} == {'fpga/experimental/ntrpu_xdma_completion_guard.sv'}
    assert {name: sha(ROOT/name) for name in FILES} == before
    (out/'POLICY_VARIANT.json').write_text(json.dumps(dict(
        original_sources=before, prepared_sources=after, reference_sha256=sha(reference),
        delta='Exactly a module-name alias of ntrpu_xdma_completion_guard_held; existing benches and checkers unchanged.',
        native_ip_qualified=False), indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    a = ap.parse_args()
    prepare(a.output.resolve())
