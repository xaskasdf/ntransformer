"""Create a private shared-source experiment from the preserved native baseline.

The baseline checkpoint identifies provenance, not timing approval of this new
composition. Output must not exist; the donor is never edited.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ADDITIONS = (
    'fpga/experimental/ntrpu_peer_read_admission.sv',
    'fpga/experimental/ntrpu_peer_read_shared.sv',
    'fpga/tb/tb_ntrpu_peer_admission.sv',
    'fpga/tb/tb_ntrpu_peer_shared.sv',
    'fpga/tb/ntrpu_peer_shared_monitor.svh',
    'tools/prepare_ntrpu_peer_shared.py',
    'tools/run_ntrpu_peer_shared.py',
    'tools/check_ntrpu_peer_shared.py',
    'tools/ntrpu_peer_shared.tcl',
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(reference, output):
    reference, output = reference.resolve(), output.resolve()
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.is_relative_to(reference) or output.is_relative_to(root):
        raise ValueError('Use a new output directory outside the donor and repository')
    profile = json.loads((reference / 'NATIVE_PROFILE.json').read_text())
    before = {p.relative_to(reference).as_posix(): sha(p) for p in reference.rglob('*') if p.is_file()}
    for name, digest in profile['selected_rtl'].items():
        if before.get(name) != digest:
            raise ValueError('Reference RTL mismatch: ' + name)
    if not all((root / name).is_file() for name in ADDITIONS):
        raise ValueError('Missing shared-source experiment files')
    shutil.copytree(reference, output)
    for name in ADDITIONS:
        p = output / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, p)
    after = {p.relative_to(reference).as_posix(): sha(p) for p in reference.rglob('*') if p.is_file()}
    if before != after:
        raise ValueError('Donor changed during preparation; output is not qualified')
    result = dict(scope='One source, one lease, two fixed channels; functional simulation only',
                  reference_profile_sha256=sha(reference / 'NATIVE_PROFILE.json'),
                  reference_checkpoint_sha256=profile['selected_checkpoint_sha256'],
                  reference_sources=before,
                  additions={name: sha(output / name) for name in ADDITIONS},
                  physical_timing='UNTESTED', native_hbm='UNTESTED', pcie='NOT_INCLUDED')
    (output / 'SHARED_PROFILE.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('reference', type=Path)
    ap.add_argument('output', type=Path)
    args = ap.parse_args()
    print(json.dumps(prepare(args.reference, args.output)))
