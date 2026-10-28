"""Freeze the revision-1 donor and replace only its FIFO with registered output."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--donor', type=Path, required=True)
    args = parser.parse_args()
    donor = args.donor / 'rtl1/repo'
    before = json.loads((args.donor / 'rtl1/output/result.json').read_text())['sources']
    for name, sha in before.items():
        if hashlib.sha256((donor / name).read_bytes()).hexdigest() != sha:
            raise ValueError('Changed donor: ' + name)
    out = args.output.resolve()
    shutil.copytree(donor, out)
    shutil.copyfile(ROOT / 'fpga/experimental/hbm_dma_rev2/ntrpu_bridge_async_fifo.sv',
                    out / 'fpga/experimental/ntrpu_bridge_async_fifo.sv')
    for name in ('fpga/tb/tb_ntrpu_bridge_fifo_rev2.sv', 'tools/run_ntrpu_hbm_dma_rev2.py',
                 'tools/prepare_ntrpu_hbm_dma_rev2.py'):
        shutil.copyfile(ROOT / name, out / name)
    manifest = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob('*') if p.is_file()}
    (out / 'PREPARED_REV2.json').write_text(json.dumps(dict(donor=before, prepared=manifest), indent=2) + '\n')
    print(json.dumps(dict(output=str(out), files=len(manifest))))


if __name__ == '__main__':
    main()
