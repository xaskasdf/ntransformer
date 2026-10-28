"""Freeze revision 1 in a fresh private repository; preserve the qualified donor."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--donor', type=Path, required=True, help='Original RTL batch with OUTCOME.json and rtl2/repo')
    args = parser.parse_args()
    hashes = json.loads((args.donor / 'OUTCOME.json').read_text())['source_sha256']
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    for name, sha in hashes.items():
        source = args.donor / 'rtl2/repo' / name
        if hashlib.sha256(source.read_bytes()).hexdigest() != sha:
            raise ValueError('Donor source changed: ' + name)
        target = out / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    for name in ('ntrpu_bridge_async_fifo.sv', 'ntrpu_hbm_dma_bridge_lane.sv'):
        shutil.copyfile(ROOT / 'fpga/experimental/hbm_dma_rev1' / name, out / 'fpga/experimental' / name)
    bench = out / 'fpga/tb/tb_ntrpu_hbm_dma_bridge.sv'
    text = bench.read_text()
    anchor = '    initial begin\n        reset_case(0);'
    if text.count(anchor) != 1:
        raise ValueError('Unknown donor testbench')
    extra = (ROOT / 'fpga/tb/ntrpu_hbm_dma_rev1_cases.svh').read_text()
    text = text.replace(anchor, extra + '\n    initial begin\n'
                        '        if($test$plusargs("revision_only"))begin revision_tests();$finish;end\n'
                        '        reset_case(0);')
    bench.write_text(text)
    for name in ('tools/run_ntrpu_hbm_dma_rev1.py', 'tools/run_ntrpu_hbm_dma_cdc_probe.py',
                 'tools/ntrpu_hbm_dma_cdc_probe.tcl', 'tools/prepare_ntrpu_hbm_dma_rev1.py'):
        shutil.copyfile(ROOT / name, out / name)
    manifest = {p.relative_to(out).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in out.rglob('*') if p.is_file()}
    (out / 'PREPARED.json').write_text(json.dumps(dict(donor=hashes, prepared=manifest), indent=2) + '\n')
    print(json.dumps(dict(output=str(out), files=len(manifest))))


if __name__ == '__main__':
    main()
