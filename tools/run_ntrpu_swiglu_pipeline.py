"""Run a pragma-only HLS scheduling experiment in a new, fully copied tree."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('activation', 'quantize', 'both'), required=True)
    parser.add_argument('--tool', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    staged = output / 'staged'
    files = ['scripts/run_ntrpu_hls.py', 'tests/test_ntrpu_hls_swiglu_block.cpp']
    for kernel in ('swiglu_q8_block', 'swiglu', 'quantize_q8_0'):
        files += [f'fpga/hls/{kernel}/ntrpu_{kernel}{suffix}' for suffix in ('.h', '.cpp')]
    files.append('fpga/hls/swiglu_q8_block/run_hls.tcl')
    sha = lambda data: hashlib.sha256(data).hexdigest()
    originals = {}
    for name in files:
        target = staged / name
        target.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / name).read_bytes()
        target.write_bytes(data)
        originals[name] = sha(data)
    edits = {}
    if args.variant in ('activation', 'both'):
        name = 'fpga/hls/swiglu/ntrpu_swiglu.cpp'
        edits[name] = rb'(for \(uint32_t i = 0; i < elements; \+\+i\) \{\r?\n)#pragma HLS PIPELINE off(?=\r?\n        double g = 0, u = 0;)'
    if args.variant in ('quantize', 'both'):
        name = 'fpga/hls/quantize_q8_0/ntrpu_quantize_q8_0.cpp'
        edits[name] = rb'(for \(uint32_t lane = 0; lane < BLOCK_ELEMENTS; \+\+lane\) \{\r?\n)#pragma HLS PIPELINE off(?=\r?\n            const uint32_t column)'
    patches = {}
    for name, pattern in edits.items():
        path = staged / name
        original = path.read_bytes()
        candidate, count = re.subn(pattern, rb'\g<1>#pragma HLS PIPELINE II=1', original)
        assert count == 1 and candidate != original, name
        # Arithmetic tokens, all other directives and line endings stay intact.
        reverse = pattern.replace(b'PIPELINE off', b'PIPELINE II=1')
        restored, undo_count = re.subn(reverse, rb'\g<1>#pragma HLS PIPELINE off', candidate)
        assert undo_count == 1 and restored == original, name
        path.write_bytes(candidate)
        patches[name] = dict(before=sha(original), after=sha(candidate), replacements=count,
                             change='Selected loop only: PIPELINE off -> PIPELINE II=1')
    final = {name: sha((staged / name).read_bytes()) for name in files}
    assert all(final[name] == originals[name] for name in files if name not in edits)
    runner = staged / 'scripts/run_ntrpu_hls.py'
    command = [sys.executable, str(runner), '--kernel', 'swiglu_q8_block',
               '--part', 'xcvu5p-flva2104-2-i', '--clock-ns', '5', '--stage', 'cosim',
               '--tool', args.tool, '--output-root', str(output / 'hls')]
    if args.dry_run:
        command.append('--dry-run')
    record = dict(variant=args.variant,original_source_sha256=originals,
                  staged_source_sha256=final,patches=patches,command=command,
                  scope='Scheduling experiment only; arithmetic and baseline repository unchanged. Qualification requires numerical/RTL review.')
    (output / 'pipeline_source_patch.json').write_text(json.dumps(record, indent=2)+'\n', encoding='utf-8')
    result = subprocess.run(command, check=False)
    return result.returncode


if __name__ == '__main__':
    raise SystemExit(main())
