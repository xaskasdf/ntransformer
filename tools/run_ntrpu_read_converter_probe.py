"""Run a small, private XSIM probe of the existing AMD read converters.

The synthetic responder is not a model of HBM performance. Never run in a
qualified output directory: --output must not already exist.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from check_ntrpu_read_return import review as review_donor


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args):
    root = Path(__file__).resolve().parents[1]
    donor = args.donor.resolve()
    donor_review = review_donor(donor)
    assert donor_review['revision'] == 3
    source_paths = [
        root / 'fpga/tb/tb_ntrpu_read_converter_probe.sv',
        root / 'fpga/experimental/ntrpu_hbm_read_adapter.sv',
        root / 'tools/run_ntrpu_read_converter_probe.py',
        root / 'tools/check_ntrpu_read_converter_probe.py',
        root / 'tools/check_ntrpu_read_return.py',
    ]
    sources = {str(p.relative_to(root)): sha(p) for p in source_paths}
    original = json.loads((donor / 'result.json').read_text())
    adapter_name = 'fpga/experimental/ntrpu_hbm_read_adapter.sv'
    assert sources[adapter_name] == original['sources'][adapter_name]
    donor_files = json.loads((donor / 'FILES.json').read_text())
    donor_files.update({n: sha(donor / n) for n in ['result.json', 'FILES.json']})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    saved = output / 'sources'
    for p in source_paths:
        target = saved / p.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    for name in ('hbm_read_width', 'hbm_read_protocol'):
        for suffix in ('_wrapper.v', '_config.tsv'):
            shutil.copyfile(donor / 'suite' / (name + suffix), output / (name + suffix))
    (output / 'DONOR_REVIEW.json').write_text(json.dumps(donor_review, indent=2))
    adapter = (root / adapter_name).read_text()
    start = adapter.index('    hbm_read_width width_converter (')
    end = adapter.index('\nendmodule', start)
    instances = adapter[start:end]
    assert instances.count('width_converter (') == instances.count('protocol_converter (') == 1
    (output / 'read_converter_instances.vh').write_text(instances + '\n')
    shutil.copyfile(source_paths[0], output / source_paths[0].name)
    settings = args.settings.resolve()
    # Read the tool environment without printing credential-bearing variables.
    env_result = subprocess.run(
        ['bash', '-c', 'source "$1" >/dev/null && env -0', 'probe-env', str(settings)],
        capture_output=True, check=True, timeout=30)
    env = dict(item.decode().split('=', 1) for item in env_result.stdout.split(b'\0') if b'=' in item)
    env['RDI_DATADIR'] = str(settings.parent.parent / 'data')
    libraries = ['axi_infrastructure_v1_1_0', 'fifo_generator_v13_2_15',
                 'generic_baseblocks_v2_1_2', 'axi_data_fifo_v2_1_37',
                 'axi_register_slice_v2_1_37', 'axi_protocol_converter_v2_1_38',
                 'blk_mem_gen_v8_4_13', 'axi_dwidth_converter_v2_1_38',
                 'unisims_ver', 'unimacro_ver', 'secureip', 'xpm']
    commands = [
        ['xvlog', '--relax', '--sv', '-i', '.',
         'hbm_read_width_wrapper.v', 'hbm_read_protocol_wrapper.v', source_paths[0].name],
        ['xelab', '--relax', '--debug', 'off', '--mt', '2'] +
        [v for lib in libraries for v in ('-L', lib)] +
        ['--snapshot', 'converter_probe', 'work.tb_ntrpu_read_converter_probe'],
        ['xsim', 'converter_probe', '-runall'],
    ]
    cpus = [int(c) for c in args.cpus.split(',')]
    assert len(cpus) == 2 and len(set(cpus)) == 2
    assert set(cpus) <= os.sched_getaffinity(0)
    os.sched_setaffinity(0, cpus)
    os.nice(10)
    started = time.monotonic()
    steps = []
    for label, command in zip(('compile', 'elaborate', 'simulate'), commands):
        before = time.monotonic()
        with (output / (label + '.log')).open('w') as log:
            try:
                proc = subprocess.run(command, cwd=output, env=env,
                                      stdout=log, stderr=subprocess.STDOUT, timeout=180)
                code, timed_out = proc.returncode, False
            except subprocess.TimeoutExpired:
                code, timed_out = None, True
        steps.append(dict(stage=label, command=command, returncode=code,
                          timed_out=timed_out, seconds=round(time.monotonic()-before, 3)))
        if code != 0:
            break
    result = dict(experiment='read_converter_concurrency_probe',
                  status='EXPORTED_PENDING_REVIEW', sources=sources,
                  sources_after={str(p.relative_to(root)): sha(p) for p in source_paths},
                  donor=str(donor), donor_files=donor_files,
                  donor_unchanged=all(sha(donor/n) == h for n, h in donor_files.items()),
                  cpus=cpus, nice=10, threads=2, steps=steps,
                  seconds=round(time.monotonic()-started, 3), native_hbm=False,
                  synthesized=False, board_qualified=False)
    (output / 'result.json').write_text(json.dumps(result, indent=2))
    evidence = [p for p in output.iterdir() if p.is_file() and
                p.suffix in ('.sv', '.vh', '.v', '.tsv', '.log', '.json')]
    evidence += list(saved.rglob('*'))
    (output / 'FILES.json').write_text(json.dumps(
        {str(p.relative_to(output)).replace('\\', '/'): sha(p) for p in evidence if p.is_file()}, indent=2))
    print(json.dumps(dict(output=str(output), seconds=result['seconds'], steps=steps), indent=2))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--donor', type=Path, required=True, help='Reviewed revision-3 preflight output')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--settings', type=Path, required=True)
    ap.add_argument('--cpus', default='26,27')
    run(ap.parse_args())
