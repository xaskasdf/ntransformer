"""Bounded local frame/join/guarded-DMA composition, with frozen source donors."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
BRIDGE = {
    'ntrpu_peer_read_pkg': '7d0b994ef8244710d4d2e8ed9ed4960685ad6a89ed5ffa76708439b2dcce603b',
    'ntrpu_bridge_async_fifo': 'be809fe998385e69a52eb2fa8687b5b50e52433a5b1a42694331efc07f1f0634',
    'ntrpu_hbm_dma_bridge_lane': '7e987b09ae1157251acd460b5511f041f59ab75076ecc62b4dcb0aa5a4926a24',
    'ntrpu_hbm_dma_bridge': 'b4a10f677311bfdf29b00a353bb0d83d6a4e5181ca228c1d5b5da34d86d3848a',
}
HBM = {
    'ntrpu_hbm_read_frames': '28863cc75ac91edf265540cdb8a6c02dc110620aad4a88e11b86f97506c14b2f',
    'ntrpu_peer_read_domain': '546a411b0c0a7a707075ecede72a879f9fcd54693e50b754df892b70d6d60c55',
}
GUARDED = {
    'fpga/rtl/ntrpu_xdma_local_master.sv': '5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc',
    'fpga/experimental/ntrpu_xdma_completion_guard.sv': '6d2cfe560b8d582c7edd1ed83b8b6dda24ff7809a587758a8f8caa2e42e3c89e',
    'fpga/experimental/ntrpu_hbm_xdma_guarded.sv': '32552bf0c620a464aeda5602bf47ae352cdcb51f63bbd10a252ab90ad35f62b2',
}
NEW_RTL = ['fpga/experimental/ntrpu_hbm_source_join.sv', 'fpga/experimental/ntrpu_hbm_xdma_source.sv',
           'fpga/experimental/ntrpu_xdma_completion_guard_held.sv', 'fpga/experimental/ntrpu_hbm_xdma_guarded_held.sv',
           'fpga/tb/tb_ntrpu_hbm_xdma_source.sv']
SUPPORT = ['tools/run_ntrpu_hbm_source_join.py', 'tools/check_ntrpu_hbm_source_join.py']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, indent=2)+'\n', encoding='utf-8')


def run(output, toolchain, bridge, hbm, seeds):
    output.mkdir(parents=True, exist_ok=False)
    paths = {}
    pins = dict(GUARDED)
    for donor, group in [(bridge, BRIDGE), (hbm, HBM)]:
        for name, digest in group.items():
            name = 'fpga/experimental/'+name+'.sv'
            paths[name] = donor/name
            pins[name] = digest
    paths.update({name: ROOT/name for name in [*GUARDED, *NEW_RTL, *SUPPORT]})
    for name, digest in pins.items():
        if sha(paths[name]) != digest:
            raise ValueError('Frozen donor mismatch: '+name)
    sources = {name: sha(path) for name, path in paths.items()}
    for name, path in paths.items():
        target = output/'repo'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    chain = {p.relative_to(toolchain).as_posix(): sha(p) for p in toolchain.rglob('*') if p.is_file()}
    bins = toolchain/'mingw64/bin'
    env = dict(os.environ)
    windows = Path(env.get('SystemRoot', 'C:/Windows'))
    env['PATH'] = os.pathsep.join(map(str, [bins, windows/'System32', windows]))
    result = dict(status='FAILED', sources=sources, toolchain=chain, runs=[],
                  native_hbm=False, native_xdma=False, mapped_regions=False,
                  timing_qualified=False, board_qualified=False)
    start = time.monotonic()

    def execute(command, cwd, log):
        job = subprocess.run(list(map(str, command)), cwd=cwd, env=env, capture_output=True, text=True, timeout=30)
        text = job.stdout+job.stderr
        (cwd/log).write_text(text, encoding='utf-8')
        if job.returncode:
            raise ValueError(f'{log}: exit {job.returncode}')
        return text

    try:
        execute([bins/'iverilog.exe', '-V'], output, 'compiler-version.log')
        execute([bins/'vvp.exe', '-V'], output, 'simulator-version.log')
        rtl = [name for name in paths if name.endswith('.sv')]
        log = execute([bins/'iverilog.exe', '-g2012', '-Wall', '-s', 'tb_ntrpu_hbm_xdma_source',
                       '-o', output/'source.vvp', *[output/'repo'/name for name in rtl]], output, 'compile.log')
        warnings = [line for line in log.splitlines() if 'warning:' in line]
        result['compile_warnings'] = warnings
        if any('is sensitive to all' not in line for line in warnings):
            raise ValueError('Unexpected compile warning')
        from check_ntrpu_hbm_source_join import review
        for seed in range(seeds):
            case = output/f'seed-{seed:02}'
            case.mkdir()
            execute([bins/'vvp.exe', output/'source.vvp', f'+seed={seed}'], case, 'simulation.log')
            checked = review((case/'simulation.log').read_text())
            save(case/'REVIEW.json', checked)
            result['runs'].append(dict(seed=seed, review=checked))
        result['status'] = 'HBM_SOURCE_JOIN_LOCAL_PASS'
        result['scenarios'] = sum(len(r['review']['cases']) for r in result['runs'])
    except (ValueError, AssertionError, OSError, subprocess.TimeoutExpired) as error:
        result['error'] = str(error)
    result['seconds'] = round(time.monotonic()-start, 6)
    result['sources_after'] = {name: sha(path) for name, path in paths.items()}
    result['toolchain_after'] = {p.relative_to(toolchain).as_posix(): sha(p) for p in toolchain.rglob('*') if p.is_file()}
    if result['sources_after'] != sources or result['toolchain_after'] != chain:
        result['status'] = 'FAILED'
        result['error'] = 'Sources/toolchain changed during run'
    save(output/'result.json', result)
    save(output/'FILES.json', {p.relative_to(output).as_posix(): sha(p) for p in sorted(output.rglob('*')) if p.is_file()})
    return result


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output', type=Path)
    ap.add_argument('--toolchain', type=Path, required=True)
    ap.add_argument('--bridge-snapshot', type=Path, required=True)
    ap.add_argument('--hbm-snapshot', type=Path, required=True)
    ap.add_argument('--seeds', type=int, default=8)
    a = ap.parse_args()
    if not 1 <= a.seeds <= 32:
        ap.error('--seeds must be 1..32')
    r = run(a.output.resolve(), a.toolchain.resolve(), a.bridge_snapshot.resolve(), a.hbm_snapshot.resolve(), a.seeds)
    print(json.dumps({k: r[k] for k in ('status', 'seconds', 'error', 'scenarios') if k in r}))
    raise SystemExit(0 if r['status'] == 'HBM_SOURCE_JOIN_LOCAL_PASS' else 1)
