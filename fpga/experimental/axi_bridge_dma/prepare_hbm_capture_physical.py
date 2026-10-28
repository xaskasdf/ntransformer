"""Prepare a fresh full-board synthesis using the sealed response-only RTL delta."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def prepare(donor, unit, output):
    if output.exists():
        raise ValueError('Fresh output required')
    here = Path(__file__).resolve().parent
    proof = json.loads((unit/'PREPARED.json').read_text())
    if sha(donor/'ntrpu_hbm_read_adapter.sv') != proof['donor_adapter_sha256']:
        raise ValueError('Board adapter differs from tested unit donor')
    candidate = unit/'ntrpu_hbm_response_capture_queue.sv'
    if sha(candidate) != proof['files'][candidate.name]['sha256']:
        raise ValueError('Tested candidate hash differs')
    adapter = (donor/'ntrpu_hbm_read_adapter.sv').read_text()
    binding = 'ntrpu_hbm_read_response_queue #(.WIDTH(265)) response_cut ('
    if adapter.count(binding) != 1:
        raise ValueError('Response-only binding changed')
    replacement = 'ntrpu_hbm_response_capture_queue #(.WIDTH(265)) response_cut ('
    changed = adapter.replace(binding, replacement, 1)+'\n'+candidate.read_text()
    output.mkdir(parents=True)
    original = {}
    for p in donor.iterdir():
        if p.is_file():
            shutil.copy2(p, output/p.name); original[p.name] = sha(p)
    (output/'ntrpu_hbm_read_adapter.sv').write_text(changed, newline='\n')
    entry = output/'elaborate_hbm_native_join.tcl'
    text = entry.read_text()
    anchor = '        write_checkpoint $out/synth.dcp'
    if text.count(anchor) != 1:
        raise ValueError('Synthesis checkpoint anchor changed')
    entry.write_text(text.replace(anchor, anchor+'\n        source $source/audit_hbm_capture_physical.tcl\n        cp_probe $out/response_capture 0', 1), newline='\n')
    for name in ('audit_hbm_capture_physical.tcl', 'prepare_hbm_capture_physical.py'):
        shutil.copy2(here/name, output/name)
    # communicate waits for stdout EOF from tool AND inherited helper writers.
    # This avoids sealing a log before a late Vivado helper exit message arrives.
    runner = (donor/'run_hbm_native_join.py').read_text()
    begin = runner.index("    with (root / 'tool.log').open('w') as log:")
    end = runner.index('    unchanged = ', begin)
    block = '''    child = subprocess.Popen(['bash', '-c', 'source ' + shlex.quote(str(args.settings)) + ' && exec ' + shlex.join(argv)],
        cwd=work, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    save(root / 'own_process.json', dict(pid=child.pid, pgid=child.pid, timeout=1800, stage=args.stage))
    try:
        captured, _ = child.communicate(timeout=1800)
    except subprocess.TimeoutExpired:
        expired = True
        try: os.killpg(child.pid, signal.SIGTERM)
        except ProcessLookupError: pass
        try: captured, _ = child.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try: os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError: pass
            captured, _ = child.communicate()
    code = child.returncode
    (root / 'tool.log').write_bytes(captured)
'''
    (output/'run_hbm_native_join.py').write_text(runner[:begin]+block+runner[end:], newline='\n')
    files = {p.name:sha(p) for p in output.iterdir() if p.is_file()}
    record = dict(status='CAPTURE_FULL_BOARD_PREPARED', donor_files=original, prepared_files=files,
                  tested_candidate_sha256=sha(candidate), response_only_binding=True,
                  changed_original_files=sorted(n for n,h in original.items() if files[n]!=h),
                  full_rtl_synthesis_requested=True, physical_timing_qualified=False,
                  timeout_seconds=1800, stdout_eof_before_archive=True)
    if record['changed_original_files'] != ['elaborate_hbm_native_join.tcl', 'ntrpu_hbm_read_adapter.sv', 'run_hbm_native_join.py']:
        raise ValueError('Unexpected original file change')
    (output/'CAPTURE_PHYSICAL_BINDING.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('donor_files','prepared_files')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('donor','unit','output'): p.add_argument(key,type=Path)
    a = p.parse_args(); prepare(a.donor.resolve(),a.unit.resolve(),a.output.resolve())
