"""Review a backed DDR capture; never acquire or modify the Vivado queue."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import subprocess
import sys
import tarfile
import time
import zipfile

import review_ntrpu_ddr_activity_identity_r1 as detail


COORDINATOR = 'ddr-activity-vivado-r3-coordinator.json'
PACKAGE = 'ddr-activity-vivado-r3-package.json'
DRIVER_SHA = 'e4a625fbe0062196a7a2aba79ad92a39c2ec75df4aee5751685a80580479b94a'
PACKAGE_SHA = 'a4949e5bd65ed6ffa0be647b17450ff50b53d2f600ecf239fd4e131aa8b76f64'
REVIEWER_SHA = 'd52d3add12fbd5f5ea2620ceec131fc4bc4f6efc73307414243e2d4f92c90d93'
TERMINAL = {'activity_capture_backed_requires_review', 'diagnostic_backed_requires_inspection'}
WAITING = {'waiting_for_projected_root_bram_review_and_backup', 'running', 'reviewing', 'sealing'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest_stream(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1024 * 1024), b''):
        h.update(block)
    return h.hexdigest()


def digest(path):
    with Path(path).open('rb') as stream:
        return digest_stream(stream)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')


def readiness(state):
    require(state.get('script_sha256') == DRIVER_SHA, 'Unexpected capture driver')
    require(state.get('package_sha256') == PACKAGE_SHA, 'Unexpected capture configuration')
    if state.get('status') in TERMINAL:
        return True
    require(state.get('status') in WAITING, 'Capture stopped or unknown state; inspect without relaunch')
    return False


def safe_name(name):
    require(isinstance(name, str) and name and '\\' not in name and ':' not in name,
            'Nonportable archive member')
    p = PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and p.as_posix() == name,
            'Unsafe archive member')
    return name


def validate_terminal(root, state, *, source_sha=detail.SOURCE_SHA):
    """Recheck archive AND restored bytes, not merely an earlier receipt."""
    require(readiness(state), 'Capture has not finished its backup')
    rid = state.get('run', '')
    require(re.fullmatch(r'[0-9]{8}T[0-9]{6}Z-V8-ddr-activity-vivado-r3', rid), 'Unexpected run ID')
    folder = Path(root) / 'artifacts/ntrpu-board-vivado' / rid
    cert_path = folder / 'VERIFIED.json'
    cert = load(cert_path)
    for field in ('restoration_verified', 'local_and_nfs_verified', 'readable_and_all_members_verified'):
        require(cert.get(field) is True, 'Missing backup certification: ' + field)
    archive = folder / (rid + '.tar.gz')
    require(PurePosixPath(cert['archive']).name == archive.name, 'Archive name mismatch')
    require(archive.stat().st_size == cert['bytes'], 'Archive length mismatch')
    require(digest(archive) == cert['sha256'] == state['archive_sha256'], 'Archive hash mismatch')
    restored = folder / 'restored' / rid
    require(restored.is_dir() and not restored.is_symlink(), 'Missing restored run')
    prefix = rid + '/'
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        names = [m.name for m in members]
        require(len(names) == len(set(names)), 'Duplicate archive member')
        for member in members:
            safe_name(member.name)
            require(member.name == rid or member.name.startswith(prefix), 'Foreign archive root')
            require(member.isfile() or member.isdir(), 'Archive contains link or special file')
        hashes = json.load(tar.extractfile(prefix + 'FILES.json'))
        require(isinstance(hashes, dict) and hashes, 'Missing file manifest')
        for name in hashes:
            safe_name(name)
        file_names = {m.name[len(prefix):] for m in members if m.isfile()}
        require(file_names == set(hashes) | {'FILES.json'}, 'Archive inventory mismatch')
        require(cert['restored_files'] == cert['files'] == len(hashes), 'Certificate file count mismatch')
        actual = set()
        for path in restored.rglob('*'):
            require(not path.is_symlink(), 'Restored tree contains a link')
            require(path.resolve().is_relative_to(restored.resolve()), 'Restored path escapes run')
            if path.is_file():
                actual.add(path.relative_to(restored).as_posix())
        require(actual == file_names, 'Restored inventory mismatch')
        with tar.extractfile(prefix + 'FILES.json') as stream:
            require(digest_stream(stream) == digest(restored / 'FILES.json'), 'Restored manifest mismatch')
        for name, sha in hashes.items():
            with tar.extractfile(prefix + name) as stream:
                require(digest_stream(stream) == sha, 'Archive member hash mismatch: ' + name)
            require(digest(restored / name) == sha, 'Restored file hash mismatch: ' + name)
    require(hashes['source.zip'] == source_sha, 'Wrong immutable source package')
    parent = load(restored / 'REVIEW.json')
    require(parent['run'] == rid, 'Review run mismatch')
    require(hashes['REVIEW.json'] == cert['review_sha256'] == state['review_sha256'], 'Review binding mismatch')
    invocation = load(restored / 'invocation.json')
    require('finished_utc' in invocation, 'Invocation still active')
    require(invocation.get('source_sha256') == source_sha, 'Invocation source mismatch')
    return dict(run=rid, restored=str(restored), archive=str(archive.resolve()),
                remote_archive=cert['archive'], archive_sha256=cert['sha256'],
                certificate_sha256=digest(cert_path), review_sha256=hashes['REVIEW.json'],
                invocation_sha256=hashes['invocation.json'], verified_files=len(hashes),
                source_sha256=source_sha, manifest=hashes)


def remote_read(b, code):
    # Only read-only probes pass through this helper. Never retry mutations.
    result = subprocess.run(b.SSH + ['bash', '-se'], input=("python3 - <<'PY'\n" + code + '\nPY\n').encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
    return json.loads(result.stdout)


def backup(b, folder, report, evidence, state, config, monitor):
    """Separate supplemental archive, referring to the complete original backup."""
    parent = Path(evidence['restored'])
    files = {
        'activity-identity.json': folder / 'activity-identity.json',
        'parent-VERIFIED.json': parent.parent.parent / 'VERIFIED.json',
        'parent-REVIEW.json': parent / 'REVIEW.json',
        'parent-invocation.json': parent / 'invocation.json',
        'tools/' + Path(__file__).name: Path(__file__),
        'tools/' + Path(detail.__file__).name: Path(detail.__file__),
        'tools/test_ntrpu_ddr_activity_identity_followup_r1.py': b.ROOT / 'tools/test_ntrpu_ddr_activity_identity_followup_r1.py',
        'tools/test_ntrpu_ddr_activity_identity_r1.py': b.ROOT / 'tools/test_ntrpu_ddr_activity_identity_r1.py',
    }
    write_new(folder / 'dependency.json', {k: v for k, v in evidence.items() if k != 'manifest'})
    write_new(folder / 'capture-coordinator.json', state)
    write_new(folder / 'capture-package.json', config)
    write_new(folder / 'followup-state-at-review.json', monitor)
    for name in ('dependency.json', 'capture-coordinator.json', 'capture-package.json', 'followup-state-at-review.json'):
        files[name] = folder / name
    hashes = {name: digest(path) for name, path in files.items()}
    manifest = dict(files=hashes, dependency=load(folder / 'dependency.json'),
                    status=report['status'], ready_for_power_estimate=False,
                    vivado_launched=False, shared_queue_modified=False)
    archive = folder / 'followup.zip'
    with zipfile.ZipFile(archive, 'x', zipfile.ZIP_DEFLATED) as z:
        for name, path in files.items():
            z.write(path, name)
        z.writestr('MANIFEST.json', json.dumps(manifest, indent=2) + '\n')
    with zipfile.ZipFile(archive) as z:
        require(set(z.namelist()) == set(hashes) | {'MANIFEST.json'}, 'Supplemental inventory mismatch')
        z.extractall(folder / 'restored')
    require(all(digest(folder / 'restored' / n) == h for n, h in hashes.items()), 'Supplemental restore mismatch')
    remote = config['source']['remote'] + '/' + folder.name
    # One create/upload attempt; a lost acknowledgement leaves diagnostics for inspection.
    b.ssh('python3 -c ' + shlex.quote('from pathlib import Path;Path(' + repr(remote) + ').mkdir(exist_ok=False)'))
    b.scp(archive, b.HOST + ':' + remote + '/followup.zip')
    payload = dict(path=remote, sha256=digest(archive))
    code = r'''from pathlib import Path
import hashlib,json,zipfile
a=PAYLOAD
p=Path(a['path'])
def sha(f):
 h=hashlib.sha256()
 with f.open('rb') as s:
  for block in iter(lambda:s.read(1048576),b''):h.update(block)
 return h.hexdigest()
assert sha(p/'followup.zip')==a['sha256']
with zipfile.ZipFile(p/'followup.zip') as z:
 m=json.loads(z.read('MANIFEST.json'));files=m['files']
 assert set(z.namelist())==set(files)|{'MANIFEST.json'}
 assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in files.items())
 z.extractall(p/'restored')
assert all(sha(p/'restored'/n)==h for n,h in files.items())
assert sha(Path(m['dependency']['remote_archive']))==m['dependency']['archive_sha256']
print(json.dumps(dict(sha256=a['sha256'],restored_files=len(files),dependency_verified=True)))
'''.replace('PAYLOAD', repr(payload), 1)
    # Restoration is a mutation: do not use the read-only retry helper.
    result = subprocess.run(b.SSH + ['bash', '-se'], input=("python3 - <<'PY'\n" + code + '\nPY\n').encode(),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=60, check=True)
    remote_result = json.loads(result.stdout)
    receipt = dict(utc=b.stamp(), archive=str(archive.resolve()), remote_archive=remote + '/followup.zip',
                   sha256=digest(archive), bytes=archive.stat().st_size, files=len(files),
                   restoration_verified=True, local_and_nfs_verified=True, remote_restoration=remote_result,
                   dependency=manifest['dependency'], review_status=report['status'],
                   ready_for_power_estimate=False, board_power_qualified=False, exact_dimm_qualified=False)
    write_new(folder / 'VERIFIED.json', receipt)
    b.scp(folder / 'VERIFIED.json', b.HOST + ':' + remote + '/VERIFIED.json')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--wait-hours', type=float, default=12, help='Observer deadline; never changes a Vivado deadline')
    args = parser.parse_args()
    require(0 < args.wait_hours <= 24, 'Observer deadline must be in (0,24] hours')
    root = Path(__file__).resolve().parents[1]
    here = root / 'build/cloud-setup/20261002-board'
    sys.path.insert(0, str(here))
    import boardctl as b
    from ntrpu_queue_state_io_r1 import write_json_atomic
    config = load(here / PACKAGE)
    require(digest(here / PACKAGE) == PACKAGE_SHA, 'Configuration changed')
    require(digest(here / 'queue_ddr_activity_vivado_r3.py') == DRIVER_SHA, 'Driver changed')
    require(digest(detail.__file__) == REVIEWER_SHA, 'Supplemental reviewer changed')
    require(config['source']['source_sha256'] == detail.SOURCE_SHA, 'Source changed')
    record = here / 'ddr-activity-identity-followup-r1-monitor.json'
    tracked = [Path(__file__), Path(detail.__file__), root / 'tools/test_ntrpu_ddr_activity_identity_followup_r1.py',
               root / 'tools/test_ntrpu_ddr_activity_identity_r1.py']
    bindings = {str(path): digest(path) for path in tracked}
    state = dict(utc=b.stamp(), status='waiting_for_terminal_capture_backup', local_pid=os.getpid(),
                 script_sha256=digest(__file__), dependency=COORDINATOR, bindings=bindings,
                 vivado_launched=False, shared_queue_modified=False, wait_hours=args.wait_hours)
    write_new(record, state)  # Exclusive: a second monitor cannot duplicate work.

    def save(status, **extra):
        state.update(utc=b.stamp(), status=status, **extra)
        write_json_atomic(record, state)
        print(json.dumps(state), flush=True)

    deadline = time.monotonic() + args.wait_hours * 3600
    try:
        while True:
            prior = load(here / COORDINATOR)
            if readiness(prior):
                break
            require(time.monotonic() < deadline, 'Observer deadline reached; capture remains untouched')
            time.sleep(30)
        require(all(digest(p) == h for p, h in bindings.items()), 'Supplemental sources changed while waiting')
        save('verifying_terminal_backup', run=prior['run'])
        evidence = validate_terminal(root, prior)
        code = '''from pathlib import Path
import hashlib,json
p=Path(PATH);h=hashlib.sha256()
with p.open('rb') as s:
 for block in iter(lambda:s.read(1048576),b''):h.update(block)
print(json.dumps(dict(sha256=h.hexdigest())))
'''.replace('PATH', repr(evidence['remote_archive']), 1)
        remote = remote_read(b, code)
        require(remote['sha256'] == evidence['archive_sha256'], 'NFS parent backup changed')
        folder = b.new_local_archive_directory(prior['run'] + '-activity-identity-r1')
        save('reviewing_restored_capture', output=str(folder.resolve()))
        report = detail.review(Path(evidence['restored']))
        write_new(folder / 'activity-identity.json', report)
        # The reviewer is read-only. Check every restored file again before certifying.
        restored = Path(evidence['restored'])
        require(all(digest(restored / n) == h for n, h in evidence['manifest'].items()), 'Reviewer changed parent files')
        save('backing_supplemental_review', review_status=report['status'])
        receipt = backup(b, folder, report, evidence, prior, config, state)
        save('supplemental_review_backed_requires_inspection', review_status=report['status'],
             review_errors=report['errors'], archive_sha256=receipt['sha256'],
             restoration_verified=True, local_and_nfs_verified=True)
    except Exception as exc:
        save('stopped_requires_inspection', error=repr(exc))
        raise


if __name__ == '__main__':
    main()
