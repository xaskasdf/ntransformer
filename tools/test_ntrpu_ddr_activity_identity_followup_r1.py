"""Synthetic backup-corruption and postprocessing controls; no Vivado execution."""
import hashlib
import io
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import ntrpu_ddr_activity_identity_followup_r1 as followup


RID = '20261016T000000Z-V8-ddr-activity-vivado-r3'
REPO = Path(__file__).resolve().parents[1]


def fixture(root, *, bad_name=None, duplicate=False, active=False):
    folder = root / 'artifacts/ntrpu-board-vivado' / RID
    restored = folder / 'restored' / RID
    restored.mkdir(parents=True)
    source = b'SYNTHETIC source envelope; not an FPGA result\n'
    sha = hashlib.sha256(source).hexdigest()
    inv = dict(source_sha256=sha, frozen_source_unchanged=True, exit_code=124)
    if not active:
        inv['finished_utc'] = '2026-10-16T00:00:00Z'
    parent = dict(run=RID, status='DIAGNOSTIC_DDR_ACTIVITY_R1', errors=['Synthetic timeout'],
                  native_identity_observed=False, board_power_qualified=False, exact_dimm_qualified=False)
    values = {'source.zip': source, 'invocation.json': json.dumps(inv).encode(),
              'REVIEW.json': json.dumps(parent).encode(), 'console.log': b'SYNTHETIC timeout\n'}
    if bad_name:
        values[bad_name] = b'Invalid archive path\n'
    hashes = {n: hashlib.sha256(v).hexdigest() for n, v in values.items()}
    values['FILES.json'] = json.dumps(hashes).encode()
    archive = folder / (RID + '.tar.gz')
    with tarfile.open(archive, 'w:gz') as tar:
        for name, raw in values.items():
            info = tarfile.TarInfo(RID + '/' + name)
            info.size = len(raw)
            tar.addfile(info, io.BytesIO(raw))
            if '..' not in Path(name).parts and ':' not in name:
                target = restored / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
        if duplicate:
            info = tarfile.TarInfo(RID + '/console.log')
            info.size = len(values['console.log'])
            tar.addfile(info, io.BytesIO(values['console.log']))
    cert = dict(archive=archive.resolve().as_posix(), bytes=archive.stat().st_size,
                sha256=followup.digest(archive), files=len(hashes), restored_files=len(hashes),
                review_sha256=hashes['REVIEW.json'], restoration_verified=True,
                local_and_nfs_verified=True, readable_and_all_members_verified=True)
    followup.write_new(folder / 'VERIFIED.json', cert)
    state = dict(status='diagnostic_backed_requires_inspection', run=RID,
                 script_sha256=followup.DRIVER_SHA, package_sha256=followup.PACKAGE_SHA,
                 archive_sha256=cert['sha256'], review_sha256=cert['review_sha256'])
    return folder, restored, cert, state, sha


class BackupControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='ddr-followup-controls-')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_waiting_states_do_not_claim_terminal(self):
        for status in followup.WAITING:
            with self.subTest(status=status):
                self.assertFalse(followup.readiness(dict(status=status, script_sha256=followup.DRIVER_SHA,
                                                         package_sha256=followup.PACKAGE_SHA)))

    def test_stopped_or_unknown_state_rejected(self):
        for status in ('stopped_requires_inspection', 'observer_disconnected_waiting', 'success', None):
            with self.subTest(status=status), self.assertRaises(ValueError):
                followup.readiness(dict(status=status, script_sha256=followup.DRIVER_SHA,
                                        package_sha256=followup.PACKAGE_SHA))

    def test_wrong_coordinator_binding_rejected(self):
        for field in ('script_sha256', 'package_sha256'):
            state = dict(status='activity_capture_backed_requires_review', script_sha256=followup.DRIVER_SHA,
                         package_sha256=followup.PACKAGE_SHA)
            state[field] = '0' * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                followup.readiness(state)

    def test_complete_diagnostic_archive_accepted_without_promoting_power(self):
        folder, restored, cert, state, sha = fixture(self.root)
        result = followup.validate_terminal(self.root, state, source_sha=sha)
        self.assertEqual(result['verified_files'], 4)
        before = {n: followup.digest(restored / n) for n in result['manifest']}
        report = followup.detail.review(restored, source_sha=sha)
        self.assertEqual(report['status'], 'DIAGNOSTIC_DDR_ACTIVITY_SOURCE_IDENTITY_R1')
        self.assertTrue(report['errors'])
        self.assertIn('Native invocation failed; source audit is diagnostic only', report['errors'])
        for flag in ('ready_for_power_estimate', 'board_power_qualified', 'exact_dimm_qualified'):
            self.assertIs(report[flag], False)
        self.assertEqual(before, {n: followup.digest(restored / n) for n in result['manifest']})

    def test_success_label_cannot_bypass_byte_verification(self):
        folder, restored, cert, state, sha = fixture(self.root)
        state['status'] = 'activity_capture_backed_requires_review'
        (restored / 'console.log').write_bytes(b'Changed bytes')
        with self.assertRaisesRegex(ValueError, 'Restored file hash'):
            followup.validate_terminal(self.root, state, source_sha=sha)

    def test_state_bindings_rejected(self):
        _, _, _, state, sha = fixture(self.root)
        for field, value in [('run', '../foreign'), ('archive_sha256', '0' * 64),
                             ('review_sha256', '0' * 64), ('status', 'sealing')]:
            changed = {**state, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                followup.validate_terminal(self.root, changed, source_sha=sha)

    def test_certificate_controls(self):
        folder, _, cert, state, sha = fixture(self.root)
        for field, value in [('restoration_verified', False), ('local_and_nfs_verified', False),
                             ('readable_and_all_members_verified', False), ('restored_files', 3),
                             ('files', 3), ('bytes', 0), ('review_sha256', '0' * 64),
                             ('archive', '/foreign.tar.gz')]:
            (folder / 'VERIFIED.json').write_text(json.dumps({**cert, field: value}))
            with self.subTest(field=field), self.assertRaises(ValueError):
                followup.validate_terminal(self.root, state, source_sha=sha)

    def test_source_mismatch(self):
        _, _, _, state, _ = fixture(self.root)
        with self.assertRaisesRegex(ValueError, 'source package'):
            followup.validate_terminal(self.root, state, source_sha='0' * 64)

    def test_active_invocation_rejected_even_with_terminal_receipt(self):
        _, _, _, state, sha = fixture(self.root, active=True)
        with self.assertRaisesRegex(ValueError, 'still active'):
            followup.validate_terminal(self.root, state, source_sha=sha)

    def test_unsafe_archive_path_rejected(self):
        _, _, _, state, sha = fixture(self.root, bad_name='../outside')
        with self.assertRaisesRegex(ValueError, 'Unsafe archive'):
            followup.validate_terminal(self.root, state, source_sha=sha)

    def test_duplicate_archive_rejected(self):
        _, _, _, state, sha = fixture(self.root, duplicate=True)
        with self.assertRaisesRegex(ValueError, 'Duplicate archive'):
            followup.validate_terminal(self.root, state, source_sha=sha)

    def test_extra_restored_file_rejected(self):
        _, restored, _, state, sha = fixture(self.root)
        (restored / 'extra.txt').write_text('foreign')
        with self.assertRaisesRegex(ValueError, 'Restored inventory'):
            followup.validate_terminal(self.root, state, source_sha=sha)

    def test_changed_restored_manifest_rejected(self):
        _, restored, _, state, sha = fixture(self.root)
        (restored / 'FILES.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'Restored manifest'):
            followup.validate_terminal(self.root, state, source_sha=sha)

    def test_changed_archive_rejected(self):
        folder, _, cert, state, sha = fixture(self.root)
        archive = folder / (RID + '.tar.gz')
        raw = bytearray(archive.read_bytes())
        raw[-1] ^= 1
        archive.write_bytes(raw)
        with self.assertRaisesRegex(ValueError, 'Archive hash'):
            followup.validate_terminal(self.root, state, source_sha=sha)

    def test_separate_backup_restored_and_diagnostic_preserved(self):
        _, restored, _, state, sha = fixture(self.root)
        evidence = followup.validate_terminal(self.root, state, source_sha=sha)
        report = followup.detail.review(restored, source_sha=sha)
        output = self.root / 'supplemental'
        output.mkdir()
        followup.write_new(output / 'activity-identity.json', report)
        remote_root = self.root / 'nfs'
        remote_root.mkdir()
        config = dict(source=dict(remote=remote_root.as_posix()))
        real_run = subprocess.run

        class LocalTransport:
            ROOT = REPO
            HOST = 'local'
            SSH = ['MOCK_SSH']
            @staticmethod
            def stamp():
                return 'SYNTHETIC'
            @staticmethod
            def ssh(command):
                argv = shlex.split(command)
                if argv[:2] != ['python3', '-c']:
                    raise AssertionError('Unexpected command')
                return real_run([sys.executable, '-c', argv[2]], check=True, capture_output=True).stdout
            @staticmethod
            def scp(src, dst):
                if not str(dst).startswith('local:'):
                    raise AssertionError('Unexpected destination')
                shutil.copyfile(src, str(dst)[len('local:'):])

        def local_run(argv, **kwargs):
            self.assertEqual(argv, ['MOCK_SSH', 'bash', '-se'])
            data = kwargs.pop('input').decode()
            self.assertTrue(data.startswith("python3 - <<'PY'\n") and data.endswith('\nPY\n'))
            code = data[len("python3 - <<'PY'\n"):-len('\nPY\n')]
            return real_run([sys.executable, '-c', code], **kwargs)

        before = {n: followup.digest(restored / n) for n in evidence['manifest']}
        with patch.object(followup.subprocess, 'run', side_effect=local_run):
            receipt = followup.backup(LocalTransport(), output, report, evidence, state, config, {'synthetic': True})
        self.assertTrue(receipt['restoration_verified'] and receipt['local_and_nfs_verified'])
        self.assertEqual(receipt['review_status'], 'DIAGNOSTIC_DDR_ACTIVITY_SOURCE_IDENTITY_R1')
        self.assertFalse(receipt['ready_for_power_estimate'])
        self.assertEqual(followup.load(output / 'restored/activity-identity.json'), report)
        self.assertEqual(followup.load(remote_root / output.name / 'restored/activity-identity.json'), report)
        self.assertEqual(before, {n: followup.digest(restored / n) for n in evidence['manifest']})
        with self.assertRaises(FileExistsError):
            followup.backup(LocalTransport(), output, report, evidence, state, config, {'synthetic': True})


if __name__ == '__main__':
    unittest.main(verbosity=2)
