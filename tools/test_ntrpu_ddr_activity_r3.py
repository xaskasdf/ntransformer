"""Reject actual diagnostics while accepting escaped native-style SAIF names."""
import argparse
import json
from pathlib import Path
import tempfile

import review_ntrpu_ddr_activity_r3 as reviewer
import review_ntrpu_ddr_activity_detail_r1 as detail
import test_ntrpu_ddr_activity_detail_r1 as fixtures


def controls(folder):
    root = Path(folder)/'synthetic-run'
    capture = root/'output/sim/activity_capture'
    capture.mkdir(parents=True)
    inv = dict(finished_utc='synthetic', exit_code=0, frozen_source_unchanged=True,
               source_sha256=detail.CAPTURE_SOURCE_SHA256)
    (root/'invocation.json').write_text(json.dumps(inv))
    parent = dict(run=root.name, status='DIAGNOSTIC_DDR_ACTIVITY_R1',
                  invocation_sha256=detail.digest(root/'invocation.json'),
                  errors=['Native error diagnostics require inspection', 'Independent capture review: AssertionError()'])
    (root/'REVIEW.json').write_text(json.dumps(parent))
    (capture/'windows.tsv').write_text(fixtures.synthetic_windows())
    (capture/'CAPTURE_COMPLETE.txt').write_text('NTRPU_ACTIVITY_CAPTURE_EXPORTED; NOT_POWER_QUALIFIED\n')
    base = fixtures.synthetic_saif()
    escaped = base.replace('(logic_a ', r'(\\byte_num\[0\]\.clb2phy_wr_dq_array\ \[0\]\[0\] ', 1)
    for i in (0, 1):
        (capture/f'prbs_{i}.saif').write_text(escaped)
    (root/'output/sim/simulate.log').write_text('NTRPU_ACTIVITY_IDENTITY dut_simulation=FALSE tg_simulation=FALSE\n')
    echoes = '#     puts stderr "NTRPU_ACTIVITY_INTEGRATION_ERROR $message"\n##     puts stderr "NTRPU_ACTIVITY_CAPTURE_ERROR $message"\n'
    (root/'console.log').write_text(echoes)
    (root/'vivado.log').write_text(echoes)
    before = {str(p.relative_to(root)): detail.digest(p) for p in root.rglob('*') if p.is_file()}
    result = reviewer.review(root)
    assert not result['errors'], result
    assert result['status'] == 'CAPTURE_ACTIVITY_REVIEWED_IDENTITY_AND_MAPPING_PENDING'
    assert sum(map(len, result['recognized_tcl_source_echoes'].values())) == 4
    assert all(s['signals'] == 25 for s in result['activity']['saifs'])
    assert result['parent_status'] == parent['status'] and not result['ready_for_power_estimate']
    after = {str(p.relative_to(root)): detail.digest(p) for p in root.rglob('*') if p.is_file()}
    assert before == after
    rejected = []
    for name, text, location in [
        ('native_error', 'ERROR: actual failure', 'console.log'),
        ('native_fatal', 'FATAL: actual failure', 'console.log'),
        ('capture_error', 'NTRPU_ACTIVITY_CAPTURE_ERROR failed', 'console.log'),
        ('source_like_runtime_error', '# ERROR: actual failure', 'console.log'),
        ('nonexact_echo', '# puts stderr "NTRPU_ACTIVITY_CAPTURE_ERROR failure"', 'console.log'),
        ('echo_in_simulation', echoes, 'output/sim/simulate.log'),
    ]:
        path = root/location; original = path.read_text()
        path.write_text(original+'\n'+text+'\n')
        assert reviewer.review(root)['native_error_diagnostics'], name
        path.write_text(original); rejected.append(name)
    for name, text in [('missing_escape', escaped.replace(r'array\ \[0', r'array \[0', 1)),
                       ('truncated', escaped[:-4]),
                       ('duration', escaped.replace('(DURATION 20480000)', '(DURATION 20480001)')),
                       ('reset_pulse', escaped.replace('(sys_rst (T0 20480000) (T1 0)', '(sys_rst (T0 20479999) (T1 1)'))]:
        (capture/'prbs_1.saif').write_text(text)
        assert reviewer.review(root)['errors'], name
        (capture/'prbs_1.saif').write_text(escaped); rejected.append(name)
    for name, changes in [('nonzero_exit', dict(exit_code=1)), ('changed_source', dict(frozen_source_unchanged=False)),
                          ('wrong_source', dict(source_sha256='0'*64))]:
        (root/'invocation.json').write_text(json.dumps({**inv, **changes}))
        assert reviewer.review(root)['errors'], name
        (root/'invocation.json').write_text(json.dumps(inv)); rejected.append(name)
    return dict(status='PASS_DDR_ACTIVITY_R3_CONTROLS', rejected=rejected, synthetic=True, native_saif=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='ddr-activity-r3-', dir=args.output.parent) as folder:
        result = controls(folder)
    with args.output.open('x', encoding='utf-8') as f:
        json.dump(result, f, indent=2); f.write('\n')
    print(json.dumps(result))
