"""Synthetic corruption controls. No RTL, native SAIF, or power result."""
import argparse
import copy
import csv
import io
import json
from pathlib import Path
import tempfile
import subprocess
import sys

import review_ntrpu_ddr_activity_detail_r1 as detail


def synthetic_saif(overrides=None, *, unit='ps', factor=1, duration=20480000, escape_pins=False):
    names = {name: dict(T0=duration * (expected == 0), T1=duration * expected, TX=0, TZ=0, TC=0)
             for name, expected in detail.STEADY.items()}
    names['c0_ddr4_clk'] = dict(T0=duration // 2, T1=duration // 2, TX=0, TZ=0, TC=8192)
    for rank in range(4):
        for pin in ('cs_n', 'cke', 'odt'):
            names[f'c0_ddr4_{pin}[{rank}]'] = dict(T0=duration // 2, T1=duration // 2, TX=0, TZ=0, TC=20)
    names['logic_a'] = dict(T0=duration // 2, T1=duration // 2, TX=0, TZ=0, TC=100)
    names['logic_b'] = dict(T0=duration // 2, T1=duration // 2, TX=0, TZ=0, TC=100)
    for name, values in (overrides or {}).items():
        if values is None:
            names.pop(name)
        else:
            names[name] = {**names.get(name, {}), **values}
    nested = {}
    for name, values in names.items():
        scope, _, leaf = name.rpartition('/')
        if escape_pins:
            leaf = leaf.replace('[', r'\[').replace(']', r'\]')
        nested.setdefault(scope, []).append('(' + leaf + ' ' + ' '.join(f'({k} {v})' for k, v in values.items()) + ')')
    body = '(NET ' + ' '.join(nested.pop('')) + ')'
    for scope, entries in nested.items():
        body += f'(INSTANCE {scope} (NET ' + ' '.join(entries) + '))'
    return (f'(SAIFILE (SAIFVERSION "2.0") (DIRECTION "backward") (DESIGN) (DATE "Synthetic only") '
            f'(DIVIDER /) (TIMESCALE {factor} {unit}) (DURATION {duration}) '
            '(INSTANCE ntrpu_ddr_example_activity_tb (INSTANCE dut ' + body + ')))\n')


def synthetic_windows():
    rows = [['window', 'edge', 'metric', 'value'], ['all', 'scope', 'objects_selected', '100']]
    for i in range(2):
        for edge, j in [('start', i), ('end', i + 1)]:
            row = dict(cycles=4096 * j + 10000, sample_ps=20480000 * j + 50000000,
                       aw_count=10 * j, b_count=9 * j, ar_count=10 * j,
                       rlast_count=9 * j, w_beats=80 * j, r_beats=72 * j,
                       w_bytes=5120 * j, w_stalls=10 * j, r_stalls=5 * j,
                       pending_w=j, pending_r=j, simulator_time=f'{20480 * j + 50000} ns')
            rows += [[f'prbs_{i}', edge, k, str(v)] for k, v in row.items()]
    out = io.StringIO(newline='')
    csv.writer(out, delimiter='\t', lineterminator='\n').writerows(rows)
    return out.getvalue()


def run_controls(folder):
    folder = Path(folder)
    saif = folder / 'sample.saif'
    windows = folder / 'windows.tsv'
    baseline = synthetic_saif()
    time_table = synthetic_windows()
    rejected, observed = [], []

    def parse(text):
        saif.write_text(text, encoding='utf-8')
        return detail.parse_saif(saif)

    def reject(label, function):
        try:
            function()
        except (ValueError, KeyError):
            rejected.append(label)
        else:
            raise AssertionError('Accepted corrupted evidence: ' + label)

    reference = parse(baseline)
    summary = detail.summarize_saif(reference)
    assert not summary['issues'] and summary['signals'] == 25
    assert len(summary['rank_pin_observations']) == 4
    assert all(p['commands_observed'] is None and p['refresh_observed'] is None for p in summary['rank_pin_observations'])
    # Equivalent time units, quoted/escaped vector names and DUT-only roots.
    scaled = parse(synthetic_saif(unit='ns', duration=20480, escape_pins=True))
    assert reference['signals'].keys() == scaled['signals'].keys()
    assert detail.compare_saifs(reference, scaled)['changed_signals'] == 0
    assert not detail.summarize_saif(scaled)['issues']
    direct = baseline.replace('(INSTANCE ntrpu_ddr_example_activity_tb ', '', 1)[:-2] + '\n'
    assert parse(direct)['signals'] == reference['signals']
    quoted = baseline.replace('(c0_ddr4_cs_n[0]', '("c0_ddr4_cs_n[0]"')
    assert parse(quoted)['signals'] == reference['signals']
    # Exercise lexer chunk boundaries, escaped delimiters, quotes, and spaces.
    header = '(DATE "' + ' ' * 65530 + 'x\\"y")'
    assert parse(baseline.replace('(DATE "Synthetic only")', header))['signals'] == reference['signals']
    for label, text in {
        'wrong_root': baseline.replace('(SAIFILE ', '(ANYTHING ', 1),
        'second_root': baseline + baseline,
        'truncated': baseline[:-3],
        'wrong_version': baseline.replace('"2.0"', '"3.0"'),
        'wrong_direction': baseline.replace('"backward"', '"forward"'),
        'wrong_divider': baseline.replace('(DIVIDER /)', '(DIVIDER .)'),
        'missing_duration': baseline.replace('(DURATION 20480000)', ''),
        'wrong_duration': baseline.replace('(DURATION 20480000)', '(DURATION 20480001)'),
        'zero_scale': baseline.replace('(TIMESCALE 1 ps)', '(TIMESCALE 0 ps)'),
        'nan_scale': baseline.replace('(TIMESCALE 1 ps)', '(TIMESCALE NaN ps)'),
        'wrong_unit': baseline.replace('(TIMESCALE 1 ps)', '(TIMESCALE 1 ticks)'),
        'duplicate_header': baseline.replace('(DURATION 20480000)', '(DURATION 20480000) (DURATION 20480000)'),
        'foreign_scope': baseline.replace('(INSTANCE dut ', '(INSTANCE memory ', 1),
        'missing_counter': baseline.replace('(TX 0)', '', 1),
        'negative_counter': baseline.replace('(TC 0)', '(TC -1)', 1),
        'unknown_counter': baseline.replace('(TX 0)', '(UNKNOWN 0)', 1),
        'duplicate_counter': baseline.replace('(TX 0)', '(TX 0) (TX 0)', 1),
        'residency_mismatch': baseline.replace('(T0 20480000)', '(T0 20479999)', 1),
        'canonical_duplicate': baseline.replace('(logic_a ', '(c0_ddr4_cs_n\\[0\\] ', 1),
        'malformed_counter_nesting': baseline.replace('(TC 0)', '(TC (0))', 1),
    }.items():
        reject(label, lambda text=text: parse(text))
    for name, change, label in [
        ('sys_rst', dict(T0=20479999, T1=1, TC=2), 'reset_pulse'),
        ('ntrpu_mig_reset', dict(T0=20479999, TX=1), 'reset_unknown'),
        ('c0_init_calib_complete', dict(T0=20480000, T1=0), 'calibration_low'),
        ('sys_rst', dict(TC=2), 'zero_duration_reset_glitch'),
        ('c0_data_compare_error', dict(IG=1), 'critical_ig'),
        ('c0_ddr4_read_err', dict(TB=1), 'critical_tb'),
        ('c0_ddr4_aresetn', None, 'missing_reset_release'),
        ('c0_ddr4_clk', dict(TC=8191), 'clock_missing_transition'),
        ('c0_ddr4_clk', dict(T0=10239999, T1=10240001), 'clock_duty'),
        ('c0_ddr4_clk', dict(TB=1), 'clock_tb'),
        ('c0_ddr4_clk', dict(IG=1), 'clock_ig'),
    ]:
        value = detail.summarize_saif(parse(synthetic_saif({name: change})))
        assert value['issues'], label
        observed.append(label)
    unknown = detail.summarize_saif(parse(synthetic_saif({'logic_a': dict(T0=10239999, TX=1)})))
    assert unknown['signals_with_x_z_or_tb'] == 1
    assert unknown['highest_unknown_residency'][0]['signal'] == 'dut/logic_a'
    observed.append('noncritical_unknown_reported')
    cancel = parse(synthetic_saif({'logic_a': dict(TC=150), 'logic_b': dict(TC=50)}))
    compared = detail.compare_saifs(reference, cancel)
    assert compared['transitions_before'] == compared['transitions_after']
    assert compared['summed_absolute_per_signal_transition_difference'] == 100 and compared['changed_signals'] == 2
    assert not compared['statistical_representativeness_established']
    observed.append('equal_total_conceals_per_signal_changes')
    altered = copy.deepcopy(reference)
    altered['signals']['dut/foreign'] = altered['signals'].pop('dut/logic_a')
    reject('same_count_wrong_inventory', lambda: detail.compare_saifs(reference, altered))

    def check_windows(text):
        windows.write_text(text, encoding='utf-8')
        return detail.review_windows(windows)

    valid = check_windows(time_table)
    assert valid['windows'][0]['rates']['accepted_write_bytes_per_second'] == '2.5E+8'
    assert valid['windows'][0]['rates']['accepted_read_bytes_per_second'] == '2.25E+8'
    cases = {
        'wrong_columns': time_table.replace('metric\tvalue', 'value\tmetric', 1),
        'extra_metric': time_table + 'prbs_0\tstart\tunknown\t0\n',
        'duplicate_metric': time_table + 'prbs_0\tend\tcycles\t14096\n',
        'missing_metric': time_table.replace('prbs_0\tend\tcycles\t14096\n', ''),
        'cycles_mismatch': time_table.replace('prbs_0\tend\tcycles\t14096', 'prbs_0\tend\tcycles\t14095'),
        'time_interval_mismatch': time_table.replace('prbs_0\tend\tsimulator_time\t70480 ns', 'prbs_0\tend\tsimulator_time\t70481 ns'),
        'future_sample': time_table.replace('prbs_0\tend\tsample_ps\t70480000', 'prbs_0\tend\tsample_ps\t70480001'),
        'stale_sample': time_table.replace('prbs_0\tstart\tsimulator_time\t50000 ns', 'prbs_0\tstart\tsimulator_time\t50005 ns'),
        'bad_pending': time_table.replace('prbs_0\tend\tpending_w\t1', 'prbs_0\tend\tpending_w\t2'),
        'noncontiguous_counter': time_table.replace('prbs_1\tstart\tw_bytes\t5120', 'prbs_1\tstart\tw_bytes\t5119'),
        'write_mask_exceeds_bus': time_table.replace('prbs_0\tend\tw_bytes\t5120', 'prbs_0\tend\tw_bytes\t5121'),
        'more_than_one_beat_per_cycle': time_table.replace('prbs_0\tend\tw_beats\t80', 'prbs_0\tend\tw_beats\t4097'),
        'accepted_and_stalled_overlap': time_table.replace('prbs_0\tend\tw_stalls\t10', 'prbs_0\tend\tw_stalls\t4096'),
        'rlast_exceeds_read_beats': time_table.replace('prbs_0\tend\tr_beats\t72', 'prbs_0\tend\tr_beats\t8'),
        'counter_regression': time_table.replace('prbs_1\tend\tw_bytes\t10240', 'prbs_1\tend\tw_bytes\t5119'),
    }
    for label, text in cases.items():
        assert text != time_table, label
        reject(label, lambda text=text: check_windows(text))
    # Equivalent simulator units compare by value, not text spelling.
    equivalent = time_table.replace('prbs_1\tstart\tsimulator_time\t70480 ns', 'prbs_1\tstart\tsimulator_time\t70.480 us')
    assert len(check_windows(equivalent)['windows']) == 2
    observed.append('equivalent_time_units')
    # End-to-end receipt binding and CLI writes: synthetic run names make the
    # provenance explicit, and byte hashes prove archived inputs stay untouched.
    run = folder / 'synthetic-terminal-run'
    capture = run / 'output/sim/activity_capture'
    capture.mkdir(parents=True)
    for i in range(2):
        (capture / f'prbs_{i}.saif').write_text(baseline)
    (capture / 'windows.tsv').write_text(time_table)
    (capture / 'CAPTURE_COMPLETE.txt').write_text('NTRPU_ACTIVITY_CAPTURE_EXPORTED; NOT_POWER_QUALIFIED\n')
    inv = dict(finished_utc='SYNTHETIC_ONLY', exit_code=0, frozen_source_unchanged=True,
               source_sha256=detail.CAPTURE_SOURCE_SHA256)
    (run / 'invocation.json').write_text(json.dumps(inv))
    parent = dict(run=run.name, status='CAPTURE_EXPORTED_REQUIRES_ACTIVITY_REVIEW', errors=[],
                  invocation_sha256=detail.digest(run / 'invocation.json'), native_identity_observed=True)
    (run / 'REVIEW.json').write_text(json.dumps(parent))
    native_result = detail.review(run)
    assert native_result['status'] == 'ACTIVITY_DETAIL_EXPORTED_REQUIRES_REVIEW'
    assert not native_result['ready_for_power_estimate'] and not native_result['board_power_qualified']
    assert native_result['activity']['rank_command_coverage'] == 'NOT_ESTABLISHED_PIN_OCCUPANCY_ONLY'
    assert native_result['activity']['refresh_coverage'].startswith('NOT_ESTABLISHED')
    observed.append('complete_synthetic_receipt_still_not_power_qualified')
    for label, changes in [('wrong_source_package', dict(source_sha256='0' * 64)),
                           ('native_timeout', dict(exit_code=124)),
                           ('frozen_inputs_changed', dict(frozen_source_unchanged=False))]:
        (run / 'invocation.json').write_text(json.dumps({**inv, **changes}))
        assert detail.review(run)['status'] == 'DIAGNOSTIC_DDR_ACTIVITY_DETAIL_R1', label
        observed.append(label)
    (run / 'invocation.json').write_text(json.dumps({k: v for k, v in inv.items() if k != 'finished_utc'}))
    reject('active_run', lambda: detail.review(run))
    (run / 'invocation.json').write_text(json.dumps(inv))
    for label, changes in [('parent_wrong_invocation', dict(invocation_sha256='0' * 64)),
                           ('parent_native_identity_missing', dict(native_identity_observed=False)),
                           ('parent_rejected_capture', dict(errors=['native error']))]:
        (run / 'REVIEW.json').write_text(json.dumps({**parent, **changes}))
        assert detail.review(run)['status'] == 'DIAGNOSTIC_DDR_ACTIVITY_DETAIL_R1', label
        observed.append(label)
    (run / 'REVIEW.json').write_text(json.dumps(parent))
    before = {str(p.relative_to(run)): detail.digest(p) for p in run.rglob('*') if p.is_file()}
    command = [sys.executable, detail.__file__, str(run), '--output']
    result = subprocess.run(command + [str(folder / 'supplement.json')], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for label, output in [('cli_refuses_overwrite', folder / 'supplement.json'),
                          ('cli_refuses_archived_output', run / 'alteration.json')]:
        result = subprocess.run(command + [str(output)], capture_output=True, text=True)
        assert result.returncode != 0, label
        rejected.append(label)
    after = {str(p.relative_to(run)): detail.digest(p) for p in run.rglob('*') if p.is_file()}
    assert before == after
    observed.append('cli_preserves_all_archived_input_bytes')
    return dict(status='PASS_SYNTHETIC_DDR_ACTIVITY_DETAIL_CONTROLS', rejected=rejected,
                diagnostic_observations=observed, reviewer_sha256=detail.digest(Path(detail.__file__)),
                controls_sha256=detail.digest(Path(__file__)), synthetic=True,
                native_elaborated=False, native_saif=False, power_estimated=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix='ddr-activity-detail-', dir=args.output.parent) as folder:
        result = run_controls(folder)
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
