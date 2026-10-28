"""Reconstruct reset-first diagnostic pins; errors remain functional failures."""
import argparse
import re
from decimal import Decimal
from pathlib import Path
from check_ntrpu_ddr_phase import records


def review(text, open_row, lag):
    assert open_row in (0, 1) and lag in (12500, 2000000)
    assert 'Time resolution is 1 ps' in text
    assert not re.search(r'(?i)fatal|NTRPU_MODEL_INCOMPLETE|WARNING:', text)
    assert records(text, 'NTRPU_MODEL_LAG') == [dict(phase_ps='313', cke_lag_ps=str(lag))]
    assert records(text, 'NTRPU_MODEL_TIMING') == [dict(
        full_delays='1', cold_reset_ps='200000000', cold_cke_ps='500000000', warm_cke_ps='500000000')]
    warm = records(text, 'NTRPU_MODEL_WARM_RESET')
    cold = records(text, 'NTRPU_MODEL_COLD_DONE')
    done = records(text, 'NTRPU_MODEL_WINDOW_COMPLETE')
    assert len(warm) == len(cold) == len(done) == 1
    assert int(warm[0]['open_row']) == int(done[0]['open_row']) == open_row
    assert int(warm[0]['low_ps']) == 2600000
    reset = int(warm[0]['time_ps'])
    end = int(done[0]['time_ps'])
    assert reset % 1250 == 313 and reset > int(cold[0]['time_ps'])
    pins = records(text, 'NTRPU_MODEL_PINS')
    assert len(pins) == 7 and len({p['time_ps'] for p in pins}) == 7
    expected = [(0, 0, 0), (200000000, 1, 0), (700000000, 1, 1),
                (reset, 0, 1), (reset + lag, 0, 0),
                (reset + 2600000, 1, 0), (reset + 502600000, 1, 1)]
    assert [(int(p['time_ps']), int(p['rst']), int(p['cke'])) for p in pins] == expected
    for pin in pins:
        assert set(pin) == {'time_ps', 'edge', 'ck', 'rst', 'cke', 'cs'}
        t = int(pin['time_ps'])
        assert pin['cs'] == '1' and int(pin['edge']) == (t + 625) // 1250
        assert int(pin['ck']) == int(t % 1250 >= 625)
    assert 2600000 - lag >= 10000
    commands = records(text, 'NTRPU_MODEL_COMMAND')
    modes = [('1000', 0, 3, 0), ('1000', 1, 2, 0x424), ('1000', 1, 1, 0x40),
             ('1000', 1, 0, 0), ('1000', 0, 2, 0), ('1000', 0, 1, 0x401),
             ('1000', 0, 0, 0x314), ('1110', 0, 0, 0x714)]
    decoded = [(c['opcode'], int(c['bg']), int(c['ba']), int(c['addr'], 16)) for c in commands]
    assert decoded == modes + ([('0000', 0, 0, 0x123)] if open_row else []) + modes
    times = [int(c['time_ps']) for c in commands]
    assert times == sorted(set(times))
    assert all(t % 1250 == 625 and int(c['edge']) == (t + 625) // 1250
               for c, t in zip(commands, times))
    assert all(700600000 <= t < int(cold[0]['time_ps']) for t in times[:8])
    if open_row:
        assert int(cold[0]['time_ps']) < times[8] < reset
    after = times[8 + open_row:]
    assert all(reset + 503200000 <= t < end for t in after)
    for offset in (0, 8 + open_row):
        assert all(times[offset+i+1] - times[offset+i] >= 20000 for i in range(6))
        assert times[offset+7] - times[offset+6] >= 200000
    initializations = [int(v) for v in re.findall(r'Initialization complete @(\d+)', text)]
    assert initializations and initializations[0] < int(cold[0]['time_ps'])
    assert initializations == sorted(set(initializations))
    assert all(t < int(cold[0]['time_ps']) or after[0] <= t < end for t in initializations)
    errors = []
    for line in text.splitlines():
        if 'ERROR:' not in line:
            continue
        match = re.fullmatch(r'.+\.(lmr_decode|zq_decode):ERROR:(.+) @(\d+)', line)
        assert match is not None, line
        kind, message, stamp = match.groups()
        assert int(stamp) in after
        assert message == ({'lmr_decode': 'All banks must be Precharged prior to cmdLMR (1 are open)',
                            'zq_decode': ' cmdZQ Failure. All banks must be Precharged.'}[kind])
        assert after.index(int(stamp)) < 7 if kind == 'lmr_decode' else int(stamp) == after[7]
        errors.append(dict(kind=kind, time_ps=int(stamp), message=message))
    assert len({e['time_ps'] for e in errors}) == len(errors)
    finish = re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)', text)
    assert finish and {int(Decimal(v) * {'ps': 1, 'ns': 1000, 'us': 1000000}[u]) for v, u in finish} == {end}
    return dict(trace_status='DIAGNOSTIC_TRACE_VALID', open_row=open_row, phase_ps=313,
                cke_lag_ps=lag, reset_low_ps=2600000, cke_low_before_release_ps=2600000-lag,
                commands=len(commands), pin_transitions=len(pins), reset_time_ps=reset,
                completed_time_ps=end, initialization_times_ps=initializations, model_errors=errors,
                result='WARM_RESET_MODEL_ERRORS_PRESENT' if errors else 'NO_ERRORS_IN_WINDOW_NOT_RESET_QUALIFICATION')


if __name__ == '__main__':
    import json
    p = argparse.ArgumentParser()
    p.add_argument('log', type=Path)
    p.add_argument('--row', type=int, choices=[0, 1], required=True)
    p.add_argument('--lag', type=int, choices=[12500, 2000000], required=True)
    a = p.parse_args()
    print(json.dumps(review(a.log.read_text(), a.row, a.lag), indent=2))
