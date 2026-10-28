"""Negative controls for the cold MIG periodic witness; never changes a log."""
import argparse
import json
from pathlib import Path
import re

from check_ntrpu_multibuffer_periodic import review
from check_ntrpu_pcie_ddr_periodic import PIN_DELAY


def controls(text):
    baseline = review(text)
    if baseline['periodic']['warning_lines'] != 0:
        raise ValueError('These controls require the zero-warning cold MIG trace')
    results = []

    def reject(name, mutated, expected):
        if mutated == text:
            raise ValueError('Ineffective mutation: '+name)
        try:
            review(mutated)
        except ValueError as error:
            if expected not in str(error):
                raise ValueError(name+' failed for an unintended reason: '+str(error)) from error
            results.append(dict(mutation=name, rejection=str(error)))
        else:
            raise ValueError('Corruption escaped: '+name)

    def first(prefix, predicate=lambda line: True):
        return next(line for line in text.splitlines()
                    if line.startswith(prefix) and predicate(line))

    config = first('NTRPU_DDR_PERIODIC_CONFIG ')
    reject('wrong_configuration', text.replace(config, config.replace('interval=200', 'interval=201'), 1),
           'exact periodic configuration')
    cas = first('NTRPU_DDR_READ_CAS ', lambda line: 'injected=1' in line)
    cas_ps = int(re.search(r'ps=(\d+)', cas)[1])
    pin = first('NTRPU_DDR_PIN_CMD ', lambda line: f'ps={cas_ps+PIN_DELAY} ' in line)
    # Move only this injected read to an unwritten BL8 block, preserving its
    # MC/pin association, bank, rank, row and exact timing. No warning is added.
    adr = int(re.search(r'adr=([0-9a-f]+)', pin)[1], 16)
    new_pin = re.sub(r'adr=[0-9a-f]+', f'adr={(adr & ~1023) | 0x3f9:05x}', pin)
    new_cas = re.sub(r'col=[0-9a-f]+', 'col=3f9', cas)
    reject('unwritten_periodic_read_without_warning',
           text.replace(cas, new_cas, 1).replace(pin, new_pin, 1),
           'warning iff physical BL8 address unwritten')
    write_lines = []
    for line in text.splitlines():
        if line.startswith('NTRPU_DDR_PIN_CMD ') and 'act_n=1 ' in line:
            address = int(re.search(r'adr=([0-9a-f]+)', line)[1], 16)
            if (address >> 14) & 7 == 4:
                continue
        write_lines.append(line)
    reject('removed_physical_write_history', '\n'.join(write_lines),
           'warning iff physical BL8 address unwritten')
    reject('missing_cas', text.replace(cas+'\n', '', 1), 'one physical READ/return per MC CAS')
    ret = first('NTRPU_DDR_READ_RETURN ', lambda line: 'injected=1' in line)
    reject('missing_return', text.replace(ret+'\n', '', 1), 'one physical READ/return per MC CAS')
    reject('periodic_leaks_to_user', text.replace(ret, ret.replace('native=0', 'native=1'), 1),
           'injected response blocked from native user data')
    ret_ps = int(re.search(r'ps=(\d+)', ret)[1])
    reject('wrong_return_latency', text.replace(ret, ret.replace(f'ps={ret_ps}', f'ps={ret_ps+5000}'), 1),
           'exact measured CAS/pin/return timing')
    rank = re.search(r'rank=([0-9a-f]+)', cas)[1]
    reject('wrong_cas_rank', text.replace(cas, cas.replace('rank='+rank, 'rank='+str((int(rank, 16)+1)%4)), 1),
           'MC CAS matches physical READ/open row')
    accept = first('NTRPU_DDR_PERIODIC ', lambda line: 'accept=1 ' in line)
    reject('missing_accept', text.replace(accept+'\n', '', 1), 'completion follows accepted request')
    summary = first('NTRPU_DDR_PERIODIC_PASS ')
    reject('wrong_terminal_count', text.replace(summary, summary.replace('returns=12', 'returns=11'), 1),
           'terminal periodic counters match all observations')
    drain = first('NTRPU_DDR_PERIODIC_DRAIN ')
    reject('missing_final_drain', text.replace(drain+'\n', '', 1), 'one final periodic drain')
    reject('unexplained_known_warning', text+'\nModel:WARNING: Reading unwritten address: C:0 BG:0 B:0 R:0 C:0 dm:0 @1\n',
           'every warning matched exactly once')
    reject('unknown_model_warning', text+'\nModel:WARNING: unexpected diagnostic\n', 'unknown model warning')
    return dict(status='PERIODIC_NEGATIVE_CONTROLS_PASS', controls=results)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', type=Path)
    args = parser.parse_args()
    print(json.dumps(controls(args.log.read_text()), indent=2))
