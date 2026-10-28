"""Negative controls for the independent CDC evidence reviewers."""
import argparse
import json
import re
from pathlib import Path
from review_hbm_cdc import review as integration_review
from review_hbm_cdc_queues import review as queues_review
from audit_hbm_cdc_trace import trace_text


def check(integration, queues):
    original = (queues / 'case-00-seed-0.log').read_text()
    watchdog = (integration / 'case-22-seed-0.log').read_text()
    queues_review(original, 0, 0)
    integration_review(watchdog, 22, 0)
    lines = original.splitlines()
    event_lines = [i for i, line in enumerate(lines) if line.startswith('HQ EVENT')]
    command_lines = [i for i, line in enumerate(lines) if line.startswith('HQ CMD')]
    controls = []

    def altered_line(index, transform):
        altered = list(lines)
        altered[index] = transform(altered[index])
        return '\n'.join(altered)

    def reject(name, text, reviewer=queues_review, case=0):
        try:
            reviewer(text, case, 0)
        except ValueError as exc:
            controls.append(dict(control=name, rejected=True, reason=str(exc)))
        else:
            raise ValueError('Altered evidence accepted: ' + name)

    reject('payload bit', altered_line(event_lines[0], lambda s: re.sub(
        r'data=([0-9a-f]+)', lambda m: 'data=' + format(int(m[1], 16) ^ 1, 'x'), s)))
    reject('command missing', altered_line(command_lines[0], lambda _: ''))
    reject('high address lost', altered_line(command_lines[0], lambda s: re.sub(
        r'addr=([0-9a-f]+)', lambda m: 'addr=' + format(int(m[1], 16) & 0xffffffff, 'x'), s)))
    cpl = next(i for i in event_lines if 'cpl=1' in lines[i])
    reject('CPL identity lost', altered_line(cpl, lambda s: re.sub(r'tag=\w+', 'tag=0', s)))
    reject('CPL byte count', altered_line(cpl, lambda s: re.sub(r'bytes=\d+', 'bytes=64', s)))
    reject('CPL status', altered_line(cpl, lambda s: re.sub(r'status=\d+', 'status=2', s)))
    shuffled = list(lines)
    same_lane = re.search(r'lane=(\d)', lines[cpl])[1]
    data = next(i for i in reversed(event_lines) if i < cpl and 'lane=' + same_lane in lines[i])
    shuffled[cpl], shuffled[data] = shuffled[data], shuffled[cpl]
    reject('CPL overtakes data', '\n'.join(shuffled))
    reject('stale replay', original + '\n' + lines[event_lines[0]])
    reject('ownership lost', original.replace('HQ RETAIN commands=2 returns=6 fault=1',
                                             'HQ RETAIN commands=0 returns=0 fault=1'))
    reject('no saturation coverage', re.sub(r'command_stalls=\d+', 'command_stalls=0', original))
    reject('watchdog evidence missing', re.sub(r'^HC WATCHDOG.*$', '', watchdog, flags=re.M), integration_review, 22)
    reject('wrong stopped level', watchdog.replace('level=0', 'level=1'), integration_review, 22)
    vcd=(integration/'case-22-seed-0/hbm_cdc.vcd').read_text()
    trace_text(vcd,22)

    def symbol(name):
        return re.search(r'\$var\s+\w+\s+1\s+(\S+)\s+'+name+r'\s+\$end',vcd)[1]

    def trace_reject(name, altered):
        try:
            trace_text(altered,22)
        except (ValueError,StopIteration) as exc:
            controls.append(dict(control=name,rejected=True,reason=str(exc)))
        else:
            raise ValueError('Altered trace accepted: '+name)

    run=symbol('reader_run')
    trace_reject('trace without clock halt',re.sub(r'^0'+re.escape(run)+r'\s*$', '',vcd,flags=re.M))
    require_point='#600000\n'
    if require_point not in vcd:raise ValueError('Missing control injection point')
    trace_reject('trace lost ownership',vcd.replace(require_point,require_point+'0'+symbol('owned')+'\n',1))
    trace_reject('trace clock still running',vcd.replace(require_point,require_point+'1'+symbol('reader_clk')+'\n',1))
    fault=symbol('reader_management_fault')
    delayed=re.sub(r'^1'+re.escape(fault)+r'\s*$', '',vcd,flags=re.M)
    if '#1200000\n' not in delayed:raise ValueError('Missing delayed-fault injection point')
    trace_reject('trace late fault',delayed.replace('#1200000\n','#1200000\n1'+fault+'\n',1))
    return dict(status='HBM_CDC_NEGATIVE_CONTROLS_PASS', rejected=len(controls), controls=controls)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('integration', type=Path)
    parser.add_argument('queues', type=Path)
    args = parser.parse_args()
    print(json.dumps(check(args.integration, args.queues), indent=2))
