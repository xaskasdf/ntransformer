"""Review the two-clock Brandon integration without changing single-clock gates."""
import argparse
import json
from pathlib import Path
import re
from check_ntrpu_ffn_host import records
from check_ntrpu_ffn_transfer import review as transfer_review, mutations as transfer_mutations


def review(text, fixture, host_period_ps):
    result = transfer_review(text, fixture)
    errors = result['errors']

    def need(ok, message):
        if not ok:
            errors.append(message)

    config = records(text, 'NTRPU_ASYNC_CONFIG')
    need(host_period_ps in (4000, 6200), 'qualified host profile')
    need(len(config) == 1, 'one clock/reset configuration')
    if len(config) != 1:
        result.update(status='FAIL')
        return result
    conf = config[0]
    need(conf == dict(host_period_ps=host_period_ps, host_phase_ps=700,
                      device_period_ps=5000, reset_edges=64, time_ps=conf.get('time_ps')), 'clock/reset contract')
    need(conf.get('time_ps', 0) >= 64*max(host_period_ps, 5000), 'reset held before traffic')

    def edges(time_ps, period, phase):
        return max(0, (time_ps-phase-period//2)//period+1)

    metrics = records(text, 'NTRPU_ASYNC_METRICS')
    seen = records(text, 'NTRPU_ASYNC_IRQ_SEEN')
    ack = records(text, 'NTRPU_ASYNC_IRQ_ACK')
    for rows in (metrics, seen, ack):
        need([r.get('id') for r in rows] == [1, 2, 3], 'three clock/IRQ case records')
    for row, irq, cleared, legacy in zip(metrics, seen, ack, result.get('transfer_cases', [])):
        start = row.get('start_ps', 0)
        times = [start]
        for stage in ('load', 'compute', 'read'):
            duration = row.get(stage+'_ps', 0)
            need(duration > 0, 'positive elapsed duration '+stage)
            times.append(times[-1]+duration)
        need(start > conf['time_ps'] or start == conf['time_ps'], 'traffic after reset')
        need(row.get('end_ps') == times[-1], 'elapsed interval sum')
        for stage, a, b, old in zip(('load', 'compute', 'read'), times, times[1:],
                                   ('load_cycles', 'control_compute_cycles', 'result_cycles')):
            need(row.get('host_'+stage+'_edges') == edges(b, host_period_ps, 700)-edges(a, host_period_ps, 700),
                 'host edges reconstructed from absolute time '+stage)
            need(legacy.get(old) == edges(b, 5000, 0)-edges(a, 5000, 0),
                 'legacy cycle field uses fabric clock '+stage)
        need(times[1] < irq.get('time_ps', 0) < times[2], 'IRQ seen before published completion')
        need(cleared.get('time_ps', 0) > times[3] and 0 <= cleared.get('host_wait', -1) < 16,
             'IRQ acknowledgement after host result consumption')
    for before, after, cleared in zip(metrics, metrics[1:], ack):
        need(after['start_ps'] > cleared['time_ps'] > before['end_ps'], 'next sample after IRQ acknowledgement')
    result.update(status='FAIL' if errors else 'PASS', clock_config=conf, async_metrics=metrics,
                  irq_seen=seen, irq_ack=ack, legacy_cycle_domain='device clock, 5000ps',
                  scope='AMD asynchronous AXI4/AXI4-Lite -> unchanged FFN fabric/HLS -> behavioral RAM; '+
                        'retained completion IRQ synchronized by XPM and acknowledged through MMIO. '+
                        'Three pinned Brandon FFNs with synthetic inputs and coordinated initial reset. '+
                        'Behavioral FIFO omits synchronization delays; no physical latency/throughput, '+
                        'XDMA integration, reset supervisor, in-flight reset qualification, DDR or full inference.')
    return result


def mutations(text):
    def bump(prefix, field):
        return re.sub(r'(^'+prefix+r'[^\n]*\b'+field+r'=)(\d+)',
                      lambda m: m[1]+str(int(m[2])+1), text, count=1, flags=re.M)
    return dict(transfer_mutations(text),
                clock=bump('NTRPU_ASYNC_CONFIG', 'host_period_ps'),
                host_edges=bump('NTRPU_ASYNC_METRICS', 'host_load_edges'),
                elapsed=bump('NTRPU_ASYNC_METRICS', 'end_ps'),
                irq_missing=text.replace('NTRPU_ASYNC_IRQ_SEEN id=1 ', 'REMOVED_IRQ id=1 ', 1),
                irq_early=re.sub(r'(^NTRPU_ASYNC_IRQ_ACK id=1[^\n]*time_ps=)\d+', r'\g<1>1', text, count=1, flags=re.M))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('log', type=Path)
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--host-period-ps', type=int, required=True, choices=(4000, 6200))
    args = parser.parse_args()
    result = review(args.log.read_text(encoding='utf-8'), args.fixture, args.host_period_ps)
    print(json.dumps(result, indent=2))
    raise SystemExit(result['status'] != 'PASS')
