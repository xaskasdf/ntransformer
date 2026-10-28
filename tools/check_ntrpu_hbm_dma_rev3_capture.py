"""Verify new rev3 RTL -> synthesis -> bounded CDC, without physical signoff."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

from check_ntrpu_hbm_dma_rev2_review import digest, require, verify_stage, check_reports, check_capture
from check_ntrpu_hbm_dma_rev2_physical import rows


def cache_registers(text, placed=False):
    regs = rows(text)
    counts = Counter(r['group'] for r in regs)
    require(set(counts) == {'window_end', 'published_end', 'source_capacity', 'target_capacity'}, 'Cache group missing')
    require(len({r['cell'] for r in regs}) == len(regs), 'Duplicate cache register')
    for r in regs:
        require(r['clock'] == 'reader_clk' and r['ref'].startswith('FD'), 'Wrong cache register/clock')
        require(re.fullmatch(re.escape(r['group']) + r'_reg\[\d+\](?:_replica.*)?', r['cell']), 'Wrong cache cell scope')
        if placed:
            require(r['loc'].startswith('SLICE_X'), 'Unplaced cache register')
    return dict(counts)


def review(behavior, synth, capture):
    out = behavior / 'output'
    result = json.loads((out / 'result.json').read_text())
    for name, sha in json.loads((out / 'FILES.json').read_text()).items():
        require(digest(out / name) == sha, 'Behavior evidence changed: ' + name)
    require(result['status'] == 'REV3_BEHAVIOR_PASS' and result['sources'] == result['sources_after'], 'Wrong behavioral donor')
    for name, sha in result['sources'].items():
        require(digest(behavior / 'repo' / name) == sha, 'Behavior source changed')
    require(result['prior_bridge_scenarios'] == 144 and result['prior_fifo_runs'] == 72 and
            result['new_directed_scenarios'] == 72 and result['vectors']['vectors'] == 10740 and
            len(result['mutation_controls']) == 4, 'Incomplete behavioral qualification')
    sr, cr = verify_stage(synth), verify_stage(capture)
    for module in ('ntrpu_peer_read_pkg', 'ntrpu_bridge_async_fifo', 'ntrpu_hbm_dma_bridge_lane', 'ntrpu_hbm_dma_bridge'):
        name = 'fpga/experimental/' + module + '.sv'
        require(sr['sources'][name] == result['sources'][name], 'Synthesized different RTL')
    require({n for n in sr['sources'] if n.endswith('.sv')} ==
            {'fpga/experimental/' + n + '.sv' for n in ('ntrpu_peer_read_pkg', 'ntrpu_bridge_async_fifo',
                'ntrpu_hbm_dma_bridge_lane', 'ntrpu_hbm_dma_bridge')}, 'Reference/test RTL included in synthesis')
    require(digest(synth / 'output/reports/synth.dcp') == digest(capture / 'repo/evidence/synth.dcp'), 'Wrong synthesized DCP')
    console = (synth / 'output/console.log').read_text()
    require('Synth 8-7137' not in console and 'Synth 8-4767' not in console, 'RAM/reset warning requires review')
    utilization = (synth / 'output/reports/utilization.rpt').read_text()
    fields = [v.strip() for v in next(l for l in utilization.splitlines() if '| ntrpu_hbm_dma_bridge ' in l).split('|')]
    resource = dict(zip(('lut_total', 'lut_logic', 'lut_ram', 'srl', 'ff'), map(int, fields[3:8])))
    require(resource['lut_total'] == resource['lut_logic'] + resource['lut_ram'] and resource['lut_ram'] == 768
            and resource['srl'] == 0 and resource['ff'] > 9268, 'Unexpected resource accounting')
    cache_text = (synth / 'output/reports/cache_registers.tsv').read_text()
    cache = cache_registers(cache_text)
    altered_caches = [cache_text.replace('\treader_clk\t', '\tdma_clk\t', 1),
                      '\n'.join(l for l in cache_text.splitlines() if not l.startswith('target_capacity\t')),
                      cache_text + cache_text.splitlines()[1] + '\n']
    for altered in altered_caches:
        try:
            cache_registers(altered)
        except ValueError:
            continue
        raise ValueError('Corrupted cache inventory passed')
    cells = rows((synth / 'output/reports/cells.tsv').read_text())
    require(sum(r['async_reg'] == '1' for r in cells) == 98, 'Changed synchronizer count')
    r = capture / 'output/reports'
    names = ('bindings.tsv', 'exceptions.rpt', 'ignored_exceptions.rpt', 'bus_skew_estimated.rpt', 'cdc.rpt', 'effective_constraints.xdc')
    texts = [(r / n).read_text() for n in names]
    cdc = check_reports(*texts)
    args = [(r / n).read_text() for n in ('data_bindings.tsv', 'data_paths.tsv', 'exceptions.rpt')]
    slack = check_capture(*args)
    require('NTRPU_BRIDGE_REV2_CAPTURE_REVIEW_DONE' in (capture / 'output/console.log').read_text(), 'Incomplete topology review')
    # Require inherited checks to reject missing paths and broad exclusions.
    mutations = [(check_reports, texts[:5] + [texts[5] + '\nset_clock_groups -asynchronous -group reader_clk -group dma_clk\n']),
                 (check_reports, ['\n'.join(texts[0].splitlines()[:-1])] + texts[1:]),
                 (check_capture, [args[0], '\n'.join(args[1].splitlines()[:-1]), args[2]])]
    for fn, altered in mutations:
        try:
            fn(*altered)
        except ValueError:
            continue
        raise ValueError('Corrupted capture evidence passed')
    return dict(status='REV3_CAPTURE_VERIFIED_NOT_PHYSICALLY_QUALIFIED', resources=resource,
                cache_registers=cache, cdc=cdc, first_stage_bits=45, capture_bits=1318, gray_buses=12,
                min_capture_estimated_slack_ns=slack, negative_controls=len(mutations) + len(altered_caches),
                synth_seconds=sr['seconds'], capture_seconds=cr['seconds'],
                constrained_dcp_sha256=digest(r / 'controls_constrained.dcp'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('behavior', 'synth', 'capture'):
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(review(a.behavior, a.synth, a.capture), indent=2))
