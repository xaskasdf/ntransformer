"""Review passive per-stage counters; require an unchanged functional trace.

Counters overlap unless explicitly partitioned. AXI pending/no-R-valid cycles
are observations, not a claim that memory stalls every MAC in those cycles.
The caller must additionally qualify numerical/ABI2 evidence and provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_ffn_host import records


def functional_trace(text):
    return '\n'.join(line for line in text.splitlines()
                     if line.startswith('NTRPU_') and not line.startswith('NTRPU_FFN_PROFILE '))+'\n'


def review(text, baseline):
    errors = []
    def need(ok, why):
        if not ok:
            errors.append(why)
    rows = records(text, 'NTRPU_FFN_PROFILE')
    expected_keys = set(('id phase start end cycles ar r aw w b bytes ar_wait r_wait r_backpressure '
                         'aw_wait w_wait b_wait lane0 lane1 both blocks done0 done1 run block_fire '
                         'block_backpressure block_empty block_neither act_read act_hls act_write '
                         'act_other act_starts act_dones').split())
    need(not re.search(r'(?mi)^\s*(Fatal:|Error:|FAIL\b)', text), 'no simulation failure')
    need(functional_trace(text) == functional_trace(baseline), 'unchanged numerical/host/IRQ/timing trace')
    need([(r.get('id'), r.get('phase')) for r in rows] ==
         [(i, p) for i in (1, 2, 3) for p in range(4)], 'twelve ordered stage records')
    for row in rows:
        need(set(row) == expected_keys, 'complete profile schema')
        need(row.get('cycles',0)>0, 'positive stage duration before computing ratios')
    if errors:
        return dict(status='FAIL', errors=errors)
    jobs = []
    host = records(text, 'NTRPU_FFN_HOST_CASE')
    memory = records(text, 'NTRPU_FFN_MEMORY_CASE')
    need(len(host) == len(memory) == 3, 'three independent completion summaries')
    for ident in (1, 2, 3):
        stages = [r for r in rows if r['id'] == ident]
        mem = next((r for r in memory if r['id'] == ident), None)
        h = next((r for r in host if r['id'] == ident), None)
        if mem is None or h is None:
            continue
        for p, row in enumerate(stages):
            c = row['cycles']
            need(c == row['end']-row['start']+1 and c > 0, 'inclusive stage interval')
            need(row['start'] > 0 and (p == 0 or row['start'] > stages[p-1]['end']), 'serial stage order')
            need(row['r'] == [3244, 3244, 180, 3312][p], 'per-stage read beats from geometry')
            need(row['bytes'] == [5760, 5760, 1472, 1024][p], 'per-stage exact output bytes')
            need(row['w']*64 == row['bytes'] and row['aw'] == row['b'] > 0, 'write drain and strobes')
            for name in ('ar', 'r', 'aw', 'w', 'b', 'ar_wait', 'r_wait', 'r_backpressure',
                         'aw_wait', 'w_wait', 'b_wait', 'lane0', 'lane1', 'both', 'run'):
                need(0 <= row[name] <= c, 'cycle counter bounds '+name)
            if p == 2:
                need(sum(row[n] for n in ('act_read', 'act_hls', 'act_write', 'act_other')) == c,
                     'SwiGLU state partition')
                need(row['act_starts'] == row['act_dones'] == 23, '23 quantized activation blocks')
                need(all(row[n] == 0 for n in ('lane0', 'lane1', 'both', 'blocks', 'done0', 'done1',
                                               'run', 'block_fire', 'block_backpressure', 'block_empty', 'block_neither')),
                     'no matrix counters in activation stage')
            else:
                blocks = 5888 if p == 3 else 5760
                need(row['blocks'] == row['block_fire'] == blocks, 'one accepted MAC block per row/K')
                need(row['done0'] == row['done1'] == blocks//2, 'each lane finishes its exact block count')
                need(row['run'] == sum(row[n] for n in ('block_fire','block_backpressure','block_empty','block_neither')),
                     'GEMV RUN handshake partition')
                need(max(0,row['lane0']+row['lane1']-c) <= row['both'] <= min(row['lane0'],row['lane1']),
                     'lane occupancy intersection')
                need(all(row[n] == 0 for n in ('act_read','act_hls','act_write','act_other','act_starts','act_dones')),
                     'no activation counters in matrix stage')
        for name in ('ar','r','aw','w','b','bytes'):
            need(sum(r[name] for r in stages) == mem[name], 'independent memory total '+name)
        stage_cycles = sum(r['cycles'] for r in stages)
        need(stage_cycles <= h['compute_cycles'] < mem['cycles'], 'hardware/controller/host interval nesting')
        jobs.append(dict(id=ident, compute_cycles=h['compute_cycles'], stage_cycles=stage_cycles,
                         controller_overhead_cycles=h['compute_cycles']-stage_cycles,
                         simulated_compute_us=h['compute_cycles']*0.005,
                         stages=[dict(**r, share_of_compute=r['cycles']/h['compute_cycles'],
                                      simulated_us=r['cycles']*0.005,
                                      read_bytes=r['r']*64,
                                      read_bytes_per_stage_cycle=r['r']*64/r['cycles'],
                                      mean_lane_occupancy=(r['lane0']+r['lane1'])/(2*r['cycles'])) for r in stages]))
    return dict(status='FAIL' if errors else 'PASS', errors=errors, jobs=jobs,
                functional_trace_sha256=hashlib.sha256(functional_trace(text).encode()).hexdigest(),
                scope='Passive RTL stage and occupancy counters at 5 ns; synthetic RAM peer and three Brandon FFNs. '
                      'No physical DDR/PCIe bandwidth or full-model throughput. Pending-R and pending-B intervals overlap compute.')


def mutations(text):
    def replace(field, value):
        return re.sub(r'(^NTRPU_FFN_PROFILE [^\n]*\b'+field+r'=)\d+',
                      lambda m:m[1]+str(value), text, count=1, flags=re.M)
    result={name:replace(name,0) for name in ('cycles','start','end','r','bytes','blocks','done0','run')}
    result['duplicate_stage']=text+'\n'+next(l for l in text.splitlines() if l.startswith('NTRPU_FFN_PROFILE '))+'\n'
    result['missing_stage']=re.sub(r'^NTRPU_FFN_PROFILE [^\n]*\n','',text,count=1,flags=re.M)
    result['act_partition']=re.sub(r'(^NTRPU_FFN_PROFILE [^\n]*phase=2 [^\n]*act_hls=)\d+',lambda m:m[1]+'0',text,count=1,flags=re.M)
    result['functional_change']=text.replace('NTRPU_FFN_MEMORY_PASS ', 'NTRPU_CHANGED_PASS ', 1)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path)
    parser.add_argument('baseline',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=review(args.log.read_text(),args.baseline.read_text())
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(result['status'])
    raise SystemExit(result['status'] != 'PASS')
