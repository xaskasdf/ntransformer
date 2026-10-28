import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import struct
import sys

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent

def percentile(values, q):
    values = sorted(values)
    pos = (len(values)-1)*q
    lo = int(pos); hi = math.ceil(pos)
    return values[lo] + (values[hi]-values[lo])*(pos-lo)

def digest(path):
    with path.open('rb') as stream: return hashlib.file_digest(stream, 'sha256').hexdigest()

cases = []
for n in [32,128,512]:
    label = f'p{n}-d64'
    directory = root/label
    rows = [json.loads(line) for line in (directory/'measurements.jsonl').read_text().splitlines()]
    setup = rows[0]
    assert setup['kind'] == 'setup' and setup['prompt_tokens'] == n
    phases = [r for r in rows if r['kind'] == 'phase']
    completions = [r for r in rows if r['kind'] == 'completion']
    assert len(phases) == 8 and len(completions) == 4
    command = json.loads((root/(label+'-command.json')).read_text())
    assert command['returncode'] == 0
    logs = (root/(label+'.log')).read_text()
    assert 'FAIL' not in logs
    caches = re.findall(r'expert cache: (\d+) lookups, (\d+) hits \([\d.]+%\), (\d+) evictions', logs)
    assert len(caches) == len(phases)
    last = [0,0,0]
    for row, cache in zip(phases,caches):
        curr = list(map(int,cache))
        lookups,hits,evictions = [v-u for v,u in zip(curr,last)]
        last = curr
        count = row['tokens']; calls = count*setup['layers']
        assert lookups == calls*6*3 and hits <= lookups
        assert row['routed_gpu_calls'] == calls and row['shared_gpu_calls'] == calls
        assert row['routed_cpu_calls'] == 0 and row['shared_cpu_calls'] == 0
        assert row['routed_kernel_launches'] == calls*4
        assert row['shared_kernel_launches'] == calls*3
        assert row['tail_kernel_launches'] == calls*2
        assert row['routed_upload_bytes'] == (lookups-hits)*6266880
        assert len(row['token_seconds']) == count and all(t > 0 for t in row['token_seconds'])
        assert abs(sum(row['token_seconds'])-row['seconds']) < 0.05
        row.update(cache_lookups=lookups,cache_hits=hits,cache_evictions=evictions,cache_hit_fraction=hits/lookups)
    for i,row in enumerate(completions):
        assert row['round'] == i and row['repeat_bitwise_equal']
        assert row['finite_logit_values'] == 65*setup['vocab']
        assert row['predicted_ids'] == completions[0]['predicted_ids']
        assert len(row['predicted_ids']) == 65
        expected_eos = next((j for j,token in enumerate(row['predicted_ids']) if token==setup['eos_id']),-1)
        assert row['first_eos_prediction']==expected_eos
    binary = (directory/'final-logits.f32').read_bytes()
    assert len(binary) == setup['vocab']*4
    floats = struct.unpack('<'+str(setup['vocab'])+'f',binary)
    assert all(math.isfinite(v) for v in floats)
    assert max(range(len(floats)), key=floats.__getitem__) == completions[0]['predicted_ids'][-1]
    result = dict(label=label,setup=setup,cold={},reused={},first_eos_prediction=completions[0]['first_eos_prediction'],
                  final_logits_sha256=digest(directory/'final-logits.f32'),
                  prompt_ids_sha256=digest(directory/'prompt-ids.json'),completion_sha256=digest(directory/'completion.txt'),
                  first_ttft_seconds=completions[0]['ttft_seconds'],
                  first_ready_plus_setup_ttft_seconds=setup['setup_wall_seconds']+completions[0]['ttft_seconds'],
                  reused_ttft_mean_seconds=statistics.mean(c['ttft_seconds'] for c in completions[1:]))
    resources = (root/(label+'-resources.txt')).read_text()
    def resource_int(pattern): return int(re.search(pattern,resources).group(1))
    result['process_resources'] = dict(max_rss_kib=resource_int(r'Maximum resident set size \(kbytes\):\s*(\d+)'),
        major_faults=resource_int(r'Major \(requiring I/O\) page faults:\s*(\d+)'),
        minor_faults=resource_int(r'Minor \(reclaiming a frame\) page faults:\s*(\d+)'))
    for phase in ['prefill','decode']:
        selected = [r for r in phases if r['phase']==phase]
        result['cold'][phase] = {k:v for k,v in selected[0].items() if k!='token_seconds'}
        result['cold'][phase]['tokens_per_second'] = selected[0]['tokens']/selected[0]['seconds']
        warm = selected[1:]
        durations = [r['seconds'] for r in warm]
        latencies = [t for r in warm for t in r['token_seconds']]
        result['reused'][phase] = dict(tokens_per_second=sum(r['tokens'] for r in warm)/sum(durations),
            mean_seconds=statistics.mean(durations),min_seconds=min(durations),max_seconds=max(durations),
            mean_token_ms=statistics.mean(latencies)*1000,p50_token_ms=percentile(latencies,.5)*1000,
            p95_token_ms=percentile(latencies,.95)*1000,max_token_ms=max(latencies)*1000,
            cache_hit_fraction=sum(r['cache_hits'] for r in warm)/sum(r['cache_lookups'] for r in warm),
            mean_upload_bytes=statistics.mean(r['routed_upload_bytes'] for r in warm),
            mean_evictions=statistics.mean(r['cache_evictions'] for r in warm),
            phases_mean_seconds={k:statistics.mean(r[k] for r in warm) for k in ['moe','attention','engram','hc','other','qkv','rope','compression','attention_loop','attention_tail','router','routed','shared_cpu']})
    first_eos = result['first_eos_prediction']
    if first_eos > 1:
        # Describes only the pre-EOS prefix of these fixed-work replays, not a
        # separate EOS-stopping workload with a different steady cache state.
        emission_times = [t for r in phases if r['phase']=='decode' and r['round']>0
                          for t in r['token_seconds'][:first_eos-1]]
        result['pre_eos_prefix'] = dict(non_eos_prediction_count=first_eos,
            reused_new_token_count=first_eos-1,
            reused_decode_tokens_per_second=len(emission_times)/sum(emission_times),
            reused_eos_detection_seconds=statistics.mean(r['token_seconds'][first_eos-1]
                for r in phases if r['phase']=='decode' and r['round']>0))
    (directory/'measurements-with-cache.json').write_text(json.dumps(phases,indent=2)+'\n')
    cases.append(result)

with (root/'telemetry.csv').open() as stream:
    telemetry = list(csv.DictReader(stream,skipinitialspace=True))
gpu = {}
for index in ['0','1']:
    samples = [r for r in telemetry if r['index'].strip() == index]
    def field(key):
        return [float(r[key].split()[0]) for r in samples]
    gpu[index] = dict(samples=len(samples),max_memory_mib=max(field('memory.used [MiB]')),
        mean_utilization_pct=statistics.mean(field('utilization.gpu [%]')),
        max_utilization_pct=max(field('utilization.gpu [%]')),
        mean_power_w=statistics.mean(field('power.draw [W]')),max_power_w=max(field('power.draw [W]')))
observations = [json.loads(line) for line in (root/'compute-processes.jsonl').read_text().splitlines()]
for sample in observations:
    apps = [line for line in sample['apps'].splitlines() if line.strip()]
    assert len(apps) <= 1, 'foreign GPU process during benchmark'
    for line in apps:
        assert 'GPU-67e3ec29-9bbc-da80-baf1-8f982193ee6c' in line and 'benchmark_ds41' in line
summary = dict(cases=cases,telemetry=gpu,compute_process_samples=len(observations),
               validation='all 12 rounds finite, repeated logits bitwise equal, all phases use GPU routed/shared/tail; no observed competing GPU process',
               finite_logit_values_checked=sum(4*65*c['setup']['vocab'] for c in cases))
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({c['label']:{phase:round(c['reused'][phase]['tokens_per_second'],4) for phase in ['prefill','decode']} for c in cases},indent=2))
