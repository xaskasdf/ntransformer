from common import root
import json,statistics
for phase in ['prefill','decode']:
    means={}
    for arm in ['serial','batched']:
        rows=[json.loads(line) for run in [1,2] for line in (root/(arm+str(run)+'.log')).read_text().splitlines() if line.startswith('{')]
        selected=[r for r in rows if r['round']>=0 and r['phase']==phase]
        means[arm]=statistics.mean(r['seconds'] for r in selected)
    print(phase,means,'gain_pct',(means['serial']/means['batched']-1)*100)
for label in ['trace-baseline','trace-batched']:
    if not (root/(label+'-command.json')).exists():continue
    rows=[json.loads(line) for line in (root/(label+'.log')).read_text().splitlines() if line.startswith('{')]
    for phase in ['prefill','decode']:
        selected=[r for r in rows if r['round']>=0 and r['phase']==phase]
        print(label,phase,{k:statistics.mean(r[k] for r in selected) for k in ['host_launch','host_callback','host_wait','stream_compute','host_input','host_lookup','stream_readback']})
