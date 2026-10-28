import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
for path in sorted(ROOT.glob('*/measurements.jsonl')):
    records = [json.loads(line) for line in path.read_text().splitlines()]
    selected = [r for r in records if r.get('phase') == 'decode' and r['round'] > 0]
    if not selected: continue
    n = sum(r['tokens'] for r in selected)
    result = dict(case=path.parent.name, repeats=len(selected), tps=n/sum(r['seconds'] for r in selected))
    result.update({k: 1000*sum(r[k] for r in selected)/n for k in selected[0] if k.startswith(('moe_host_', 'moe_stream_'))})
    print(json.dumps(result, indent=2))
