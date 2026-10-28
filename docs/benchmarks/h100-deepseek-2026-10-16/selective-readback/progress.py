import json
from pathlib import Path
import statistics
root = Path(__file__).resolve().parent
print((root / 'status.json').read_text())
for validation in sorted(root.glob('nt-*-validation.json')):
    name = validation.name.removesuffix('-validation.json')
    rows = [json.loads(line) for line in (root / name / 'measurements.jsonl').read_text().splitlines()]
    out = {}
    for phase in ['prefill', 'decode']:
        values = [r for r in rows if r['kind'] == 'phase' and r['round'] >= 2 and r['phase'] == phase]
        out[phase] = dict(seconds=statistics.median(r['seconds'] for r in values),
            tokens_per_second=values[0]['tokens'] / statistics.median(r['seconds'] for r in values))
    print(name, json.dumps(out))
