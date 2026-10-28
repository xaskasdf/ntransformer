"""Review independent queries against the existing operation plan and legacy oracle."""
import re
import check_ntrpu_attention_root_guard_r1 as legacy

LAYERS = list(range(16)) + [16, 0x10000000, 0x8000000f, 0xffffffff]


def review(text, seed):
    result = legacy.review(text, seed)
    assert 'CACHEQUERY_FAIL' not in text
    rows = []
    for line in text.splitlines():
        if line.startswith('CACHEQUERY_CHECK'):
            match = re.fullmatch(r'CACHEQUERY_CHECK step=(\d+) tag=(\d+) command=(\d+) layer=([0-9a-f]{8}) valid=(\d+) prefix=(\d+) legacy_valid=(\d+) legacy_prefix=(\d+) reference_valid=(\d+) reference_prefix=(\d+)', line)
            assert match, line
            values = match.groups()
            rows.append(tuple(int(v, 16 if i == 3 else 10) for i, v in enumerate(values)))
    wanted = []
    for step, (tag, valid, fault, prefixes) in enumerate(legacy.expected()):
        for command in range(17):
            for layer in LAYERS:
                wanted.append((step, tag, command, layer, int(bool(valid) and layer < 16),
                    prefixes[layer] if layer < 16 else 0, valid, prefixes[command], valid, prefixes[command]))
    assert rows == wanted, 'Independent layer/authority/ACK order or legacy interface mismatch'
    assert len(rows) == 32640
    assert text.splitlines().count(f'CACHEQUERY_PASS seed={seed} checks=32640 steps=96') == 1
    return dict(result, queries=len(rows), invalid_layer_queries=96*17*4, legacy_outputs_compared=10,
                operation_layer_distinct=True, arithmetic_qualified=False)


def controls(text, seed):
    mutations = dict(
        missing=re.sub(r'^CACHEQUERY_CHECK[^\n]*\n', '', text, count=1, flags=re.M),
        malformed=text.replace('CACHEQUERY_CHECK step=0', 'CACHEQUERY_CHECK step=x', 1),
        invalid_layer=text.replace('layer=10000000', 'layer=00000000', 1),
        completion=text.replace('checks=32640', 'checks=32639', 1),
    )
    sample=next(line for line in text.splitlines() if line.startswith('CACHEQUERY_CHECK') and 'valid=1 prefix=1' in line)
    for name, old, new in [('layer','layer=00000000','layer=00000001'),
                          ('value','valid=1 prefix=1','valid=1 prefix=2'),
                          ('authority','valid=1 prefix=1','valid=0 prefix=1'),
                          ('operation','command=0','command=1'),
                          ('reference','reference_prefix=1','reference_prefix=2'),
                          ('tag','tag=50','tag=10')]:
        assert old in sample, (name, sample)
        mutations[name]=text.replace(sample,sample.replace(old,new,1),1)
    rejected=[]
    for name, candidate in mutations.items():
        assert candidate != text
        try:review(candidate, seed)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted corruption: '+name)
    return legacy.controls(text, seed)+rejected
