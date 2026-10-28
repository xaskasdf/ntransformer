from collections import OrderedDict
import json
import struct
import prepare as p
b, ROOT = p.b, p.ROOT
b.status = lambda *args, **kwargs: None
b.ENV = p.DEFAULT | dict(ASAN_OPTIONS='detect_leaks=1:halt_on_error=1', UBSAN_OPTIONS='halt_on_error=1')
b.command('compile-cache-sanitized', [p.PREFIX/'bin/x86_64-conda-linux-gnu-g++', '-std=c++20', '-O1', '-g',
    '-fsanitize=address,undefined', '-fno-omit-frame-pointer', '-I'+str(ROOT/'source/src'),
    ROOT/'source/tests/test_ds41_cache.cpp', '-o', ROOT/'cache-sanitized'])
b.command('cache-sanitized', [ROOT/'cache-sanitized'])
b.command('compile-replay-selftest', [p.PREFIX/'bin/x86_64-conda-linux-gnu-g++', '-std=c++20', '-O2',
    '-I'+str(ROOT/'source/src'), ROOT/'replay_ds41_cache.cpp', '-o', ROOT/'replay-selftest-bin'])
directory=ROOT/'replay-selftest'; directory.mkdir(exist_ok=True)
lru=OrderedDict(); records=[]
for batch, keys in enumerate([[1,2],[1,2],[3,4],[5],[1,2]],1):
    for key in keys:
        hit=key in lru
        if hit:lru.move_to_end(key)
        else:
            if len(lru)==4:lru.popitem(last=False)
            lru[key]=None
        records.append((key,batch,int(hit)))
paths=[]
for name, entries in [('first',records[:7]),('last',records[7:])]:
    path=directory/(name+'.u64');paths.append(path)
    path.write_bytes(struct.pack('<4Q',0x4e54434143484531,4,544,len(entries))+
                     b''.join(struct.pack('<3Q',*entry) for entry in entries))
for percent in [0,80]:
    label='replay-selftest-'+str(percent)
    b.command(label,[ROOT/'replay-selftest-bin',percent,*paths])
    rows=[json.loads(line) for line in (ROOT/(label+'.log')).read_text().splitlines()]
    if percent==0:assert all(row['hit_mismatches']==0 for row in rows)
    assert rows[-1]['misses']==(2 if percent==0 else 0)
bad=directory/'truncated.u64';bad.write_bytes(paths[0].read_bytes()[:-1])
assert b.command('replay-reject-truncated',[ROOT/'replay-selftest-bin',0,bad],required=False)!=0
assert 'invalid or nonmonotonic trace record' in (ROOT/'replay-reject-truncated.log').read_text()
b.save('cpu-validation.json',dict(cache_sanitizers='ASan+UBSan passed',replay_independent_lru=True,
    replay_scan_retention=True,replay_truncation_rejected=True))
print('PASS CPU sanitizer and replay validation',flush=True)
