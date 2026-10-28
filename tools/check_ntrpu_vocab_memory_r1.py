"""Independent complete-vocabulary output, segmentation and ownership review."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from check_ntrpu_qkv_memory_r1 import bursts, SCHEMA as QKV_SCHEMA

MANIFEST_SHA = '5b25655c14ffcccc1b6a8a05ef57e94011c2bf2e7f8fc34e52c2ecf7a789d7d1'
FIRST_TAG = 0xfffff000
SCHEMA = {k:v for k,v in QKV_SCHEMA.items() if k not in ('AR', 'R', 'REVOKE')}
SCHEMA['CONFIG'] = 'seed rows segments tail_rows k_blocks tile_k watchdog phase_limit'
HEX = {'sequence', 'epoch', 'tag', 'address', 'data'}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''): h.update(block)
    return h.hexdigest()


def load_fixture(folder):
    folder = Path(folder)
    assert sha(folder/'manifest.json') == MANIFEST_SHA
    manifest = json.loads((folder/'manifest.json').read_text())
    assert manifest['rows'] == 128256 and manifest['raw_weight_bytes_verified'] == 279085056
    assert len(manifest['segments']) == 16 and manifest['segments'][-1]['rows'] == 5376
    for name, digest in manifest['files'].items(): assert sha(folder/name) == digest, name
    words = lambda name, width: [int.from_bytes(raw[i:i+width], 'little')
        for raw in [(folder/name).read_bytes()] for i in range(0, len(raw), width)]
    return dict(manifest=manifest, fp64=words('logits-fp64.bin', 8),
                fp32=words('logits-fp32.bin', 64), logits=(folder/'logits-fp32.bin').read_bytes())


def parse(text):
    assert not re.search(r'(?im)^\s*(?:Fatal:|ERROR:|VOCABM_FAIL)', text), 'Simulator failure'
    result = []
    previous = 0
    for line in text.splitlines():
        if not line.startswith('VOCABM_'): continue
        words = line.split()
        kind = words[0][7:]
        assert kind in SCHEMA, ('Unexpected event', kind)
        fields = [word.split('=') for word in words[1:]]
        assert all(len(x) == 2 for x in fields) and [x[0] for x in fields] == SCHEMA[kind].split()
        row = {k:int(v, 16 if k in HEX else 10) for k,v in fields}
        assert all(v >= 0 for v in row.values())
        if 'cycle' in row:
            assert row['cycle'] >= previous, 'Event time moved backward'
            previous = row['cycle']
        result.append((kind, row))
    return result


def review(text, fixture):
    events = parse(text)
    assert events[0] == ('CONFIG', dict(seed=1, rows=128256, segments=16, tail_rows=5376,
        k_blocks=64, tile_k=32, watchdog=65536, phase_limit=16777216))
    assert events[-1][0] == 'PASS'
    assert sum(k == 'CONFIG' for k,r in events) == sum(k == 'PASS' for k,r in events) == 1
    job_ids = [row['job'] for kind,row in events if kind == 'BEGIN']
    assert job_ids == [1,2,3]
    jobs, phases = [], []
    for job, mode in ((1,1), (2,2), (3,0)):
        part = [(k,r) for k,r in events if r.get('job') == job]
        assert part[0][0] == 'BEGIN' and part[-1][0] == 'END'
        by_kind = defaultdict(list)
        for kind,row in part: by_kind[kind].append(row)
        for name in ('BEGIN', 'COMPLETE', 'PUBLIC', 'SOURCE_ACK', 'END'):
            assert len(by_kind[name]) == 1, (job,name)
        begin, complete, public, release, end = (by_kind[name][0] for name in ('BEGIN', 'COMPLETE', 'PUBLIC', 'SOURCE_ACK', 'END'))
        tag = FIRST_TAG+(4096 if job == 3 else 0)
        assert (begin['mode'], begin['sample'], begin['target'], begin['tag']) == (mode,0,0,tag)
        assert begin['sequence'] == public['sequence'] == 0x1234567800000000+job
        assert begin['epoch'] == public['epoch'] == 0xfedcba9800000000+job
        assert begin['cycle'] < complete['cycle'] < public['cycle'] <= release['cycle'] < end['cycle']
        assert complete['elapsed'] == complete['cycle']-begin['cycle'] and complete['mode'] == mode
        status = 1 if mode == 1 else 7 if mode == 2 else 0
        assert (public['status'], public['valid'], public['revoked']) == (status,int(mode == 0),0)
        expected_phases = [] if mode == 1 else [0] if mode == 2 else list(range(16))
        assert all(row['phase'] in expected_phases for kind,row in part if 'phase' in row)
        assert [r['phase'] for r in by_kind['MATRIX']] == expected_phases
        assert [r['phase'] for r in by_kind['MATRIX_ACK']] == expected_phases
        assert len(by_kind['CANCEL']) == len(by_kind['CHILD_CANCEL']) == int(mode == 2)
        if mode == 2:
            assert by_kind['CANCEL'][0]['phase'] == by_kind['CHILD_CANCEL'][0]['phase'] == 0
            assert by_kind['CANCEL'][0]['reads'] >= 70
        if mode == 1:
            assert {k for k,r in part} == {'BEGIN', 'COMPLETE', 'PUBLIC', 'SOURCE_ACK', 'END'}
        for phase in expected_phases:
            phase_events = [(k,r) for k,r in part if r.get('phase') == phase]
            pe = defaultdict(list)
            for kind,row in phase_events: pe[kind].append(row)
            for name in ('MATRIX','MATRIX_DONE','MATRIX_ACK'): assert len(pe[name]) == 1
            start, done, ack = (pe[name][0] for name in ('MATRIX','MATRIX_DONE','MATRIX_ACK'))
            rows = 5376 if phase == 15 else 8192
            cancelled = mode == 2
            groups = 1 if cancelled else rows//32
            assert start['tag'] == done['tag'] == tag+phase*256
            assert start['cycle'] < done['cycle'] <= ack['cycle'] <= release['cycle']
            assert done['elapsed'] == done['cycle']-start['cycle'] < 16777216
            assert (done['status'],done['rows'],done['bytes']) == ((7,0,0) if cancelled else (0,rows,rows*4))
            assert ack['public'] == int(cancelled or phase == 15)
            if cancelled or phase == 15:
                assert public['cycle'] <= ack['cycle']
            else:
                assert ack['cycle'] < by_kind['MATRIX'][phase+1]['cycle']
            assert len(pe['GROUP']) == len(pe['GROUP_DONE']) == groups
            assert len(pe['TILE']) == (1 if cancelled else groups*2)
            for group in range(groups):
                g,gd = pe['GROUP'][group],pe['GROUP_DONE'][group]
                assert (g['group'],g['tag']) == (group,tag+phase*256+group)
                assert (gd['group'],gd['status'],gd['bytes']) == (group,7 if cancelled else 0,0 if cancelled else 128)
                assert start['cycle'] < g['cycle'] < gd['cycle'] < done['cycle']
                if group: assert pe['GROUP_DONE'][group-1]['cycle'] < g['cycle']
            for i,tile in enumerate(pe['TILE']):
                assert (tile['tile'],tile['address']) == (i,0x100000000+4032+phase*17825792+i*34816)
                assert pe['GROUP'][i//2]['cycle'] < tile['cycle'] < pe['GROUP_DONE'][i//2]['cycle']
            assert len(pe['MAC']) == (0 if cancelled else rows)
            for i,mac in enumerate(pe['MAC']):
                assert (mac['row'],mac['data']) == (i,fixture['fp64'][phase*8192+i])
                assert pe['GROUP'][i//32]['cycle'] < mac['cycle'] < pe['GROUP_DONE'][i//32]['cycle']
            expected_aw = []
            if not cancelled:
                for g in range(groups): expected_aw += bursts(0xa000000000+4032+phase*32768+g*128,128)
            assert len(pe['AW']) == len(pe['B']) == len(expected_aw)
            expected_last = [int(i == count-1) for address,count in expected_aw for i in range(count)]
            assert len(pe['W']) == len(expected_last) == (0 if cancelled else rows//16)
            for i,(address,count) in enumerate(expected_aw):
                aw,reply = pe['AW'][i],pe['B'][i]
                assert (aw['aw'],aw['address'],aw['beats']) == (i,address,count)
                assert (reply['aw'],reply['response']) == (i,0)
                assert aw['cycle'] < reply['cycle'] < done['cycle']
            for i,word in enumerate(pe['W']):
                assert (word['word'],word['data'],word['last']) == (i,fixture['fp32'][phase*512+i],expected_last[i])
                assert pe['GROUP'][i//2]['cycle'] < word['cycle'] < pe['GROUP_DONE'][i//2]['cycle']
            expected = dict(groups=groups, tiles=len(pe['TILE']), results=len(pe['MAC']),
                            aw=len(pe['AW']), w=len(pe['W']), b=len(pe['B']))
            assert all(done[k] == v for k,v in expected.items())
            if cancelled:
                assert 3 <= done['ar'] <= 21 and 70 <= done['r'] <= 1152
            else:
                assert (done['ar'],done['r']) == (groups*21,groups*1152)
            phases.append(dict(job=job, phase=phase, rows=done['rows'], cycles=done['elapsed'],
                               read_beats=done['r'], write_beats=done['w'], status=done['status']))
        matrix_count = len(expected_phases)
        assert complete['matrices'] == matrix_count
        assert public['matrices'] == (16 if mode == 0 else 0)
        assert complete['good'] == (16 if mode == 0 else 0)
        assert public['bytes'] == (513024 if mode == 0 else 0)
        for key in ('ar','r','aw','w','b','results'):
            assert complete[key] == sum(row[key] for row in by_kind['MATRIX_DONE'])
        jobs.append(dict(job=job, mode=mode, cycles=complete['elapsed'], results=complete['results']))
    # Every job event must belong to the planned three commands.
    assert all(row.get('job',1) in job_ids for kind,row in events)
    total = events[-1][1]
    for key,value in dict(seed=1,jobs=3,publics=3,source_acks=3,cancels=1,matrices=17,
                          groups=4009,results=128256,aw=4134,w=8016,b=4134).items():
        assert total[key] == value, key
    for key in ('ar','r'):
        assert total[key] == sum(row[key] for kind,row in events if kind == 'MATRIX_DONE')
    assert total['cycles'] >= max(row.get('cycle',0) for kind,row in events)
    assert all(total[k] > 0 for k in ('ar_stalls','r_stalls','aw_stalls','w_stalls'))
    return dict(status='PASS_FULL_VOCAB_R1_ARITHMETIC', jobs=jobs, segments=phases,
        rows=128256, output_bytes=513024, output_sha256=hashlib.sha256(fixture['logits']).hexdigest(),
        cycles=total['cycles'], full_command_cycles=jobs[-1]['cycles'], total=total,
        live_normalization_qualified=False, physical_timing_qualified=False, whole_model_rtl_qualified=False,
        scope='Full vocabulary RTL from the CPU-crosschecked final ABI1 input. All FP64 row results and FP32 AXI writes compared; bounds rejection and active early cancellation/recovery.')


def output_files(directory, fixture):
    directory = Path(directory)
    def read(name, count):
        lines = (directory/name).read_text().splitlines()
        assert len(lines) == count and all(re.fullmatch('[0-9a-fA-F]{128}', line) for line in lines)
        return b''.join(bytes.fromhex(line)[::-1] for line in lines)
    raw = read('vocab-result.hex',8016)
    assert raw == fixture['logits']
    for phase in range(16):
        rows = 5376 if phase == 15 else 8192
        assert read('vocab-phase-%02d.hex'%phase,rows//16) == raw[phase*32768:phase*32768+rows*4]
    return dict(files=17, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def controls(text, fixture):
    variants = [('missing_pass', re.sub(r'^VOCABM_PASS.*\n?', '', text, flags=re.M)),
        ('missing_tail', re.sub(r'^VOCABM_MATRIX_DONE .*phase=15 .*\n?', '', text, flags=re.M)),
        ('lost_fp64', re.sub(r'^VOCABM_MAC .*\n?', '', text, count=1, flags=re.M)),
        ('wrong_fp32', re.sub(r'(VOCABM_W .*?data=)[0-9a-fA-F]+', r'\g<1>0', text, count=1)),
        ('missing_write', re.sub(r'^VOCABM_W .*\n?', '', text, count=1, flags=re.M)),
        ('missing_cancel', re.sub(r'^VOCABM_CHILD_CANCEL .*\n?', '', text, count=1, flags=re.M)),
        ('early_public', text.replace('valid=1 revoked=0 matrices=16', 'valid=1 revoked=0 matrices=15',1)),
        ('wrong_tag', text.replace('tag=0000000100000000','tag=0000000100000001',1)),
        ('missing_source_release', re.sub(r'^VOCABM_SOURCE_ACK .*\n?', '', text, count=1, flags=re.M)),
        ('native_error', text+'\nFatal: injected failure\n')]
    rejected=[]
    for name,bad in variants:
        assert bad != text, ('Control did not modify transcript',name)
        try: review(bad,fixture)
        except (AssertionError,ValueError,KeyError,IndexError): rejected.append(name)
        else: raise AssertionError('Corrupt transcript accepted: '+name)
    return rejected
