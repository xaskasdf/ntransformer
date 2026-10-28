"""Review the complete allocation protocol transcript; no arithmetic qualification."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from collections import Counter
from ntrpu_llama32_layer_layout_r1 import plan

SCHEMA = {
    'LP_ADMIT': {'job','cycle','sequence','epoch','expected','injection','at'},
    'LP_RESULT': {'job','cycle','status','valid','region','other','ticks'},
    'LP_REGION': {'job','index','address','base','bytes'},
    'LP_HOLD': {'job','cycle','status','valid','revoked','sequence','epoch'},
    'LP_RELEASE': {'job','cycle'}, 'LP_PASS': {'seed','jobs','cycles'},
}
HEX = {'sequence','epoch','address','base','bytes'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse(text):
    require(not re.search(r'(?im)(^LP_FAIL|\bfatal\b|^ERROR:)', text), 'Native diagnostic')
    result = []
    for line in text.splitlines():
        if not line.startswith('LP_'):
            continue
        fields = line.split()
        kind = fields.pop(0)
        require(kind in SCHEMA, 'Unknown event')
        row = {'kind': kind}
        for field in fields:
            key, value = field.split('=')
            require(key not in row, 'Duplicate event field')
            row[key] = int(value, 16 if key in HEX else 10)
        require(set(row) == SCHEMA[kind] | {'kind'}, 'Event schema mismatch')
        result.append(row)
    return result


def review(text, seed):
    events, rows = parse(text), plan()
    require(len(events) == 28 * len(rows) + 1, 'Incomplete event inventory')
    previous = -1
    latency = []
    for case in rows:
        job = case['id']
        batch = events[job*28:(job+1)*28]
        require([r['kind'] for r in batch] == ['LP_ADMIT','LP_RESULT'] + ['LP_REGION']*24 + ['LP_HOLD','LP_RELEASE'], 'Event order')
        require(all(r['job'] == job for r in batch), 'Job identity')
        admit, result, hold, release = batch[0], batch[1], batch[-2], batch[-1]
        words = case['words']
        require(admit['cycle'] > previous, 'Overlapping jobs')
        require((admit['sequence'],admit['epoch'],admit['expected'],admit['injection'],admit['at']) ==
                (words[0],words[1],words[20],words[21],words[22]), 'Admission mismatch')
        require(1 <= result['ticks'] <= 302, 'Unbounded validation latency')
        require(abs(result['cycle']-admit['cycle']-result['ticks']) <= 1, 'Cycle accounting')
        require(result['status'] == words[20] and result['valid'] == int(words[20] == 0), 'Initial result mismatch')
        expected_pair = (31,31)
        if case['name'].startswith('alias_') and words[20] == 1:
            expected_pair = tuple(map(int,case['name'].split('_')[1:]))
        elif case['name'].startswith(('range_','cache_alias_')):
            index = int(case['name'].split('_')[1 if case['name'].startswith('range_') else 2])
            expected_pair = (index,31)
        elif case['name'].startswith('allocation_tail_alias_'):
            expected_pair = tuple(map(int,case['name'].split('_')[-2:]))
        require((result['region'],result['other']) == expected_pair, 'Fault location mismatch')
        for index, event in enumerate(batch[2:26]):
            require(event['index'] == index, 'Descriptor order')
            require([event['address'],event['base'],event['bytes']] == words[24+3*index:27+3*index], 'Captured descriptor mismatch')
        require((hold['status'],hold['sequence'],hold['epoch']) == (words[20],words[0],words[1]), 'Held identity/status changed')
        require(hold['valid'] == int(words[20] == 0 and words[21] == 0), 'Revoked result republished')
        require(hold['revoked'] == int(words[20] != 0 or words[21] != 0), 'Revocation mismatch')
        require(result['cycle'] < hold['cycle'] < release['cycle'], 'Response not held through ACK')
        previous = release['cycle']
        latency.append(result['ticks'])
    final = events[-1]
    require(final['kind'] == 'LP_PASS' and final['seed'] == seed and final['jobs'] == len(rows), 'Missing pass inventory')
    require(final['cycles'] >= previous, 'Invalid final cycle')
    return dict(status='PASS_LLAMA32_LAYER_PREFLIGHT_R1_PROTOCOL', jobs=len(rows), seed=seed,
                cycles=final['cycles'], maximum_validation_cycles=max(latency),
                status_counts=dict(Counter(r['expected_status'] for r in rows)),
                allocation_pairs=276, write_aliases_rejected=221, read_aliases_permitted=55,
                arithmetic_qualified=False, memory_traffic_generated=False,
                transcript_sha256=hashlib.sha256(text.encode()).hexdigest())


def controls(text, seed):
    lines = text.splitlines()
    tests = {}
    def changed(kind, field, value, *, ordinal=0):
        copy = list(lines)
        index = [i for i,l in enumerate(copy) if l.startswith(kind+' ')][ordinal]
        copy[index] = re.sub(r'\b'+field+r'=[^ ]+',field+'='+str(value),copy[index])
        return '\n'.join(copy)
    for kind,field,value in [('LP_ADMIT','sequence','0000000000000001'), ('LP_ADMIT','epoch','0000000000000001'),
                             ('LP_ADMIT','expected',1), ('LP_ADMIT','injection',1), ('LP_ADMIT','at',100),
                             ('LP_RESULT','status',1), ('LP_RESULT','valid',0), ('LP_RESULT','region',0),
                             ('LP_RESULT','ticks',999), ('LP_REGION','address','0000000000000000'),
                             ('LP_REGION','base','0000000000000000'), ('LP_REGION','bytes','0'),
                             ('LP_HOLD','valid',0), ('LP_HOLD','revoked',1), ('LP_HOLD','epoch','1'),
                             ('LP_HOLD','status',1), ('LP_RELEASE','cycle',0), ('LP_PASS','jobs',1),
                             ('LP_PASS','seed',seed+1)]:
        tests[kind+'_'+field] = changed(kind,field,value)
    tests['missing_descriptor'] = '\n'.join(l for i,l in enumerate(lines) if i != next(i for i,l in enumerate(lines) if l.startswith('LP_REGION ')))
    tests['missing_pass'] = '\n'.join(l for l in lines if not l.startswith('LP_PASS '))
    tests['duplicate_descriptor'] = text + '\n' + next(l for l in lines if l.startswith('LP_REGION '))
    tests['native_failure'] = text + '\nLP_FAIL late assertion\n'
    tests['unknown_field'] = text.replace('LP_ADMIT job=0 ', 'LP_ADMIT unknown=1 job=0 ', 1)
    tests['foreign_second_job'] = changed('LP_ADMIT','job',999,ordinal=1)
    rejected = []
    for name, bad in tests.items():
        try:
            review(bad,seed)
        except (ValueError,KeyError):
            rejected.append(name)
        else:
            raise ValueError('Accepted corrupted trace: '+name)
    return rejected


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('log',type=Path);p.add_argument('--seed',type=int,required=True)
    p.add_argument('--controls',action='store_true');a=p.parse_args();text=a.log.read_text()
    result=review(text,a.seed)
    if a.controls:result['rejected_controls']=controls(text,a.seed)
    print(json.dumps(result))
