"""Produce hardware-domain vectors, retaining every binary protocol failure.

Only the deliberate unknown-enabled-payload scenario is excluded: hardware
cannot classify X/Z as digital values. Unknown bits elsewhere were already
proved irrelevant by the four-state reference and are concretized as zero.
Every retained expected state must match a freshly executed reference.
"""
import argparse,hashlib,json
from pathlib import Path
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors


def generate(vectors,manifest):
    original=[line.split() for line in vectors.splitlines() if line.strip()]
    v=Vectors();changed_bits=0;excluded=[]
    for scenario in manifest['scenarios']:
        if scenario['name']=='fault_rc_unknown_payload':
            excluded.append(dict(name=scenario['name'],reason='Four-state-only assertion; no binary hardware input represents X/Z.'))
            continue
        v.case_begin(scenario['name'])
        selected=original[scenario['first_vector']:scenario['last_vector']+1]
        assert all(int(r[0])==scenario['id'] for r in selected)
        for row in selected[4:]:
            def value(raw):
                nonlocal changed_bits
                changed_bits+=sum(c in 'xz' for c in raw)
                return int(raw.replace('x','0').replace('z','0'),2)
            rd,cd=value(row[10]),value(row[14])
            v.step(rq_beat=(rd,int(row[11],16)),rc_beat=(cd,int(row[15],16)),
                rq_valid=int(row[8]),rq_ready=int(row[9],16),rc_valid=int(row[12]),rc_ready=int(row[13]),
                cold=int(row[2]),profile=int(row[3]),reset=int(row[4]),link=int(row[5]),
                requester=int(row[6],16),attr=int(row[7],16),expect_error=bool(int(row[17])),native_ps=int(row[1]))
            assert v.rows[-1].split()[16:]==row[16:],('concretization changed expected behavior',scenario['name'])
        v.case.update(last_vector=len(v.rows)-1,reference_fault=v.fault,
                      peak_tags=v.model.peak_tags,peak_reads=v.model.peak_reads,peak_span_bytes=v.model.peak_span_bytes)
    assert len(excluded)==1 and changed_bits>0
    return v,dict(excluded_scenarios=excluded,irrelevant_unknown_bits_concretized=changed_bits)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('input',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();text=a.input.read_text();m=json.loads(a.input.with_suffix('.json').read_text())
    assert hashlib.sha256(a.input.read_bytes()).hexdigest()==m['vectors_sha256']
    v,scope=generate(text,m)
    with a.output.open('x') as f:f.write('\n'.join(v.rows)+'\n')
    result=dict(schema=1,rows=len(v.rows),scenarios=v.scenarios,scope=scope,
        vectors_sha256=hashlib.sha256(a.output.read_bytes()).hexdigest(),
        input_vectors_sha256=m['vectors_sha256'],input_manifest_sha256=hashlib.sha256(a.input.with_suffix('.json').read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        generator_dependency_sha256=hashlib.sha256(Path(__file__).with_name('generate_ntrpu_pcie_inventory_vectors_r2.py').read_bytes()).hexdigest())
    with a.output.with_suffix('.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),sha256=result['vectors_sha256'],**scope)))
