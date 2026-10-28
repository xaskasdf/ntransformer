"""Produce binary stage inputs from the archived RTL trace, retaining its reference."""
import argparse
import hashlib
import json
import re
from pathlib import Path
from check_ntrpu_pcie_metadata_effect_r3 import ROW, binary_reference, review


def summarize(raw, lo, hi):
    replacement = None
    words = []
    for n in range(lo, hi):
        if not (raw >> (672+n)) & 1:
            continue
        if (raw >> (656+n)) & 1:
            replacement = (raw >> (30*n)) & ((1 << 30)-1)
            words = []
        else:
            words.append((raw >> (480+11*n)) & 2047)
    value = 0 if replacement is None else (1 << 56) | (replacement << 26)
    if words:
        value |= min(len(words),2) << 24
        value |= words[0] << 13
        value |= min(sum(words),2049) << 1
    return value


def generate(evidence):
    logs=evidence['logs']
    review(logs['normal/simulate.log'], logs['mutant/simulate.log'])
    rows=[]
    previous_metadata, previous_effect = '00000000', '000000000000000'
    for line in logs['normal/simulate.log'].splitlines():
        if not line.startswith('EFFECT_ROW '):continue
        fields=ROW.fullmatch(line.replace('X','x').replace('Z','z')).groups()
        index, reset, accept, cut, raw, valid, effect, metadata, reference=fields
        if re.search('[xz]',raw):continue
        raw=int(raw,16);cut=int(cut)
        if reset=='0':want_metadata,want_effect='00000000','000000000000000'
        elif accept=='0':want_metadata,want_effect=previous_metadata,previous_effect
        else:want_metadata,want_effect=binary_reference(raw)
        assert (metadata,effect)==(want_metadata,want_effect), 'Filtering changed clocked state'
        left,right=summarize(raw,0,cut),summarize(raw,cut,16)
        rows.append(f'{reset} {accept} {left:015x} {right:015x} {raw>>688:08x} {valid} {effect} {metadata}')
        previous_metadata,previous_effect=metadata,effect
    assert len(rows)==12803
    rows += ['0 1 000000000000000 000000000000000 00000000 0 000000000000000 00000000',
             '1 0 000000000000000 000000000000000 00000000 0 000000000000000 00000000']
    return '\n'.join(rows)+'\n'


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('evidence');p.add_argument('output');a=p.parse_args()
    evidence=Path(a.evidence);out=Path(a.output)
    text=generate(json.loads(evidence.read_text()))
    with out.open('x',newline='\n') as f:f.write(text)
    manifest=dict(rows=12805,source_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        vector_sha256=hashlib.sha256(out.read_bytes()).hexdigest(),
        generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Binary inputs and original sequential-reference outputs; 12803 retained raw rows plus final reset and bubble. No X/Z gate-level equivalence claim.')
    with out.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(manifest))
