"""Protocol trace, cycle-by-cycle public equivalence and speculative-read coverage."""
import re
import check_ntrpu_attention_quantize_owner_r2 as owner

def review(text, seed):
    result=owner.review(text,seed)
    lines=[line for line in text.splitlines() if line.startswith('AQBRAM_')]
    assert len(lines)==1, 'Missing/duplicate equivalence certificate'
    match=re.fullmatch(r'AQBRAM_PASS seed=(\d+) outputs=(\d+) samples=(\d+) private_reads=(\d+) unrequested=(\d+)',lines[0])
    assert match, 'Equivalence certificate schema'
    actual,outputs,samples,reads,unrequested=map(int,match.groups())
    assert actual==seed and outputs==17
    assert samples==2*result['summary']['cycles']-1 and samples>360000
    assert reads==21693 and unrequested==1725
    result.update(public_outputs=outputs,equivalence_samples=samples,private_reads=reads,
                  unrequested_private_reads=unrequested,physical_timing_qualified=False)
    return result

def controls(text,seed):
    rejected=owner.controls(text,seed)
    line=next(line for line in text.splitlines() if line.startswith('AQBRAM_'))
    mutations={'missing-equivalence':text.replace(line,''),'duplicate-equivalence':text+'\n'+line,
       'equivalence-seed':text.replace(line,line.replace(f'seed={seed} ',f'seed={seed+1} ')),
       'equivalence-outputs':text.replace(line,line.replace('outputs=17','outputs=16')),
       'equivalence-samples':text.replace(line,re.sub(r'samples=\d+','samples=1',line)),
       'private-reads':text.replace(line,line.replace('private_reads=21693','private_reads=0')),
       'unrequested-reads':text.replace(line,line.replace('unrequested=1725','unrequested=0'))}
    for name,changed in mutations.items():
        assert changed!=text
        try:review(changed,seed)
        except AssertionError:rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    assert len(rejected)==33
    return rejected
