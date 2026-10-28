"""Audit range-variant evidence without changing prior qualification rules."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from prepare_hbm_range import SOURCE_SHA, variant
from review_hbm_source import review as source_case, CASES as SOURCE_CASES
from review_hbm_cdc import review as cdc_case, CASES as CDC_CASES
from review_hbm_clocktree import review as physical_review

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def files(root):
    manifest=json.loads((root/'FILES.json').read_text())
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file() and 'work' not in p.relative_to(root).parts and 'project' not in p.relative_to(root).parts and '.Xil' not in p.parts}
    if actual!=set(manifest)|{'FILES.json'}:raise ValueError('File inventory mismatch')
    for n,h in manifest.items():
        if sha(root/n)!=h:raise ValueError('Evidence checksum mismatch: '+n)

def sources(source):
    record=json.loads((source/'RANGE_VARIANT.json').read_text())
    if record['donor_files']['ntrpu_axi_bridge_hbm_source.sv']!=SOURCE_SHA:raise ValueError('Wrong original source')
    for name,h in record['donor_files'].items():
        if name=='ntrpu_axi_bridge_hbm_source.sv':continue
        if sha(source/name)!=h:raise ValueError('Unrelated composition change: '+name)
    reference=(source/'reference_hbm_source.sv').read_text().replace('module reference_hbm_source (','module ntrpu_axi_bridge_hbm_source (')
    if hashlib.sha256(reference.encode()).hexdigest()!=SOURCE_SHA:raise ValueError('Reference changed')
    if (source/'ntrpu_axi_bridge_hbm_source.sv').read_text()!=variant(reference):raise ValueError('Unapproved source changes')
    if sha(source/'ntrpu_axi_bridge_hbm_source.sv')!=record['variant_sha256']:raise ValueError('Variant hash changed')
    return record['variant_sha256']

def review(root):
    files(root);result=json.loads((root/'RESULT.json').read_text())
    if result['status']!='HBM_RANGE_EXPORTED' or not result['sources_unchanged']:raise ValueError('Incomplete range phase')
    for n,h in result['sources'].items():
        if sha(root/'source'/n)!=h:raise ValueError('Executed snapshot changed')
    digest=sources(root/'source')
    if any(s['returncode'] or s['timed_out'] for s in result['steps']):raise ValueError('Failed tool step')
    engine=result['engine'];summary=dict(status='HBM_RANGE_REVIEWED',engine=engine,variant_sha256=digest,
                                      native_hbm_pcie_ip=False,board_qualified=False)
    if engine in ('icarus','xsim'):
        text=(root/'range-predicate.log').read_text()
        if re.findall(r'HR PASS known=(\d+) unknown=(\d+)',text)!=[('380448','28480')] or 'MISMATCH' in text:raise ValueError('Incomplete predicate comparison')
        reviewed=[]
        for suite,cases,check in [('source',SOURCE_CASES,source_case),('cdc',CDC_CASES,cdc_case)]:
            for seed in range(2):
                for case in cases:
                    reviewed.append(dict(suite=suite,**check((root/f'{suite}-case-{case:02}-seed-{seed}.log').read_text(),case,seed)))
        if result['reviews']!=[dict(suite='range',known=380448,unknown=28480)]+reviewed:raise ValueError('Review receipt differs from transactions')
        if len(result['steps'])!=(88 if engine=='icarus' else 91):raise ValueError('Incomplete simulation steps')
        summary.update(predicate_binary_cases=380448,predicate_xz_cases=28480,protocol_cases=len(reviewed),
                       bytes_checked=sum(r['bytes_checked'] for r in reviewed),retention_cases=sum(r['retained'] for r in reviewed))
    else:
        physical=physical_review(root/'physical')
        if engine=='physical':
            output=root/'synth'
            if (output/'STATUS.txt').read_text().strip()!='HBM_CLOCKED_CANONICAL_SYNTH_EXPORTED':raise ValueError('Missing synthesis completion')
            if sha(output/'synth.dcp')!=sha(root/'physical/source/donor.dcp'):raise ValueError('Wrong routed synthesis donor')
            profile=json.loads((root/'source/HEAD_NATIVE_PROFILE.json').read_text())
            for n,expected in profile['selected_read_configs'].items():
                actual=dict(line.split('\t',1) for line in (output/n).read_text().splitlines())
                if actual!=expected:raise ValueError('Native converter profile changed')
            checks=(output/'check_timing.rpt').read_text()
            for kind in ('loops','latch_loops','no_clock','multiple_clock','unconstrained_internal_endpoints'):
                if not re.search(r'checking '+kind+r' \(0\)',checks):raise ValueError('Structural issue '+kind)
        elif engine=='refine':
            if sha(root/'source/refinement_donor.dcp')!=result['refinement_donor_sha256']:raise ValueError('Refinement donor changed')
            if sha(root/'refinement/refined.dcp')!=sha(root/'physical/source/donor.dcp'):raise ValueError('Wrong refined checkpoint reopened')
            if (root/'refinement/STATUS.txt').read_text().strip()!='HBM_RANGE_REFINEMENT_EXPORTED':raise ValueError('Refinement incomplete')
            def xdc(p):return Counter(l for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith('#'))
            if xdc(root/'refinement/before.xdc')!=xdc(root/'refinement/after.xdc'):raise ValueError('Refinement changed exported constraints')
        summary['physical']=physical
    return summary

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);a=ap.parse_args();print(json.dumps(review(a.root),indent=2))
