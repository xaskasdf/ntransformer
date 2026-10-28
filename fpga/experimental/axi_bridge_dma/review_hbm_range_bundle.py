"""Review a restored range-optimization bundle and replay its SMT queries."""
import argparse
import json
from pathlib import Path
import sys
import zipfile
from review_hbm_range import review,files,sha
from review_hbm_clocktree import review as review_physical

def clocks(path):
    return clocks_text(path.read_text())

def clocks_text(text):
    text=text.split('Intra Clock Table',1)[1].split('Inter Clock Table',1)[0]
    result={}
    for line in text.splitlines():
        fields=line.split()
        if fields and fields[0] in ('pcie_clk','reader_clk','mgmt_clk','mem_clk'):
            result[fields[0]]=dict(wns_ns=float(fields[1]),tns_ns=float(fields[2]),setup_failures=int(fields[3]),
                                   whs_ns=float(fields[5]),hold_failures=int(fields[7]))
    if len(result)!=4:raise ValueError('Incomplete per-clock timing')
    return result

def review_bundle(root):
    manifest=json.loads((root/'MANIFEST.json').read_text())
    if {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}!=set(manifest)|{'MANIFEST.json'}:raise ValueError('Bundle inventory mismatch')
    if any(sha(root/n)!=h for n,h in manifest.items()):raise ValueError('Bundle checksum mismatch')
    selection=json.loads((root/'SELECTION.json').read_text());reviews={}
    for role in ('local','xsim','physical','refine'):
        reviews[role]=review(root/selection[role])
        if reviews[role]!=json.loads((root/(role.upper()+'_REVIEW.json')).read_text()):raise ValueError('Review receipt mismatch')
    if len({r['variant_sha256'] for r in reviews.values()})!=1:raise ValueError('Different RTL across engines')
    if sha(root/selection['physical']/'physical/output/routed.dcp')!=sha(root/selection['refine']/'source/refinement_donor.dcp'):raise ValueError('Refinement not derived from selected implementation')
    proof=root/selection['proof'];files(proof);receipt=json.loads((proof/'RESULT.json').read_text())
    if receipt['source_sha256']!=reviews['local']['variant_sha256'] or receipt['assumptions'] or receipt['input_bits']!=77:raise ValueError('Proof scope mismatch')
    expected={'equivalence':'unsat','four_state_expression_model':'unsat','missing_page_check':'sat',
              'wrong_stack_bit':'sat','reject_exact_end':'sat','four_state_plain_upper_bits':'sat','four_state_missing_positive_length':'sat'}
    if {q['name']:q['result'] for q in receipt['queries']}!=expected:raise ValueError('Incomplete solver evidence')
    sys.path.insert(0,str(root/'deps'))
    import z3
    for query in receipt['queries']:
        solver=z3.Solver();solver.set(timeout=30000);solver.from_string((proof/(query['name']+'.smt2')).read_text())
        if str(solver.check())!=expected[query['name']]:raise ValueError('Stored SMT query does not reproduce')
        if query['result']=='sat':
            values=query['counterexample'];widths={'address':64,'length':13,'address_known_mask':64,'length_known_mask':13}
            for n,v in values.items():solver.add(z3.BitVec(n,widths[n])==v)
            if solver.check()!=z3.sat:raise ValueError('Counterexample is not a witness')
    negative=root/selection['negative'];n=json.loads((negative/'RESULT.json').read_text())
    if n['hdl_counterexamples']!=5 or len(n['evidence_controls'])!=5:raise ValueError('Missing negative controls')
    if review(negative/'fixture')!=reviews['local']:raise ValueError('Negative fixture not restored')
    for case in n['evidence_controls']:
        if not case['rejected'] or sha(negative/case['name']/Path(case['path']).name)!=case['sha256']:raise ValueError('Negative evidence changed')
    if 'WITNESSES PASS count=5' not in (negative/'run.log').read_text():raise ValueError('HDL witnesses incomplete')
    physical_negative=root/selection['physical_negative'];pn=json.loads((physical_negative/'REVIEW.json').read_text())
    if pn['count']<16 or not all(c['rejected'] for c in pn['checks']):raise ValueError('Missing physical negative controls')
    if review_physical(physical_negative/'fixture')!=reviews[selection['selected_physical']]['physical']:raise ValueError('Physical negative fixture not restored')
    for phase in json.loads((root/'PHASES.json').read_text()):
        receipt=json.loads((root/phase['backup']).read_text());archive=root/phase['archive']
        if sha(archive)!=receipt['archive_sha256'] or archive.stat().st_size!=receipt['archive_bytes']:raise ValueError('VM archive changed')
    completed=json.loads((root/'PROCESS_COMPLETION.json').read_text())
    if len(completed['jobs'])!=3:raise ValueError('Missing private process inventory')
    if any(j['runner_live'] or j['remaining_groups'] for j in completed['jobs']):raise ValueError('Private VM work remains active')
    prior=root/'provenance/20261011-hbm-clocktree.zip'
    if sha(prior)!='fa0ce3672b851d1b40bcd6c243e84911c8a123589ff52a6a9a014a8b643907f5':raise ValueError('Prior physical evidence changed')
    with zipfile.ZipFile(prior) as z:
        earlier=json.loads(z.read('SELECTION.json'))
        before=clocks_text(z.read(earlier['reopen']+'/output/audit/timing.rpt').decode())
    first=clocks(root/selection['physical']/'physical/output/audit/timing.rpt')
    after=clocks(root/selection[selection['selected_physical']]/'physical/output/audit/timing.rpt')
    expected={n:dict(before=before[n],first_route=first[n],after=after[n],setup_delta_ns=round(after[n]['wns_ns']-before[n]['wns_ns'],3)) for n in before}
    if expected!=json.loads((root/'TIMING_COMPARISON.json').read_text()):raise ValueError('Timing comparison does not match reports')
    physical=reviews[selection['selected_physical']]['physical']
    return dict(status='HBM_RANGE_BUNDLE_REVIEWED',files=len(manifest),protocol_cases=168,predicate_cases_per_engine=408928,
                smt_queries=7,hdl_counterexamples=5,evidence_negative_controls=5,physical_negative_controls=pn['count'],
                wns_ns=physical['wns_ns'],whs_ns=physical['whs_ns'],internal_timing_met=physical['internal_timing_met'],
                native_hbm_pcie_ip=False,board_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);a=ap.parse_args();print(json.dumps(review_bundle(a.root),indent=2))
