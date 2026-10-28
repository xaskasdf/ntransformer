"""Derive one explicit tracker delta from the sealed mapped public baseline."""
import argparse
import json
from pathlib import Path
import re
import shutil
from run_ntrpu_peer_read import sha, save

ROOT=Path(__file__).resolve().parents[1]
ALIAS='fpga/experimental/ntrpu_hbm_read_tracker.sv'
CANDIDATE='fpga/experimental/ntrpu_hbm_read_head_tracker.sv'
ADAPTER='fpga/experimental/ntrpu_hbm_read_adapter.sv'
DELTA={ALIAS,ADAPTER}
NEW_FILES=[CANDIDATE, 'fpga/tb/tb_ntrpu_tracker_head.sv', 'fpga/tb/tb_ntrpu_read_tracker.sv',
           *('tools/'+n+'_ntrpu_tracker_head'+suffix+'.py' for n,suffix in
             [('prepare',''),('run','_unit'),('check','_public'),('run','_physical'),('check','_physical')])]


def verify_profile(repo):
    p=json.loads((repo/'TRACKER_HEAD_PROFILE.json').read_text())
    result=json.loads((repo/'HEAD_BASELINE_RESULT.json').read_text())
    review=json.loads((repo/'HEAD_BASELINE_REVIEW.json').read_text())
    assert sha(repo/'HEAD_BASELINE_RESULT.json')==p['baseline_result_sha256']==review['result_sha256']
    assert sha(repo/'HEAD_BASELINE_REVIEW.json')==p['baseline_review_sha256']
    assert review['status']=='REGIONS_PUBLIC_PREFLIGHT_REVIEWED'
    assert result['sources']==result['sources_after']==p['baseline_sources']
    assert all(sha(repo/n)==h for n,h in p['baseline_sources'].items() if n not in DELTA)
    assert sha(repo/'HEAD_REFERENCE.sv')==p['baseline_sources'][ALIAS]
    expected=(repo/CANDIDATE).read_text().replace('module ntrpu_hbm_read_head_tracker #','module ntrpu_hbm_read_tracker #')
    assert (repo/ALIAS).read_text()==expected and sha(repo/CANDIDATE)==p['candidate_sha256']
    assert sha(repo/ALIAS)==p['alias_sha256']
    original=(repo/'HEAD_BASELINE_ADAPTER.sv').read_text()
    assert sha(repo/'HEAD_BASELINE_ADAPTER.sv')==p['baseline_sources'][ADAPTER]
    assert (repo/ADAPTER).read_text()==embed(original,(repo/'HEAD_REFERENCE.sv').read_text(),expected)
    assert sha(repo/ADAPTER)==p['adapter_sha256']
    unit=json.loads((repo/'HEAD_UNIT_RESULT.json').read_text())
    assert unit['sources']==unit['sources_after']
    assert all(sha(repo/n)==h for n,h in unit['sources'].items() if n not in DELTA)
    assert unit['sources'][ALIAS]==p['baseline_sources'][ALIAS]
    assert p['native_requalification_required'] and not p['native_hbm_controller']
    assert p['part']=='xcvu47p-fsvh2892-2-e' and p['clocks_ns']==[5.0,2.222]
    return p


def embed(adapter,reference,candidate):
    pattern=r'\bmodule ntrpu_hbm_read_tracker\b.*?\bendmodule\b'
    blocks=[re.findall(pattern,s,re.S) for s in (adapter,reference,candidate)]
    assert all(len(b)==1 for b in blocks) and blocks[0][0]==blocks[1][0]
    return adapter.replace(blocks[0][0],blocks[2][0])


def prepare(baseline,unit,out):
    r=json.loads((baseline/'output/result.json').read_text())
    assert r['sources']==r['sources_after']
    assert all(sha(baseline/'repo'/n)==h for n,h in r['sources'].items())
    assert all(sha(baseline/'output'/n)==h for n,h in json.loads((baseline/'output/FILES.json').read_text()).items())
    ur=json.loads((unit/'REVIEW.json').read_text());u=json.loads((unit/'output/result.json').read_text())
    assert ur['status']=='TRACKER_HEAD_UNIT_REVIEWED' and ur['result_sha256']==sha(unit/'output/result.json')
    assert u['sources']==u['sources_after'] and u['sources'][CANDIDATE]==sha(ROOT/CANDIDATE)
    assert all(sha(unit/'repo'/n)==h for n,h in u['sources'].items())
    assert all(sha(ROOT/n)==u['sources'][n] for n in NEW_FILES if n in u['sources'])
    shutil.copytree(baseline/'repo',out)
    for n in NEW_FILES:
        assert not (out/n).exists();shutil.copyfile(ROOT/n,out/n)
    shutil.copyfile(out/ALIAS,out/'HEAD_REFERENCE.sv')
    (out/ALIAS).write_text((out/CANDIDATE).read_text().replace(
        'module ntrpu_hbm_read_head_tracker #','module ntrpu_hbm_read_tracker #'),encoding='utf-8',newline='\n')
    shutil.copyfile(out/ADAPTER,out/'HEAD_BASELINE_ADAPTER.sv')
    (out/ADAPTER).write_text(embed((out/ADAPTER).read_text(),(out/'HEAD_REFERENCE.sv').read_text(),
                                (out/ALIAS).read_text()),encoding='utf-8',newline='\n')
    for source,name in [(baseline/'output/result.json','HEAD_BASELINE_RESULT.json'),
                        (baseline/'REVIEW.json','HEAD_BASELINE_REVIEW.json'),
                        (unit/'output/result.json','HEAD_UNIT_RESULT.json'),(unit/'REVIEW.json','HEAD_UNIT_REVIEW.json')]:
        shutil.copyfile(source,out/name)
    save(out/'TRACKER_HEAD_PROFILE.json',dict(baseline_result_sha256=sha(baseline/'output/result.json'),
         baseline_review_sha256=sha(baseline/'REVIEW.json'),baseline_sources=r['sources'],
         candidate_sha256=sha(out/CANDIDATE),alias_sha256=sha(out/ALIAS),
         adapter_sha256=sha(out/ADAPTER),
         unit_result_sha256=ur['result_sha256'],unit_review_sha256=sha(unit/'REVIEW.json'),
         part='xcvu47p-fsvh2892-2-e',clocks_ns=[5.0,2.222],
         native_requalification_required=True,native_hbm_controller=False,board_qualified=False,
         delta='The standalone tracker and its embedded adapter copy change identically; registered head length, empty and last-release bypass.'))
    verify_profile(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('baseline','unit','output'):ap.add_argument(n,type=Path)
    a=ap.parse_args();prepare(a.baseline.resolve(),a.unit.resolve(),a.output.resolve())
