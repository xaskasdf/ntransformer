"""Bind a new native snapshot to the exact timing-closed cached-head RTL."""
import argparse
import json
from pathlib import Path
import shutil
from prepare_ntrpu_peer_commit import once
from prepare_ntrpu_tracker_head import verify_profile,ALIAS,ADAPTER,CANDIDATE
from run_ntrpu_peer_read import sha,save

ROOT=Path(__file__).resolve().parents[1]
NEW_FILES=['fpga/tb/ntrpu_tracker_head_native_monitor.svh',
           *('tools/'+n+'_ntrpu_tracker_head_native.py' for n in ('prepare','run','check'))]
SUPPORT='tools/check_ntrpu_head_native_regions_support.py'


def bench(text):
    text=once(text,'module tb_ntrpu_peer_regions_full','module tb_ntrpu_tracker_head_native')
    text=once(text,'    initial begin #150000;',
              '    `include "ntrpu_tracker_head_native_monitor.svh"\n    initial begin #150000;')
    return once(text,'        $display("SN ALL_PASS native=%0d",NATIVE);$finish;',
                '        head_summary();\n        $display("SN ALL_PASS native=%0d",NATIVE);$finish;')


def verify(repo):
    head=verify_profile(repo)
    p=json.loads((repo/'HEAD_NATIVE_PROFILE.json').read_text())
    r=json.loads((repo/'HEAD_NATIVE_ROUTE_RESULT.json').read_text())
    review=json.loads((repo/'HEAD_NATIVE_ROUTE_REVIEW.json').read_text())
    assert sha(repo/'HEAD_NATIVE_ROUTE_RESULT.json')==p['selected_result_sha256']==review['result_sha256']
    assert sha(repo/'HEAD_NATIVE_ROUTE_REVIEW.json')==p['selected_review_sha256']
    assert review['status']=='TRACKER_HEAD_ROUTED_INTERNAL_TIMING_PASS'
    assert review['checkpoint_sha256']==p['selected_checkpoint_sha256']
    assert r['sources']==r['sources_after'] and all(sha(repo/n)==h for n,h in r['sources'].items())
    assert all(sha(repo/n)==h for n,h in p['selected_rtl'].items())
    assert {n for n,h in p['baseline_rtl'].items() if p['selected_rtl'][n]!=h}=={ALIAS,ADAPTER}
    assert p['candidate_sha256']==head['candidate_sha256']==sha(repo/CANDIDATE)
    assert sha(repo/SUPPORT)==p['regions_checker_sha256']
    assert (repo/'fpga/tb/tb_ntrpu_tracker_head_native.sv').read_text()==bench((repo/'fpga/tb/tb_ntrpu_peer_regions_full.sv').read_text())
    for suffix in ('','_run'):
        expected=(repo/f'tools/ntrpu_peer_regions_full{suffix}.tcl').read_text().replace('peer_regions_full','tracker_head_native').replace('REGIONS_NATIVE','HEAD_NATIVE')
        assert (repo/f'tools/ntrpu_tracker_head_native{suffix}.tcl').read_text()==expected
    assert not p['integrated_timing_qualified'] and not p['pcie_integrated'] and not p['board_qualified']
    return p


def prepare(selected,out):
    r=json.loads((selected/'output/result.json').read_text())
    review=json.loads((selected/'REVIEW.json').read_text())
    assert review['result_sha256']==sha(selected/'output/result.json')
    assert sha(selected/'output/suite/routed.dcp')==review['checkpoint_sha256']
    assert all(sha(selected/'output'/n)==h for n,h in json.loads((selected/'output/FILES.json').read_text()).items())
    assert r['sources']==r['sources_after'] and all(sha(selected/'repo'/n)==h for n,h in r['sources'].items())
    shutil.copytree(selected/'repo',out)
    for n in NEW_FILES:
        assert not (out/n).exists();shutil.copyfile(ROOT/n,out/n)
    assert not (out/SUPPORT).exists()
    shutil.copyfile(ROOT/'tools/check_ntrpu_peer_regions_full.py',out/SUPPORT)
    for suffix in ('','_run'):
        text=(out/f'tools/ntrpu_peer_regions_full{suffix}.tcl').read_text().replace('peer_regions_full','tracker_head_native').replace('REGIONS_NATIVE','HEAD_NATIVE')
        (out/f'tools/ntrpu_tracker_head_native{suffix}.tcl').write_text(text,encoding='utf-8',newline='\n')
    (out/'fpga/tb/tb_ntrpu_tracker_head_native.sv').write_text(bench((out/'fpga/tb/tb_ntrpu_peer_regions_full.sv').read_text()),encoding='utf-8',newline='\n')
    shutil.copyfile(selected/'output/result.json',out/'HEAD_NATIVE_ROUTE_RESULT.json')
    shutil.copyfile(selected/'REVIEW.json',out/'HEAD_NATIVE_ROUTE_REVIEW.json')
    profile=json.loads((out/'REGIONS_NATIVE_PROFILE.json').read_text())
    profile.update(baseline_rtl=profile['selected_rtl'],
        selected_rtl={p.relative_to(out).as_posix():sha(p) for directory in ('fpga/experimental','fpga/rtl') for p in (out/directory).glob('*.sv')},
        selected_result_sha256=sha(out/'HEAD_NATIVE_ROUTE_RESULT.json'),selected_review_sha256=sha(out/'HEAD_NATIVE_ROUTE_REVIEW.json'),
        selected_checkpoint_sha256=review['checkpoint_sha256'],candidate_sha256=sha(out/CANDIDATE),
        regions_checker_sha256=sha(out/SUPPORT),integrated_timing_qualified=False,pcie_integrated=False,board_qualified=False,
        qualification='New observed-top native elaboration required. Cached native IP is not inherited RTL qualification.')
    save(out/'HEAD_NATIVE_PROFILE.json',profile);verify(out)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('selected','output'):ap.add_argument(n,type=Path)
    a=ap.parse_args();prepare(a.selected.resolve(),a.output.resolve())
