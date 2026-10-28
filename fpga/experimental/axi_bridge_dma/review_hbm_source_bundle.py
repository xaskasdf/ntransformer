"""Reaudit the restored HBM frontier and distinguish structural from physical gates."""
import argparse
import json
from pathlib import Path
import re
from review_hbm_source import bundle,require,sha,READER_SHA
from check_hbm_source import check

PINNED={
    'ntrpu_axi_bridge_guard_v3.sv':'0f495c618cbf4bc37356445af9b7bd0600039e7b75b3ba0fd95f06d050472e24',
    'ntrpu_axi_bridge_writer.sv':'885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6',
    'ntrpu_axi_bridge_endpoint_client.sv':'0a3c338a09046b21a71bc6eff951a98990a445310d8e39f3ec24249a6899321b',
    'ntrpu_axi_bridge_mapper.sv':'f832d7e777cffc617499ee46e114a9aa9b6a6c77d3f059971b6836b2466f95c9',
    'ntrpu_xdma_local_master.sv':'5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc',
}

def manifest(root):
    files=json.loads((root/'FILES.json').read_text())
    require(set(files)|{'FILES.json'}=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()},'File set mismatch')
    for n,h in files.items():require(sha(root/n)==h,'Changed file: '+n)
    return len(files)

def synthesis(root):
    count=manifest(root);r=json.loads((root/'RESULT.json').read_text())
    require(r['status']=='HBM_SOURCE_CLIENT_SYNTH_EXPORTED' and r['returncode']==0 and not r['timed_out'] and r['sources_unchanged'],'Synthesis incomplete')
    checks=dict(re.findall(r'checking (\w+) \((\d+)\)',(root/'output/check_timing.rpt').read_text()))
    for key in ('loops','latch_loops','no_clock','multiple_clock','unconstrained_internal_endpoints'):
        require(checks.get(key)=='0','Unmet structural check: '+key)
    require((root/'output/STATUS.txt').read_text().strip()=='HBM_SOURCE_CLIENT_SYNTHESIZED_NOT_PHYSICALLY_QUALIFIED','Missing structural verdict')
    return dict(files=count,checks=checks,checkpoint_sha256=sha(root/'output/synth.dcp'),black_boxes=0,latches=0,physical_timing_qualified=False)

def review(root):
    count=manifest(root);local=bundle(root/'local');xsim=bundle(root/'xsim');synth=synthesis(root/'synth')
    require(local['cases']==xsim['cases']==40 and local['bytes_checked']==xsim['bytes_checked'],'Simulation coverage differs')
    names=list(PINNED)+['ntrpu_axi_bridge_hbm_source.sv','ntrpu_axi_bridge_hbm_client.sv']
    for name in names:
        values=[sha(root/phase/'source'/name) for phase in ('local','xsim','synth')]+[sha(root/'source'/name)]
        require(len(set(values))==1,'Different RTL across phases: '+name)
        if name in PINNED:require(values[0]==PINNED[name],'Frozen module changed: '+name)
    require(sha(root/'local/source/ntrpu_hbm_read_frames.sv')==sha(root/'xsim/source/ntrpu_hbm_read_frames.sv')==READER_SHA,'Reader donor changed')
    require(sha(root/'local/source/tb_axi_bridge_hbm_source.sv')==sha(root/'xsim/source/tb_axi_bridge_hbm_source.sv')==sha(root/'source/tb_axi_bridge_hbm_source.sv'),'Different selected benches')
    old=root/'retained/synth-loops'
    manifest(old)
    try:synthesis(old)
    except ValueError as error:require(str(error)=='Unmet structural check: loops','Unexpected old synthesis rejection')
    else:raise ValueError('Old combinational loop was accepted')
    processes=json.loads((root/'reviews/PROCESS_COMPLETION.json').read_text())
    require(not any(p['active'] for phase in processes['phases'] for p in phase['processes']),'Private process still active')
    return dict(status='HBM_SOURCE_FRONTIER_RESTORED',files=count,local=local,xsim=xsim,synthesis=synth,
                negative_controls=check(root/'xsim'),original_loop_rejected=True,
                native_hbm_pcie_simulation=False,cdc_200_250_implemented=False,physical_timing_qualified=False,board_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('restored',type=Path)
    print(json.dumps(review(ap.parse_args().restored),indent=2))
