"""Recheck the restored, self-contained CDC experiment bundle."""
import argparse
import json
import re
from pathlib import Path
from review_hbm_source import require, sha
from review_hbm_cdc import bundle as integration_review, FIFO_SHA
from review_hbm_cdc_queues import bundle as queues_review
from check_hbm_cdc import check
from audit_hbm_cdc_trace import audit

PINS = {
    'ntrpu_axi_bridge_hbm_source.sv': '0454df4e72a9007b59274a0eb814b756e8b8bc90d581f4df39fee900d8d6a9f6',
    'ntrpu_axi_bridge_hbm_client.sv': 'c3ef3b58e798187f647b7faf0d7008cda47833768c4daad256e229d297bd77a5',
    'ntrpu_axi_bridge_guard_v3.sv': '0f495c618cbf4bc37356445af9b7bd0600039e7b75b3ba0fd95f06d050472e24',
    'ntrpu_axi_bridge_writer.sv': '885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6',
    'ntrpu_axi_bridge_endpoint_client.sv': '0a3c338a09046b21a71bc6eff951a98990a445310d8e39f3ec24249a6899321b',
    'ntrpu_axi_bridge_mapper.sv': 'f832d7e777cffc617499ee46e114a9aa9b6a6c77d3f059971b6836b2466f95c9',
    'ntrpu_xdma_local_master.sv': '5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc',
}


def manifest(root):
    files = json.loads((root/'FILES.json').read_text())
    require(set(files)|{'FILES.json'} == {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}, 'Bundle file set mismatch')
    for name, digest in files.items():
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()) and sha(path)==digest, 'Bundle hash mismatch: '+name)
    return len(files)


def synthesis(root):
    manifest(root)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='HBM_CDC_SYNTH_EXPORTED' and result['returncode']==0 and not result['timed_out'] and result['sources_unchanged'], 'Synthesis incomplete')
    require(not result['native_hbm_pcie_ip'] and not result['physical_timing_qualified'], 'Unsupported synthesis scope')
    require((root/'output/STATUS.txt').read_text().strip()=='HBM_CDC_SYNTHESIZED_NOT_PHYSICALLY_QUALIFIED', 'Wrong synthesis verdict')
    for name, digest in result['sources'].items():
        require(sha(root/'source'/name)==digest, 'Synthesis source mismatch')
    text=(root/'output/check_timing.rpt').read_text()
    counts={name:int(count) for name,count in re.findall(r'checking (\w+) \((\d+)\)',text)}
    require(all(counts.get(name)==0 for name in ('loops','latch_loops','no_clock','multiple_clock','unconstrained_internal_endpoints')), 'Failed structural check')
    require(counts.get('no_input_delay')==1602 and counts.get('no_output_delay')==1602, 'External interface scope changed')
    cdc=(root/'output/cdc.rpt').read_text()
    summary={rule:dict(severity=severity,count=int(count)) for rule,severity,count in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',cdc,re.M)}
    require(summary=={'CDC-3':dict(severity='Info',count=7),'CDC-6':dict(severity='Warning',count=8),'CDC-15':dict(severity='Warning',count=1458)}, 'CDC inventory changed; review required')
    require('xcvu47p-fsvh2892' in cdc and 'Synthesized' in cdc, 'Wrong target/state')
    util=(root/'output/utilization.rpt').read_text()
    row=next(line for line in util.splitlines() if re.match(r'\| ntrpu_axi_bridge_hbm_cdc\s+\|',line))
    values=[field.strip() for field in row.split('|')[1:-1]]
    require(values[2:7]==['1538','698','840','0','1598'], 'Resource inventory changed')
    return dict(status='HBM_CDC_STRUCTURE_REVIEWED',lut=1538,ff=1598,cdc=summary,missing_input_delays=1602,missing_output_delays=1602,dcp_sha256=sha(root/'output/synth.dcp'),physical_cdc_qualified=False)


def review(root):
    count=manifest(root)
    selected=json.loads((root/'SELECTION.json').read_text())
    paths={name:(root/path).resolve() for name,path in selected['phases'].items()}
    require(all(path.is_relative_to(root.resolve()) for path in paths.values()), 'Phase escapes bundle')
    results={name:json.loads((path/'RESULT.json').read_text()) for name,path in paths.items()}
    checks={}
    for name in ('integration_icarus','integration_xsim'):
        checks[name]=integration_review(paths[name])
        checks[name+'_traces']=audit(paths[name])
        for file,digest in PINS.items():
            require(sha(paths[name]/'source'/file)==digest, 'Qualified baseline changed: '+file)
    for name in ('queues_icarus','queues_xsim'):
        checks[name]=queues_review(paths[name])
    for prefix in ('integration','queues'):
        require(results[prefix+'_icarus']['sources']==results[prefix+'_xsim']['sources'], 'Simulator source drift')
    checks['synthesis']=synthesis(paths['synthesis'])
    for name,path in paths.items():
        require(sha(path/'source/ntrpu_axi_bridge_hbm_cdc.sv')==selected['cdc_sha256'], 'CDC drift: '+name)
        require(sha(path/'source/ntrpu_bridge_async_fifo.sv')==FIFO_SHA, 'FIFO drift: '+name)
    require(sha(root/'source/ntrpu_axi_bridge_hbm_cdc.sv')==selected['cdc_sha256'], 'Published CDC drift')
    checks['negative']=check(paths['integration_xsim'],paths['queues_xsim'])
    attempts=selected['unselected_attempts']
    require(len(attempts)==1, 'Missing failed attempt history')
    failed=root/attempts[0]
    manifest(failed)
    require(json.loads((failed/'RESULT.json').read_text())['status']=='FAILED','Failed reset-stimulus attempt was promoted')
    processes=json.loads((root/'PROCESS_COMPLETION.json').read_text())
    require(len(processes['phases'])==3 and not any(p['active'] for phase in processes['phases'] for p in phase['processes']), 'Private jobs still active')
    return dict(status='HBM_CDC_BUNDLE_REVIEWED',files=count,checks=checks,native_hbm_pcie_simulation=False,physical_cdc_qualified=False,board_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bundle',type=Path)
    print(json.dumps(review(parser.parse_args().bundle),indent=2))
