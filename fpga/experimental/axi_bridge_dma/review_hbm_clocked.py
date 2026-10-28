"""Independent, fail-closed review of canonical reset and clock-loss evidence."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
from prepare_hbm_clocked import PINNED,variants,composition

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def review_source(source):
    if sha(source/'HEAD_NATIVE_PROFILE.json')!='dcbf8e04bd18ae6fd78623f16760d85f03b6f77f18bbd75ed996bbf5e3867cc1':raise ValueError('Selected native profile changed')
    for name,digest in PINNED.items():
        if sha(source/name)!=digest:raise ValueError('Frozen source mismatch: '+name)
    profile=json.loads((source/'HEAD_NATIVE_PROFILE.json').read_text())
    for name in ('ntrpu_hbm_read_return','ntrpu_hbm_read_frames','ntrpu_hbm_read_adapter','ntrpu_hbm_read_return_cut','ntrpu_hbm_startup'):
        area='rtl' if name=='ntrpu_hbm_startup' else 'experimental'
        if sha(source/(name+'.sv'))!=profile['selected_rtl'][f'fpga/{area}/{name}.sv']:raise ValueError('Selected reader source changed: '+name)
    expected=variants((source/'ntrpu_axi_bridge_guard_v3.sv').read_text(),(source/'ntrpu_axi_bridge_hbm_client.sv').read_text())
    for name,text in zip(('ntrpu_axi_bridge_guard_v3_reset_export.sv','ntrpu_axi_bridge_hbm_client_reset_export.sv'),expected):
        if (source/name).read_text()!=text:raise ValueError('Export variant changed: '+name)
    if (source/'ntrpu_axi_bridge_hbm_clocked.sv').read_text()!=composition((source/'ntrpu_axi_bridge_hbm_client.sv').read_text(),(source/'ntrpu_hbm_read_return.sv').read_text()):raise ValueError('Composition changed')
    return True

def review_case(text,case,seed):
    if re.search(r'(?im)^\s*(?:Fatal|Error|FATAL|ERROR):|HC CHECK|HC TIMEOUT',text):raise ValueError('Simulation diagnostic')
    match=re.findall(r'HC PASS case=(\d+) seed=(\d+) comparisons=(\d+) cold_domains=(\d+) retained_lanes=(\d+)',text)
    if len(match)!=1:raise ValueError('One completion required')
    c,s,n,d,l=map(int,match[0])
    if (c,s,d,l)!=(case,seed,4,2) or n<=40:raise ValueError('Case contract mismatch')
    faults=re.findall(r'HC FAULT case=(\d+) latency_ns=([\d.]+) mem=(\d) reader=(\d) pcie=(\d)',text)
    latency=None
    if case not in (0,10):
        if len(faults)!=1 or int(faults[0][0])!=case:raise ValueError('Missing fault measurement')
        _,delay,mem,reader,pcie=faults[0];latency=float(delay)
        if not 0<=latency<901:raise ValueError('Detection out of bound')
        if case in (1,2,11,12) and mem!='1':raise ValueError('Memory watchdog did not detect loss')
        if case in (3,4) and reader!='1':raise ValueError('Reader watchdog did not detect loss')
        if case in (5,6,7) and pcie!='1':raise ValueError('PCIe guard did not detect loss')
    elif faults:raise ValueError('Unexpected fault in healthy case')
    if case==10 and text.count('HC ABSENT_RECOVERED case=10')!=1:raise ValueError('Absent-clock startup not exercised')
    return dict(case=case,seed=seed,comparisons=n,latency_ns=latency)

def review(root,packaging_audit=None):
    files=json.loads((root/'FILES.json').read_text())
    for n,h in files.items():
        if sha(root/n)!=h:raise ValueError('Evidence hash mismatch: '+n)
    result=json.loads((root/'RESULT.json').read_text())
    recovered=False
    if result['status']!='HBM_CLOCKED_EXPORTED' or not result['sources_unchanged']:
        if packaging_audit is None:raise ValueError('Runner incomplete')
        candidates=[r for r in json.loads(packaging_audit.read_text()) if r['engine']==result['engine']]
        if len(candidates)!=1:raise ValueError('One packaging audit required')
        record=candidates[0];excluded={'xsim-runner.log','synth-runner.log'}
        if result['status']!='FAILED' or result['error']!='Input sources changed' or result['sources_unchanged']:raise ValueError('Not a packaging-only failure')
        if record['bad_copies'] or set(record['changed'])!=excluded or record['source_hashes']!=result['sources']:raise ValueError('Packaging recovery source mismatch')
        if {n for n,h in result['sources'].items() if record['stage_hashes'].get(n)!=h}!=excluded:raise ValueError('A real input changed')
        if {n for n in result['sources'] if Path(n).suffix not in ('.sv','.py','.tcl','.json')}!=excluded:raise ValueError('Unexpected unclassified input')
        if any(sha(root/'source'/n)!=h for n,h in result['sources'].items()):raise ValueError('Executed snapshot changed')
        recovered=True
    review_source(root/'source')
    for step in result['steps']:
        if step['returncode'] or step['timed_out']:raise ValueError('Failed step')
    if result['engine']=='synth':
        output=root/'output'
        if (output/'STATUS.txt').read_text().strip()!='HBM_CLOCKED_CANONICAL_SYNTH_EXPORTED':raise ValueError('Missing synthesis completion')
        profile=json.loads((root/'source/HEAD_NATIVE_PROFILE.json').read_text())
        for name,expected in profile['selected_read_configs'].items():
            actual=dict(line.rstrip('\n').split('\t',1) for line in (output/name).read_text().splitlines())
            if actual!=expected:raise ValueError('Selected IP configuration changed: '+name)
        rows=list(csv.DictReader((output/'canonical_resets.tsv').open(),delimiter='\t'))
        if len(rows)!=17 or len({r['consumer'] for r in rows})!=17:raise ValueError('Incomplete canonical connectivity')
        drivers={'pcie':'endpoint/guard/cold_a_reg[2]/Q','management':'endpoint/guard/cold_m_reg[2]/Q',
                 'reader':'reader/startup/source_release_reg[2]/Q','memory':'reader/startup/memory_release_reg[2]/Q'}
        consumers={'pcie':{'transport/p_cold_rst_n','memory_watch/p_cold_rst_n','endpoint/client/cold_rst_n','endpoint/source_join/cold_rst_n'},
                   'management':{'transport/m_cold_rst_n','memory_watch/m_cold_rst_n'},
                   'reader':{'transport/r_cold_rst_n','memory_watch/r_cold_rst_n'}|{f'reader/channels[{n}].{cell}/{port}' for n in range(2) for cell,port in [('frame','rst_n'),('crossing','s_axi_aresetn')]},
                   'memory':{'memory_watch/mem_cold_rst_n'}|{f'reader/channels[{n}].{cell}/{port}' for n in range(2) for cell,port in [('adapter','rst_n'),('crossing','m_axi_aresetn')]}}
        if {(r['domain'],r['consumer']) for r in rows}!={(d,c) for d,cs in consumers.items() for c in cs}:raise ValueError('Unexpected reset consumer set')
        for row in rows:
            if row['driver']!=drivers[row['domain']]:raise ValueError('Duplicate/noncanonical reset source')
        checks=(output/'check_timing.rpt').read_text()
        for kind in ('loops','latch_loops','no_clock','multiple_clock','unconstrained_internal_endpoints'):
            if not re.search(r'checking '+kind+r' \(0\)',checks):raise ValueError('Structural issue '+kind)
        inventory={m[0]:dict(severity=m[1],count=int(m[2])) for m in re.findall(r'^(CDC-\d+)\s+(\w+)\s+(\d+)\s+', (output/'cdc.rpt').read_text(),re.M)}
        if inventory!={'CDC-3':dict(severity='Info',count=29),'CDC-6':dict(severity='Warning',count=8),
                      'CDC-9':dict(severity='Info',count=12),'CDC-15':dict(severity='Warning',count=1460)}:raise ValueError('Unexpected CDC inventory')
        if not re.search(r'Checks found:\s+0\b',(output/'drc.rpt').read_text()):raise ValueError('Unexpected DRC finding')
        timing=(output/'timing.rpt').read_text()
        if 'Timing constraints are not met.' not in timing:raise ValueError('Expected structural-only timing evidence changed')
        return dict(status='HBM_CLOCKED_SYNTH_REVIEWED',packaging_recovered=recovered,canonical_connections=len(rows),native_reader_converters=True,
                    native_hbm_pcie_ip=False,physical_timing_qualified=False,cdc_inventory=inventory,timing_constraints_met=False)
    reviews=[review_case((root/f'case-{c:02}-seed-{s}.log').read_text(),c,s) for s in range(2) for c in range(13)]
    return dict(status='HBM_CLOCKED_CONTROL_REVIEWED',packaging_recovered=recovered,engine=result['engine'],cases=len(reviews),
                max_fault_latency_ns=max(r['latency_ns'] or 0 for r in reviews),native_memory_data=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path)
    ap.add_argument('--packaging-audit',type=Path);a=ap.parse_args()
    print(json.dumps(review(a.root,a.packaging_audit),indent=2))
