"""Review clock probe evidence without treating OOC results as board closure."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open() as f:return list(csv.DictReader(f,delimiter='\t'))
def review(root):
    manifest=json.loads((root/'FILES.json').read_text())
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual!=set(manifest)|{'FILES.json'}:raise ValueError('Evidence file set changed')
    for n,h in manifest.items():
        if sha(root/n)!=h:raise ValueError('Evidence hash mismatch: '+n)
    result=json.loads((root/'RESULT.json').read_text());mode=result['mode']
    if result['status']!='HBM_CLOCKTREE_EXPORTED' or result['returncode'] or result['timed_out'] or not result['sources_unchanged']:raise ValueError('Incomplete phase')
    if sha(root/'source/donor.dcp')!=result['donor_sha256']:raise ValueError('Donor mismatch')
    for name,digest in result['sources'].items():
        if sha(root/'source'/name)!=digest:raise ValueError('Source snapshot mismatch: '+name)
    out=root/'output';audit=out/'audit'
    if (out/'STATUS.txt').read_text().strip()!=f'HBM_CLOCKTREE_{mode}_EXPORTED':raise ValueError('Missing Tcl completion')
    if mode!='reopen':
        inserted=rows(out/'clock_insertion.tsv')
        if len(inserted)!=4 or any(r['original_loads']!=r['after_loads'] or int(r['original_loads'])<=0 for r in inserted):raise ValueError('Clock insertion changed loads')
        bindings=rows(out/'bindings.tsv')
        if Counter(r['kind'] for r in bindings)!=Counter(gray=8,capture=4,control=19):raise ValueError('Constraint group inventory changed')
        if len(rows(out/'reset_arcs.tsv'))!=12:raise ValueError('Async control arc count changed')
    paths=rows(audit/'paths.tsv')
    if Counter(r['kind'] for r in paths)!=Counter(gray=24,capture=1458,control=19):raise ValueError('CDC path coverage incomplete')
    if len({r['to'] for r in paths})!=1501:raise ValueError('Duplicate CDC endpoint')
    for r in paths:
        expected=2.222 if r['to']=='memory_watch/echo_mem_reg[0]/D' else 4.0
        if abs(float(r['requirement_ns'])-expected)>0.001:raise ValueError('Missing/overridden CDC constraint')
    if mode!='preflight' and min(float(r['slack_ns']) for r in paths)<0:raise ValueError('CDC max-delay failure')
    clocks=rows(audit/'clocks.tsv')
    expected={'clk':('pcie_clk',4.0),'reader_clk':('reader_clk',5.0),'mgmt_clk':('mgmt_clk',10.0),'mem_clk':('mem_clk',2.222)}
    if {r['port'] for r in clocks}!=set(expected) or len(clocks)!=4:raise ValueError('Clock set mismatch')
    for r in clocks:
        c,p=expected[r['port']]
        if r['clock']!=c or float(r['period_ns'])!=p or r['buffer']!='clock_probe_'+c or r['input_startpoint']!=r['port'] or int(r['clock_loads'])<1:raise ValueError('Clock source contract mismatch')
        if mode!='preflight' and (not r['site'].startswith('BUFGCE_') or r['route_status']!='ROUTED'):raise ValueError('Internal clock tree is not routed')
    reset_buffer_count=0;sync_setup=None;sync_hold=None
    if (audit/'interstage.tsv').exists():
        stages=rows(audit/'interstage.tsv')
        if len(stages)!=176 or len({(r['from'],r['to']) for r in stages})!=88:raise ValueError('Synchronizer coverage mismatch')
        if mode!='preflight' and min(float(r['slack_ns']) for r in stages)<0:raise ValueError('Synchronizer setup/hold failure')
        sync_setup=min(float(r['slack_ns']) for r in stages if r['type']=='max')
        sync_hold=min(float(r['slack_ns']) for r in stages if r['type']=='min')
        capture=rows(audit/'capture_topology.tsv')
        if len(capture)!=1458 or any(r['leaf_loads']!='1' or r['driver_ref'] not in ('RAMD32','RAMS32') for r in capture):raise ValueError('Capture topology changed')
        resets=rows(audit/'canonical_resets.tsv')
        if len(resets)!=17 or len({r['driver'] for r in resets})!=4:raise ValueError('Canonical reset count changed')
        canonical={'pcie':'endpoint/guard/cold_a_reg[2]/Q','management':'endpoint/guard/cold_m_reg[2]/Q',
                   'reader':'reader/startup/source_release_reg[2]/Q','memory':'reader/startup/memory_release_reg[2]/Q'}
        buffers={}
        if (audit/'control_buffers.tsv').exists():
            for b in rows(audit/'control_buffers.tsv'):
                if b['input_driver'] not in canonical.values() or b['ce_type']!='ASYNC' or b['clock_loads']!='0' or b['i_inverted'] not in ('0','FALSE','false'):raise ValueError('Unsafe reset distribution buffer')
                inverted=b['ce_inverted'] in ('1','TRUE','true')
                if b['ce_inverted'] not in ('0','1','TRUE','FALSE','true','false') or b['ce_ref']!=('GND' if inverted else 'VCC'):raise ValueError('Reset buffer is not always enabled')
                buffers[b['buffer']+'/O']=b['input_driver']
            reset_buffer_count=len(buffers)
        elif mode=='reopen':raise ValueError('Missing control buffer audit')
        for r in resets:
            if r['driver']!=canonical[r['domain']]:raise ValueError('Wrong canonical reset domain')
            physical=r.get('physical_driver',r['driver'])
            if buffers.get(physical,physical)!=r['driver']:raise ValueError('Wrong physical reset origin')
    elif mode!='preflight':raise ValueError('Missing topology audit')
    checks=(audit/'check_timing.rpt').read_text()
    for n in ('loops','latch_loops','no_clock','multiple_clock','unconstrained_internal_endpoints'):
        if not re.search(r'checking '+n+r' \(0\)',checks):raise ValueError('Structural issue '+n)
    if 'No ignored timing exceptions found.' not in (audit/'exceptions_ignored.rpt').read_text():raise ValueError('Ignored timing exceptions')
    skew_section=(audit/'bus_skew.rpt').read_text().split('2. Bus Skew Report Per Constraint\n')[-2]
    skew_rows=re.findall(r'^\s*(?:Slow|Fast)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*$',skew_section,re.M)
    if len(skew_rows)!=16:raise ValueError('Expected 8 private and 8 native IP bus skew bounds')
    if sum(float(r[0])==2.0 for r in skew_rows)!=8:raise ValueError('Private Gray skew bounds changed')
    if mode!='preflight' and min(float(r[2]) for r in skew_rows)<0:raise ValueError('Bus skew failure')
    gap_count=None
    if mode!='preflight':
        gaps=rows(audit/'route_gaps.tsv');gap_count=len(gaps)
        for gap in gaps:
            if gap['leaf_drivers'] or gap['ports'] not in expected or gap['net']!='clock_probe_'+expected[gap['ports']][0]+'_input':raise ValueError('Nonboundary route gap')
        if len({r['ports'] for r in gaps})!=len(gaps):raise ValueError('Duplicate boundary clock gap')
        route=(audit/'route_status.rpt').read_text()
        if not re.search(r'nets with routing errors\.*\s*:\s*0',route,re.I):raise ValueError('Route error count unavailable or nonzero')
    cdc={m[0]:dict(severity=m[1],count=int(m[2])) for m in re.findall(r'^(CDC-\d+)\s+(\w+)\s+(\d+)\s+',(audit/'cdc.rpt').read_text(),re.M)}
    if any(r['severity']=='Critical' for r in cdc.values()):raise ValueError('Critical CDC finding')
    timing=(audit/'timing.rpt').read_text()
    summary=re.search(r'WNS\(ns\).*?\n\s*[- ]+\n\s*([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)\s+\d+\s+([-\d.]+)\s+([-\d.]+)\s+(\d+)',timing,re.S)
    if not summary:raise ValueError('Timing summary missing')
    wns,tns,setup,whs,ths,hold,wpws,tpws,pulse=summary.groups()
    timing_pass=int(setup)==int(hold)==int(pulse)==0
    inventories={}
    for report in ('drc','methodology'):
        p=audit/(report+'.rpt')
        if not p.exists():
            if mode!='preflight':raise ValueError('Missing '+report+' report')
            continue
        inventory={}
        for rule,severity,description,count in re.findall(r'^\|\s*([A-Z]+-\d+)\s*\|\s*([^|]+?)\s*\|\s*([^|]+?)\s*\|\s*(\d+)\s*\|',p.read_text(),re.M):
            inventory[rule]=dict(severity=severity.strip(),count=int(count))
        if any(r['severity'] not in ('Info','Warning') for r in inventory.values()):raise ValueError('Severe '+report+' finding')
        inventories[report]=inventory
    if mode=='reopen':
        def normalized(p):return re.sub(r'^# Command Used: write_xdc .*$', '# Command Used: write_xdc <output>',p.read_text(),flags=re.M)
        if normalized(out/'before.xdc')!=normalized(out/'after.xdc'):raise ValueError('Read-only audit changed constraints')
    return dict(status='HBM_CLOCKTREE_REVIEWED',mode=mode,cdc_paths=len(paths),cdc_min_slack_ns=min(float(r['slack_ns']) for r in paths),
                bus_skew_bounds=len(skew_rows),max_private_gray_skew_ns=max(float(r[1]) for r in skew_rows if float(r[0])==2.0),clock_input_gaps=gap_count,
                clocks=len(clocks),reset_distribution_buffers=reset_buffer_count,cdc_inventory=cdc,wns_ns=float(wns),whs_ns=float(whs),wpws_ns=float(wpws),setup_failing_endpoints=int(setup),
                synchronizer_setup_slack_ns=sync_setup,synchronizer_hold_slack_ns=sync_hold,
                hold_failing_endpoints=int(hold),pulse_failing_endpoints=int(pulse),internal_timing_met=timing_pass,
                report_inventories=inventories,
                native_hbm_pcie_ip=False,board_qualified=False)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);a=ap.parse_args()
    print(json.dumps(review(a.root),indent=2))
