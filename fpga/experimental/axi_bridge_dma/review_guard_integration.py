"""Offline evidence review for the frozen native-source integration experiment.

This checks preservation and report consistency, not board timing closure.
Inputs are trusted experiment archives, not arbitrary third-party Python code.
"""
import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re

PINNED = {
    'ntrpu_axi_bridge_guard_v2.sv': 'bb9a758d8cffeba6b96d238ddc295de11e126ee64dc9057a34a460bb1b054a8d',
    'ntrpu_axi_bridge_endpoint_client.sv': '0a3c338a09046b21a71bc6eff951a98990a445310d8e39f3ec24249a6899321b',
    'ntrpu_axi_bridge_writer.sv': '885509ebee499bc14700c0dbac1b9db99513e857615ba7ef1ada46925e7b4ed6',
    'ntrpu_axi_bridge_mapper.sv': 'f832d7e777cffc617499ee46e114a9aa9b6a6c77d3f059971b6836b2466f95c9',
    'ntrpu_xdma_local_master.sv': '5318eb1381e87224f0dafe7d31aec4067df800b3229a1fad6f6b4614d685d3fc',
}


def require(ok, message):
    if not ok:raise ValueError(message)


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def rows(text):return list(csv.DictReader(io.StringIO(text),delimiter='\t'))


def manifest(root):
    root=root.resolve();files=json.loads((root/'FILES.json').read_text())
    for name,digest in files.items():
        p=(root/name).resolve()
        require(p.is_relative_to(root) and p.is_file(), 'Missing or escaping manifest file: '+name)
        require(sha(p)==digest,'Hash mismatch: '+name)
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Manifest file set mismatch')
    return len(files)


def cdc(text):
    summary=re.findall(r'^CDC-(\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',text,re.M)
    require(summary==[('3','Info','3'),('9','Info','8')],'Changed scoped CDC coverage/severity')
    detail=re.findall(r'^\s*\d+\s+CDC-(\d+)\s+Info\s+.+?\s+(\d+)\s+(None|False Path|Max Delay Datapath Only)\s+(\S+)\s+(\S+)\s*$',text,re.M)
    require(len(detail)==11 and all(depth=='3' for _,depth,_,_,_ in detail),'Missing CDC detail or changed depth')
    require(Counter(kind for _,_,kind,_,_ in detail)=={'Max Delay Datapath Only':3,'False Path':8},'CDC exceptions incomplete')
    return {'CDC-3':3,'CDC-9':8}


def check_stages(before,after):
    def parse(text):
        data=rows(text);keys={(r['from'],r['to'],r['type']) for r in data}
        require(len(data)==len(keys)==56,'Synchronizer stage coverage')
        chains=['cold_m','cold_a','echo_a','echo_m','armed_a','stop_a']
        chains += [f'availability[{i}].release_{c}' for i in range(4) for c in ('m','a')]
        expected={(f'guarded/guard/{c}_reg[{i}]/C',f'guarded/guard/{c}_reg[{i+1}]/D',kind)
                  for c in chains for i in range(2) for kind in ('max','min')}
        require(keys==expected,'Wrong synchronizer stage endpoints')
        for row in data:
            expected_clock='mgmt_clk' if re.search(r'(?:cold_m|echo_m|release_m)_reg',row['from']) else 'axi_aclk_out'
            require(row['clock']==expected_clock,'Wrong interstage clock')
        return {(r['from'],r['to'],r['type']):(float(r['slack']),r['clock']) for r in data}
    a=parse(before);b=parse(after)
    require(a==b,'Interstage timing changed after the async-control cuts')
    return {kind:min(v[0] for k,v in b.items() if k[2]==kind) for kind in ('max','min')}


def cuts(text,before,after):
    data=rows(text)
    expected={(f'ntrpu_bridge_i/inst/pcie4c_ip_i/inst/{reg}/C',f'guarded/guard/availability[{n}].release_m_reg[{i}]/CLR')
              for n,reg in enumerate(('user_lnk_up_reg','user_reset_reg','user_reset_reg')) for i in range(3)}
    require(len(data)==9 and {(r['from'],r['to']) for r in data}==expected,'Native reset exception endpoints')
    for r in data:
        expected_ref='FDCE' if 'user_lnk_up_reg' in r['from'] else 'FDPE'
        require(r['source_ref']==expected_ref and r['source_clock']=='axi_aclk_out' and r['dest_clock']=='mgmt_clk' and r['before_paths']=='1','Native reset source/clock evidence')
        require(r['after_timed']=='0' and r['after_ignored']=='1','Native reset exception did not remove the timing requirement')
    def constraints(t):return Counter(line for line in t.splitlines() if re.match(r'set_(?:false_path|max_delay|min_delay|multicycle_path|clock_groups)\b',line))
    a=constraints(before);b=constraints(after)
    require(not a-b,'Existing timing exception removed')
    added=list((b-a).elements())
    require(len(added)==9 and all(line.startswith('set_false_path -from ') for line in added),'Unexpected added timing exception')
    require({tuple(re.findall(r'NAME == "([^"]+)"',line)) for line in added}==expected,'Added XDC is broader than native reset pairs')
    return len(added)


def consumers(text):
    data=rows(text);counts=Counter(r['signal'] for r in data)
    require(set(counts)=={'client_cold_rst_n','client_link_up','client_stop'},'Missing real control consumer')
    for r in data:
        require(r['endpoint'].startswith('guarded/client/') and r['endpoint_clock']=='axi_aclk_out','Client control consumer in wrong domain')
        if r['signal']=='client_cold_rst_n':require(r['driver']=='guarded/guard/cold_a_reg[2]/Q','Cold reset provenance')
        elif r['signal']=='client_stop':require(r['driver']=='guarded/guard/stop_a_reg[2]/Q','Stop provenance')
    return dict(counts)


def corruption_checks(out):
    """Ensure missing stages, broader exceptions, and concealed failures reject."""
    read=lambda n:(out/n).read_text()
    c=read('cdc.rpt');b=read('stages_before.tsv');a=read('stages_after.tsv')
    cut=read('native_cuts.tsv');x=read('before.xdc');y=read('after.xdc');control=read('control_consumers.tsv')
    checks={
        'critical_cdc':lambda:cdc(c.replace('CDC-3  Info','CDC-3  Critical',1)),
        'missing_cdc_group':lambda:cdc(c.replace('CDC-9  Info          8','CDC-9  Info          7',1)),
        'unconstrained_cdc':lambda:cdc(c.replace('Max Delay Datapath Only','None',1)),
        'missing_interstage':lambda:check_stages(b,'\n'.join(a.splitlines()[:-1])+'\n'),
        'wrong_interstage_clock':lambda:check_stages(b,a.replace('mgmt_clk','axi_aclk_out',1)),
        'cut_interstage_data':lambda:check_stages(b,a.replace('/D','/CLR',1)),
        'lost_client_consumer':lambda:consumers('\n'.join(line for line in control.splitlines() if not line.startswith('client_stop\t'))+'\n'),
        'foreign_consumer_clock':lambda:consumers(control.replace('axi_aclk_out','mgmt_clk',1)),
        'wrong_native_source':lambda:cuts(cut.replace('user_lnk_up_reg','user_reset_reg',1),x,y),
        'missing_native_cut':lambda:cuts('\n'.join(cut.splitlines()[:-1])+'\n',x,y),
        'whole_clock_cut':lambda:cuts(cut,x,y+'\nset_clock_groups -asynchronous -group mgmt_clk -group axi_aclk_out\n'),
        'removed_constraint':lambda:cuts(cut,x+'\nset_max_delay 1 -from x -to y\n',y),
    }
    rejected=[]
    for name,fn in checks.items():
        try:fn()
        except ValueError:rejected.append(name)
        else:raise ValueError('Corrupt evidence accepted: '+name)
    return rejected


def review(native,audit,mapping):
    counts={name:manifest(p) for name,p in [('native',native),('audit',audit),('mapping',mapping)]}
    for root,status in [(native,'NATIVE_GUARDED_SYNTHESIS_EXPORTED'),(audit,'NATIVE_BOUNDARY_AUDIT_EXPORTED')]:
        result=json.loads((root/'RESULT.json').read_text())
        require(result['status']==status and result['returncode']==0 and result['inputs_unchanged'] and not result['timed_out'],'Incomplete native phase')
        require(result['physical_route_executed'] is False and result['board_timing_closed'] is False,'Unsupported physical closure claim')
    require(json.loads((audit/'INPUT.json').read_text())['sha256']==sha(native/'output/native_scoped.dcp'),'Wrong checkpoint audited')
    for name,digest in PINNED.items():
        require(sha(native/'source'/name)==digest and sha(mapping/'source'/name)==digest,'Changed frozen dependency: '+name)
    info=json.loads((native/'prepared/PREPARED.json').read_text())
    require(info['provider_example_board_xdc_used'] is False and info['board_pinout_approved'] is False,'Unapproved board substitution')
    top=(native/'source/ntrpu_axi_bridge_guard_native.sv').read_text()
    require(sha(native/'source/ntrpu_axi_bridge_guard_native.sv')==info['native_top_sha256'],'Native generated top mismatch')
    require('board.probe.' not in top and len(info['private_connections'])==len(set(info['private_connections']))==35,'Incomplete private bus bindings')
    boundary={r['pin']:r for r in rows((native/'output/scoped/boundary.tsv').read_text())}
    require(boundary['guarded/clk']['driver_types']=='BUFG_GT' and boundary['guarded/clk']['clocks']=='axi_aclk_out','Native AXI clock replaced')
    require(boundary['guarded/mgmt_clk']['drivers']=='management_buffer/O' and boundary['guarded/mgmt_clk']['driver_types']=='BUFGCE' and boundary['guarded/mgmt_clk']['clocks']=='mgmt_clk','Management clock provenance')
    cells=rows((native/'output/scoped/guard_cells.tsv').read_text())
    require(sum(r['async_reg'] in ('1','TRUE') for r in cells)==42,'Missing ASYNC_REG stages')
    out=audit/'output';read=lambda name:(out/name).read_text()
    cdc_result=cdc(read('cdc.rpt'))
    stages=check_stages(read('stages_before.tsv'),read('stages_after.tsv'))
    cut_count=cuts(read('native_cuts.tsv'),read('before.xdc'),read('after.xdc'))
    control_counts=consumers(read('control_consumers.tsv'))
    timing_check=dict(re.findall(r'\d+\. checking (\w+) \((\d+)\)',read('check_timing.rpt')))
    for key in ('no_clock','multiple_clock','unconstrained_internal_endpoints','loops'):
        require(timing_check.get(key)=='0','Clock or internal timing coverage changed: '+key)
    require(timing_check.get('no_input_delay')=='1589' and timing_check.get('no_output_delay')=='511','External timing contract coverage changed')
    gates={(r['object'],r['property']):r['value'] for r in rows(read('physical_gate.tsv'))}
    require(gates.get(('refclk_ibuf','LOC'))=='' and gates.get(('placement','executed'))=='false' and gates.get(('routing','executed'))=='false','Unexpected physical gate state')
    methodology=read('methodology.rpt');drc=read('drc.rpt')
    require('TIMING-6#' not in methodology and 'TIMING-7#' not in methodology,'Untreated asynchronous guard paths remain')
    require(all(rule in drc for rule in ('AVAL-326','NSTD-1','UCIO-1')),'Board DRC evidence missing')
    require('Timing constraints are not met.' in read('timing.rpt'),'Synthesis timing verdict changed; re-review physically')
    require('No waivers found.' in (native/'output/scoped/waivers.rpt').read_text(),'Unexpected waivers')
    spec=importlib.util.spec_from_file_location('frozen_mapping_review',mapping/'source/run_mapping_local.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    checks=[module.review((mapping/f'seed-{seed}/simulation.log').read_text(),seed) for seed in range(4)]
    require(sum(c['scenario_count'] for c in checks)==44,'Mapping coverage')
    return dict(status='NATIVE_INTEGRATION_EVIDENCE_REVIEWED_BOARD_CONTRACT_PENDING',files=counts,
                mapping_scenarios=44,native_simulation_executed_this_batch=False,scoped_cdc=cdc_result,
                async_reg=42,interstage_checks=56,interstage_minimum_slack_ns=stages,
                native_reset_exceptions=cut_count,control_endpoint_counts=control_counts,
                corruptions_rejected=corruption_checks(out),
                missing_input_delays=1589,missing_output_delays=511,placement_executed=False,routing_executed=False,
                board_timing_closed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    for n in ('native','audit','mapping'):ap.add_argument(n,type=Path,nargs='?')
    ap.add_argument('--bundle',type=Path)
    a=ap.parse_args()
    if a.bundle:
        if any((a.native,a.audit,a.mapping)):ap.error('Use --bundle or three restored directories')
        total=manifest(a.bundle)
        result=review(*(a.bundle/'runs'/n for n in ('native','audit','mapping')))
        result['bundle_manifest_files']=total
    else:
        if not all((a.native,a.audit,a.mapping)):ap.error('Three restored directories required')
        result=review(a.native,a.audit,a.mapping)
    print(json.dumps(result,indent=2))
