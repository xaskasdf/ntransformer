"""Independent physical-path and retained-constraint review for the CDC block."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re

SYNTH_SHA='299a340e165ff18ab52addd769f26773718336d078c89b6b1c093ddce3690444'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def table(path):
    return list(csv.DictReader(path.read_text().splitlines(),delimiter='\t'))


def number(value):
    result=float(value)
    require(math.isfinite(result),'Non-finite timing value')
    return result


def manifest(root):
    files=json.loads((root/'FILES.json').read_text())
    require(set(files)|{'FILES.json'}=={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()},'File inventory mismatch')
    for name,digest in files.items():
        path=(root/name).resolve()
        require(path.is_relative_to(root.resolve()) and sha(path)==digest,'File hash mismatch: '+name)
    return len(files)


def expected_groups():
    groups={}
    for lane in range(2):
        for fifo,width,wclock,rclock in [('commands',136,'pcie_clk','reader_clk'),('returns',593,'reader_clk','pcie_clk')]:
            prefix=f'lanes[{lane}].{fifo}/'
            for src,dst,sc,dc in [('wg','ws1',wclock,rclock),('rg','rs1',rclock,wclock)]:
                groups[f'l{lane}_{fifo}_{src}']=dict(kind='gray',sc=sc,dc=dc,
                    endpoints={prefix+f'{dst}_reg[{bit}]/D' for bit in range(3)},prefix=prefix+src+'_reg')
            groups[f'l{lane}_{fifo}_data']=dict(kind='capture',sc=wclock,dc=rclock,
                endpoints={prefix+f'output_data_reg[{bit}]/D' for bit in range(width)},prefix=prefix+'memory_reg')
    for src,dst,sc,dc in [('request_reg','echo_r','mgmt_clk','reader_clk'),
                         ('echo_r_reg[2]','echo_m','reader_clk','mgmt_clk'),
                         ('reader_failed_reg','failed_m','reader_clk','mgmt_clk'),
                         ('armed_reg','armed_p','mgmt_clk','pcie_clk'),
                         ('management_fault_reg','failed_p','mgmt_clk','pcie_clk'),
                         ('management_fault_reg','failed_r','mgmt_clk','reader_clk'),
                         ('stop_held_reg','stop_r','pcie_clk','reader_clk')]:
        groups[dst]=dict(kind='control',sc=sc,dc=dc,endpoints={dst+'_reg[0]/D'},prefix=src+'/')
    return groups


def paths_review(rows,interstage):
    groups=expected_groups()
    require(len(rows)==1489,'Missing or duplicate CDC timing paths')
    margins={kind:[] for kind in ('gray','capture','control')}
    for name,expected in groups.items():
        selected=[row for row in rows if row['group']==name]
        require(len(selected)==len(expected['endpoints']) and {row['to'] for row in selected}==expected['endpoints'],'CDC endpoint coverage: '+name)
        for row in selected:
            require(row['kind']==expected['kind'] and row['from'].startswith(expected['prefix']),'CDC driver mismatch: '+name)
            require((row['start_clock'],row['end_clock'])==(expected['sc'],expected['dc']),'Wrong CDC clock: '+name)
            require(number(row['requirement_ns'])==4.0,'Missing/overridden max delay: '+name)
            require(0<=number(row['datapath_ns'])<=4.0 and number(row['slack_ns'])>=0,'Failed CDC path: '+name)
            margins[row['kind']].append(number(row['slack_ns']))
    pairs=set()
    for lane in range(2):
        for fifo in ('commands','returns'):
            for a,b in [('ws1','ws2'),('rs1','rs2')]:
                for bit in range(3):
                    pairs.add((f'lanes[{lane}].{fifo}/{a}_reg[{bit}]',f'lanes[{lane}].{fifo}/{b}_reg[{bit}]'))
    for chain in ('echo_r','echo_m','started_m','failed_m','armed_p','failed_p','stop_r','failed_r'):
        for bit in range(2):
            pairs.add((f'{chain}_reg[{bit}]',f'{chain}_reg[{bit+1}]'))
    expected={(a,b,kind) for a,b in pairs for kind in ('max','min')}
    require(len(interstage)==80 and {(r['from'],r['to'],r['type']) for r in interstage}==expected,'Missing/duplicate synchronizer timing')
    require(all(number(r['slack_ns'])>=0 and r['source_loc'].startswith('SLICE_') and r['dest_loc'].startswith('SLICE_') for r in interstage),'Failed/unplaced synchronizer path')
    return dict(cdc_paths=len(rows),groups=len(groups),min_slack_ns={kind:min(values) for kind,values in margins.items()},interstage_paths=len(interstage),interstage_setup_ns=min(number(r['slack_ns']) for r in interstage if r['type']=='max'),interstage_hold_ns=min(number(r['slack_ns']) for r in interstage if r['type']=='min'))


def xdc_review(xdc):
    require(not re.search(r'^\s*(set_clock_groups|set_false_path|set_multicycle_path|set_disable_timing|create_waiver)\b',xdc,re.M),'Broad cut, waiver or disabled arc')
    require(len(re.findall(r'^set_max_delay\b',xdc,re.M))==19,'Wrong max-delay group count')
    skew=[line for line in xdc.splitlines() if line.startswith('set_bus_skew')]
    require(len(skew)==8,'Wrong Gray skew group count')
    require(all(re.search(r'\s2\.000\s*$',line) for line in skew),'Changed skew budget')


def report_review(route,skew,timing):
    def nets(label):
        match=re.search(r'# of '+re.escape(label)+r'\.+\s*:\s*(\d+)\s*:',route)
        require(match is not None,'Missing route count: '+label)
        return int(match[1])
    routable=nets('routable nets');routed=nets('fully routed nets')
    require(routable>0 and routed==routable and nets('nets with routing errors')==0,'Incomplete/erroneous routing')
    require(nets('nets with explicit gaps')==3,'Changed parent clock gap count')
    summary=re.search(r'1\. Bus Skew Report Summary\n-+\n(.*?)\n2\. Bus Skew Report Per Constraint\n-+',skew,re.S)
    require(summary is not None,'Missing bus-skew summary')
    skew_rows=re.findall(r'^\s*(Slow|Fast)\s+([\d.-]+)\s+([\d.-]+)\s+([\d.-]+)\s*$',summary[1],re.M)
    require(len(skew_rows)==8,'Missing skew groups')
    for _,bound,actual,slack in skew_rows:
        require(number(bound)==2.0 and 0<=number(actual)<=2 and number(slack)>=0 and abs(number(slack)-(2-number(actual)))<.002,'Failed bus skew')
    details=re.split(r'^Id:\s*\d+\s*$',skew,flags=re.M)[1:]
    require(len(details)==8,'Missing skew detail')
    seen=set()
    for index,detail in enumerate(details):
        require(re.search(r'^Endpoints: 3\s*$',detail,re.M) and re.search(r'^Requirement: 2\.000ns\s*$',detail,re.M),'Skew endpoint/budget mismatch')
        dest=re.search(r'Endpoint Destination:\s*(lanes\[[01]\]\.(?:commands|returns)/(?:ws1|rs1)_reg)\[\d\]/D',detail)
        require(dest is not None,'Skew endpoint not a FIFO pointer')
        seen.add(dest[1])
        actual=re.search(r'Actual Bus Skew:\s*([\d.-]+)ns',detail)
        require(actual is not None and number(actual[1])==number(skew_rows[index][2]),'Skew summary/detail disagreement')
    expected={f'lanes[{lane}].{fifo}/{chain}_reg' for lane in range(2) for fifo in ('commands','returns') for chain in ('ws1','rs1')}
    require(seen==expected,'Duplicated/missing skew coverage')
    section=timing.split('| Design Timing Summary')[1].split('| Clock Summary')[0]
    matches=[line.split() for line in section.splitlines() if re.match(r'^\s*-?\d+\.\d+\s+-?\d+\.\d+\s+\d+\s+\d+',line)]
    require(len(matches)==1 and len(matches[0])==12,'Missing timing summary')
    values=list(map(number,matches[0]))
    require(all(values[i]>=0 for i in (0,4,8)) and all(values[i]==0 for i in (1,2,5,6,9,10)),'Timing failure')
    require(all(values[i]>0 for i in (3,7,11)),'Empty timing analysis')
    return dict(routable_nets=routable,routed_nets=routed,parent_clock_gaps=3,max_bus_skew_ns=max(number(r[2]) for r in skew_rows),min_bus_skew_slack_ns=min(number(r[3]) for r in skew_rows),wns_ns=values[0],whs_ns=values[4],wpws_ns=values[8])


def comparable_xdc(text):
    # Vivado records the destination filename in this one generated comment.
    # Normalize only that comment, preserving every constraint and other byte.
    normalized,count=re.subn(r'^# Command Used: write_xdc -constraints all [^\n]+/(?:before|after)\.xdc$',
                             '# Command Used: write_xdc -constraints all <audit-output>',text,flags=re.M)
    require(count==1,'Unexpected write_xdc header')
    return normalized


def review(root):
    file_count=manifest(root)
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status'] in ('HBM_CDC_PHYSICAL_EXPORTED','HBM_CDC_REOPEN_EXPORTED') and result['returncode']==0 and not result['timed_out'] and result['sources_unchanged'],'Incomplete physical phase')
    require(sha(root/'source/donor.dcp')==result['donor_sha256'],'Checkpoint provenance mismatch')
    for name,digest in result['sources'].items():
        require(sha(root/'source'/name)==digest,'Changed tool source')
    is_reopen=result['status']=='HBM_CDC_REOPEN_EXPORTED'
    if not is_reopen:
        require(result['donor_sha256']==SYNTH_SHA,'Wrong synthesis donor')
    audit=root/'output/audit'
    require((audit/'STATUS.txt').read_text().strip()=='HBM_CDC_PHYSICAL_AUDIT_EXPORTED_PENDING_REVIEW','Missing audit completion')
    checked=paths_review(table(audit/'paths.tsv'),table(audit/'interstage.tsv'))
    reports=report_review((audit/'route_status.rpt').read_text(),(audit/'bus_skew.rpt').read_text(),(audit/'timing.rpt').read_text())
    xdc=(audit/'effective.xdc').read_text()
    xdc_review(xdc)
    checks={name:int(value) for name,value in re.findall(r'checking (\w+) \((\d+)\)',(audit/'check_timing.rpt').read_text())}
    require(all(checks.get(name)==0 for name in ('loops','latch_loops','no_clock','multiple_clock','unconstrained_internal_endpoints')),'Incomplete structural timing')
    require(checks.get('no_input_delay')==1602 and checks.get('no_output_delay')==1602,'External timing scope changed')
    require('No ignored timing exceptions found.' in (audit/'exceptions_ignored.rpt').read_text(),'Ignored exception')
    coverage=re.findall(r'^\s*\d+\s+Max Delay Datapath Only\s+max_dpo=4\s+-\s+(\d+) cells\s+(\d+) pins\s+(\d+)\s+100\.00\s+100\.00\s*$',(audit/'exceptions.rpt').read_text(),re.M)
    require(len(coverage)==19 and sum(int(r[2]) for r in coverage)==1489 and all(len(set(r))==1 for r in coverage),'Incomplete constraint coverage')
    cdc={rule:(severity,int(count)) for rule,severity,count in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',(audit/'cdc.rpt').read_text(),re.M)}
    require(cdc=={'CDC-3':('Info',7),'CDC-6':('Warning',8),'CDC-15':('Warning',1458)},'Changed CDC inventory')
    def rules(name):
        return {rule:(severity,int(count)) for rule,severity,count in re.findall(r'^\|\s+([A-Z]+-\d+)\s+\|\s+(Warning|Critical Warning|Error)\s+\|[^|]+\|\s+(\d+)\s+\|',(audit/name).read_text(),re.M)}
    require(rules('drc.rpt')=={'RTSTAT-10':('Warning',1)},'Changed DRC inventory')
    require(rules('methodology.rpt')=={'TIMING-18':('Warning',1000),'XDCB-5':('Warning',7)},'Changed methodology inventory')
    reset_rows=table(audit/'reset_endpoints.tsv')
    require(bool(reset_rows),'Missing reset topology')
    for row in reset_rows:
        require(row['reset_startpoints']=={'pcie_clk':'p_cold_rst_n','reader_clk':'r_cold_rst_n','mgmt_clk':'m_cold_rst_n'}[row['clock']],'Crossed reset domains')
    if is_reopen:
        require(comparable_xdc((root/'output/before.xdc').read_text())==comparable_xdc((root/'output/after.xdc').read_text()),'Read-only audit changed constraints')
        topology=table(audit/'topology.tsv')
        require(len(topology)==1489 and {r['to'] for r in topology}=={r['to'] for r in table(audit/'paths.tsv')},'Topology coverage changed')
        require(all(r['leaf_loads']=='1' and r['driver_ref'] in ('RAMD32','RAMS32') for r in topology if r['kind']=='capture'),'RAM bypass or extra consumer')
        stages=table(audit/'first_stage_loads.tsv')
        require(len(stages)==40 and len({r['source'] for r in stages})==40,'Intermediate synchronizer fanout mismatch')
        parents=table(audit/'parent_clocks.tsv')
        require(len(parents)==3 and {r['port'] for r in parents}=={'p_clk','r_clk','mgmt_clk'} and all(r['leaf_drivers']=='0' for r in parents),'Parent clocks unexpectedly implemented')
        gaps=table(audit/'clock_gaps.tsv')
        require(len(gaps)==3 and {r['ports'] for r in gaps}=={'p_clk','r_clk','mgmt_clk'},'Non-clock route gap')
        boundary=table(audit/'boundary_nets.tsv')
        require(len(boundary)==1469 and all(r['leaf_loads']=='0' and r['ports'] for r in boundary),'Unclassified boundary DRC net')
    return dict(status='HBM_CDC_PHYSICAL_PATHS_REVIEWED',files=file_count,**checked,**reports,reset_endpoints=len(reset_rows),reopened_readonly=is_reopen,cdc_warnings=1466,missing_input_delays=1602,missing_output_delays=1602,scope='OOC internal CDC; parent clocks and interfaces remain unqualified',native_hbm_pcie_ip=False,board_qualified=False)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',type=Path)
    print(json.dumps(review(parser.parse_args().phase),indent=2))
