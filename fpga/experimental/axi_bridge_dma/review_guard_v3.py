"""Offline checks for V3 reset-source sharing, RTL traces and native CDC."""
import argparse
import json
import math
from pathlib import Path
import re
from review_guard import review as legacy_review, vcd_rows
from review_guard_v2_controls import review_controls
from review_guard_integration import manifest, require, rows, sha, PINNED
from review_guard_physical_candidate import timing, route_status


def cdc(text):
    summary=re.findall(r'^CDC-(\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',text,re.M)
    require(summary==[('3','Info','3'),('9','Info','6')],'V3 scoped CDC coverage/severity changed')
    detail=re.findall(r'^\s*\d+\s+CDC-(\d+)\s+Info\s+.+?\s+(\d+)\s+(None|False Path|Max Delay Datapath Only)\s+(\S+)\s+(\S+)\s*$',text,re.M)
    require(len(detail)==9 and all(d=='3' for _,d,_,_,_ in detail),'V3 synchronizer depth changed')
    require(sum(e=='False Path' for _,_,e,_,_ in detail)==6 and sum(e=='Max Delay Datapath Only' for _,_,e,_,_ in detail)==3,'V3 CDC exceptions changed')
    return {'CDC-3':3,'CDC-9':6,'critical':0,'warning':0}


def constraints(text):
    actual=[]
    for line in text.splitlines():
        if re.match(r'set_(?:false_path|max_delay|min_delay|clock_groups|multicycle_path)\b',line) and 'guarded/guard/' in line:
            pins=re.findall(r'NAME == "([^"]+)"',line)
            require(len(pins)==2,'V3 exception is not an exact pin pair')
            if line.startswith('set_max_delay'):
                bound=10.0 if pins[1].endswith('echo_m_reg[0]/D') else 4.0
                require('-datapath_only' in line and float(line.split()[-1])==bound,'First-stage delay bound changed')
            actual.append((line.split()[0],tuple(pins)))
    g='guarded/guard/'
    expected={('set_max_delay',(g+f,g+t)) for f,t in [('request_reg/C','echo_a_reg[0]/D'),('echo_a_reg[2]/C','echo_m_reg[0]/D'),('armed_reg/C','armed_a_reg[0]/D')]}
    expected|={('set_false_path',(g+f'availability[{n}].release_m_reg[2]/C',g+f'availability[{n}].release_a_reg[{i}]/CLR')) for n in range(3) for i in range(3)}
    expected|={('set_false_path',(g+'fault_reg/C',g+f'stop_a_reg[{i}]/PRE')) for i in range(3)}
    expected|={('set_false_path',(f'ntrpu_bridge_i/inst/pcie4c_ip_i/inst/{reg}/C',g+f'availability[{n}].release_m_reg[{i}]/CLR')) for n,reg in [(0,'user_lnk_up_reg'),(1,'user_reset_reg')] for i in range(3)}
    require(len(actual)==21 and set(actual)==expected,'Unexpected V3 timing cuts')
    require(not re.search(r'^set_clock_groups\b.*(?:mgmt_clk|axi_aclk_out)',text,re.M),'Whole clock cut')
    return dict(exact_false_paths=18,first_stage_max_delays=3)


def alias_sources(alias,reference):
    require(len(reference)==2 and {a['port'] for a in reference}=={'axi_aresetn','axi_ctl_aresetn'},'Missing loaded reference alias proof')
    require(reference[0]['driver'] and len({a['driver'] for a in reference})==1 and len({a['startpoint'] for a in reference})==1,'Loaded reference outputs are not aliases')
    require(len(alias)==2 and {a['port'] for a in alias}=={'axi_aresetn','axi_ctl_aresetn'},'Native alias coverage')
    primary=next(a for a in alias if a['port']=='axi_aresetn');ctl=next(a for a in alias if a['port']=='axi_ctl_aresetn')
    require(primary['driver']==reference[0]['driver'] and primary['startpoint']==reference[0]['startpoint']=='ntrpu_bridge_i/inst/pcie4c_ip_i/inst/user_reset_reg/C','Unexpected native reset source')
    require((ctl['driver']==primary['driver'] and ctl['startpoint']==primary['startpoint']) or (ctl['driver']=='' and ctl['startpoint']=='REMOVED_UNUSED_ALIAS'),'Native reset outputs are distinct')


def structure(out,placed):
    alias=rows((out/'native_reset_alias.tsv').read_text())
    reference=rows((out.parent/'reference_alias.tsv').read_text())
    alias_sources(alias,reference)
    sync=rows((out/'synchronizers.tsv').read_text())
    chains=['cold_m','cold_a','echo_a','echo_m','armed_a','stop_a']+[f'availability[{n}].release_{d}' for n in range(3) for d in ('m','a')]
    require(len(sync)==36 and {r['cell'] for r in sync}=={f'guarded/guard/{c}_reg[{i}]' for c in chains for i in range(3)},'Wrong physical synchronizer topology')
    if placed:require(all(r['loc'].startswith('SLICE_') and r['bel'] for r in sync),'Unplaced synchronizer')
    stages=rows((out/'interstage.tsv').read_text())
    require(len(stages)==48 and {(r['from'],r['to'],r['type']) for r in stages}=={(f'guarded/guard/{c}_reg[{i}]/C',f'guarded/guard/{c}_reg[{i+1}]/D',t) for c in chains for i in range(2) for t in ('max','min')},'Missing interstage timing')
    for r in stages:
        clock='mgmt_clk' if re.search(r'(?:cold_m|echo_m|release_m)_reg',r['from']) else 'axi_aclk_out'
        require(r['clock']==clock and math.isfinite(float(r['slack'])),'Wrong interstage clock/slack')
        if placed:require(float(r['slack'])>=0,'Unmet interstage timing')
    checked=dict(re.findall(r'\d+\. checking (\w+) \((\d+)\)',(out/'check_timing.rpt').read_text()))
    require(all(checked.get(k)=='0' for k in ('no_clock','multiple_clock','unconstrained_internal_endpoints','loops')),'Internal timing coverage incomplete')
    require('No waivers found.' in (out/'waivers.rpt').read_text(),'Unexpected waiver')
    xdc=(out/'effective.xdc').read_text()
    require('HD.CLK_SRC' not in xdc and 'set_input_delay' not in xdc and 'set_output_delay' not in xdc,'Unexpected external clock/timing assumption')
    return dict(alias=alias,loaded_reference_alias=reference,async_registers=36,interstage_paths=48,
                interstage_min={k:min(float(r['slack']) for r in stages if r['type']==k) for k in ('max','min')},
                cdc=cdc((out/'cdc.rpt').read_text()),constraints=constraints(xdc),timing=timing((out/'timing.rpt').read_text()),
                inputs_without_delay=int(checked['no_input_delay']),outputs_without_delay=int(checked['no_output_delay']))


def topology(log,vcd,bad,clocks):
    if bad:
        require('V3_SHARED_RESET_CONTRACT' in log and 'V3_TOPOLOGY_PASS' not in log,'Miswired shared reset was not rejected')
        return dict(expected_contract_rejection=True)
    require('Fatal:' not in log and 'Error:' not in log,'Topology simulator failure')
    require(f'V3_TOPOLOGY_PASS axi_half={clocks[0]} mgmt_half={clocks[1]} phase={clocks[2]}' in log,'Missing topology verdict')
    staggered=False;separate=False;final={}
    for _,before,a in vcd_rows(vcd):
        final=a
        if a.get('cold_rst_n')!=1:continue
        if a.get('ctl_n')==0 and a.get('sl')==1 and a.get('dl')==0:staggered=True
        if a.get('df')==1 and a.get('sf')==0 and a.get('sl')==1:separate=True
        if a.get('sf')==1:require(a.get('sl')==0 and a.get('ss')==1 and a.get('sc')==1,'Shared fault loses ownership reset contract')
    require(staggered and separate,'Distinct reset independence unobserved')
    require(all(final.get(k)==1 for k in ('df','sf','ds','ss','dc','sc')) and all(final.get(k)==0 for k in ('dl','sl')),'Topology final fault state')
    return dict(staggered_release_observed=True,independent_control_reset_observed=True)


def rtl(root,topology_root=None):
    count=manifest(root);result=json.loads((root/'RESULT.json').read_text())
    require(result['sources_unchanged'],'RTL sources changed')
    if topology_root is None:require(result['status']=='GUARD_V3_RTL_TRACE_PASS','RTL phase failed')
    else:
        manifest(topology_root);fixed=json.loads((topology_root/'RESULT.json').read_text())
        require(fixed['status']=='GUARD_V3_TOPOLOGY_TRACE_PASS' and fixed['sources_unchanged'] and len(fixed['reviews'])==4,'Portable topology phase failed')
        require(result['status']=='GUARD_V3_RTL_TRACE_PASS' or result.get('error')=='Failed step: topology-0-0','Unexpected initial RTL failure')
        strip=lambda t:re.sub(r'// synthesis translate_off.*?// synthesis translate_on','',t,flags=re.S)
        require(strip((root/'source/ntrpu_axi_bridge_guard_v3.sv').read_text())==strip((topology_root/'source/ntrpu_axi_bridge_guard_v3.sv').read_text()),'Synthesizable body differs across RTL phases')
    profiles=[(2,5,1),(7,2,2)];names=[]
    for shared in (0,1):
        for pi,clocks in enumerate(profiles):
            for case in range(17):
                name=f'legacy_v3_{shared}-{pi}-{case:02}';names.append(name)
                legacy_review((root/(name+'.log')).read_text(),(root/name/'guard.vcd').read_text(),case,clocks)
        for case in range(16):
            name=f'controls_v3_{shared}-{case:02}';names.append(name)
            review_controls((root/(name+'.log')).read_text(),(root/name/'guard.vcd').read_text(),case)
    require(set(names)<={r['name'] for r in result['reviews']},'Incomplete functional cases')
    top_root=topology_root or root
    for pi,clocks in enumerate(profiles):
        for bad in (0,1):
            name=f'topology-{pi}-{bad}';names.append(name)
            topology((top_root/(name+'.log')).read_text(),(top_root/name/'guard.vcd').read_text(),bad,clocks)
    if topology_root is None:require(len(result['reviews'])==104 and {r['name'] for r in result['reviews']}==set(names),'Incomplete RTL cases')
    return dict(files=count,cases=104,positive_cases=102,expected_wiring_rejections=2,profiles=2,portable_topology_substitution=topology_root is not None)


def native(root):
    count=manifest(root);r=json.loads((root/'RESULT.json').read_text())
    require(r['status']=='V3_SHARED_NATIVE_SYNTHESIS_EXPORTED' and r['returncode']==0 and r['inputs_unchanged'] and not r['timed_out'],'Native synthesis failed')
    for n in ('ntrpu_axi_bridge_endpoint_client.sv','ntrpu_axi_bridge_writer.sv','ntrpu_axi_bridge_mapper.sv','ntrpu_xdma_local_master.sv'):require(sha(root/'source'/n)==PINNED[n],'Unrelated RTL changed')
    return dict(files=count,evidence=structure(root/'output/optimized',False))


def route(root,checkpoint,reference):
    count=manifest(root);r=json.loads((root/'RESULT.json').read_text())
    require(r['status']=='V3_SHARED_NATIVE_ROUTE_EXPORTED' and r['returncode']==0 and r['inputs_unchanged'] and not r['timed_out'],'Native route failed')
    inp=json.loads((root/'INPUT.json').read_text())
    require(sha(checkpoint)==inp['sha256'] and sha(reference)==inp['reference_sha256'],'Route checkpoint/reference mismatch')
    evidence=structure(root/'output/v3_audit',True)
    require(cdc((root/'output/routed/cdc.rpt').read_text())==evidence['cdc'],'CDC snapshots disagree')
    buffers=rows((root/'output/guard_buffers.tsv').read_text())
    for b in buffers:
        require(b['ref']=='BUFGCE' and b['driver']=='guarded/guard/cold_a_reg[2]/Q' and b['ce_ref']=='VCC' and b['clock_loads']=='0' and int(b['reset_loads'])>0,'Unexpected guard buffer role')
    return dict(files=count,evidence=evidence,route=route_status((root/'output/route_status.rpt').read_text()),guard_buffers=buffers,board_timing_closed=False,joint_hbm_implemented=False,native_postroute_simulation=False)


def recovered(root,origin,reference):
    count=manifest(root);manifest(origin)
    r=json.loads((root/'RESULT.json').read_text());inp=json.loads((root/'INPUT.json').read_text())
    require(r['status']=='V3_SHARED_RECOVERY_EXPORTED' and r['returncode']==0 and r['inputs_unchanged'] and not r['timed_out'],'Recovery failed')
    require(inp['sha256']==sha(origin/'output/synth.dcp') and inp['reference_sha256']==sha(reference),'Recovery input/reference changed')
    original=json.loads((origin/'RESULT.json').read_text())
    require(original['inputs_unchanged'],'Original synthesis sources changed')
    for n in ('ntrpu_axi_bridge_endpoint_client.sv','ntrpu_axi_bridge_writer.sv','ntrpu_axi_bridge_mapper.sv','ntrpu_xdma_local_master.sv'):require(sha(origin/'source'/n)==PINNED[n],'Unrelated RTL changed')
    seed=json.loads((origin/'SEED_FILES.json').read_text())
    require(seed['ntrpu_bridge_ex.srcs/sources_1/ip/ntrpu_bridge/ntrpu_bridge.xci']=='bf4a478249f14eb4a3db8d3317d4040fa374ffc09e4936f149a14e41dffe0a7a','Different native IP configuration')
    return dict(files=count,evidence=structure(root/'output/optimized',False),synth_checkpoint_reused=True)


def probe(root,checkpoint):
    count=manifest(root);r=json.loads((root/'RESULT.json').read_text())
    require(r['status']=='V3_FINDINGS_PROBE_EXPORTED' and r['returncode']==0 and r['inputs_unchanged'] and not r['timed_out'],'Read-only probe failed')
    require(json.loads((root/'INPUT.json').read_text())['sha256']==sha(checkpoint),'Probe checkpoint changed')
    out=root/'output'
    clean=lambda text:[line for line in text.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    require(clean((out/'before.xdc').read_text())==clean((out/'after.xdc').read_text()),'Probe modified constraints')
    constraints((out/'after.xdc').read_text())
    require('No waivers found.' in (out/'waivers.rpt').read_text(),'Probe used waivers')
    warnings=re.findall(r'There is a large setup violation of (-?\d+\.\d+) ns between (\S+) \(clocked by [^\n]+?\) and (\S+) \(clocked by axi_aclk_out\)',(out/'methodology.rpt').read_text())
    require(len(warnings)==11 and len({to for _,_,to in warnings})==11,'Methodology endpoint coverage changed')
    require({fr for _,fr,_ in warnings}=={'ntrpu_bridge_i/inst/pcie4c_ip_i/inst/ntrpu_bridge_pcie4c_ip_pcie_4_0_pipe_inst/pcie_4_c_e4_inst/CORECLK'},'Unexpected methodology source')
    endpoints={to for _,_,to in warnings}
    require(all(to.startswith('ntrpu_bridge_i/') for to in endpoints),'Finding outside native bridge')
    paths=rows((out/'methodology_paths.tsv').read_text())
    require(len(paths)==22 and {(p['endpoint'],p['mode']) for p in paths}=={(to,mode) for to in endpoints for mode in ('timed','ignored')},'Missing methodology path queries')
    for p in paths:
        require(int(p['count']) in (0,1),'Unexpected timing query count')
        if p['mode']=='timed':
            require(p['count']=='1' and math.isfinite(float(p['slack'])),'Missing timed methodology path')
            require(p['end_clock']=='axi_aclk_out','Wrong methodology destination clock')
            endpoint=p['endpoint'];bit=re.search(r'\[(\d+)\]/CE$',endpoint)
            require(bit is not None,'Unexpected methodology endpoint')
            kind='rdata' if 's_axi_ctl_rdata_reg[' in endpoint else 'fsm'
            report=(out/f'path_{kind}_{bit[1]}_timed.rpt').read_text()
            slack=re.search(r'Slack \((?:MET|VIOLATED)\)\s*:\s*(-?\d+\.\d+)ns',report)
            require(endpoint in report and slack is not None and float(slack[1])==float(p['slack']),'Methodology report/query mismatch')
    return dict(files=count,methodology_warnings=11,methodology_slack_min_ns=min(float(s) for s,_,_ in warnings),
                queried_timing_slack_min_ns=min(float(p['slack']) for p in paths if p['mode']=='timed'),
                queried_paths=paths,cdc=cdc((out/'cdc.rpt').read_text()),timing=timing((out/'timing.rpt').read_text()),
                queried_paths_meet_constraints=all(float(p['slack'])>=0 for p in paths if p['mode']=='timed'),
                constraints_unchanged=True,warnings_preserved=True,board_timing_closed=False)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('phase',choices=('rtl','native','route','recovered'));ap.add_argument('evidence',type=Path);ap.add_argument('--checkpoint',type=Path);ap.add_argument('--origin',type=Path);ap.add_argument('--reference',type=Path);ap.add_argument('--topology',type=Path)
    a=ap.parse_args();result=recovered(a.evidence,a.origin,a.reference) if a.phase=='recovered' else route(a.evidence,a.checkpoint,a.reference) if a.phase=='route' else rtl(a.evidence,a.topology) if a.phase=='rtl' else native(a.evidence)
    print(json.dumps(result,indent=2))
