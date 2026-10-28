"""Derive the PCIe candidate pin table from a preserved native-checkpoint inventory."""
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import re
from review_guard_integration import manifest,require


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def properties(path):
    data=defaultdict(dict)
    with path.open() as f:
        for r in csv.DictReader(f,delimiter='\t'):
            require(r['property'] not in data[r['object']],'Duplicate property')
            data[r['object']][r['property']]=r['value']
    return data


def active_xdc(text):
    return [s.strip() for s in text.splitlines() if s.strip() and not s.lstrip().startswith('#')]


def decode_lanes(rows,pins):
    require(len(rows)==64 and len({r['port'] for r in rows})==64,'Expected 64 serial ports')
    result=[]
    for r in rows:
        match=re.fullmatch(r'pci_exp_(rx|tx)(p|n)\[(\d+)\]',r['port'])
        require(match is not None,'Unexpected serial port')
        direction,polarity,lane=match.groups();lane=int(lane)
        require(0<=lane<16 and r['loc']==f'GTYE4_CHANNEL_X0Y{15-lane}','Lane-to-channel mapping changed')
        require(r['direction']==('IN' if direction=='rx' else 'OUT'),'Serial direction')
        require(r['ref']=='GTYE4_CHANNEL' and r['primitive_pin']=='GTY'+direction.upper()+polarity.upper(),'Serial primitive')
        matches=[p for p in r['package_candidates'].split(',') if re.fullmatch('MGTY'+direction.upper()+polarity.upper()+r'\d+_\d+',pins[p]['PIN_FUNC'])]
        require(len(matches)==1,'Ambiguous package polarity')
        ball=matches[0];pin=pins[ball]
        require(pin['IS_BONDED']=='1' and pin['BANK']==str(127-lane//4),'Unbonded or different GT bank')
        result.append(dict(port=r['port'],lane=lane,direction=r['direction'],polarity=polarity.upper(),
                           primitive_pin=r['primitive_pin'],site=r['loc'],bank=int(pin['BANK']),ball=ball,
                           pin_function=pin['PIN_FUNC'],pair_ball=pin['DIFF_PAIR_PIN'],
                           ecad_net=f'NTRPU_PCIE_{direction.upper()}{lane}_{polarity.upper()}',connector_pin=None,
                           status='CANDIDATE_NOT_JOINT_HBM_QUALIFIED'))
    require(len({r['ball'] for r in result})==64,'Duplicate ball')
    by_port={r['port']:r for r in result}
    for r in result:
        other=r['port'].replace('p[','n[') if r['polarity']=='P' else r['port'].replace('n[','p[')
        require(r['pair_ball']==by_port[other]['ball'],'Differential pair mismatch')
    return sorted(result,key=lambda r:(r['lane'],r['direction'],r['polarity']))


def cdc_inventory(text):
    summary=re.findall(r'^CDC-(\d+)\s+(Info|Warning|Critical)\s+(\d+)\s',text,re.M)
    require(summary==[('3','Info','164'),('9','Info','30'),('15','Warning','288')],'Global CDC changed; review required')
    warnings=[]
    for line in text.splitlines():
        if not re.match(r'^\s*\d+\s+CDC-15\s+Warning',line):continue
        source,dest=line.split()[-2:]
        a=re.fullmatch(r'(ntrpu_bridge_i/.*gen_cpll_cal_gtye4\.gen_cpll_cal_inst\[(\d+)\].*/U_TXOUTCLK_FREQ_COUNTER)/testclk_cnt_reg\[(\d+)\]/C',source)
        b=re.fullmatch(r'(ntrpu_bridge_i/.*/U_TXOUTCLK_FREQ_COUNTER)/freq_cnt_o_reg\[(\d+)\]/D',dest)
        require(a and b and a[1]==b[1] and a[3]==b[2] and 'False Path' in line,'Unclassified CDC warning')
        warnings.append(dict(rule='CDC-15',severity='Warning',instance=int(a[2]),bit=int(a[3]),source=source,destination=dest))
    require(len(warnings)==288 and {(r['instance'],r['bit']) for r in warnings}=={(i,b) for i in range(16) for b in range(18)},'Incomplete CPLL CDC inventory')
    return warnings


def build(root,out):
    count=manifest(root);r=root/'output';result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='NATIVE_PHYSICAL_PREFLIGHT_EXPORTED' and result['returncode']==0 and result['inputs_unchanged'],'Incomplete preflight')
    require(active_xdc((r/'before.xdc').read_text())==active_xdc((r/'after.xdc').read_text()),'Preflight changed constraints')
    require('No waivers found.' in (r/'waivers.rpt').read_text(),'Waivers introduced')
    pins=properties(r/'package.tsv');require(len(pins)==2892,'Package coverage')
    lanes=decode_lanes(list(csv.DictReader((r/'serial_lanes.tsv').open(),delimiter='\t')),pins)
    cells=properties(r/'physical_cells.tsv')
    require([(p['REF_NAME'],p.get('LOC')) for p in cells.values() if p.get('REF_NAME','').startswith('PCIE')]==[('PCIE4CE4','PCIE4CE4_X0Y0')],'Wrong PCIe hard block')
    warnings=cdc_inventory((r/'full_cdc.rpt').read_text())
    fixed=[]
    for port,ball,function in [('sys_clk_p','AR40','MGTREFCLK0P_125'),('sys_clk_n','AR41','MGTREFCLK0N_125'),('sys_rst_n','BF41','IO_T3U_N12_PERSTN0_65')]:
        p=pins[ball];require(p['PIN_FUNC']==function and p['IS_BONDED']=='1','Candidate special pin mismatch')
        fixed.append(dict(port=port,ball=ball,bank=int(p['BANK']),pin_function=function,pair_ball=p['DIFF_PAIR_PIN'],
                          iostandard='LVCMOS18' if port=='sys_rst_n' else None,status='EXPLORATORY_PCIE_ONLY'))
    require(pins['AR40']['DIFF_PAIR_PIN']=='AR41' and pins['AR41']['DIFF_PAIR_PIN']=='AR40','Reference pair')
    contract=dict(schema=1,status='EXPLORATORY_VU47P_PCIE_CONTRACT',part='xcvu47p-fsvh2892-2-e',
                  architecture_decision='Prioritize VU47P HBM node; keep DDR node separate',
                  board_approved=False,joint_hbm_implemented=False,connector_selected=False,
                  preflight_files=count,preflight_manifest_sha256=sha(root/'FILES.json'),
                  checkpoint=json.loads((root/'INPUT.json').read_text()),pcie=dict(generation=3,lanes=16,block='PCIE4CE4_X0Y0',gt_banks=[124,125,126,127],serial_ports=lanes),
                  candidate_external_pins=fixed,
                  clocks={'pcie_refclk':{'period_ns':10,'source':'external PCIe reference, electrical budget pending'},
                          'axi_aclk_out':{'period_ns':4,'source':'native AMD BUFG_GT, not replaced'},
                          'mgmt_clk':{'period_ns':10,'source':'independent external OOC contract through BUFGCE','package_pin':None,'oscillator':None}},
                  configuration_reset={'port':'configuration_rst_n','package_pin':None,'independent_of_perst':True},
                  perst_electrical={'fpga_candidate_voltage':1.8,'host_interface_translation':'unselected; no direct-connection approval','shared_bank_consumers':'must be reviewed with final HBM/auxiliary design'},
                  fabric_boundary={'clock':'axi_aclk_out','inputs_without_delay':1589,'outputs_without_delay':511,'timing_budget':None,'zero_delay_constraints_added':False},
                  global_cdc={'CDC-3_info':164,'CDC-9_info':30,'CDC-15_warning':288,'critical':0,'warning_class':'16 CPLL calibration counters, 18 bits each; retained without waiver'},
                  pending=['HBM IP and clocks in same top','management clock source and pin','configuration reset source','electrical PERST contract','fabric producers and consumers','connector/lane order at PCB','power SI/PI and thermal budgets'])
    out.mkdir(parents=True,exist_ok=False)
    (out/'contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    with (out/'pcie_lane_map.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(lanes[0]));w.writeheader();w.writerows(lanes)
    with (out/'cdc_warning_inventory.tsv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(warnings[0]),delimiter='\t');w.writeheader();w.writerows(warnings)
    return contract


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('preflight',type=Path);ap.add_argument('new_output',type=Path)
    a=ap.parse_args();c=build(a.preflight,a.new_output)
    print(json.dumps(dict(status=c['status'],serial_ports=len(c['pcie']['serial_ports']),cdc=c['global_cdc'])))
