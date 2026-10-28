"""Review closed native pad/clock evidence and derive an experimental reservation."""
import argparse
from collections import Counter, defaultdict
import copy
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from review_hbm_bar_control import closed, rows
from review_hbm_loader_readiness import expand_xdc

DONOR_SHA = '73570c8e393c01ead212b5a11c283d346ea54213749b1941265bcdb82b7a551d'
PAIRS = {'fabric_ref_clk_in':('BK31','BL31'), 'hbm_ref0_clk_in':('BK33','BL33'),
         'hbm_ref1_clk_in':('BH32','BJ32'), 'mgmt_clk_in':('BJ33','BJ34')}
OOC = ['configuration_rst_n','fabric_clock_locked','fabric_clock_reset',
       'hbm_ref0_clk_in','hbm_ref1_clk_in','mgmt_clk_in']


def require(ok, why):
    if not ok: raise ValueError(why)


def properties(table):
    result = defaultdict(dict)
    for row in table:
        require(row['property'] not in result[row['object']], 'Duplicate property')
        result[row['object']][row['property']] = row['value']
    return dict(result)


def clock_candidates(pins, gc, sites):
    require(len(pins) == 2892 and len(gc) == 96 and len({r['ball'] for r in gc}) == 96, 'Package/GCIO coverage')
    actual = {r['ball']:r for r in gc}
    require(set(actual) == {n for n,p in pins.items() if p['IS_GLOBAL_CLK']=='1' and p['IS_BONDED']=='1'}, 'GCIO inventory differs')
    for ball,r in actual.items():
        p = pins[ball]; bank = int(p['BANK'])
        require(r['function'] == p['PIN_FUNC'] and r['bank'] == p['BANK'], 'GCIO function/bank')
        require(64 <= bank <= 75 and r['slr'] == f'SLR{(bank-64)//4}' and r['clock_region'] == f'X4Y{bank-64}', 'GCIO SLR/region mapping')
        require(p['DIFF_PAIR_PIN'] in actual and pins[p['DIFF_PAIR_PIN']]['DIFF_PAIR_PIN'] == ball, 'GCIO pair reciprocity')
    hbm = [r for r in sites if r['type']=='HBM_REF_CLK']
    require(len(hbm)==2 and {r['site'] for r in hbm}=={'HBM_REF_CLK_X0Y0','HBM_REF_CLK_X0Y1'} and
            all(r['slr']=='SLR0' and r['clock_region']=='X4Y0' for r in hbm), 'HBM reference site locations')
    selected = []
    for port,(positive,negative) in PAIRS.items():
        p,n = pins[positive],pins[negative]
        require(p['DIFF_PAIR_PIN']==negative and n['DIFF_PAIR_PIN']==positive and p['IS_MASTER']=='1' and n['IS_MASTER']=='0', 'Selected differential pair')
        require(all(pins[b]['IS_USED']=='0' and actual[b]['slr']=='SLR0' and pins[b]['BANK']=='64' for b in (positive,negative)), 'Selected clock pin unavailable')
        selected.append(dict(port=port,positive=positive,negative=negative,bank=64,slr='SLR0',
            region=actual[positive]['clock_region'],frequency_mhz=100,
            state='EXPERIMENTAL_RESERVATION_NOT_BOARD_RELEASE',
            current_pad_present=port=='fabric_ref_clk_in',
            current_pad_mode='single-ended' if port=='fabric_ref_clk_in' else 'OOC boundary without input pad',
            final_clock_source=None,final_io_standard=None))
    return selected


def classify_ports(ports, pins, roots):
    expected = {f'pci_exp_{d}{p}[{i}]' for d in ('rx','tx') for p in ('p','n') for i in range(16)}
    expected |= set(OOC)|{'fabric_ref_clk_in','sys_clk_p','sys_clk_n','sys_rst_n'}
    require({r['port'] for r in ports}==expected and len(expected)==74, 'Native port coverage')
    serial = [r for r in ports if r['port'].startswith('pci_exp_')]
    require(len(serial)==64, 'Serial connectivity coverage')
    lanes = []
    for r in serial:
        direction, polarity, lane = re.fullmatch(r'pci_exp_(rx|tx)(p|n)\[(\d+)\]',r['port']).groups()
        lane=int(lane); channel=15-lane; bank=124+channel//4
        require(r['ref']=='GTYE4_CHANNEL' and r['loc']==f'GTYE4_CHANNEL_X0Y{channel}', 'Native lane mapping')
        require(r['leaf_pin'].endswith('/GTY'+direction.upper()+polarity.upper()), 'GT signal polarity')
        require(r['direction']==('IN' if direction=='rx' else 'OUT'), 'GT direction')
        function=f'MGTY{direction.upper()}{polarity.upper()}{channel%4}_{bank}'
        matches=[name for name,p in pins.items() if p['PIN_FUNC']==function and p['IS_BONDED']=='1']
        require(len(matches)==1, 'Unique bonded GT terminal')
        ball=matches[0]
        lanes.append(dict(port=r['port'],lane=lane,channel=r['loc'],bank=bank,ball=ball,
                          pair=pins[ball]['DIFF_PAIR_PIN'],function=function,connector_pin=None))
    require(len({r['ball'] for r in lanes})==64, 'Serial ball alias')
    by_port={r['port']:r for r in lanes}
    for r in lanes:
        other=r['port'].replace('p[','n[') if 'p[' in r['port'] else r['port'].replace('n[','p[')
        require(r['pair']==by_port[other]['ball'], 'Serial differential pair')
    expected_roots={'sys_clk_p':'IBUFDS_GTE4','fabric_ref_clk_in':'INBUF',
                    'mgmt_clk_in':'BUFGCE','hbm_ref0_clk_in':'BUFGCE','hbm_ref1_clk_in':'BUFGCE'}
    require(len(roots)==5 and {r['port'] for r in roots}==set(expected_roots), 'Clock root coverage')
    require(all(r['ref']==expected_roots[r['port']] and float(r['period_ns'])==10. for r in roots), 'Clock root/period')
    physical={'sys_clk_p':'IBUFDS_GTE4','sys_clk_n':'IBUFDS_GTE4','sys_rst_n':'INBUF','fabric_ref_clk_in':'INBUF'}
    for port,ref in physical.items():
        found=[r for r in ports if r['port']==port]
        require(len(found)==1 and found[0]['ref']==ref and found[0]['package_pin']=='', 'Existing unassigned pad')
    for r in ports:
        if r['port'] in OOC:
            require(r['ref'] not in ('INBUF','IBUF','IBUFDS','IBUFDS_GTE4','OBUF') and not r['package_pin'], 'OOC boundary acquired a pad')
    return sorted(lanes,key=lambda r:r['port'])


def review(root):
    result=closed(root)
    require(result['stage']=='inspect' and result['checkpoint_sha256']==DONOR_SHA, 'Wrong donor')
    out=root/'output'
    require((out/'STATUS.txt').read_text().strip()=='HBM_BOARD_BOUNDARY_EXPORTED_NO_ASSIGNMENTS', 'Wrong inventory stage')
    before=expand_xdc((out/'before.xdc').read_text());after=expand_xdc((out/'after.xdc').read_text())
    require(before==after, 'Inventory changed constraints')
    require(not any(re.match(r'set_property (PACKAGE_PIN|IOSTANDARD) ',s) for s in before), 'Donor contains explicit board pin assignments')
    pins=properties(rows(root,'package.tsv'));gc=rows(root,'gc_sites.tsv');sites=rows(root,'resource_sites.tsv')
    selected=clock_candidates(pins,gc,sites)
    ports=rows(root,'ports.tsv');roots=rows(root,'clock_roots.tsv');lanes=classify_ports(ports,pins,roots)
    cdc=(out/'full_cdc.rpt').read_text()
    counts=[(a,b,int(c)) for a,b,c in re.findall(r'^(CDC-\d+)\s+(Info|Warning|Critical)\s+(\d+)\s+',cdc,re.M)]
    require(counts==[('CDC-3','Info',200),('CDC-6','Warning',9),('CDC-9','Info',36),('CDC-15','Warning',2425)], 'Expanded CDC changed')
    require('No waivers found.' in (out/'waivers.rpt').read_text(), 'Waivers present')
    drc=(out/'drc.rpt').read_text()
    findings=re.findall(r'^\| ([A-Z]+-\d+)\s*\| (Critical Warning|Error)\s*\| [^|]+\| (\d+)\s*\|$',drc,re.M)
    require(findings==[(r,'Critical Warning','1') for r in ('AVAL-326','NSTD-1','UCIO-1')], 'Physical findings changed')
    require('Problem ports: fabric_ref_clk_in, sys_rst_n.' in drc and 'Problem ports: fabric_ref_clk_in, sys_clk_n, sys_clk_p, sys_rst_n.' in drc, 'DRC affected ports')
    readiness=rows(root,'readiness_paths.tsv')
    starts={'availability/fault_reg/C','availability/ready_mgmt_reg[0]/C','availability/ready_mgmt_reg[1]/C'}
    require(len(readiness)==6 and {(r['from'],r['type']) for r in readiness}=={(s,t) for s in starts for t in ('max','min')}, 'Readiness timing coverage')
    require(all(r['to']=='loader_readiness/ready_reg/D' and float(r['requirement_ns'])==(10. if r['type']=='max' else 0.) for r in readiness), 'Readiness source requirements')
    negative=[]
    def reject(name,fn):
        try:fn()
        except ValueError:negative.append(name)
        else:raise ValueError('Negative control escaped: '+name)
    bad=copy.deepcopy(gc);bad[0]['slr']='SLR9'
    reject('wrong_slr',lambda:clock_candidates(pins,bad,sites))
    badpins=copy.deepcopy(pins);badpins['BK31']['DIFF_PAIR_PIN']='BK33'
    reject('wrong_pair',lambda:clock_candidates(badpins,gc,sites))
    reject('missing_hbm_site',lambda:clock_candidates(pins,gc,[r for r in sites if r['site']!='HBM_REF_CLK_X0Y1']))
    reject('missing_clock_root',lambda:classify_ports(ports,pins,roots[:-1]))
    badports=copy.deepcopy(ports);next(r for r in badports if r['port']=='pci_exp_rxp[0]')['loc']='GTYE4_CHANNEL_X0Y0'
    reject('swapped_lane',lambda:classify_ports(badports,pins,roots))
    return dict(status='HBM_BOARD_BOUNDARY_REVIEWED',checkpoint_sha256=DONOR_SHA,
        package_balls=2892,gcio_pins=96,hbm_slr='SLR0',eligible_hbm_clock_pairs=16,
        native_port_bits=74,serial_port_bits=64,existing_unassigned_pad_bits=4,
        ooc_boundary_ports=OOC,clock_reservations=selected,serial_lanes=lanes,
        explicit_io_standards_present=False,property_defaults_are_not_board_constraints=True,
        full_cdc_findings=counts,readiness_estimated_hold_ns=min(float(r['slack_ns']) for r in readiness if r['type']=='min'),
        negative_controls=negative,physical_route_executed=False,board_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path)
    a=p.parse_args();value=json.dumps(review(a.root),indent=2)+'\n'
    if a.output:
        with a.output.open('x') as f:f.write(value)
    else:print(value,end='')
