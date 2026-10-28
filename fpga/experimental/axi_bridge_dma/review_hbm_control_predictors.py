"""Check native predictor evidence against frozen RTL and independent inventories."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode = True
from prepare_hbm_control_predictors import generated


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok, why):
    if not ok:raise ValueError(why)
def table(p):return list(csv.DictReader(p.read_text().splitlines(), delimiter='\t'))


def topology(root, kind, candidate):
    label = kind+('_candidate' if candidate else '_reference')
    out = root/label
    rows = table(out/'control.tsv')
    props = {(r['object'], r['property']):r['value'] for r in rows}
    require(len(rows)==len(props), 'Duplicate control properties')
    port = 'h_rready' if kind=='tracker' else 'm_valid'
    cell = 'rready_reg_reg' if kind=='tracker' else 'valid_reg_reg'
    d_starts = None
    if kind=='tracker' or candidate:
        require(props[port, 'DRIVER']==cell+'/Q' and props[port, 'STARTPOINTS']==cell+'/C', 'Direct output register')
        require(props[cell, 'REF_NAME']=='FDCE' and props[cell, 'INIT']=="1'b0", 'Control primitive/initial state')
        require(props[cell, 'ASYNC_REG'] not in ('1','true','TRUE') and props[cell, 'DONT_TOUCH'] not in ('1','true','TRUE'), 'Control register attributes')
        require(props[cell+'/C', 'CLOCK']=='hbm_clk' and props[cell+'/CE', 'DRIVER_PRIMITIVE']=='VCC', 'Control clock/enable')
        require(props[cell+'/CLR', 'STARTPOINTS']=='rst_n', 'Control reset source')
        driver = props[cell+'/CLR', 'DRIVER']; inversion = props[cell+'/CLR', 'IS_INVERTED']
        if driver=='rst_n':require(inversion=='1', 'Direct reset polarity')
        else:
            inv = driver.removesuffix('/O')
            require(driver.endswith('/O') and inversion=='0' and props[inv, 'REF_NAME']=='LUT1' and props[inv, 'INIT']=="2'h1", 'Explicit reset polarity')
        d_starts = props[cell+'/D', 'STARTPOINTS'].split(',')
        require(bool(d_starts) and ('rst_n' not in d_starts if candidate else 'rst_n' in d_starts), 'Predictor reset fan-in')
    else:
        driver = props[port, 'DRIVER'].removesuffix('/O')
        require(props[driver, 'REF_NAME'].startswith('LUT') and 'rst_n' in props[port, 'STARTPOINTS'].split(','), 'Return baseline combinational path')
    clock = (out/'constraints.xdc').read_text()
    require('2.222' in clock and not re.search(r'^\s*(set_false_path|set_clock_groups|set_multicycle_path)',clock,re.M), 'Clock/exception changed')
    queues = ('response_cut/', 'validated_cut/') if kind=='tracker' else ('',)
    width = 265 if kind=='tracker' else 519
    expected = {f'{q}slot{s}_reg[{b}]' for q in queues for s in range(2) for b in range(width)}
    payload = table(out/'enables.tsv')
    require(len(payload)==len(expected) and {r['cell'] for r in payload}==expected, 'Payload inventory')
    prefix = 'response_cut/' if kind=='tracker' else ''
    allowed = {'rst_n',prefix+'count_reg[0]/C',prefix+'count_reg[1]/C',prefix+'write_slot_reg/C'}
    for r in payload:
        require(r['primitive']=='FDRE' and r['clock']=='hbm_clk', 'Payload type/clock')
        if r['cell'].startswith(prefix):
            starts = set(r['startpoints'].split(','))
            require(starts and '' not in starts and starts<=allowed, 'Free-slot capture enable dependency')
    ownership = table(out/'ownership.tsv')
    owners = {prefix+'count_reg[0]',prefix+'count_reg[1]',prefix+'write_slot_reg'}
    expected = {(cell,pin) for cell in owners for pin in ('CE','D','R')}
    require(len(ownership)==9 and {(r['cell'],r['pin']) for r in ownership}==expected, 'Ownership input inventory')
    valid = 'h_rvalid' if kind=='tracker' else 's_valid'
    for owner in owners:
        require(any(valid in r['startpoints'].split(',') for r in ownership if r['cell']==owner), 'Ownership lost VALID')
    cells = table(out/'cells.tsv')
    require(len(cells)==len({r['cell'] for r in cells}), 'Duplicate primitive inventory')
    return dict(control_driver=props[port,'DRIVER'], d_startpoints=d_starts,
                payload_registers=len(payload), ownership_inputs=len(ownership),
                primitives=dict(Counter(r['primitive'] for r in cells)), checkpoint_sha256=sha(out/'unit.dcp'))


def review(root, donor):
    result = json.loads((root/'RESULT.json').read_text())
    require(result['status']=='HBM_CONTROL_PREDICTORS_EXPORTED' and result['sources_unchanged'] and result['error'] is None, 'Failed native result')
    files = json.loads((root/'FILES.json').read_text())
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    require(actual==set(files)|{'FILES.json'}, 'Archive inventory')
    for name, digest in files.items():require(sha(root/name)==digest, 'Evidence digest: '+name)
    source = root/'source'; binding = json.loads((source/'PREPARED.json').read_text())
    require(binding['status']=='HBM_CONTROL_PREDICTORS_PREPARED' and sha(donor)==binding['donor_adapter_sha256'], 'Donor binding')
    for name,digest in result['sources'].items():require(sha(source/name)==digest, 'Native source: '+name)
    require(set(result['sources'])==set(binding['files'])|{'PREPARED.json'}, 'Source inventory')
    for name,digest in binding['files'].items():require(sha(source/name)==digest, 'Prepared source: '+name)
    for name,text in generated(donor).items():require((source/name).read_text()==text, 'Collateral RTL change: '+name)
    negatives = json.loads((root/'NEGATIVE_CONTROLS.json').read_text())
    expected = {'stale_space','stale_credit','late_fault','request_fault','sync_reset',
                'return_stale','return_sync_reset','return_overwrite'}
    require(negatives==result['negative_controls'] and len(negatives)==8 and {n['name'] for n in negatives}==expected, 'Negative coverage')
    order = ['compile','queue-elaborate','queue-run','tracker-elaborate','tracker-run','return-elaborate','return-run']
    for n in negatives:order += [n['name']+suffix for suffix in ('-compile','-elaborate','-run')]
    order += ['synthesis']
    require([s['name'] for s in result['steps']]==order, 'Native stage order/inventory')
    for s in result['steps']:
        require(not s['timed_out'] and s['returncode']==0, 'Native timeout/process error')
        if s['name'] not in {n+'-run' for n in expected}:
            require(not re.search(r'(?im)^\s*(fatal|error):', (root/(s['name']+'.log')).read_text()), 'Positive diagnostics')
    for n in negatives:
        log = (root/(n['name']+'-run.log')).read_text()
        require('Fatal:' in log and n['reason'] in log and 'ALL PASS' not in log, 'Negative invariant')
        require(sha(root/('mutation-'+n['name'])/n['file'])==n['sha256'], 'Mutation hash')
    cases = {}
    for label,marker in [('queue','REGISTERED_READY_QUEUE'),('tracker','REGISTERED_READY_TRACKER'),('return','RETURN_VALID_QUEUE')]:
        log = (root/(label+'-run.log')).read_text()
        require(marker+' ALL PASS' in log, 'Simulation completion')
        cases[label] = [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
                        for line in log.splitlines() if line.startswith(marker+' PASS ')]
    for label,bins in [('queue',24),('return',12)]:
        rows=cases[label];require(len(rows)==2 and {r['width'] for r in rows}=={265,519}, 'Width coverage')
        for r in rows:
            require(r['bins']==bins and r['pushes']==r['pops']+r['discarded'] and r['checks']>200000, 'Queue coverage/conservation')
            if label=='return':require(r['async_resets']==1 and r['stalls']>100, 'Return async/full coverage')
    require(len(cases['tracker'])==4 and {r['pending'] for r in cases['tracker']}=={1,2,4,8}, 'Tracker depths')
    for r in cases['tracker']:
        require(r['quarantines']==4 and r['full_stalls']==24 and r['invalid_requests']==3 and r['aborted']>0 and
                r['ar_r_pairs']==int(r['pending']>1) and r['delivered']+r['aborted']==r['accepted_r'] and
                r['concurrent_bad_requests']==3*int(r['pending']>1), 'Tracker fault/reset/credit coverage')
    require((root/'output/STATUS.txt').read_text().strip()=='HBM_CONTROL_PREDICTORS_TOPOLOGY_EXPORTED', 'Missing topology completion')
    return dict(status='HBM_CONTROL_PREDICTORS_REVIEWED', seconds=result['seconds'], cases=cases,
                checks=sum(r['checks'] for rows in cases.values() for r in rows),
                negative_controls=negatives, topology={kind+('_candidate' if c else '_reference'):topology(root/'output',kind,c)
                    for kind in ('tracker','return') for c in (False,True)},
                full_board_integrated=False, physical_timing_qualified=False, native_pcie_hbm_traffic=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path);p.add_argument('donor',type=Path)
    a=p.parse_args();print(json.dumps(review(a.root,a.donor),indent=2))
