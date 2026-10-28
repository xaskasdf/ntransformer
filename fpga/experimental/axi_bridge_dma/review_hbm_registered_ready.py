"""Independently check registered-ready unit evidence; no board timing claim."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
from prepare_hbm_registered_ready import module,transform


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)
def table(path):return list(csv.DictReader(path.read_text().splitlines(),delimiter='\t'))


def topology(out,variant):
    root=out/variant
    rows=table(root/'enables.tsv')
    expected={f'{q}/slot{s}_reg[{b}]' for q in ('response_cut','validated_cut') for s in range(2) for b in range(265)}
    require(len(rows)==1060 and {x['cell'] for x in rows}==expected,'Payload inventory')
    for row in rows:
        require(row['primitive']=='FDRE' and row['clock']=='hbm_clk','Payload primitive or clock')
        if row['cell'].startswith('response_cut/'):
            starts=set(row['startpoints'].split(','))
            allowed={'rst_n','response_cut/count_reg[0]/C','response_cut/count_reg[1]/C','response_cut/write_slot_reg/C'}
            require(starts and '' not in starts and starts<=allowed,'Payload enable external dependency')
    ownership=table(root/'ownership.tsv')
    owners={'response_cut/count_reg[0]','response_cut/count_reg[1]','response_cut/write_slot_reg'}
    expected_inputs={(cell,pin) for cell in owners for pin in ('CE','D','R')}
    require(len(ownership)==9 and {(x['cell'],x['pin']) for x in ownership}==expected_inputs,'Ownership input inventory')
    for cell in owners:
        inputs=set().union(*(set(x['startpoints'].split(',')) for x in ownership if x['cell']==cell))
        require('h_rvalid' in inputs,'Ownership lost RVALID')
    ready=table(root/'rready.tsv');require(len(ready)==1 and ready[0]['port']=='h_rready','READY port inventory')
    ready=ready[0]
    if variant=='candidate':
        require(ready['driver']=='rready_reg_reg/Q' and ready['primitive']=='FDCE' and ready['pin']=='Q',
                'RREADY not directly registered')
        require(ready['startpoints']=='rready_reg_reg/C','Combinational path to RREADY')
        rows=table(root/'reset.tsv');props={(x['object'],x['property']):x['value'] for x in rows}
        require(len(rows)==len(props),'Duplicate reset properties')
        name='rready_reg_reg'
        require(props[name,'REF_NAME']=='FDCE' and props[name,'INIT']=="1'b0",'READY reset primitive')
        require(props[name,'ASYNC_REG'] not in ('1','true','TRUE') and props[name,'DONT_TOUCH'] not in ('1','true','TRUE'),'READY is ordinary mutable register')
        require(props[name+'/C','CLOCK']=='hbm_clk' and props[name+'/CLR','STARTPOINTS']=='rst_n','READY clock or reset source')
        driver=props[name+'/CLR','DRIVER'];inverted=props[name+'/CLR','IS_INVERTED']
        if driver=='rst_n':require(inverted=='1','Wrong direct reset polarity')
        else:
            inv=driver.removesuffix('/O')
            require(inverted=='0' and props[inv,'REF_NAME']=='LUT1' and props[inv,'INIT']=="2'h1"
                    and props[inv+'/I0','STARTPOINTS']=='rst_n','Wrong explicit reset inverter')
    else:require('rst_n' in ready['startpoints'].split(',') and ready['primitive'].startswith('LUT'),'Baseline lacks combinational reset path')
    cells=table(root/'cells.tsv');require(len({x['cell'] for x in cells})==len(cells),'Duplicate primitive inventory')
    clocks=(root/'constraints.xdc').read_text()
    require('2.222' in clocks and not re.search(r'^\s*(set_false_path|set_clock_groups|set_multicycle_path)',clocks,re.M),'Unit clock/exception changed')
    return dict(ready=ready,response_payload_registers=530,validated_payload_registers=530,
                response_enables_from_rvalid=0,ownership_depends_on_rvalid=True,
                primitives=dict(Counter(x['primitive'] for x in cells)),checkpoint_sha256=sha(root/'tracker.dcp'))


def review(root,donor):
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='HBM_REGISTERED_READY_EXPORTED' and result['sources_unchanged'] and result['error'] is None,'Native result failed')
    source=root/'source';binding=json.loads((source/'PREPARED.json').read_text())
    require(sha(donor)==binding['donor_adapter_sha256'],'Frozen donor changed')
    for name,digest in result['sources'].items():require(sha(source/name)==digest,'Native source changed: '+name)
    for name,digest in binding['files'].items():require(sha(source/name)==digest,'Prepared source changed: '+name)
    tracker=module(donor.read_text(),'ntrpu_hbm_read_tracker')
    require((source/'reference_tracker.sv').read_text()==tracker,'Reference tracker changed')
    require((source/'candidate_tracker.sv').read_text()==transform(tracker),'Collateral tracker change')
    for name,rtl in [('reference_queue.sv','ntrpu_hbm_read_response_queue'),('reference_capture_queue.sv','ntrpu_hbm_response_capture_queue')]:
        require((source/name).read_text()==module(donor.read_text(),rtl),'Reference queue changed')
    expanded='concurrent_bad_requests=' in (source/'tb_hbm_registered_ready_tracker.sv').read_text()
    negative_runs={'stale_space-run','stale_credit-run','late_fault-run','sync_reset-run'}
    if expanded:negative_runs.add('request_fault-run')
    require(all(not step['timed_out'] and (step['returncode'] in (0,1) if step['name'] in negative_runs else step['returncode']==0)
                for step in result['steps']),'Native timeout or process failure')
    require(len(result['steps'])==(21 if expanded else 18),'Native stage inventory')
    for step in result['steps']:
        if step['name'].endswith('-run') and step['name'] not in ('queue-run','tracker-run'):continue
        require(not re.search(r'(?im)^\s*(fatal|error):',(root/(step['name']+'.log')).read_text()),'Positive native diagnostics')
    queue=(root/'queue-run.log').read_text();tracker_log=(root/'tracker-run.log').read_text()
    require('REGISTERED_READY_QUEUE ALL PASS' in queue and 'REGISTERED_READY_TRACKER ALL PASS' in tracker_log,'Missing simulation completion')
    qrows=[]
    for line in queue.splitlines():
        if line.startswith('REGISTERED_READY_QUEUE PASS '):qrows.append({k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)})
    require(len(qrows)==2 and {x['width'] for x in qrows}=={265,519},'Queue width coverage')
    for row in qrows:require(row['bins']==24 and row['pushes']==row['pops']+row['discarded'] and row['checks']>1000000,'Queue conservation/coverage')
    trows=[]
    for line in tracker_log.splitlines():
        if line.startswith('REGISTERED_READY_TRACKER PASS '):trows.append({k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)})
    require(len(trows)==4 and {x['pending'] for x in trows}=={1,2,4,8},'Credit depth coverage')
    for row in trows:
        require(row['quarantines']==4 and row['full_stalls']==24 and row['invalid_requests']==3 and row['aborted']>0
                and row['ar_r_pairs']==int(row['pending']>1) and row['delivered']+row['aborted']==row['accepted_r'],
                'Tracker fault/reset/capacity coverage')
        if expanded:require(row.get('concurrent_bad_requests')==3*int(row['pending']>1),'Concurrent request fault coverage')
    negatives=json.loads((root/'NEGATIVE_CONTROLS.json').read_text())
    require(negatives==result['negative_controls'] and {x['name']+'-run' for x in negatives}==negative_runs and len(negatives)==len(negative_runs),'Negative control coverage')
    for item in negatives:
        log=(root/(item['name']+'-run.log')).read_text()
        require('Fatal:' in log and item['reason'] in log and 'REGISTERED_READY_TRACKER ALL PASS' not in log,'Mutation not rejected as intended')
        require(sha(root/('mutation-'+item['name'])/'candidate_tracker.sv')==item['sha256'],'Mutation evidence changed')
    require((root/'output/STATUS.txt').read_text().strip()=='HBM_REGISTERED_READY_TOPOLOGY_EXPORTED','Missing synthesis proof')
    return dict(status='HBM_REGISTERED_READY_REVIEWED',seconds=result['seconds'],concurrent_request_faults_checked=expanded,queue_cases=qrows,tracker_cases=trows,
                checks=sum(x['checks'] for x in qrows+trows),negative_controls=negatives,
                reference=topology(root/'output','reference'),candidate=topology(root/'output','candidate'),
                full_board_integrated=False,physical_timing_qualified=False,native_pcie_hbm_traffic=False,board_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('donor',type=Path)
    a=p.parse_args();print(json.dumps(review(a.root,a.donor),indent=2))
