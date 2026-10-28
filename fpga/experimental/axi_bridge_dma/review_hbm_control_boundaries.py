"""Review isolated control-boundary evidence; never infer board timing closure."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
from prepare_hbm_control_boundaries import generated


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(ok,why):
    if not ok:raise ValueError(why)
def table(p):return list(csv.DictReader(p.read_text().splitlines(),delimiter='\t'))


def control(out,port,cell,candidate):
    rows=table(out/(port+'.tsv'));props={(r['object'],r['property']):r['value'] for r in rows}
    require(len(rows)==len(props),'Duplicate control properties')
    if cell is None:
        driver=props[port,'DRIVER'].removesuffix('/O')
        require(props[driver,'REF_NAME'].startswith('LUT') and 'rst_n' in props[port,'STARTPOINTS'].split(','),'Reference VALID gate')
        return dict(driver=props[port,'DRIVER'],registered=False)
    require(props[port,'DRIVER']==cell+'/Q' and props[port,'STARTPOINTS']==cell+'/C','Direct registered output')
    require(props[cell,'REF_NAME']=='FDCE' and props[cell,'INIT']=="1'b0",'Control primitive or INIT')
    require(props[cell,'ASYNC_REG'] not in ('1','true','TRUE') and props[cell,'DONT_TOUCH'] not in ('1','true','TRUE'),'Ordinary register attributes')
    require(props[cell+'/C','CLOCK']=='hbm_clk' and props[cell+'/CE','DRIVER_PRIMITIVE']=='VCC','Control clock or CE')
    require(props[cell+'/CLR','STARTPOINTS']=='rst_n','Reset source')
    driver=props[cell+'/CLR','DRIVER'];polarity=props[cell+'/CLR','IS_INVERTED']
    if driver=='rst_n':require(polarity=='1','Direct reset polarity')
    else:
        inv=driver.removesuffix('/O')
        require(polarity=='0' and props[inv,'REF_NAME']=='LUT1' and props[inv,'INIT']=="2'h1" and props[inv+'/I0','STARTPOINTS']=='rst_n','Explicit reset polarity')
    starts=props[cell+'/D','STARTPOINTS'].split(',')
    require(starts and '' not in starts and len(starts)==len(set(starts)) and 'rst_n' not in starts,'Predictor reset fan-in')
    if port=='h_rready':
        lengths=sorted(x for x in starts if x.startswith('h_arlen_reg'))
        require(props[cell+'/D','ARLEN_STARTPOINTS']==','.join(lengths),'Credit length inventory')
        require(not lengths if candidate else bool(lengths),'Credit adder dependency')
    return dict(driver=props[port,'DRIVER'],registered=True,d_startpoints=starts)


def topology(root,candidate):
    out=root/('candidate' if candidate else 'reference')
    result={p:control(out,p,c,candidate) for p,c in [('h_rready','rready_reg_reg'),('p_rvalid','validated_cut/valid_reg_reg' if candidate else None)]}
    xdc=(out/'constraints.xdc').read_text()
    require('2.222' in xdc and not re.search(r'^\s*set_(false_path|clock_groups|multicycle_path|max_delay|min_delay)',xdc,re.M),'Clock or exceptions')
    payload=table(out/'enables.tsv')
    expected={f'{q}/slot{s}_reg[{b}]' for q in ['response_cut','validated_cut'] for s in range(2) for b in range(265)}
    require(len(payload)==len(expected) and {r['cell'] for r in payload}==expected,'Payload inventory')
    allowed={'rst_n','response_cut/count_reg[0]/C','response_cut/count_reg[1]/C','response_cut/write_slot_reg/C'}
    for r in payload:
        require(r['primitive']=='FDRE' and r['clock']=='hbm_clk','Payload primitive or clock')
        if r['cell'].startswith('response_cut/'):
            starts=set(r['startpoints'].split(','))
            require(starts and '' not in starts and starts<=allowed,'Capture free-slot dependency')
    owners=table(out/'ownership.tsv')
    expected={(f'{q}/{c}',pin) for q in ['response_cut','validated_cut'] for c in ['count_reg[0]','count_reg[1]','write_slot_reg'] for pin in ['CE','D','R']}
    require(len(owners)==18 and {(r['cell'],r['pin']) for r in owners}==expected,'Ownership inventory')
    for q in ['response_cut','validated_cut']:
        for c in ['count_reg[0]','count_reg[1]','write_slot_reg']:
            starts={v for r in owners if r['cell']==q+'/'+c for v in r['startpoints'].split(',')}
            require('h_rvalid' in starts if q=='response_cut' else bool(starts&{'response_cut/count_reg[0]/C','response_cut/count_reg[1]/C'}),'Ownership lost upstream VALID')
    cells=table(out/'cells.tsv');require(len(cells)==len({r['cell'] for r in cells}),'Primitive inventory')
    result.update(payload_registers=len(payload),ownership_inputs=len(owners),primitives=dict(Counter(r['primitive'] for r in cells)),checkpoint_sha256=sha(out/'unit.dcp'))
    return result


def review(root,donor):
    result=json.loads((root/'RESULT.json').read_text())
    require(result['status']=='HBM_CONTROL_BOUNDARIES_EXPORTED' and result['error'] is None and result['sources_unchanged'],'Native completion')
    files=json.loads((root/'FILES.json').read_text())
    require({p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}==set(files)|{'FILES.json'},'Evidence inventory')
    for name,digest in files.items():require(sha(root/name)==digest,'Evidence digest: '+name)
    source=root/'source';binding=json.loads((source/'PREPARED.json').read_text())
    require(binding['status']=='HBM_CONTROL_BOUNDARIES_PREPARED' and sha(donor)==binding['donor_adapter_sha256'],'Donor binding')
    require(set(result['sources'])==set(binding['files'])|{'PREPARED.json'},'Source inventory')
    for record in (result['sources'],binding['files']):
        for name,digest in record.items():require(sha(source/name)==digest,'Source digest: '+name)
    for name,text in generated(donor).items():require((source/name).read_text()==text,'Untested RTL composition: '+name)
    negatives=json.loads((root/'NEGATIVE_CONTROLS.json').read_text())
    names=['credit_stale','stale_space','late_request_fault','validated_stale','validated_no_admission','validated_sync_reset','validated_overwrite']
    require(negatives==result['negative_controls'] and [n['name'] for n in negatives]==names,'Negative coverage')
    order=['compile','tracker-elaborate','tracker-run','validated-elaborate','validated-run','credit-elaborate','credit-run']
    for name in names:order += [name+suffix for suffix in ['-compile','-elaborate','-run']]
    order+=['synthesis']
    require([s['name'] for s in result['steps']]==order,'Stage order')
    for s in result['steps']:
        require(not s['timed_out'] and s['returncode']==0,'Native process error or timeout')
        if s['name'] not in {n+'-run' for n in names}:
            require(not re.search(r'(?im)^\s*(error|fatal):',(root/(s['name']+'.log')).read_text()),'Positive diagnostics')
    for n in negatives:
        log=(root/(n['name']+'-run.log')).read_text()
        require('Fatal:' in log and n['reason'] in log and 'ALL PASS' not in log,'Negative invariant')
        require(sha(root/('mutation-'+n['name'])/n['file'])==n['sha256'],'Mutation digest')
    cases={}
    for label,marker in [('tracker','REGISTERED_READY_TRACKER'),('validated','VALIDATED_BOUNDARY'),('credit','CREDIT_PREDICATE')]:
        log=(root/(label+'-run.log')).read_text();require(marker+' ALL PASS' in log,'Missing bank completion')
        cases[label]=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)} for line in log.splitlines() if line.startswith(marker+' PASS ')]
    require(cases['credit']==[dict(checks=8224,excluded=32)],'Credit exhaustive coverage')
    require(len(cases['validated'])==2 and {r['width'] for r in cases['validated']}=={265,519},'Queue widths')
    for r in cases['validated']:
        require(r['bins']==12 and r['async_resets']==1 and r['pushes']==r['pops']+r['discarded'] and r['checks']>200000 and r['stalls']>100,'Queue reset/conservation coverage')
    require(len(cases['tracker'])==4 and {r['pending'] for r in cases['tracker']}=={1,2,4,8},'Tracker depths')
    for r in cases['tracker']:
        require(r['quarantines']==4 and r['full_stalls']==24 and r['invalid_requests']==3 and r['aborted']>0 and r['ar_r_pairs']==int(r['pending']>1) and r['delivered']+r['aborted']==r['accepted_r'] and r['concurrent_bad_requests']==3*int(r['pending']>1),'Tracker faults/credits/conservation')
    require((root/'output/STATUS.txt').read_text().strip()=='HBM_CONTROL_BOUNDARIES_TOPOLOGY_EXPORTED','Native topology completion')
    return dict(status='HBM_CONTROL_BOUNDARIES_REVIEWED',seconds=result['seconds'],cases=cases,
                checks=sum(r['checks'] for rows in cases.values() for r in rows),negative_controls=negatives,
                topology={k:topology(root/'output',k=='candidate') for k in ['reference','candidate']},
                full_board_integrated=False,physical_timing_qualified=False,native_pcie_hbm_traffic=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('donor',type=Path)
    a=p.parse_args();print(json.dumps(review(a.root,a.donor),indent=2))
