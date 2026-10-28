"""Classify every warning on this checkpoint, without waivers or fixed old net counts."""
import argparse,json,re
from pathlib import Path
from collections import Counter,defaultdict
from check_ntrpu_dual_hbm import sha
from check_ntrpu_hbm_netlist import rows,lut_used

def review(out):
    r=json.loads((out/'result.json').read_text());p=json.loads((out/'PRIOR_REVIEW.json').read_text())
    assert r['experiment']=='dual_hbm_drc' and r['status']=='EXPORTED_PENDING_REVIEW'
    assert r['returncode']==0 and not r['timed_out'] and r['sources']==r['sources_after']
    assert r['donor_unchanged'] and r['input_unchanged']
    assert p['status']=='DUAL_HBM_ROUTED_INTERNAL_TIMING_PASS' and p['result_sha256']==r['prior_result_sha256']
    assert p['checkpoint_sha256']==r['input_sha256']==sha(out/'input.dcp')
    assert all(sha(out/n)==h for n,h in json.loads((out/'FILES.json').read_text()).items())
    log=(out/'console.log').read_text();assert 'NTRPU_HBM_DRC_AUDIT_COMPLETE' in log and not re.search(r'^ERROR:',log,re.M)
    violations=defaultdict(dict)
    for v in rows(out,'violations.tsv'):violations[v['violation']][v['property']]=v['value']
    inventory=Counter(v.split('#')[0] for v in violations)
    assert inventory=={'PDCN-1569':12,'RTSTAT-10':1} and len(violations)==13
    assert all(v['SEVERITY']=='Warning' and v['IS_WAIVED']=='0' for v in violations.values())
    properties=defaultdict(dict)
    for v in rows(out,'cells.tsv'):properties[(v['violation'],v['cell'])][v['property']]=v['value']
    pins=rows(out,'pins.tsv');lut_evidence=[]
    for name,values in violations.items():
        if not name.startswith('PDCN-1569'):continue
        found=[]
        for (violation,cell),props in properties.items():
            if violation!=name:continue
            for pin in pins:
                if pin['violation']==name and pin['cell']==cell and '(pin '+pin['pin']+')' in values['DESCRIPTION']:
                    logical=pin['pin'].rsplit('/',1)[1]
                    assert lut_used(props['REF_NAME'],props['INIT'],logical) is False
                    found.append(dict(cell=cell,pin=logical,ref=props['REF_NAME'],init=props['INIT'],net=pin['net']))
        assert len(found)==1
        lut_evidence.append(dict(violation=name,**found[0]))
    nets=[v for v in rows(out,'nets.tsv') if v['violation'].startswith('RTSTAT-10')]
    loads=rows(out,'loads.tsv');reported=int(re.match(r'(\d+) net\(s\)',violations['RTSTAT-10#1']['DESCRIPTION'])[1])
    assert len(nets)==len({n['net'] for n in nets})==reported and reported>0
    classification=Counter();details=[];unresolved=[]
    for net in nets:
        inputs=[v for v in loads if v['violation']==net['violation'] and v['net']==net['net'] and v['direction']=='IN']
        used=[lut_used(v['ref_name'],v['init'],v['pin'].rsplit('/',1)[1]) for v in inputs]
        if not inputs and net['ports'] and net['route_status']=='HIERPORT':kind='external_ooc_port_without_internal_load'
        elif not inputs and not net['ports'] and net['route_status']=='NOLOADS':kind='no_logical_leaf_loads'
        elif inputs and all(v is False for v in used) and not net['ports']:kind='only_functionally_unused_lut_inputs'
        else:kind='requires_further_review';unresolved.append(dict(net=net,inputs=inputs,functional_use=used))
        classification[kind]+=1;details.append(dict(net=net['net'],kind=kind))
    assert re.search(r'# of nets with routing errors[.\s]*:\s*0',(out/'route_status.rpt').read_text(),re.I)
    return dict(status='DUAL_HBM_DRC_'+('OBJECTS_REVIEWED' if not unresolved else 'OPEN'),seconds=r['seconds'],
                result_sha256=sha(out/'result.json'),checkpoint_sha256=r['input_sha256'],warnings=13,rules=dict(inventory),
                unused_lut_pins=lut_evidence,rtstat_nets=reported,rtstat_classification=dict(classification),net_evidence=details,
                unresolved=unresolved,warnings_waived=False,drc_clean=False,external_io_qualified=False,board_qualified=False)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);a=ap.parse_args();print(json.dumps(review(a.output),indent=2))
