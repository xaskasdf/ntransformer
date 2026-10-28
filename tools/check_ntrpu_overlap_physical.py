"""Review compact physical stages and both routed functional comparisons."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_candidate_transport import require
from check_ntrpu_transport_study import physical, netlist, mutations
from check_ntrpu_transport_overlap import extra, negative


def routed_extra(text):
    result=extra(text);expected=defaultdict(list);actual=defaultdict(list)
    samples=Counter();matches=Counter();count=0;case=0
    for line in text.splitlines():
        f=line.split()
        if line.startswith('TE S '):case=int(f[3]);samples[case]+=1
        elif line.startswith('TE FETCH '):expected[case,'FETCH',int(f[3])].append((int(f[5],16),))
        elif line.startswith('TE W '):expected[case,'W',int(f[3])].append((int(f[7],16),(1<<64)-1,int(f[6])))
        elif line.startswith('TE CPL '):expected[case,'CPL',int(f[3])].append((int(f[4]),int(f[5])))
        elif line.startswith('TN MATCH '):
            count+=1;require(int(f[3])==count,'Missing extra routed sample');matches[int(f[2])]+=1
        elif line.startswith('TN FETCH '):actual[int(f[2]),'FETCH',int(f[3])].append((int(f[4],16),))
        elif line.startswith('TN W '):actual[int(f[2]),'W',int(f[3])].append((int(f[4],16),int(f[5],16),int(f[6])))
        elif line.startswith('TN CPL '):actual[int(f[2]),'CPL',int(f[3])].append((int(f[4]),int(f[5])))
    require(set(samples)=={1,2,3} and samples==matches and count>0,'Extra routed sample coverage')
    require(re.findall(r'TN ALL_PASS checks=(\d+)',text)==[str(count)],'Extra routed completion')
    require(expected==actual,'Extra routed bytes/metadata/status mismatch')
    result.update(status='ROUTED_EXTRA_EQUIVALENCE_PASS',routed_samples=count,output_ports=28)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--review',type=Path,required=True);a=ap.parse_args();out=a.output
    record=json.loads((out/'result.json').read_text());files=json.loads((out/'FILES.json').read_text())
    require(record['variant']=='compact' and record['status']=='EXPORTED_PENDING_REVIEW','Incomplete compact stage')
    require(record['sources']==record['sources_after'] and record['inputs']==record['inputs_after'] and
            record['rtl_evidence']==record['rtl_evidence_after'],'Inputs changed')
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    require(all(sha(out/n)==h for n,h in files.items()),'Evidence hashes')
    evidence=out/'rtl_evidence';old=json.loads((evidence/'result.json').read_text())
    old_files=json.loads((evidence/'FILES.json').read_text())
    require(old['variant']=='compact' and old['status']=='EXPORTED_PENDING_REVIEW' and old['sources']==old['sources_after'],'RTL baseline provenance')
    require(sha(evidence/'result.json')==old_files['result.json'],'Baseline record hash')
    require(all(sha(p)==old_files['suite/generated/'+p.name] for p in evidence.glob('*.sv')),'Baseline generated hashes')
    stage=record['stage'];console=(out/'console.log').read_text()
    require('NTRPU_TRANSPORT_STUDY_COMPLETE '+stage in console and not re.search(r'^ERROR:',console,re.M),'Tool completion')
    if stage in ('synth','netlist'):
        meta=json.loads((out/'suite/generated/COMPACT_PHYSICAL.json').read_text())
        require(meta['variant']=='compact' and meta['clock_period_ns']==5 and meta['interface_registers_added']==0,'Physical fixture scope')
        require(meta['baseline_result_sha256']==sha(evidence/'result.json'),'Baseline identity')
        require(all(sha(out/'suite/generated'/(n+'.sv'))==h==sha(evidence/(n+'.sv'))
                    for n,h in meta['validated_generated_sha256'].items()),'Physical/generated logic differs')
    if stage=='netlist':
        text=(out/'suite/simulation.log').read_text();other=(out/'suite/extra.log').read_text()
        require(not any(re.search(r'^\s*(Fatal:|ERROR:|FAIL )',s,re.M) for s in (text,other)),'Simulation error')
        result=netlist(text);result['extra']=routed_extra(other)
        result['trace_negative_controls']=mutations(text,netlist)
        result['extra_negative_controls']=negative(other,routed_extra,['TN W ','TN FETCH ','TN CPL ','TN MATCH '])
        comp=json.loads((out/'suite/generated/COMPARISON.json').read_text())
        require(comp['latency_edges']==0 and comp['startup_ns']==150 and comp['reversible_renaming'] and
                not comp['sdf'] and not comp['force'] and comp['extra_cases']==3,'Comparison scope')
        require(comp['netlist_sha256']==record['inputs']['physical_funcsim.v']==record['prior']['sha256'],'Netlist provenance')
    else:
        result=physical(out,stage);utilization=(out/'suite/utilization.rpt').read_text();resources={}
        for key in ('CLB LUTs','CLB Registers','Block RAM Tile','RAMB36/FIFO*','RAMB18'):
            found=re.search(r'^\|\s*'+re.escape(key)+r'\*?\s*\|\s*([0-9.]+)\s*\|',utilization,re.M)
            if found:resources[key]=float(found[1])
        require(all(k in resources for k in ('CLB LUTs','CLB Registers','Block RAM Tile')),'Missing resource counts')
        result['resources']=resources
    result.update(stage=stage,variant='compact',seconds=record['seconds'],checker_sha256=sha(Path(__file__)))
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cases','timing_checks','drc','negative_controls')}))


if __name__=='__main__':main()
