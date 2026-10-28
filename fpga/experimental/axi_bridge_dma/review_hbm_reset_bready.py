"""Independently review the complete-loader BREADY experiment."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import sys
sys.dont_write_bytecode=True
from prepare_hbm_reset_bready import candidate,once


def require(value,why):
    if not value:raise ValueError(why)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def table(p):return list(csv.DictReader(p.read_text().splitlines(),delimiter='\t'))


def topology(out):
    require((out/'STATUS.txt').read_text().strip()=='RESET_BREADY_TOPOLOGY_EXPORTED','Missing topology')
    utilization={}
    for variant,top in [('reference','ntrpu_hbm_weight_loader_reference'),('candidate','ntrpu_hbm_bready_loader')]:
        commands=[line.strip() for line in (out/(variant+'.xdc')).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        require(commands==['create_clock -period 4.000 -name clk [get_ports clk]',
                           'create_clock -period 2.222 -name mem_clk [get_ports mem_clk]',
                           'current_instance -quiet'],'Unexpected unit constraints')
        rows=[line.split('|')[1:-1] for line in (out/(variant+'_utilization.rpt')).read_text().splitlines() if line.startswith('|')]
        rows=[[item.strip() for item in row] for row in rows if row and row[0].strip()==top]
        require(len(rows)==1 and rows[0][1]=='(top)','Utilization inventory')
        keys=['luts','logic_luts','lutram','srl','ff','ramb36','ramb18','uram','dsp']
        require(len(rows[0][2:])==len(keys),'Utilization columns')
        utilization[variant]=dict(zip(keys,map(int,rows[0][2:])))
    a=table(out/'reference_ports.tsv');b=table(out/'candidate_ports.tsv')
    require(a==b and len(a)==len({x['port'] for x in a}) and len(a)>100,'Port inventory')
    result=[]
    for variant in ['reference','candidate']:
        rows=table(out/(variant+'.tsv'));props={(int(r['channel']),r['object'],r['property']):r['value'] for r in rows}
        require(len(rows)==len(props) and {x[0] for x in props}=={0,1},'Property inventory')
        for ch in range(2):
            p=lambda obj,key:props[ch,obj,key]
            port=f'h_bready[{ch}]';driver=p(port,'DRIVER');name=driver.rsplit('/',1)[0]
            if variant=='reference':
                require(p(name,'REF_NAME').startswith('LUT') and 'mem_cold_rst_n' in p(port,'STARTPOINTS').split(','),'Reference BREADY gate')
                continue
            require(driver==f'bready_reg_reg[{ch}]/Q' and p(name,'REF_NAME')=='FDCE' and p(name,'INIT')=="1'b0",'Registered BREADY driver')
            require(p(name,'ASYNC_REG') not in ('1','true','TRUE') and p(name,'DONT_TOUCH') not in ('1','true','TRUE'),'Ordinary register attributes')
            require(p(name+'/C','CLOCK')=='mem_clk' and p(name+'/CE','DRIVER_PRIMITIVE')=='VCC','Register clock or enable')
            require(p(port,'STARTPOINTS')==name+'/C','Output bypass')
            starts=p(name+'/D','STARTPOINTS').split(',')
            require(starts and '' not in starts and len(starts)==len(set(starts)) and not set(starts)&{'mem_cold_rst_n','cold_rst_n','bus_rst_n'},'Reset in BREADY D fan-in')
            require(p(name+'/CLR','STARTPOINTS')=='mem_cold_rst_n','Asynchronous clear source')
            encoding=p(name,'RESET_ENCODING');inv=p(name+'/CLR','IS_INVERTED')
            if encoding=='PRIMITIVE_INVERTED_CLR':require(inv=='1','Direct clear polarity')
            else:
                require(encoding=='EXPLICIT_INVERTER' and inv=='0','Explicit clear polarity')
                inverters=[obj for (c,obj,k),v in props.items() if c==ch and k=='REF_NAME' and v=='LUT1']
                require(len(inverters)==1 and p(inverters[0],'INIT')=="2'h1" and p(inverters[0]+'/I0','INPUT_PORT')=='mem_cold_rst_n','Reset inverter proof')
            result.append(dict(channel=ch,driver=driver,d_startpoints=starts,reset_encoding=encoding))
    return dict(channels=result,identical_ports=len(a),utilization=utilization,physical_timing_qualified=False)


def review(root,donor):
    r=json.loads((root/'RESULT.json').read_text());s=root/'source'
    require(r['status']=='RESET_BREADY_UNIT_EXPORTED' and r['error'] is None and r['sources_unchanged'],'Native unit failure')
    for n,h in r['sources'].items():require(sha(s/n)==h,'Native source changed')
    binding=json.loads((s/'PREPARED.json').read_text())
    require(binding['status']=='RESET_BREADY_UNIT_PREPARED','Preparation status')
    for n,h in binding['files'].items():require(sha(s/n)==h,'Prepared source changed')
    for n,h in binding['donor_files'].items():require(sha(donor/n)==h,'Donor changed')
    original=(donor/'ntrpu_hbm_weight_loader.sv').read_text()
    require((s/'candidate_loader.sv').read_text()==candidate(original),'Untested loader delta')
    require((s/'reference_loader.sv').read_text()==once(original,'module ntrpu_hbm_weight_loader (','module ntrpu_hbm_weight_loader_reference ('),'Reference loader differs')
    steps={x['name']:x for x in r['steps']};require(len(steps)==len(r['steps']) and not any(x['timed_out'] for x in steps.values()),'Step inventory or timeout')
    completed=json.loads((root/'COMPLETED.json').read_text())
    require(completed==[dict(bank='loader',seed=i) for i in range(4)]+[dict(bank='predicate',seed=0)],'Bank inventory')
    cases=[];checks=0
    for seed in range(4):
        log=(root/f'loader-seed-{seed}.log').read_text()
        matches=re.findall(r'WL PASS seed=(\d+) groups=(\d+) checks=(\d+) aw=(\d+) w=(\d+) b=(\d+)',log)
        eq=re.findall(r'BREADY_EQUIVALENCE PASS seed=(\d+) checks=(\d+) outputs=(\d+)',log)
        require(len(matches)==len(eq)==1,'Loader completion markers')
        a=list(map(int,matches[0]));e=list(map(int,eq[0]))
        require(a[0]==e[0]==seed and a[1]==10 and a[2]>100 and e[1]>1000 and e[2]==len(binding['compared_outputs']),'Loader coverage')
        require(steps[f'loader-seed-{seed}']['returncode']==0,'Loader simulation exit')
        checks+=a[2]+e[1];cases.append(dict(seed=seed,groups=a[1],checks=a[2],equivalence_checks=e[1],aw=a[3],w=a[4],b=a[5]))
    log=(root/'predicate-seed-0.log').read_text()
    require(re.findall(r'BREADY_PREDICATE PASS checks=(\d+)',log)==['4096'],'Predicate coverage')
    checks+=4096
    expected={'stale':'predictor truth table','early':'predictor truth table','ignore_response':'predictor truth table','wrong_lane':'predictor truth table','synchronous_clear':'asynchronous BREADY clear'}
    negatives=json.loads((root/'NEGATIVE_CONTROLS.json').read_text())
    require(len(negatives)==len(expected) and {x['name'] for x in negatives}==set(expected) and negatives==r['negative_controls'],'Negative controls inventory')
    for v in negatives:
        name=v['name'];log=(root/(name+'-run.log')).read_text();step=steps[name+'-run']
        require(v['reason']==expected[name] and v['reason'] in log and (step['returncode'] or re.search(r'(?im)^\s*(?:error|fatal):',log)),'Mutation failed for wrong reason')
        require(sha(root/('mutation-'+name)/v['file'])==v['sha256'],'Mutation source changed')
    bad={x+'-run' for x in expected}
    require(all(x['returncode']==0 for n,x in steps.items() if n not in bad),'Unexpected failed step')
    for n in steps.keys()-bad:
        require(not re.search(r'(?im)^\s*(?:error|fatal):',(root/(n+'.log')).read_text()),'Error in positive native step')
    return dict(status='RESET_BREADY_UNIT_REVIEWED',seconds=r['seconds'],checks=checks,cases=cases,
                compared_outputs=len(binding['compared_outputs']),negative_controls=len(negatives),topology=topology(root/'output'),
                checkpoint_sha256=sha(root/'output/candidate.dcp'),full_board_integrated=False,physical_timing_qualified=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('donor',type=Path)
    a=p.parse_args();print(json.dumps(review(a.root,a.donor),indent=2))
