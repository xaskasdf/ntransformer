"""Negative checks on copies of real short-pulse traces, without rerunning RTL."""
import argparse
import json
from pathlib import Path
import re
from review_guard import vcd_rows
from review_guard_v2_controls import review_controls


def encode(states):
    names=sorted(set().union(*(set(values) for _,values in states)))
    codes={name:'v'+str(i) for i,name in enumerate(names)}
    lines=['$timescale 1ps $end']
    for name in names:
        width=max([1]+[v[name].bit_length() for _,v in states if isinstance(v.get(name),int)])
        lines.append(f'$var wire {width} {codes[name]} {name} $end')
    lines.append('$enddefinitions $end');previous={}
    for now,values in states:
        lines.append('#'+str(now))
        for name,value in values.items():
            if name not in previous or previous[name]!=value:
                lines.append('b'+('x' if value is None else format(value,'b'))+' '+codes[name])
        previous=values
    return '\n'.join(lines)+'\n'


def check(root):
    evidence={};variants=[]
    for case in (0,1,12):
        log=(root/f'controls-{case:02}.log').read_text()
        trace=(root/f'controls-{case:02}/guard.vcd').read_text()
        review_controls(log,trace,case)
        states=[(now,dict(after)) for now,_,after in vcd_rows(trace)]
        # Verify lossless semantic re-encoding before mutating any observations.
        review_controls(log,encode(states),case)
        start=int(re.search(r'^V2 CONTROL_DROP .* t=(\d+)$',log,re.M)[1])
        evidence[case]=(log,states,start)

    def change(name,case,field,value,when):
        log,states,start=evidence[case]
        altered=[(t,dict(v)) for t,v in states];count=0
        for i,(t,values) in enumerate(altered):
            if when(t,values,start,i,len(altered)):
                replacement=value(values) if callable(value) else value
                if values.get(field)!=replacement:values[field]=replacement;count+=1
        if not count:raise ValueError('No-op mutation: '+name)
        variants.append((name,case,log,encode(altered)))

    after=lambda t,v,s,i,n:s+100<=t<s+1000
    last=lambda t,v,s,i,n:i==n-1
    change('admission_reopens',0,'client_link_up',1,after)
    change('operational_cold_reset',0,'client_cold_rst_n',0,after)
    change('stop_missing_on_fault',0,'client_stop',0,lambda t,v,s,i,n:v.get('fault')==1)
    change('wrong_input_falls',0,'native_status',13,lambda t,v,s,i,n:t==s)
    change('pulse_return_delayed',0,'native_status',14,lambda t,v,s,i,n:t==s+100)
    change('management_clock_gap',0,'mgmt_clk',0,lambda t,v,s,i,n:s+1000<t<s+28000)
    change('fault_not_sticky',0,'fault',0,last)
    change('wrong_fault_code',0,'fault_code',2,last)
    change('clock_not_alive_at_end',0,'clock_alive',0,last)
    change('axi_changes_while_stopped',1,'axi_clk',1,after)
    change('axi_never_resumes',1,'axi_clk',0,lambda t,v,s,i,n:t>=s and v.get('clock_enable')==1)
    change('overlap_pulse_missing',12,'native_status',lambda v:v['native_status']|2,
           lambda t,v,s,i,n:t>=s and isinstance(v.get('native_status'),int))
    log,states,start=evidence[0]
    variants.append(('wrong_case_verdict',0,log.replace('V2_CONTROL_PASS case=0','V2_CONTROL_PASS case=9'),encode(states)))
    rejected={}
    for name,case,log,trace in variants:
        try:review_controls(log,trace,case)
        except ValueError as error:rejected[name]=str(error)
        else:raise ValueError('Corrupted trace accepted: '+name)
    return dict(status='GUARD_V2_PULSE_REVIEWER_CONTROLS_PASS',actual_cases_reencoded=3,rejected_controls=rejected,
                mutations_are_synthetic=True,additional_rtl_runs=0)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('run',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();result=check(a.run)
    with a.output.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(result))
