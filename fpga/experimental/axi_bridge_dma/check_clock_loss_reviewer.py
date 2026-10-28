"""In-memory synthetic tests of the reviewer. NOT native or RTL validation."""
import copy
import json
from collections import defaultdict

from review_clock_loss_native import CLOCKS, NATIVE, P, review


def fixture(level):
    start=20001 if level==0 else 22001
    held=1300000;end=held+1;recovered=1940000;fault=665000
    times=sorted(set(range(0,recovered+1,2000))|set(range(0,recovered+1,5000))|
                 {start,held,end,recovered,fault,1355000})
    rows=[]
    for t in times:
        owned=int(t>=6000);failed=int(t>=fault)
        clk=level if start<=t<end else (t//2000)%2
        values={name:clk for name in CLOCKS};values.update({name:1 for name in NATIVE})
        fields=dict(phase=3,mgmt_clk=(t//5000)%2,configuration_rst_n=1,cold_rst_n=1,
            owned=owned,src_req_valid=owned,src_req_ready=0,src_valid=0,src_cpl_valid=0,ack_valid=0,
            result_valid=0,cmd_ready=1-owned,cfg_ready=0,cmd_valid=int(4000<=t<8000),
            cmd_tag=(9<<64)|3,cmd_generation=2,cmd_source=4096,cmd_bytes=64,
            src_req_tag=((9<<64)|3) if owned else 0,src_req_address=4096 if owned else 0,
            src_req_bytes=64 if owned else 0,guard_armed=1,guard_fault=failed,guard_code=2*failed,
            guard_alive=int(t<fault or t>=1355000),link_up=1-failed,stop=failed,
            fault=int(t>=1302000),mapper_fault=int(t>=1302000),pause_active=int(start<=t<end),
            pause_complete=int(t>=end),clock_case_done=int(t==recovered),stop_level=level)
        values.update({P+k:v for k,v in fields.items()});rows.append((t,values))
    log=f'''SYNTHETIC_REVIEWER_FIXTURE_NOT_NATIVE
BN DATA_PASS bytes=192 maps=2 clock_period_ps=4000
CL PAUSE level={level} t={start}
CL HELD level={level} t={held} code=2
CL RELEASE level={level} t={end}
CL RECOVERED level={level} t={recovered}
BRIDGE_CLOCK_LOSS_PASS level={level} timeout=64 hold_cycles=128 recovery_cycles=64
'''
    return log,rows


def vcd(rows):
    names=sorted(rows[0][1]);groups=defaultdict(list)
    for index,name in enumerate(names):groups[tuple(name.strip('/').split('/')[:-1])].append((name,'v'+str(index)))
    lines=['$timescale 1ps $end'];codes={}
    for scopes,entries in groups.items():
        lines.extend('$scope module '+s+' $end' for s in scopes)
        for name,code in entries:
            width=max(1,max(values[name].bit_length() for _,values in rows))
            lines.append(f'$var wire {width} {code} {name.rsplit("/",1)[1]} $end');codes[name]=code
        lines.extend('$upscope $end' for _ in scopes)
    lines.append('$enddefinitions $end');previous={}
    for t,values in rows:
        lines.append('#'+str(t))
        for name,value in values.items():
            if previous.get(name)!=value:lines.append('b'+format(value,'b')+' '+codes[name])
        previous=values
    return '\n'.join(lines)+'\n'


def check():
    positive=[]
    for level in (0,1):
        log,rows=fixture(level);positive.append(review(log,vcd(rows)))
    log,rows=fixture(0);variants=[]
    def changed(name,signals,when,reason):
        altered=copy.deepcopy(rows)
        for t,values in altered:
            if when(t):
                for signal,value in signals.items():values[signal]=value
        variants.append((name,log,altered,reason))
    changed('axi_toggles_while_paused',{k:1 for k in CLOCKS},lambda t:t==100000,'Actual public clocks held')
    changed('management_clock_stops',{P+'mgmt_clk':0},lambda t:100000<t<200000,'Continuous independent')
    changed('perst_asserts',{'/board/sys_rst_n':0},lambda t:t==100000,'No native reset/link loss')
    changed('cold_reset_asserts',{P+'cold_rst_n':0},lambda t:t==100000,'Retained cold reset')
    changed('source_identity_changes',{P+'src_req_tag':7},lambda t:t==100000,'Retained source identity')
    changed('source_accepts_during_pause',{P+'src_req_ready':1},lambda t:t==100000,'No new admission')
    changed('false_success',{P+'result_valid':1},lambda t:t==100000,'No new admission')
    changed('writer_fault_without_edge',{P+'fault':1},lambda t:t==100000,'AXI fault registers frozen')
    changed('automatic_rearm',{P+'guard_fault':0},lambda t:t==1500000,'Fault sticky')
    changed('wrong_fault_cause',{P+'guard_code':1},lambda t:t==700000,'clock-only cause')
    changed('clock_does_not_return',{k:0 for k in CLOCKS},lambda t:t>=1300001,'AXI resumes')
    changed('wrong_admitted_tag',{P+'cmd_tag':99},lambda t:t<6000,'Exactly one attributed third command')
    changed('guard_uses_different_clock',{P+'guard/axi_clk':1},lambda t:t==100000,'observe the same public AXI clock')
    changed('late_watchdog_detection',{P+'guard_fault':0,P+'guard_code':0,P+'guard_alive':1,P+'link_up':1,P+'stop':0},
            lambda t:665000<=t<855000,'detection within 76')
    variants.append(('wrong_event_time',log.replace('t=1300000 code=2','t=1300002 code=2'),rows,'event order'))
    rejected=[]
    for name,bad_log,bad_rows,reason in variants:
        try:review(bad_log,vcd(bad_rows))
        except ValueError as error:
            if reason not in str(error):raise AssertionError((name,reason,str(error)))
            rejected.append(dict(name=name,reason=str(error)))
        else:raise AssertionError('Mutation escaped: '+name)
    return dict(status='REVIEWER_SYNTHETIC_CONTROLS_PASS',synthetic_positive_cases=len(positive),
                rejected_controls=rejected,native_executed=False,rtl_compiled=False)


if __name__=='__main__':print(json.dumps(check(),indent=2))
