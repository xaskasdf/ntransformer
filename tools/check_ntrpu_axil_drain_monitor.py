"""Inspect AXI-Lite ledger evidence without claiming DMA or CDC qualification."""
import re


def review(text):
    def need(ok,message):
        if not ok:raise ValueError(message)
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',text,re.M),'simulator failure')
    expected=['fresh_request_and_identity','aw_before_w_and_blocked_b','w_before_aw',
        'read_and_blocked_r','held_valid_and_external_fence','multiple_and_simultaneous_transactions',
        'unexpected_b','b_before_both_halves','unexpected_r','aw_overflow','w_overflow','ar_overflow',
        'sequencer_waits_for_real_ledger']
    cases=re.findall(r'^DRAIN_CASE_PASS case=(\d+) name=(\S+) ps=(\d+)$',text,re.M)
    need([(int(n),name) for n,name,ps in cases]==list(enumerate(expected)),'13 ordered cases')
    need(text.splitlines().count('DRAIN_SUITE_PASS cases=13 native_endpoint=0 cdc=0 scope=axil_only')==1,'complete scoped suite')
    rows=[]
    for line in text.splitlines():
        if not line.startswith('DRAIN_EVENT '):continue
        r=dict(re.findall(r'(\w+)=(\S+)',line))
        rows.append({k:v if k=='name' else int(v,16 if k in {'tag','ack_tag'} else 10) for k,v in r.items()})
    need([r['ps'] for r in rows]==sorted(r['ps'] for r in rows),'event order')
    for r in rows:
        if r['name']=='ACK':
            need(r['ack']==r['req']==r['fenced']==1 and r['fault']==r['aw']==r['w']==r['ar']==0,'drained ACK requirements')
            need(r['tag']==r['ack_tag'],'ACK belongs to request')
        else:need(r['ack']==0,'incomplete transaction cannot ACK')
    def event(case,name):
        group=[r for r in rows if r['case']==case and r['name']==name]
        need(len(group)==1,'unique '+str(case)+' '+name);return group[0]
    for case,name,inventory in [(1,'AW_ONLY',(1,0,0)),(1,'B_BLOCKED',(1,1,0)),
            (2,'W_ONLY',(0,1,0)),(3,'AR_PENDING',(0,0,1)),(3,'R_BLOCKED',(0,0,1)),
            (4,'VALID_HELD',(0,0,0)),(5,'FULL_BALANCED',(3,3,3))]:
        r=event(case,name);need(tuple(r[k] for k in ['aw','w','ar'])==inventory and not r['fault'],'pending inventory '+name)
    for case in range(6,12):need(event(case,'FAULT')['fault']==1,'sticky fault '+str(case))
    high=[r['tag'] for r in rows if r['case']==0 and r['name']=='ACK']
    need(high==[0x100000001,0x200000001,0x200000001],'full width and reissued request identity')
    integration=[]
    for line in text.splitlines():
        if line.startswith('DRAIN_INTEGRATION '):integration.append(dict(re.findall(r'(\w+)=(\S+)',line)))
    need(integration==[
        dict(event='AW_ONLY',state='2',local_reset='0',endpoint_reset='0',aw='1',w='0'),
        dict(event='B_BLOCKED',state='2',local_reset='0',endpoint_reset='0',aw='1',w='1'),
        dict(event='LOCAL_HELD',state='3',local_reset='1',endpoint_reset='0',aw='0',w='0')],
        'sequencer integration release order')
    need('$finish called at time' in text,'simulator completion')
    return dict(status='PASS_AXIL_DRAIN_13_CASES_SAME_DOMAIN_SEQUENCE',cases=expected,event_count=len(rows),
        clock_domain_count=1,bench_counter_bits=2,synthesis_default_counter_bits=4,
        native_endpoint_qualified=False,cdc_qualified=False,physical_design_qualified=False,
        scope='AXI4-Lite ledger and same-domain sequencer fixture only. External fence and non-quiescence actions are controlled test inputs; no DMA/burst/CDC/internal-IP drain certificate.')
