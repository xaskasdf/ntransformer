"""Review atomic phase certificates and measured two-clock simulator events."""
import re


def review(text):
    def need(ok,message):
        if not ok:raise ValueError(message)
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',text,re.M),'simulator failure')
    names=['normal_delayed_ack','wrong_identity_fields','preasserted_matching_ack',
        'peer_stopped_before_request','peer_stopped_during_action','management_stopped_during_return',
        'canceled_request_drains','replacement_while_busy','multiple_replacements_coalesce',
        'generation_identity','same_identity_not_rearmed','sequencer_axil_ledger_across_clocks']
    cases=re.findall(r'^MAILBOX_CASE_PASS case=(\d+) name=(\S+) ps=(\d+)$',text,re.M)
    need([(int(n),name) for n,name,ps in cases]==list(enumerate(names)),'12 ordered cases')
    need(text.splitlines().count('MAILBOX_SUITE_PASS cases=12 deliveries=13 answers=10 obsolete=3 mgmt_period_ps=10000 peer_period_ps=7000 native_endpoint=0')==1,'complete scoped suite')
    records={n:[] for n in ['INTENT','DELIVER','ACCEPT','ACK','EVENT']}
    for line in text.splitlines():
        for name,rows in records.items():
            if not line.startswith('MAILBOX_'+name+' '):continue
            r=dict(re.findall(r'(\w+)=(\S+)',line))
            rows.append({k:v if k=='name' else int(v,16 if k in {'tag','generation'} else 10) for k,v in r.items()})
    intents,deliveries,accepts,acks,events=(records[n] for n in records)
    need([len(x) for x in [intents,deliveries,accepts,acks]]==[15,13,13,10],'conserved transfer counts')
    key=lambda r:(r['generation'],r['tag'],r['kind'])
    need(len({key(r) for r in deliveries})==13,'unique delivered identities')
    expected=[r for r in intents if r['case']!=10 and not (r['case']==8 and r['tag']==0x10000000b)]
    need([key(r) for r in deliveries]==[key(r) for r in expected],'only delivered or latest queued request')
    need([key(r) for r in accepts]==[key(r) for r in deliveries],'one peer certificate per request')
    obsolete=lambda r:r['case']==6 or (r['case']==7 and r['tag']==0x100000008) or (r['case']==8 and r['tag']==0x10000000a)
    need([key(r) for r in acks]==[key(r) for r in accepts if not obsolete(r)],'obsolete completions never relabeled')
    for rows in [deliveries,accepts,acks]:need([r['ps'] for r in rows]==sorted(r['ps'] for r in rows),'clock event order')
    for r in deliveries+accepts:need(r['ps']%3500==1301,'actual peer edge including stopped clock phase')
    for r in acks:need(r['ps']%5000==1,'actual management edge including stopped clock phase')
    latency=[]
    for r in acks:
        sent=next(x for x in accepts if key(x)==key(r));delivered=next(x for x in deliveries if key(x)==key(r))
        need(r['ps']>sent['ps']>delivered['ps'],'proof before management ACK')
        latency.append(dict(case=r['case'],peer_certificate_to_management_ack_ps=r['ps']-sent['ps']))
    need(len(events)==10,'directed boundary events')
    by_name={r['name']:r for r in events};need(len(by_name)==10,'unique boundary names')
    for n in ['WRONG_FIELDS','PREASSERTED_ACK','PEER_STOP_BEFORE','PEER_STOP_DURING','MGMT_STOPPED',
              'CANCELED','REPLACED','COALESCED','OLD_GENERATION','REPLAY_SUPPRESSED']:
        need(by_name[n]['ack']==0,'no false ACK '+n)
    for n,count in [('CANCELED',1),('REPLACED',2),('COALESCED',3)]:need(by_name[n]['obsolete']==count,'obsolete inventory '+n)
    need(by_name['REPLAY_SUPPRESSED']['deliveries']==12 and by_name['REPLAY_SUPPRESSED']['answers']==9 and by_name['REPLAY_SUPPRESSED']['busy']==0,'last-key replay suppressed')
    integration=[x for x in text.splitlines() if x.startswith('MAILBOX_INTEGRATION ')]
    need(integration==[
        'MAILBOX_INTEGRATION event=AW_ONLY state=2 local_reset=0 endpoint_reset=0 aw=1 w=0',
        'MAILBOX_INTEGRATION event=B_BLOCKED state=2 local_reset=0 endpoint_reset=0 aw=1 w=1',
        'MAILBOX_INTEGRATION event=LOCAL_HELD state=3 local_reset=1 endpoint_reset=0 aw=0 w=0'],'ledger-to-sequencer through real CDC')
    need('$finish called at time' in text,'simulator completion')
    running=[r['peer_certificate_to_management_ack_ps'] for r in latency if r['case']!=5]
    return dict(status='PASS_TAGGED_MAILBOX_12_CASES_AXIL_SEQUENCE_CDC',cases=names,
        deliveries=13,certificates=13,management_answers=10,obsolete_discarded=3,
        certificate_return_latency_ps=latency,running_clock_return_min_ps=min(running),running_clock_return_max_ps=max(running),
        scope='Actual XPM_CDC_HANDSHAKE, AXI-Lite ledger and sequencer fixture at 10ns/7ns clocks. External isolation/owner action are controlled. No native endpoint, physical CDC/timing or complete DMA drain approval.',
        native_endpoint_qualified=False,physical_design_qualified=False)
