"""Cross-check both sides of AMD AXI CDC, then replay the cold MIG bank.

Control/barrier and readback remain test drivers in the MIG UI domain.
This gate does not qualify control CDC, native PCIe or warm resets.
"""
from collections import defaultdict
from check_ntrpu_pcie_ingress_mig import review as review_memory, need


def review(text, host_period_ps=4000, host_phase_ps=700):
    memory=review_memory(text,1)
    traces=defaultdict(list); config=None; final=None
    fields={'AW':{'addr','id','len','size','burst','lock','ps'},
            'W':{'data','strb','last','ps'}, 'B':{'id','resp','ps'}}
    for line in text.splitlines():
        if not line.startswith('NTRPU_CDC_LINE_'):continue
        label,*tokens=line.split();kind=label.removeprefix('NTRPU_CDC_LINE_')
        f=dict(t.split('=',1) for t in tokens)
        need(len(f)==len(tokens),'duplicate CDC field')
        if kind=='CONFIG':
            need(config is None and not traces and final is None,'CDC config context')
            need(set(f)=={'host_period_ps','host_phase_ps','ui_period_ps','ps'},'CDC config schema')
            config={k:int(v) for k,v in f.items()}
            need(config['host_period_ps']==host_period_ps and config['host_phase_ps']==host_phase_ps and config['ui_period_ps']==5000,'CDC clocks')
            need(config['ps']==memory['start_ps'],'CDC cold admission')
        elif kind=='PASS':
            need(final is None and config is not None and set(f)=={'aw','w','b','ps'},'CDC final context/schema')
            final={k:int(v) for k,v in f.items()}
            need(all(final[k]==194 for k in ['aw','w','b']) and final['ps']==memory['summary']['ps'],'CDC final counts/time')
        else:
            need(kind in {d+'_'+c for d in ['HOST','DEVICE'] for c in fields},'CDC marker')
            domain,channel=kind.split('_')
            need(config is not None and final is None and set(f)==fields[channel],'CDC event context/schema')
            record={k:int(v,16 if k in {'addr','id','data','strb'} else 10) for k,v in f.items()}
            ps=record['ps'];need(ps>config['ps'],'CDC traffic before start')
            if traces[kind]:need(ps>traces[kind][-1]['ps'],'CDC per-channel time order')
            if domain=='HOST':need((ps-host_phase_ps-host_period_ps//2)%host_period_ps==0,'host clock edge')
            traces[kind].append(record)
    need(config is not None and final is not None,'CDC final proof absent')
    # The extra device monitor must describe exactly the same handshakes as
    # the original memory checker; matching only the two extra logs is weaker.
    for channel in fields:
        originals=[]
        for line in text.splitlines():
            if line.startswith('NTRPU_MIG_LINE_'+channel+' '):
                f=dict(t.split('=',1) for t in line.split()[1:])
                originals.append({k:int(v,16 if k in {'addr','id','data','strb'} else 10) for k,v in f.items() if k!='cycle'})
        devices=traces['DEVICE_'+channel];hosts=traces['HOST_'+channel]
        need(len(originals)==len(devices)==len(hosts)==194,'CDC conserved channel count')
        for original,device,host in zip(originals,devices,hosts):
            need(all(device[k]==v for k,v in original.items()),'CDC versus memory monitor')
            need({k:v for k,v in device.items() if k!='ps'}=={k:v for k,v in host.items() if k!='ps'},'CDC payload/ID/sideband changed')
            if channel=='AW':need(device['burst']==1 and device['lock']==0,'CDC write profile')
            need((host['ps']>device['ps']) if channel=='B' else (device['ps']>host['ps']),'CDC causal direction')
            need(device['ps']<final['ps'] and host['ps']<final['ps'],'CDC retirement time')
    offsets={r['ps']%5000 for c in fields for r in traces['DEVICE_'+c]}
    need(len(offsets)==1,'device events do not share UI clock')
    latencies={}
    for channel in fields:
        samples=[abs(d['ps']-h['ps']) for d,h in zip(traces['DEVICE_'+channel],traces['HOST_'+channel])]
        latencies[channel]={'min_ps':min(samples),'max_ps':max(samples),'samples':len(samples)}
    for i in range(194):
        need(traces['DEVICE_B'][i]['ps']>max(traces['DEVICE_AW'][i]['ps'],traces['DEVICE_W'][i]['ps']),'device B before request')
        need(traces['HOST_B'][i]['ps']>max(traces['HOST_AW'][i]['ps'],traces['HOST_W'][i]['ps']),'host B before request')
        if i+1<194:
            need(min(traces['HOST_AW'][i+1]['ps'],traces['HOST_W'][i+1]['ps'])>traces['HOST_B'][i]['ps'],'fixture must serialize host responses before next request')
    return dict(status='PASS_ASYNC_INGRESS_COLD_MIG',qualified=True,clocks=config,
                cdc_channel_latencies=latencies,host_transactions=194,memory=memory,
                latency_scope='Observed behavioral simulation handshake latency only. AMD independent-clock FIFO models do not model synchronization delays; these numbers do not qualify physical CDC latency or metastability.',
                scope='AMD AXI4 asynchronous 64-bit/512-bit/4-bit-ID transport, one host transaction outstanding, cold coordinated resets; actual ingress/controller/receiver/capacity guard and generated generic MIG. Control, isolation and readback remain UI-clock BFMs; no physical CDC timing/metastability, native PCIe, host control CDC, exact DIMM, warm reset, exhaustive capacity or inference claim.')
