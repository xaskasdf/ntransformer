"""Review real dual AMD CDC control/data, cold MIG and passive DDR witnesses.

Host transactions, external barrier and direct MIG readback are test drivers;
this is not yet a native PCIe or physical synchronization qualification.
"""
from collections import defaultdict
from check_ntrpu_pcie_ingress_cdc import review as data_review
from check_ntrpu_ingress_periodic import review as periodic_review


def need(ok, message):
    if not ok:
        raise ValueError(message)


def fixture():
    writes=[(0x20,0x11223344,15,0),(0x20,0xaabbccdd,5,0)]
    reads=[(0,0x4952544e,0),(1,0,3),(0x10000000,0,3),(0x90,0,3),(0x20,0x11bb33dd,0)]
    locations=[0,0x100000040,0x800000000,0x1000000000,0x1800000000,0x1fffffffc0]
    for job in range(1,14):
        tag=0x1000+job;window=0x1000fffc0 if job%2 else 0x100000000
        dest=locations[job-1] if job<=6 else {7:0x2000000000,8:0x1000000000000000,9:0xffffffffffffffc0,10:locations[1],11:locations[5],12:0,13:0}[job]
        def desc(epoch,win,addr):
            for i,value in enumerate((epoch,tag,win,addr)):
                writes.extend([(0x20+i*8,value & 0xffffffff,15,0),(0x24+i*8,value>>32,15,0)])
        desc(job,window,dest)
        writes.extend([(0x10,1,15,0),(0x20,0,15,2),(0x10,4,15,2)])
        if job==10:writes.append((0x10,2,15,0))
        status=5 if job in {7,8,9} else 1 if job==10 else 4 if job==12 else 0
        reads.extend([(0x60,job,0),(0x64,0,0),(0x68,tag,0),(0x6c,0,0),(0x70,status,0)])
        writes.extend([(0x40,job,15,0),(0x44,0,15,0),(0x48,tag,15,0),
                       (0x4c,1,15,0),(0x10,4,15,2),(0x4c,0,15,0),(0x10,4,15,0)])
        for epoch,win,addr in [(job,window,dest),(job+1,window+1,dest),(job+1,window,dest+1)]:
            desc(epoch,win,addr);writes.append((0x10,1,15,2))
    assert len(writes)==588 and len(reads)==70
    return writes,reads


def review(text,host_period_ps=4000,host_phase_ps=700):
    result=data_review(text,host_period_ps,host_phase_ps)
    result['periodic']=periodic_review(text)
    traces=defaultdict(list);intents=defaultdict(list);final=[]
    fields={'AW':{'awaddr','ps'},'W':{'wdata','wstrb','ps'},'B':{'bresp','ps'},
            'AR':{'araddr','ps'},'R':{'rdata','rresp','ps'}}
    for line in text.splitlines():
        if not line.startswith('NTRPU_HOST_CONTROL_'):continue
        label,*tokens=line.split();kind=label.removeprefix('NTRPU_HOST_CONTROL_')
        f=dict(t.split('=',1) for t in tokens)
        need(len(f)==len(tokens),'duplicate control field')
        if kind=='PASS':
            need(set(f)=={'writes','reads','jobs','ps'} and not final,'control final schema/count')
            final.append({k:int(v) for k,v in f.items()});continue
        need(not final,'control event after final')
        if kind in {'MMIO','READ'}:
            expected={'n','addr','data','resp','ps'} | ({'strb'} if kind=='MMIO' else set())
            need(set(f)==expected,'intent schema')
            row={k:int(v,16 if k in {'addr','data','strb'} else 10) for k,v in f.items()}
            need(row['n']==len(intents[kind])+1,'intent numbering')
            intents[kind].append(row)
        else:
            need(kind in {d+'_'+c for d in ['HOST','DEVICE'] for c in fields},'control marker')
            domain,channel=kind.split('_');need(set(f)==fields[channel],'control event schema')
            row={k:int(v,10 if k=='ps' else 16) for k,v in f.items()}
            need(row['ps']>result['clocks']['ps'],'control before admission')
            if traces[kind]:need(row['ps']>traces[kind][-1]['ps'],'control per-channel ordering')
            if domain=='HOST':need((row['ps']-host_phase_ps-host_period_ps//2)%host_period_ps==0,'control host edge')
            traces[kind].append(row)
    writes,reads=fixture()
    need([(r['addr'],r['data'],r['strb'],r['resp']) for r in intents['MMIO']]==writes,'control write fixture')
    need([(r['addr'],r['data'],r['resp']) for r in intents['READ']]==reads,'control read fixture')
    need(final==[dict(writes=588,reads=70,jobs=13,ps=result['memory']['summary']['ps'])],'control final')
    latencies={}
    for channel in fields:
        hosts,devices=traces['HOST_'+channel],traces['DEVICE_'+channel]
        n=588 if channel in {'AW','W','B'} else 70
        need(len(hosts)==len(devices)==n,'control conserved counts')
        samples=[]
        for host,device in zip(hosts,devices):
            need({k:v for k,v in host.items() if k!='ps'}=={k:v for k,v in device.items() if k!='ps'},'control CDC payload')
            # AMD lite_async holds source READY until dest_ack returns through
            # xpm_cdc_handshake (installed v2_1_36, lines514-646). Thus receiver
            # handshake precedes source acknowledgement, unlike the data FIFO.
            # These are acknowledgement delays, NOT forward payload latency.
            need((device['ps']>host['ps']) if channel in {'B','R'} else (host['ps']>device['ps']),'control CDC acknowledgement order')
            need(host['ps']<final[0]['ps'] and device['ps']<final[0]['ps'],'control final drain')
            samples.append(abs(host['ps']-device['ps']))
        latencies[channel]=dict(min_ps=min(samples),max_ps=max(samples),samples=n,
            measurement='receiver_handshake_to_source_acknowledgement')
    offsets={r['ps']%5000 for c in fields for r in traces['DEVICE_'+c]}
    need(len(offsets)==1,'control UI edge')
    for i,(a,v,s,b) in enumerate(writes):
        for domain in ['HOST','DEVICE']:
            aw,w,resp=(traces[domain+'_'+c][i] for c in ['AW','W','B'])
            need((aw['awaddr'],w['wdata'],w['wstrb'],resp['bresp'])==(a,v,s,b),'control write observations')
            need(resp['ps']>max(aw['ps'],w['ps']),'control B before request')
        need(intents['MMIO'][i]['ps']-traces['HOST_B'][i]['ps']==host_period_ps//2,'write intent versus handshake')
        if i:
            need(min(traces['HOST_AW'][i]['ps'],traces['HOST_W'][i]['ps'])>traces['HOST_B'][i-1]['ps'],'control write serialized')
    for i,(a,v,r) in enumerate(reads):
        for domain in ['HOST','DEVICE']:
            ar,resp=(traces[domain+'_'+c][i] for c in ['AR','R'])
            need((ar['araddr'],resp['rdata'],resp['rresp'])==(a,v,r),'control read observations')
            need(resp['ps']>ar['ps'],'control R before request')
        need(intents['READ'][i]['ps']-traces['HOST_R'][i]['ps']==host_period_ps//2,'read intent versus handshake')
    result.update(status='PASS_HOST_CONTROL_DATA_CDC_MIG_PERIODIC',control=dict(writes=588,reads=70,
        rejected_writes=sum(b!=0 for _,_,_,b in writes),decode_error_reads=sum(r!=0 for _,_,r in reads),
        partial_strobe_writes=sum(s!=15 for _,_,s,_ in writes),channel_latencies=latencies),
        scope='Actual AMD AXI4 data and AXI-Lite control CDC; UI control and guarded ingress drive generated generic cold MIG. Host AXI transactions, externally trusted producer isolation and direct memory readback remain BFMs. STOP is initiated at phase3 then applied after MMIO latency. No nativePCIe, actual upstream fence, physical CDC latency/metastability, warm reset, exactDIMM, exhaustivecapacity, physicaltiming or inference qualification.')
    return result
