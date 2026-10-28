"""Independent management and ownership review of the guarded rev3 composition."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
from check_ntrpu_xdma_stop_inventory_r4 import require, markers, vcd_rows
from check_ntrpu_hbm_xdma_native import LENGTHS, DEST, WB
from ntrpu_pcie_vcd import logic_vcd_rows
from ntrpu_pcie_straddle import LogicWord

ADDR = [0x1004,0x1048,0x1040,0x1004,0x1104,0x1148,0x1140,0x1104]


def management(rows, epoch):
    pub=[0,0];raw=[0,0];sent=[0,0];done_times=[[],[]]
    pending=None;poll=[];certificate=None;serial=0;boot=False;last_fault=0
    held={};reads=0;credits=[];ar_stalls=0;last_count=[0,0]
    for r in rows:
        t=r['time']
        require(r['faults'] & last_fault == last_fault,'management sticky fault');last_fault=r['faults']
        for name,v in [('published_count',pub),('observed_count',raw),('delivered_count',sent)]:
            require(r[name]==v[0]|v[1]<<32,'management inventory '+name)
        require(r['awvalid']==r['wvalid']==r['m_req_write']==r['m_req_strobe']==0,'management read-only')
        for stem,valid,ready,fields in [
            ('request','m_req_valid','m_req_ready',['m_req_address','m_req_identity']),
            ('AR','arvalid','arready',['araddr','arprot']),('R','rvalid','rready',['rdata','rresp'])]:
            payload=tuple(r[n] for n in fields)
            if stem in held:require(r[valid] and payload==held[stem],'stability '+stem)
            if r[valid] and not r[ready]:held[stem]=payload
            else:held.pop(stem,None)
        ar_stalls+=bool(r['arvalid'] and not r['arready'])
        if r['m_req_valid'] and r['m_req_ready']:
            require(pending is None,'single management transaction');serial+=1
            require(r['m_req_address']==ADDR[(serial-1)%8] and r['m_req_identity']==epoch<<64|serial,'address/identity order')
            pending={'address':r['m_req_address'],'identity':r['m_req_identity'],'ar':None,'r':None}
        if r['arvalid'] and r['arready']:
            require(pending and pending['ar'] is None and r['araddr']==pending['address'] and r['arprot']==0,'real AR')
            pending['ar']=t;reads+=1
        if r['rvalid'] and r['rready']:
            require(pending and pending['ar'] is not None and pending['r'] is None,'real R')
            pending['r']=(r['rdata'],r['rresp'],t)
        if r['m_rsp_valid'] and r['m_rsp_ready']:
            require(pending and pending['r'],'no invented response')
            data,code,rt=pending['r']
            require((data,code,pending['identity'])==(r['m_rsp_data'],r['m_rsp_code'],r['m_rsp_identity']),'AXI response identity')
            poll.append((pending['address'],data,rt,code==0 and not r['m_rsp_timeout']));pending=None
            if len(poll)==8:
                good=all(x[3] for x in poll)
                require([x[0] for x in poll]==ADDR,'full readback sequence')
                for c in range(2):
                    a,n,s,z=poll[c*4:c*4+4]
                    good &= a[1]==z[1]==0xf83e7f and s[1]&~5==0 and last_count[c]<=n[1]<=pub[c]
                    last_count[c]=n[1]
                if not boot:
                    good &= poll[1][1]==poll[5][1]==0;boot=bool(good)
                certificate=copy.deepcopy(poll) if good else None;poll=[]
        if r['admit_descriptor']:require(boot,'admission before complete boot readback')
        if r['faults']:require(r['admit_descriptor']==r['validated_done']==0,'fault blocks credits and descriptor grants')
        for c in range(2):
            if r['dsc_load']>>c&1:
                require(r['admit_descriptor']>>c&1 and pub[c]-sent[c]<2,'bounded actual descriptor load');pub[c]+=1
            if r['native_done']>>c&1:
                require(raw[c]<pub[c],'Done backed by published descriptor');raw[c]+=1;done_times[c].append(t)
            if r['validated_done']>>c&1:
                require(r['core_ready'] and r['exclusive_owner'] and not r['reset_request'] and not r['external_fault']
                        and not r['wait_expired'] and not r['m_timeout_latched'] and r['native_run']==3,'live credit authority')
                require(certificate and sent[c]<raw[c] and sent[c]+1<=certificate[c*4+1][1],'credit certificate')
                require(done_times[c][sent[c]]<certificate[c*4+1][2]<t,'Done before count before credit')
                sent[c]+=1;credits.append([t,c])
    return dict(reads=reads,credits=credits,published=pub,raw_done=raw,delivered=sent,faults=last_fault,ar_stalls=ar_stalls)


def extract_local(vcd):
    top='/tb_ntrpu_hbm_xdma_guarded/'
    cases={i:dict(dma=[],reader=[]) for i in range(1,12)}
    words={top+n for n in ['dsc_src','dsc_dst','dsc_len','dsc_ctl','axis_data','axis_keep','axis_last']}
    for t,b,a in logic_vcd_rows(vcd,words):
        phase=b.get(top+'phase')
        if phase not in cases:continue
        row={n[len(top):]:v for n,v in b.items() if n.startswith(top) and '/' not in n[len(top):]};row['time']=t
        if b[top+'dma_rst_n']==1 and b[top+'dma_clk']==0 and a[top+'dma_clk']==1:
            for name in ['m_req_valid','m_req_ready','m_req_write','m_req_address','m_req_strobe','m_req_identity',
                         'm_rsp_valid','m_rsp_ready','m_rsp_data','m_rsp_code','m_rsp_identity','m_rsp_timeout','m_timeout_latched']:
                row[name]=b[top+'dut/'+name]
            for suffix in ['awvalid','wvalid','arvalid','arready','araddr','arprot','rvalid','rready','rdata','rresp']:
                row[suffix]=b[top+'m_axil_'+suffix]
            row.update(faults=row['guard_faults'],native_done=row['dma_done'],native_run=row['dma_run'],
                       core_ready=row['management_core_ready'] and row['dma_link_up'],
                       reset_request=row['management_reset_request'],external_fault=row['management_fault'] or row['dma_error'],
                       wait_expired=row['management_wait_expired'])
            cases[phase]['dma'].append(row)
        if b[top+'rst_n']==1 and b[top+'clk']==0 and a[top+'clk']==1:cases[phase]['reader'].append(row)
    return cases


def review_local_events(cases):
    results=[]
    expected_faults={3:8,4:8,5:4096,6:512,7:2,8:1,9:4,10:1,11:8}
    for phase,events in cases.items():
        m=management(events['dma'],phase)
        require(m['published']==m['raw_done']==[2,2],'four actual descriptors and eventual Done')
        require(m['faults']==expected_faults.get(phase,0),'expected fault category')
        require(m['delivered']==([2,2] if phase<=2 else [0,0]),'credits only for healthy epochs')
        desc=[[],[]];packets=[[],[]];data=[bytearray(),bytearray()];held=[None,None]
        for r in events['dma']:
            for c in range(2):
                def part(n,w):
                    v=r[n]
                    if isinstance(v,LogicWord):
                        mask=((1<<w)-1)<<(c*w)
                        require(v.known&mask==mask,'unknown active channel '+n);v=v.value
                    require(v is not None,'unknown handshake '+n)
                    return (v>>(c*w))&((1<<w)-1)
                if part('dsc_load',1):
                    i=c+2*len(desc[c]);require(i<4,'bounded descriptor count')
                    require([part(n,w) for n,w in [('dsc_src',64),('dsc_dst',64),('dsc_len',28),('dsc_ctl',16)]]==
                            [WB[i],DEST[i],LENGTHS[i],16],'full descriptor fields');desc[c].append(r['time'])
                valid,ready=part('axis_valid',1),part('axis_ready',1)
                payload=tuple(part(n,w) for n,w in [('axis_data',512),('axis_keep',64),('axis_last',1)]) if valid else None
                if held[c]:require(valid and payload==held[c],'AXIS stable')
                held[c]=payload if valid and not ready else None
                if valid and ready:
                    require(payload[1]==(1<<64)-1,'AXIS keep');data[c].extend(payload[0].to_bytes(64,'little'))
                    if payload[2]:
                        i=c+2*len(packets[c]);require(i<4 and len(desc[c])>len(packets[c]),'packet attribution')
                        expected=bytes(((j*37)^(j>>7)^((i+1)*53))&255 for j in range(LENGTHS[i]))
                        require(data[c]==expected,'exact stream payload');packets[c].append(r['time']);data[c].clear()
        require([len(x) for x in packets]==[2,2] and not any(data),'four complete payloads')
        admitted=[[],[]];retired=[0,0];owned=[0,0];acks=[[],[]];local=[[],[]];held_results=[None,None]
        credit_times=[[t for t,c in m['credits'] if c==channel] for channel in range(2)]
        quarantine_samples=0
        for r in events['reader']:
            require(r['contexts_held']==owned[0]|owned[1]<<2,'ownership inventory')
            if r['submit_valid'] and r['submit_ready']:
                c=r['submit_channel'];admitted[c].append(r['submit_context']);owned[c]+=1
            for c in range(2):
                if r['ack_valid']>>c&1:acks[c].append(r['time'])
                if r['local_valid']>>c&1:local[c].append(r['time'])
                valid,ready=r['result_valid']>>c&1,r['result_ready']>>c&1
                fields=tuple((r[n]>>(c*w))&((1<<w)-1) for n,w in [('result_context',1024),('result_status',4),('result_bytes',64)])
                if held_results[c]:require(valid and fields==held_results[c],'stable result under backpressure')
                held_results[c]=fields if valid and not ready else None
                if valid and ready:
                    q=retired[c];require(q<2 and fields[0]==admitted[c][q],'error/success report identity')
                    require(fields[1:]==((0,LENGTHS[c+2*q]) if phase<=2 else (1,0)),'result status/length')
                    if phase<=2:
                        require(len(acks[c])>q and len(local[c])>q and r['time']>max(acks[c][q],local[c][q],credit_times[c][q]),
                                'success requires local CPL, validated DMA and ACK');owned[c]-=1
                    retired[c]+=1
            if phase>2 and retired==[2,2]:
                require(owned==[2,2],'error reports must not free ownership');quarantine_samples+=1
        require(retired==[2,2] and [len(x) for x in admitted]==[2,2],'four attributed reports')
        require(owned==([0,0] if phase<=2 else [2,2]),'final retained contexts')
        if phase>2:require(quarantine_samples>=80,'late events observed while quarantined')
        results.append(dict(phase=phase,management=m,payload_bytes=4544,retired_reports=4,
                            retained_contexts=sum(owned),quarantine_samples=quarantine_samples))
    return results


def review_local(path):
    log=(path/'simulation.log').read_text();require('FATAL' not in log and 'HG ASSERT' not in log,'local simulator diagnostics')
    end=markers(log,'HG PASS');require(len(end)==1 and end[0]['cases']=='11','local completion')
    cases=extract_local((path/'guarded.vcd').read_text());result=review_local_events(cases)
    require(len(markers(log,'HG CASE'))==11,'case evidence')
    negatives=[]
    for name in ['credit_without_readback','retained_context_erased','wrong_error_identity','payload_changed','clear_status_read']:
        bad=copy.deepcopy(cases)
        if name=='credit_without_readback':
            next(r for r in bad[1]['dma'] if r['m_req_valid'])['validated_done']=1
        elif name=='retained_context_erased':bad[3]['reader'][-1]['contexts_held']=0
        elif name=='wrong_error_identity':
            next(r for r in bad[3]['reader'] if r['result_valid'] & r['result_ready'])['result_context']^=1
        elif name=='payload_changed':
            row=next(r for r in bad[1]['dma'] if r['axis_valid'] & r['axis_ready'] & 1)
            v=row['axis_data'];row['axis_data']=LogicWord(v.value^1,v.known,v.width)
        else:next(r for r in bad[1]['dma'] if r['m_req_valid'] and r['m_req_ready'])['m_req_address']=0x1044
        try:review_local_events(bad)
        except ValueError:negatives.append(name)
        else:raise ValueError('undetected mutation '+name)
    return dict(status='GUARDED_BRIDGE_LOCAL_PASS',cases=result,negative_controls=negatives)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('path',type=Path)
    print(json.dumps(review_local(ap.parse_args().path),indent=2))
