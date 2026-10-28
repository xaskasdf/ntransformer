"""Reconstruct the seven appended XDMA/guard/MIG transactions independently.

This transport subset does not replace full periodic or source review.
"""
import re

from check_ntrpu_pcie_ddr_ffn import rows
from check_ntrpu_pcie_ddr_periodic import records,require,review as review_baseline,review_periodic
from check_ntrpu_pcie_ffn import reference

BASE=0x1e0000fc0
DESCRIPTOR=0x600000100
SOURCE=0x600000fc0
DESTINATION=0x700000fc0
CASES=[(1,1,0),(2,0,1),(3,1,0),(4,1,1),(5,1,0),(6,0,0),(7,1,0)]


def check_drain(text,periodic,marker):
    drains=records(text,marker,['begin_ps','end_ps','falling_edges','requests','returns'])
    require(len(drains)==1,'one passive drain for '+marker)
    d=drains[0];phase=(periodic['configuration']['ps']+2500)%5000
    first=d['begin_ps']+(phase-d['begin_ps'])%5000
    if first==d['begin_ps']:first+=5000
    require(0<d['falling_edges']<200 and d['end_ps']==first+5000*(d['falling_edges']-1),
            'bounded falling-edge drain')
    counts=periodic['counters']
    require(d['requests']==d['returns']==counts['requests']==counts['accepts']==counts['returns']==counts['periodic_cas'],
            'all internal operations completed at drain')
    require(int(rows(text,'NTRPU_DDR_READ_RETURN')[-1]['ps'])<d['end_ps'], 'last actual response before drain')
    return d


def review_transfers(text):
    require(not re.search(r'(?mi)\b(?:ERROR|FATAL)\s*:|^\s*FAIL\b|^\s*CRITICAL WARNING:',text), 'no simulator/model error')
    marker='NTRPU_DMA_ERRORS_BEGIN '
    require(text.count(marker)==1,'one appended error suite')
    prefix,stage=text.split(marker)
    baseline=review_baseline(prefix)
    baseline['periodic_drain']=check_drain(prefix,baseline['periodic'],'NTRPU_DDR_PERIODIC_DRAIN')
    begin_suite=records(text,'NTRPU_DMA_ERRORS_BEGIN',['time_ps'])
    require(begin_suite==[dict(time_ps=baseline['periodic_drain']['end_ps'])], 'error suite starts after complete baseline drain')
    require(not re.search(r'^NTRPU_PCIE_FFN_(?:JOB|DONE|IRQ|ACK|RETIRE|SERVICE_ACCEPT) ',stage,re.M),
            'no unexpected compute/interrupt activity during DMA error cases')
    require(not rows(stage,'NTRPU_PCIE_DDR_STATE'),'no admission/reset transition during error suite')
    descriptions=records(stage,'NTRPU_DMA_ERROR_DESCRIPTOR',['case','direction','bad','bytes','time_ps'],
                         ['address','source','destination','raw'])
    passes=records(stage,'NTRPU_DMA_ERROR_CASE_PASS',['case','direction','bad','completed','polls','time_ps'],['status'])
    controls=records(stage,'NTRPU_DMA_ERROR_CONTROL',['case','direction','time_ps'],['value'])
    starts=records(stage,'NTRPU_DMA_ERROR_AXI_BEGIN',['case','direction','bad','ps'])
    host_pass=records(stage,'NTRPU_DMA_ERROR_HOST_PASS',['case','descriptors','source','received','time_ps'])
    summaries=records(stage,'NTRPU_DMA_ERROR_AXI_PASS',['case','scope','aw','w','b','ar','r','ps'])
    require(len(descriptions)==len(passes)==len(controls)==len(starts)==len(host_pass)==7 and len(summaries)==21,
            'one complete set of seven cases/three interfaces')
    for channel in ['AW','W','B','AR','R']:
        for r in rows(stage,'NTRPU_DMA_ERROR_AXI_'+channel):
            require(int(r['case']) in range(1,8) and int(r['scope']) in range(3),
                    'no bus events outside reviewed cases/interfaces')
    require({(r['case'],r['scope']) for r in summaries}=={(c,s) for c in range(1,8) for s in range(3)},
            'one summary for each case/interface')
    expected=reference()[7][:64]
    csr_reads=[(int(ps),int(value,16),int(address,16)) for ps,value,address in re.findall(
        r'^\[\s*(\d+)\] : Data read ([0-9a-fA-F]+) from Address ([0-9a-fA-F]+)$',stage,re.M)]
    result=[]
    for i,(case,direction,bad) in enumerate(CASES):
        d,p,c,begin,h=descriptions[i],passes[i],controls[i],starts[i],host_pass[i]
        for r in [d,p,begin]:
            require((r['case'],r['direction'],r['bad'])==(case,direction,bad),'case order/identity')
        require(c['case']==h['case']==case and c['direction']==direction,'control/host identity')
        addr=BASE+((1<<37) if bad else 0)
        source,destination=(addr,DESTINATION) if direction else (SOURCE,addr)
        desc=0xad4b0013 | (64<<32) | (source<<64) | (destination<<128)
        require((d['address'],d['source'],d['destination'],d['bytes'],d['raw'])==
                (DESCRIPTOR,source,destination,64,desc),'exact full-width descriptor bytes')
        require(d['time_ps']<=begin['ps']<c['time_ps']<p['time_ps'] and
                (i==0 or passes[i-1]['time_ps']<d['time_ps']),'sequential isolated cases')
        require(c['value']==0x00fffe7e,'error logging enabled, stopped incrementing control')
        csr_base=0x1000 if direction else 0
        require((c['time_ps'],c['value'],csr_base+4) in csr_reads,'control value from actual root read completion')
        mask=(0x200 if direction else 0x4000) if bad else 0
        state_mask=0x47 if bad else 7  # PG195: idle_stopped is status, not another error.
        require((p['status'] & ~state_mask)==mask and not p['status']&1 and
                (p['completed'] in (0,1) if bad else p['status']==6 and p['completed']==1) and 0<p['polls']<=100,
                'actual terminal XDMA DECERR or successful recovery status')
        polls=records(stage,'NTRPU_DMA_ERROR_STATUS',['case','direction','time_ps'],['value'])
        polls=[r for r in polls if r['case']==case]
        require(len(polls)==p['polls'] and polls[-1]['value']==p['status'],'terminal status is last actual poll')
        require(all(r['direction']==direction and c['time_ps']<r['time_ps']<p['time_ps'] and
                    r['value'] & ~state_mask & ~mask == 0 for r in polls),'poll direction/time/error class')
        require(all((r['time_ps'],r['value'],csr_base+0x40) in csr_reads for r in polls),
                'status values from actual root read completions')
        completed_reads=[(ps,value) for ps,value,address in csr_reads if
                         address==csr_base+0x48 and polls[-1]['time_ps']<ps<p['time_ps']]
        require(len(completed_reads)==1 and completed_reads[0][1]==p['completed'],
                'completion count from actual root read completion')
        received=bytearray();packet=None;desc_bytes=payload_bytes=0;requests=[]
        for line in stage.splitlines():
            if not line.startswith('NTRPU_PCIE_FFN_CQ_'):continue
            r=dict(re.findall(r'(\w+)=([^ ]+)',line))
            if not begin['ps']<=int(r['time_ps'])<=p['time_ps']:continue
            if line.startswith('NTRPU_PCIE_FFN_CQ_REQUEST '):
                require(packet is None,'no overlapping host packet')
                address=int(r['addr'],16);length=int(r['bytes']);typ=int(r['type'])
                # Actual generated XDMA: reads carry Relaxed Ordering, writes do not.
                # Match both CQ and the independent serialized PCIe header below.
                require(int(r['tc'])==0 and int(r['attr'])==(2 if typ==0 else 0) and 0<length<=256 and
                        (address&4095)+length<=4096,'host packet attributes/page/size')
                requests.append(r)
                if typ==0:
                    if address==DESCRIPTOR+desc_bytes and desc_bytes+length<=32:desc_bytes+=length
                    else:
                        require(not direction and address==SOURCE+payload_bytes and payload_bytes+length<=64,
                                'actual host read maps descriptor or H2C source')
                        payload_bytes+=length
                else:
                    require(typ==1 and direction and address==DESTINATION+len(received) and len(received)+length<=64,
                            'C2H posted write stays inside separate host result buffer')
                    packet=[address,length,bytearray()]
            else:
                require(packet is not None,'CQ data follows actual request')
                address=int(r['addr'],16);length=int(r['bytes']);first=int(r['first']);left=int(r['left'])
                require(address==packet[0] and left==packet[1]-len(packet[2]) and
                        first==(16 if not packet[2] else 0) and length==min(left,64-first),'C2H accepted lane progression')
                packet[2].extend(int(r['data'],16).to_bytes(64,'little')[first:first+length])
                if len(packet[2])==packet[1]:received.extend(packet[2]);packet=None
        require(packet is None and desc_bytes==32 and payload_bytes==(0 if direction else 64),'complete actual descriptor/payload requests')
        if direction:
            require(received==(bytes(64) if bad else expected) or bad and received==b'',
                    'exact numerical sentinel, or observed zero/no data for rejected read')
        else:require(not received,'H2C causes no host payload write')
        require((h['descriptors'],h['source'],h['received'])==(desc_bytes,payload_bytes,len(received)) and
                begin['ps']<h['time_ps']<=p['time_ps'],'host summary reconstructed from accepted CQ')
        # Check independent packet log counts/headers against accepted CQ.
        logged=[r for r in rows(stage,'NTRPU_PCIE_RX_LOG_PACKET') if r['channel']=='0' and
                begin['ps']<=int(r['time_ps'])<=p['time_ps']]
        require(len(logged)==len(requests),'one independent packet log per CQ request')
        for log,request in zip(logged,requests):
            write=int(request['type'])==1;count=int(request['bytes'])
            require(int(log['head'],16)==((0x60000000 if write else 0x20002000)|count//4) and
                    int(log['bytes'])==16+(count if write else 0) and int(log['time_ps'])>=int(request['time_ps']),
                    'actual 64-bit PCIe header and byte count')
        for scope in range(3):
            observed={channel:records(stage,'NTRPU_DMA_ERROR_AXI_'+channel.upper(),
                ['case','scope','ps']+({'aw':['len','size','burst','id'],'ar':['len','size','burst','id'],
                                     'w':['last'],'b':['id','resp'],'r':['id','resp','last']}[channel]),
                {'aw':['addr'],'ar':['addr'],'w':['data','strb'],'b':[],'r':['data']}[channel])
                      for channel in ['aw','w','b','ar','r']}
            observed={k:[r for r in v if r['case']==case and r['scope']==scope] for k,v in observed.items()}
            want={k:int(not(scope==2 and bad) and (k in ['ar','r'] if direction else k in ['aw','w','b'])) for k in observed}
            require({k:len(v) for k,v in observed.items()}==want,'exact upstream/MIG counts, no invalid forwarding')
            summary=next(r for r in summaries if r['case']==case and r['scope']==scope)
            require({k:summary[k] for k in want}==want and begin['ps']<summary['ps']<=p['time_ps'],
                    'interface summary from accepted handshakes')
            for channel,events in observed.items():
                for r in events:
                    require(begin['ps']<r['ps']<summary['ps'],'bus event inside its case')
                    require(r['ps']% (4000 if scope==0 else 5000)==
                            (0 if scope==0 else baseline['ui_phase_ps']),'actual host/MIG clock sampling')
                    if channel in ['aw','ar']:
                        require((r['addr'],r['len'],r['size'],r['burst'])==(addr,0,6,1) and r['id']<16,
                                'full untruncated AXI address/length/size/burst/ID')
                    elif channel=='w':
                        data=bytes(0xc3^n for n in range(64)) if bad else expected
                        require(r['data'].to_bytes(64,'little')==data and r['strb']==(1<<64)-1 and r['last']==1,
                                'actual full W payload and strobes')
                    elif channel=='b':
                        aw,w=observed['aw'][0],observed['w'][0]
                        require(r['id']==aw['id'] and r['resp']==(3 if bad else 0) and r['ps']>max(aw['ps'],w['ps']),
                                'B follows complete actual write and reports precise error')
                    else:
                        ar=observed['ar'][0]
                        require(r['id']==ar['id'] and r['resp']==(3 if bad else 0) and r['last']==1 and r['ps']>ar['ps'],
                                'R follows actual AR and reports precise error')
                        require(r['data'].to_bytes(64,'little')==(bytes(64) if bad else expected),'actual rejected zero read or exact sentinel')
        result.append(dict(case=case,direction=direction,bad=bad,status=p['status'],idle_stopped=(p['status']>>6)&1,completed=p['completed'],
                           c2h_bytes=len(received),elapsed_ps=p['time_ps']-begin['ps']))
    outcomes=records(stage,'NTRPU_DMA_ERRORS_PASS',['cases','rejected','valid_reads','valid_writes','time_ps'])
    require(len(outcomes)==1,'one complete seven-case outcome')
    outcome=outcomes[0]
    require((outcome['cases'],outcome['rejected'],outcome['valid_reads'],outcome['valid_writes'])==(7,2,4,1) and
            passes[-1]['time_ps']<outcome['time_ps'],'complete seven-case outcome after last transfer')
    return dict(status='PASS_ERROR_TRANSFERS_ONLY',baseline=baseline,cases=result,
                rejected=2,valid_reads=4,valid_writes=1,
                scope='Transport subset only; complete periodic chain and provenance still require their own review.')


def mutations(text):
    variants={}
    def change(name,marker,key,base=10,case=None,scope=None):
        candidates=rows(text,marker)
        row=next(r for r in candidates if (case is None or int(r.get('case',-1))==case) and
                 (scope is None or int(r.get('scope',-1))==scope))
        line=next(l for l in text.splitlines() if l.startswith(marker+' ') and
                  dict(re.findall(r'(\w+)=([^ ]+)',l))==row)
        value=int(row[key],base)^1
        altered=line.replace(key+'='+row[key],key+'='+(format(value,'x') if base==16 else str(value)),1)
        variants[name]=text.replace(line,altered,1)
    for case in range(1,8):
        change('case_status_'+str(case),'NTRPU_DMA_ERROR_CASE_PASS','status',16,case)
        change('descriptor_'+str(case),'NTRPU_DMA_ERROR_DESCRIPTOR','raw',16,case)
    for marker,key,base,case in [
        ('AW','addr',16,2),('AW','len',10,2),('AW','size',10,2),('AW','burst',10,2),
        ('W','data',16,2),('W','strb',16,2),('W','last',10,2),
        ('B','resp',10,2),('B','id',10,2),('B','ps',10,2),
        ('AR','addr',16,4),('R','resp',10,4),('R','data',16,4),('R','last',10,4),
        ('R','id',10,4),('R','data',16,3),('W','data',16,6),('R','data',16,7)]:
        change('bus_'+marker+'_'+key+'_'+str(case),'NTRPU_DMA_ERROR_AXI_'+marker,key,base,case,0)
    for marker in ['NTRPU_DMA_ERROR_HOST_PASS','NTRPU_DMA_ERROR_AXI_PASS','NTRPU_DMA_ERRORS_PASS',
                   'NTRPU_DMA_ERROR_DESCRIPTOR','NTRPU_DMA_ERROR_CONTROL','NTRPU_DMA_ERROR_AXI_BEGIN']:
        line=next(l for l in text.splitlines() if l.startswith(marker+' '))
        variants['missing_'+marker]=text.replace(line+'\n','',1)
        variants['duplicate_'+marker]=text.replace(line,line+'\n'+line,1)
    change('wrong_control','NTRPU_DMA_ERROR_CONTROL','value',16,2)
    change('changed_scope','NTRPU_DMA_ERROR_AXI_AW','scope',10,2,0)
    line=next(l for l in text.splitlines() if l.startswith('NTRPU_DMA_ERROR_AXI_AW case=2 scope=1 '))
    row=dict(re.findall(r'(\w+)=([^ ]+)',line))
    variants['invalid_mig_forward']=text.replace(line,line+'\n'+line.replace('scope=1','scope=2',1),1)
    variants['truncated_address']=text.replace(line,line.replace('addr='+row['addr'],'addr='+format(BASE,'016x')),1)
    # A missing header, altered payload or duplicated posted packet must be
    # rejected independently of the app's numerical summary.
    boundary=text.index('NTRPU_DMA_ERRORS_BEGIN ')
    stage=text[boundary:]
    line=next(l for l in stage.splitlines() if l.startswith('NTRPU_PCIE_FFN_CQ_DATA '))
    row=dict(re.findall(r'(\w+)=([^ ]+)',line));value=int(row['data'],16)^(1<<(8*int(row['first'])))
    variants['host_payload']=text[:boundary]+stage.replace(line,line.replace('data='+row['data'],'data='+format(value,'x'),1),1)
    variants['missing_host_header']=text[:boundary]+re.sub(r'^NTRPU_PCIE_FFN_CQ_REQUEST .+\n','',stage,count=1,flags=re.M)
    for name,marker,old,new in [
        ('read_relaxed_ordering','NTRPU_PCIE_FFN_CQ_REQUEST type=0 ','attr=2','attr=0'),
        ('read_no_snoop','NTRPU_PCIE_FFN_CQ_REQUEST type=0 ','attr=2','attr=3'),
        ('write_relaxed_ordering','NTRPU_PCIE_FFN_CQ_REQUEST type=1 ','attr=0','attr=2'),
        ('read_serialized_attribute','NTRPU_PCIE_RX_LOG_PACKET channel=0 ','head=20002008','head=20000008')]:
        line=next(l for l in stage.splitlines() if l.startswith(marker) and old in l)
        variants[name]=text[:boundary]+stage.replace(line,line.replace(old,new,1),1)
    variants['prefixed_error']=text+'\nModel:ERROR: deliberate evidence corruption\n'
    return variants


def review(text):
    from decimal import Decimal
    result=review_transfers(text)
    periodic=review_periodic(text,summary_marker='NTRPU_DDR_PERIODIC_FINAL_PASS')
    d=check_drain(text,periodic,'NTRPU_DDR_PERIODIC_FINAL_DRAIN')
    finish=re.findall(r'\$finish called at time : ([0-9.]+) (ps|ns|us)',text)
    require(finish and {int(Decimal(v)*{'ps':1,'ns':1000,'us':1000000}[u]) for v,u in finish}=={d['end_ps']},
            'actual finish coincides with final drain')
    require(int(rows(text,'NTRPU_DMA_ERRORS_PASS')[0]['time_ps'])==d['end_ps'], 'suite result at final drain')
    result.update(status='PASS_COLD_PCIE_DDR_DMA_ERRORS',periodic=periodic,periodic_drain=d)
    return result
