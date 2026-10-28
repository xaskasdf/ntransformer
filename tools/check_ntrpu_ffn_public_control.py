"""Check ABI2 using accepted host MMIO, IRQ and memory observations only.

CSR elapsed cycles are reported as CSR values; no internal job/done edge
is invented to validate them. This does not claim internal CDC latency.
"""
import re
from check_ntrpu_ffn_host import records
from check_ntrpu_ffn_ddr_cold_control import CAPACITY,region_base,expected_abi2_writes

def review_control(text,seed,host_period_ps,result):
    errors=result['errors']
    def need(ok,why):
        if not ok:raise ValueError(why)
    try:
        ids=[1,2,3]
        host=records(text,'NTRPU_FFN_HOST_CASE')
        writes=records(text,'NTRPU_FFN_HOST_WRITE')
        reads=records(text,'NTRPU_FFN_IRQ_READ')
        mem={r['id']:r for r in result['cases']}
        events={n:records(text,'NTRPU_FFN_IRQ_'+n) for n in ['BEGIN','SERVICE_REQUEST','EDGE','ACK','END']}
        for n in ['JOB','DONE','EVENT','SERVICE','RETIRE']:
            need(not records(text,'NTRPU_FFN_IRQ_'+n),'no invented internal event '+n)
        need([r['id'] for r in host]==ids,'three ordered host outcomes')
        for n in ['BEGIN','SERVICE_REQUEST','END']:need([r['id'] for r in events[n]]==ids,'one public '+n+' per job')
        need([w['order'] for w in writes]==[i%3 for i in range(len(writes))],'AW/W skew order')
        for rows in [reads,writes,events['EDGE'],events['ACK']]:need(all(r['id'] in ids for r in rows),'known owner')
        for row in reads+events['EDGE']+events['ACK']:
            need((row['time_ps']-700)%host_period_ps==host_period_ps//2,'public observation on host edge')
        # Primitive initialization may resolve after time zero; active operation
        # and all management samples must have fully known DUT state.
        for line in text.splitlines():
            if line.startswith(('NTRPU_DMA_REC_OUT ','NTRPU_DMA_REC_M ')):
                fields=dict(re.findall(r'(\w+)=(\S+)',line))
                if int(fields['ps'])>=1000:
                    need(all(fields[k] in {'0','1'} for k in ['ext','rst','hrst','da','ha','run','dh','dr','mh','mr']),'known DUT state after first nanosecond')
                else:need(fields['ext']=='1','initial settle occurs only under external reset')
        timings=[]
        for i in ids:
            mode=(i-1)%3;h=host[i-1];case=mem[i]
            ws=[r for r in writes if r['id']==i];rs=[r for r in reads if r['id']==i]
            ev={n:[r for r in rows if r['id']==i] for n,rows in events.items()}
            begin,request,end=ev['BEGIN'][0],ev['SERVICE_REQUEST'][0],ev['END'][0]
            need([(r['addr'],r['data'],r['resp']) for r in ws]==expected_abi2_writes(i,32,32,0,(2,2,2)),'exact public MMIO write schedule')
            need([(r['addr'],r['data']) for r in rs[:5]]==[(0,0x4652544e),(4,0x20000),(8,15),(0x18,i-1),(0x1c,i-1)],'identity/ABI/counters')
            need(rs[5]['addr']==0xc and rs[5]['data']&0x187==0x100,'idle before START')
            need(begin['mode']==mode and begin['fault']==0 and (begin['sequence'],begin['epoch'])==(0xfffffffe,i+600),'captured owner/mode')
            need(all(h[k]==v for k,v in dict(reset=0,status=0,stages=4,flags=0,cancel=0,sequence=0xfffffffe,epoch=i+600,bytes=128,stage_bytes=704).items()),'host completion metadata')
            need(h['bytes']==case['down_bytes'] and h['stage_bytes']==case['bytes'],'public physical byte agreement')
            p=h['polls'];need(p>0,'completion polling')
            polls=rs[6:6+p]
            need(len(polls)==p and all(r['addr']==0xc for r in polls),'completion polls')
            need(all(not r['data']&4 for r in polls[:-1]) and polls[-1]['data']&4,'first observed completion flag')
            complete_ps=polls[-1]['time_ps']
            bs=[r for r in records(text,'NTRPU_DDR_FFN_B') if r['id']==i and r['host']==0]
            need(bs and max(r['ps'] for r in bs)<complete_ps,'all output writes acknowledged before observed completion')
            need(begin['time_ps']<complete_ps<request['time_ps']<end['time_ps'],'public lifecycle ordering')
            need(end['mode']==mode and 0<end['retire_polls']<128,'bounded retirement')
            need(h['writes']==len(ws) and h['reads']==len(rs)==p+19+end['retire_polls']+int(mode==1),'MMIO accounting')
            need(0<h['compute_cycles']<h['total_cycles']<250000,'CSR elapsed range')
            need(h['program_cycles']>0 and h['program_cycles']+h['poll_cycles']<h['total_cycles'],'host elapsed range')
            need(h['total_cycles']>case['cycles'],'consumer finishes before retirement')
            middle=rs[6+p:6+p+13]
            need([r['addr'] for r in middle]==list(range(0x40,0x64,4))+[0x64,0x68,0x18,0x1c],'metadata register order')
            values={r['addr']:r['data'] for r in middle}
            need(values[0x40]==0x1c00 and values[0x44]==128 and values[0x48]==704,'status/phase/byte registers')
            need(values[0x4c]==values[0x64]==0xfffffffe and values[0x50]==values[0x68]==i+600,'admitted/completed owner registers')
            need(values[0x54]==values[0x60]==0 and values[0x58]+(values[0x5c]<<32)==h['compute_cycles'],'diagnostics and reported elapsed value')
            need(values[0x18]==values[0x1c]==i,'exactly-once completion counters')
            trailing=rs[6+p+13:]
            need(trailing and all(r['addr']==0xc for r in trailing) and trailing[-1]['data']&0x187==0x100,'idle after retirement')
            for row in rs:
                if row['addr']==0xc:
                    need(not row['data']&~0x1ff and not row['data']&0x80,'reserved/fault status bits')
                    need(not(row['data']&4 and row['data']&3),'completion excludes a new active job')
            if mode==2:need(not ev['EDGE'] and not ev['ACK'],'polling mode has no IRQ')
            else:
                need([r['level'] for r in ev['EDGE']]==[1,0] and len(ev['ACK'])==1,'one IRQ and transport ACK')
                rise,fall=[r['time_ps'] for r in ev['EDGE']];ack=ev['ACK'][0]['time_ps']
                need(begin['time_ps']<rise<=ack<fall<end['time_ps'],'public IRQ/ACK/retirement ordering')
                need(fall>request['time_ps'],'IRQ retained until software service requested')
                if mode==0:need(ack<request['time_ps'],'transport-first ACK order')
                else:
                    need(trailing[0]['data']&0x24==0x24,'completion held after software service')
                    need(request['time_ps']<trailing[0]['time_ps']<ack,'software-first ACK order')
            timings.append(dict(id=i,mode=mode,csr_compute_cycles=h['compute_cycles'],observed_completion_ps=complete_ps,
                service_request_to_idle_ps=end['time_ps']-request['time_ps']))
        need(all(events['END'][i]['time_ps']<events['BEGIN'][i+1]['time_ps'] for i in range(2)),'serial owner retirement')
        result.update(host_cases=host,mmio_writes=len(writes),mmio_reads=len(reads),public_timings=timings,
            completed_ids=ids,aborted_ids=[],host_period_ps=host_period_ps,
            scope='Public MMIO/IRQ lifecycle and independent memory oracle. CSR elapsed cycles are reported, not independently timed at unobserved internal job/done pins. No internal phase or CDC latency claim.')
    except (ValueError,KeyError,IndexError,TypeError) as exc:errors.append(str(exc))
    result['status']='FAIL' if errors else 'PASS'
    return result
