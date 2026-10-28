"""Review cold PIPE/XDMA -> supervised FFN -> MIG/DDR numerical evidence."""
import argparse
from collections import Counter, deque
import json
from pathlib import Path
import re

from check_ntrpu_pcie_ffn import reference, require, review as review_pcie


def rows(text, marker):
    return [dict(re.findall(r'(\w+)=([^ ]+)', line)) for line in text.splitlines()
            if line.startswith(marker+' ')]


def review(text):
    require(not re.search(r'(?mi):\s*WARNING\s*:',text),
            'unreviewed hierarchically prefixed model warning')
    result=review_transfers(text)
    result['status']='PASS'
    return result


def review_transfers(text):
    """Numerical/protocol subset; model warnings require a separate review."""
    # Micron diagnostics prefix ERROR with the instance/task hierarchy.
    # A start-of-line-only tool diagnostic check would miss those messages.
    require(not re.search(r'(?mi)\b(?:ERROR|FATAL)\s*:|^\s*FAIL\b|^\s*CRITICAL WARNING:',text),
            'tool or hierarchically prefixed DDR model error')
    result=review_pcie(text,concurrent=True,rx_split=True)
    ui=rows(text,'NTRPU_DDR_FFN_UI')
    require(len(ui)==2 and [int(r['sample']) for r in ui]==[1,2],'two actual MIG UI samples')
    phase=int(ui[0]['ps'])%5000
    require(int(ui[1]['ps'])-int(ui[0]['ps'])==5000,'actual MIG UI period')
    boot=rows(text,'NTRPU_PCIE_DDR_ADMITTED')
    require(len(boot)==1,'one cold admission')
    admitted=int(boot[0]['time_ps'])
    state=rows(text,'NTRPU_PCIE_DDR_STATE')
    require(state and any(r['pcie_resetn']=='0' and r['hrst']=='0' for r in state),'observed actual XDMA reset')
    held=[r for r in state if r['ph']=='1']
    live=[r for r in state if all(r[k]=='1' for k in ['pcie_resetn','link','rst','hrst','da','ha','run','pr','calib']) and r['ui_rst']=='0']
    require(held and live and int(held[0]['ps'])<int(live[0]['ps'])<=admitted,'held/ready/admission ordering')
    require(int(ui[0]['ps'])<admitted,'calibration before host admission')
    for r in state:
        if int(r['ps'])>=admitted:
            require(all(r[k]=='1' for k in ['pcie_resetn','link','rst','hrst','da','ha','run','pr','calib']) and r['ui_rst']=='0','cold operation keeps both peers admitted')

    physical_w=deque();write=None;read=None
    aw=ar=wb=rb=bs=0;last_w_ps=0;last_b_ps=0
    accepted_w_times=[];associated_w_times=[]
    numeric_read=None
    for line in text.splitlines():
        marker=line.split(' ',1)[0]
        if marker not in {'NTRPU_PCIE_FFN_RAM_AW','NTRPU_PCIE_FFN_RAM_AR',
                         'NTRPU_PCIE_FFN_RAM_W','NTRPU_PCIE_FFN_RAM_R',
                         'NTRPU_PCIE_DDR_W','NTRPU_PCIE_DDR_R','NTRPU_PCIE_DDR_B',
                         'NTRPU_PCIE_FFN_JOB','NTRPU_PCIE_FFN_DONE'}:continue
        r=dict(re.findall(r'(\w+)=([^ ]+)',line))
        ps=int(r.get('ps',r.get('time_ps')))
        require(ps>admitted and ps%5000==phase,'actual transfer/core observation on admitted MIG UI edge')
        if marker=='NTRPU_PCIE_FFN_RAM_AW':
            require(write is None,'one memory write awaiting B')
            write=dict(addr=int(r['addr'],16),beats=int(r['beats']),ident=int(r['id']),aw_ps=ps,received=0)
            aw+=1
        elif marker=='NTRPU_PCIE_DDR_W':
            data=int(r['data'],16);mask=int(r['strb'],16);last=int(r['last'])
            require(mask==2**64-1 and last in {0,1},'actual MIG W strobes/LAST known')
            physical_w.append((data,last,ps));accepted_w_times.append(ps);wb+=1
        elif marker=='NTRPU_PCIE_FFN_RAM_W':
            require(write is not None and physical_w,'association must have accepted AW and W')
            data,last,when=physical_w.popleft();i=write['received']
            require(i<write['beats'] and last==int(i+1==write['beats']),'raw W LAST matches accepted AW length')
            require(int(r['addr'],16)==write['addr']+64*i and int(r['id'])==write['ident'],'association address/ID from actual AW')
            require(int(r['data'],16)==data and ps>=max(when,write['aw_ps']),'association contains only actual accepted W data')
            write['received']+=1;last_w_ps=when;associated_w_times.append(ps)
        elif marker=='NTRPU_PCIE_DDR_B':
            require(write is not None and not physical_w and write['received']==write['beats'],'B drains exact accepted W burst')
            require(int(r['id'])==write['ident'] and int(r['resp'])==0,'actual MIG B identity/OKAY')
            require(ps>max(write['aw_ps'],last_w_ps),'B follows complete AW/W')
            write=None;bs+=1;last_b_ps=ps
        elif marker=='NTRPU_PCIE_FFN_RAM_AR':
            require(read is None,'one memory read burst')
            read=dict(addr=int(r['addr'],16),beats=int(r['beats']),ident=int(r['id']),ps=ps,received=0);ar+=1
        elif marker=='NTRPU_PCIE_FFN_RAM_R':
            require(numeric_read is None and read is not None,'one numerical R association')
            numeric_read=r
        elif marker=='NTRPU_PCIE_DDR_R':
            require(read is not None and numeric_read is not None,'actual R linked to address/data observation')
            i=read['received'];n=numeric_read
            require(int(n['addr'],16)==read['addr']+64*i and int(n['id'])==int(r['id'])==read['ident'],'actual R address/ID')
            require(int(n['time_ps'])==ps and ps>read['ps'] and int(n['data'],16)==int(r['data'],16),'actual R time/data')
            require(int(r['resp'])==0 and int(r['last'])==int(i+1==read['beats']),'actual R response/LAST')
            numeric_read=None;rb+=1;read['received']+=1
            if read['received']==read['beats']:read=None
        elif marker=='NTRPU_PCIE_FFN_DONE' and r['sequence']=='00000001':
            require(write is None and not physical_w and last_b_ps<ps,'FFN completion after all output B responses')
    require(write is None and read is None and not physical_w and numeric_read is None,'all MIG transactions drained')
    require(wb*64==sum(map(len,reference()))==4032 and rb*64==result['ram_read_bytes']==4096,'exact physical W/R byte totals')
    require(aw==bs>0 and ar>0,'complete address/response coverage')
    passes=rows(text,'NTRPU_PCIE_DDR_PASS')
    require(len(passes)==1,'one complete DDR outcome')
    summary=passes[0]
    require({k:int(summary[k]) for k in ['aw','w','b','ar','r']}==dict(aw=aw,w=wb,b=bs,ar=ar,r=rb),'reconstructed DDR totals')
    ranks=Counter()
    for r in rows(text,'NTRPU_DDR_FFN_ACT'):
        rank=int(r['rank']);require(rank in range(4),'valid observed rank')
        ranks[rank]+=1;require(int(r['count'])==ranks[rank],'rank activation sequence')
    require(sum(ranks.values())>0 and [int(summary['rank'+str(i)]) for i in range(4)]==[ranks[i] for i in range(4)],'observed DRAM commands agree with summary')
    jobs=rows(text,'NTRPU_PCIE_FFN_JOB');done=rows(text,'NTRPU_PCIE_FFN_DONE')
    require((int(done[0]['time_ps'])-int(jobs[0]['time_ps']))%5000==0,'exact measured compute cycle interval')
    result.update(status='PASS_TRANSFERS_ONLY',mig_counts=dict(aw=aw,w=wb,b=bs,ar=ar,r=rb),
        rank_activations=[ranks[i] for i in range(4)],ui_period_ps=5000,ui_phase_ps=phase,
        host_admitted_ps=admitted,w_association_delayed_beats=sum(a!=z for a,z in zip(accepted_w_times,associated_w_times)),
        scope='Cold PIPE/XDMA, real generated MIG and generic DDR simulation with supervised guarded FFN. Passive RAM-labelled records associate accepted bytes; DUT read responses come only from MIG. No exact DIMM, warm reset, full-capacity PCIe coverage, driver, physical timing or hardware throughput qualification.')
    return result


def mutations(text):
    variants={}
    for marker in ['PCIE_DDR_ADMITTED','PCIE_DDR_PASS','DDR_FFN_UI','DDR_FFN_ACT',
                   'PCIE_DDR_W','PCIE_DDR_R','PCIE_DDR_B','PCIE_FFN_RAM_AW',
                   'PCIE_FFN_RAM_AR','PCIE_FFN_RAM_W','PCIE_FFN_RAM_R',
                   'PCIE_FFN_DONE','PCIE_FFN_ACK','PCIE_FFN_SERVICE_ACCEPT','PCIE_FFN_PASS']:
        variants['missing_'+marker]=re.sub(r'^NTRPU_'+marker+r' .+\n','',text,count=1,flags=re.M)
    for marker,key,base in [('PCIE_DDR_W','data',16),('PCIE_DDR_W','strb',16),('PCIE_DDR_W','last',10),
                            ('PCIE_DDR_W','ps',10),('PCIE_DDR_R','data',16),('PCIE_DDR_R','resp',10),
                            ('PCIE_DDR_R','id',10),('PCIE_DDR_R','last',10),('PCIE_DDR_R','ps',10),
                            ('PCIE_DDR_B','id',10),('PCIE_DDR_B','resp',10),('PCIE_DDR_B','ps',10),
                            ('PCIE_FFN_RAM_AW','addr',16),('PCIE_FFN_RAM_AW','beats',10),
                            ('PCIE_FFN_RAM_W','data',16),('PCIE_FFN_RAM_R','data',16),
                            ('PCIE_FFN_MMIO_R','data',16),
                            ('PCIE_FFN_IRQ','level',10),('PCIE_FFN_DMA','status',16),
                            ('PCIE_DDR_PASS','w',10),('DDR_FFN_ACT','rank',10)]:
        def alter(m,base=base):
            value=int(m[2],base)^1
            return m[1]+(format(value,'x') if base==16 else str(value))
        variants['corrupt_'+marker+'_'+key]=re.sub(r'(^NTRPU_'+marker+r' .*?\b'+key+r'=)([0-9a-f]+)',alter,text,count=1,flags=re.M)
    variants['model_error']=text+'\nERROR: injected DDR model diagnostic\n'
    variants['prefixed_model_error']=text+'\nmodel.zq_decode:ERROR: cmdZQ Failure. All banks must be Precharged. @1206615625\n'
    variants['prefixed_model_warning']=text+'\nModel:WARNING: Reading unwritten address: C:0 BG:0 B:0 R:0 C:1 dm:0 @29193811\n'
    def corrupt_cq_payload(m):
        line=m[0];r=dict(re.findall(r'(\w+)=([^ ]+)',line))
        value=int(r['data'],16)^(1<<(int(r['first'])*8))
        return line.replace('data='+r['data'],'data='+format(value,'x'))
    variants['corrupt_cq_payload']=re.sub(r'^NTRPU_PCIE_FFN_CQ_DATA .+',corrupt_cq_payload,text,count=1,flags=re.M)
    variants['missing_reset_observation']=re.sub(r'^NTRPU_PCIE_DDR_STATE .*\bpcie_resetn=0\b.*\n','',text,flags=re.M)
    variants['early_admission']=re.sub(r'(^NTRPU_PCIE_DDR_ADMITTED time_ps=)\d+',r'\g<1>0',text,count=1,flags=re.M)
    return variants


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log',type=Path);parser.add_argument('--output',type=Path)
    args=parser.parse_args();result=review(args.log.read_text(encoding='utf-8'))
    if args.output:args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
