"""Independent event ledger for boundary sweeps; adapted from the frozen performance oracle."""
import argparse
import hashlib
import json
from pathlib import Path
import re

from check_ntrpu_candidate_transport import require, pattern
from check_ntrpu_transport_study import distribution, performance
from prepare_ntrpu_return_boundary import bench, PROFILES
from run_ntrpu_return_cut import SOURCES as FROZEN_SOURCES


def capacity(jobs, credits, peak):
    for lane in (0,1):
        count=sum(j['ch']==lane and ('AW' in j or j['out']>0) and 'B' not in j for j in jobs.values())
        require(count<=credits,'Target obligation capacity exceeded')
        peak[lane]=max(peak[lane],count)


def boundary(text):
    rows=[];case=0;jobs={};config=None;last_cycle=-1;samples=0;bytes_checked=0
    for line in text.splitlines():
        f=line.split()
        if line.startswith('TE CASE '):
            require(case==len(rows),'Incomplete performance case')
            case=int(f[3]);require(case==len(rows)+1,'Case order')
            jobs={};order=[[],[]];counts={k:[0,0] for k in ('FETCH','AW','W','B','CPL')}
            config=None;active=None;budget=0;writes=[];fetch_cycles=[];release=None;peak_banks=peak_queued=0
            peak_target=[0,0];w_before_aw=0;origin=int(f[2])
        elif line.startswith('BN CONFIG '):
            require(int(f[2])==case and config is None,'Duplicate config')
            config=tuple(map(int,f[3:]));profile,beats,count,credits,bdelay,pdelay=config
            require(profile==(case-1)//9 and credits==(1,2,4)[((case-1)//3)%3] and beats==(1,8,64)[(case-1)%3] and count==64,'Workload configuration')
            require(bdelay==(4,160,32,32,32)[profile] and pdelay==(64 if profile==2 else 0),'Latency configuration')
        elif line.startswith('BN RELEASE '):
            require(config[0]==3 and release is None and counts['CPL']==[0,64],'Blocked lane did not preserve peer progress')
            release=int(f[2])
        elif line.startswith('TE S '):
            cycle=int(f[2]);require(cycle>last_cycle and int(f[3])==case,'Sample sequence');last_cycle=cycle;samples+=1
            require(f[4:6]==['0','0'],'Unexpected stop/fault')
            require(int(f[14])==budget,'Sample payload budget')
            if config: capacity(jobs,credits,peak_target)
            queued=int(f[7],16);banks=int(f[8],16).bit_count()
            require(max(queued&15,queued>>4)<=10,'Queue overflow')
            peak_banks=max(peak_banks,banks);peak_queued=max(peak_queued,queued&15,queued>>4)
            if int(f[6]):require(all(j['done'] for j in jobs.values()) and budget==0,'False drain')
        elif line.startswith('TE CASE_PASS '):
            require(config and int(f[2])==case and list(map(int,f[3:]))==[64,64,64,64],'Workload totals')
            require(len(jobs)==128 and all(j['done'] for j in jobs.values()) and budget==0,'Lost job')
            require((release is not None)==(profile==3),'Missing blocked-peer case')
            total=128*beats*64;require(len(writes)*64==total,'Exact byte count');bytes_checked+=total
            begin=min(j['SUB'] for j in jobs.values());end=max(j['CPL'] for j in jobs.values())
            trim=len(writes)//8;window=sorted(writes)[trim:len(writes)-trim]
            span=window[-1]-window[0]+1
            latency={name:distribution([j[b]-j[a]+inclusive for j in jobs.values()])
                     for name,a,b,inclusive in [('submit_to_admit','SUB','ADMIT',0),('admit_to_reserve','ADMIT','RES',0),
                       ('reserve_to_fetch','RES','FETCH',0),('fetch_to_first_input','FETCH','first_in',0),
                       ('fill_span','first_in','last_in',1),('last_input_to_first_write','last_in','first_w',0),
                       ('last_write_to_response','last_w','B',0),('response_to_completion','B','CPL',0),
                       ('submit_to_completion','SUB','CPL',0)]}
            row=dict(case=case,profile=('fast','slow_b','payload_latency','lane0_blocked','aw_pause')[profile],
                     block_bytes=beats*64,jobs=128,bytes=total,cycles=end-begin+1,
                     batch_bytes_per_cycle=round(total/(end-begin+1),6),
                     batch_gbps_at_200mhz=round(total/(end-begin+1)/5,4),
                     trimmed_write_bytes_per_cycle=round(len(window)*64/span,6),
                     trimmed_write_gbps_at_200mhz=round(len(window)*64/span/5,4),
                     peak_banks=peak_banks,peak_queued_per_lane=peak_queued,latency_cycles=latency,
                     credits_per_lane=credits,b_delay_cycles=bdelay,payload_delay_cycles=pdelay,
                     peak_target_obligations=peak_target,w_beats_before_aw=w_before_aw,
                     fetch_interval_cycles=distribution([b-a for a,b in zip(fetch_cycles,fetch_cycles[1:])]))
            if release is not None:
                first=min(j['SUB'] for j in jobs.values() if j['ch']==1)
                last=max(j['CPL'] for j in jobs.values() if j['ch']==1)
                require(all(j['first_w']>release for j in jobs.values() if j['ch']==0),'Blocked lane wrote before release')
                row['peer_while_blocked']=dict(bytes=64*beats*64,cycles=last-first+1,gbps_at_200mhz=round(64*beats*64/(last-first+1)/5,4))
            rows.append(row)
        elif line.startswith('TE ') and f[1] in ('SUB','ADMIT','RES','FETCH','IN','AW','W','B','CORE_CPL','CPL'):
            kind=f[1];cyc=int(f[2]);require(cyc==last_cycle,'Unpaired event sample')
            if kind=='IN':
                tag,beat,data=int(f[3]),int(f[4]),int(f[5],16)
                require(active==tag and budget>0 and jobs[tag]['in']==beat and data==pattern(tag,beat),'Input ownership/bytes')
                j=jobs[tag];require(cyc-j['FETCH']>=pdelay+1,'Payload latency bypassed');j.setdefault('first_in',cyc);j['last_in']=cyc;j['in']+=1;budget-=1
                if budget==0:active=None
                continue
            ch,tag=int(f[3]),int(f[4]);require(ch in (0,1),'Lane')
            if kind=='SUB':
                require(config and tag not in jobs and tag==1+ch+2*len(order[ch]),'Duplicate or unordered publication')
                desc=int(f[5],16);expected=(0x76543210<<200)|(tag<<136)|((0x100000000+tag*4096)<<72)|((0x800000000+tag*4096)<<8)|(beats-1)
                require(desc==expected,'Descriptor metadata')
                jobs[tag]=dict(ch=ch,desc=desc,SUB=cyc,done=False,**{'in':0,'out':0});order[ch].append(tag)
                continue
            require(tag in jobs and jobs[tag]['ch']==ch,'Unknown job');j=jobs[tag]
            if kind=='W':
                beat,last,data=int(f[5]),int(f[6]),int(f[7],16)
                require(order[ch][counts[kind][ch]]==tag and 'FETCH' in j and beat==j['out']<j['in'],'Write order/input causality')
                require(data==pattern(tag,beat) and last==(beat==beats-1),'Write data/WLAST')
                j.setdefault('first_w',cyc);j['last_w']=cyc;j['out']+=1;counts[kind][ch]+=last;writes.append(cyc)
                w_before_aw+=int('AW' not in j);capacity(jobs,credits,peak_target)
                continue
            predecessor={'ADMIT':'SUB','RES':'ADMIT','FETCH':'RES','AW':'FETCH','B':'AW','CORE_CPL':'B','CPL':'CORE_CPL'}[kind]
            require(predecessor in j and kind not in j and cyc>=j[predecessor],'Job causality/duplicate event')
            if kind in counts:
                require(order[ch][counts[kind][ch]]==tag,'Per-lane ordering');counts[kind][ch]+=1
            if kind=='FETCH':
                require(active is None and budget==0 and int(f[5],16)==j['desc'],'Fetch ownership')
                active=tag;budget=beats
                fetch_cycles.append(cyc)
            elif kind=='AW':require(int(f[5],16)==0x100000000+tag*4096 and int(f[6])==beats,'AXI address')
            elif kind=='B':
                require(j['out']==beats and f[5]=='0','Write/response')
                require(cyc>=max(j['AW'],j['last_w'])+bdelay+1,'Premature target response')
            elif kind=='CPL':require(f[5]=='0' and not j['done'],'Completion status');j['done']=True
            j[kind]=cyc
            if kind=='AW':
                capacity(jobs,credits,peak_target)
                if profile==4:require((cyc-1-origin)%64>=32,'AW accepted during target pause')
    require(len(rows)==45 and len(re.findall(r'^BN ALL_PASS cases=45 ',text,re.M))==1,'Incomplete performance suite')
    return dict(status='RETURN_BOUNDARY_SWEEP_PASS',samples=samples,bytes=bytes_checked,rows=rows,
                timing_basis='5 ns simulation cycles; numerical GB/s conversion, not board throughput',
                window='Batch: first SUB through last CPL inclusive; trimmed W window removes first/last 1/8 of beat events')


def controls(text):
    rejected=[]
    for prefix in ['TE IN ','TE W ','TE AW ','TE B ','TE CPL ','TE CASE_PASS ']:
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith(prefix));del lines[i]
        try: boundary('\n'.join(lines))
        except (ValueError,IndexError,KeyError): rejected.append('missing_'+prefix.strip())
        else: raise ValueError('Undetected trace loss '+prefix)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('TE W '));f=lines[i].split();f[-1]=format(int(f[-1],16)^1,'x');lines[i]=' '.join(f)
    try: boundary('\n'.join(lines))
    except (ValueError,IndexError,KeyError): rejected.append('corrupted_write')
    else: raise ValueError('Undetected write corruption')
    # Independent capacity ledger must reject a two-obligation trace at one credit.
    try: capacity({1:dict(ch=0,AW=1,out=0),2:dict(ch=0,out=1)},1,[0,0])
    except ValueError: rejected.append('AW_and_W_obligations_overflow')
    else: raise ValueError('Capacity oracle failed')
    return rejected


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args();out=a.output
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((out/'result.json').read_text());files=json.loads((out/'FILES.json').read_text())
    require(r['experiment']=='return_boundary_sweep','Experiment')
    require(all(sha(out/n)==h for n,h in files.items()),'Evidence hash mismatch')
    require(all(r[n]==r[n+'_after'] for n in ['sources','inputs','rtl_evidence','plan_evidence']),'Changed sources/inputs')
    old=json.loads((out/'inputs/result.json').read_text())
    require(old['sources']==old['sources_after']=={n:r['sources'][n] for n in FROZEN_SOURCES},'Frozen source bridge')
    if r['status']!='EXPORTED_PENDING_REVIEW':
        require(r['status']=='FAILED_OR_TIMEOUT','Invalid result')
        result=dict(status='TIMEOUT_DIAGNOSTIC' if r['timed_out'] else 'TOOL_FAILURE_DIAGNOSTIC')
    else:
        console=(out/'console.log').read_text();s=out/'suite';g=s/'generated'
        require('NTRPU_RETURN_BOUNDARY_COMPLETE' in console and not re.search(r'^ERROR:',console,re.M),'Tool completion')
        for n in ['RETURN_CUT.json','AXI_INTEGRATION.json']:
            require(json.loads((out/'inputs'/n).read_text())==json.loads((g/n).read_text()),'Circuit fixture changed')
        props=lambda p:dict(l.split('\t',1) for l in p.read_text().splitlines())
        prior_ip=props(out/'inputs/ip_config.tsv');current_ip=props(s/'ip_config.tsv')
        # IP_FILE is project-local metadata; every other property must match.
        require({k:v for k,v in prior_ip.items() if k!='IP_FILE'}==
                {k:v for k,v in current_ip.items() if k!='IP_FILE'},'IP configuration changed')
        require(all(v['IP_FILE'].endswith('/sources_1/ip/ntrpu_mem_regslice/ntrpu_mem_regslice.xci')
                    for v in (prior_ip,current_ip)),'Unexpected IP source path')
        fixture=json.loads((g/'BOUNDARY.json').read_text());require(fixture['profiles']==PROFILES and fixture['cases']==45,'Sweep manifest')
        require((g/'tb_ntrpu_return_boundary.sv').read_text()==bench((g/'tb_ntrpu_transport_perf.sv').read_text()),'Unexpected bench changes')
        root=Path(__file__).resolve().parents[1]
        for n,h in fixture['unchanged_circuit']['logic_sha256'].items():
            p=g/(n+'.sv');p=p if p.exists() else root/'fpga/rtl'/(n+'.sv');require(sha(p)==h,'Circuit byte mismatch '+n)
        log=(s/'boundary.log').read_text();require(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',log,re.M),'Simulation error')
        result=boundary(log);result['negative_controls']=controls(log)
        result['ip_metadata_path_change']=[prior_ip['IP_FILE'],current_ip['IP_FILE']]
        baseline=performance((out/'inputs/perf.log').read_text())['rows'][:3]
        rows=[x for x in result['rows'] if x['profile']=='fast' and x['credits_per_lane']==4]
        result['baseline_comparison']=[dict(block_bytes=a['block_bytes'],prior_cycles=a['cycles'],sweep_cycles=b['cycles'],equal=a['cycles']==b['cycles']) for a,b in zip(baseline,rows)]
        require(all(x['equal'] for x in result['baseline_comparison']),'Fast reference changed')
        require(any(x['w_beats_before_aw']>0 for x in result['rows'] if x['profile']=='aw_pause'),'Missing W-before-AW coverage')
        require(all(max(x['peak_target_obligations'])==x['credits_per_lane'] for x in result['rows'] if x['profile']=='slow_b' and x['block_bytes']==64),'Credits not saturated')
    result.update(seconds=r['seconds'],stage='boundary',checker_sha256=sha(Path(__file__)),scope='Same-clock behavioral sensitivity only; no PCIe/HBM/CDC integration or new physical qualification')
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'}))


if __name__=='__main__':main()
