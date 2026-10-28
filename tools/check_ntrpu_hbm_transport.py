"""Independent replay from submitted descriptors through AXI3 and completions."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def need(ok,why):
    if not ok:raise ValueError(why)

def pattern(tag,beat):
    return sum(((tag*0x9e3779b9)^(beat*0x10203)^(k*0x01010101))%2**32<<(32*k) for k in range(16))

def replay(text):
    cases=[];current=None;finished=False
    for line in text.splitlines():
        f=line.split()
        if len(f)<2 or f[0] not in ('TE','HT'):continue
        if f[:2]==['TE','CASE']:
            need(current is None or current['pass'] is not None,'Unfinished case')
            need(int(f[3])==len(cases)+1,'Case order')
            current=dict(number=int(f[3]),events=[],states=[],pass_=None)
            current['pass']=None;cases.append(current)
        elif f[:2]==['HT','ALL_PASS']:
            need(f[2]=='cases=17' and len(cases)==17,'Suite count');finished=True
        elif f[:2]==['TE','CASE_PASS']:
            need(current is not None and current['pass'] is None,'Unexpected pass')
            need(int(f[2])==current['number'],'Pass case mismatch');current['pass']=list(map(int,f[3:]))
        elif current is not None:
            if f[:2]==['TE','S']:current['states'].append(f)
            else:current['events'].append(f)
    need(finished and len(cases)==17,'Missing suite completion')
    results=[]
    for case in cases:
        number=case['number'];events=case['events'];jobs={};order=[[],[]]
        children=[{},{}];parent_order=[[],[]];completions=[[],[]];source=[]
        # First reconstruct each independent channel; link them afterwards.
        for f in events:
            prefix,kind=f[:2]
            if prefix=='TE' and kind=='SUB':
                cycle,lane,tag=map(int,f[2:5]);desc=int(f[5],16)
                need(tag not in jobs and tag==(desc>>136)&((1<<64)-1),'Descriptor tag')
                beats=(desc&255)+1;addr=(desc>>72)&((1<<64)-1)
                need(addr==(1<<34 if number==15 else (1<<32)+tag*4096),'Descriptor destination')
                jobs[tag]=dict(lane=lane,desc=desc,beats=beats,address=addr,SUB=cycle,data=[],out=[],children=[])
                order[lane].append(tag)
            elif prefix=='TE' and kind=='FETCH':
                cycle,lane,tag=map(int,f[2:5]);p=jobs[tag]
                need('FETCH' not in p and lane==p['lane'] and int(f[5],16)==p['desc'] and cycle>=p['SUB'],'Fetch owner/descriptor')
                p['FETCH']=cycle;source.append(tag)
            elif prefix=='TE' and kind=='IN':
                cycle,tag,beat=map(int,f[2:5]);data=int(f[5],16);p=jobs[tag]
                need('FETCH' in p and cycle>=p['FETCH'] and beat==len(p['data'])<p['beats'],'Input ownership/count')
                need(data==pattern(tag,beat),'Input payload');p['data'].append((cycle,data))
            elif prefix=='TE' and kind=='AW':
                cycle,lane,tag=map(int,f[2:5]);addr=int(f[5],16);beats=int(f[6]);p=jobs[tag]
                need('AW' not in p and 'FETCH' in p and lane==p['lane'],'Parent AW owner')
                need(addr==p['address'] and beats==p['beats'] and cycle>=p['FETCH'],'Parent AW metadata')
                p['AW']=cycle;parent_order[lane].append(tag)
            elif prefix=='TE' and kind=='W':
                cycle,lane,tag,beat,last=map(int,f[2:7]);data=int(f[7],16);p=jobs[tag]
                need(lane==p['lane'] and beat==len(p['out'])<len(p['data']),'Parent W owner/count')
                need(last==(beat==p['beats']-1) and data==p['data'][beat][1] and cycle>=p['data'][beat][0],'Parent W data/causality')
                p['out'].append((cycle,data))
            elif prefix=='TE' and kind in ('B','CPL'):
                cycle,lane,tag,status=map(int,f[2:]);p=jobs[tag]
                need(kind not in p and lane==p['lane'],'Duplicate/misowned response')
                p[kind]=(cycle,status)
                if kind=='CPL':completions[lane].append(tag)
            elif prefix=='HT' and kind in ('AW','W','B'):
                cycle,lane,index=map(int,f[2:5]);c=children[lane].setdefault(index,dict(data=[]))
                if kind=='AW':
                    need('AW' not in c,'Duplicate child AW');c.update(AW=cycle,address=int(f[5],16),beats=int(f[6]))
                    need(1<=c['beats']<=16,'Child length')
                elif kind=='W':
                    beat=int(f[5]);need(beat==len(c['data'])<16,'Child data count/order')
                    c['data'].append((cycle,int(f[6],16),int(f[7],16),int(f[8])))
                else:
                    need('B' not in c and int(f[5])==0,'Duplicate/wrong child B');c['B']=(cycle,int(f[6]))
        total_bytes=0;peak_parents=0;peak_children=0;w_before_aw=0
        for lane in range(2):
            # Valid parents occupy exact consecutive fragments; rejects use none.
            cursor=0
            for tag in parent_order[lane]:
                p=jobs[tag];size=p['beats']*64
                permitted=p['address']+size<=1<<34
                offset=0
                if permitted:
                    while offset<size:
                        need(cursor in children[lane],'Missing child');c=children[lane][cursor];cursor+=1
                        need('AW' in c and c['address']==p['address']+offset,'Child address/order')
                        need(c['AW']>=p['AW'] and offset+c['beats']*32<=size,'Child AW causality/coverage')
                        need(len(c['data'])==c['beats'],'Missing child data')
                        for k,(cy,data,mask,last) in enumerate(c['data']):
                            word=offset//32+k;beat=word//2;half=word%2
                            need(beat<len(p['out']) and cy>=p['out'][beat][0],'Child before parent data')
                            need(mask==0xffffffff and data==(p['out'][beat][1]>>(half*256))%2**256,'Child payload/strobe')
                            need(last==(k==c['beats']-1),'Child WLAST');total_bytes+=32
                            w_before_aw+=int(cy<c['AW'])
                        if 'B' in c:
                            need(c['B'][0]>max(c['AW'],c['data'][-1][0]),'Premature child response')
                            need(c['B'][1]==(2 if number==5 and tag==400 else 0),'Target error injection')
                        else:need(number in (16,17),'Missing child response')
                        offset+=c['beats']*32;p['children'].append(c)
                    need(len(p['children'])==(size+511)//512,'Excess fragments')
                need(len(p['out'])==p['beats'] and len(p['data'])==p['beats'],'Incomplete parent payload')
                if number not in (16,17):
                    need('B' in p and 'CPL' in p,'Missing parent completion')
                    need(all('B' in c for c in p['children']),'Aggregate before child responses')
                    expected=max((c['B'][1] for c in p['children']),default=3)
                    need(p['B'][1]==p['CPL'][1]==expected,'Response aggregation/completion status')
                    need(p['B'][0]>=max([p['AW'],p['out'][-1][0]]+[c['B'][0] for c in p['children']]),'Premature parent B')
                    need(p['CPL'][0]>=p['B'][0],'Premature completion')
                else:need('B' not in p and 'CPL' not in p,'Quarantined response escaped')
            need(cursor==len(children[lane]),'Extra child transaction')
            need(completions[lane]==order[lane][:len(completions[lane])],'Completion order')
            need(parent_order[lane]==order[lane][:len(parent_order[lane])],'Parent order')
            transitions=[]
            for tag in parent_order[lane]:
                p=jobs[tag];transitions.append((p['AW'],1))
                if 'B' in p:transitions.append((p['B'][0],-1))
            n=0
            for _,delta in sorted(transitions):n+=delta;peak_parents=max(peak_parents,n);need(0<=n<=4,'Parent credits')
            transitions=[]
            for c in children[lane].values():
                transitions.append((min(c['AW'],c['data'][0][0]),1))
                if 'B' in c:transitions.append((c['B'][0],-1))
            n=0
            for _,delta in sorted(transitions):n+=delta;peak_children=max(peak_children,n);need(0<=n<=32,'Child credits')
        need(case['pass']==[len(x) for x in order]+[len(x) for x in completions],'Pass inventory')
        for f in events:
            if f[:2]==['TE','FENCED']:
                cy=int(f[2]);need(all('CPL' in p and p['CPL'][0]<=cy for p in jobs.values()),'Fence precedes completion')
            if f[:2]==['TE','DRAIN']:
                cy=int(f[2]);need(all('CPL' in p and p['CPL'][0]<=cy for p in jobs.values()),'Drain precedes completion')
        need(len(case['states'])>0,'Missing inventory samples')
        if number in (16,17):
            q=[f for f in events if f[:2]==['HT','QUARANTINE']]
            need(len(q)==1 and list(map(int,q[0][3:]))==[number,16 if number==16 else 0,1 if number==17 else 0],'Bad-B quarantine coverage')
        if number==14:
            q=[f for f in events if f[:2]==['HT','SATURATED']]
            need(len(q)==1 and int(q[0][3])==4 and peak_parents==4,'Parent saturation coverage')
        if number==15:need(len(children[0])==0 and jobs[960]['CPL'][1]==3,'Local reject coverage')
        if number==8:need(w_before_aw>0,'W-before-AW coverage')
        results.append(dict(case=number,submitted=len(jobs),completed=sum(map(len,completions)),
                            children=sum(map(len,children)),bytes=total_bytes,peak_parents=peak_parents,
                            peak_children=peak_children,w_before_aw=w_before_aw))
    return dict(status='HBM_TRANSPORT_FUNCTIONAL_PASS',cases=results,
                submitted=sum(r['submitted'] for r in results),completed=sum(r['completed'] for r in results),
                children=sum(r['children'] for r in results),bytes=sum(r['bytes'] for r in results))

def controls(text):
    result=[]
    for prefix in ('TE SUB ','TE FETCH ','TE IN ','TE AW ','TE W ','TE B ','TE CPL ','HT AW ','HT W ','HT B ','HT QUARANTINE '):
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith(prefix));del lines[i]
        try:replay('\n'.join(lines))
        except (ValueError,KeyError,IndexError):result.append('remove_'+prefix.strip())
        else:raise ValueError('Undetected trace removal: '+prefix)
    for index,name in ((6,'data'),(7,'strobe')):
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('HT W '));f=lines[i].split();f[index]=format(int(f[index],16)^1,'x');lines[i]=' '.join(f)
        try:replay('\n'.join(lines))
        except (ValueError,KeyError,IndexError):result.append('corrupt_'+name)
        else:raise ValueError('Undetected '+name)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args();out=a.output
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    files=json.loads((out/'FILES.json').read_text());r=json.loads((out/'result.json').read_text())
    need(all(sha(out/n)==h for n,h in files.items()),'Evidence hashes')
    need(r['sources']==r['sources_after'] and r['generated']==r['generated_after'],'Sources changed')
    need(r['experiment']=='hbm_transport' and r['stage']=='sim' and r['status']=='EXPORTED_PENDING_REVIEW','Tool failure')
    text=(out/'suite/simulation.log').read_text();need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',text,re.M),'Simulation failure')
    result=replay(text);result.update(negative_controls=controls(text),seconds=r['seconds'],checker_sha256=sha(Path(__file__)),
                                   scope='Same-clock functional composition; behavioral AXI3 sinks; no hard HBM, CDC or timing qualification')
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))

if __name__=='__main__':main()
