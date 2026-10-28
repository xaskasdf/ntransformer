"""Replay parent/child writes independently, including enabled bytes and responses."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def need(ok, why):
    if not ok: raise ValueError(why)

def permitted(address, beats, size, burst, identity, lock):
    return (size==6 and burst==1 and identity==0 and lock==0 and address%64==0
            and address%4096+beats*64<=4096 and address+beats*64<=1<<34)

def payload(case, job, beat):
    return int.from_bytes(bytes((case*29+job*17+beat*7+k)%256 for k in range(64)), 'little')

def strobes(case, job, beat):
    return ((1<<64)-1,0x8000000100000001,0xffffffff,0)[(job+beat)%4] if case in (2,9) else (1<<64)-1

def child_response(case, child):
    if case!=3: return 0
    errors={0:{3:2},1:{4:3},2:{1:2,5:3},3:{1:3,5:2},4:{7:2},5:{7:3},6:{0:2}}
    return errors.get(child//8,{}).get(child%8,0)


def check_bytes(child, parent):
    need('AW' in child and len(child['data'])==child['beats'],'Incomplete child')
    enabled=0
    for beat,(cycle,data,strb,last) in enumerate(child['data']):
        need(last==(beat==child['beats']-1),'Child WLAST/count')
        word=child['offset']//32+beat;source=word//2;half=word%2
        need(source<len(parent['data']),'Child data before source data')
        expected,mask,accepted=parent['data'][source]
        need(cycle>=accepted,'Child data before accepted source data')
        expected=(expected>>(half*256))&((1<<256)-1);mask=(mask>>(half*32))&0xffffffff
        need(strb==mask,'Downsized strobes');bits=sum(255<<(8*k) for k in range(32) if mask>>k&1)
        need((data^expected)&bits==0,'Enabled data bytes changed');enabled+=mask.bit_count()
    return enabled


def replay(text):
    rows=[];case=0;last_cycle=-1;finished=False
    for line in text.splitlines():
        if not line.startswith('HA '):continue
        f=line.split();kind=f[1]
        if kind=='CASE':
            need(case==len(rows),'Unfinished case');cycle,case,njobs,delay=map(int,f[2:])
            need(case==len(rows)+1 and 1<=case<=11,'Case order')
            need(njobs==(24 if case==9 else 11 if case==8 else 1 if case>=10 else 8),'Case workload')
            need(delay==({3:32,6:64,9:41}.get(case,4)),'Target latency')
            parents=[];children=[];sw=sb=ma=mw=mb=0;wb=0;peak_parents=peak_children=0
            closed=None;released=False;quarantine=False;w_before_aw=0;enabled_bytes=0
        elif kind=='ALL_PASS':
            need(f[2]=='cases=11' and len(rows)==11 and not finished,'Final marker');finished=True;continue
        elif kind=='PASS':
            totals=list(map(int,f[2:]));need(totals==[case,len(parents),sw,sb,ma,mb],'Pass totals')
            need(len(parents)==njobs and sw==njobs and wb==0 and ma==mw==len(children),'Lost request/data')
            if case>=10:need(quarantine and sb==mb==0 and len(children)==1,'Missing quarantine')
            else:need(sb==njobs and mb==len(children),'Lost completion')
            need(released==(case==6),'Held-response coverage')
            need(closed is None,'Closed admission not reopened')
            need(all(not p['ok'] or p['addressed']==p['beats']*64 for p in parents),'Address coverage')
            enabled_bytes=sum(check_bytes(c,parents[c['parent']]) for c in children)
            row=dict(case=case,parents=njobs,completed=sb,local_rejects=sum(not p['ok'] for p in parents),
                     children=len(children),child_responses=mb,bus_bytes=sum(p['beats']*64 for p in parents if p['ok']),
                     enabled_bytes=enabled_bytes,peak_parents=peak_parents,peak_children=peak_children,w_before_aw=w_before_aw,
                     parent_statuses=[p.get('status') for p in parents],quarantined=case>=10)
            rows.append(row);continue
        else:
            cycle=int(f[2])
            need(not finished and case>0,'Event outside case')
            if kind=='SA':
                index=int(f[3]);address=int(f[4],16);beats,size,burst,identity,lock=map(int,f[5:])
                need(index==len(parents)<njobs and closed is None,'Parent admission/order')
                ok=permitted(address,beats,size,burst,identity,lock)
                parent=dict(address=address,beats=beats,identity=identity,ok=ok,SA=cycle,data=[],children=[],addressed=0)
                if ok:
                    for offset in range(0,beats*64,512):
                        parent['children'].append(len(children))
                        children.append(dict(parent=index,data=[]))
                parents.append(parent);peak_parents=max(peak_parents,len(parents)-sb)
                need(len(parents)-sb<=4,'Parent credits exceeded')
            elif kind=='SW':
                index,beat=int(f[3]),int(f[4]);data,strb,last=int(f[5],16),int(f[6],16),int(f[7])
                need(index==sw<len(parents),'Source data order');p=parents[index]
                need(beat==len(p['data'])<p['beats'] and last==(beat==p['beats']-1),'Source WLAST/count')
                need(data==payload(case,index,beat) and strb==strobes(case,index,beat),'Source workload data/strobes')
                p['data'].append((data,strb,cycle));sw+=last
            elif kind=='MA':
                index=int(f[3]);address=int(f[4],16);beats,size,burst,identity=map(int,f[5:])
                need(index==ma<len(children),'Unexpected child address');c=children[index]
                p=parents[c['parent']];offset=p['addressed']
                need(address==p['address']+offset and 1<=beats<=16 and offset+beats*32<=p['beats']*64,'Fragment address/length')
                need(size==5 and burst==1 and identity==0,'HBM attributes')
                c.update(AW=cycle,offset=offset,beats=beats);p['addressed']+=beats*32;ma+=1
            elif kind=='MW':
                index,beat=int(f[3]),int(f[4]);data,strb,last=int(f[5],16),int(f[6],16),int(f[7])
                need(index==mw<len(children),'Unexpected child data');c=children[index];p=parents[c['parent']]
                need(beat==len(c['data'])<16,'Child beat count');w_before_aw+=int('AW' not in c)
                c['data'].append((cycle,data,strb,last));mw+=last;wb=0 if last else beat+1
            elif kind=='MB':
                index,identity,resp=map(int,f[3:]);need(index==mb<len(children),'Child response order');c=children[index]
                need(identity==0 and resp==child_response(case,index),'Injected response mismatch')
                need('AW' in c and len(c['data'])==c['beats'],'Response before complete child')
                check_bytes(c,parents[c['parent']])
                need(cycle>=max(c['AW'],c['data'][-1][0])+delay+1,'Premature target response')
                c['B']=cycle;c['status']=resp;mb+=1
            elif kind=='SB':
                index,identity,resp=map(int,f[3:]);need(index==sb<len(parents),'Parent response order');p=parents[index]
                need(identity==p['identity'] and len(p['data'])==p['beats'],'Parent ID or incomplete input data')
                need(all('B' in children[i] for i in p['children']),'Premature aggregate B')
                expected=max((children[i]['status'] for i in p['children']),default=3)
                need(resp==expected,'Aggregated/local response status')
                p['status']=resp;p['B']=cycle;sb+=1
            elif kind=='RELEASE':
                need(case==6 and len(parents)==4 and sb==0 and not released,'Parent reply retention');released=True
            elif kind=='CLOSE':
                need(case==7 and int(f[3])==len(parents)==2 and closed is None,'Admission closure');closed=cycle
            elif kind=='REOPEN':
                need(case==7 and closed is not None and len(parents)==sb==2 and ma==mb==mw,'Premature reopening');closed=None
            elif kind=='QUARANTINE':
                need(case in (10,11) and not quarantine,'Unexpected quarantine')
                need(list(map(int,f[3:]))==[1,0,16 if case==10 else 0,0 if case==10 else 1],'Wrong bad response probe')
                need(ma==mw==1 and mb==sb==0,'Bad response was consumed')
                quarantine=True
            else:raise ValueError('Unknown event '+kind)
            outstanding=sum(bool('AW' in c or c['data']) and 'B' not in c for c in children)
            need(outstanding<=32,'HBM child capacity');peak_children=max(peak_children,outstanding)
        need(cycle>=last_cycle,'Time moved backwards');last_cycle=cycle
    need(finished and len(rows)==11,'Incomplete suite')
    need(rows[0]['children']==24 and rows[2]['children']==64,'Long burst coverage')
    need(rows[7]['local_rejects']==10 and rows[7]['completed']==11,'Reject/valid successor coverage')
    need(rows[5]['peak_parents']==4,'Parent saturation missing')
    return dict(status='HBM_ADAPTER_FUNCTIONAL_PASS',rows=rows,parents=sum(r['parents'] for r in rows),
                completed=sum(r['completed'] for r in rows),children=sum(r['children'] for r in rows),
                child_responses=sum(r['child_responses'] for r in rows),enabled_bytes=sum(r['enabled_bytes'] for r in rows))


def controls(text):
    rejected=[]
    for prefix in ['HA SA ','HA SW ','HA MA ','HA MW ','HA MB ','HA SB ','HA QUARANTINE ']:
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith(prefix));del lines[i]
        try:replay('\n'.join(lines))
        except (ValueError,IndexError,KeyError):rejected.append('missing_'+prefix.strip())
        else:raise ValueError('Undetected trace removal '+prefix)
    for field,name in [(5,'data'),(6,'strobe')]:
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('HA MW '));f=lines[i].split();f[field]=format(int(f[field],16)^1,'x');lines[i]=' '.join(f)
        try:replay('\n'.join(lines))
        except (ValueError,IndexError,KeyError):rejected.append('corrupt_'+name)
        else:raise ValueError('Undetected '+name)
    return rejected


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True);a=ap.parse_args();out=a.output
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    r=json.loads((out/'result.json').read_text());files=json.loads((out/'FILES.json').read_text())
    need(all(sha(out/n)==h for n,h in files.items()),'Evidence hash mismatch');need(r['sources']==r['sources_after'],'Sources changed')
    need(r['experiment']=='hbm_write_adapter','Wrong experiment')
    if r['status']!='EXPORTED_PENDING_REVIEW':
        need(r['status']=='FAILED_OR_TIMEOUT','Invalid status');result=dict(status='TOOL_FAILURE_DIAGNOSTIC')
    else:
        need('NTRPU_HBM_ADAPTER_COMPLETE sim' in (out/'console.log').read_text(),'Missing tool marker')
        expected={'hbm_width':dict(PROTOCOL='AXI4',READ_WRITE_MODE='WRITE_ONLY',ADDR_WIDTH='34',SI_DATA_WIDTH='512',MI_DATA_WIDTH='256',SI_ID_WIDTH='4',MAX_SPLIT_BEATS='16',ACLK_ASYNC='0',FIFO_MODE='0'),
                  'hbm_protocol':dict(SI_PROTOCOL='AXI4',MI_PROTOCOL='AXI3',READ_WRITE_MODE='WRITE_ONLY',ADDR_WIDTH='34',DATA_WIDTH='256',ID_WIDTH='4',TRANSLATION_MODE='2')}
        for name,props in expected.items():
            actual=dict(l.split('\t',1) for l in (out/'suite'/(name+'_config.tsv')).read_text().splitlines())
            need(all(actual['CONFIG.'+k]==v for k,v in props.items()),'IP configuration')
        log=(out/'suite/simulation.log').read_text();need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL )',log,re.M),'Simulation failure')
        result=replay(log);result['negative_controls']=controls(log)
    result.update(seconds=r['seconds'],checker_sha256=sha(Path(__file__)),scope='Functional write conversion with AMD IP and behavioral target; no hard HBM, CDC, timing closure or transport integration')
    a.review.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
