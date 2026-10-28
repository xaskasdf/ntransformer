"""Independently reconcile KV descriptors, AXI requests/responses and cache bytes."""
import argparse
import hashlib
import json
from pathlib import Path
import re

BASE=0xa000000000
LIMIT=1<<64
def need(ok,why):
    if not ok:raise ValueError(why)
def specifications():
    cases=[(1,1,0,0,0,0,0,0),(2,17,16,15,7,3,4032,0),(17,65,64,48,28,15,64,0),(64,131072,131071,131008,31,15,4032,0)]
    cases += [(64,131072,131071,131008,h,h%16,64,0) for h in range(32)]
    cases += [(0,131072,63,0,0,0,0,1),(65,131072,64,0,0,0,0,2)]
    cases += [(2,131072,63,0,3,0,0,s) for s in range(3,24)]
    cases += [(16,131072,63,0,31,15,0,s) for s in range(24,38)]
    for s in range(38,43):cases += [(16,131072,63,0,31,15,0,s),(1,1,0,0,0,0,0,0)]
    cases += [(64,131072,131071,131008,31,15,4032,s) for s in (43,44)]
    result=[]
    for ident,(n,cap,pos,start,qhead,layer,offset,s) in enumerate(cases,1):
        d=dict(id=ident,scenario=s,n=n,capacity=cap,position=pos,start=start,layer=layer,qhead=qhead,kvhead=qhead//4,
            committed=pos+1,cache=BASE+offset,region=BASE,bytes=cap*65536+8192,
            sequence=0x1234567800000000+ident,epoch=0xfedcba9800000000+ident,
            cache_sequence=0x1234567800000000+ident,cache_epoch=0xfedcba9800000000+ident,expected_epoch=0xfedcba9800000000+ident)
        if s==3:d['cache']+=1
        if s==4:d['bytes']=offset+cap*65536-1
        if s==5:d['region']=d['cache']+64
        if s==6:d.update(cache=LIMIT-4096,region=LIMIT-4096,bytes=8192)
        if s==7:d.update(cache=LIMIT-4096,region=0,bytes=LIMIT-1)
        if s==8:d['expected_epoch']+=1
        if s==9:d['cache_epoch']+=1
        if s==10:d['cache_sequence']+=1
        if s==11:d['sequence']=0
        if s==12:d['epoch']=0
        if s==13:d['capacity']=0
        if s==14:d['capacity']=131073
        if s==15:d['position']=cap
        if s==16:d['committed']=pos
        if s==17:d['committed']=cap+1
        if s==18:d['start']=(1<<32)-1
        if s==19:d['start']=pos
        if s==20:d['layer']=16
        if s==21:d['qhead']=32
        if s==22:d['kvhead']^=1
        if s==23:d['qhead']=256
        result.append(d)
    need(len(result)==85,'Case count');return result

def admission(d):
    if d['cache']%64 or not d['bytes'] or not 1<=d['capacity']<=131072 or not 1<=d['n']<=64 or not 0<=d['position']<d['capacity'] or not d['position']<d['committed']<=d['capacity'] or d['start']+d['n']>d['position']+1 or not 0<=d['layer']<16 or not 0<=d['qhead']<32 or d['kvhead']!=d['qhead']//4:return 1
    if not d['sequence'] or not d['epoch'] or d['sequence']!=d['cache_sequence'] or d['epoch']!=d['cache_epoch'] or d['epoch']!=d['expected_epoch']:return 2
    end=d['region']+d['bytes'];cache_end=d['cache']+65536*d['capacity']
    if end>=LIMIT or cache_end>=LIMIT or d['cache']<d['region'] or cache_end>end:return 1
    return 0

def address(d,plane,index=0):
    layer_bytes=d['capacity']*4096;plane_bytes=d['capacity']*2048;head_bytes=d['capacity']*256
    return d['cache']+d['layer']*layer_bytes+plane*plane_bytes+d['kvhead']*head_bytes+d['start']*256+index*64
def plan(d):
    out=[]
    for plane in (0,1):
        index=0
        while index<d['n']*4:
            addr=address(d,plane,index);beats=min(d['n']*4-index,(4096-addr%4096)//64)
            out.append(dict(id=d['id'],plane=plane,index=index,addr=addr,beats=beats));index+=beats
    return out
def memory(d,addr):
    result=0
    for lane in range(16):
        word=0x3f000000|(((addr>>2)^(addr>>33)^(addr>>40)^(lane*29+d['id']*17))&0x7fffff)
        if d['scenario']==43:word=[0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff][lane%6]
        result |= word<<(32*lane)
    return result

HEX={'cache','region','bytes','sequence','epoch','cache_sequence','cache_epoch','expected_epoch','addr','tail','data'}
def fields(line,prefix,keys):
    need(line.startswith(prefix+' '),'Marker order');parts=line[len(prefix)+1:].split()
    need(len(parts)==len(keys),'Field count');out={}
    for part,key in zip(parts,keys):
        pair=part.split('=');need(len(pair)==2 and pair[0]==key,'Field order/name')
        need(re.fullmatch('[0-9a-f]+' if key in HEX else '[0-9]+',pair[1]) is not None,'Malformed numeric field')
        out[key]=int(pair[1],16 if key in HEX else 10)
    return out

def review(text,seed,watchdog):
    need(not re.search(r'(?im)^\s*(?:fatal:|error:|fail\b)',text),'Simulation diagnostic')
    lines=[s for s in text.splitlines() if s.startswith('NTRPU_KV_')]
    need(lines[:2]==[f'NTRPU_KV_SEED seed={seed}',f'NTRPU_KV_CONFIG watchdog={watchdog}'],'Seed/config')
    cursor=2;results=[];totals=dict(outputs=0,bursts=0,resets=0);published=0
    for d in specifications():
        begin=fields(lines[cursor],'NTRPU_KV_BEGIN',list(d));cursor+=1;need(begin==d,'Stimulus coverage/identity')
        error=admission(d);s=d['scenario'];expected_plan=plan(d) if error==0 else []
        ars=[];responses={};output_count=0;active=None;received=0;cancel=None;faults=[]
        while True:
            need(cursor<len(lines),'Missing completion');line=lines[cursor];cursor+=1
            if line.startswith('NTRPU_KV_AR '):
                ar=fields(line,'NTRPU_KV_AR',['id','plane','index','addr','beats'])
                need(active is None and len(ars)<len(expected_plan) and ar==expected_plan[len(ars)],'AXI request/range/order')
                if cancel:need(len(ars)<cancel['bursts']+cancel['pending_ar'],'New request after cancellation')
                ars.append(ar);active=dict(ar,received=0)
            elif line.startswith('NTRPU_KV_R '):
                r=fields(line,'NTRPU_KV_R',['id','plane','index','addr','resp','rid','last','tail'])
                need(active is not None,'Unowned R response');i=active['received'];index=active['index']+i
                addr=active['addr']+i*64;tail=memory(d,addr)>>(15*32);resp=rid=0;last=int(i==active['beats']-1)
                if s in (24,25) and active['plane']==s-24 and index==d['n']*4-1:tail=0x7fc12345;faults.append('nonfinite')
                if s==26 and active['plane']==0 and index==d['n']*4-1:resp=2;faults.append('rresp')
                if s==27 and active['plane']==1 and index==d['n']*4-1:rid=1;faults.append('rid')
                if s==28 and len(ars)==1:last=int(i==2)
                if s==29 and len(ars)==1:last=int(i==active['beats'])
                expected=dict(id=d['id'],plane=active['plane'],index=index,addr=addr,resp=resp,rid=rid,last=last,tail=tail)
                need(r==expected,'Actual AXI response/fault evidence');key=(r['plane'],r['index']);need(key not in responses,'Duplicate beat')
                responses[key]=r;active['received']+=1;received+=1
                if last:
                    if s not in (28,29):need(active['received']==active['beats'],'Burst length')
                    else:need(len(ars)==1 and active['received']==(3 if s==28 else active['beats']+1),'Malformed RLAST coverage');faults.append('rlast')
                    active=None
            elif line.startswith('NTRPU_KV_DATA '):
                out=fields(line,'NTRPU_KV_DATA',['id','plane','index','last','data'])
                plane,index=divmod(output_count,d['n']*4);key=(plane,index);need(key in responses,'Output without owned response')
                response=responses[key];need(response['resp']==response['rid']==0 and response['tail']&0x7f800000!=0x7f800000,'Bad response published to private cache')
                want=dict(id=d['id'],plane=plane,index=index,last=int(output_count+1==d['n']*8),data=memory(d,address(d,plane,index)))
                need(out==want,'Physical KV bytes/order');output_count+=1
            elif line.startswith('NTRPU_KV_CANCEL '):
                need(cancel is None and s in (30,31,32,33,37),'Unexpected cancellation')
                cancel=fields(line,'NTRPU_KV_CANCEL',['id','cycle','bursts','pending_ar'])
                need(cancel['id']==d['id'] and cancel['bursts']==len(ars) and cancel['pending_ar'] in (0,1),'Cancel frontier')
            elif line.startswith('NTRPU_KV_RESET '):
                end=fields(line,'NTRPU_KV_RESET',['id','scenario','outputs','bursts','returned'])
                need(38<=s<=42 and end==dict(id=d['id'],scenario=s,outputs=output_count,bursts=len(ars),returned=received),'Reset evidence')
                totals['resets']+=1;result=dict(d,reset=True,outputs=output_count,bursts=len(ars),returned=received);break
            elif line.startswith('NTRPU_KV_CASE '):
                end=fields(line,'NTRPU_KV_CASE',['id','scenario','status','cycles','outputs','bursts','returned'])
                want_status=error or (4 if s in (24,25) else 3 if 26<=s<=29 else 7 if s in (30,31,32,33,37) else 8 if 34<=s<=36 else 0)
                need(not 38<=s<=42 and active is None,'Terminal/reset ownership')
                need(end['id']==d['id'] and end['scenario']==s and end['status']==want_status,'Status identity')
                need(end['outputs']==output_count and end['bursts']==len(ars) and end['returned']==received,'Counter reconciliation')
                need(0<end['cycles']<2*watchdog+10000,'Cycle bounds')
                if want_status==0:
                    need(ars==expected_plan and output_count==received==d['n']*8,'Full successful read');published+=output_count
                elif error or s==30:need(not ars and not responses and output_count==0,'Rejected descriptor touched memory')
                if s in (24,25,26,27,28,29):need(len(faults)==1,'Missing injected error')
                if s in (30,31,32,33,37):need(cancel is not None,'Missing cancellation')
                if 34<=s<=36:need(len(ars)==1 and end['cycles']>=watchdog,'Watchdog drain/new request')
                result=dict(d,reset=False,**{k:end[k] for k in ('status','cycles','outputs','bursts','returned')});break
            else:raise ValueError('Unknown KV marker')
        totals['outputs']+=output_count;totals['bursts']+=len(ars);results.append(result)
    need(cursor==len(lines)-1,'Trailing/missing cases');summary=fields(lines[cursor],'NTRPU_KV_PASS',['cases','resets','outputs','bursts','ar_stalls','output_stalls','done_stalls'])
    need(summary['cases']==85 and summary['resets']==5 and all(summary[k]==totals[k] for k in totals),'Summary reconciliation')
    need(all(summary[k]>0 for k in ('ar_stalls','output_stalls','done_stalls')),'Backpressure coverage')
    return dict(status='PASS_KV_TILE_LOADER_R1_TRANSCRIPT',seed=seed,watchdog=watchdog,cases=results,summary=summary,
        publishable_beats=published,publishable_bytes=published*64,private_failed_or_reset_beats=totals['outputs']-published,
        scope='AXI simulation against generated memory bytes; cache lease/committed-prefix authority is a trusted input. No actual DDR, KV append publisher or attention integration.')

def controls(text,seed,watchdog):
    first=re.search(r'(?m)^NTRPU_KV_DATA .*$',text)[0];ar=re.search(r'(?m)^NTRPU_KV_AR .*$',text)[0]
    bad={'missing_data':text.replace(first+'\n','',1),'duplicate_data':text.replace(first,first+'\n'+first,1),
        'missing_request':text.replace(ar+'\n','',1),'wrong_status':text.replace('scenario=24 status=4','scenario=24 status=0',1),
        'missing_summary':text.replace('NTRPU_KV_PASS','REMOVED'),'diagnostic':text+'\nFATAL: injected\n',
        'missing_reset':re.sub(r'(?m)^NTRPU_KV_RESET .*\n','',text,count=1),
        'wrong_epoch':text.replace('expected_epoch=fedcba9800000001','expected_epoch=fedcba9800000000',1)}
    bad['data_bit']=text.replace(first,re.sub(r'data=([0-9a-f]{128})',lambda m:f'data={int(m[1],16)^1:0128x}',first),1)
    bad['address_high_bit']=text.replace(ar,re.sub(r'addr=([0-9a-f]{16})',lambda m:f'addr={int(m[1],16)^(1<<40):016x}',ar),1)
    bad['erase_rresp']=text.replace('resp=2 rid=0','resp=0 rid=0',1)
    bad['erase_nonfinite']=text.replace('tail=7fc12345','tail=3f000000',1)
    rejected=[]
    for name,mutation in bad.items():
        need(mutation!=text,'Mutation unchanged')
        try:review(mutation,seed,watchdog)
        except (ValueError,IndexError,ZeroDivisionError):rejected.append(name)
        else:raise AssertionError('Accepted '+name)
    return rejected

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('folder',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--controls',action='store_true');a=p.parse_args()
    suites=[]
    for name,seed,watchdog in [('seed-1.log',1,32),('seed-7.log',7,32),('seed-42.log',42,32),('seed-1-default.log',1,65536)]:
        path=a.folder/name;text=path.read_text();result=review(text,seed,watchdog);result['log_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
        if a.controls:result['rejected_controls']=controls(text,seed,watchdog)
        suites.append(result)
    result=dict(status='PASS_KV_TILE_LOADER_R1_FOUR_SUITES',cases=340,resets=20,suites=suites,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(status=result['status'],cases=340,resets=20,suites=[{k:v for k,v in s.items() if k!='cases'} for s in suites])))
if __name__=='__main__':main()
