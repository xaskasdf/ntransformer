"""Check result-writer traffic, errors and reset coverage independently of Tcl."""
import re

CASES=[(1,0,0,0),(7,4032,0,0),(8,0,0,0),(9,4032,0,0),(17,4032,0,0),(31,0,0,0),(32,4032,0,0),
 (0,0,1,1),(33,0,1,1),(2,0,2,1),(9,4032,3,1),(2,0,4,2),(2,0,5,1),(2,0,6,1),
 (32,4032,7,3),(32,4032,8,3),(32,4032,9,3),(32,4032,10,3),
 (32,4032,11,8),(32,4032,12,8),(32,4032,13,8),
 *[(9,0,m,5) for m in range(14,21)],(9,0,21,6),(9,0,22,5),(9,0,23,7),(9,0,24,7),
 (32,0,25,0),(32,0,26,0),(32,0,27,0),(32,0,28,0),(32,0,29,0),
 (9,4032,31,0),(32,4032,32,0),(9,4032,33,0),(9,0,34,5),(17,4032,0,0)]


def review(text,seed,netlist=False):
    errors=[]
    def need(ok,why):
        if not ok:errors.append(why)
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)',text,re.M|re.I),'Executed failure')
    need(re.findall(r'^NTRPU_WRITER_SEED seed=(\d+)\s*$',text,re.M)==[str(seed)],'Seed')
    if netlist:need(re.findall(r'^NTRPU_WRITER_CONFIG watchdog=(\d+)\s*$',text,re.M)==['65536'],'Mapped watchdog stimulus')
    rows=[{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',l)} for l in text.splitlines() if l.startswith('NTRPU_WRITER_CASE ')]
    need([r.get('id') for r in rows]==list(range(1,43)),'Case identities')
    for r,want in zip(rows,CASES):
        n,offset,mode,status=want;reset=mode in {28,29}
        need(set(r)=={'id','rows','offset','scenario','status','reset','cycles','inputs','aw','w','b','bytes'},'Fields')
        need(tuple(r.get(k) for k in ['rows','offset','scenario','status'])==want,'Case contract '+str(r.get('id')))
        need(r.get('reset')==int(reset),'Reset contract')
        need(0<r.get('cycles',0)<(80000 if netlist else 2000),'Latency')
        inputs=0 if status in {1,2} or mode==24 else n-1 if mode==19 else n+1 if mode==20 else n
        need(r.get('inputs')==inputs,'Input drain')
        if status in {1,2,5,6,7}: need(all(r.get(k)==0 for k in ['aw','w','b','bytes']),'No writes permitted')
        elif reset:
            need(r.get('w')==(1 if mode==28 else 4) and r.get('bytes')==(64 if mode==28 else 256),'Reset traffic')
            need(0<=r.get('aw',-1)<=1 and r.get('b')==0,'Reset response')
        else:
            beats=(n+7)//8;bursts=1+int(offset+beats*64>4096);size=n*8
            if mode in {7,9,11,12,13}:beats=1;bursts=1;size=64
            need((r.get('aw'),r.get('w'),r.get('b'),r.get('bytes'))==(bursts,beats,bursts,size),'AXI counts')
            if status==8:need(r.get('cycles',0)>=(65584 if netlist else 80),'Timeout stimulus')
    markers=re.findall(r'^NTRPU_WRITER_PASS (.+)$',text,re.M);need(len(markers)==1,'Suite marker')
    summary={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',markers[0])} if markers else {}
    need(summary.get('cases')==42 and summary.get('resets')==2,'Suite cases/reset')
    for k in ['aw','w','bytes']:need(summary.get(k)==sum(r.get(k,0) for r in rows),'Total '+k)
    for k in ['aw_stalls','w_stalls','done_stalls','aw_before_w','w_before_aw']:need(summary.get(k,0)>0,'Coverage '+k)
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=rows,summary=summary)
