"""Native reset variant of the independently qualified ownership/DDR ledger."""
import re
from collections import Counter
from check_ntrpu_pcie_fragment_pool import fields,require


def review(text):
    """Check actual ownership, DDR bytes and reset order across the last loss."""
    require(not re.search(r'(?mi)^\s*(?:ERROR:|FATAL:|FAIL )|\bModel:ERROR:',text),'simulator failure')
    def rows(prefix):return [fields(x[len(prefix):]) for x in text.splitlines() if x.startswith(prefix)]
    endings=rows('NTRD_NATIVE_RESET_RETENTION_PASS '); markers=rows('NTRD_NATIVE_RESET_HEALTHY_END ')
    require(len(endings)==len(markers)==1,'one complete guarded fixture')
    end=endings[0]; begin=int(markers[0]['ps'])
    events_=rows('NTRD_NATIVE_RESET_EVENT ')
    require([x['name'] for x in events_]==['ready','perst_assert','endpoint_down','release_memory_b','memory_verified','perst_release','link_return'],'loss event order')
    ready,loss,endpoint_down,release,verified,perst_release,link_return=map(lambda x:int(x['ps']),events_)
    require(ready<loss<=endpoint_down<release<verified<=perst_release<link_return<int(end['ps']),'actual native reset stages')
    require(end['coordination']=='blocked','missing coordinator is explicit')
    require(begin<ready<loss<release<int(end['ps']),'ordered loss phases')
    cookie=0x100000010; epoch=0x100000013
    require(int(events_[0]['cookie'],16)==cookie and int(events_[0]['epoch'],16)==epoch,'independent last identity')
    loss_rows=rows('NTRD_NATIVE_RESET_LOSS ')
    require(len(loss_rows)==int(end['loss_samples'])>100,'complete loss sample inventory')
    times=[int(x['ps']) for x in loss_rows]
    require(0<times[0]-ready<=5000 and times==list(range(times[0],times[-1]+1,5000)),'consecutive source edges after loss arm')
    require(0<int(end['ps'])-times[-1]<5000,'loss trace reaches finish')
    for row in loss_rows:
        require(int(row['cookie'],16)==cookie and int(row['epoch'],16)==epoch and row['bh']=='1','ownership preserved every sampled edge')
        require(row['mrst']=='3' and row['lifetime']=='1' and int(row['fault'],16)==0,'memory lifetime and boundary integrity')
        require(row['occupied0']=='1' and row['occupied1']=='0','completion descriptor retained independently of reusable staging bank')
        if row['invalid']=='1':require(row['boundary']=='1','invalidation follows boundary loss')
        if row['ack']=='1':require(row['invalid']==row['boundary']=='1','owner ACK requires latched invalidation')
        if int(row['cv'],16):
            require(int(row['cv'],16)==1 and int(row['cr'],16)==0 and int(row['ct'],16)==cookie-1 and int(row['cs'],16)==0,'late completion held intact')
    invalid_at=int(next(x['ps'] for x in loss_rows if x['invalid']=='1'))
    source_ack_at=int(next(x['ps'] for x in loss_rows if x['ack']=='1'))
    require(source_ack_at==invalid_at+5000,'owner latches invalidation on next actual source edge')
    mgmt=[x for x in rows('NTRD_GUARD_NATIVE ') if int(x['ps'])>=loss]
    first=lambda key,value:int(next(x['ps'] for x in mgmt if x[key]==value))
    ack_at=first('ack','1'); reset_at=first('req','1')
    require(loss<first('loss','1')<=ack_at<reset_at<release,'management loss then real owner ACK then transport reset')
    faulted=[x for x in mgmt if x['state']=='6']
    require(faulted and all(x['ha']==x['sa']=='0' and x['hrst']==x['srst']==x['held']==x['ready']=='0' and x['req']==x['loss']==x['ack']=='1' for x in faulted),'transport remains held awaiting an actual coordinator')
    for key in ['invalid','ack','boundary']:
        bits=[int(x[key],16) for x in loss_rows]
        require(set(bits)=={0,1} and bits==sorted(bits),'sticky '+key)
    for key in ['hrst','srst']:
        bits=[int(x[key],16) for x in loss_rows]
        groups=[v for i,v in enumerate(bits) if i==0 or v!=bits[i-1]]
        require(groups==[1,0],'one guarded runtime reset kept asserted '+key)
    host=[x for x in rows('NTRD_HOST_DATA ') if int(x['ps'])>=loss]
    require(sum(x['reset']=='0' for x in host)>=64,'runtime reset spans64 real host edges')
    require(sum(x['srst']=='0' for x in loss_rows)>=64,'runtime reset spans64 real source edges')
    life=[x for x in rows('NTRD case=0 LIFE_TRACE ') if int(x['ps'])>begin]
    require(life and all(x['cold']==x['poolreset']==x['admit']=='1' and x['memreset']=='3' and x['restart']==x['synced']=='0' for x in life),'no memory pool reset or admission loss')
    cycles={x['cycle']:int(x['ps']) for x in life}
    own=[x for x in rows('NTRD case=0 OWN_TRACE ') if x['cycle'] in cycles]
    counts=Counter()
    pairs={'arms':('av','ar'),'lines':('lv','lr'),'commands':('cv','cr'),'data':('dv','dr'),'done':('ov','ore'),'ack':('kv','kr')}
    witnessed={}
    for row in own:
        for name,(v,r) in pairs.items():
            if row[v]==row[r]=='1':counts[name]+=1;witnessed[name]=cycles[row['cycle']]
        counts['returns']+=bool(int(row['pv'],16)&int(row['pr'],16))
        require(row['rv']=='0','no last-job publication')
        if cycles[row['cycle']]>=ready:require(row['kr']=='0' and row['ov']=='0','late completion cannot become DONE or permit retirement')
    require({k:counts[k] for k in [*pairs,'returns']}==dict(arms=1,lines=1,commands=1,data=1,done=0,ack=0,returns=0),'last job transfer obligations')
    pattern=sum(((18*37+k*19+7*7)&255)<<(k*8) for k in range(64))
    memory={kind:[x for x in rows('MANAGED_MIG_'+kind+' ') if int(x['ps'])>begin] for kind in ['AW','W','B','AR','R']}
    require(all(len(v)==1 for v in memory.values()),'one last DDR write/read transaction')
    aw,w,b,ar,r=[memory[k][0] for k in ['AW','W','B','AR','R']]
    require(all(x['channel']=='0' for x in [aw,w,b,ar,r]),'last transfer to channel0')
    require(all(x['job']=='18' for x in [aw,w,b]) and ar['identity']==r['identity']=='18','last DDR identity')
    require(int(aw['address'],16)==int(ar['address'],16)==0x1fffffffc0,'last line of128GiB')
    require([int(aw[k],16) for k in ['len','id','size','burst','cache']]==[0,0,6,1,0],'last AW attributes')
    require(int(w['data'],16)==int(r['data'],16)==pattern and int(w['strb'],16)==2**64-1 and w['last']==r['last']=='1','all64 bytes survive host reset and DDR readback')
    require(b['status']==r['status']=='0' and b['id']==r['id']=='0','successful DDR responses')
    cpl_at=int(next(x['ps'] for x in loss_rows if x['cv']=='1'))
    require(witnessed['commands']<int(aw['ps'])<ready and witnessed['data']<int(w['ps'])<ready,'pool acceptance before DDR acceptance')
    require(release<int(b['ps'])<cpl_at<int(ar['ps'])<int(r['ps'])<int(end['ps']),'late B then quarantined CPL then readback')
    expected=dict(arms=19,commands=17,data=17,returns=16,done=16,ack=18,aw=17,w=17,b=17,reads=29,frames=37)
    require(all(int(end[k])==v for k,v in expected.items()),'whole bank conservation summary')
    require(int(end['cookie'],16)==cookie and int(end['epoch'],16)==epoch and int(end['banks'],16)==0 and int(end['pending'],16)==0 and end['occupied0']=='1' and end['occupied1']=='0','last reservation remains owned')
    return dict(status='PASS_NATIVE_RESET_RETENTION_PEER_COORDINATION_BLOCKED',counts=expected,loss_ps=loss,owner_ack_ps=ack_at,transport_reset_ps=reset_at,
                memory_b_ps=int(b['ps']),quarantined_cpl_ps=cpl_at,readback_ps=int(r['ps']),loss_source_samples=len(loss_rows),
                physical_throughput_claimed=False)

