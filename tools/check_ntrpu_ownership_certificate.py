"""Independent public-wave audit: ownership invalidation is not transport drain."""
from collections import Counter, deque
import re
from check_ntrpu_drain_sections_r2 import vcd_rows
from check_ntrpu_xdma_stop_inventory_r4 import markers, require

EPOCH = 0x100000001
OWNER_TAG = 0x800000010000abcd
KEYS = '''clk mgmt_clk rst_n source_enable epoch bridge_epoch lease_held bridge_lease_held
session_lost session_invalid arm_valid arm_ready arm_epoch arm_tag arm_destination arm_channel
retire_ready line_valid line_ready line_epoch line_tag line_address line_data
cmd_valid cmd_ready cmd_addr cmd_tag cmd_channel cmd_len data_valid data_ready data
done_valid done_ready done_epoch done_tag done_status result_valid result_ready
result_epoch result_tag result_status ack_valid ack_ready cpl_valid cpl_ready cpl_tag
next_cookie proof_request generation proof_tag proof_ack proof_generation proof_ack_tag
peer_request peer_answer peer_generation peer_tag busy obsolete fault_mgmt revoked_source
epoch_probe observed_epoch finished s_axi_awvalid s_axi_awready s_axi_awaddr s_axi_awid
s_axi_awlen s_axi_awsize s_axi_awburst s_axi_awlock s_axi_wvalid s_axi_wready s_axi_wdata
s_axi_wstrb s_axi_wlast s_axi_bvalid s_axi_bready s_axi_bid s_axi_bresp'''.split()
MARK_FIELDS = dict(epoch=('epoch',16),be=('bridge_epoch',16),gen=('generation',16),
    tag=('proof_tag',16),lost=('session_lost',2),invalid=('session_invalid',2),
    held=('lease_held',2),bh=('bridge_lease_held',2),cmd=('cmd_valid',2),
    data=('data_valid',2),result=('result_valid',2),ack=('proof_ack',2),
    busy=('busy',2),obsolete=('obsolete',10),fault=('fault_mgmt',2),
    revoked=('revoked_source',2),enable=('source_enable',2))
COMMON = ['LIVE_OWNER_BLOCKED','STOPPED_BEFORE_LOSS_EDGE','REPLACED_WHILE_STOPPED',
    'REGISTERED_LOSS_SOURCE_STOPPED','FIRST_CERTIFICATE','RETIRE_AND_REARM_BLOCKED',
    'RENEWAL_SOURCE_STOPPED','RENEWED_CERTIFICATE','PENDING_TRAFFIC_PRESERVED']


def review(sim, wave, top='tb_ntrpu_ownership_certificate'):
    require(top in {'tb_ntrpu_ownership_certificate','tb_ntrpu_ownership_certificate_r2'},'known bank top')
    require(not re.search(r'^\s*(Fatal:|Failure:|ERROR:|CRITICAL WARNING:)',sim,re.M),'native simulation failure')
    require(markers(sim,'OWNERSHIP_CERTIFICATE_SUITE_PASS') == [dict(cases='12',real_owner='1',native_xpm='1',reset_driven='0')], 'suite scope')
    require('$finish called at time' in sim,'native finish')
    passed=markers(sim,'OC_CASE_PASS'); events=markers(sim,'OC_EVENT')
    require(len(passed)==12 and {int(r['id']) for r in passed}==set(range(12)),'all twelve cases')
    cases=[]
    for i in range(12):
        ev=[r for r in events if int(r['id'])==i]
        names=COMMON+(['PUBLIC_EPOCH_CONTRADICTION','REVOCATION_RETAINED'] if i>=6 else ['WRONG_GENERATION_BLOCKED'])+['TERMINAL']
        require([r['name'] for r in ev]==names,'scenario coverage and order')
        require([int(r['ps']) for r in ev]==sorted({int(r['ps']) for r in ev}),'strict event times')
        period=5000 if (i//3)%2==0 else 37000
        end=next(r for r in passed if int(r['id'])==i)
        for r in ev+[end]:
            require((int(r['phase']),int(r['period']),int(r['tail']))==(i%3,period,i//6),'case parameters')
        require(int(end['checks'])>30 and int(end['positive'])>10,'positive exercised checks')
        payload=sum((((i+1)*44+k*19)&255)<<(8*k) for k in range(64))
        destination=0x1fffffffc0 if (i+1)%2 else 0x100000040
        cases.append(dict(id=i,prefix=f'/{top}/scenarios[{i}].test_case/',
            events=ev,query=0,counts=Counter(),last_edge=None,last_clk=None,period=period,
            payload=payload,destination=destination,aw=deque(),w=deque(),active=False,
            mgmt=0,source=0,positive=0,seen_low=False,identity=None,lost_at=None,
            fault_at=None,revoked_at=None,ack_rises=[],peer_good=0,quiet_proofs=[]))
    final=0
    for t,before,after in vcd_rows(wave):
        final=t
        for c in cases:
            p=c['prefix'];i=c['id']
            require(all(p+k in after for k in KEYS),'all public owner and provider ports captured')
            s={k:after[p+k] for k in KEYS};old={k:before[p+k] for k in KEYS}
            while c['query']<len(c['events']) and int(c['events'][c['query']]['ps'])<=t:
                row=c['events'][c['query']];qt=int(row['ps'])
                states=[old,s] if qt==t else [old]
                require(any(all(state[key]==int(row[field],base) for field,(key,base) in MARK_FIELDS.items()) for state in states),'event and waveform disagree')
                c['query']+=1
            if s['rst_n']!=1:
                require(not c['active'],'operational reset of retained ownership');continue
            c['active']=True
            require(all(s[k] is not None for k in KEYS),'known public values after configuration')
            require(s['epoch_probe']==0 or i>=6,'only labeled adversarial tail may change observation')
            require(s['observed_epoch']==(s['epoch']^(0x100000000 if s['epoch_probe'] else 0)),'exact public epoch probe')
            if s['session_lost']:
                if c['lost_at'] is None:c['lost_at']=t
                require(s['session_invalid'] and s['arm_ready']==s['retire_ready']==s['line_ready']==s['cpl_ready']==0,'loss blocks owner reuse and completion acceptance')
                require(s['epoch']==s['bridge_epoch']==EPOCH and s['lease_held']==s['bridge_lease_held']==1,'ownership ledger retained')
            else:require(c['lost_at'] is None,'registered session loss sticky')
            if s['revoked_source'] and c['revoked_at'] is None:c['revoked_at']=t
            if s['fault_mgmt'] and c['fault_at'] is None:c['fault_at']=t
            if c['fault_at'] is not None:require(s['fault_mgmt']==1 and s['proof_ack']==0,'permanent management revocation')
            ident=(s['generation'],s['proof_tag'])
            if ident!=c['identity']:c['seen_low']=False;c['identity']=ident
            if old['mgmt_clk']==0 and s['mgmt_clk']==1 and old['rst_n']==1:
                c['mgmt']+=1
                if old['proof_request'] and not old['proof_ack']:c['seen_low']=True
                if old['proof_ack']:c['positive']+=1
            if s['proof_ack']:
                require(s['proof_request']==1 and s['fault_mgmt']==0 and s['session_lost']==1,'live proof prerequisite')
                require(s['proof_generation']==s['generation']==EPOCH and s['proof_ack_tag']==s['proof_tag'],'proof bound to actual owner epoch and fresh tag')
                require(c['seen_low'],'management fresh ACK-low challenge')
            if old['proof_ack']==0 and s['proof_ack']==1:c['ack_rises'].append((t,s['proof_tag']))
            if old['clk']==0 and s['clk']==1 and old['rst_n']==1:
                require(old['source_enable']==1,'no source edge while stopped')
                c['source']+=1
                if c['last_clk'] is not None:require(t-c['last_clk']>=c['period'],'source clock too fast')
                c['last_clk']=t
                good=old['peer_request'] and old['session_lost'] and old['session_invalid'] and old['peer_generation']==old['epoch'] and old['peer_tag']!=0 and not old['epoch_probe']
                c['peer_good']=c['peer_good']+1 if good else 0
                if old['peer_answer']==0 and s['peer_answer']==1:
                    # Tag 1 was captured before loss; later requests also need
                    # their capture edge before the three quiet observations.
                    require(c['peer_good']>=(4 if old['peer_tag']==1 else 5),'registered proof observation interval')
                    c['quiet_proofs'].append(c['peer_good'])
                previous=c['last_edge']
                for valid,ready,fields in [
                    ('cmd_valid','cmd_ready',['cmd_addr','cmd_tag','cmd_channel','cmd_len']),
                    ('data_valid','data_ready',['data']),
                    ('line_valid','line_ready',['line_epoch','line_tag','line_address','line_data']),
                    ('result_valid','result_ready',['result_epoch','result_tag','result_status']),
                    ('s_axi_bvalid','s_axi_bready',['s_axi_bid','s_axi_bresp'])]:
                    if previous and previous[valid] and not previous[ready]:
                        require(old[valid] and all(old[k]==previous[k] for k in fields),'held VALID or payload changed: '+valid)
                count=c['counts']
                for name,a,b in [('arm','arm_valid','arm_ready'),('line','line_valid','line_ready'),
                    ('cmd','cmd_valid','cmd_ready'),('data','data_valid','data_ready'),
                    ('done','done_valid','done_ready'),('retire','ack_valid','ack_ready')]:
                    if old[a] and old[b]:count[name]+=1
                count['completion']+=(old['cpl_valid']&old['cpl_ready']).bit_count()
                if old['arm_valid'] and old['arm_ready']:
                    require(old['arm_epoch']==EPOCH and old['arm_tag']==OWNER_TAG and old['arm_destination']==c['destination'],'real joint ARM identity and high address')
                if old['s_axi_awvalid'] and old['s_axi_awready']:
                    require([old['s_axi_'+k] for k in ['awaddr','awid','awlen','awsize','awburst','awlock']]==[0x1000fffc0,(i+1)%16,0,6,1,0],'ingress AW attributes')
                    c['aw'].append(t);count['aw']+=1
                if old['s_axi_wvalid'] and old['s_axi_wready']:
                    half=count['w'];require(half<2,'two NTRPUF01 frame halves')
                    expected=0x313046555052544e|(EPOCH<<64)|(OWNER_TAG<<128)|((half*32)<<192)|(((c['payload']>>(half*256))&((1<<256)-1))<<256)
                    require(old['s_axi_wdata']==expected and old['s_axi_wstrb']==(1<<64)-1 and old['s_axi_wlast']==1,'all frame bytes/header/strobes')
                    c['w'].append(t);count['w']+=1
                if old['s_axi_bvalid']:
                    require(c['aw'] and c['w'],'B without both accepted AW/W')
                    require(old['s_axi_bid']==(i+1)%16 and old['s_axi_bresp']==0,'B identity and status')
                    if old['s_axi_bready']:c['aw'].popleft();c['w'].popleft();count['b']+=1
                if old['line_valid']:
                    require((old['line_epoch'],old['line_tag'],old['line_address'],old['line_data'])==(EPOCH,OWNER_TAG,c['destination'],c['payload']),'collector output payload')
                if old['cmd_valid']:
                    require((old['cmd_addr'],old['cmd_tag'],old['cmd_channel'],old['cmd_len'])==(c['destination'],0xffffffff,(i+1)%2,0),'command address/cookie/channel/length')
                if old['data_valid']:
                    require(count['cmd']==1 and old['data']==c['payload'],'real data requires prior command and correct full payload')
                if old['done_valid']:
                    require((old['done_epoch'],old['done_tag'],old['done_status'])==(EPOCH,OWNER_TAG,0) and i%3==2,'only pre-loss result is published')
                if old['result_valid']:
                    require((old['result_epoch'],old['result_tag'],old['result_status'])==(EPOCH,OWNER_TAG,0) and i%3==2,'held result immutable')
                c['last_edge']=old
    metrics=[]
    for c in cases:
        i=c['id'];ev={r['name']:r for r in c['events']}
        require(c['query']==len(c['events']),'all log observations covered by waveform')
        require(c['counts']==Counter(arm=1,line=1,cmd=1,data=1,done=int(i%3==2),retire=0,completion=int(i%3==2),aw=2,w=2,b=2),'independent exact transfer scoreboard')
        require(not c['aw'] and not c['w'],'no forgotten ingress frame')
        require([tag for t,tag in c['ack_rises']]==[2,3],'obsolete tag never publishes; renewal fresh')
        require(len(c['quiet_proofs'])==3,'old, current and renewed peer proof observations')
        require(ev['FIRST_CERTIFICATE']['obsolete']=='1' and ev['RENEWED_CERTIFICATE']['obsolete']=='1','one obsolete transfer')
        for name in ['STOPPED_BEFORE_LOSS_EDGE','REGISTERED_LOSS_SOURCE_STOPPED','RENEWAL_SOURCE_STOPPED']:
            require(ev[name]['enable']=='0' and ev[name]['ack']=='0','stopped source cannot manufacture new proof')
        require(ev['STOPPED_BEFORE_LOSS_EDGE']['lost']=='0' and ev['REGISTERED_LOSS_SOURCE_STOPPED']['lost']=='1','registered ownership loss needs edge')
        require(ev['PENDING_TRAFFIC_PRESERVED']['ack']=='1','ownership certificate survives irreversible pending data handshakes')
        if i>=6:
            latency=c['fault_at']-c['revoked_at']
            require(20000<=latency<=30000,'three-stage native management fault crossing')
            require(ev['PUBLIC_EPOCH_CONTRADICTION']['enable']=='0' and ev['REVOCATION_RETAINED']['fault']=='1','source-stop revocation and retained fault')
        else:
            latency=None
            require(c['fault_at'] is None and ev['WRONG_GENERATION_BLOCKED']['busy']=='1' and ev['WRONG_GENERATION_BLOCKED']['ack']=='0','mismatched epoch remains unqualified mailbox obligation')
        metrics.append(dict(id=i,phase=i%3,source_period_ps=c['period'],source_edges=c['source'],
            management_edges=c['mgmt'],positive_management_edges=c['positive'],
            loss_to_first_certificate_ps=c['ack_rises'][0][0]-c['lost_at'],
            revocation_to_management_ps=latency,transfers=dict(c['counts'])))
    require(0<final<100000000,'bounded complete waveform')
    return dict(status='PASS_REAL_OWNER_CERTIFICATE_PUBLIC_WAVES',cases=12,events=len(events),
        metrics=metrics,final_time_ps=final,real_owner_jobs=12,ingress_frame_bytes=1536,
        destination_data_bytes=768,retirements_after_loss=0,real_xpm=True,
        adversarial_port_probe_cases=6,qualified_complete_drain=False,
        qualified_native_endpoint=False,qualified_reset=False,qualified_physical_board=False)
