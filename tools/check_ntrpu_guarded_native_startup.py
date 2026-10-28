"""Match management observations to native VCD pins and enforce the cold gate."""
from check_ntrpu_pcie_fragment_pool import fields,require

ROOT='/board/ntrpu_bypass/'
EXTRA={'/board/sys_rst_n':1,'/board/EP/user_lnk_up':1,
       **{ROOT+k:1 for k in 'mgmt_clk guard_arm guard_peers_held guard_peers_ready native_requests_idle host_reset_req host_admit source_admit loss_mgmt guard_invalidate owner_invalid_ack'.split()},
       ROOT+'guard_state':4}
PINS=dict(cold=ROOT+'lifetime_reset_n',link='/board/EP/user_lnk_up',epreset='/board/EP/user_resetn',perst='/board/sys_rst_n',
          loss=ROOT+'loss_mgmt',invalid=ROOT+'guard_invalidate',ack=ROOT+'owner_invalid_ack',req=ROOT+'host_reset_req',
          hrst=ROOT+'host_rst_n',srst=ROOT+'transport_rst_n',ha=ROOT+'host_admit',sa=ROOT+'source_admit',state=ROOT+'guard_state',
          held=ROOT+'guard_peers_held',ready=ROOT+'guard_peers_ready',idle=ROOT+'native_requests_idle',arm=ROOT+'guard_arm')


def sample_management(vcd):
    header,body=vcd.split('$enddefinitions $end',1)
    scope=[];names={};widths={}
    for line in header.splitlines():
        f=line.split()
        if not f:continue
        if f[0]=='$scope':scope.append(f[2])
        elif f[0]=='$upscope':scope.pop()
        elif f[0]=='$var':
            name='/'+'/'.join(scope+[f[4]])
            require(name not in names,'unique management VCD pin')
            names[name]=f[3];widths[f[3]]=int(f[2])
    require(set(PINS.values())|set(EXTRA)<=set(names),'all guard/native pins observed')
    for n,w in EXTRA.items():require(widths[names[n]]==w,'guard pin width')
    values={};updates={};now=0;samples=[]
    def raw(n):return values.get(names[n],'x'*widths[names[n]])
    def flush():
        if raw(ROOT+'mgmt_clk')=='0' and updates.get(names[ROOT+'mgmt_clk'])=='1':
            samples.append(dict(ps=now,**{k:raw(n) for k,n in PINS.items()}))
        values.update(updates)
    for line in body.splitlines():
        line=line.strip()
        if not line or line[0]=='$':continue
        if line[0]=='#':
            flush();updates={};n=int(line[1:]);require(n>=now,'management VCD chronology');now=n
        else:
            value,code=line[1:].split() if line.startswith('b') else (line[0],line[1:])
            value=value.lower()
            require(code in widths and set(value)<=set('01xz') and len(value)<=widths[code],'management VCD encoding')
            updates[code]=value.rjust(widths[code],value[0] if value[0] in 'xz' else '0')
    flush()
    return samples


def review_startup(sim,vcd,host,source,finish):
    rows=[fields(x[len('NTRD_GUARD_NATIVE '):]) for x in sim.splitlines() if x.startswith('NTRD_GUARD_NATIVE ')]
    require(rows and all(set(x)=={'ps',*(set(PINS)-{'arm'})} for x in rows),'guard trace schema')
    expected=[x for x in sample_management(vcd) if x['ps']<finish]
    require([int(x['ps']) for x in rows]==[x['ps'] for x in expected]==list(range(5000,finish,10000)),'every management edge witnessed')
    for r,z in zip(rows,expected):
        for k in PINS:
            if k=='arm':continue
            if k=='state':require(int(r[k],16)==int(z[k],2),'actual guard state')
            else:require(r[k]==z[k],'actual guard/native pin '+k)
    for r in expected:
        require(r['loss']==r['invalid']==r['ack']=='0','cold session never loses ownership')
        state=int(r['state'],2)
        require(state in (0,1,2,3),'cold states only')
        if state in (0,1):require(r['req']=='1' and r['hrst']==r['srst']==r['ha']==r['sa']=='0','held transport and closed admission')
        if state==2:require(r['req']=='0' and r['ha']==r['sa']=='0','release before admission')
        if state==3:require(r['cold']==r['link']==r['epreset']==r['perst']==r['hrst']==r['srst']==r['ready']==r['arm']=='1' and r['req']=='0','native healthy RUN prerequisites')
        if r['cold']=='1' and r['hrst']=='0':require(r['idle']=='1','no native request while cold host transport held')
    states=[]
    for r in expected:
        s=int(r['state'],2)
        if not states or states[-1]!=s:states.append(s)
    require(states==[0,1,2,3],'one clear/hold/release/run progression')
    release=next(i for i,r in enumerate(expected) if int(r['state'],2)==2)
    run=next(i for i,r in enumerate(expected) if int(r['state'],2)==3)
    require(all(expected[release-1][k]=='1' for k in ('held','idle','arm','link','epreset','perst','cold')),'actual peers held before release')
    require(all(expected[run-1][k]=='1' for k in ('ready','hrst','srst','link','epreset','perst')),'actual peers ready before RUN')
    require(expected[-1]['ha']==expected[-1]['sa']=='1','both admissions open at terminal')
    for key in ('arm','ha','sa'):
        seq=[r[key] for r in expected];require(set(seq)=={'0','1'} and seq==sorted(seq),'one monotonic '+key)
    lifetime=next(r['ps'] for r in source if r['lifetime']=='1')
    held_host=sum(r['ps']>=lifetime and r['reset']=='0' for r in host)
    held_source=sum(r['ps']>=lifetime and r['transport']=='0' for r in source)
    require(min(held_host,held_source)>=64,'64 actual local hold edges in both domains')
    terminal=[fields(x[len('NTRD_GUARDED_NATIVE_PASS '):]) for x in sim.splitlines() if x.startswith('NTRD_GUARDED_NATIVE_PASS ')]
    require(len(terminal)==1 and set(terminal[0])=={'ps','accepted','delivered','abandoned'},'one guarded terminal')
    end=terminal[0];require(int(end['ps'])==finish and int(end['abandoned'],16)==0,'no terminal retirement or divergent finish')
    counts=[35,35,35,294,294,294,9,9]
    for key in ('accepted','delivered'):
        value=int(end[key],16);require(value>>512==0 and [(value>>(64*i))&((1<<64)-1) for i in range(8)]==counts,'all eight retained channel counters '+key)
    return dict(status='PASS_NATIVE_COLD_GUARD_STARTUP',management_samples=len(rows),states=states,
        release_ps=expected[release]['ps'],run_ps=expected[run]['ps'],held_host_edges=held_host,held_source_edges=held_source,
        retained_channel_counts=counts,scope='Actual native link/reset pins and cold idle/ready observations. No runtime endpoint reset acknowledgement or link recovery qualification.')
