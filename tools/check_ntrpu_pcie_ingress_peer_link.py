"""Native PCIe -> actual ingress/controller/receiver -> destination AXI witness.

Preserve the qualified PCIe/ingress checker unchanged by filtering only the
explicit additional signal set. Separately sample destination pins at the
pre-edge values and validate stability, full data, epochs and final-B drain.
"""
import re
from check_ntrpu_pcie_bypass_native import bits, require
from check_ntrpu_pcie_bypass_smoke import fields
from check_ntrpu_pcie_line_link import ROOT, review as ingress_review, sample as ingress_sample

WIDTHS = dict(peer_epoch=64,peer_quarantined=1,peer_axi_busy=1,bridge_phase=3,destination_status=3,
              m_axi_awaddr=64,m_axi_awid=4,m_axi_awlen=8,m_axi_awsize=3,m_axi_awburst=2,
              m_axi_awcache=4,m_axi_awprot=3,m_axi_awlock=1,m_axi_awvalid=1,m_axi_awready=1,
              m_axi_wdata=512,m_axi_wstrb=64,m_axi_wlast=1,m_axi_wvalid=1,m_axi_wready=1,
              m_axi_bid=4,m_axi_bresp=2,m_axi_bvalid=1,m_axi_bready=1)


def sample(text):
    header,body=text.split('$enddefinitions $end',1)
    scope=[];names={};widths={};filtered_header=[]
    for line in header.splitlines():
        f=line.split();keep=True
        if f and f[0]=='$scope':scope.append(f[2])
        elif f and f[0]=='$upscope':scope.pop()
        elif f and f[0]=='$var':
            name='/'+ '/'.join(scope+[f[4]])
            require(name not in names,'duplicate raw signal')
            names[name]=f[3];widths[f[3]]=int(f[2])
            if name.startswith(ROOT) and name[len(ROOT):] in WIDTHS:
                require(int(f[2])==WIDTHS[name[len(ROOT):]],'peer signal width')
                keep=False
        if keep:filtered_header.append(line)
    extra={ROOT+k for k in WIDTHS}
    require(extra.issubset(names),'missing peer signals')
    remaining_codes={code for name,code in names.items() if name not in extra}
    def decode(line):
        if line.startswith('b'):return line[1:].split()
        return line[0],line[1:]
    filtered=[]
    for line in body.splitlines():
        clean=line.strip()
        if not clean or clean[0] in '$#' or decode(clean)[1] in remaining_codes:filtered.append(line)
    clean='\n'.join(filtered_header)+'\n$enddefinitions $end\n'+'\n'.join(filtered)
    _,ingress=ingress_sample(clean)
    require(len(names)==ingress['signals']+len(WIDTHS)==105,'exact peer-link inventory')
    values={};updates={};now=0;events=[];states={};held={};stalls=dict(MAW=0,MW=0,MB=0)
    channels={'MAW':('aw',['addr','id','len','size','burst','cache','prot','lock']),
              'MW':('w',['data','strb','last']), 'MB':('b',['id','resp'])}
    def raw(name):return values.get(names[name],'x'*widths[names[name]])
    def flush():
        clock=names['/board/EP/user_clk']
        if values.get(clock)=='0' and updates.get(clock)=='1' and raw('/board/EP/user_resetn')=='1':
            state={k:bits(raw(ROOT+k)) for k in WIDTHS}
            states[now]=state
            for ch,(prefix,fs) in channels.items():
                payload={f:state['m_axi_'+prefix+f] for f in fs}
                valid,ready=state['m_axi_'+prefix+'valid'],state['m_axi_'+prefix+'ready']
                if ch in held:require(valid==1 and payload==held.pop(ch),'peer '+ch+' unstable')
                if valid:
                    if ready:events.append(dict(channel=ch,ps=now,**payload))
                    else:held[ch]=payload;stalls[ch]+=1
            require(state['peer_epoch']==ingress['states'][now]['epoch'],'desynchronized epoch')
            require(not (ingress['states'][now]['result_valid'] and state['peer_axi_busy']),'result before peer drain')
        values.update(updates)
    for line in body.splitlines():
        line=line.strip()
        if not line or line.startswith('$'):continue
        if line.startswith('#'):
            flush();updates={};new=int(line[1:]);require(new>=now,'peer VCD chronology');now=new
        else:
            raw_value,code=decode(line);raw_value=raw_value.lower()
            require(code in widths and set(raw_value)<=set('01xz') and len(raw_value)<=widths[code],'peer VCD encoding')
            updates[code]=raw_value.rjust(widths[code],raw_value[0] if raw_value[0] in 'xz' else '0')
    flush();require(not held,'undrained destination VALID')
    return clean,ingress,dict(events=events,states=states,stalls=stalls,signals=len(names),end_ps=now)


def review(sim,tx,rx,vcd):
    clean,ingress,peer=sample(vcd)
    result=ingress_review(sim,tx,rx,clean)
    rows={ch:[e for e in peer['events'] if e['channel']==ch] for ch in ['MAW','MW','MB']}
    require(all(len(r)==3 for r in rows.values()),'destination counts')
    aux={ch:[e for e in ingress['events'] if e['channel']==ch] for ch in ['ARM','LINE','DONE','RESULT']}
    logs=[(s.split()[0].removeprefix('NTRPU_PEER_LINK_'),fields(s)) for s in sim.splitlines() if s.startswith('NTRPU_PEER_LINK_')]
    require(len(logs)==9 and set(ch for ch,_ in logs)==set(rows),'destination log inventory')
    measurements=[]
    for index,job in enumerate([0,1,3]):
        aw,w,b=[rows[ch][index] for ch in ['MAW','MW','MB']]
        arm,line,done,retired=aux['ARM'][job],aux['LINE'][index],aux['DONE'][index],aux['RESULT'][job]
        require(aw['addr']==line['address']==0x1fffffffc0 and aw['id']==aw['len']==aw['cache']==aw['prot']==aw['lock']==0 and aw['size']==6 and aw['burst']==1,'destination full address/attributes')
        require(w['data']==line['data'] and w['strb']==(1<<64)-1 and w['last']==1,'destination full line bytes')
        require(b['id']==b['resp']==0,'destination B response')
        require(line['ps']<min(aw['ps'],w['ps']) and max(aw['ps'],w['ps'])<b['ps']<done['ps']<retired['ps'],'whole pipeline order')
        require(peer['states'][b['ps']]['peer_axi_busy']==1 and peer['states'][done['ps']]['peer_axi_busy']==0,'B releases destination busy')
        require(peer['states'][done['ps']]['bridge_phase']==5,'DONE phase')
        for ch,e in [('MAW',aw),('MW',w),('MB',b)]:
            row=[f for k,f in logs if k==ch][index]
            require(set(row)==set(e)-{'channel'},'destination log fields')
            require(all(int(row[k],10 if k in {'ps','len','size','burst','last','resp','lock'} else 16)==v for k,v in e.items() if k!='channel'),'destination log/native mismatch')
        measurements.append(dict(job=job,line_ps=line['ps'],aw_ps=aw['ps'],w_ps=w['ps'],b_ps=b['ps'],done_ps=done['ps'],result_ps=retired['ps'],line_to_b_ps=b['ps']-line['ps'],b_to_done_ps=done['ps']-b['ps']))
    for job in range(4):
        arm,retired=aux['ARM'][job],aux['RESULT'][job]
        a,r=peer['states'][arm['ps']],peer['states'][retired['ps']]
        require(a['peer_epoch']==job and a['peer_quarantined']==1 and a['peer_axi_busy']==a['bridge_phase']==0,'joint rearm preconditions')
        require(r['peer_epoch']==job+1 and r['peer_axi_busy']==r['bridge_phase']==r['destination_status']==0,'retired peer state')
        phases=[]
        for ps,state in peer['states'].items():
            if arm['ps']<=ps<=retired['ps']:
                phase=state['bridge_phase']
                if not phases or phase!=phases[-1]:phases.append(phase)
        require(phases==([0] if job==2 else [0,1,2,3,4,5,0]),'native bridge phase sequence')
    require(peer['stalls']['MAW']>0 and peer['stalls']['MW']>0,'destination backpressure coverage')
    result.update(status='PASS_NATIVE_PCIE_INGRESS_PEER',signals=peer['signals'],destination_stalls=peer['stalls'],destination_measurements=measurements,
                  scope='Actual generated XDMA bypass into unchanged qualified ingress/controller/peer receiver RTL. Three exact 64-byte AXI writes, one source conflict and recovery without reset. Destination AXI response and trusted isolation are BFMs. No MIG/capacity filter/host-visible completion/CDC/P2P/routed/physical/inference qualification. Additional stop/foreign-B/error cases belong to the separate standalone qualification.')
    return result
