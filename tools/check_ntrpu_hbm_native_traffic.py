"""Replay native AXI3 write/read traces independently, including byte strobes."""
import argparse,csv,hashlib,json,re
from collections import Counter,defaultdict
from pathlib import Path
from check_ntrpu_hbm_native_link import binding as write_binding

def need(ok,why):
    if not ok:raise ValueError(why)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def base(out):
    files=json.loads((out/'FILES.json').read_text());r=json.loads((out/'result.json').read_text())
    need(all(sha(out/n)==h for n,h in files.items()),'Evidence hashes')
    need(r['experiment']=='hbm_native_traffic' and r['status']=='EXPORTED_PENDING_REVIEW','Tool failed')
    need(r['sources']==r['sources_after'],'Changed sources')
    return r

def compile_review(out,r):
    suite=out/'suite';manifest=json.loads((suite/'BINDING.json').read_text())
    template=(suite/'native_hbm.veo').read_text();old=(suite/'write_binding/ntrpu_hbm_native_link.sv').read_text()
    initial=write_binding(template,old)
    source=(suite/'ntrpu_hbm_native_rw.sv').read_text()
    connections=dict(re.findall(r'\.([A-Za-z0-9_]+)\(([^)]*)\)',source))
    widths={'araddr':34,'arburst':2,'arid':6,'arlen':4,'arsize':3,'arvalid':1,'arready':1,
            'rdata':256,'rdata_parity':32,'rid':6,'rlast':1,'rresp':2,'rvalid':1,'rready':1}
    expected=dict(initial)
    for lane,index in enumerate((0,16)):
        for name,width in widths.items():
            expected[f'AXI_{index:02}_{name.upper()}']='s_'+name+(f'[{lane*width}+:{width}]' if width>1 else f'[{lane}]')
    need(connections==expected==manifest['connections'],'Native read/write binding')
    need(sha(suite/'ntrpu_hbm_native_rw.sv')==manifest['binding_sha256'],'Binding hash')
    need(sha(suite/'tb_ntrpu_hbm_native_traffic.sv')==manifest['bench_sha256'],'Bench hash')
    config=dict(l.split('\t',1) for l in (suite/'native_hbm_config.tsv').read_text().splitlines())
    required={'USER_HBM_DENSITY':'16GB','USER_HBM_STACK':'2','USER_AXI_CLK_FREQ':'450',
              'USER_AXI_CLK1_FREQ':'450','USER_SWITCH_ENABLE_00':'TRUE','USER_SWITCH_ENABLE_01':'TRUE'}
    need(all(config['CONFIG.'+k]==v for k,v in required.items()),'Native HBM configuration')
    need(all(config[f'CONFIG.USER_MC{i}_USER_PARITY_EN']=='false' for i in range(16)),'Unexpected parity mode')
    need(all(config[f'CONFIG.USER_MC{i}_{key}']=='true' for i in range(16)
             for key in ('DM_EN','EN_DATA_MASK')),'Native byte masking disabled')
    snapshots=json.loads((out/'SNAPSHOT_INPUTS.json').read_text())
    prefix='xsim.dir/tb_ntrpu_hbm_native_traffic_behav/'
    for name in ('xsimk','xsim.mem','xsim.reloc'):
        need(snapshots.get(prefix+name,{}).get('bytes',0)>0,'Incomplete executable snapshot')
    for i in (0,1):
        name=f'xpm_internal_config_file_sim_{i}.mem'
        need(snapshots[name]['sha256']==sha(suite/'model_inputs'/name),'Native model configuration')
    log=(out/'console.log').read_text()
    need('NTRPU_NATIVE_TRAFFIC_COMPILED' in log and not re.search(r'^ERROR:',log,re.M),'Compilation failure')
    need('Built simulation snapshot tb_ntrpu_hbm_native_traffic_behav' in log,'No executable snapshot')
    if (suite/'INCREMENTAL.json').exists():
        cache=json.loads((suite/'INCREMENTAL.json').read_text())
        need(cache['donor_unchanged'] and cache['copied_files']>0,'Incremental donor modified')
        need(cache['native_config_sha256']==sha(suite/'native_hbm_config.tsv'),'Incremental native configuration changed')
    return dict(status='NATIVE_TRAFFIC_SNAPSHOT_REVIEWED',sources=r['sources'],
                snapshot_manifest_sha256=sha(out/'SNAPSHOT_INPUTS.json'),ports=[0,16],
                instantiated_native_controller=True,traffic_validated=False,parity_validated=False,
                native_config_sha256=sha(suite/'native_hbm_config.tsv'),byte_masking_enabled=True,snapshot_files=len(snapshots),
                snapshot_bytes=sum(v['bytes'] for v in snapshots.values()))

def pattern(number,lane,beat):
    return sum(((number*0x9e3779b9)^(lane*0x6a09e667)^(beat*0x10203)^(k*0x01010101))%2**32<<(32*k) for k in range(8))

def replay(text):
    need(not re.search(r'^\s*(Fatal:|ERROR:)',text,re.M),'Simulation failure')
    groups=defaultdict(lambda:defaultdict(list));passes=[];cases=[];release=[];boot=[];end=[];thermal=[];thermal_end=[];count_rows=[]
    for line in text.splitlines():
        f=line.split()
        if f[:1]!=['NT']:continue
        kind=f[1]
        if kind=='RELEASE':release.append(int(f[2]))
        elif kind=='BOOT':boot.append(int(f[2]))
        elif kind=='CASE':cases.append((int(f[2]),int(f[3])))
        elif kind=='CASE_PASS':passes.append(int(f[2]))
        elif kind=='THERMAL':thermal.append(f[2:])
        elif kind=='THERMAL_END':thermal_end.append(dict(x.split('=') for x in f[2:]))
        elif kind=='COUNTS':count_rows.append({k:int(v) for k,v in (x.split('=') for x in f[2:])})
        elif kind=='ALL_PASS':end.append({k:int(v) for k,v in (x.split('=') for x in f[2:])})
        elif kind in ('AW','W','B','AR','R','BS','RS'):
            time,number,lane=map(int,f[2:5]);need(lane in (0,1) and 1<=number<=5,'Bad lane/case')
            groups[(number,lane)][kind].append((time,f[5:]))
    need(len(release)==len(boot)==len(end)==1 and release[0]<boot[0],'Initialization evidence')
    need(len(thermal)==len(thermal_end)==1 and int(thermal[0][0])==boot[0],'Thermal observations missing')
    alarm_start=thermal[0][1:3];alarm_end=[thermal_end[0]['cattrip0'],thermal_end[0]['cattrip1']]
    need(all(v.lower() in ('0','x','z') for v in alarm_start+alarm_end),'Asserted thermal alarm')
    unknown_cycles=int(thermal_end[0]['unknown_cycles']);need(unknown_cycles>=0,'Invalid thermal unknown count')
    if any(v.lower() in ('x','z') for v in alarm_start+alarm_end):need(unknown_cycles>0,'Missing unknown thermal accounting')
    need([v[1] for v in cases]==passes==list(range(1,6)),'Scenario completion')
    need(len(groups)==10 and boot[0]<=cases[0][0],'Missing transactions or premature traffic')
    memory={};enabled=0;total_beats=0;case_reports=[];ordering=Counter();stall_totals=Counter()
    for number in range(1,6):
        offset={1:0x1000,2:0x2000,3:0x3000,4:0x2000,5:0x1fffffe0}[number]
        beats={1:1,2:16,3:8,4:16,5:1}[number]
        for lane in (0,1):
            g=groups[number,lane];address=(lane<<33)+offset
            need({'AW','W','B','AR','R','BS'}<=set(g)<={'AW','W','B','AR','R','BS','RS'},'Missing AXI channel/stall evidence')
            need(all(len(g[k])==1 for k in ('AW','B','AR')),'Duplicate address/response')
            aw_t,aw=g['AW'][0];b_t,b=g['B'][0];ar_t,ar=g['AR'][0]
            need(int(aw[0],16)==int(ar[0],16)==address,'Address/stack mapping')
            need(int(aw[1])==int(ar[1])==beats and address%32==0,'Length/alignment')
            need(int(aw[2])==(number*7+lane*17)%64 and int(ar[2])==(number*11+lane*19)%64,'Request IDs')
            need(b==[aw[2],'0'],'BID/BRESP')
            need(len(g['W'])==len(g['R'])==beats,'Lost/extra data beat')
            need(aw_t>=cases[number-1][0] and min(t for t,_ in g['W'])>=cases[number-1][0],'Traffic before case')
            need(b_t>max(aw_t,max(t for t,_ in g['W'])) and ar_t>b_t,'Write/read causality')
            need(len(g['BS'])>=7,'Fewer than seven sampled B stalls')
            for time,stalled in g['BS']:
                need(max(aw_t,max(t for t,_ in g['W']))<time<b_t and stalled==b,'Stalled B response mismatch')
            need(len({time for time,_ in g['BS']})==len(g['BS']),'Duplicate stalled B sample')
            stall_totals['B']+=len(g['BS']);stall_totals['R']+=len(g.get('RS',[]))
            ordering['w_before_aw' if g['W'][0][0]<aw_t else 'aw_before_or_with_w']+=1
            for beat,(time,w) in enumerate(g['W']):
                value,mask=int(w[0],16),int(w[1],16);last=int(w[2]);word=address+beat*32
                need(value==pattern(number,lane,beat),'Write payload pattern')
                want_mask=(0x55555555 if beat%2 else 0xaaaaaaaa) if number==4 else 0xffffffff
                need(mask==want_mask and last==(beat+1==beats),'Write strobe/WLAST')
                old=memory.get(word,0)
                for byte in range(32):
                    if mask>>byte&1:
                        old=(old&~(255<<(byte*8)))|(((value>>(byte*8))&255)<<(byte*8));enabled+=1
                memory[word]=old
            for beat,(time,response) in enumerate(g['R']):
                value=int(response[0],16);rid,resp,last=map(int,response[1:])
                need(time>ar_t and (beat==0 or time>g['R'][beat-1][0]),'Read time ordering')
                need(rid==int(ar[2]) and resp==0 and last==(beat+1==beats),'RID/RRESP/RLAST')
                need(value==memory[address+beat*32],'Readback memory mismatch')
            for time,stalled in g.get('RS',[]):
                next_read=next((v for t,v in g['R'] if t>time),None)
                need(time>ar_t and stalled==next_read,'Stalled R data/response mismatch')
            need(len({time for time,_ in g.get('RS',[])})==len(g.get('RS',[])),'Duplicate stalled R sample')
            total_beats+=beats
        case_reports.append(dict(number=number,lanes=2,beats_per_lane=beats,partial_write=number==4))
    totals=end[0]
    need(int(thermal_end[0]['time_ps'])==totals['time_ps'],'Thermal end-time mismatch')
    need(all(totals[k]==v for k,v in dict(cases=5,aw=10,w=84,b=10,ar=10,r=84).items()),'Final counts')
    need(totals['b_stalls']>=70 and totals['r_stalls']>0,'Missing response backpressure')
    need(totals['b_stalls']==stall_totals['B'] and totals['r_stalls']==stall_totals['R'],'Observed stall totals')
    need(len(count_rows)==1 and all(totals[k]==v for k,v in count_rows[0].items()),'Independent count row')
    # A slave may legally wait for AW before accepting early-offered W data.
    # Report actual handshake ordering instead of requiring WREADY before AW.
    need(sum(ordering.values())==10,'Write ordering inventory')
    return dict(status='NATIVE_HBM_WRITE_READBACK_PASS',cases=case_reports,write_bursts=10,read_bursts=10,
                write_beats=total_beats,read_beats=total_beats,read_bytes=total_beats*32,
                strobed_write_bytes=enabled,unique_bytes=len(memory)*32,ordering=dict(ordering),
                response_stalls=dict(B=totals['b_stalls'],R=totals['r_stalls']),
                thermal_observation=dict(start=alarm_start,end=alarm_end,unknown_cycles=unknown_cycles,
                                         start_temperatures=thermal[0][3:5],thermal_validated=False),
                boot_time_ps=boot[0],end_time_ps=totals['time_ps'],ports=[0,16])

def negative_controls(text):
    results=[]
    for kind in ('RELEASE','BOOT','THERMAL','THERMAL_END','COUNTS','CASE','CASE_PASS','AW','W','B','BS','AR','R','RS','ALL_PASS'):
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('NT '+kind+' '));del lines[i]
        try:replay('\n'.join(lines))
        except (ValueError,KeyError,IndexError):results.append('remove_'+kind)
        else:raise ValueError('Undetected removal '+kind)
    for kind,column,radix in (('AW',5,16),('W',5,16),('W',6,16),('B',5,10),('BS',5,10),('AR',5,16),('R',5,16),('R',6,10),('R',8,10),('RS',5,16)):
        lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('NT '+kind+' '));f=lines[i].split()
        v=int(f[column],radix)^1;f[column]=format(v,'x') if radix==16 else str(v);lines[i]=' '.join(f)
        try:replay('\n'.join(lines))
        except (ValueError,KeyError,IndexError):results.append('corrupt_'+kind+'_'+str(column))
        else:raise ValueError('Undetected corruption '+kind)
    lines=text.splitlines();i=next(i for i,l in enumerate(lines) if l.startswith('NT THERMAL '));f=lines[i].split();f[3]='1';lines[i]=' '.join(f)
    try:replay('\n'.join(lines))
    except (ValueError,KeyError,IndexError):results.append('assert_thermal_alarm')
    else:raise ValueError('Undetected thermal alarm')
    return results

def traffic_review(out,r):
    need(r['donor_unchanged'] and r['runtime_inputs_unchanged'],'Snapshot inputs changed')
    prior=json.loads((out/'PRIOR_REVIEW.json').read_text())
    need(prior['status']=='NATIVE_TRAFFIC_SNAPSHOT_REVIEWED' and prior['sources']==r['sources'],'Unreviewed snapshot sources')
    need(prior['snapshot_manifest_sha256']==r['manifest_sha256']==sha(out/'SNAPSHOT_INPUTS.json'),'Snapshot review identity')
    text=(out/'evidence/simulate.log').read_text();result=replay(text)
    with (out/'evidence/progress.tsv').open() as f:progress=list(csv.DictReader(f,delimiter='\t'))
    need(progress and all(progress[-1][name]=='1' for name in ('apb_complete_0','apb_complete_1','init_seq_complete_0','init_seq_complete_1','booted')),'Native initialization flags')
    last=progress[-1]
    need([last[f'DRAM_{i}_STAT_CATTRIP'].lower() for i in (0,1)]==[v.lower() for v in result['thermal_observation']['end']],'Thermal probe/trace mismatch')
    need(int(last['thermal_unknown_cycles'],16)==result['thermal_observation']['unknown_cycles'],'Thermal counter mismatch')
    need(all(int(last[k],16)==v for k,v in dict(aw_count=10,w_count=84,b_count=10,ar_count=10,r_count=84).items()),'Native probe transaction counts')
    need(int(last['b_stalls'],16)==result['response_stalls']['B'] and int(last['r_stalls'],16)==result['response_stalls']['R'],'Native probe stall counts')
    result.update(negative_trace_controls=negative_controls(text),progress_samples=len(progress),
                  native_ip_simulation=True,internal_responder_model=True,physical_hardware=False,
                  transport_composed=False,cdc_integration=False,parity_validated=False,
                  timing_qualified=False,reset_after_traffic=False,thermal_validated=False)
    return result

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--review',type=Path,required=True)
    a=ap.parse_args();r=base(a.output);result=compile_review(a.output,r) if r['stage']=='compile' else traffic_review(a.output,r)
    result.update(seconds=r['seconds'],checker_sha256=sha(Path(__file__)))
    a.review.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
