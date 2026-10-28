"""Audit full-matrix jobs, group completions, AXI traffic and stored FP64 bits."""
import re
import struct

from check_ntrpu_gemv_memory import bursts

CASES = [(33,64,32,0,0),(65,32,32,0,0),(64,9,4,0,0),
         (64,32,32,1,3),(64,32,32,2,3),(64,32,32,3,7),
         (64,32,32,4,0),(33,33,32,0,1),(64,32,32,5,1),(33,64,32,0,0)]


def review(text, seed, fixture=None, selected_job=0):
    errors = []
    def need(ok, why):
        if not ok:
            errors.append(why)
    def records(prefix):
        return [{k:int(v) for k,v in re.findall(r'(\w+)=(\d+)', line)}
                for line in text.splitlines() if line.startswith(prefix+' ')]
    need(not re.search(r'^\s*(Fatal:|ERROR:|FAIL\b)', text, re.M|re.I), 'Executed failure')
    need(records('NTRPU_MATRIX_MEMORY_SEED') == [dict(seed=seed)], 'Seed')
    jobs = ([j for j in fixture['jobs'] if not selected_job or j['id']==selected_job]
            if fixture else [])
    contracts = ([(j['rows'],j['k_blocks'],j['tile_k'],0,0) for j in jobs] if fixture else CASES)
    ids = [j['id'] for j in jobs] if fixture else list(range(1,11))
    need(bool(contracts), 'Empty selection')
    if selected_job:
        need(records('NTRPU_MATRIX_MEMORY_FIXTURE') == [dict(selected=selected_job,jobs=len(fixture['jobs']))], 'Fixture selection')
    rows = records('NTRPU_MATRIX_MEMORY_CASE')
    groups = records('NTRPU_MATRIX_MEMORY_GROUP')
    need([r.get('id') for r in rows] == ids, 'Case identities/order')
    expected_groups = []
    expected_values = {}
    published = 0
    for index, (r, contract) in enumerate(zip(rows, contracts)):
        ident=ids[index]; n,k,chunk,mode,status=contract
        need(tuple(r.get(f) for f in ['rows','blocks','chunk','mode','status'])==contract, 'Case contract '+str(ident))
        reset=mode==4; rejected=status==1
        group_count=0 if rejected else (n+31)//32
        computed=0 if rejected else 32 if mode in {1,3} else n
        written=0 if rejected else 32 if mode in {1,3} else 40 if reset else n
        need(r.get('reset')==int(reset), 'Reset '+str(ident))
        need(r.get('groups')==group_count and r.get('computed')==computed, 'Group/compute count '+str(ident))
        need(0<r.get('cycles',0)<(10000000 if fixture else 200000), 'Latency bound '+str(ident))
        need((0<r.get('first_output',0)<r.get('cycles',0)) if written else r.get('first_output')==0, 'First output '+str(ident))
        ar=weight_beats=tile_count=aw=bus_w=0
        weight_address=4032
        for g in range(group_count):
            gn=min(32,n-g*32)
            ar+=bursts(4032,k*64)
            if mode!=3 or g==0:
                for kb in range(0,k,chunk):
                    size=gn*min(chunk,k-kb)*34
                    ar+=bursts(weight_address,size);weight_beats+=size//64;tile_count+=1
                    weight_address+=size
            group_written=max(0,min(gn,written-g*32))
            if group_written:
                aw+=bursts(4032+g*256,((group_written+7)//8)*64)
                bus_w+=(group_written+7)//8
            if not(reset and g==1):
                gs=status if g==1 and mode in {1,2,3} else 0
                expected_groups.append(dict(case=ident,group=g,rows=gn,status=gs,bytes=group_written*8))
        expected_traffic=(tile_count,ar,group_count*k,weight_beats,aw,bus_w,aw-int(reset),written*8)
        need(tuple(r.get(f) for f in ['tiles','ar','activation_beats','weight_beats','aw','w','b','bytes'])==expected_traffic,
             'AXI traffic '+str(ident))
        if status==0 and not reset:
            published+=n
        for row in range(written):
            bits=(jobs[index]['reference_fp64_bits'][row] if fixture else struct.pack('>d',float(sum(
                ((b+lane)%5-2)*((b+row*3+lane)%7-3) for b in range(k) for lane in range(32)))).hex())
            expected_values[(ident*1000+row//32,row%32)]=bits
    need([{k:v for k,v in g.items() if k!='cycles'} for g in groups]==expected_groups, 'Group completion contracts/order')
    for r in rows:
        timeline=[g.get('cycles',0) for g in groups if g.get('case')==r.get('id')]
        need(all(0<a<b for a,b in zip(timeline,timeline[1:]+[r.get('cycles',0)])), 'Group completion time')
    values=[]
    for line in text.splitlines():
        if line.startswith('NTRPU_STORE_VALUE '):
            f=dict(re.findall(r'(\w+)=([^ ]+)',line))
            try:
                key=(int(f['case']),int(f['row']))
            except (KeyError, ValueError):
                need(False,'Malformed physical value');continue
            need(f.get('bits')==expected_values.get(key), 'Physical FP64 oracle '+str(key))
            values.append(key)
    need(values==list(expected_values), 'Written identities/order')
    summaries=records('NTRPU_MATRIX_MEMORY_PASS')
    need(len(summaries)==1, 'Suite marker')
    summary=summaries[0] if summaries else {}
    need(summary.get('cases')==len(contracts), 'Suite cases')
    for key in ['cycles','computed','groups','resets']:
        need(summary.get(key)==sum(r.get('reset' if key=='resets' else key,0) for r in rows), 'Total '+key)
    need(summary.get('read_beats')==sum(r.get('activation_beats',0)+r.get('weight_beats',0) for r in rows), 'Total reads')
    need(summary.get('write_bytes')==len(values)*8==sum(r.get('bytes',0) for r in rows), 'Total written bytes')
    need(summary.get('ar_stalls',0)>0 and summary.get('done_stalls',0)>0, 'Backpressure coverage')
    activation=sum(r.get('activation_beats',0) for r in rows)*64
    weights=sum(r.get('weight_beats',0) for r in rows)*64
    write_bus=sum(r.get('w',0) for r in rows)*64
    return dict(status='FAIL' if errors else 'PASS',errors=errors,seed=seed,cases=rows,groups=groups,summary=summary,
                written_private_values=len(values),published_values=published,
                traffic=dict(activation_bytes=activation,weight_bytes=weights,write_strobed_bytes=len(values)*8,
                             write_bus_bytes=write_bus,write_bursts=sum(r.get('aw',0) for r in rows),
                             bus_bytes_per_job_cycle=(activation+weights+write_bus)/max(1,summary.get('cycles',0))))
