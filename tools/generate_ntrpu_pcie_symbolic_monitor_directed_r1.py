"""Raw traffic separating current guards, each pending origin, and late publication."""
import hashlib
import json
from pathlib import Path
import sys
from generate_ntrpu_pcie_inventory_vectors_r2 import Vectors, place, rq_header, rq, rc
from generate_ntrpu_pcie_symbolic_ced_vectors import close_without_drain

def generate():
    v=Vectors();witnesses={}
    v.case_begin('delayed_arming_and_immediate_guards')
    # Only empty startup rows are rewritten; the independent model remains empty.
    for index in (2,3):
        row=v.rows[index].split();row[3]='0';row[16]='0';v.rows[index]=' '.join(row)
    v.observing=False
    for _ in range(6):v.step(profile=0)
    witnesses['arming_origin']=len(v.rows);v.step()
    for _ in range(10):v.step()
    v.end()
    v.case_begin('posted_write_and_each_pipeline_stage')
    for _ in range(8):v.step()
    write=(place((0,rq_header(0,0x300,1,write=True)+b'abcd')),15 | 1<<20 | 1<<26 | 4<<28)
    witnesses['posted_origin']=len(v.rows);v.step(rq_beat=write)
    for _ in range(8):v.step()
    v.step(rq_beat=write);v.step(rq_beat=write)
    for _ in range(8):v.step()
    v.step(rq_beat=write,rq_ready=0);v.step(rq_beat=write,rq_ready=0)
    v.step(rq_beat=write)
    for _ in range(8):v.step()
    v.end()
    v.case_begin('read_completion_last_retirement')
    for _ in range(8):v.step()
    v.step(rq_beat=rq(13));v.step();v.step()
    witnesses['completion_origin']=len(v.rows);v.step(rc_beat=rc(13))
    for _ in range(8):v.step()
    v.end()
    v.case_begin('isolated_late_requester_fault')
    for _ in range(8):v.step()
    v.step(rq_beat=rq(14));data,user=rc(14)
    witnesses['late_fault_origin']=len(v.rows);v.step(rc_beat=(data^(1<<48),user),expect_error=True)
    for _ in range(12):v.step()
    v.end()
    v.case_begin('early_fault_survives_profile_restore')
    for _ in range(8):v.step()
    witnesses['early_fault_origin']=len(v.rows);v.step(profile=0,expect_error=True)
    witnesses['profile_restored_origin']=len(v.rows);v.step()
    for _ in range(10):v.step()
    v.end()
    for distance in range(6):
        v.case_begin('cold_busy_distance_'+str(distance))
        for _ in range(8):v.step()
        v.step(rq_beat=rq(40+distance))
        for _ in range(distance):v.step()
        close_without_drain(v)
    v.case_begin('final_cold_rearm')
    for _ in range(8):v.step()
    v.end()
    return v,witnesses

if __name__=='__main__':
    out=Path(sys.argv[1]);v,w=generate();text='\n'.join(v.rows)+'\n'
    with out.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    manifest=dict(rows=len(v.rows),scenarios=v.scenarios,witnesses=w,
        vectors_sha256=hashlib.sha256(text.encode()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        native_rows=0,scope='Directed monitor arming,posted drain,held traffic,read completion,isolated late fault,early fault after profile restoration,and six busy cold distances. Between-edge X/Z guard probes belong to the independent testbench.')
    with out.with_suffix('.json').open('x') as f:json.dump(manifest,f,indent=2);f.write('\n')
    print(json.dumps(dict(rows=len(v.rows),scenarios=len(v.scenarios),witnesses=w,sha256=manifest['vectors_sha256'])))
