"""Reconstruct the DDR reset diagnostic from public pin samples, not TB claims."""
import argparse,json,re
from pathlib import Path

def records(text,tag):
    return [dict(re.findall(r'(\w+)=([^ ]+)',line[len(tag)+1:]))
            for line in text.splitlines() if line.startswith(tag+' ')]

def review(text,open_row,phase,lead):
    assert 'Time resolution is 1 ps' in text
    assert 'NTRPU_MODEL_INCOMPLETE' not in text and 'FATAL' not in text and 'Fatal' not in text
    assert records(text,'NTRPU_MODEL_PHASE')==[dict(phase_ps=str(phase),cke_lead_ps=str(lead))]
    assert records(text,'NTRPU_MODEL_TIMING')==[dict(full_delays='1',cold_reset_ps='200000000',cold_cke_ps='500000000',warm_cke_ps='500000000')]
    warm=records(text,'NTRPU_MODEL_WARM_RESET');done=records(text,'NTRPU_MODEL_WINDOW_COMPLETE')
    cold=records(text,'NTRPU_MODEL_COLD_DONE')
    assert len(warm)==len(done)==len(cold)==1
    assert int(warm[0]['open_row'])==int(done[0]['open_row'])==open_row
    assert int(warm[0]['low_ps'])==2600000
    raw_pins=records(text,'NTRPU_MODEL_PINS')
    # Two pin events separated by #0 schedule identical postponed $strobe
    # samples at the same simulation time. Preserve and count that duplicate.
    pins=[]
    for sample in raw_pins:
        if pins and sample==pins[-1]:continue
        pins.append(sample)
    assert len(raw_pins)-len(pins)==int(lead==0)
    assert len({p['time_ps'] for p in pins})==len(pins)
    for pin in pins:
        assert set(pin)=={'time_ps','edge','ck','rst','cke','cs'}
        assert pin['cs']=='1' and all(pin[k] in {'0','1'} for k in ['ck','rst','cke'])
        t=int(pin['time_ps']);edge=int(pin['edge'])
        assert edge==(t+625)//1250
        assert int(pin['ck'])==int(t%1250>=625)
    # Cold low/release/CKE, then optional separate CKE fall, warm low/release/CKE.
    assert len(pins)==6+int(lead>0)
    assert [(int(p['time_ps']),int(p['rst']),int(p['cke'])) for p in pins[:3]]==[
        (0,0,0),(200000000,1,0),(700000000,1,1)]
    reset=int(warm[0]['time_ps'])
    expected=[(reset-lead,1,0)] if lead else []
    expected += [(reset,0,0),(reset+2600000,1,0),(reset+502600000,1,1)]
    assert [(int(p['time_ps']),int(p['rst']),int(p['cke'])) for p in pins[3:]]==expected
    assert reset%1250==phase and reset>int(cold[0]['time_ps'])
    assert int(done[0]['time_ps'])>reset+502600000
    commands=records(text,'NTRPU_MODEL_COMMAND')
    expected_mode=[('1000',0,3,0),('1000',1,2,0x424),('1000',1,1,0x40),
                   ('1000',1,0,0),('1000',0,2,0),('1000',0,1,0x401),
                   ('1000',0,0,0x314),('1110',0,0,0x714)]
    sequence=[(c['opcode'],int(c['bg']),int(c['ba']),int(c['addr'],16)) for c in commands]
    assert sequence==expected_mode+([('0000',0,0,0x123)] if open_row else [])+expected_mode
    times=[int(c['time_ps']) for c in commands]
    assert times==sorted(set(times))
    assert all(int(c['edge'])==(t+625)//1250 and t%1250==625 for c,t in zip(commands,times))
    assert all(700600000<=t<int(cold[0]['time_ps']) for t in times[:8])
    if open_row:assert int(cold[0]['time_ps'])<times[8]<reset-lead
    assert all(reset+503200000<=t<int(done[0]['time_ps']) for t in times[8+open_row:])
    for offset in [0,8+open_row]:
        assert all(times[offset+i+1]-times[offset+i]>=20000 for i in range(6))
        assert times[offset+7]-times[offset+6]>=200000
    initializations=[int(v) for v in re.findall(r'Initialization complete @(\d+)',text)]
    assert initializations and initializations[0]<int(cold[0]['time_ps'])
    errors=[line for line in text.splitlines() if 'ERROR:' in line]
    classified=[]
    for line in errors:
        m=re.fullmatch(r'.+\.(lmr_decode|zq_decode):ERROR:(.+) @(\d+)',line)
        assert m is not None,line
        kind,message,stamp=m.groups();stamp=int(stamp)
        assert stamp>=reset+503200000 and stamp in times[8+open_row:]
        assert message in {'All banks must be Precharged prior to cmdLMR (1 are open)',
                           ' cmdZQ Failure. All banks must be Precharged.'}
        classified.append(dict(kind=kind,time_ps=stamp,message=message))
    return dict(trace_status='DIAGNOSTIC_TRACE_VALID',open_row=open_row,phase_ps=phase,cke_lead_ps=lead,
        reset_low_ps=2600000,cke_after_release_ps=500000000,
        commands=len(commands),pin_transitions=len(pins),duplicate_postponed_samples=len(raw_pins)-len(pins),reset_time_ps=reset,
        completed_time_ps=int(done[0]['time_ps']),initialization_times_ps=initializations,
        model_errors=classified,
        result='WARM_RESET_MODEL_ERRORS_PRESENT' if errors else 'NO_ERRORS_IN_WINDOW_NOT_RESET_QUALIFICATION')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=Path)
    p.add_argument('--row',type=int,choices=[0,1],required=True)
    p.add_argument('--phase',type=int,choices=[0,313],required=True)
    p.add_argument('--lead',type=int,choices=[0,20000],required=True)
    a=p.parse_args();print(json.dumps(review(a.log.read_text(),a.row,a.phase,a.lead),indent=2))
