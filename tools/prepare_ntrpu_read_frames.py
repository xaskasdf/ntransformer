"""Compose a private one/two-frame candidate over the eight-credit tracker."""
import argparse,json
from pathlib import Path
from prepare_ntrpu_read_window import prepare as window,sha,replace_once

def prepare(root,out,frames,binding,stub,legacy=False):
    assert frames in (1,2)
    window(root,out,8,binding,stub)
    inherited=json.loads((out/'VARIANT.json').read_text())
    extra=['fpga/experimental/ntrpu_hbm_read_frames.sv','fpga/tb/ntrpu_read_frames_body.svh',
           'tools/prepare_ntrpu_read_frames.py','tools/check_ntrpu_read_frames.py']
    for name in extra:
        p=out/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((root/name).read_bytes())
    path=out/'fpga/experimental/ntrpu_hbm_read_frame.sv'
    path.write_text(path.read_text()+'\n'+(root/'fpga/experimental/ntrpu_hbm_read_frames.sv').read_text())
    path=out/'fpga/experimental/ntrpu_hbm_read_return.sv'
    path.write_text(replace_once(path.read_text(),'ntrpu_hbm_read_frame frame (',
                                'ntrpu_hbm_read_frames #(.FRAMES('+str(frames)+')) frame ('))
    path=out/'fpga/tb/tb_ntrpu_read_return.sv';bench=path.read_text()
    if legacy:
        bench=bench.replace('cmd_ready==0&&native_reads','cmd_ready=='+str(3 if frames==2 else 0)+'&&native_reads')
        bench=bench.replace('!drained&&cmd_ready==0,"COMPLETION_RESERVATION"',
                            '!drained&&cmd_ready=='+str(3 if frames==2 else 0)+',"COMPLETION_RESERVATION"')
    else:
        head=bench[:bench.index('    integer scenario=')]
        head=replace_once(head,'COMMON=0,NATIVE=0','COMMON=0,NATIVE=0,FRAMES='+str(frames))
        monitor=bench[bench.index('    for(genvar w=0;w<2;w++)begin: window_monitor'):bench.index('    initial begin #200000;')]
        bench=head+(root/'fpga/tb/ntrpu_read_frames_body.svh').read_text()+monitor+'\n    initial begin #200000;$fatal(1,"RF_WATCHDOG");end\nendmodule\n'
    path.write_text(bench)
    if not legacy:
        path=out/'tools/ntrpu_read_return.tcl'
        flow=path.read_text().replace('RR ALL_PASS','RF ALL_PASS')
        flow=flow.replace('$native?4:20','$native?4:'+str(12 if frames==2 else 4))
        path.write_text(flow)
    path=out/'fixtures/stub.sv'
    path.write_text(path.read_text().replace('[0:127]','[0:1023]').replace('request_write[lane]>=128','request_write[lane]>=1024'))
    for name in ['tools/run_ntrpu_read_return.py','tools/run_ntrpu_read_return_native.py','tools/check_ntrpu_read_return_native.py']:
        path=out/name;text=path.read_text()
        text=text.replace("experiment='read_window'","experiment='read_frames'")
        text=text.replace("experiment='read_window_native'","experiment='read_frames_native'")
        text=text.replace("r['experiment']=='read_window_native'","r['experiment']=='read_frames_native'")
        text=text.replace('READ_WINDOW_PREFLIGHT_REVIEWED','READ_FRAMES_PREFLIGHT_REVIEWED')
        text=text.replace('READ_WINDOW_NATIVE_TRAFFIC_REVIEWED','READ_FRAMES_NATIVE_TRAFFIC_REVIEWED')
        text=text.replace('READ_WINDOW_NATIVE_SNAPSHOT_REVIEWED','READ_FRAMES_NATIVE_SNAPSHOT_REVIEWED')
        text=text.replace("'READ_RETURN_NATIVE_SNAPSHOT_REVIEWED','READ_FRAMES_NATIVE_SNAPSHOT_REVIEWED'",
                          "'READ_RETURN_NATIVE_SNAPSHOT_REVIEWED','READ_WINDOW_NATIVE_SNAPSHOT_REVIEWED','READ_FRAMES_NATIVE_SNAPSHOT_REVIEWED'")
        if name.endswith('check_ntrpu_read_return_native.py'):
            text=text.replace('from check_ntrpu_read_return import sha,replay,controls','from check_ntrpu_read_frames import sha,replay,controls')
        path.write_text(text)
    checker=(root/('tools/check_ntrpu_read_window.py' if legacy else 'tools/check_ntrpu_read_frames.py')).read_text()
    if legacy:
        checker=checker.replace("r['experiment']=='read_window'","r['experiment']=='read_frames'").replace('READ_WINDOW_PREFLIGHT_REVIEWED','READ_FRAMES_LEGACY_REVIEWED')
    (out/'tools/check_ntrpu_read_return.py').write_text(checker)
    result=dict(variant='read_frames',frames=frames,credits=8,legacy=legacy,
                inherited_variant=inherited,originals={n:sha(root/n) for n in extra},
                prepared={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*') if p.is_file() and p.name!='VARIANT.json'})
    (out/'VARIANT.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('output',type=Path);ap.add_argument('--frames',type=int,choices=(1,2),required=True)
    ap.add_argument('--binding',type=Path,required=True);ap.add_argument('--stub',type=Path,required=True);ap.add_argument('--legacy',action='store_true')
    a=ap.parse_args();r=prepare(Path(__file__).resolve().parents[1],a.output,a.frames,a.binding,a.stub,a.legacy)
    print(json.dumps(dict(frames=r['frames'],files=len(r['prepared']))))
