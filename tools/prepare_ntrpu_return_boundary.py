"""Generate a boundary sensitivity bench around the frozen registered-return RTL."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from prepare_ntrpu_registered_dispatch import one

PROFILES = [
    dict(name='fast', b_delay=4, payload_delay=0),
    dict(name='slow_b', b_delay=160, payload_delay=0),
    dict(name='payload_latency', b_delay=32, payload_delay=64),
    dict(name='lane0_blocked', b_delay=32, payload_delay=0),
    dict(name='aw_pause', b_delay=32, payload_delay=0),
]


def bench(text):
    text = one(text, 'module tb_ntrpu_transport_perf;', 'module tb_ntrpu_return_boundary;')
    text = one(text, 'integer profile=0,block_beats=1,origin=0;',
               'integer profile=0,block_beats=1,origin=0;\n    integer target_credits=4,b_delay=4,payload_delay=0,source_due=0;')
    text = one(text, '&&(profile!=1||(cycle-origin)%8==0)', '')
    text = one(text, '&&(profile!=2||(cycle-origin)%2==0)', '&&cycle>=source_due')
    text = one(text, 'm_awready[ch]=rst_n&&!hold_aw[ch];',
               'm_awready[ch]=rst_n&&!hold_aw[ch]&&aws[ch]-bs[ch]<target_credits&&(profile!=4||(cycle-origin)%64>=32);')
    text = one(text, 'm_wready[ch]=rst_n&&!hold_w[ch]&&(profile!=3||(cycle-origin)%2==0);',
               'm_wready[ch]=rst_n&&!hold_w[ch]&&ws[ch]-bs[ch]<target_credits;')
    text = one(text, 'src_beats=lengths[t];src_index=0;',
               'src_beats=lengths[t];src_index=0;source_due=cycle+payload_delay;')
    text = one(text, 'cycle+4;b_tail[p]++;', 'cycle+b_delay;b_tail[p]++;')
    text = one(text, 'aws[p]++;$display',
               'aws[p]++;need(aws[p]-bs[p]<=target_credits,"TARGET_AW_CREDITS");$display')
    text = one(text, 'output_beats[t]++;if(m_wlast[p])ws[p]++;',
               'need(ws[p]+1-bs[p]<=target_credits,"TARGET_W_CREDITS");output_beats[t]++;if(m_wlast[p])ws[p]++;')
    start = text.index('    initial begin\n        #150;')
    end = text.index('    initial begin #1000000;', start)
    body = '''    initial begin
        #150;
        for(integer mode=0;mode<5;mode++)begin
            for(integer credit=0;credit<3;credit++)begin
                for(integer size=0;size<3;size++)begin
                    profile=mode;target_credits=1<<credit;block_beats=size==0?1:size==1?8:64;
                    b_delay=mode==0?4:mode==1?160:32;payload_delay=mode==2?64:0;
                    reset_case(mode*9+credit*3+size+1);
                    $display("BN CONFIG %0d %0d %0d 64 %0d %0d %0d",scenario,profile,block_beats,target_credits,b_delay,payload_delay);
                    if(profile==3)hold_w=1;
                    fork
                        begin for(integer job0=0;job0<64;job0++)push_one(0,1+2*job0,block_beats);end
                        begin for(integer job1=0;job1<64;job1++)push_one(1,2+2*job1,block_beats);end
                        begin
                            if(profile==3)begin
                                wait_cpl(1,64);need(cpls[0]==0,"BLOCKED_PEER_INDEPENDENCE");
                                $display("BN RELEASE %0d %0d",cycle,scenario);hold_w=0;
                            end
                        end
                    join
                    drain_all;pass_case;
                end
            end
        end
        $display("BN ALL_PASS cases=45 cycle=%0d",cycle);$finish;
    end
'''
    text = text[:start]+body+text[end:]
    return text.replace('tries<10000', 'tries<100000').replace('#1000000;', '#5000000;')


def main():
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('output', type=Path)
    a = ap.parse_args(); out = a.output.resolve()
    subprocess.run([sys.executable, '-B', str(Path(__file__).with_name('prepare_ntrpu_return_cut.py')), str(out)], check=True)
    original = out/'tb_ntrpu_transport_perf.sv'; target = out/'tb_ntrpu_return_boundary.sv'
    target.write_text(bench(original.read_text()))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    (out/'BOUNDARY.json').write_text(json.dumps(dict(profiles=PROFILES, credits=[1,2,4], beats=[1,8,64],
        jobs_per_lane=64, cases=45, period_ns=5, source_bench_sha256=sha(original), bench_sha256=sha(target),
        unchanged_circuit=json.loads((out/'RETURN_CUT.json').read_text()),
        scope='Behavioral target at existing AXI ports; no PCIe, HBM, CDC or width converter'), indent=2)+'\n')


if __name__ == '__main__':
    main()
