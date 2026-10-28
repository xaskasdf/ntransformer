"""Generate isolated throughput, physical-clock and routed comparison fixtures."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

from prepare_ntrpu_candidate_dispatch import ports
from prepare_ntrpu_registered_dispatch import one
from prepare_ntrpu_candidate_netlist import ALWAYS, SCALAR_DATA, CHANNEL_DATA

ROOT = Path(__file__).resolve().parents[1]
DIRECT = ['ntrpu_buffer_pool_allocator', 'ntrpu_buffer_pool_allocator_decoded',
          'ntrpu_rv_elastic', 'ntrpu_candidate_transport']
GENERATED = ['ntrpu_destination_candidate', 'ntrpu_peer_multibuffer_offer',
             'ntrpu_peer_multibuffer_reserved', 'ntrpu_candidate_dispatch']


PERF = '''    initial begin
        for(integer mode=0;mode<5;mode++)begin
            for(integer size=0;size<3;size++)begin
                profile=mode;block_beats=size==0?1:size==1?8:64;
                reset_case(mode*3+size+1);
                $display("PF CONFIG %0d %0d %0d 64",scenario,profile,block_beats);
                if(profile==4)hold_w=1;
                fork
                    begin for(integer job0=0;job0<64;job0++)push_one(0,1+2*job0,block_beats);end
                    begin for(integer job1=0;job1<64;job1++)push_one(1,2+2*job1,block_beats);end
                    begin
                        if(profile==4)begin
                            wait_cpl(1,64);
                            need(cpls[0]==0,"BLOCKED_PEER_INDEPENDENCE");
                            $display("PF RELEASE %0d %0d",cycle,scenario);hold_w=0;
                        end
                    end
                join
                drain_all;pass_case;
            end
        end
        $display("PF ALL_PASS cases=15 cycle=%0d",cycle);$finish;
    end
'''


def performance(text):
    text = one(text, 'module tb_ntrpu_candidate_transport;', 'module tb_ntrpu_transport_perf;')
    text = one(text, '    integer error_tag=-1;', '    integer error_tag=-1;\n    integer profile=0,block_beats=1,origin=0;')
    text = text.replace('    task ', '    task automatic ')
    # Named-block disable can terminate other concurrent activations of that
    # task. Return only from the producer whose handshake actually occurred.
    text = one(text, 'step;submit_valid[ch]=0;disable waiting;end step;end',
               'step;submit_valid[ch]=0;return;end step;end')
    text = text.replace('tries<5000', 'tries<20000')
    text = one(text, 'need(t<1024&&submitted[p]<1024,"TAG_BOUNDS");',
               'need(t<1024&&submitted[p]<1024,"TAG_BOUNDS");need(lengths[t]==0,"UNIQUE_PUBLICATION_TAG");')
    text = one(text, 'fetch_ready=rst_n&&!hold_fetch;',
               'fetch_ready=rst_n&&!hold_fetch&&(profile!=1||(cycle-origin)%8==0);')
    text = one(text, '(cycle%11!=4)', '(profile!=2||(cycle-origin)%2==0)')
    text = one(text, '&&cycle%5!=0', '')
    text = one(text, '&&cycle%7!=0', '&&(profile!=3||(cycle-origin)%2==0)')
    text = one(text, '&&cycle%13!=0', '')
    text = one(text, 'cycle+20+order[p*1024+b_tail[p]]%7', 'cycle+4')
    text = one(text, 'scenario=number;rst_n=1;', 'scenario=number;origin=cycle;rst_n=1;')
    begin = text.index('    initial begin\n        reset_case(1);')
    end = text.index('    initial begin #1000000;', begin)
    return text[:begin]+PERF+text[end:]


def physical(signals):
    lines = ['`timescale 1ns/1ps', '// Same interfaces, explicit clock buffer; no timing sampler registers.',
             'module ntrpu_transport_physical (']
    lines += [f'    {d} wire {w} {n}'+(',' if i+1<len(signals) else '') for i,(d,w,n) in enumerate(signals)]
    lines += [');', '    wire transport_clk;',
              '    (* DONT_TOUCH = "yes" *) BUFGCE parent_clock (.I(clk),.CE(1\'b1),.O(transport_clk));',
              '    ntrpu_candidate_transport core (.clk(transport_clk),.*);', 'endmodule', '']
    return '\n'.join(lines)


def comparison(text, signals):
    text = one(text, 'module tb_ntrpu_candidate_transport;', 'module tb_ntrpu_transport_netlist;')
    text = one(text, 'ntrpu_candidate_transport dut', 'ref_ntrpu_candidate_transport dut')
    text = one(text, '    initial begin\n        reset_case(1);', '    initial begin\n        #150; // Respect unmodified UNISIM global startup reset.\n        reset_case(1);')
    text = one(text, '$finish;', '$display("TN ALL_PASS checks=%0d",tn_checks);$finish;')
    out = [(d,w,n) for d,w,n in signals if d=='output']
    assert {n for _,_,n in out} == set(ALWAYS)|set(SCALAR_DATA)|set(CHANNEL_DATA)
    lines = ['    integer tn_checks=0;']
    lines += [f'    wire {w} route_{n};' for _,w,n in out]
    lines += ['    ntrpu_transport_physical routed_dut (']
    lines += [f'        .{n}({"route_" if d=="output" else ""}{n})'+(',' if i+1<len(signals) else '')
              for i,(d,w,n) in enumerate(signals)]
    lines += ['    );', '    always @(posedge clk)begin', '        if(rst_n)begin']
    for n in ALWAYS:
        lines.append(f'            if(route_{n} !== {n})$fatal(1,"TN_MISMATCH {n} case=%0d",scenario);')
    for n,v in SCALAR_DATA.items():
        lines.append(f'            if({v} && route_{n} !== {n})$fatal(1,"TN_MISMATCH {n} case=%0d",scenario);')
    lines += ['            for(integer lane=0;lane<2;lane++)begin']
    for n,(v,w) in CHANNEL_DATA.items():
        a=f'[lane*{w}+:{w}]'
        lines.append(f'                if({v}[lane] && route_{n}{a} !== {n}{a})$fatal(1,"TN_MISMATCH {n} case=%0d",scenario);')
    lines += ['                if(route_m_wvalid[lane]&&m_wready[lane])',
              '                    $display("TN W %0d %0d %h %h %b",scenario,lane,route_m_wdata[lane*512+:512],route_m_wstrb[lane*64+:64],route_m_wlast[lane]);',
              '                if(route_cpl_valid[lane]&&cpl_ready[lane])',
              '                    $display("TN CPL %0d %0d %0d %0d",scenario,lane,route_cpl_tag[lane*64+:64],route_cpl_status[lane*2+:2]);',
              '            end', '            if(route_fetch_valid&&fetch_ready)',
              '                $display("TN FETCH %0d %b %h",scenario,route_fetch_channel,route_fetch_desc);',
              '            tn_checks++;$display("TN MATCH %0d %0d",scenario,tn_checks);',
              '        end', '    end', '']
    return one(text, 'endmodule', '\n'.join(lines)+'endmodule')


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path)
    ap.add_argument('--netlist',type=Path);args=ap.parse_args();out=args.output.resolve()
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_candidate_dispatch.py'),str(out)],check=True)
    signals=ports((ROOT/'fpga/rtl/ntrpu_registered_dispatch.sv').read_text())
    original=(ROOT/'fpga/tb/tb_ntrpu_candidate_transport.sv').read_text()
    bench=performance(original)
    (out/'tb_ntrpu_transport_perf.sv').write_text(bench)
    (out/'PERF.diff').write_text(''.join(difflib.unified_diff(original.splitlines(True),bench.splitlines(True))))
    (out/'ntrpu_transport_physical.sv').write_text(physical(signals))
    if args.netlist:
        net=args.netlist.read_text();top=net.split('module ntrpu_transport_physical\n',1)[1].split('\n  wire',1)[0]
        assert re.findall(r'^  (input|output) (\[[^\]]+\])?\s*(\w+);',top,re.M)==signals, 'Routed ports changed'
        sources={n:((ROOT/'fpga/rtl' if n in DIRECT else out)/(n+'.sv')).read_text() for n in DIRECT+GENERATED}
        modules={m for t in sources.values() for m in re.findall(r'^module (\w+)',t,re.M)}
        mapping={n:'ref_'+n for n in modules};inverse={v:k for k,v in mapping.items()}
        assert not set(mapping.values()) & set(re.findall(r'^module (\w+)',net,re.M))
        regex=re.compile(r'\b('+'|'.join(sorted(mapping))+r')\b')
        reverse=re.compile(r'\b('+'|'.join(sorted(inverse))+r')\b')
        for n,t in sources.items():
            renamed=regex.sub(lambda m:mapping[m[0]],t)
            assert reverse.sub(lambda m:inverse[m[0]],renamed)==t
            (out/('ref_'+n+'.sv')).write_text(renamed)
        (out/'tb_ntrpu_transport_netlist.sv').write_text(comparison(original,signals))
        (out/'COMPARISON.json').write_text(json.dumps(dict(netlist_sha256=hashlib.sha256(args.netlist.read_bytes()).hexdigest(),
            latency_edges=0,startup_ns=150,sdf=False,force=False,reversible_renaming=True,ports=signals),indent=2)+'\n')
    (out/'STUDY_FILES.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in out.iterdir() if p.is_file()},indent=2)+'\n')


if __name__=='__main__':main()
