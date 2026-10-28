"""Generate an isolated dispatcher with registered, validated queue heads."""
import argparse
import difflib
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def one(text,old,new):
    assert text.count(old)==1,old
    return text.replace(old,new)


def emit(out,source,name,change):
    before=(ROOT/source).read_text();after=change(before)
    target=out/name;target.write_text(after)
    (out/(name+'.diff')).write_text(''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile=source,tofile=name)))
    return dict(source=source,source_sha256=hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                generated=name,generated_sha256=hashlib.sha256(target.read_bytes()).hexdigest())


def dispatcher(text):
    text=one(text,'module ntrpu_destination_dispatch #(','module ntrpu_destination_registered #(')
    text=one(text,'    output wire [1:0] head_valid,','    output wire [1:0] head_valid, head_good,')
    text=one(text,'        reg [231:0] fifo [0:DEPTH-1];','''        reg [231:0] fifo [0:DEPTH-1];
        reg cached_valid,cached_good;
        reg [231:0] cached_desc;
        wire [231:0] read_desc=fifo[rd[g]];
        wire [63:0] read_address=read_desc[135:72];
        wire [8:0] read_beats={1'b0,read_desc[7:0]}+9'd1;
        wire [14:0] read_end={3'b0,read_address[11:0]}+{read_beats,6'b0};
        // Cache metadata only, never receiver availability or a resource grant.
        // The FIFO entry stays occupied until the real reserve_take handshake.
        always @(posedge clk)begin
            if(!rst_n)begin cached_valid<=0;cached_good<=0;cached_desc<=0;end
            else if(reserve_take[g])cached_valid<=0;
            else if(!cached_valid && count[g]!=0)begin
                cached_desc<=read_desc;
                cached_good<=read_address[5:0]==0 && read_beats<=64 && read_end<=4096;
                cached_valid<=1;
            end
        end''')
    text=one(text,'        assign head_valid[g] = rst_n && count[g]!=0;',
             '        assign head_valid[g] = rst_n && cached_valid;\n        assign head_good[g] = cached_good;')
    text=one(text,"        assign head_desc[g*232+:232] = count[g]!=0 ? fifo[rd[g]] : 232'b0;",
             "        assign head_desc[g*232+:232] = cached_desc;")
    return text


def bench(text):
    text=one(text,'module tb_ntrpu_reserved_dispatch;','module tb_ntrpu_registered_dispatch;')
    text=one(text,'ntrpu_reserved_dispatch dut','ntrpu_registered_dispatch dut')
    text=one(text,'            for(p=0;p<2;p++)if(submit_valid[p]&&submit_ready[p])begin',
        '''            $display("RH HEAD %0d %b %b %h %h",cycle,dut.head_valid,dut.head_good,
                dut.head_desc[231:0],dut.head_desc[463:232]);
            for(p=0;p<2;p++)if(submit_valid[p]&&submit_ready[p])begin''')
    return text


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);args=ap.parse_args()
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_reserved_receiver.py'),str(args.output)],check=True)
    manifest=[emit(args.output,'fpga/rtl/ntrpu_destination_dispatch.sv','ntrpu_destination_registered.sv',dispatcher),
              emit(args.output,'fpga/tb/tb_ntrpu_reserved_dispatch.sv','tb_ntrpu_registered_dispatch.sv',bench)]
    (args.output/'REGISTERED_TRANSFORMATION.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__=='__main__':main()
