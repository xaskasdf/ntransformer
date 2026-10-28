"""Generate isolated HBM startup and asynchronous transport test fixtures."""
import argparse
import json
import hashlib
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]

def native_pins(binding):
    declarations=[];pins=['.axi_clk(mem_clk)', '.axi_cold_reset_n(mem_rst_n)']
    for name,direction,width in binding['management']:
        if direction=='output':
            declarations.append('    wire '+(f'[{width-1}:0] ' if width>1 else '')+name+';');value=name
        elif name.startswith('HBM_REF_CLK'):value='ref_clk'
        elif name.endswith('PCLK'):value='apb_clk'
        elif name.endswith('PRESET_N'):value='apb_rst_n'
        else:value=f"{width}'b0"
        pins.append(f'.{name}({value})')
    return declarations,pins

def boot(out):
    binding=json.loads((out/'binding/BINDING.json').read_text())
    declarations,pins=native_pins(binding)
    sys.path.insert(0,str(ROOT/'tools'))
    from prepare_ntrpu_hbm_native_link import WRITE
    for name,(direction,width) in WRITE.items():
        pins.append(f'.s_{name}('+ (f"{width*2}'b0" if direction=='input' else '')+')')
    text='''`timescale 1ns/1ps
module tb_ntrpu_hbm_boot;
    reg mem_clk=0,ref_clk=0,apb_clk=0;
    always #1.111 mem_clk=~mem_clk;
    always #5 ref_clk=~ref_clk;
    initial begin #1.7;forever #5 apb_clk=~apb_clk;end
    reg mem_rst_n=0,apb_rst_n=0;
'''+ '\n'.join(declarations)+'\n    ntrpu_hbm_native_link dut(\n        '+',\n        '.join(pins)+');\n'+'''
    reg [1:0] previous=0;
    always @(posedge apb_clk)begin
        if({apb_complete_1,apb_complete_0}!==previous)begin
            $display("BOOT STATUS %0t %b %b",$time,apb_complete_1,apb_complete_0);
            previous={apb_complete_1,apb_complete_0};
        end
    end
    initial begin
        #4700;
        @(negedge apb_clk);apb_rst_n=1;
        @(negedge mem_clk);mem_rst_n=1;
        $display("BOOT RELEASE %0t",$time);
        wait(apb_complete_0===1'b1&&apb_complete_1===1'b1);
        repeat(32)@(posedge apb_clk);
        if(DRAM_0_STAT_CATTRIP||DRAM_1_STAT_CATTRIP)$fatal(1,"BOOT_THERMAL_TRIP");
        $display("BOOT PASS %0t",$time);$finish;
    end
    initial begin #5000000;$fatal(1,"BOOT_TIMEOUT");end
endmodule
'''
    (out/'tb_ntrpu_hbm_boot.sv').write_text(text)


def asynchronous(out, reference):
    from prepare_ntrpu_hbm_transport import one
    generated=out/'generated'
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_hbm_transport.py'),
                    str(generated),'--reference',str(reference)],check=True)
    text=(generated/'tb_ntrpu_hbm_transport.sv').read_text()
    text=one(text,'module tb_ntrpu_hbm_transport;','module tb_ntrpu_hbm_cdc;')
    text=one(text,'ntrpu_hbm_transport dut(.*);','ntrpu_hbm_transport_async dut(.*);')
    text=one(text,'    reg rst_n=0,stop=0,fault_in=0;', '''    reg cold_reset_n=0,stop=0,fault_in=0;
    wire rst_n,mem_rst_n,started;
    wire [11:0] inflight;
    reg [1:0] init_complete_async=0;
    reg mem_clk=0,mem_pause=0;
    realtime mem_half=1.111;
    initial begin #0.731;forever begin #(mem_half);if(!mem_pause)mem_clk=~mem_clk;end end
    function automatic longint stamp();return longint'($realtime*1000.0);endfunction''')
    text=one(text,'        rst_n=0;stop=0;', '        cold_reset_n=0;init_complete_async=0;mem_pause=0;stop=0;')
    text=one(text,'repeat(32)step;scenario=number;rst_n=1;$display("TE CASE %0d %0d",cycle,scenario);step;',
        '''repeat(32)step;scenario=number;$display("TE CASE %0d %0d",cycle,scenario);
        cold_reset_n=1;
        if(number!=18)begin
            repeat(12)step;init_complete_async=3;
            repeat(40)step;need(started&&!fault,"STARTUP_TIMEOUT");
        end else repeat(12)step;
        step;''')
    # CDC owns additional parent AW slots. Adapter limit remains exactly four.
    text=one(text,'aws[0]==4&&bs[0]==0&&ha[0]>=24&&!drained',
        'aws[0]>=4&&bs[0]==0&&ha[0]>=24&&!drained&&dut.memory[0].adapter.guard.used==4')
    text=one(text,'$display("HT SATURATED %0d %0d %0d",cycle,aws[0],ha[0]);',
        '$display("HT SATURATED %0d %0d %0d",cycle,dut.memory[0].adapter.guard.used,ha[0]);')
    text=one(text,'        $display("HT ALL_PASS cases=17 cycle=%0d",cycle);$finish;',
        (ROOT/'fpga/tb/ntrpu_hbm_cdc_cases.svh').read_text())
    begin=text.index('    // Two independent AXI3 sinks.')
    target=text[begin:].replace('@(negedge clk)','@(negedge mem_clk)').replace('@(posedge clk)','@(posedge mem_clk)')
    target=re.sub(r'\brst_n\b','mem_rst_n',target)
    target=re.sub(r'\bcycle\b','hc',target)
    text=text[:begin]+target
    # Events from both clocks must share a physical timebase for causality.
    text=re.sub(r'(\$display\("(?:TE|HT) [^"\n]*"\s*,\s*)(?:cycle|hc)\b',r'\1stamp()',text)
    monitor='''    always @(negedge clk)begin
        #0.1;
        if(rst_n)begin
            need(inflight[5:0]==aws[0]-bs[0]&&inflight[11:6]==aws[1]-bs[1],"CDC_LEDGER");
            need(inflight[5:0]<=16&&inflight[11:6]<=16,"CDC_SOURCE_CREDIT_LIMIT");
            need(!drained||inflight==0,"CDC_DRAIN_WAITS_RESPONSES");
            if(!started)need(submit_ready==0&&!fetch_valid&&h_awvalid==0&&h_wvalid==0&&!drained,"CDC_BEFORE_START");
            $display("CD STATE %0d %0d %b %b %b %h %b %b",stamp(),scenario,started,fault,drained,inflight,submit_ready,mem_rst_n);
        end
    end
'''
    text=one(text,'endmodule',monitor+'endmodule')
    (generated/'tb_ntrpu_hbm_cdc.sv').write_text(text)
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (generated/'CDC.json').write_text(json.dumps(dict(
        reference=str(reference),clock_domains=2,timestamp_units='picoseconds',
        cases=21,sources={p.name:sha(p) for p in generated.glob('*.sv')}),indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('output',type=Path);ap.add_argument('--stage',choices=('boot','async'),required=True)
    ap.add_argument('--reference',type=Path)
    a=ap.parse_args();out=a.output
    if a.stage=='async':
        assert a.reference;asynchronous(out,a.reference);return
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_hbm_native_link.py'),str(out/'native_hbm.veo'),str(out/'binding')],check=True)
    boot(out)

if __name__=='__main__':main()
