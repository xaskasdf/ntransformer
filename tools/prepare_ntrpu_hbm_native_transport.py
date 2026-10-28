"""Join approved async transport fixtures to the unchanged native HBM binding."""
import argparse, hashlib, json, subprocess, sys
from pathlib import Path
from prepare_ntrpu_hbm_native_link import WRITE
from prepare_ntrpu_hbm_native_traffic import READ
from prepare_ntrpu_hbm_cdc import native_pins

ROOT = Path(__file__).resolve().parents[1]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def replace(text, old, new):
    assert text.count(old) == 1, old
    return text.replace(old, new)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('output',type=Path);ap.add_argument('reference',type=Path)
    a=ap.parse_args();out=a.output;ref=a.reference
    files=json.loads((ref/'FILES.json').read_text())
    record=json.loads((ref/'result.json').read_text())
    assert record['experiment']=='hbm_cdc' and record['stage']=='async'
    assert record['sources']==record['sources_after']
    for n,h in record['sources'].items():
        if n.startswith('fpga/rtl/'): assert sha(ROOT/n)==h,n
    composition=json.loads((ref/'suite/generated/COMPOSITION.json').read_text())
    generated=out/'generated';generated.mkdir()
    selected=[*('suite/generated/'+n+'.sv' for n in composition['frozen_logic']),
              'suite/generated/tb_ntrpu_hbm_cdc.sv','suite/generated/COMPOSITION.json']
    for n in selected: assert sha(ref/n)==files[n],n
    for n,h in composition['frozen_logic'].items():
        source=ref/'suite/generated'/(n+'.sv');assert sha(source)==h
        (generated/source.name).write_bytes(source.read_bytes())
    subprocess.run([sys.executable,'-B',str(ROOT/'tools/prepare_ntrpu_hbm_native_traffic.py'),str(out)],check=True)
    binding=json.loads((out/'BINDING.json').read_text())
    text=(ref/'suite/generated/tb_ntrpu_hbm_cdc.sv').read_text()
    text=text[:text.index('    task automatic reset_case')]
    text=replace(text,'module tb_ntrpu_hbm_cdc;','module tb_ntrpu_hbm_native_transport;')
    text=replace(text,'reg [1:0] init_complete_async=0;',
        'wire [1:0] init_complete_async={apb_complete_1,apb_complete_0};')
    for name,(direction,width) in WRITE.items():
        if direction=='output':
            text=replace(text,f'reg [{width*2-1}:0] h_{name}=0;',f'wire [{width*2-1}:0] h_{name};')
    text=replace(text,"destination=scenario==15?64'h400000000:64'h100000000+64'(tag)*4096;",
        "destination=(tag%2?64'h200000000:64'h0)+64'h10000+64'(tag/2)*4096;")
    text=replace(text,'            $display("TE S ', '            if(started)$display("TE S ')
    declarations,pins=native_pins(binding['original'])
    for name in WRITE: pins.append(f'.s_{name}(h_{name})')
    for name,(direction,width) in READ.items():
        declarations.append(f'    {"reg" if direction=="input" else "wire"} [{width*2-1}:0] s_{name}'+('=0;' if direction=='input' else ';'))
        pins.append(f'.s_{name}(s_{name})')
    text+='''
    reg ref_clk=0,apb_clk=0,apb_rst_n=0;
    always #5 ref_clk=~ref_clk;
    initial begin #1.7;forever #5 apb_clk=~apb_clk;end
'''+ '\n'.join(declarations)+'\n    ntrpu_hbm_native_rw hbm(\n        '+',\n        '.join(pins)+');\n'
    text+=(ROOT/'fpga/tb/ntrpu_hbm_native_transport_body.svh').read_text()+'\nendmodule\n'
    (generated/'tb_ntrpu_hbm_native_transport.sv').write_text(text)
    # Same public binding, a deliberately limited AXI memory only for bench preflight.
    stub=(out/'ntrpu_hbm_native_rw.sv').read_text().split('    wire [63:0] parity;')[0]
    stub=stub.replace('output wire','output reg')
    stub+=(ROOT/'fpga/tb/ntrpu_hbm_native_transport_stub.svh').read_text()+'\nendmodule\n'
    (generated/'ntrpu_hbm_native_stub.sv').write_text(stub)
    (out/'COMPOSITION.json').write_text(json.dumps(dict(
        reference_result_sha256=sha(ref/'result.json'),frozen_logic=composition['frozen_logic'],
        native_binding_sha256=sha(out/'ntrpu_hbm_native_rw.sv'),
        bench_sha256=sha(generated/'tb_ntrpu_hbm_native_transport.sv'),
        cold_epochs=1,write_ports=[0,16],readback_ports=[0,16],
        clocks_mhz=[200,450],cases=4,expected_descriptors=24),indent=2)+'\n')

if __name__=='__main__':main()
