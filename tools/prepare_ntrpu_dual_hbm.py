"""Generate test-only HBM pins from a frozen, reviewed native binding header."""
import re
from pathlib import Path


def prepare(binding: Path, output: Path):
    header=binding.read_text().split(');',1)[0]
    ports=re.findall(r'^\s*(input|output) wire\s*(\[\d+:0\])?\s*(\w+)\s*,?\s*$',header,re.M)
    assert len(ports)>40 and {n for _,_,n in ports}>={'s_rdata','apb_complete_0','apb_complete_1'}
    declarations=['reg ref_clk=0,apb_clk=0,apb_rst_n=0;',
                  'always #5 ref_clk=~ref_clk;',
                  'initial begin #1.7;forever #5 apb_clk=~apb_clk;end',
                  'reg [1:0] write_open=3,reply_open=3,bad_bid=0,bad_bresp=0;',
                  'wire [1:0] memory_wready,memory_bvalid;',
                  'wire [11:0] memory_bid;wire [3:0] memory_bresp;',
                  'assign h_wready=memory_wready & write_open;',
                  'assign h_bvalid=memory_bvalid & reply_open;',
                  "assign h_bid=memory_bid ^ {5'b0,bad_bid[1],5'b0,bad_bid[0]};",
                  "assign h_bresp=memory_bresp ^ {bad_bresp[1],1'b0,bad_bresp[0],1'b0};"]
    pins=[]
    special=dict(axi_clk='mem_clk',axi_cold_reset_n='mem_rst_n',
                 s_wready='memory_wready',s_wvalid='h_wvalid & write_open',
                 s_bid='memory_bid',s_bresp='memory_bresp',s_bvalid='memory_bvalid',s_bready='h_bready & reply_open')
    for direction,width,name in ports:
        if name in special:value=special[name]
        elif name.startswith('s_'):
            if name.startswith(('s_ar','s_r')):
                declarations.append(f'{"reg" if direction=="input" else "wire"} {width} {name}'+('=0;' if direction=='input' else ';'))
                value=name
            else:value='h_'+name[2:]
        elif direction=='output':
            declarations.append(f'wire {width} {name};');value=name
        elif name.startswith('HBM_REF_CLK'):value='ref_clk'
        elif name.endswith('PCLK'):value='apb_clk'
        elif name.endswith('PRESET_N'):value='apb_rst_n'
        else:value="'0"
        pins.append(f'.{name}({value})')
    output.write_text('\n'.join(declarations)+'\nntrpu_hbm_native_rw hbm(\n'+',\n'.join(pins)+');\n')
