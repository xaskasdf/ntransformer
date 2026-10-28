"""Bind the reviewed read/write HBM model to a test-only fault interposer."""
import re
from pathlib import Path

def prepare(binding:Path,output:Path):
    ports=re.findall(r'^\s*(input|output) wire\s*(\[\d+:0\])?\s*(\w+)\s*,?\s*$',binding.read_text().split(');',1)[0],re.M)
    assert len(ports)>40 and {n for _,_,n in ports}>={'s_rdata','apb_complete_0','apb_complete_1'}
    declarations=['reg ref_clk=0,apb_clk=0,apb_rst_n=0;',
                  'always #5 ref_clk=~ref_clk;',
                  'initial begin #1.7;forever #5 apb_clk=~apb_clk;end']
    pins=[]
    special=dict(axi_clk='mem_clk',axi_cold_reset_n='mem_rst_n',s_arvalid='h_arvalid & ar_open',
                 s_arready='raw_arready',s_rvalid='raw_rvalid',s_rready='h_rready & r_open & ~stray',
                 s_rdata='raw_rdata',s_rid='raw_rid',s_rresp='raw_rresp',s_rlast='raw_rlast')
    for direction,width,name in ports:
        if name in special:value=special[name]
        elif name=='s_rdata_parity':value=''
        elif name.startswith('s_'):value='h_'+name[2:]
        elif direction=='output':declarations.append(f'wire {width} {name};');value=name
        elif name.startswith('HBM_REF_CLK'):value='ref_clk'
        elif name.endswith('PCLK'):value='apb_clk'
        elif name.endswith('PRESET_N'):value='apb_rst_n'
        else:value="'0"
        pins.append(f'.{name}({value})')
    output.write_text('\n'.join(declarations)+'\nntrpu_hbm_native_rw memory(\n'+',\n'.join(pins)+');\n')
