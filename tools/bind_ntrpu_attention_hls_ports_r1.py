"""Check actual Vitis attention ports before emitting a zero-latency binding.

Only the frozen single-port ap_memory contract is supported. Unknown ports,
extra memory ports and changed widths/directions fail before RTL is emitted.
This is interface inspection, not numerical or timing qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

TOP='ntrpu_attention_tile_fp32_r1'


def ports(text,top):
    clean=re.sub(r'/\*.*?\*/|//[^\n]*|\(\*.*?\*\)','',text,flags=re.S)
    match=re.search(r'\bmodule\s+'+re.escape(top)+r'\s*\((.*?)\)\s*;(.*?)\bendmodule',clean,re.S)
    assert match, 'Expected non-ANSI generated module'
    names=[n.strip() for n in match[1].split(',')]
    assert all(re.fullmatch(r'\w+',n) for n in names) and len(set(names))==len(names)
    result={}
    for row in re.finditer(r'^\s*(input|output)\s+(?:wire\s+|reg\s+)?(?:\[(\d+)\s*:\s*(\d+)\]\s*)?(\w+)\s*;',match[2],re.M):
        direction,high,low,name=row.groups();assert name not in result
        assert low is None or int(low)==0, 'Unexpected port range'
        result[name]=dict(direction=direction,width=int(high)+1 if high is not None else 1)
    assert set(result)==set(names), ('Undeclared or unsupported ports',set(names)^set(result))
    return result,match[2]


def contract(actual):
    mapping={};roles=[]
    def bind(candidates,direction,width,signal):
        found=[n for n in candidates if n in actual]
        assert len(found)==1, ('Missing/ambiguous role',candidates,found)
        name=found[0];assert name not in mapping
        assert actual[name]==dict(direction=direction,width=width), ('Port mismatch',name,actual[name])
        mapping[name]=signal;roles.append(dict(port=name,direction=direction,width=width,signal=signal))
        return name
    for name,direction,width,signal in [('ap_clk','input',1,'clk'),('ap_rst','input',1,'~rst_n'),
            ('ap_start','input',1,'hls_start'),('ap_done','output',1,'hls_done'),
            ('ap_ready','output',1,'hls_ready'),('ap_idle','output',1,'unused_idle'),('ap_return','output',32,'hls_return')]:
        bind([name],direction,width,signal)
    for name,width in [('tile_start',32),('tile_count',32),('query_position',32),('layer',32),
                       ('query_head',32),('kv_head',32),('sequence',64),('epoch',64)]:
        bind([name,name+'_r'],'input',width,'hls_'+name)
    for aliases,signal,aw,dw,write in [(['query','query_r'],'query',6,32,False),
            (['keys','keys_r'],'keys',12,32,False),(['values','values_r'],'values',12,32,False),
            (['state_input','state_input_r'],'state_in',7,64,False),
            (['state_output','state_output_r'],'state_out',7,64,True),(['output_r','output'],'output',6,32,True)]:
        matches=[prefix for prefix in aliases if prefix+'_address0' in actual]
        assert len(matches)==1, ('Memory prefix',aliases,matches)
        prefix=matches[0]
        bind([prefix+'_address0'],'output',aw,'hls_'+signal+'_address')
        bind([prefix+'_ce0'],'output',1,'hls_'+signal+'_ce')
        if write:
            bind([prefix+'_we0'],'output',1,'hls_'+signal+'_we')
            bind([prefix+'_d0'],'output',dw,'hls_'+signal+'_d')
        else:
            bind([prefix+'_q0'],'input',dw,'hls_'+signal+'_q')
    assert set(mapping)==set(actual), ('Extra core ports',set(actual)-set(mapping))
    assert len(mapping)==35
    return mapping,roles


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rtl',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    raw=args.rtl.read_bytes();text=raw.decode();actual,body=ports(text,TOP);mapping,roles=contract(actual)
    assert re.search(r'always\s*@\s*\(posedge\s+ap_clk\s*\)',body)
    assert re.search(r'if\s*\(ap_rst\s*==\s*1\x27b1\)',body)
    assert not re.search(r'(?:posedge|negedge)\s+ap_rst',body), 'Asynchronous reset needs a separate binding'
    declarations=['    input wire clk','    input wire rst_n']
    for row in roles:
        if row['signal'] in ('clk','~rst_n','unused_idle'):continue
        width='' if row['width']==1 else '['+str(row['width']-1)+':0] '
        declarations.append('    '+row['direction']+' wire '+width+row['signal'])
    sv='// Generated only after exact inspection of the supplied Vitis module.\n'
    sv+='// Core SHA256: '+hashlib.sha256(raw).hexdigest()+'\n`default_nettype none\n'
    sv+='module ntrpu_attention_hls_ports_r1 (\n'+',\n'.join(declarations)+'\n);\n'
    sv+='    wire unused_idle;\n    '+TOP+' core (\n'
    sv+=',\n'.join('        .'+name+'('+signal+')' for name,signal in mapping.items())+'\n    );\nendmodule\n`default_nettype wire\n'
    args.output.mkdir(parents=True);output=args.output/'ntrpu_attention_hls_ports_r1.sv'
    output.write_text(sv,encoding='utf-8',newline='\n')
    report=dict(status='INSPECTED_ATTENTION_R1_GENERATED_PORTS',top=TOP,core_path=str(args.rtl),
        core_sha256=hashlib.sha256(raw).hexdigest(),ports=roles,synchronous_active_high_core_reset=True,
        binding_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        added_latency_cycles=0,numerical_qualified=False,physical_timing_qualified=False)
    (args.output/'binding.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__':main()
