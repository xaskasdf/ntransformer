"""Build a synthesizable private composition from the frozen AMD example."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def ports(text):
    header=text[:text.index(');')]
    result={}
    for line in header.splitlines():
        match=re.fullmatch(r'\s*(input|output) wire\s+(\[[^]]+\]\s*)?([\w,]+),?\s*',line)
        if not match:continue
        direction,width,names=match.groups()
        for name in names.strip(',').split(','):
            if name in result:raise ValueError('Duplicate port '+name)
            result[name]=(direction,(width or '').strip())
    if not {'clk','cfg_valid','m_axil_araddr','guard_fault_code'}.issubset(result):raise ValueError('Incomplete guarded client ports')
    return result


def prepare(example,client,output):
    original=example/'imports/xilinx_axi_pcie4_ep.v';text=original.read_text()
    before=hashlib.sha256(original.read_bytes()).hexdigest()
    parsed=ports(client.read_text())
    references=re.findall(r'board\.probe\.(\w+)',text)
    private={name for name in parsed if name.startswith('m_')}
    if len(references)!=35 or set(references)!=private:raise ValueError('Expected exactly the 35 reviewed requester/control connections')
    output.mkdir(parents=True,exist_ok=False)
    text=text.replace('module xilinx_axi_pcie4_ep','module ntrpu_axi_bridge_guard_native',1)
    text=text.replace('board.probe.','')
    internal={'clk','mgmt_clk','native_link_up','native_axi_reset_n','native_ctl_reset_n','perst_n'}|private
    additions=['  input wire mgmt_clk_in']
    for name,(direction,width) in parsed.items():
        if name not in internal:additions.append(f'  {direction} wire {width} {name}')
    anchor='  input                                       sys_rst_n\n);'
    if text.count(anchor)!=1:raise ValueError('Example top header changed')
    text=text.replace(anchor,'  input                                       sys_rst_n,\n'+',\n'.join(additions)+'\n);')
    wires=[]
    for name in sorted(private):
        width=parsed[name][1].replace('AXI_ADDR_BITS','64')
        wires.append(f'    wire {width} {name};')
    body='\n'.join(wires)+'''
    wire mgmt_clk;
    // The management source remains an external, independent clock contract.
    BUFGCE management_buffer(.I(mgmt_clk_in),.CE(1'b1),.O(mgmt_clk));
    (* KEEP_HIERARCHY="YES" *) ntrpu_axi_bridge_guarded_client guarded(
        .clk(axi_aclk_out),.mgmt_clk(mgmt_clk),
        .native_link_up(user_lnk_up),.native_axi_reset_n(axi_aresetn),
        .native_ctl_reset_n(axi_ctl_aresetn),.perst_n(sys_rst_n_c),.*);
'''
    if text.count('endmodule')!=1:raise ValueError('One top module expected')
    text=text.replace('endmodule',body+'\nendmodule')
    if 'board.probe' in text:raise ValueError('Simulation-only reference survived')
    (output/'ntrpu_axi_bridge_guard_native.sv').write_text(text,encoding='utf-8')
    info=dict(status='PREPARED_NOT_SYNTHESIZED',example_sha256=before,
              guarded_client_sha256=hashlib.sha256(client.read_bytes()).hexdigest(),
              native_top_sha256=hashlib.sha256((output/'ntrpu_axi_bridge_guard_native.sv').read_bytes()).hexdigest(),
              client_ports=parsed,private_connections=sorted(private),
              native_clock_source='Original AMD axi_aclk_out; no extra AXI clock buffer',
              management_clock_source='Independent external mgmt_clk_in through BUFGCE; board source unassigned',
              provider_example_board_xdc_used=False,board_pinout_approved=False)
    (output/'PREPARED.json').write_text(json.dumps(info,indent=2)+'\n')
    return info


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('example',type=Path)
    ap.add_argument('client',type=Path);ap.add_argument('output',type=Path)
    a=ap.parse_args();print(json.dumps(prepare(a.example,a.client,a.output)))
