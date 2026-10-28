"""Adapt the frozen cold bank to the exact routed netlist's passive observers.

No model, transaction, numerical assertion or production RTL changes. Hold
cold reset through GSR; the compiled FIRST_COOKIE value is already pinned by
the synthesis/route provenance. Full ARADDR remains at the AMD bridge port.
The optimized RRESP zero bit is observed at its retained physical register.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re


def verify_netlist(text):
    modules={m[1]:m[2] for m in re.finditer(r'(?ms)^module (\S+)\s+(.*?)^endmodule\s*$',text)}
    kind='ntrpu_pcie_fragment_dual_host_cold'
    top=modules[kind]
    assert 'parameter' not in top
    assert 'wire [7:2]d_axil_araddr;' in top and 'wire [1:1]d_axil_rresp;' in top
    path=['control_cdc','bridge','inst',r'\gen_clock_conv.gen_async_lite_conv.clock_conv_lite_resp_r','handshake']
    hierarchy=[]
    for instance in path:
        match=re.search(r'^\s*(\S+)\s+'+re.escape(instance)+r'\s*\(',modules[kind],re.M)
        assert match, (kind,instance)
        hierarchy.append(dict(parent=kind,instance=instance,module=match[1]))
        kind=match[1]
    match=re.search(r"FDRE #\(\s*\.INIT\(1'b0\)\)\s*\\dest_hsdata_ff_reg\[32\]\s*\(.*?\);",modules[kind],re.S)
    assert match
    block=match[0]
    for field in [".D(1'b0)",".R(1'b0)",'.Q(dest_hsdata_ff[32])','.C(dest_clk)','.CE(dest_hsdata_en)']:
        assert field in block,field
    assert 'output [31:0]m_axi_araddr;' in modules['ntrpu_host_axil_cdc']
    return dict(hierarchy=hierarchy,retained_zero_register=block)


def adapt(source,netlist):
    audit=verify_netlist(netlist)
    changes={
        "ntrpu_pcie_fragment_dual_host_cold #(.FIRST_COOKIE(64'hffffffff)) dut (.*);":
            'ntrpu_pcie_fragment_dual_host_cold dut (.*);',
        'dut.d_axil_araddr':'dut.control_cdc.bridge.m_axi_araddr',
        'dut.d_axil_rresp':
            r'{dut.control_cdc.bridge.m_axi_rresp[1],dut.control_cdc.bridge.inst.\gen_clock_conv.gen_async_lite_conv.clock_conv_lite_resp_r .handshake.dest_hsdata_ff[32]^dut.control_cdc.bridge.m_axi_rresp[1]}',
        'initial begin external_reset_async=1;#40;external_reset_async=0;end':
            'initial begin external_reset_async=1;#200;if(glbl.GSR!==0) $fatal(1,"MANAGED_NETLIST_GSR_ACTIVE");$display("MANAGED_NETLIST_GSR_RELEASE time=%0t gsr=%0b",$time,glbl.GSR);external_reset_async=0;end',
    }
    for old,new in changes.items():
        assert source.count(old)==1,(old,source.count(old))
        source=source.replace(old,new,1)
    audit['changes']=changes
    return source,audit


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('netlist');parser.add_argument('output');parser.add_argument('audit')
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    path=root/'fpga/tb/tb_ntrpu_pcie_managed_host_cold_r2.sv';raw=path.read_bytes()
    netlist=Path(args.netlist).read_bytes();text,audit=adapt(raw.decode(),netlist.decode())
    with Path(args.output).open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    audit.update(source_sha256=hashlib.sha256(raw).hexdigest(),netlist_sha256=hashlib.sha256(netlist).hexdigest(),
        generated_sha256=hashlib.sha256(Path(args.output).read_bytes()).hexdigest())
    with Path(args.audit).open('x',encoding='utf-8') as f:json.dump(audit,f,indent=2);f.write('\n')
