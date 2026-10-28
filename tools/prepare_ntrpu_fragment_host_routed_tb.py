"""Observe the routed response's retained zero bit, not an unused IP port.

Routing leaves the optional encoded RRESP[0] port unconnected. The retained
handshake bit32 has INIT=0,D=0,R=0 in the reviewed netlist; observe its Q.
This changes only a passive display expression, never stimulus or DUT logic.
The reviewer must verify that exact constant-register structure in the input.
"""
from pathlib import Path
from prepare_ntrpu_fragment_host_netlist_tb import adapt as synth_adapt


def adapt(text):
    text = synth_adapt(text)
    old = 'dut.control_cdc.bridge.m_axi_rresp[0]'
    assert text.count(old) == 1
    retained = (r'dut.control_cdc.bridge.inst.'
                r'\gen_clock_conv.gen_async_lite_conv.clock_conv_lite_resp_r '
                '.handshake.dest_hsdata_ff[32]')
    return text.replace(old, retained)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    source = root / 'fpga/tb/tb_ntrpu_pcie_fragment_host.sv'
    target = root / 'fpga/tb/tb_ntrpu_pcie_fragment_host_routed_netlist_r2.sv'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('// Passive routed observer; retained constant bit verified by reviewer.\n')
        stream.write(adapt(source.read_text(encoding='utf-8')))
