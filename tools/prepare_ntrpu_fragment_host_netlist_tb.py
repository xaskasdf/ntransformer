"""Create the same host/MIG bank with observers at preserved netlist pins.

Synthesis moves AXI-Lite decode logic across hierarchy and merges equal
response bits. Observe the actual AMD bridge pins; undo only the existing
reversible RRESP transport coding when displaying the UI-side response.
No stimulus, assertion, model or production RTL is changed.
"""
from pathlib import Path


def adapt(text):
    assert text.count('dut.data_path.d_axi_wready') == 1
    text = text.replace('dut.data_path.d_axi_wready',
                        'dut.data_path.ingress_cdc.m_axi_wready')
    response = 'dut.control_cdc.bridge.m_axi_rresp'
    text = text.replace('dut.d_axil_rresp',
                        '{' + response + '[1],' + response + '[0]^' + response + '[1]}')
    text = text.replace('dut.d_axil_', 'dut.control_cdc.bridge.m_axi_')
    return text


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1]
    source = root / 'fpga/tb/tb_ntrpu_pcie_fragment_host.sv'
    target = root / 'fpga/tb/tb_ntrpu_pcie_fragment_host_netlist.sv'
    with target.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('// Same host/MIG bank; only passive observer paths adapted to synthesis.\n')
        stream.write(adapt(source.read_text(encoding='utf-8')))
