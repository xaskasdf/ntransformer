"""Read-only, pinned source-boundary audit. Not a PCIe adversarial simulation."""
import argparse
import hashlib
import json
from pathlib import Path
import re

TOP_SHA = '3c64aaa56ea4601147f9cf91963fda4356f4faa3127ea394505c84e6b723b988'
APP_SHA = '735953b55f6bad1f283711b87204f856941d99c583ebba769e17f69417930be6'
XCI_SHA = 'bf4a478249f14eb4a3db8d3317d4040fa374ffc09e4936f149a14e41dffe0a7a'
AXIL = 'awaddr awprot awvalid awready wdata wstrb wvalid wready bresp bvalid bready araddr arprot arvalid arready rdata rresp rvalid rready'.split()
DATA = 'awaddr awid awlen awsize awburst awvalid awready wdata wstrb wlast wvalid wready bvalid bready bresp bid'.split()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(content):
    return hashlib.sha256(content).hexdigest()


def uncomment(text):
    return re.sub(r'/\*.*?\*/|//[^\n]*', '', text, flags=re.S)


def ports(text, module, instance):
    matches = re.findall(r'\b' + module + r'\s+' + instance + r'\s*\((.*?)\)\s*;', text, re.S)
    require(len(matches) == 1, 'Unique instance: ' + instance)
    pairs = re.findall(r'\.(\w+)\s*\(([^()]*)\)', matches[0])
    require(len(pairs) == len(dict(pairs)), 'Unique named ports: ' + instance)
    return {k: re.sub(r'\s+', '', v) for k, v in pairs}


def audit(top_bytes, app_bytes, xci_bytes):
    top, app = map(lambda b: uncomment(b.decode()), (top_bytes, app_bytes))
    bridge = ports(top, 'ntrpu_bridge', 'ntrpu_bridge_i')
    target = ports(top, 'xilinx_axi_pcie_ep_app', 'xilinx_axi_pcie_ep_app_i')
    bram = ports(app, 'axi_bram_ctrl_0', 'AXI_BRAM_CTL')
    for field in AXIL:
        require(bridge.get('s_axil_' + field) == 'board.probe.m_axil_' + field,
                'Private control binding: ' + field)
    for field in DATA:
        require(bridge.get('s_axib_' + field) == 'board.probe.m_' + field,
                'Private requester binding: ' + field)
    for field in 'awaddr araddr wdata wstrb awvalid wvalid arvalid'.split():
        require(bridge.get('m_axib_' + field) == 'm_axi_' + field, 'Incoming bridge binding: ' + field)
    expected_target = dict(awaddr='awaddr', araddr='araddr', wdata='wdata', wstrb='wstrb',
                           awvalid='m_axi_awvalid', wvalid='m_axi_wvalid', arvalid='m_axi_arvalid')
    for field, net in expected_target.items():
        require(target.get('s_axi_' + field) == net, 'Incoming target binding: ' + field)
        require(bram.get('s_axi_' + field) == 's_axi_' + field, 'BRAM-only target: ' + field)
    require(not any('probe.' in v for v in target.values()), 'No private control at incoming target')
    require('s_axil_' not in app and 'board.' not in app, 'No control/hierarchical escape in app')
    # Complete reviewed file identity also covers aliases, additional drivers,
    # instance sets and response paths, beyond the selected pin checks above.
    require(sha(top_bytes) == TOP_SHA, 'Reviewed endpoint source identity')
    require(sha(app_bytes) == APP_SHA, 'Reviewed BRAM application identity')
    ip = json.loads(xci_bytes)['ip_inst']
    require((ip['component_reference'], ip['ip_revision']) == ('xilinx.com:ip:xdma:4.2', '4'), 'Native IP version')
    params = {k: v[0]['value'] for k, v in ip['parameters']['component_parameters'].items()}
    expected = dict(functional_mode='AXI_Bridge', device_port_type='PCI_Express_Endpoint_device',
                    axilite_master_en='false', axist_bypass_en='false', xdma_axilite_slave='true',
                    ext_xvc_vsec_enable='false', en_debug_ports='false', pf0_msix_enabled='false',
                    pf0_bar0_enabled='true', pf0_bar0_type='Memory', pf0_bar0_size='128',
                    pf0_bar0_scale='Kilobytes', pf0_bar0_64bit='false', axibar_num='1',
                    axibar_0='0x0000000010000000', axibar_highaddr_0='0x0000000010000fff',
                    pciebar2axibar_0='0x0000000000000000')
    expected.update({f'pf0_bar{i}_enabled': 'false' for i in range(1, 6)})
    for key, value in expected.items():
        require(params.get(key) == value, 'Reviewed IP profile: ' + key)
    require(sha(xci_bytes) == XCI_SHA, 'Reviewed complete XCI identity')
    return dict(status='PINNED_HOST_BOUNDARY_SOURCE_PASS', private_axil_connections=len(AXIL),
                private_requester_connections=len(DATA), incoming_target='128 KiB AMD example BRAM',
                source_sha256=dict(endpoint=sha(top_bytes), app=sha(app_bytes), xci=sha(xci_bytes)),
                checked_profile=expected, opaque_native_core=True, host_adversarial_simulation=False,
                physical_isolation_proven=False,
                note='A source boundary and documented VSEC accessibility, not proof of all PCIe transactions.')


def read_inputs(example):
    xci = list(example.rglob('ntrpu_bridge.xci'))
    require(len(xci) == 1, 'Exactly one bridge XCI')
    return ((example/'imports/xilinx_axi_pcie4_ep.v').read_bytes(),
            (example/'imports/xilinx_axi_pcie_ep_app.v').read_bytes(), xci[0].read_bytes())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(*read_inputs(args.example))
    with args.output.open('x', encoding='utf-8') as out:
        json.dump(result, out, indent=2); out.write('\n')
    print(json.dumps(result))
