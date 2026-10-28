"""Connect the client only in a private copy of the generated AMD example."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def patch(example):
    marker = example / 'NTRPU_BRIDGE_BINDING.json'
    if marker.exists():
        raise ValueError('Example already patched')
    client = Path(__file__).with_name('ntrpu_axi_bridge_endpoint_client.sv').read_text()
    names = re.findall(r'^\s*(?:input|output) wire (?:\[[^\]]+\] )?(\w+)[,\n]', client, re.M)
    bindings = {}
    for name in names:
        if name.startswith('m_axil_'):
            bindings['s_axil_' + name[7:]] = name
        elif name.startswith(('m_aw', 'm_w', 'm_b')):
            bindings['s_axib_' + name[2:]] = name
    if len(bindings) != 35:
        raise ValueError('Expected exactly 35 client bus connections')
    changes = []
    staged = {}
    for name in ('board.v', 'sample_tests.vh', 'xilinx_axi_pcie4_ep.v'):
        path = example / 'imports' / name
        before = path.read_bytes()
        text = before.decode()
        if name == 'board.v':
            assert text.count('module board;') == 1
            text = text.replace('module board;', 'module board;\n    ntrpu_axi_bridge_native probe();')
        elif name == 'sample_tests.vh':
            text = text[:text.index('else if(testname')] + '''
else if(testname == "ntrpu_bridge_native") begin
    TSK_SYSTEM_INITIALIZATION;
    TSK_BAR_INIT;
    // The stock BAR task enables memory decoding but not Bus Master Enable.
    TSK_TX_TYPE0_CONFIGURATION_WRITE(DEFAULT_TAG, 12'h04, 32'h00000007, 4'h1);
    TSK_TX_CLK_EAT(32);
    TSK_TX_TYPE0_CONFIGURATION_READ(DEFAULT_TAG, 12'h04, 4'h1);
    TSK_WAIT_FOR_READ_DATA;
    if ((P_READ_DATA & 32'h7) !== 32'h7) $fatal(1, "Bridge command not enabled");
    board.probe.run_probe;
    $finish;
end
'''
        else:
            for port, signal in bindings.items():
                pattern = r'\.' + port + r'\s*\(([^)]*)\)'
                matches = re.findall(pattern, text)
                assert len(matches) == 1, port
                assert not matches[0].strip() or re.fullmatch(r"\d+'b[01]+", matches[0].strip()), port
                text = re.sub(pattern, '.' + port + '(board.probe.' + signal + ')', text)
            # Correct the stock example's zero literal widths; preserve its BRAM target.
            for port in ('m_axib_bid', 'm_axib_rid'):
                text, count = re.subn(r'\.' + port + r"\s*\(3'b0\)", '.' + port + "(4'b0)", text)
                assert count == 1
        after = text.encode()
        staged[path] = after
        changes.append(dict(path='imports/' + name, before=hashlib.sha256(before).hexdigest(),
                            after=hashlib.sha256(after).hexdigest()))
    for path, content in staged.items():
        path.write_bytes(content)
    marker.write_text(json.dumps(dict(changes=changes, connections=bindings,
        control_master='board.probe.dut.mapper', incoming_target='original AMD BRAM app',
        synthesizable_board_top=False), indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('example', type=Path)
    patch(parser.parse_args().example.resolve())
